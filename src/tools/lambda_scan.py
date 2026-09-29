"""lambda_scan: chốt thang đo và lưới giá trị của λ TRƯỚC khi tốn lượt huấn luyện nào.

    python -m src.tools.lambda_scan --workdir data/stage_a --fit fit.qwen1_5b_base.v1.jsonl
    python -m src.tools.lambda_scan --workdir data/stage_a --fit fit.qwen1_5b_base.jsonl --div-scale raw

Câu hỏi nó trả lời: với mỗi λ, thành phần đa dạng có thật sự đổi tập được chọn không, và ở bao nhiêu câu.
Lưới {0; 0,25; 0,5; 1} được đoán trước khi có dữ liệu; nếu cả bốn giá trị đều gần như không đổi tập nào so
với λ = 0 thì Div chỉ có mặt trên giấy, và mọi lượt huấn luyện để chọn λ là phí.

Cách đọc:
  - cột "đổi so với λ=0": tỷ lệ câu hỏi (có hơn k ứng viên) mà tập chọn khác tập của λ = 0
  - cột "trùng Div-only": tỷ lệ câu mà tập chọn trùng tập có Div lớn nhất, tức Div đã lấn át
  - lưới đề xuất: các λ đầu tiên đạt khoảng 10%, 25% và 50% câu đổi tập, cộng thêm 0

Không chọn λ tốt nhất: việc đó cần huấn luyện trên validation_matched.jsonl. Công cụ này chỉ bảo đảm lưới
đem đi thử trải đúng vùng mà λ có tác dụng.

Chỉ đọc file, không cần GPU. RSR không đổi định nghĩa ngày 29/09, nên dùng file fit cũ (.v1) cũng được;
383 chuỗi dài ở file cũ bị chấm trên 2.048 token đầu, nên chạy lại với file mới để xác nhận.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl, split_tid
from src.stage_b.objective import (base_of_subsets, choose, distance_matrix, div_of_subsets, div_scale,
                                   fit_from_rsr, subsets)

DEFAULT_LAMS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
                10.0]
TARGETS = (0.10, 0.25, 0.50)


def build_questions(fit_rows: Sequence[Mapping], qual: Mapping[str, float], tids: Sequence[str],
                    vecs: np.ndarray) -> tuple[dict, dict]:
    """Ghép ba nguồn theo tid, nhóm theo câu hỏi. Trả về (câu hỏi, thống kê chuỗi bị thiếu)."""
    row_of = {t: i for i, t in enumerate(tids)}
    by_q: dict[str, list] = defaultdict(list)
    miss = {"qual": 0, "embed": 0}
    for r in fit_rows:
        t = r["tid"]
        if qual.get(t) is None:
            miss["qual"] += 1
            continue
        if t not in row_of:
            miss["embed"] += 1
            continue
        by_q[r.get("qid") or split_tid(t)[0]].append((t, float(r["rsr"]), float(qual[t]), row_of[t]))
    out = {}
    for q, items in by_q.items():
        items.sort()
        out[q] = {"tids": [x[0] for x in items],
                  "fit": fit_from_rsr([x[1] for x in items]),
                  "qual": np.array([x[2] for x in items]),
                  "vecs": vecs[[x[3] for x in items]]}
    return out, miss


def scan(questions: Mapping[str, Mapping], lams: Sequence[float], k: int, how: str,
         a: float = 1.0, b: float = 1.0) -> dict:
    """Với mỗi λ: tỷ lệ câu đổi tập so với λ = 0, tỷ lệ trùng tập Div lớn nhất, và thang đo hai vế."""
    changed = {lam: 0 for lam in lams}
    div_only = {lam: 0 for lam in lams}
    div_chosen = {lam: [] for lam in lams}
    base_range, d_range, n_q = [], [], 0
    for q in questions.values():
        n = len(q["tids"])
        if n <= k:
            continue                                          # không có gì để chọn
        n_q += 1
        sets = subsets(n, k)
        base = base_of_subsets(q["fit"], q["qual"], sets, a, b)
        div_raw = div_of_subsets(distance_matrix(q["vecs"]), sets)
        div = div_scale(div_raw, how)
        base_range.append(float(base.max() - base.min()))
        d_range.append(float(div.max() - div.min()))
        ref = choose(base, div, 0.0)
        top_div = int(np.argmax(div))
        for lam in lams:
            c = choose(base, div, lam)
            changed[lam] += c != ref
            div_only[lam] += c == top_div
            div_chosen[lam].append(float(div_raw[c]))
    if n_q == 0:
        raise ValueError(f"không có câu hỏi nào có hơn k = {k} ứng viên")
    rows = [{"lam": lam, "changed": changed[lam] / n_q, "div_only": div_only[lam] / n_q,
             "div_raw_mean": statistics.mean(div_chosen[lam])} for lam in lams]
    return {"n_questions": n_q, "rows": rows,
            "base_range_median": statistics.median(base_range),
            "div_range_median": statistics.median(d_range)}


def suggest_grid(rows: Sequence[Mapping], targets: Sequence[float] = TARGETS) -> list[float]:
    """λ = 0 cộng λ nhỏ nhất đạt từng mức đổi tập. Mức nào không đạt được thì bỏ."""
    grid = [0.0]
    for t in targets:
        hit = next((r["lam"] for r in rows if r["lam"] > 0 and r["changed"] >= t), None)
        if hit is not None and hit not in grid:
            grid.append(hit)
    return grid


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--fit", default="fit.qwen1_5b_base.jsonl", help="file của b1_fit, chỉ đọc cột rsr")
    ap.add_argument("--embeddings", default="embeddings.npz")
    ap.add_argument("--div-scale", choices=["subsets", "raw", "both"], default="both")
    ap.add_argument("--k", type=int, help="mặc định selection.k trong cấu hình")
    args = ap.parse_args(argv)

    cfg = load_config()
    wd = resolve_path(cfg, args.workdir)
    k = args.k or cfg["selection"]["k"]
    a, b = cfg["selection"]["a"], cfg["selection"]["b"]
    fit_rows = read_jsonl(wd / args.fit)
    if not fit_rows:
        raise SystemExit(f"Không đọc được {wd / args.fit}")
    qual = {r["tid"]: r.get("qual") for r in read_jsonl(wd / "quality.jsonl")}
    npz = np.load(wd / args.embeddings)
    questions, miss = build_questions(fit_rows, qual, [str(t) for t in npz["tids"]], npz["vectors"])
    print(f"[λ] {len(fit_rows)} chuỗi trong {args.fit}, ghép được {sum(len(q['tids']) for q in questions.values())}"
          f" chuỗi thuộc {len(questions)} câu hỏi; thiếu Qual {miss['qual']}, thiếu biểu diễn {miss['embed']}")
    print(f"[λ] k = {k}, a = {a}, b = {b}; lưới hiện tại trong cấu hình {cfg['selection']['lambda_grid']}\n")

    for how in (["subsets", "raw"] if args.div_scale == "both" else [args.div_scale]):
        r = scan(questions, DEFAULT_LAMS, k, how, a, b)
        label = "Div min-max trong các tập con của câu (0 đến 1)" if how == "subsets" else "Div giữ nguyên"
        print(f"=== {label}, {r['n_questions']} câu có hơn {k} ứng viên")
        print(f"    chênh lệch trung vị giữa tập tốt nhất và kém nhất trong một câu:"
              f" Σ Fit·Qual {r['base_range_median']:.3f}, Div {r['div_range_median']:.3f}"
              f" (tỷ số {r['base_range_median'] / max(r['div_range_median'], 1e-12):.2f})")
        print(f"    {'λ':>6}  {'đổi so với λ=0':>15}  {'trùng Div-only':>15}  {'Div thô TB của tập chọn':>24}")
        for row in r["rows"]:
            print(f"    {row['lam']:>6.2f}  {row['changed']:>15.1%}  {row['div_only']:>15.1%}"
                  f"  {row['div_raw_mean']:>24.3f}")
        print(f"    lưới đề xuất (0 và các mức 10%, 25%, 50% câu đổi tập): {suggest_grid(r['rows'])}\n")
    print("Tỷ số ở dòng chênh lệch cho biết λ cỡ nào thì hai vế ngang sức. Lưới nên trải quanh giá trị đó.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

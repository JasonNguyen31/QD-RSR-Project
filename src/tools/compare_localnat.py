"""compare_localnat: LocalNat trên bước của GLM giống GRAPE tới đâu ở từng k, để chốt số bước ngữ cảnh k.

    python -m src.tools.compare_localnat                                   # 1,5 tỷ, file mặc định
    python -m src.tools.compare_localnat --student qwen1_5b --workdir data/stage_a

Đọc  <workdir>/fit.<student>_base.jsonl             RSR, GRAPE, LARK, cột local_nat cũ (cắt theo dấu câu, k = 4)
     <workdir>/localnat.<student>_base.glm.jsonl    b1_localnat ghi: local_nat_k1, local_nat_k4, ...
Ghi  không gì cả. Không cần GPU.

Câu hỏi cần trả lời: với bước là nhóm câu (trung vị 6 bước mỗi chuỗi), k = 4 cho mô hình thấy gần hết phần đứng
trước, nên LocalNat có thể chỉ là GRAPE dưới tên khác. Khi đó phương án Local Naturalness không còn là một baseline
riêng. In ra ba thứ, đều tính TRONG TỪNG câu hỏi vì b2_select chọn trong từng câu hỏi:
  1. tương quan hạng trung bình giữa các tín hiệu (mọi tín hiệu đưa về hướng càng cao càng phù hợp, RSR đảo dấu);
  2. tập top-k mà từng bản LocalNat chọn trùng tập của GRAPE, RSR và cột local_nat cũ tới đâu. Đây là thứ quyết
     định dữ liệu huấn luyện có khác nhau hay không; phá hoà theo đúng thứ tự cố định của b2_select;
  3. chuỗi phải cắt dự phòng có bị chấm lệch hệ thống so với chuỗi GLM cắt được không (thứ hạng phần trăm trung
     bình trong câu hỏi; không lệch thì gần 50%).
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl
from src.stage_b.b2_select import order_key, pick_topk
from src.stage_b.signals import add_lark
from src.tools.compare_judges import spearman

HIGHER_IS_BETTER = {"rsr": False}                # mọi cột khác đều càng cao càng phù hợp
OLD = "local_nat"
OLD_LABEL = "local_nat_cũ"


def merge(fit_rows: Sequence[Mapping], ln_rows: Sequence[Mapping]) -> tuple[list[dict], list[str]]:
    """Ghép hai file theo tid, chỉ giữ chuỗi có ở cả hai; ĝ của LARK tính lại trên đúng nhóm đó.
    Trả về (dòng đã ghép, tên các cột LocalNat mới theo thứ tự k)."""
    fit = {r["tid"]: r for r in fit_rows}
    cols = sorted((c for c in (ln_rows[0] if ln_rows else {}) if c.startswith("local_nat_k")),
                  key=lambda c: int(c[len("local_nat_k"):]))
    rows = []
    for r in ln_rows:
        f = fit.get(r["tid"])
        if f is None:
            continue
        row = {"tid": r["tid"], "qid": r["qid"], "steps_source": r.get("steps_source", "glm"),
               "rsr": f["rsr"], "grape": f["grape"], "n_tokens": f["n_tokens"],
               "brier": f["brier"], "mean_surprisal": f["mean_surprisal"], **{c: r[c] for c in cols}}
        if OLD in f:
            row[OLD_LABEL] = f[OLD]
        rows.append(row)
    return add_lark(rows), cols


def by_question(rows: Sequence[Mapping], min_chains: int, seed: int) -> dict:
    """Nhóm theo câu hỏi, giữ câu có từ min_chains chuỗi, xếp chuỗi theo thứ tự cố định của b2_select."""
    by_q: dict[str, list] = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    return {q: sorted(v, key=lambda r: order_key(seed, r["tid"])) for q, v in by_q.items() if len(v) >= min_chains}


def oriented(row: Mapping, name: str) -> float:
    v = float(row[name])
    return v if HIGHER_IS_BETTER.get(name, True) else -v


def correlations(groups: Mapping[str, Sequence[Mapping]], names: Sequence[str]) -> dict:
    """Tương quan hạng Spearman trung bình trong từng câu hỏi, cho mọi cặp tín hiệu."""
    acc: dict[tuple, list] = defaultdict(list)
    for rs in groups.values():
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                s = spearman([oriented(r, x) for r in rs], [oriented(r, y) for r in rs])
                if s is not None:
                    acc[(x, y)].append(s)
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def topk_overlap(groups: Mapping[str, Sequence[Mapping]], a: str, b: str, k: int) -> dict:
    """Hai tín hiệu chọn top-k giống nhau tới đâu, trên các câu có hơn k chuỗi."""
    same = shared = n = 0
    for rs in groups.values():
        if len(rs) <= k:
            continue
        n += 1
        pa = set(pick_topk(np.array([oriented(r, a) for r in rs]), k, "max")["idx"])
        pb = set(pick_topk(np.array([oriented(r, b) for r in rs]), k, "max")["idx"])
        same += pa == pb
        shared += len(pa & pb)
    return {"questions": n, "same_set": same / n if n else None, "sample_share": shared / (n * k) if n else None}


def fallback_rank(groups: Mapping[str, Sequence[Mapping]], col: str) -> dict:
    """Thứ hạng phần trăm trung bình (0 thấp nhất, 1 cao nhất) trong câu hỏi của các chuỗi không do GLM cắt."""
    ranks = []
    for rs in groups.values():
        if len(rs) < 2:
            continue
        vals = [float(r[col]) for r in rs]
        for r in rs:
            if not str(r["steps_source"]).startswith("glm"):
                v = float(r[col])
                below, equal = sum(x < v for x in vals), sum(x == v for x in vals) - 1
                ranks.append((below + 0.5 * equal) / (len(vals) - 1))
    return {"n": len(ranks), "mean": statistics.mean(ranks) if ranks else None}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--student", default="qwen1_5b")
    ap.add_argument("--fit", help="mặc định fit.<student>_base.jsonl")
    ap.add_argument("--localnat", help="mặc định localnat.<student>_base.glm.jsonl")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(student=args.student, overrides=args.override)
    wd = resolve_path(cfg, args.workdir)
    tag = f"{cfg['student']['key']}_base"
    fit_rows = read_jsonl(wd / (args.fit or f"fit.{tag}.jsonl"))
    ln_rows = read_jsonl(wd / (args.localnat or f"localnat.{tag}.glm.jsonl"))
    if not fit_rows or not ln_rows:
        raise SystemExit(f"Thiếu dữ liệu trong {wd}: file fit {len(fit_rows)} dòng, file LocalNat {len(ln_rows)} dòng.")
    rows, cols = merge(fit_rows, ln_rows)
    k = int(cfg["selection"]["k"])
    groups = by_question(rows, int(cfg["data"]["min_correct_per_question"]), int(cfg["selection"]["random_seed"]))
    sources = defaultdict(int)
    for r in rows:
        sources[r["steps_source"]] += 1
    print(f"[localnat] {len(rows)} chuỗi có ở cả hai file, {len(groups)} câu có từ "
          f"{cfg['data']['min_correct_per_question']} chuỗi | nguồn bước: "
          + ", ".join(f"{s} {n}" for s, n in sorted(sources.items())))

    names = cols + ([OLD_LABEL] if OLD_LABEL in rows[0] else []) + ["grape", "lark", "rsr", "n_tokens"]
    m = correlations(groups, names)
    w = max(len(n) for n in names) + 2
    print("\n1. Tương quan hạng trung bình trong từng câu hỏi (RSR đã đảo dấu, càng cao càng phù hợp)\n")
    print(" " * w + "".join(f"{n:>15}" for n in names))
    for x in names:
        cells = [f"{'1.00':>15}" if x == y else
                 (f"{v:>15.2f}" if (v := m.get((x, y), m.get((y, x)))) is not None else f"{'—':>15}") for y in names]
        print(f"{x:<{w}}" + "".join(cells))

    refs = ["grape", "rsr"] + ([OLD_LABEL] if OLD_LABEL in rows[0] else [])
    print(f"\n2. Tập top-{k} trùng nhau tới đâu (câu có hơn {k} chuỗi): tỷ lệ câu chọn ĐÚNG CÙNG tập, và tỷ lệ mẫu chung\n")
    print(f"   {'':<{w}}" + "".join(f"{r:>26}" for r in refs))
    for c in cols + ([OLD_LABEL] if OLD_LABEL in rows[0] else []):
        cells = []
        for r in refs:
            if r == c:
                cells.append(f"{'—':>26}")
                continue
            o = topk_overlap(groups, c, r, k)
            cells.append(f"{o['same_set']:>14.1%} / {o['sample_share']:>8.1%}" if o["questions"] else f"{'—':>26}")
        print(f"   {c:<{w}}" + "".join(cells))
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            o = topk_overlap(groups, a, b, k)
            if o["questions"]:
                print(f"   {a} so với {b}: cùng tập {o['same_set']:.1%}, mẫu chung {o['sample_share']:.1%} "
                      f"({o['questions']} câu)")

    print("\n3. Chuỗi không do GLM cắt (cứu từ phản hồi thô coi như GLM): thứ hạng phần trăm trung bình trong câu hỏi")
    for c in cols:
        f = fallback_rank(groups, c)
        print(f"   {c}: " + (f"{f['mean']:.0%} trên {f['n']} chuỗi (không lệch thì gần 50%)" if f["n"] else "không có chuỗi nào"))

    print("\nCách đọc: bản nào tương quan với GRAPE sát 1 và chọn gần như cùng tập với GRAPE thì không còn là một")
    print("baseline riêng. Bài gốc (bản mới nhất) cho biết cửa sổ ngữ cảnh 5% đến 25% số bước đứng trước là vùng xếp")
    print("hạng đúng, 50% đến 75% thì hội tụ về cách chấm toàn cục; xem thêm mục 5 của python -m src.tools.audit_steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

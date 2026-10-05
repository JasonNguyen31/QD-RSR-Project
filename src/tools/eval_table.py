"""eval_table: dựng bảng so sánh các mô hình từ kết quả của c1_evaluate. Không cần GPU, không gọi mạng.

    python -m src.tools.eval_table                                  # mọi mô hình 1,5 tỷ đã đánh giá, mọi seed có sẵn
    python -m src.tools.eval_table --seeds 42                       # chỉ seed 42
    python -m src.tools.eval_table --versus correct_only.k3         # thêm chênh lệch ghép cặp so với một mốc
    python -m src.tools.eval_table --benchmarks val --versus fit_quality.k3      # bảng chọn lambda
    python -m src.tools.eval_table --format tsv > bang.tsv          # dán vào bảng tính

Đọc  outputs/results/eval/<student>_base/<tên>/seed<seed>/summary.json     chỉ số từng bộ (c1_evaluate ghi)
     .../<bộ>.jsonl                                                        từng lượt sinh; chỉ cần khi dùng --versus
Ghi  không gì cả; in bảng Markdown (dán thẳng vào Notion) hoặc TSV.

Ba bảng: Acc@4 (chỉ số chính, trung bình 4 lượt), Pass@4 (có ít nhất một lượt đúng), và bảng chẩn đoán (số token
trung bình, tỷ lệ lượt dừng, chạm trần, không có \\boxed). Có từ hai seed trở lên thì mỗi ô là trung bình ± độ lệch
chuẩn mẫu qua các seed. Cột "TB tầng 1" là trung bình cộng của bốn bộ tầng 1, chỉ in khi mô hình có đủ cả bốn.

Thư mục seed<seed>.limit<N> (phép đo tốc độ của --limit) bị bỏ qua. Mô hình nào thiếu bộ, thiếu seed, hoặc được
đánh giá với cài đặt khác các mô hình còn lại đều được nêu ở cuối: bảng không âm thầm so những thứ không so được.

--versus <tên>: với từng bộ, chênh lệch Acc@4 của mỗi mô hình so với mô hình mốc, tính GHÉP CẶP theo câu hỏi (hai
mô hình trả lời cùng một tập câu), kèm khoảng tin cậy 95% bằng bootstrap trên câu hỏi; chỉ dùng các seed mà cả
hai mô hình cùng có. Khoảng chứa 0 thì chênh lệch chưa phân biệt được với nhiễu. Bộ AIME chỉ có 30 câu: một lượt sinh đúng thêm đã là 0,83 điểm.
"""
from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.common.config import load_config, path_of, resolve_path
from src.common.io_utils import iter_jsonl, read_json

METRICS = ("acc@4", "pass@4")
BENCH_LABEL = {"math500": "MATH-500", "amc23": "AMC 2023", "aime24": "AIME 2024", "aime25": "AIME 2025",
               "gsm8k": "GSM8K", "bbh": "BBH", "val": "Kiểm định"}
ORDER = ["no_filter", "correct_only", "token_length", "grape", "local_naturalness", "rsr", "lark", "qd_rsr",
         "fit_only", "quality_only", "diversity_only", "fit_quality", "fit_diversity", "no_prefilter"]
MUST_MATCH = ("n_samples", "temperature", "top_p", "top_k", "max_new_tokens", "seed", "bbh_per_task")
_SEED_DIR = re.compile(r"seed(\d+)")


# ============================================================ đọc kết quả
def find_runs(root: Path) -> dict:
    """{tên tập chọn: {seed: summary}} cho mọi lần đánh giá đủ (bỏ thư mục .limitN)."""
    runs: dict[str, dict] = defaultdict(dict)
    for path in sorted(root.glob("*/*/summary.json")):
        m = _SEED_DIR.fullmatch(path.parent.name)
        if m:
            runs[path.parent.parent.name][int(m.group(1))] = {**read_json(path), "_dir": path.parent}
    return dict(runs)


def sort_key(tag: str) -> tuple:
    name = tag.split(".k", 1)[0]
    lam = re.search(r"\.lam([0-9.]+)", tag)
    return (ORDER.index(name) if name in ORDER else len(ORDER), tag.split(".lam")[0], float(lam.group(1)) if lam else -1.0)


def mean_std(values: Sequence[float]) -> tuple[float, float | None]:
    """Trung bình và độ lệch chuẩn MẪU (chia n − 1); một giá trị thì không có độ lệch chuẩn."""
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None)


def cell(values: Sequence[float], scale: float = 100.0, digits: int = 1) -> str:
    if not values:
        return "—"
    m, s = mean_std([v * scale for v in values])
    return f"{m:.{digits}f}" + (f" ± {s:.{digits}f}" if s is not None else "")


def collect(runs: Mapping[str, Mapping[int, Mapping]], seeds: Sequence[int] | None, benches: Sequence[str],
            key: str) -> dict:
    """{tên: {bộ: [giá trị của từng seed]}} cho một trường của summary (acc@4, mean_new_tokens, ...)."""
    out: dict[str, dict] = {}
    for tag, by_seed in runs.items():
        out[tag] = {b: [s["benchmarks"][b][key] for sd, s in sorted(by_seed.items())
                        if (seeds is None or sd in seeds) and b in s.get("benchmarks", {}) and key in s["benchmarks"][b]]
                    for b in benches}
    return out


def tier1_mean(runs: Mapping[int, Mapping], seeds: Sequence[int] | None, tier1: Sequence[str], key: str) -> list[float]:
    """Trung bình cộng của các bộ tầng 1 cho từng seed có đủ mọi bộ đó."""
    out = []
    for sd, s in sorted(runs.items()):
        b = s.get("benchmarks", {})
        if (seeds is None or sd in seeds) and all(t in b and key in b[t] for t in tier1):
            out.append(statistics.mean(b[t][key] for t in tier1))
    return out


# ============================================================ so ghép cặp theo câu hỏi
def question_acc(path: Path, n_samples: int) -> dict[str, float]:
    """Tỷ lệ lượt đúng của từng câu hỏi trong một file <bộ>.jsonl. Câu chưa đủ lượt thì bỏ."""
    hits: dict[str, dict] = defaultdict(dict)
    for r in iter_jsonl(path):
        if r["sample"] < n_samples:
            hits[r["qid"]][r["sample"]] = bool(r["correct"])
    return {q: sum(v.values()) / n_samples for q, v in hits.items() if len(v) == n_samples}


def pooled_question_acc(by_seed: Mapping[int, Mapping], seeds: Sequence[int] | None, bench: str, n_samples: int) -> dict:
    """Tỷ lệ đúng của từng câu, lấy trung bình qua các seed đã có kết quả cho bộ này."""
    per_seed = [question_acc(s["_dir"] / f"{bench}.jsonl", n_samples) for sd, s in sorted(by_seed.items())
                if (seeds is None or sd in seeds) and (s["_dir"] / f"{bench}.jsonl").exists()]
    if not per_seed:
        return {}
    common = set.intersection(*(set(p) for p in per_seed))
    return {q: statistics.mean(p[q] for p in per_seed) for q in common}


def paired_bootstrap(a: Mapping[str, float], b: Mapping[str, float], n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Chênh lệch trung bình a − b trên các câu chung (thang 0 đến 1) và khoảng tin cậy 95% bằng bootstrap câu hỏi."""
    qids = sorted(set(a) & set(b))
    if len(qids) < 2:
        return None
    d = np.array([a[q] - b[q] for q in qids])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(qids), "diff": float(d.mean()), "lo": float(lo), "hi": float(hi)}


# ============================================================ in bảng
def render(header: Sequence[str], rows: Sequence[Sequence[str]], fmt: str) -> str:
    if fmt == "tsv":
        return "\n".join("\t".join(r) for r in [header, *rows])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def check_settings(runs: Mapping[str, Mapping[int, Mapping]], seeds: Sequence[int] | None) -> list[str]:
    """Cảnh báo khi các mô hình được đánh giá với cài đặt không so được với nhau."""
    seen: dict[tuple, list] = defaultdict(list)
    for tag, by_seed in runs.items():
        for sd, s in by_seed.items():
            if seeds is None or sd in seeds:
                st = s.get("settings", {})
                seen[tuple(repr(st.get(k)) for k in MUST_MATCH)].append(f"{tag} seed {sd}")
    if len(seen) <= 1:
        return []
    out = ["CÀI ĐẶT ĐÁNH GIÁ KHÁC NHAU giữa các mô hình (" + ", ".join(MUST_MATCH) + "); các ô không so được với nhau:"]
    for vals, who in seen.items():
        out.append(f"  {', '.join(vals)}: {', '.join(who[:6])}" + (f" và {len(who) - 6} lần khác" if len(who) > 6 else ""))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="qwen1_5b")
    ap.add_argument("--root", help="thư mục kết quả, mặc định outputs/results/eval/<student>_base")
    ap.add_argument("--seeds", type=int, nargs="+", help="chỉ dùng các seed này (mặc định mọi seed có kết quả)")
    ap.add_argument("--benchmarks", nargs="+", help="chỉ in các bộ này (mặc định mọi bộ có kết quả)")
    ap.add_argument("--only", nargs="+", metavar="TÊN", help="chỉ in các tập chọn có tên bắt đầu bằng một trong các chuỗi này")
    ap.add_argument("--versus", metavar="TÊN", help="tên tập chọn làm mốc cho phép so ghép cặp, ví dụ correct_only.k3")
    ap.add_argument("--boot", type=int, default=2000, help="số lần lấy mẫu lại của bootstrap")
    ap.add_argument("--format", choices=["md", "tsv"], default="md")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(student=args.student, overrides=args.override)
    root = resolve_path(cfg, args.root) if args.root else path_of(cfg, "out_results") / "eval" / f"{cfg['student']['key']}_base"
    runs = find_runs(root)
    if args.only:
        runs = {t: v for t, v in runs.items() if any(t.startswith(p) for p in args.only)}
    if args.seeds:
        runs = {t: v for t, v in runs.items() if any(sd in args.seeds for sd in v)}
    if not runs:
        raise SystemExit(f"Không thấy summary.json nào trong {root} (cấu trúc <tên>/seed<seed>/summary.json).")
    tags = sorted(runs, key=sort_key)
    tier1 = list(cfg["eval"]["tiers"][0])
    known = [b for t in cfg["eval"]["tiers"] for b in t] + ["val"]
    present = {b for v in runs.values() for s in v.values() for b in s.get("benchmarks", {})}
    benches = [b for b in (args.benchmarks or known + sorted(present - set(known))) if b in present]
    if not benches:
        raise SystemExit(f"Không mô hình nào có kết quả cho {args.benchmarks or 'bộ nào'}; đang có: {sorted(present)}")
    show_avg = all(b in benches for b in tier1)
    seeds_of = {t: sorted(sd for sd in runs[t] if args.seeds is None or sd in args.seeds) for t in tags}
    n_seeds = sorted({len(v) for v in seeds_of.values()})
    labels = [BENCH_LABEL.get(b, b) for b in benches]
    n_q = {b: next((s["benchmarks"][b].get("n_questions") for v in runs.values() for s in v.values()
                    if b in s.get("benchmarks", {})), None) for b in benches}

    print(f"Kết quả ở {root}")
    print(f"{len(tags)} mô hình; seed: " + "; ".join(sorted({', '.join(map(str, v)) for v in seeds_of.values()}))
          + (" (ô là trung bình ± độ lệch chuẩn mẫu qua các seed)" if n_seeds[-1] > 1 else " (một seed, chưa có độ lệch chuẩn)"))
    print("Số câu mỗi bộ: " + ", ".join(f"{BENCH_LABEL.get(b, b)} {n_q[b]}" for b in benches))

    for metric in METRICS:
        data = collect(runs, args.seeds, benches, metric)
        rows = []
        for t in tags:
            row = [t] + [cell(data[t][b]) for b in benches]
            if show_avg:
                row.append(cell(tier1_mean(runs[t], args.seeds, tier1, metric)))
            rows.append(row + [str(len(seeds_of[t]))])
        name = "Acc@4 (%), chỉ số chính" if metric == "acc@4" else "Pass@4 (%)"
        print(f"\n### {name}\n")
        print(render(["Phương án", *labels, *(["TB tầng 1"] if show_avg else []), "Số seed"], rows, args.format))

    print("\n### Chẩn đoán: token trung bình mỗi lượt / % lượt dừng / % chạm trần / % không có \\boxed\n")
    diag = {k: collect(runs, args.seeds, benches, k) for k in ("mean_new_tokens", "stopped_share", "hit_cap_share", "no_boxed_share")}
    rows = []
    for t in tags:
        row = [t]
        for b in benches:
            if not diag["mean_new_tokens"][t][b]:
                row.append("—")
                continue
            row.append(f"{statistics.mean(diag['mean_new_tokens'][t][b]):.0f} / "
                       + " / ".join(f"{100 * statistics.mean(diag[k][t][b]):.0f}"
                                    for k in ("stopped_share", "hit_cap_share", "no_boxed_share")))
        rows.append(row)
    print(render(["Phương án", *labels], rows, args.format))

    if args.versus:
        if args.versus not in runs:
            raise SystemExit(f"--versus {args.versus}: không có mô hình này. Đang có: {', '.join(tags)}")
        k = int(cfg["eval"]["n_samples"])
        rows = []
        for t in tags:
            if t == args.versus:
                continue
            shared = sorted(set(seeds_of[t]) & set(seeds_of[args.versus]))     # chỉ so trên các seed cả hai cùng có
            row = [t]
            for b in benches:
                r = paired_bootstrap(pooled_question_acc(runs[t], shared, b, k),
                                     pooled_question_acc(runs[args.versus], shared, b, k), args.boot) if shared else None
                if r is None:
                    row.append("—")
                else:
                    mark = " *" if r["lo"] > 0 or r["hi"] < 0 else ""
                    row.append(f"{100 * r['diff']:+.1f} [{100 * r['lo']:+.1f}; {100 * r['hi']:+.1f}]{mark}")
            rows.append(row + [", ".join(map(str, shared)) or "—"])
        print(f"\n### Chênh lệch Acc@4 so với {args.versus} (điểm %), ghép cặp theo câu hỏi, khoảng tin cậy 95% bootstrap\n")
        print(render(["Phương án", *labels, "Seed dùng chung"], rows, args.format))
        print("\nDấu * là khoảng tin cậy không chứa 0. Không có dấu * thì chênh lệch chưa phân biệt được với nhiễu "
              "của tập câu hỏi; bootstrap này chưa tính nhiễu giữa các seed huấn luyện.")

    notes = check_settings(runs, args.seeds)
    for t in tags:
        lacking = [BENCH_LABEL.get(b, b) for b in benches if len(collect({t: runs[t]}, args.seeds, [b], "acc@4")[t][b]) < len(seeds_of[t])]
        if lacking:
            notes.append(f"{t}: thiếu kết quả ở {', '.join(lacking)} cho ít nhất một seed")
    if len(n_seeds) > 1:
        notes.append("Số seed khác nhau giữa các mô hình (xem cột Số seed): ô một seed không có độ lệch chuẩn.")
    commits = {s.get("eval_commit") for v in runs.values() for s in v.values()}
    if len(commits) > 1:
        notes.append(f"Các lần đánh giá chạy ở nhiều commit khác nhau: {sorted(map(str, commits))}")
    res = [f"{BENCH_LABEL.get(b, b)} {100 / (n_q[b] * int(cfg['eval']['n_samples'])):.2f}" for b in benches if n_q[b]]
    print("\nMột lượt sinh đúng thêm làm Acc@4 đổi (điểm %): " + ", ".join(res))
    if notes:
        print("\nLƯU Ý:")
        for n in notes:
            print(f"- {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

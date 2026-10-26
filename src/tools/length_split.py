"""length_split: tách phần chênh lệch Acc@4 do lượt chạm trần độ dài khỏi phần còn lại. Chỉ đọc file đánh giá.

    python -m src.tools.length_split                                         # hai phép so mặc định, MATH-500, mọi seed
    python -m src.tools.length_split --seeds 42 43 44 --benchmarks math500 amc23
    python -m src.tools.length_split --pair fq_select.k3 quality_only.k3     # tự chọn phép so, nhắc lại được nhiều lần

Đọc  outputs/results/eval/<student>_base/<tên>/seed<seed>/summary.json và <bộ>.jsonl (c1_evaluate ghi)
Ghi  không gì cả; in bảng Markdown (dán thẳng vào Notion). Không cần GPU, không cần torch, không gọi mạng.

Vì sao cần. Lượt sinh chạm trần eval.max_new_tokens bị cắt và vẫn chấm như bình thường, thường là sai vì chưa kịp viết
\\boxed. Quality-only viết dài hơn (khoảng 905 đến 925 token, chạm trần 10 đến 11%) so với FQ-Select (khoảng 685 token,
4 đến 5%), nên một phần của chênh lệch FQ-Select hơn Quality-only có thể chỉ là Quality-only bị cắt nhiều hơn chứ không
suy luận kém hơn. Cùng câu hỏi đặt cho Quality-only với Correct-Only theo chiều ngược lại.

Cách tách. Với mỗi phép so A với B, các câu chung được chia theo ĐIỀU KIỆN CẮT:
    câu không bị cắt   không lượt nào của A và của B (mọi seed đang dùng) chạm trần;
    câu có bị cắt      có ít nhất một lượt của A hoặc của B chạm trần.
Chênh lệch trung bình của mọi câu bằng ĐÚNG trung bình có trọng số của hai nhóm (trọng số là tỷ lệ câu), nên báo cáo in
phần đóng góp của từng nhóm và hai phần cộng lại bằng tổng:
    Δ tất cả = (n sạch / n) · Δ sạch  +  (n cắt / n) · Δ cắt
Phần "câu không bị cắt" là chênh lệch mà việc cắt độ dài không thể giải thích. Chênh lệch ghép cặp theo câu hỏi, khoảng
tin cậy 95% bằng bootstrap trên câu hỏi (cùng hàm paired_bootstrap với eval_table), nên đọc được cạnh bảng của eval_table.

Hai điều cần nhớ khi đọc. (1) Nhóm "không bị cắt" được chọn SAU khi biết kết quả sinh, và câu mà cả hai mô hình viết ngắn
thường là câu dễ; vì vậy Δ sạch nhỏ hơn Δ tất cả chưa phải bằng chứng rằng chênh lệch chỉ do độ dài. Hãy đọc nó cùng tỷ lệ
chạm trần và tỷ lệ đúng trong các lượt chạm trần ở bảng dưới. (2) Khoảng tin cậy chỉ tính nhiễu của tập câu hỏi, chưa tính
nhiễu giữa các seed huấn luyện. Chạy riêng từng seed và trung bình ba seed đều được in để thấy chênh lệch có nhất quán không.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from typing import Mapping, Sequence

from src.common.config import load_config, path_of, resolve_path
from src.common.io_utils import iter_jsonl
from src.tools.eval_table import BENCH_LABEL, MUST_MATCH, find_runs, paired_bootstrap

DEFAULT_PAIRS = (("fq_select.k3", "quality_only.k3"), ("quality_only.k3", "correct_only.k3"))


# ============================================================ đọc từng lượt sinh
def is_capped(row: Mapping, cap: int) -> bool:
    """Lượt chạm trần: chưa dừng ở token kết thúc mà đã đủ cap token. Đúng định nghĩa hit_cap_share của c1_evaluate."""
    return (not row["stopped"]) and int(row["n_tokens"]) >= cap


def load_generations(path, n_samples: int, cap: int) -> dict[str, dict]:
    """qid -> {correct, capped, tokens}, mỗi trường là danh sách n_samples phần tử theo thứ tự lượt. Câu chưa đủ lượt bị bỏ."""
    got: dict[str, dict] = defaultdict(dict)
    for r in iter_jsonl(path):
        if r["sample"] < n_samples:
            got[r["qid"]][r["sample"]] = (bool(r["correct"]), is_capped(r, cap), int(r["n_tokens"]))
    out = {}
    for qid, by_sample in got.items():
        if len(by_sample) == n_samples:
            rows = [by_sample[i] for i in range(n_samples)]
            out[qid] = {"correct": [x[0] for x in rows], "capped": [x[1] for x in rows], "tokens": [x[2] for x in rows]}
    return out


def check_comparable(summary_a: Mapping, summary_b: Mapping, names: Sequence[str]) -> None:
    """Hai mô hình phải được đánh giá với cùng cài đặt (số lượt, nhiệt độ, trần token, seed), nếu không thì không so ghép cặp."""
    sa, sb = summary_a.get("settings", {}), summary_b.get("settings", {})
    diff = [f"{k}: {sa.get(k)!r} so {sb.get(k)!r}" for k in MUST_MATCH if sa.get(k) != sb.get(k)]
    if diff:
        raise SystemExit(f"{names[0]} và {names[1]} được đánh giá với cài đặt khác nhau ({'; '.join(diff)}); "
                         f"không so ghép cặp được.")


# ============================================================ tách
def _mean(xs: Sequence[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def question_table(gens_by_seed: Mapping[int, Mapping[str, Mapping]], seeds: Sequence[int]) -> dict:
    """Mỗi câu có mặt ở MỌI seed đang dùng: Acc@4 trung bình qua seed và cờ 'có lượt chạm trần ở một seed nào đó'."""
    common = set.intersection(*(set(gens_by_seed[s]) for s in seeds)) if seeds else set()
    return {q: {"acc": statistics.mean(statistics.mean(gens_by_seed[s][q]["correct"]) for s in seeds),
                "capped": any(any(gens_by_seed[s][q]["capped"]) for s in seeds)} for q in common}


def generation_stats(gens_by_seed: Mapping[int, Mapping[str, Mapping]], seeds: Sequence[int],
                     qids: Sequence[str]) -> dict:
    """Số liệu mức lượt sinh trên các câu đã dùng: token trung bình, tỷ lệ chạm trần, và tỷ lệ đúng trong các lượt chạm trần."""
    toks, capped, capped_ok = [], 0, 0
    n = 0
    for s in seeds:
        for q in qids:
            g = gens_by_seed[s][q]
            for t, c, ok in zip(g["tokens"], g["capped"], g["correct"]):
                toks.append(t)
                n += 1
                if c:
                    capped += 1
                    capped_ok += ok
    return {"generations": n, "mean_tokens": _mean(toks), "capped_share": capped / n if n else None,
            "capped": capped, "capped_correct_share": capped_ok / capped if capped else None}


def split_pair(gens_a: Mapping[int, Mapping], gens_b: Mapping[int, Mapping], seeds: Sequence[int],
               n_boot: int = 2000) -> dict:
    """Tách chênh lệch Acc@4 của A so với B (A trừ B) theo điều kiện cắt, trên các câu chung của mọi seed đang dùng."""
    ta, tb = question_table(gens_a, seeds), question_table(gens_b, seeds)
    qids = sorted(set(ta) & set(tb))
    cut = {q: ta[q]["capped"] or tb[q]["capped"] for q in qids}
    groups = {"all": qids, "clean": [q for q in qids if not cut[q]], "cut": [q for q in qids if cut[q]]}
    out = {"seeds": list(seeds), "questions": len(qids)}
    for name, group in groups.items():
        a = {q: ta[q]["acc"] for q in group}
        b = {q: tb[q]["acc"] for q in group}
        r = paired_bootstrap(a, b, n_boot) if group else None
        out[name] = {"n": len(group), "share": len(group) / len(qids) if qids else None,
                     "acc_a": _mean(list(a.values())), "acc_b": _mean(list(b.values())),
                     "diff": _mean([a[q] - b[q] for q in group]), "lo": r["lo"] if r else None, "hi": r["hi"] if r else None}
    # phần đóng góp vào tổng; nhóm rỗng đóng góp 0
    for name in ("clean", "cut"):
        g = out[name]
        g["contribution"] = g["share"] * g["diff"] if g["n"] else 0.0
    out["gen_a"] = generation_stats(gens_a, seeds, qids)
    out["gen_b"] = generation_stats(gens_b, seeds, qids)
    return out


# ============================================================ in bảng
def _pp(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}"


def _diff(g: Mapping) -> str:
    if g["diff"] is None:
        return "n/a"
    if g["lo"] is None:
        return f"{100 * g['diff']:+.1f}"
    mark = " *" if g["lo"] > 0 or g["hi"] < 0 else ""
    return f"{100 * g['diff']:+.1f} [{100 * g['lo']:+.1f}; {100 * g['hi']:+.1f}]{mark}"


def render(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def print_split(a_name: str, b_name: str, bench: str, res: Mapping) -> None:
    seeds = ", ".join(map(str, res["seeds"]))
    avg = " (trung bình qua seed)" if len(res["seeds"]) > 1 else ""
    print(f"\n#### {a_name} so với {b_name}, {BENCH_LABEL.get(bench, bench)}, seed {seeds}{avg}, {res['questions']} câu chung\n")
    rows = []
    for key, label in (("all", "Mọi câu"), ("clean", "Câu không bị cắt"), ("cut", "Câu có lượt chạm trần")):
        g = res[key]
        part = "" if key == "all" else f"{100 * g['contribution']:+.1f}"
        rows.append([label, str(g["n"]), _pp(g["share"], 0) + "%" if g["share"] is not None else "n/a",
                     _pp(g["acc_a"]), _pp(g["acc_b"]), _diff(g), part])
    print(render(["Nhóm câu", "Số câu", "Tỷ lệ", f"Acc@4 {a_name}", f"Acc@4 {b_name}", "Δ [95%]", "Đóng góp vào Δ tất cả"], rows))
    total = res["all"]["diff"]
    parts = res["clean"]["contribution"] + res["cut"]["contribution"]
    if total is not None:
        print(f"\nΔ tất cả {100 * total:+.1f} = phần câu không bị cắt {100 * res['clean']['contribution']:+.1f} "
              f"+ phần câu có lượt chạm trần {100 * res['cut']['contribution']:+.1f}"
              + ("" if abs(parts - total) < 1e-9 else f"   (LỆCH {100 * (parts - total):+.3f}: báo lỗi)"))
    gr = []
    for name, key in ((a_name, "gen_a"), (b_name, "gen_b")):
        g = res[key]
        gr.append([name, f"{g['mean_tokens']:.0f}" if g["mean_tokens"] is not None else "n/a",
                   _pp(g["capped_share"]) + "%", str(g["capped"]),
                   _pp(g["capped_correct_share"]) + "%" if g["capped_correct_share"] is not None else "n/a"])
    print("\nMức lượt sinh trên các câu chung:\n")
    print(render(["Mô hình", "Token TB", "Chạm trần", "Số lượt chạm trần", "Đúng trong lượt chạm trần"], gr))


# ============================================================ dòng lệnh
def run_pair(runs: Mapping, a: str, b: str, bench: str, seeds_arg: Sequence[int] | None, n_samples: int,
             n_boot: int) -> list[dict]:
    """Các kết quả tách của một phép so: từng seed riêng, rồi trung bình mọi seed chung (nếu có từ hai seed)."""
    for name in (a, b):
        if name not in runs:
            raise SystemExit(f"Không có kết quả của {name}. Đang có: {', '.join(sorted(runs))}")
    shared = sorted((set(runs[a]) & set(runs[b])) & (set(seeds_arg) if seeds_arg else set(runs[a]) & set(runs[b])))
    gens: dict[str, dict[int, dict]] = {a: {}, b: {}}
    used = []
    for s in shared:
        sa, sb = runs[a][s], runs[b][s]
        check_comparable(sa, sb, (f"{a} seed {s}", f"{b} seed {s}"))
        cap = int(sa["settings"]["max_new_tokens"]) if "max_new_tokens" in sa.get("settings", {}) else None
        if cap is None:
            raise SystemExit(f"{a} seed {s}: summary.json không ghi settings.max_new_tokens; không biết trần token.")
        files = [sm["_dir"] / f"{bench}.jsonl" for sm in (sa, sb)]
        if bench not in sa.get("benchmarks", {}) or bench not in sb.get("benchmarks", {}) or not all(f.exists() for f in files):
            print(f"[chú ý] bỏ seed {s} của {a} so {b}: thiếu kết quả {bench}")
            continue
        gens[a][s] = load_generations(files[0], n_samples, cap)
        gens[b][s] = load_generations(files[1], n_samples, cap)
        used.append(s)
    if not used:
        raise SystemExit(f"{a} và {b} không có seed chung nào có kết quả {bench}.")
    out = [split_pair(gens[a], gens[b], [s], n_boot) for s in used]
    if len(used) > 1:
        out.append(split_pair(gens[a], gens[b], used, n_boot))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="qwen1_5b")
    ap.add_argument("--root", help="thư mục kết quả, mặc định outputs/results/eval/<student>_base")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"),
                    help="phép so A trừ B, tên tập chọn như fq_select.k3; nhắc lại được. Mặc định: hai phép so của H2")
    ap.add_argument("--benchmarks", nargs="+", default=["math500"])
    ap.add_argument("--seeds", type=int, nargs="+", help="chỉ dùng các seed này (mặc định mọi seed cả hai cùng có)")
    ap.add_argument("--boot", type=int, default=2000, help="số lần lấy mẫu lại của bootstrap")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(student=args.student, overrides=args.override)
    root = resolve_path(cfg, args.root) if args.root else path_of(cfg, "out_results") / "eval" / f"{cfg['student']['key']}_base"
    runs = find_runs(root)
    if not runs:
        raise SystemExit(f"Không thấy summary.json nào trong {root} (cấu trúc <tên>/seed<seed>/summary.json).")
    pairs = [tuple(p) for p in args.pair] if args.pair else list(DEFAULT_PAIRS)
    k = int(cfg["eval"]["n_samples"])

    print(f"Kết quả ở {root}")
    print("Δ là A trừ B, điểm phần trăm của Acc@4, ghép cặp theo câu hỏi; * là khoảng tin cậy 95% không chứa 0.")
    for bench in args.benchmarks:
        for a, b in pairs:
            for res in run_pair(runs, a, b, bench, args.seeds, k, args.boot):
                print_split(a, b, bench, res)
    print("\nĐọc kèm: nhóm 'câu không bị cắt' được chọn sau khi biết kết quả sinh và thiên về câu dễ, nên Δ của nó nhỏ hơn "
          "Δ tất cả chưa chứng minh chênh lệch chỉ do độ dài. Khoảng tin cậy chưa tính nhiễu giữa các seed huấn luyện; "
          "xem từng seed và trung bình để thấy chênh lệch có nhất quán không.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

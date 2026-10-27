"""diagnose_fq: mổ xẻ vì sao FQ-Select không hơn GRAPE. Chỉ đọc file, không cần GPU, không gọi mạng.

    python -m src.tools.diagnose_fq --student qwen1_5b
    python -m src.tools.diagnose_fq --student qwen1_5b --pair grape.k3 fq_select.k3 --pair fq_select.k3 quality_only.k3

Đọc  data/stage_a/candidates.jsonl, quality.jsonl, fit.<student>_base.jsonl, embeddings.npz (qua b2_select.Inputs,
     nên kho chuỗi, thứ tự cố định và cách tính hạng GIỐNG HỆT lúc chọn)
     data/stage_b/<student>_base/train.<tên>.jsonl của các tập cần so
Ghi  không gì cả; in bảng Markdown.

Ba câu hỏi, sinh ra sau buổi báo cáo giữa kỳ 25/10 (thầy hỏi vấn đề nằm ở đâu):
  1. Fam và Qual có đo cùng một thứ không. Spearman TRONG TỪNG CÂU giữa hai tín hiệu, và với hai nửa của Qual
     (điểm quy tắc, điểm giám khảo). Tương quan âm nghĩa là cộng hai thứ hạng thì chúng triệt tiêu nhau.
  2. Tín hiệu nào thưởng độ dài. Spearman trong từng câu giữa mỗi tín hiệu và số token của mô hình học.
  3. Mỗi mô hình dạy đứng ở đâu theo từng tín hiệu: hạng chuẩn hoá trung bình (1 tốt nhất, 0 kém nhất).
  4. Hai tập chọn khác nhau ở đâu: với từng cặp, chuỗi bị bỏ và chuỗi được thêm đến từ mô hình dạy nào, dài bao nhiêu,
     có hạng Fam và Qual thế nào.
Chỉ tính trên các câu có từ 4 chuỗi đúng trở lên, vì câu có đúng 3 chuỗi thì mọi phương án lấy hết, không có lựa chọn.
Đây là phân tích MÔ TẢ sau khi đã biết kết quả, không phải phép kiểm định giả thuyết đặt trước.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.common.config import load_config, path_of, resolve_path
from src.common.io_utils import read_jsonl
from src.stage_b.b2_select import Inputs, norm_ranks, signal_values, student_tag
from src.tools.compare_judges import spearman

DEFAULT_PAIRS = (("grape.k3", "fq_select.k3"), ("fq_select.k3", "quality_only.k3"), ("grape.k3", "quality_only.k3"))
CORR_PAIRS = (("fam", "qual"), ("fam", "rule"), ("fam", "llm"), ("rule", "llm"),
              ("fam", "len"), ("qual", "len"), ("rule", "len"), ("llm", "len"))


def columns(q: Mapping, quality: Mapping[str, Mapping]) -> dict:
    """Các cột của một câu, theo đúng thứ tự chuỗi trong q. Mọi cột theo chiều 'càng cao càng được chọn', trừ len."""
    return {"fam": np.asarray(signal_values(q, "grape"), dtype=float),
            "qual": np.asarray(q["qual"], dtype=float),
            "rule": np.array([float(quality[t]["rule_score"]) for t in q["tids"]]),
            "llm": np.array([float(quality[t]["llm_score"]) for t in q["tids"]]),
            "len": np.array([float(x) for x in q["n_tokens"]])}


def within_corr(questions: Mapping[str, Mapping], quality: Mapping[str, Mapping], min_n: int = 4) -> dict:
    """Spearman trong từng câu cho mỗi cặp cột: trung vị, trung bình, tỷ lệ âm, số câu tính được."""
    vals: dict[tuple, list] = defaultdict(list)
    for q in questions.values():
        if len(q["tids"]) < min_n:
            continue
        c = columns(q, quality)
        for a, b in CORR_PAIRS:
            r = spearman(list(c[a]), list(c[b]))
            if r is not None:
                vals[(a, b)].append(r)
    return {p: {"median": statistics.median(v), "mean": statistics.fmean(v),
                "neg": sum(x < 0 for x in v) / len(v), "n": len(v)} for p, v in vals.items() if v}


def teacher_ranks(questions: Mapping[str, Mapping], quality: Mapping[str, Mapping], min_n: int = 4) -> dict:
    """Hạng chuẩn hoá trung bình của chuỗi mỗi mô hình dạy theo từng tín hiệu, và độ dài trung bình."""
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for q in questions.values():
        if len(q["tids"]) < min_n:
            continue
        c = columns(q, quality)
        ranks = {k: norm_ranks(c[k], "max") for k in ("fam", "qual", "rule", "llm")}
        for i, t in enumerate(q["teacher"]):
            for k, r in ranks.items():
                acc[t][k].append(float(r[i]))
            acc[t]["len"].append(float(c["len"][i]))
    return {t: {k: statistics.fmean(v) for k, v in d.items()} | {"n": len(d["fam"])} for t, d in acc.items()}


def chain_index(questions: Mapping[str, Mapping], quality: Mapping[str, Mapping]) -> dict:
    """tid -> teacher, số token, hạng Fam, hạng Qual trong câu của nó."""
    out = {}
    for q in questions.values():
        c = columns(q, quality)
        rf, rq = norm_ranks(c["fam"], "max"), norm_ranks(c["qual"], "max")
        for i, t in enumerate(q["tids"]):
            out[t] = {"qid": q["qid"], "teacher": q["teacher"][i], "len": float(c["len"][i]),
                      "fam": float(rf[i]), "qual": float(rq[i]), "n": len(q["tids"])}
    return out


def tid_sets(rows: Sequence[Mapping]) -> dict:
    by_q: dict[str, set] = defaultdict(set)
    for r in rows:
        by_q[r["qid"]].add(r["tid"])
    return by_q


def swap(a: Mapping[str, set], b: Mapping[str, set], index: Mapping[str, Mapping], min_n: int = 4) -> dict:
    """So tập A với tập B: chuỗi chỉ A có (bị bỏ khi chuyển A sang B) và chuỗi chỉ B có (được thêm)."""
    qids = sorted(q for q in set(a) & set(b) if index[next(iter(a[q]))]["n"] >= min_n)
    dropped = [t for q in qids for t in a[q] - b[q]]
    added = [t for q in qids for t in b[q] - a[q]]

    def stats(tids: Sequence[str]) -> dict:
        if not tids:
            return {"n": 0}
        rows = [index[t] for t in tids]
        return {"n": len(rows), "teacher": Counter(r["teacher"] for r in rows),
                **{k: statistics.fmean(r[k] for r in rows) for k in ("len", "fam", "qual")}}

    def share(s: Mapping[str, set]) -> Counter:
        return Counter(index[t]["teacher"] for q in qids for t in s[q])

    return {"questions": len(qids), "identical": sum(a[q] == b[q] for q in qids),
            "share_a": share(a), "share_b": share(b), "dropped": stats(dropped), "added": stats(added)}


def pct(c: Counter, teachers: Sequence[str]) -> str:
    tot = sum(c.values()) or 1
    return ", ".join(f"{t} {100 * c[t] / tot:.1f}%" for t in teachers)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--student", default="qwen1_5b")
    ap.add_argument("--min-n", type=int, default=4, help="chỉ tính câu có ít nhất chừng này chuỗi đúng")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"),
                    help="so hai tập, tên như trong train.<tên>.jsonl; nhắc lại được nhiều lần")
    args = ap.parse_args(argv)

    cfg = load_config(student=args.student)
    inp = Inputs(cfg, resolve_path(cfg, args.workdir), f"fit.{student_tag(cfg)}.jsonl")
    qs = inp.questions("correct", "rank_sum")
    teachers = sorted({t for q in qs.values() for t in q["teacher"]})
    many = sum(len(q["tids"]) >= args.min_n for q in qs.values())
    print(f"# diagnose_fq ({inp.fit_name})\n\n{len(qs)} câu, {many} câu có từ {args.min_n} chuỗi đúng trở lên "
          f"(chỉ các câu này có lựa chọn và được tính dưới đây).\n")

    print("## 1-2. Spearman trong từng câu\n\n| Cặp | Trung vị | Trung bình | Tỷ lệ âm | Số câu |\n|---|---|---|---|---|")
    for (a, b), s in within_corr(qs, inp.quality, args.min_n).items():
        print(f"| {a} ~ {b} | {s['median']:+.3f} | {s['mean']:+.3f} | {100 * s['neg']:.1f}% | {s['n']} |")

    print("\n## 3. Hạng chuẩn hoá trung bình theo mô hình dạy (1 = tốt nhất trong câu)\n\n"
          "| Mô hình dạy | Số chuỗi | Fam | Qual | Quy tắc | Giám khảo | Token TB |\n|---|---|---|---|---|---|---|")
    for t, s in sorted(teacher_ranks(qs, inp.quality, args.min_n).items()):
        print(f"| {t} | {s['n']} | {s['fam']:.3f} | {s['qual']:.3f} | {s['rule']:.3f} | {s['llm']:.3f} | {s['len']:.0f} |")

    index = chain_index(qs, inp.quality)
    outdir = path_of(cfg, "data_stage_b") / student_tag(cfg)
    print("\n## 4. Hai tập chọn khác nhau ở đâu")
    for a, b in (args.pair or DEFAULT_PAIRS):
        pa, pb = outdir / f"train.{a}.jsonl", outdir / f"train.{b}.jsonl"
        missing = [str(p) for p in (pa, pb) if not Path(p).exists()]
        if missing:
            print(f"\n### {a} → {b}\n\nThiếu {', '.join(missing)}, bỏ qua.")
            continue
        s = swap(tid_sets(read_jsonl(pa)), tid_sets(read_jsonl(pb)), index, args.min_n)
        print(f"\n### {a} → {b}\n\n{s['questions']} câu, {s['identical']} câu "
              f"({100 * s['identical'] / max(1, s['questions']):.1f}%) chọn giống hệt.\n\n"
              f"- Thành phần {a}: {pct(s['share_a'], teachers)}\n- Thành phần {b}: {pct(s['share_b'], teachers)}\n")
        print("| | Số chuỗi | Mô hình dạy | Token TB | Hạng Fam | Hạng Qual |\n|---|---|---|---|---|---|")
        for label, d in (("bị bỏ", s["dropped"]), ("được thêm", s["added"])):
            if d["n"]:
                print(f"| {label} | {d['n']} | {pct(d['teacher'], teachers)} | {d['len']:.0f} | "
                      f"{d['fam']:.3f} | {d['qual']:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Chuyển hai file câu hỏi pilot cũ sang lược đồ mới để chạy lại pilot bằng prompt \\boxed{} (Notion mục R1).

    python -m src.tools.convert_old_pilot
    python -m src.tools.convert_old_pilot --workdir data/pilot/run2_boxed

Giữ NGUYÊN qid cũ (gsm8k_pilot_000, mathpilot_000) để so sánh đúng trên những câu đã đo 96,7% và 62,7%.
Ghi <workdir>/questions.jsonl, đã xáo trộn cố định (seed 42) để --limit-questions luôn lấy mẫu trộn GSM8K và MATH.
Không đụng vào các file pilot cũ.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Mapping, Sequence

from src.common.answers import extract_gold
from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl, write_jsonl
from src.stage_a.a1_prepare import parse_level


def convert(gsm_rows: Sequence[Mapping], math_rows: Sequence[Mapping], seed: int = 42) -> tuple[list[dict], list[str]]:
    out, warnings = [], []
    for r in gsm_rows:
        gold = extract_gold("gsm8k", r["solution"])
        if gold is None:
            raise SystemExit(f"{r['qid']}: không tách được đáp án sau '####'")
        if "answer" in r and str(r["answer"]).replace(",", "").strip() != gold:
            warnings.append(f"{r['qid']}: trường answer cũ ({r['answer']!r}) khác đáp án tách từ lời giải ({gold!r}); dùng đáp án tách từ lời giải")
        out.append({"qid": r["qid"], "source": "gsm8k", "level": None, "orig_index": r.get("orig_index"),
                    "question": r["question"], "solution": r["solution"], "gold": gold})
    for r in math_rows:
        gold = extract_gold("math", r["solution"])
        if gold is None:
            raise SystemExit(f"{r['qid']}: lời giải không có \\boxed hợp lệ, không chấm được")
        out.append({"qid": r["qid"], "source": "math", "level": parse_level(r.get("level")), "type": r.get("type"),
                    "question": r["question"], "solution": r["solution"], "gold": gold})
    qids = [r["qid"] for r in out]
    if len(qids) != len(set(qids)):
        raise SystemExit("Có qid trùng giữa hai file pilot.")
    random.Random(seed).shuffle(out)
    return out, warnings


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gsm", default="data/pilot/questions.jsonl")
    ap.add_argument("--math", default="data/pilot/questions_math.jsonl")
    ap.add_argument("--workdir", default="data/pilot/run2_boxed")
    ap.add_argument("--force", action="store_true", help="ghi đè questions.jsonl đã có trong workdir")
    args = ap.parse_args(argv)

    cfg = load_config()
    workdir = resolve_path(cfg, args.workdir)
    target: Path = workdir / cfg["stage_a_files"]["questions"]
    if target.exists() and not args.force:
        raise SystemExit(f"{target} đã tồn tại. Dùng --force nếu chắc chắn muốn ghi đè.")

    rows, warnings = convert(read_jsonl(resolve_path(cfg, args.gsm)), read_jsonl(resolve_path(cfg, args.math)))
    for w in warnings:
        print("cảnh báo:", w)
    write_jsonl(target, rows)
    by = {}
    for r in rows:
        key = r["source"] if r["level"] is None else f"{r['source']}-L{r['level']}"
        by[key] = by.get(key, 0) + 1
    print(f"Đã ghi {len(rows)} câu vào {target}: {dict(sorted(by.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

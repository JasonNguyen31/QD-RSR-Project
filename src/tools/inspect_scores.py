"""Mở xem các chuỗi có điểm cao nhất và thấp nhất, và kiểm tra nhóm chuỗi phải dùng lớp cứu hộ.

    python -m src.tools.inspect_scores --workdir data/stage_a
    python -m src.tools.inspect_scores --workdir data/stage_a --field llm_score --show 3

Hai việc còn nợ ở Notion B5:
  1. Kiểm tra bằng mắt vài chuỗi điểm cao nhất và thấp nhất, xem điểm có hợp lý không.
  2. Xem 561 chuỗi phải dùng lớp cứu hộ (JSON hỏng, chỉ lấy được điểm tổng) có bị lệch điểm hệ thống không.

Không gọi mạng, chỉ đọc file.
"""
from __future__ import annotations

import argparse
import statistics
from typing import Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl


def head(text: str, n: int = 420) -> str:
    t = " ".join((text or "").split())
    return t[:n] + ("…" if len(t) > n else "")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--field", default="rule_score", choices=["rule_score", "llm_score", "qual"])
    ap.add_argument("--show", type=int, default=3, help="số chuỗi in ra mỗi đầu")
    ap.add_argument("--chars", type=int, default=420)
    args = ap.parse_args(argv)

    cfg = load_config()
    wd = resolve_path(cfg, args.workdir)
    files = cfg["stage_a_files"]
    quality = {r["tid"]: r for r in read_jsonl(wd / "quality.jsonl")}
    cands = {c["tid"]: c for c in read_jsonl(wd / files["candidates"])}
    judge = {r["tid"]: r for r in read_jsonl(wd / "judge.jsonl")}
    if not quality:
        raise SystemExit(f"Không thấy quality.jsonl trong {wd}.")

    rows = [r for r in quality.values() if r.get(args.field) is not None]
    vals = [r[args.field] for r in rows]
    print(f"[xem điểm] {len(rows)} chuỗi có {args.field}: nhỏ nhất {min(vals):.3f}, "
          f"trung vị {statistics.median(vals):.3f}, lớn nhất {max(vals):.3f}")

    rows.sort(key=lambda r: r[args.field])
    for label, group in [("THẤP NHẤT", rows[:args.show]), ("CAO NHẤT", rows[-args.show:][::-1])]:
        print(f"\n{'=' * 78}\n{args.field} {label}\n{'=' * 78}")
        for r in group:
            c = cands.get(r["tid"], {})
            j = judge.get(r["tid"], {})
            print(f"\n{r['tid']}  {args.field}={r[args.field]:.3f}  số từ={r.get('n_words')}  "
                  f"llm={r.get('llm_score')}  cứu hộ={j.get('salvaged', '-')}")
            print("  " + head(c.get("text", "(không thấy văn bản)"), args.chars))

    salv = [t for t, r in judge.items() if r.get("salvaged")]
    if salv:
        a = [quality[t]["llm_score"] for t in salv if t in quality and quality[t]["llm_score"] is not None]
        b = [r["llm_score"] for t, r in quality.items() if t not in set(salv) and r["llm_score"] is not None]
        print(f"\n{'=' * 78}\nNHÓM PHẢI CỨU HỘ\n{'=' * 78}")
        print(f"  {len(salv)} chuỗi ({len(salv) / len(judge):.1%} số chuỗi đã chấm)")
        if a and b:
            print(f"  điểm trung bình: cứu hộ {statistics.mean(a):.3f} so với bình thường {statistics.mean(b):.3f}"
                  f"  (chênh {statistics.mean(a) - statistics.mean(b):+.3f})")
            print("  Chênh nhỏ thì nhóm cứu hộ không lệch hệ thống, giữ lại được. Chênh lớn thì phải loại")
            print("  chúng khỏi tập ứng viên như đã làm với nhóm không chấm được.")
        from collections import Counter
        print(f"  cách cứu hộ: {dict(Counter(judge[t]['salvaged'] for t in salv))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

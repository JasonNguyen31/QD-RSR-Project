"""So RSR của dự án (b1_fit) với mã chính thức của bài RSR (github.com/UmeanNever/RankSurprisalRatio).

Thêm 10/11/2026 cho phản biện bài báo: chứng minh cài đặt RSR khớp bản gốc. Ba bước:

  1. export   rút Q câu hỏi (seed cố định), ghi MỌI lời giải đúng của chúng theo định dạng messages mà mã chính
              thức đọc, đúng định dạng b1_fit chấm: system prompt huấn luyện, đề ở vai user, lời giải ở vai assistant.
        python scripts/compare_rsr_official.py export --questions 30

  2. chạy mã chính thức (cần GPU; mô hình học bản base như file fit, 16-bit, cắt thứ hạng 100, dài tối đa 3072):
        python ~/RankSurprisalRatio/rsr_cal.py --model_path=Qwen/Qwen2.5-1.5B --model_name=qwen1_5b_base \
          --data_root=data/rsr_check --data_path=data/rsr_check/rsr_check.jsonl --output_root=data/rsr_check/out \
          --batch_size=2 --dtype=bfloat16 --rank_clip_r=100 --max_model_len=3072 --chat_template=qwen \
          --infer_save_format=""

  3. compare  đọc kết quả của mã chính thức và cột rsr của fit.qwen1_5b_base.jsonl, in độ lệch và độ trùng tập chọn.
        python scripts/compare_rsr_official.py compare
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, resolve_path  # noqa: E402
from src.common.io_utils import read_jsonl  # noqa: E402
from src.common.prompts import SYSTEM_PROMPT_TRAIN  # noqa: E402

OUT = Path("data/rsr_check")


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb)


def export(args, cfg, wd: Path) -> int:
    files = cfg["stage_a_files"]
    cands = read_jsonl(wd / files["candidates"])
    questions = {q["qid"]: q for q in read_jsonl(wd / files["questions"])}
    fit = {r["tid"] for r in read_jsonl(wd / args.fit)}
    by_q = defaultdict(list)
    for c in cands:
        if c["tid"] in fit:
            by_q[c["qid"]].append(c)
    qids = sorted(q for q, v in by_q.items() if len(v) >= 4)
    random.Random(args.seed).shuffle(qids)
    rows = []
    for qid in qids[: args.questions]:
        for c in by_q[qid]:
            rows.append({"id": c["tid"], "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_TRAIN},
                {"role": "user", "content": questions[qid]["question"]},
                {"role": "assistant", "content": c["text"]}]})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rsr_check.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Đã ghi {path}: {args.questions} câu hỏi, {len(rows)} lời giải. Chạy bước 2 trong docstring.")
    return 0


def compare(args, cfg, wd: Path) -> int:
    hits = sorted(OUT.glob("out/_sample_metrics/*/rsr_check.jsonl"))
    if not hits:
        raise SystemExit(f"Chưa có kết quả của mã chính thức trong {OUT}/out/_sample_metrics; chạy bước 2 trước.")
    official = {r["sample_id"]: r for r in read_jsonl(hits[-1])}
    ours = {r["tid"]: r for r in read_jsonl(wd / args.fit)}
    tids = [t for t in official if t in ours]
    a = [official[t]["rank_surprisal_ratio"] for t in tids]
    b = [ours[t]["rsr"] for t in tids]
    rel = sorted(abs(x - y) / abs(x) for x, y in zip(a, b))
    print(f"So {len(tids)} lời giải (mã chính thức có {len(official)}, file {args.fit} có {len(ours)})")
    print(f"  Pearson {pearson(a, b):.5f}   Spearman {pearson(rank(a), rank(b)):.5f}")
    print(f"  lệch tương đối: trung vị {100 * rel[len(rel) // 2]:.3f}%   lớn nhất {100 * rel[-1]:.3f}%")
    tl = [(official[t]["resp_token_length"], ours[t].get("n_tokens")) for t in tids]
    same_len = sum(x == y for x, y in tl)
    print(f"  số token được chấm trùng nhau ở {same_len}/{len(tids)} lời giải")
    by_q = defaultdict(list)
    for t, x, y in zip(tids, a, b):
        by_q[t.split("|")[0]].append((t, x, y))
    same_pick, same_order = 0, 0
    for v in by_q.values():
        po = sorted(v, key=lambda r: (r[1], r[0]))       # RSR là cực tiểu: lấy 3 lời giải RSR thấp nhất
        pu = sorted(v, key=lambda r: (r[2], r[0]))
        same_pick += {r[0] for r in po[:3]} == {r[0] for r in pu[:3]}
        same_order += [r[0] for r in po] == [r[0] for r in pu]
    print(f"  tập 3 lời giải RSR thấp nhất trùng nhau ở {same_pick}/{len(by_q)} câu; "
          f"thứ tự đầy đủ trùng ở {same_order}/{len(by_q)} câu")
    worst = max(zip(tids, a, b), key=lambda r: abs(r[1] - r[2]) / abs(r[1]))
    print(f"  lệch nhất: {worst[0]}  chính thức {worst[1]:.4f}  dự án {worst[2]:.4f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["export", "compare"])
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--student", default="qwen1_5b")
    ap.add_argument("--fit", default="fit.qwen1_5b_base.jsonl")
    ap.add_argument("--questions", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args(argv)
    cfg = load_config(student=args.student)
    wd = resolve_path(cfg, args.workdir)
    return export(args, cfg, wd) if args.step == "export" else compare(args, cfg, wd)


if __name__ == "__main__":
    raise SystemExit(main())

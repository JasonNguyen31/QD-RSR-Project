"""Đo độ ổn định của mô hình giám khảo: chấm lặp cùng một chuỗi nhiều lần.

    python -m src.tools.judge_stability --workdir data/stage_a --n 20 --repeats 5 --max-cost 1

Paper đã hứa báo cáo phép đo này (Notion B5). Nó khác với phép so ba giám khảo đã làm: phép kia hỏi
"hai giám khảo có nhìn giống nhau không", phép này hỏi "MỘT giám khảo có nhất quán với chính nó không".

Đọc  <workdir>/candidates.jsonl và questions.jsonl
Ghi  <workdir>/judge_stability.jsonl  mỗi lượt chấm một dòng, ghi dần nên chạy bù được
     <workdir>/judge_stability.json   tóm tắt để trích vào paper

Cách đọc: nhiệt độ đặt 0 nên về lý thuyết điểm phải giống hệt nhau mỗi lần. Độ lệch chuẩn lớn hơn 0 cho
thấy mô hình vẫn có ngẫu nhiên nội tại. Con số đáng lo là tỷ lệ chuỗi có biên độ (lớn nhất trừ nhỏ nhất)
từ 0,2 trở lên, vì hai chuỗi cách nhau 0,2 điểm hoàn toàn có thể đảo thứ tự khi chọn lọc.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import JsonlWriter, now_iso, read_jsonl, write_json
from src.common.prompts import build_judge_prompt
from src.stage_a.a3_filter import group_of
from src.stage_a.a4_score_quality import make_judge

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it


def sample_chains(cands: Sequence[Mapping], questions: Mapping[str, Mapping], n: int, seed: int) -> list[dict]:
    """Rút n chuỗi, chia đều theo nhóm độ khó, để kết luận không chỉ đúng cho bài dễ."""
    by_group: dict[str, list] = defaultdict(list)
    for c in cands:
        by_group[group_of(questions[c["qid"]])].append(c)
    rng = random.Random(seed)
    groups = sorted(by_group)
    per = max(1, n // len(groups))
    out: list[dict] = []
    for g in groups:
        out.extend(rng.sample(by_group[g], min(per, len(by_group[g]))))
    rng.shuffle(out)
    return out[:n]


def summarize(rows: Sequence[Mapping], repeats: int) -> dict:
    by_tid: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_tid[r["tid"]].append(r["overall_score"])
    full = {t: v for t, v in by_tid.items() if len(v) >= 2}
    if not full:
        return {"chains": 0}
    stds = [statistics.pstdev(v) for v in full.values()]
    spans = [max(v) - min(v) for v in full.values()]
    identical = sum(1 for v in full.values() if max(v) == min(v))
    return {
        "chains": len(full), "repeats_target": repeats,
        "identical_every_repeat": identical,
        "identical_share": round(identical / len(full), 4),
        "std_mean": round(statistics.mean(stds), 4),
        "std_max": round(max(stds), 4),
        "span_mean": round(statistics.mean(spans), 4),
        "span_max": round(max(spans), 4),
        "span_ge_0.2": sum(1 for s in spans if s >= 0.2 - 1e-9),
        "span_ge_0.2_share": round(sum(1 for s in spans if s >= 0.2 - 1e-9) / len(full), 4),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--n", type=int, default=20, help="số chuỗi đem chấm lặp")
    ap.add_argument("--repeats", type=int, default=5, help="số lần chấm mỗi chuỗi")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-model")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cost", type=float)
    ap.add_argument("--out", default="judge_stability.jsonl")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.override)
    wd: Path = resolve_path(cfg, args.workdir)
    files = cfg["stage_a_files"]
    cands = read_jsonl(wd / files["candidates"])
    questions = {q["qid"]: q for q in read_jsonl(wd / files["questions"])}
    if not cands:
        raise SystemExit(f"Không có ứng viên trong {wd}.")

    chosen = sample_chains(cands, questions, args.n, args.seed)
    model_id = args.judge_model or cfg["judge"]["model_id"]
    path = wd / args.out
    done: dict[str, int] = defaultdict(int)
    for r in read_jsonl(path):
        done[r["tid"]] += 1
    tasks = [(c, i) for c in chosen for i in range(args.repeats) if i >= done[c["tid"]]]
    print(f"[ổn định] {len(chosen)} chuỗi × {args.repeats} lần = {len(chosen) * args.repeats} lượt; "
          f"đã có {sum(done.values())}; cần chấm {len(tasks)}")
    if tasks:
        judge = make_judge(cfg, model_id)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with JsonlWriter(path) as w:
            def work(task):
                c, i = task
                if args.max_cost is not None and getattr(judge, "cost", 0.0) >= args.max_cost:
                    return None
                q = questions[c["qid"]]
                try:
                    res = judge.score(build_judge_prompt(q["question"], c["text"], q.get("solution")))
                except Exception as exc:  # noqa: BLE001
                    return f"{c['tid']} lần {i}: {type(exc).__name__}"
                w.append({"tid": c["tid"], "qid": c["qid"], "repeat": i, "model": model_id,
                          "overall_score": res["overall_score"], "ts": now_iso()})
                return None
            ex = ThreadPoolExecutor(max_workers=args.workers)
            futs = [ex.submit(work, t) for t in tasks]
            errs = [e for f in tqdm(as_completed(futs), total=len(futs), desc="chấm lặp") if (e := f.result())]
            ex.shutdown(wait=True)
        if errs:
            print(f"[ổn định] {len(errs)} lượt lỗi, ví dụ: {errs[:3]}", file=sys.stderr)
        print(f"[ổn định] chi phí {getattr(judge, 'cost', 0.0):.4f} USD")

    rows = read_jsonl(path)
    s = summarize(rows, args.repeats)
    if not s.get("chains"):
        raise SystemExit("Chưa đủ dữ liệu để tóm tắt.")
    print(f"\n[ổn định] {s['chains']} chuỗi được chấm ít nhất hai lần, mô hình {model_id}")
    print(f"  giống hệt nhau ở mọi lần : {s['identical_every_repeat']}/{s['chains']} = {s['identical_share']:.1%}")
    print(f"  độ lệch chuẩn            : trung bình {s['std_mean']:.4f}, lớn nhất {s['std_max']:.4f}")
    print(f"  biên độ (max trừ min)    : trung bình {s['span_mean']:.4f}, lớn nhất {s['span_max']:.4f}")
    print(f"  biên độ từ 0,2 trở lên   : {s['span_ge_0.2']}/{s['chains']} = {s['span_ge_0.2_share']:.1%}")
    print("\n  Tỷ lệ cuối là con số đáng lo nhất: hai ứng viên cách nhau dưới 0,2 điểm có thể đảo thứ tự")
    print("  giữa hai lần chấm, tức việc chọn lọc phụ thuộc vào may rủi ở đúng những cặp sát nhau.")
    write_json(wd / args.out.replace(".jsonl", ".json"), {"model": model_id, **s})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

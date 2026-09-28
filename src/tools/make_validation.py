"""Rút tập kiểm định chọn lambda theo đúng phân bố của tập huấn luyện.

    python -m src.tools.make_validation            # ghi data/raw/validation_matched.jsonl
    python -m src.tools.make_validation --check    # chỉ in thống kê, không ghi

Vì sao cần: tập kiểm định cũ do a1_prepare tách có 50% GSM8K và 24 câu MATH mức 1-2, trong khi tập huấn luyện
chỉ 10% GSM8K và toàn MATH mức 3-5. Lambda là trọng số của thành phần đa dạng, mà đa dạng gần như vô tác dụng
trên bài dễ, nên chọn lambda trên tập cũ sẽ kéo lambda về 0. Không sửa a1_prepare, vì tập huấn luyện được rút
SAU bước tách tập kiểm định cũ: đổi bước đó là đổi luôn 2.000 câu huấn luyện của Giai đoạn A.

Nguồn: data/raw/gsm8k_pool.jsonl và math_pool.jsonl (đã loại MATH-500 và tập kiểm định cũ).
Loại thêm: câu huấn luyện (theo qid và theo đề đã chuẩn hoá), câu của mọi lô thử trong data/pilot, câu không
tách được đáp án. Số câu mỗi tầng tỷ lệ với data.train_composition. Chạy lại cho ra đúng cùng một tập.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import now_iso, read_jsonl, write_jsonl
from src.stage_a.a1_prepare import norm


def stratum_sizes(composition: Sequence[Mapping], n: int) -> list[tuple[str, int | None, int]]:
    """Chia n theo tỷ lệ tập huấn luyện, làm tròn theo phần dư lớn nhất để tổng đúng bằng n."""
    total = sum(s["n"] for s in composition)
    raw = [(s["source"], s["level"], n * s["n"] / total) for s in composition]
    out = [[src, lvl, int(x)] for src, lvl, x in raw]
    rest = n - sum(o[2] for o in out)
    for i in sorted(range(len(raw)), key=lambda i: -(raw[i][2] - int(raw[i][2])))[:rest]:
        out[i][2] += 1
    return [tuple(o) for o in out]


def build_matched(pools: Mapping[str, Sequence[Mapping]], sizes, exclude_qids: set[str],
                  exclude_texts: set[str], seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen: list[dict] = []
    for src, lvl, k in sizes:
        stratum = sorted((r for r in pools[src] if r.get("level") == lvl and r.get("gold") is not None
                          and r["qid"] not in exclude_qids and norm(r["question"]) not in exclude_texts),
                         key=lambda r: r["qid"])
        if len(stratum) < k:
            raise SystemExit(f"Tầng {src} mức {lvl} chỉ còn {len(stratum)} câu sạch, cần {k}.")
        chosen.extend(rng.sample(stratum, k))
    return chosen


def check_clean(val: Sequence[Mapping], exclude_qids: set[str], exclude_texts: set[str]) -> None:
    qids = [r["qid"] for r in val]
    if len(set(qids)) != len(qids):
        raise SystemExit("Tập kiểm định có qid trùng nhau.")
    if set(qids) & exclude_qids:
        raise SystemExit("Tập kiểm định trùng qid với tập huấn luyện.")
    if any(norm(r["question"]) in exclude_texts for r in val):
        raise SystemExit("Tập kiểm định trùng đề với tập huấn luyện hoặc lô thử.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--train", default="data/stage_a/questions.jsonl")
    ap.add_argument("--pilot-glob", default="data/pilot/*/questions*.jsonl")
    ap.add_argument("--check", action="store_true", help="chỉ in thống kê, không ghi file")
    args = ap.parse_args(argv)

    cfg = load_config()
    spec = cfg["data"]["val_matched"]
    seed = cfg["project"]["data_seed"] + spec["seed_offset"]
    raw = resolve_path(cfg, args.raw)
    pools = {"gsm8k": read_jsonl(raw / "gsm8k_pool.jsonl"), "math": read_jsonl(raw / "math_pool.jsonl")}
    train = read_jsonl(resolve_path(cfg, args.train))
    pilot = [r for p in sorted(glob.glob(str(resolve_path(cfg, args.pilot_glob)))) for r in read_jsonl(p)]
    old_val = read_jsonl(raw / "validation.jsonl") if (raw / "validation.jsonl").exists() else []

    ex_qids = {r["qid"] for r in train}
    ex_texts = {norm(r["question"]) for r in list(train) + pilot}
    sizes = stratum_sizes(cfg["data"]["train_composition"], spec["n"])
    val = build_matched(pools, sizes, ex_qids, ex_texts, seed)
    check_clean(val, ex_qids, ex_texts)

    groups = Counter((r["source"], r.get("level")) for r in val)
    print(f"Tập kiểm định mới: {len(val)} câu, seed {seed}")
    for src, lvl, k in sizes:
        print(f"  {src:6s} mức {str(lvl):4s}: {groups[(src, lvl)]:3d} câu (cần {k})")
    print(f"Đã loại: {len(ex_qids)} câu huấn luyện, {len(pilot)} câu lô thử; "
          f"trùng đề với tập kiểm định cũ: {sum(norm(r['question']) in {norm(v['question']) for v in old_val} for r in val)}")
    if args.check:
        return 0

    out = raw / spec["file"]
    if out.exists():
        old = read_jsonl(out)
        if [r["qid"] for r in old] == [r["qid"] for r in val]:
            print(f"{out} đã có và giống hệt, không ghi lại.")
            return 0
        raise SystemExit(f"{out} đã có nhưng KHÁC bản vừa rút. Kiểm tra lại pool hoặc cấu hình trước khi xoá file cũ.")
    write_jsonl(out, val)
    (raw / (out.stem + "_manifest.json")).write_text(json.dumps({
        "ts": now_iso(), "seed": seed, "n": len(val),
        "strata": [{"source": s, "level": l, "n": k} for s, l, k in sizes],
        "excluded": {"train_qids": len(ex_qids), "pilot_questions": len(pilot)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

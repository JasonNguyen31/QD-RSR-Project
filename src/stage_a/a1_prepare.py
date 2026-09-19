"""
a1_prepare: dựng lại kho câu hỏi từ đầu (tải, loại trùng MATH-500, tách tập kiểm định, rút 2.000 câu huấn luyện).

Thay cho prepare_train_pools.py và verify_dedup.py cũ, giữ NGUYÊN các quy ước đã kiểm chứng:
  - norm(): chỉ gộp khoảng trắng và đổi chữ thường
  - thứ tự bảy nhóm MATH, qid = math_{i:05d} theo vị trí sau loại trùng, gsm8k_{k:05d} theo chỉ số gốc
  - tập kiểm định: seed 42, rút 100 GSM8K rồi 100 MATH từ hai pool (đúng thứ tự gọi random của bản cũ)

Thêm so với bản cũ:
  - phép thử đối chứng chạy MỖI LẦN: hàm loại trùng phải bắt đúng 500/500 trên tập test của MATH
  - tách sẵn đáp án chuẩn `gold` cho cả hai nguồn, level thành số nguyên
  - cổng tái lập: nếu data/raw đã có bản cũ thì so danh sách qid, lệch thì DỪNG, không ghi đè (trừ khi --force)
  - rút 2.000 câu huấn luyện theo data.train_composition, xáo trộn cố định để --limit-questions của a2 luôn lấy mẫu trộn

    python -m src.stage_a.a1_prepare --dry-run     # dựng và báo cáo, không ghi file
    python -m src.stage_a.a1_prepare               # ghi data/raw/*.jsonl và data/stage_a/questions.jsonl
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from src.common.answers import extract_gold
from src.common.config import load_config, resolve_path
from src.common.io_utils import iter_jsonl, now_iso, read_jsonl, write_json, write_jsonl

# Thứ tự này quyết định qid. KHÔNG được đổi.
MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def parse_level(text: str | None) -> int | None:
    m = re.fullmatch(r"\s*Level\s+(\d)\s*", text or "")
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- dựng bản ghi (thuần, không cần mạng)
def dedup_against(rows: Sequence[Mapping], banned: set[str]) -> tuple[list[Mapping], int]:
    kept = [r for r in rows if norm(r["problem"]) not in banned]
    return kept, len(rows) - len(kept)


def build_math_records(kept_rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    """qid theo vị trí trong danh sách đã loại trùng. Trả về (bản ghi có gold, các câu bị bỏ vì không tách được gold).
    Câu bị bỏ vẫn chiếm số thứ tự, nên qid của các câu còn lại không đổi."""
    out, dropped = [], []
    for i, r in enumerate(kept_rows):
        gold = extract_gold("math", r["solution"])
        if gold is None:
            dropped.append({"qid": f"math_{i:05d}", "tail": r["solution"][-160:]})
            continue
        out.append({"qid": f"math_{i:05d}", "source": "math", "level": parse_level(r.get("level")),
                    "type": r.get("type"), "question": r["problem"], "solution": r["solution"], "gold": gold})
    return out, dropped


def build_gsm_records(hf_rows: Sequence[Mapping], exclude_idx: set[int]) -> list[dict]:
    out = []
    for k, r in enumerate(hf_rows):
        if k in exclude_idx:
            continue
        out.append({"qid": f"gsm8k_{k:05d}", "source": "gsm8k", "level": None, "orig_index": k,
                    "question": r["question"], "solution": r["answer"],
                    "gold": extract_gold("gsm8k", r["answer"])})
    return out


def split_validation(gsm_pool: list[dict], math_pool: list[dict], n_each: int, seed: int):
    """Đúng thứ tự gọi random của bản cũ: GSM8K trước, MATH sau, cùng một bộ sinh số đã gieo seed."""
    rng = random.Random(seed)
    val_gsm = rng.sample(gsm_pool, n_each)
    val_math = rng.sample(math_pool, n_each)
    val_ids = {r["qid"] for r in val_gsm + val_math}
    return (val_gsm + val_math,
            [r for r in gsm_pool if r["qid"] not in val_ids],
            [r for r in math_pool if r["qid"] not in val_ids])


def sample_train(gsm_pool: Sequence[dict], math_pool: Sequence[dict], composition: Sequence[Mapping], seed: int):
    rng = random.Random(seed)
    chosen: list[dict] = []
    for spec in composition:
        pool = gsm_pool if spec["source"] == "gsm8k" else math_pool
        stratum = sorted((r for r in pool if r["level"] == spec["level"]), key=lambda r: r["qid"])
        if len(stratum) < spec["n"]:
            raise SystemExit(f"Nhóm {spec['source']} level {spec['level']} chỉ có {len(stratum)} câu, cần {spec['n']}.")
        chosen.extend(rng.sample(stratum, spec["n"]))
    rng.shuffle(chosen)  # mọi tiền tố của danh sách là mẫu trộn, thuận cho --limit-questions
    return chosen


def check_no_leak(train: Sequence[Mapping], val: Sequence[Mapping], banned: set[str]) -> None:
    train_ids, val_ids = {r["qid"] for r in train}, {r["qid"] for r in val}
    if len(train_ids) != len(train):
        raise SystemExit("Tập huấn luyện có qid trùng nhau.")
    if train_ids & val_ids:
        raise SystemExit(f"{len(train_ids & val_ids)} câu vừa nằm trong tập huấn luyện vừa nằm trong tập kiểm định.")
    hit = sum(norm(r["question"]) in banned for r in list(train) + list(val))
    if hit:
        raise SystemExit(f"{hit} câu trong tập huấn luyện/kiểm định trùng MATH-500.")


def compare_with_old(raw_dir: Path, new: Mapping[str, Sequence[Mapping]]) -> tuple[bool, list[str]]:
    """So danh sách qid với bản cũ trong data/raw. Trả về (khớp hoàn toàn, các dòng mô tả)."""
    names = {"gsm8k_pool": "gsm8k_pool.jsonl", "math_pool": "math_pool.jsonl", "validation": "validation.jsonl"}
    msgs, same, found = [], True, 0
    for key, fname in names.items():
        path = raw_dir / fname
        if not path.exists():
            continue
        found += 1
        old = [r["qid"] for r in iter_jsonl(path)]
        cur = [r["qid"] for r in new[key]]
        if old == cur:
            msgs.append(f"  {fname:<18} khớp hoàn toàn ({len(cur)} câu, cả thứ tự)")
        elif set(old) == set(cur):
            msgs.append(f"  {fname:<18} cùng tập qid ({len(cur)} câu) nhưng KHÁC THỨ TỰ")
            same = False
        else:
            only_old, only_new = set(old) - set(cur), set(cur) - set(old)
            msgs.append(f"  {fname:<18} LỆCH: cũ {len(old)}, mới {len(cur)}, chỉ có ở cũ {len(only_old)}, "
                        f"chỉ có ở mới {len(only_new)}")
            same = False
    if not found:
        msgs.append("  (chưa có bản cũ trong data/raw, không có gì để so)")
    return same, msgs


# ---------------------------------------------------------------- tải dữ liệu (cần mạng)
def load_hf_sources():
    from datasets import load_dataset  # nhập muộn để phần thuần chạy được khi không có thư viện
    print("Tải MATH-500 (chỉ để đối chiếu, không dùng để huấn luyện)...")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    banned = {norm(r["problem"]) for r in math500}
    print(f"  {len(math500)} câu, {len(banned)} câu duy nhất")

    print("Tải MATH (train và test của bảy nhóm)...")
    train_rows, test_hits, test_total = [], 0, 0
    for c in MATH_CONFIGS:
        for r in load_dataset("EleutherAI/hendrycks_math", c, split="train"):
            train_rows.append({"problem": r["problem"], "solution": r["solution"],
                               "level": r.get("level", ""), "type": c})
        for r in load_dataset("EleutherAI/hendrycks_math", c, split="test"):
            test_total += 1
            test_hits += norm(r["problem"]) in banned
    print(f"  MATH train: {len(train_rows)} câu")

    print("Tải GSM8K (train)...")
    gsm = list(load_dataset("openai/gsm8k", "main", split="train"))
    print(f"  GSM8K train: {len(gsm)} câu")
    return banned, train_rows, (test_hits, test_total), gsm


def read_exclusions(paths: Sequence[Path]) -> tuple[set[int], set[str]]:
    """Câu đã dùng cho lô pilot: chỉ số gốc GSM8K (orig_index) và qid dạng math_xxxxx."""
    gsm_idx: set[int] = set()
    qids: set[str] = set()
    for p in paths:
        if not p.exists():
            print(f"  cảnh báo: không thấy {p}, bỏ qua (pool sẽ KHÔNG loại các câu pilot trong file này)")
            continue
        rows = read_jsonl(p)
        for r in rows:
            if "orig_index" in r:
                gsm_idx.add(int(r["orig_index"]))
            elif "qid" in r:
                qids.add(r["qid"])
        print(f"  loại trừ pilot từ {p}: {len(rows)} dòng")
    return gsm_idx, qids


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", help="mặc định paths.data_raw")
    ap.add_argument("--workdir", help="nơi ghi questions.jsonl, mặc định paths.data_stage_a")
    ap.add_argument("--exclude-pilot", action="append", default=None,
                    help="file jsonl của lô pilot cần loại khỏi pool (lặp được). Mặc định data/pilot/questions.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="dựng và báo cáo, không ghi file")
    ap.add_argument("--force", action="store_true", help="vẫn ghi đè dù lệch so với bản cũ")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.override)
    raw_dir = resolve_path(cfg, args.raw_dir or cfg["paths"]["data_raw"])
    workdir = resolve_path(cfg, args.workdir or cfg["paths"]["data_stage_a"])
    seed = cfg["project"]["data_seed"]
    n_val = cfg["data"]["val"]

    banned, math_rows, (test_hits, test_total), gsm_rows = load_hf_sources()

    # --- phép thử đối chứng: hàm loại trùng phải bắt đúng toàn bộ MATH-500 trên tập test
    print(f"\nPhép thử đối chứng: MATH test trùng MATH-500 = {test_hits}/{test_total} (kỳ vọng {len(banned)})")
    if test_hits != len(banned):
        raise SystemExit("Hàm loại trùng HỎNG (không bắt đủ MATH-500 trên tập test). Dừng trước khi ghi bất cứ thứ gì.")

    kept_rows, removed = dedup_against(math_rows, banned)
    print(f"Loại {removed} câu MATH train trùng MATH-500 (kỳ vọng 0). Còn {len(kept_rows)}.")
    if removed:
        raise SystemExit("Tập train MATH không được trùng MATH-500. Kiểm tra lại nguồn dữ liệu.")
    math_all, dropped = build_math_records(kept_rows)
    no_gold = len(dropped)
    print(f"Tách đáp án chuẩn MATH: bỏ {no_gold} câu không tìm thấy \\boxed trong lời giải")
    for d in dropped:
        print(f"    {d['qid']}: ...{d['tail']!r}")

    print("\nLoại trừ câu đã dùng cho pilot:")
    pilot_files = [resolve_path(cfg, p) for p in (args.exclude_pilot or ["data/pilot/questions.jsonl"])]
    gsm_excl, qid_excl = read_exclusions(pilot_files)
    gsm_all = build_gsm_records(gsm_rows, gsm_excl)
    if qid_excl:
        unknown = qid_excl - {r["qid"] for r in math_all}
        if unknown:
            print(f"  cảnh báo: {len(unknown)} qid trong file pilot không có trong pool MATH, ví dụ {sorted(unknown)[:3]}")
        math_all = [r for r in math_all if r["qid"] not in qid_excl]
    if any(r["gold"] is None for r in gsm_all):
        raise SystemExit("Có câu GSM8K không tách được đáp án sau '####'.")

    # split_validation dùng chung một n cho cả hai nguồn như bản cũ; chặn nếu cấu hình khác nhau
    if n_val["gsm8k"] != n_val["math"]:
        raise SystemExit("data.val.gsm8k và data.val.math phải bằng nhau (bản cũ rút 100 + 100).")
    val, gsm_pool, math_pool = split_validation(gsm_all, math_all, n_val["gsm8k"], seed)

    train = sample_train(gsm_pool, math_pool, cfg["data"]["train_composition"], seed + 1000)
    check_no_leak(train, val, banned)

    print("\n=== KẾT QUẢ ===")
    print(f"  gsm8k_pool  : {len(gsm_pool):>6,}   (Notion B2: 7.273)")
    print(f"  math_pool   : {len(math_pool):>6,}   (Notion B2: 7.400)")
    print(f"  validation  : {len(val):>6,}   (Notion B2: 200)")
    comp = Counter((r["source"], r["level"]) for r in train)
    print(f"  huấn luyện  : {len(train):>6,}   " + ", ".join(
        f"{s}{'-L' + str(l) if l else ''}={n}" for (s, l), n in sorted(comp.items(), key=lambda x: str(x[0]))))

    new = {"gsm8k_pool": gsm_pool, "math_pool": math_pool, "validation": val}
    print("\nSo với bản cũ trong", raw_dir)
    same, msgs = compare_with_old(raw_dir, new)
    print("\n".join(msgs))
    if not same and not args.force:
        print("\nDỪNG: bản dựng lại lệch so với bản cũ, chưa ghi gì. Gửi kết quả này cho mình xem "
              "trước khi dùng --force.", file=sys.stderr)
        return 2

    if args.dry_run:
        print("\n[dry-run] không ghi file.")
        return 0

    write_jsonl(raw_dir / "gsm8k_pool.jsonl", gsm_pool)
    write_jsonl(raw_dir / "math_pool.jsonl", math_pool)
    write_jsonl(raw_dir / "validation.jsonl", val)
    write_jsonl(workdir / cfg["stage_a_files"]["questions"], train)
    write_json(raw_dir / "build_manifest.json", {
        "ts": now_iso(), "data_seed": seed, "train_seed": seed + 1000,
        "counts": {"gsm8k_pool": len(gsm_pool), "math_pool": len(math_pool), "validation": len(val), "train": len(train)},
        "math_removed_vs_math500": removed, "math_no_gold": no_gold, "math_no_gold_qids": [d["qid"] for d in dropped],
        "control_math_test_hits": [test_hits, test_total],
        "pilot_excluded": {"gsm8k_orig_index": len(gsm_excl), "qids": len(qid_excl)},
    })
    print(f"\nĐã ghi {raw_dir}/(gsm8k_pool, math_pool, validation, build_manifest) và "
          f"{workdir / cfg['stage_a_files']['questions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

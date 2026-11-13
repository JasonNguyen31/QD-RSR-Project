"""Đóng gói MỌI lời giải mà ba mô hình dạy đã sinh (đúng, sai và bị cắt) để công bố lên Hugging Face.
Thêm 12/11/2026 theo phản biện 07/11, Phần E: "cần công bố toàn bộ 18.000 lời giải sinh ra". Bản công bố 10/11 chỉ có
lời giải đúng (candidates.jsonl). Chỉ ĐỌC dữ liệu gốc.

    python scripts/pack_hf_generations.py --out hf_release/fq-select-math-distillation

Đọc  data/stage_a/trajectories*.jsonl   mọi lời giải đã sinh (a2_generate; có thể chia lô theo mô hình dạy)
     data/stage_a/labels.jsonl           nhãn của a3_filter: đúng/sai, lý do, đáp án tách được
Ghi  <out>/generations.jsonl             mỗi lời giải một dòng: qid, tid, teacher, sample_idx, text, finish_reason,
                                         completion_tokens, correct, reason, pred
     <out>/MANIFEST.json                 thêm (hoặc cập nhật) dòng của generations.jsonl nếu file manifest đã có
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

KEEP = ("qid", "tid", "teacher", "sample_idx", "text", "finish_reason", "completion_tokens")
LABEL_KEEP = ("correct", "reason", "pred")


def read_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_rows(stage_a: Path) -> list[dict]:
    files = sorted(stage_a.glob("trajectories*.jsonl"))
    if not files:
        raise SystemExit(f"Không thấy trajectories*.jsonl trong {stage_a}.")
    labels = {r["tid"]: r for r in read_jsonl(stage_a / "labels.jsonl")}
    rows, seen = [], set()
    for f in files:
        for r in read_jsonl(f):
            if r["tid"] in seen:                     # một lời giải chỉ xuất hiện một lần dù có chạy lại lô
                continue
            seen.add(r["tid"])
            lb = labels.get(r["tid"])
            if lb is None:
                raise SystemExit(f"{r['tid']} không có trong labels.jsonl; chạy lại a3_filter trước.")
            row = {k: r[k] for k in KEEP if k in r}
            row.update({k: lb[k] for k in LABEL_KEEP if k in lb})
            rows.append(row)
    missing = set(labels) - seen
    if missing:
        raise SystemExit(f"{len(missing)} tid có nhãn nhưng không có văn bản, ví dụ {sorted(missing)[:3]}.")
    rows.sort(key=lambda r: (r["qid"], r["teacher"], r["sample_idx"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage-a", default="data/stage_a")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = build_rows(Path(a.stage_a))
    dst = out / "generations.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    data = dst.read_bytes()
    man = out / "MANIFEST.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        m["generations.jsonl"] = {"md5": hashlib.md5(data).hexdigest(), "bytes": len(data), "lines": len(rows)}
        man.write_text(json.dumps(dict(sorted(m.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    n_ok = sum(1 for r in rows if r.get("correct"))
    n_cut = sum(1 for r in rows if r.get("finish_reason") == "length")
    print(f"Đã ghi {dst}: {len(rows)} lời giải, {n_ok} đúng, {len(rows) - n_ok} sai (trong đó {n_cut} bị cắt), "
          f"{len(data) / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Đóng gói dữ liệu công bố lên Hugging Face (thêm 10/11/2026, phản biện Phần E). Chỉ ĐỌC dữ liệu gốc, ghi vào thư mục mới.

    python scripts/pack_hf_dataset.py --out hf_release/fq-select-math-distillation

Ghi ra:
  questions.jsonl            1.895 câu hỏi của kho (và các câu đã bị loại, nếu có trong file gốc)
  candidates.jsonl           mọi lời giải đúng: qid, tid, teacher, sample_idx, text
  scores.jsonl               mỗi lời giải một dòng: tid, Qual, điểm judge, điểm quy tắc, Fam (GRAPE), RSR, LARK, ...
  selections/<quy tắc>.jsonl tập chọn của từng quy tắc: qid, tid (tập chọn không đổi theo seed huấn luyện)
  selections/<quy tắc>.json  thống kê tập chọn (tỉ lệ mô hình dạy, số token) do b2_select ghi
  eval/<quy tắc>/seed<N>/<bộ>.jsonl   từng lượt sinh: qid, sample, correct (+ số token, dừng, chạm trần nếu có);
                                      KHÔNG kèm văn bản sinh ra, để gọn
  eval/<quy tắc>/seed<N>/summary.json
  MANIFEST.json              md5 và số dòng của mọi file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

KEEP_EVAL = ("qid", "sample", "correct", "n_tokens", "new_tokens", "stopped", "hit_cap", "has_boxed", "pred")
SKIP_SCORE = {"text", "ts", "signals_version"}


def read_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(p: Path, rows) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage-a", default="data/stage_a")
    ap.add_argument("--stage-b", default="data/stage_b/qwen1_5b_base")
    ap.add_argument("--eval", default="outputs/results/eval/qwen1_5b_base")
    ap.add_argument("--fit", default="fit.qwen1_5b_base.jsonl")
    args = ap.parse_args()
    out, a, b, e = Path(args.out), Path(args.stage_a), Path(args.stage_b), Path(args.eval)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} đã có dữ liệu; xoá hoặc chọn thư mục khác để không trộn bản cũ.")
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(a / "questions.jsonl", out / "questions.jsonl")
    write_jsonl(out / "candidates.jsonl",
                ({k: r[k] for k in ("qid", "tid", "teacher", "sample_idx", "text") if k in r}
                 for r in read_jsonl(a / "candidates.jsonl")))
    quality = {r["tid"]: r for r in read_jsonl(a / "quality.jsonl")}
    fit = {r["tid"]: r for r in read_jsonl(a / args.fit)}
    rows = []
    for tid in sorted(set(quality) | set(fit)):
        row = {"tid": tid}
        row.update({f"quality.{k}": v for k, v in quality.get(tid, {}).items() if k not in SKIP_SCORE | {"tid"}})
        row.update({f"student_qwen2.5-1.5b.{k}": v for k, v in fit.get(tid, {}).items()
                    if k not in SKIP_SCORE | {"tid", "qid", "model"}})
        rows.append(row)
    write_jsonl(out / "scores.jsonl", rows)

    for sel in sorted(b.glob("select.*.json")):
        tag = sel.name[len("select."):-len(".json")]
        if tag.endswith(".tam"):
            continue                                   # tập bị khoá, không dùng để huấn luyện
        train = b / f"train.{tag}.jsonl"
        if not train.exists():
            continue
        write_jsonl(out / "selections" / f"{tag}.jsonl", ({"qid": r["qid"], "tid": r["tid"]} for r in read_jsonl(train)))
        shutil.copy2(sel, out / "selections" / f"{tag}.json")

    for f in sorted(e.glob("*/seed*/*.jsonl")):
        if ".limit" in str(f.parent):
            continue
        dst = out / "eval" / f.parent.parent.name / f.parent.name / f.name
        write_jsonl(dst, ({k: r[k] for k in KEEP_EVAL if k in r} for r in read_jsonl(f)))
        s = f.parent / "summary.json"
        if s.exists() and not (dst.parent / "summary.json").exists():
            shutil.copy2(s, dst.parent / "summary.json")

    manifest = {}
    for f in sorted(p for p in out.rglob("*") if p.is_file()):
        data = f.read_bytes()
        manifest[str(f.relative_to(out))] = {"md5": hashlib.md5(data).hexdigest(), "bytes": len(data),
                                             "lines": data.count(b"\n") if f.suffix == ".jsonl" else None}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(v["bytes"] for v in manifest.values())
    print(f"Đã ghi {len(manifest)} file, {total / 2**20:.1f} MB vào {out}")
    for k in ("questions.jsonl", "candidates.jsonl", "scores.jsonl"):
        print(f"  {k}: {manifest[k]['lines']} dòng")
    print(f"  selections: {sum(1 for k in manifest if k.startswith('selections/') and k.endswith('.jsonl'))} quy tắc")
    print(f"  eval: {sum(1 for k in manifest if k.startswith('eval/') and k.endswith('.jsonl'))} file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

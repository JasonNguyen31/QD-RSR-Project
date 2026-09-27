"""Khảo sát dữ liệu công khai của bài RSR (Umean/RSR_data) để kiểm hai khẳng định trong paper.

    python -m src.tools.inspect_rsr_ref
    python -m src.tools.inspect_rsr_ref --tokenizer Qwen/Qwen2.5-3B      # thêm độ dài theo token

Đọc  data/rsr_ref/rsr-min_for_qwen25-3b-base.json   tập RSR đã chọn cho Qwen2.5-3B bản nền
     data/rsr_ref/gptoss120b/gen*.json               ba lượt sinh của teacher ngắn nhất (gpt-oss-120b)
Mỗi file là danh sách {"messages": [system, user, assistant]}.

Hai khẳng định được kiểm (Notion mục P8, paper §4.1 phần Baselines):
  1. Chuỗi RSR chọn dài gấp khoảng 4,9 lần chuỗi của gpt-oss-120b.
  2. Trong 4.948 bài chung, chỉ 2 bài có chuỗi được chọn đến từ gpt-oss-120b (chia đều cho 11 teacher
     thì phải cỡ 450 bài). Nhận diện bằng so khớp NGUYÊN VĂN câu trả lời, nên đây là cận dưới chắc chắn.

Kết quả ngày 28/09/2026: 2/4.948 bài; tỷ lệ độ dài 4,90 theo số từ, 4,50 theo ký tự; 100% có thẻ <think>.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path


def _messages(rec: Mapping) -> list:
    m = rec["messages"]
    return ast.literal_eval(m) if isinstance(m, str) else m


def load_answers(path: Path) -> dict[str, list[str]]:
    """Câu hỏi (nội dung vai user, đã bỏ khoảng trắng đầu cuối) -> danh sách câu trả lời."""
    out: dict[str, list[str]] = {}
    for rec in json.loads(Path(path).read_text(encoding="utf-8")):
        msgs = _messages(rec)
        user = next(x["content"] for x in msgs if x["role"] == "user").strip()
        ans = next(x["content"] for x in msgs if x["role"] == "assistant")
        out.setdefault(user, []).append(ans)
    return out


def survey(selected: Mapping[str, list[str]], teacher: Mapping[str, list[str]], tokenize=None) -> dict:
    common = sorted(set(selected) & set(teacher))
    hits = sum(any(s.strip() == t.strip() for s in selected[q] for t in teacher[q]) for q in common)
    sel = [a for q in selected for a in selected[q]]
    tea = [a for q in teacher for a in teacher[q]]

    def ratio(f):
        a, b = statistics.mean(map(f, sel)), statistics.mean(map(f, tea))
        return {"selected": round(a, 1), "teacher": round(b, 1), "ratio": round(a / b, 2)}

    res = {"selected_questions": len(selected), "teacher_questions": len(teacher), "common": len(common),
           "selected_from_teacher": hits,
           "think_share": round(sum(a.lstrip().startswith("<think>") for a in sel) / len(sel), 4),
           "words": ratio(lambda s: len(s.split())), "chars": ratio(len)}
    if tokenize is not None:
        res["tokens"] = ratio(lambda s: len(tokenize(s)))
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/rsr_ref")
    ap.add_argument("--selected", default="rsr-min_for_qwen25-3b-base.json")
    ap.add_argument("--teacher", default="gptoss120b/gen*.json", help="mẫu đường dẫn các lượt sinh của teacher")
    ap.add_argument("--tokenizer", help="tên tokenizer Hugging Face; bỏ trống thì chỉ đo theo từ và ký tự")
    args = ap.parse_args(argv)

    d = resolve_path(load_config(), args.dir)
    sel_path = d / args.selected
    tea_paths = sorted(glob.glob(str(d / args.teacher)))
    if not sel_path.exists() or not tea_paths:
        raise SystemExit(f"Không thấy dữ liệu trong {d}. Cần {args.selected} và {args.teacher}.")
    selected = load_answers(sel_path)
    teacher: dict[str, list[str]] = {}
    for p in tea_paths:
        for q, a in load_answers(Path(p)).items():
            teacher.setdefault(q, []).extend(a)
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(args.tokenizer)
        tok = lambda s: t.encode(s, add_special_tokens=False)   # noqa: E731
    r = survey(selected, teacher, tok)

    print(f"Tập RSR chọn: {r['selected_questions']} câu; teacher: {r['teacher_questions']} câu "
          f"({len(tea_paths)} lượt sinh); chung: {r['common']} câu")
    print(f"1. Số câu mà chuỗi được chọn là của teacher này: {r['selected_from_teacher']}/{r['common']}")
    for k in ("words", "chars", "tokens"):
        if k in r:
            v = r[k]
            print(f"2. Độ dài trung bình theo {k}: chọn {v['selected']}, teacher {v['teacher']}, gấp {v['ratio']} lần")
    print(f"3. Tỷ lệ chuỗi được chọn có thẻ <think>: {r['think_share']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

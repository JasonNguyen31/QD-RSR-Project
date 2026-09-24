"""Đo tác động của bộ chấm mới lên nhãn đã có, TRƯỚC khi chạy lại a3.

    python -m src.tools.regrade_impact --workdir data/stage_a

Bộ chấm được bổ sung phép so tương đương bằng ký hiệu ngày 23/09/2026. Lệnh này chấm lại 18.000 chuỗi
bằng bộ chấm mới, so với nhãn cũ trong labels.jsonl, và cho biết:
  - bao nhiêu chuỗi đổi từ SAI sang ĐÚNG, theo mô hình dạy và mức độ khó
  - bao nhiêu chuỗi đổi ngược chiều (phải bằng 0; khác 0 là dấu hiệu bộ chấm mới nới lỏng quá tay)
  - bao nhiêu câu hỏi trước bị loại nay đủ 3 chuỗi đúng để quay lại
  - vài ví dụ cụ thể, để kiểm tra bằng mắt trước khi tin

Không gọi mạng, không sửa file nào. Chấm lại xong thì chạy a3_filter để ghi nhãn mới.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Sequence

from src.common.answers import is_equiv
from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl
from src.stage_a.a3_filter import group_of


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--show", type=int, default=8, help="số ví dụ in ra")
    args = ap.parse_args(argv)

    cfg = load_config()
    wd = resolve_path(cfg, args.workdir)
    files = cfg["stage_a_files"]
    labels = read_jsonl(wd / files["labels"])
    questions = {q["qid"]: q for q in read_jsonl(wd / files["questions"])}
    if not labels:
        raise SystemExit(f"Không thấy labels.jsonl trong {wd}. Chạy a3_filter trước.")
    min_correct = cfg["data"]["min_correct_per_question"]

    flips_up, flips_down = [], []
    reasons = Counter(l["reason"] for l in labels)
    print(f"[chấm lại] lý do trong nhãn cũ: {dict(reasons)}")
    new_correct: dict[str, int] = defaultdict(int)
    old_correct: dict[str, int] = defaultdict(int)
    for l in labels:
        gold = questions[l["qid"]]["gold"]
        pred = l.get("pred")
        # Chuỗi bị cắt cụt luôn tính là sai, dù đáp án có khớp: chuỗi cụt không làm mẫu huấn luyện được.
        # Bỏ điều kiện này thì công cụ đếm nhầm cả những chuỗi mà a3 vốn đã loại vì bị cắt.
        now = bool(pred) and l["reason"] != "truncated" and is_equiv(pred, gold)
        old_correct[l["qid"]] += bool(l["correct"])
        new_correct[l["qid"]] += now
        if now and not l["correct"]:
            flips_up.append((l, gold))
        elif l["correct"] and not now:
            flips_down.append((l, gold))

    print(f"[chấm lại] {len(labels)} chuỗi")
    print(f"  SAI  -> ĐÚNG : {len(flips_up)}")
    print(f"  ĐÚNG -> SAI  : {len(flips_down)}" + ("  <-- BẤT THƯỜNG, xem lại bộ chấm" if flips_down else ""))

    if flips_up:
        by_t = Counter(l["teacher"] for l, _ in flips_up)
        by_g = Counter(group_of(questions[l["qid"]]) for l, _ in flips_up)
        print(f"  theo mô hình dạy: {dict(by_t)}")
        print(f"  theo độ khó     : {dict(by_g)}")

    back = [q for q in questions if old_correct[q] < min_correct <= new_correct[q]]
    gained = sum(new_correct[q] - old_correct[q] for q in questions if new_correct[q] > old_correct[q])
    print(f"\n  câu hỏi quay lại tập huấn luyện: {len(back)}")
    print(f"  ứng viên mới cần chấm giám khảo và tính vector: {gained}")
    print(f"  ước chi phí giám khảo cho phần mới: {gained * 0.00047:.2f} USD")

    if flips_up:
        print(f"\n{'=' * 78}\nVÍ DỤ ĐỔI TỪ SAI SANG ĐÚNG\n{'=' * 78}")
        for l, gold in flips_up[:args.show]:
            print(f"  {l['tid']:<26} mô hình dự đoán {l['pred']!r:<28} đáp án chuẩn {gold!r}")
    if flips_down:
        print(f"\n{'=' * 78}\nĐỔI NGƯỢC CHIỀU, CẦN XEM KỸ\n{'=' * 78}")
        for l, gold in flips_down[:args.show]:
            print(f"  {l['tid']:<26} {l['pred']!r} so với {gold!r}")
    print("\nNhìn qua danh sách ví dụ: mỗi dòng phải thật sự là cùng một giá trị viết khác dạng.")
    print("Nếu có dòng nào hai giá trị khác nhau thì DỪNG, đừng chạy a3_filter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

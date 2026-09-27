"""Đối chiếu bộ chấm mới (src/common/answers.py) với nhãn của lô pilot đầu tiên (run1_original).

    python -m src.tools.check_grader

Bộ chấm mới cài lại quy ước MATH/PRM800K và từ 23/09/2026 có so tương đương bằng sympy, nhưng chỉ cho biểu
thức số không chứa chữ cái (xem _SYMPY_UNSAFE trong answers.py). Nhãn cũ do mã pilot đầu tiên chấm, dùng làm mốc.

GIỚI HẠN QUAN TRỌNG của phép đối chiếu này: file nhãn cũ chỉ lưu các chuỗi ĐÚNG (is_correct luôn True),
vì mã pilot cũ bỏ chuỗi sai trước khi ghi. Vì vậy phép so chỉ phát hiện được lỗi chấm nhầm đúng thành sai
(âm tính giả), KHÔNG phát hiện được lỗi chấm nhầm sai thành đúng (dương tính giả). Chương trình in ra số nhãn
đúng và sai để thấy rõ điều này. Kết quả ngày 28/09/2026: 1.189/1.189 khớp, cả 1.189 đều là nhãn đúng.

Hai phép so, tách riêng vì chúng kiểm tra hai phần khác nhau:
  1. SO KHỚP  is_equiv(pred_answer, gold_answer) với is_correct của nhãn cũ.
     Chỉ dùng hai trường đã tách sẵn, nên không phụ thuộc prompt cũ dùng '#### ' hay '\\boxed{}'.
  2. TÁCH ĐÁP ÁN  last_boxed(trajectory) với pred_answer của nhãn cũ, chỉ trên các chuỗi thật sự có \\boxed.
     Lệch ở đây thường là do bộ tách cũ dùng quy ước khác, không nhất thiết là lỗi.

Mọi trường hợp lệch đều được in ra để đọc bằng mắt. Mã thoát khác 0 nếu có lệch ở phép 1.
"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import Mapping, Sequence

from src.common.answers import is_equiv, last_boxed, normalize_answer
from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl

# Tên trường của nhãn pilot cũ
F_PRED, F_GOLD, F_OK, F_TEXT = "pred_answer", "gold_answer", "is_correct", "trajectory"


def compare_matching(rows: Sequence[Mapping]) -> tuple[Counter, list[dict]]:
    """So kết luận đúng/sai. Bỏ qua dòng mà bộ chấm cũ không tách được đáp án (pred rỗng)."""
    tally, diffs = Counter(), []
    for r in rows:
        pred, gold, old = r.get(F_PRED), r.get(F_GOLD), r.get(F_OK)
        if old is None or gold is None:
            tally["thiếu trường"] += 1
            continue
        if pred in (None, ""):
            tally["bộ chấm cũ không tách được"] += 1
            continue
        new = is_equiv(pred, gold)
        tally["khớp" if new == bool(old) else "LỆCH"] += 1
        if new != bool(old):
            diffs.append({"tid": r.get("tid"), "pred": pred, "gold": gold, "cũ": bool(old), "mới": new,
                          "pred_chuẩn": normalize_answer(pred), "gold_chuẩn": normalize_answer(gold)})
    return tally, diffs


def compare_extraction(rows: Sequence[Mapping]) -> tuple[Counter, list[dict]]:
    tally, diffs = Counter(), []
    for r in rows:
        text, pred = r.get(F_TEXT), r.get(F_PRED)
        if not text or "\\boxed" not in text:
            tally["chuỗi không có \\boxed"] += 1
            continue
        mine = last_boxed(text)
        if mine is None:
            tally["mới không tách được"] += 1
            diffs.append({"tid": r.get("tid"), "cũ": pred, "mới": None, "đuôi": text[-120:]})
        elif pred in (None, "") :
            tally["cũ không tách được"] += 1
        elif is_equiv(mine, pred):
            tally["khớp"] += 1
        else:
            tally["LỆCH"] += 1
            diffs.append({"tid": r.get("tid"), "cũ": pred, "mới": mine, "đuôi": text[-120:]})
    return tally, diffs


def report(name: str, tally: Counter, diffs: Sequence[Mapping], show: int) -> None:
    total = sum(tally.values())
    print(f"\n--- {name} ({total} dòng)")
    for k, v in tally.most_common():
        print(f"    {k:<32}{v:>6}  ({v / total:.1%})" if total else f"    {k}: {v}")
    for d in list(diffs)[:show]:
        print("    " + ", ".join(f"{k}={v!r}" for k, v in d.items()))
    if len(diffs) > show:
        print(f"    ... còn {len(diffs) - show} trường hợp nữa (tăng --show để xem thêm)")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="*", default=["data/pilot/run1_original/labeled.jsonl",
                                                    "data/pilot/run1_original/labeled_math.jsonl"])
    ap.add_argument("--show", type=int, default=15, help="số trường hợp lệch in ra mỗi phép")
    args = ap.parse_args(argv)

    cfg = load_config()
    rows: list[dict] = []
    for f in args.files:
        p = resolve_path(cfg, f)
        if not p.exists():
            print(f"cảnh báo: không thấy {p}, bỏ qua")
            continue
        got = read_jsonl(p)
        print(f"đọc {len(got):>5} dòng từ {f}")
        rows.extend(got)
    if not rows:
        raise SystemExit("Không đọc được nhãn cũ nào.")

    pos = sum(1 for r in rows if r.get(F_OK) is True)
    neg = sum(1 for r in rows if r.get(F_OK) is False)
    print(f"nhãn cũ: {pos} đúng, {neg} sai")
    if neg == 0:
        print("LƯU Ý: không có nhãn sai nào, nên phép 1 chỉ kiểm được lỗi chấm nhầm đúng thành sai.")
    m_tally, m_diffs = compare_matching(rows)
    report("PHÉP 1: so khớp đáp án (quan trọng)", m_tally, m_diffs, args.show)
    e_tally, e_diffs = compare_extraction(rows)
    report("PHÉP 2: tách \\boxed (tham khảo)", e_tally, e_diffs, args.show)

    if m_tally["LỆCH"]:
        print(f"\nCÓ {m_tally['LỆCH']} trường hợp lệch ở phép 1. Đọc từng dòng trên: nếu bộ chấm mới đúng "
              f"thì không sao, nếu bộ chấm mới sai thì phải sửa answers.py trước khi chạy B9.")
        return 1
    print("\nPhép 1 khớp hoàn toàn với nhãn cũ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

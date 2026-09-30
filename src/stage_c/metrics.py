"""Chỉ số đánh giá, tính từ các lượt sinh độc lập của cùng một câu hỏi.

Định nghĩa theo Phụ lục A.4 bài RSR (Yang 2026), chốt 30/09/2026:

    acc@k   TRUNG BÌNH tỷ lệ đúng của k lượt trên mỗi câu, rồi trung bình trên các câu.   Chỉ số chính.
    pass@k  câu tính là đúng nếu có ít nhất một trong k lượt đúng.                         Phép thử cho Div.

Hai chỉ số dùng CHUNG k lượt sinh, không tốn thêm lượt nào. Trước 30/09 dự án gọi nhầm pass@4 là Acc@4;
module này tồn tại để định nghĩa chỉ nằm ở một chỗ.
"""
from __future__ import annotations

from typing import Mapping, Sequence


def _check(results: Mapping[str, Sequence[bool]], k: int) -> None:
    if not results:
        raise ValueError("Không có câu hỏi nào để tính chỉ số.")
    short = [q for q, r in results.items() if len(r) != k]
    if short:
        raise ValueError(f"{len(short)} câu không có đúng {k} lượt, ví dụ {short[0]}. "
                         f"Mọi câu phải có cùng số lượt thì chỉ số mới so được giữa các phương án.")


def acc_at_k(results: Mapping[str, Sequence[bool]], k: int = 4) -> float:
    """Trung bình trên câu hỏi của tỷ lệ lượt đúng. results: qid -> k giá trị đúng/sai."""
    _check(results, k)
    return sum(sum(map(bool, r)) / k for r in results.values()) / len(results)


def pass_at_k(results: Mapping[str, Sequence[bool]], k: int = 4) -> float:
    """Tỷ lệ câu có ít nhất một lượt đúng."""
    _check(results, k)
    return sum(any(r) for r in results.values()) / len(results)


METRICS = {"acc@4": lambda r: acc_at_k(r, 4), "pass@4": lambda r: pass_at_k(r, 4)}


def report(results: Mapping[str, Sequence[bool]], names: Sequence[str]) -> dict[str, float]:
    """Tính các chỉ số có tên trong cfg.eval.report."""
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise KeyError(f"Chỉ số chưa định nghĩa: {unknown}. Có: {sorted(METRICS)}")
    return {n: METRICS[n](results) for n in names}

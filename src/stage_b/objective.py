"""Hàm mục tiêu của QD-RSR, phần toán thuần (chỉ cần numpy). Dùng chung cho lambda_scan và b2_select.

    F(S) = Σ_{t∈S} Fit(t)^a · Qual(t)^b  +  λ · Div(S)
    Fit  = 1 − minmax(RSR) trong từng câu hỏi       (RSR là cực tiểu, nên đảo lại cho càng cao càng hợp)
    Qual = cột qual của quality.jsonl                (đã min-max từng nửa trong câu hỏi rồi trộn)
    Div(S) = Σ_{t∈S} min_{s∈S\\{t}} ‖e_t − e_s‖₂    (e đã chuẩn hoá L2, nên mỗi khoảng cách nằm trong [0, 2])

Div không đơn điệu, không submodular, nên chọn bằng cách duyệt hết các tập con cỡ k (tối đa C(9,3) = 84).

Thang đo của Div là câu hỏi đang mở. Hàm div_scale cho hai cách:
    raw      giữ nguyên, Div của 3 chuỗi nằm trong [0, 6]
    subsets  min-max trong các tập con của chính câu hỏi đó, nên Div nằm trong [0, 1] như Fit và Qual
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np


def minmax(xs: Sequence[float]) -> np.ndarray:
    """Min-max trong một nhóm. Nhóm mà mọi giá trị bằng nhau thì trả 0,5 cho tất cả, như inspect_qual."""
    a = np.asarray(xs, dtype=float)
    lo, hi = a.min(), a.max()
    return np.full_like(a, 0.5) if hi - lo < 1e-12 else (a - lo) / (hi - lo)


def fit_from_rsr(rsr: Sequence[float]) -> np.ndarray:
    """Fit = 1 − minmax(RSR): chuỗi RSR thấp nhất trong câu hỏi được Fit = 1."""
    return 1.0 - minmax(rsr)


def distance_matrix(vecs: np.ndarray) -> np.ndarray:
    """Khoảng cách Euclid giữa mọi cặp vector (đã chuẩn hoá L2)."""
    v = np.asarray(vecs, dtype=float)
    sq = np.clip(2.0 - 2.0 * (v @ v.T), 0.0, None)
    np.fill_diagonal(sq, 0.0)
    return np.sqrt(sq)


def subsets(n: int, k: int) -> np.ndarray:
    """Mọi tập con cỡ k của n ứng viên, mỗi dòng là một tập (chỉ số tăng dần)."""
    if not 1 <= k <= n:
        raise ValueError(f"cần 1 <= k <= n, được k={k}, n={n}")
    return np.array(list(combinations(range(n), k)), dtype=int)


def div_of_subsets(dist: np.ndarray, sets: np.ndarray) -> np.ndarray:
    """Div(S) cho từng tập con: tổng khoảng cách từ mỗi phần tử tới phần tử gần nó nhất trong tập."""
    k = sets.shape[1]
    if k == 1:
        return np.zeros(len(sets))
    d = dist[sets[:, :, None], sets[:, None, :]]              # (số tập, k, k)
    d = d + np.eye(k)[None] * 1e9                             # bỏ khoảng cách tới chính nó
    return d.min(axis=2).sum(axis=1)


def div_scale(div: np.ndarray, how: str) -> np.ndarray:
    if how == "raw":
        return div
    if how == "subsets":
        return minmax(div) if len(div) > 1 else np.zeros_like(div)
    raise ValueError(f"cách chuẩn hoá Div không hợp lệ: {how}")


def base_of_subsets(fit: np.ndarray, qual: np.ndarray, sets: np.ndarray, a: float = 1.0,
                    b: float = 1.0) -> np.ndarray:
    """Σ Fit^a · Qual^b của từng tập. Số mũ 0 tắt thành phần tương ứng (0^0 tính là 1)."""
    per = np.power(fit, a) * np.power(qual, b)
    return per[sets].sum(axis=1)


def choose(base: np.ndarray, div: np.ndarray, lam: float) -> int:
    """Chỉ số tập con tối đa hoá base + λ·div. Hoà thì lấy tập đứng trước, để kết quả tất định."""
    return int(np.argmax(base + lam * div))

"""Điểm chất lượng theo quy tắc, tái lập bộ lọc LIMO như bài RSR mô tả ở Phụ lục B.2.

Bốn tiêu chí và trọng số:
    Elaborated reasoning   30%  tổng số từ của lời giải (KHÔNG đếm từ khoá)
    Self-Verification      20%  tần suất "check", "verify"
    Exploratory Approach   25%  tần suất "perhaps", "might"
    Adaptive Granularity   25%  tần suất "therefore", "since"

Ba bước theo đúng B.2:
  1. đếm số lần xuất hiện từ khoá
  2. chia cho tổng số từ để ra tần suất tương đối (so được giữa lời giải dài ngắn khác nhau)
  3. chuẩn hoá z-score RIÊNG từng tiêu chí, rồi cộng theo trọng số

Phạm vi z-score là TOÀN BỘ tập chuỗi, không phải trong từng câu hỏi. Bước chuẩn hoá min-max theo từng
câu hỏi diễn ra sau, ở b2_select, khi đưa vào hàm mục tiêu. Hai lớp chuẩn hoá này không xung đột.

Chỗ bài gốc không nói rõ, đề tài tự quyết và ghi lại ở đây:
  - "từ" tách bằng biểu thức chính quy \\b\\w+\\b, chữ thường hoá trước khi đếm
  - đếm theo từ nguyên vẹn, nên "checking" và "checked" KHÔNG tính là "check"
  - tiêu chí đầu dùng thẳng tổng số từ (số tuyệt đối), ba tiêu chí sau dùng tần suất; sau z-score cả bốn cùng thang
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Sequence

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

DEFAULT_WEIGHTS = {"elaborated": 0.30, "self_verification": 0.20, "exploratory": 0.25, "adaptive": 0.25}
DEFAULT_KEYWORDS = {
    "self_verification": ["check", "verify"],
    "exploratory": ["perhaps", "might"],
    "adaptive": ["therefore", "since"],
}


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def raw_features(text: str, keywords: Mapping[str, Sequence[str]] = DEFAULT_KEYWORDS) -> dict[str, float]:
    """Đặc trưng thô trước khi chuẩn hoá: tổng số từ, và tần suất tương đối của từng nhóm từ khoá."""
    words = tokenize(text)
    n = len(words)
    feats: dict[str, float] = {"elaborated": float(n)}
    if n == 0:
        return {**feats, **{k: 0.0 for k in keywords}}
    counts: dict[str, int] = {k: 0 for k in keywords}
    wanted = {w: k for k, kws in keywords.items() for w in kws}
    for w in words:
        k = wanted.get(w)
        if k is not None:
            counts[k] += 1
    for k in keywords:
        feats[k] = counts[k] / n
    return feats


def zscore(values: Sequence[float]) -> list[float]:
    """z-score. Nếu mọi giá trị bằng nhau (độ lệch chuẩn 0) thì trả về 0 hết, tránh chia cho 0."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n     # phương sai tổng thể, khớp numpy.std mặc định
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0] * n
    return [(v - mean) / sd for v in values]


def rule_scores(
    texts: Sequence[str],
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    keywords: Mapping[str, Sequence[str]] = DEFAULT_KEYWORDS,
) -> tuple[list[float], list[dict[str, float]]]:
    """Chấm cả tập một lượt, vì z-score cần biết toàn bộ phân bố.

    Trả về (điểm tổng hợp, đặc trưng thô từng chuỗi). Điểm là số thực có thể âm, đó là bình thường
    với z-score; bước chuẩn hoá về [0, 1] diễn ra sau ở b2_select.
    """
    missing = set(weights) - ({"elaborated"} | set(keywords))
    if missing:
        raise ValueError(f"Trọng số cho tiêu chí không có cách đo: {sorted(missing)}")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"Tổng trọng số phải bằng 1, hiện là {sum(weights.values())}")

    feats = [raw_features(t, keywords) for t in texts]
    z = {name: zscore([f[name] for f in feats]) for name in weights}
    scores = [sum(weights[name] * z[name][i] for name in weights) for i in range(len(texts))]
    return scores, feats

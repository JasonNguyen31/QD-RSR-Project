"""Kiểm thử phần toán của b1_fit. Không cần torch, không cần GPU, không nạp mô hình.

Phần chạy trên GPU chỉ có việc tạo ra bốn mảng số liệu từng token; toàn bộ phép gộp thành tín hiệu nằm
trong aggregate(), nên kiểm tra được ở đây bằng số liệu dựng tay.
"""
import math

import pytest

from src.stage_b.b1_fit import aggregate, perplexity


def make(n=4, rank=0, surp=0.5, sq=0.5, tp=0.6):
    return [rank] * n, [surp] * n, [sq] * n, [tp] * n


def test_rsr_uses_clipped_rank_counted_from_one():
    """Thứ hạng 0 nghĩa là token được đoán chắc nhất, nhưng bài gốc đếm từ 1."""
    r = aggregate([0, 0], [1.0, 1.0], [0.5, 0.5], [0.6, 0.6], rank_clip=100)
    assert r["mean_clipped_rank"] == pytest.approx(1.0)
    assert r["rsr"] == pytest.approx(2 / 2.0)


def test_rank_clip_caps_extreme_ranks():
    """Một token cực lạ không được kéo RSR lên vô hạn: đây là lý do bài gốc cắt ngưỡng."""
    no_clip = aggregate([0, 999_999], [1.0, 1.0], [0.5, 0.5], [0.6, 0.6], rank_clip=10_000_000)
    clipped = aggregate([0, 999_999], [1.0, 1.0], [0.5, 0.5], [0.6, 0.6], rank_clip=100)
    assert clipped["rsr"] < no_clip["rsr"] / 1000
    assert clipped["mean_clipped_rank"] <= 100


def test_rsr_direction_lower_is_better():
    """Chuỗi vừa sức: thứ hạng thấp nhưng vẫn còn độ bất ngờ. Chuỗi lạ: thứ hạng cao."""
    fit = aggregate([1, 2, 1], [0.8, 0.9, 0.7], [0.4] * 3, [0.5] * 3)
    odd = aggregate([500, 800, 900], [0.8, 0.9, 0.7], [0.4] * 3, [0.5] * 3)
    assert fit["rsr"] < odd["rsr"]


def test_grape_is_mean_log_probability():
    r = aggregate(*make(n=4, surp=0.5))
    assert r["grape"] == pytest.approx(-0.5)
    assert perplexity(r["grape"]) == pytest.approx(math.exp(0.5))


def test_brier_is_zero_for_a_perfectly_confident_correct_prediction():
    """Dự đoán chắc chắn và đúng: p_đúng = 1, tổng bình phương = 1, nên Brier = 1 - 2 + 1 = 0."""
    r = aggregate([0], [0.0], [1.0], [1.0])
    assert r["brier"] == pytest.approx(0.0)


def test_brier_grows_when_the_model_is_unsure():
    sure = aggregate([0], [0.1], [0.82], [0.9])
    unsure = aggregate([0], [1.6], [0.05], [0.2])
    assert unsure["brier"] > sure["brier"]


def test_brier_in_valid_range():
    """Điểm Brier của một token luôn nằm trong [0, 2] theo định nghĩa."""
    for sq, tp in [(1.0, 1.0), (0.0, 0.0), (0.5, 0.5), (0.25, 0.1)]:
        v = aggregate([0], [0.5], [sq], [tp])["brier"]
        assert -1e-9 <= v <= 2 + 1e-9, (sq, tp, v)


def test_aggregate_no_longer_writes_a_lark_field():
    """LARK là ĝ theo câu hỏi, không tính được từ một chuỗi. Ghi Brier dưới tên lark từng làm sai tín hiệu."""
    assert "lark" not in aggregate(*make())


def test_brier_matches_the_formula_of_the_paper_on_a_tiny_vocabulary():
    """‖π − δ(y)‖² tính tay trên từ vựng 3 mục, so với cách tính nhanh 1 − 2p + Σp²."""
    pi, y = [0.6, 0.3, 0.1], 1
    direct = sum((p - (1.0 if v == y else 0.0)) ** 2 for v, p in enumerate(pi))
    fast = aggregate([1], [-math.log(pi[y])], [sum(p * p for p in pi)], [pi[y]])["brier"]
    assert fast == pytest.approx(direct)


def test_zero_surprisal_does_not_divide_by_zero():
    """Chuỗi mà mô hình đoán được hoàn hảo: tổng độ bất ngờ bằng 0."""
    r = aggregate([0, 0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0])
    assert math.isfinite(r["rsr"]) and r["rsr"] > 0


def test_rejects_mismatched_or_empty_input():
    with pytest.raises(ValueError):
        aggregate([], [], [], [])
    with pytest.raises(ValueError):
        aggregate([0, 0], [1.0], [0.5, 0.5], [0.6, 0.6])


def test_token_count_recorded():
    assert aggregate(*make(n=7))["n_tokens"] == 7

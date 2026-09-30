"""Định nghĩa Acc@4 và Pass@4 (Phụ lục A.4 bài RSR, chốt 30/09/2026)."""
import pytest

from src.common.config import load_config
from src.stage_c.metrics import acc_at_k, pass_at_k, report


RES = {"q1": [True, False, False, False],   # 1/4 lượt đúng
       "q2": [True, True, True, True],
       "q3": [False, False, False, False]}


def test_acc_at_4_is_mean_over_attempts_not_any_correct():
    """Đây chính là chỗ paper từng ghi sai: Acc@4 là trung bình, không phải 'có một lượt đúng'."""
    assert acc_at_k(RES) == pytest.approx((0.25 + 1.0 + 0.0) / 3)


def test_pass_at_4_counts_any_correct():
    assert pass_at_k(RES) == pytest.approx(2 / 3)


def test_pass_never_below_acc():
    assert pass_at_k(RES) >= acc_at_k(RES)


def test_rejects_uneven_attempts():
    with pytest.raises(ValueError, match="đúng 4 lượt"):
        acc_at_k({"q1": [True, False]})
    with pytest.raises(ValueError):
        pass_at_k({})


def test_report_uses_names_from_config():
    cfg = load_config()
    out = report(RES, cfg.eval.report)
    assert list(out) == ["acc@4", "pass@4"]
    with pytest.raises(KeyError):
        report(RES, ["acc@1"])


def test_eval_settings_follow_rsr_appendix_a4():
    e = load_config().eval
    assert e.n_samples == 4
    assert (e.temperature, e.top_p, e.top_k) == (0.6, 0.95, -1)
    assert e.on_truncation == "keep"


def test_eval_length_equals_training_length():
    """Nguyên tắc của A.4: độ dài sinh khi đánh giá bằng độ dài chuỗi tối đa khi tinh chỉnh."""
    for student in ("qwen1_5b", "qwen7b"):
        cfg = load_config(student=student)
        assert cfg.eval.max_new_tokens == cfg.training.max_seq_len

import pytest

from src.common.rule_score import DEFAULT_WEIGHTS, raw_features, rule_scores, tokenize, zscore


def test_tokenize_and_raw_features():
    text = "We Check the result, then verify it. Therefore since perhaps might check."
    f = raw_features(text)
    assert f["elaborated"] == len(tokenize(text)) == 12
    assert f["self_verification"] == pytest.approx(3 / 12)   # check, verify, check
    assert f["exploratory"] == pytest.approx(2 / 12)
    assert f["adaptive"] == pytest.approx(2 / 12)


def test_inflected_forms_not_counted():
    """Quyết định của đề tài: chỉ đếm từ nguyên vẹn."""
    assert raw_features("checking checked verifies")["self_verification"] == 0.0


def test_empty_text_safe():
    f = raw_features("")
    assert f["elaborated"] == 0.0 and all(f[k] == 0.0 for k in ("self_verification", "exploratory", "adaptive"))


def test_frequency_normalization_makes_lengths_comparable():
    short = "check " * 2 + "word " * 8              # 2/10
    long = "check " * 20 + "word " * 80             # 20/100, cùng tần suất
    assert raw_features(short)["self_verification"] == pytest.approx(raw_features(long)["self_verification"])


def test_zscore_edges():
    assert zscore([]) == []
    assert zscore([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]      # không chia cho 0
    z = zscore([1.0, 2.0, 3.0])
    assert z[1] == pytest.approx(0.0) and z[0] < 0 < z[2]
    assert sum(z) == pytest.approx(0.0)


def test_rule_scores_ordering_and_zero_sum():
    texts = [
        "word " * 100,                                        # dài, không từ khoá
        "check verify perhaps might therefore since " * 5,    # ngắn, dày từ khoá
        "word " * 50 + "check therefore",
    ]
    scores, feats = rule_scores(texts)
    assert len(scores) == 3 and len(feats) == 3
    assert sum(scores) == pytest.approx(0.0)                  # z-score có tổng 0, trọng số cộng lại vẫn 0
    assert scores[1] > scores[0]                              # dày từ khoá thắng dài suông
    assert feats[0]["elaborated"] == 100.0


def test_identical_texts_give_zero():
    scores, _ = rule_scores(["check therefore"] * 4)
    assert scores == [pytest.approx(0.0)] * 4


def test_weight_validation():
    with pytest.raises(ValueError):
        rule_scores(["a"], weights={"elaborated": 0.5, "adaptive": 0.4})
    with pytest.raises(ValueError):
        rule_scores(["a"], weights={"elaborated": 0.5, "khong_co_cach_do": 0.5})
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9

import pytest

from src.common.answers import extract_gold, grade, is_equiv, last_boxed, normalize_answer


@pytest.mark.parametrize("text,expected", [
    (r"so \boxed{72}.", "72"),
    (r"\boxed{\frac{a}{b}}", r"\frac{a}{b}"),
    (r"\boxed{1} then finally \boxed{2}", "2"),
    (r"\boxed{\{1,2\}}", r"\{1,2\}"),
    (r"answer \boxed 5 done", "5"),
    (r"\fbox{7}", "7"),
    (r"\boxed{}", None),
    (r"\boxed{12", None),          # bị cắt giữa chừng
    ("không có gì", None),
    ("", None),
    (None, None),
])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


@pytest.mark.parametrize("pred,gold", [
    (r"\frac{1}{2}", "0.5"),
    (r"\dfrac{3}{4}", r"\frac34"),
    (r"\$72", "72"),
    (r"72\text{ dollars}", "72"),
    (r"72 \text{ dollars}", "72"),
    ("1,000", "1000"),
    ("18.00", "18"),
    ("x=5", "5"),
    (r"10\%", "10"),
    (r"2\sqrt3", r"2\sqrt{3}"),
    ("(1, 2)", "(1,2)"),
    (r"\text{(A)}", "(A)"),
    (r"90^\circ", "90"),
    ("3/4", r"\frac{3}{4}"),
    (".5", "0.5"),
    ("5.", "5"),
    (r"-\frac{1}{2}", "-0.5"),
])
def test_equivalent(pred, gold):
    assert is_equiv(pred, gold)


@pytest.mark.parametrize("pred,gold", [
    ("5", "6"),
    ("3.14159", "3.14"),
    (r"\frac{1}{3}", "0.333"),
    ("(1,2)", "(2,1)"),
    ("", "5"),
    (None, "5"),
    ("5", None),
])
def test_not_equivalent(pred, gold):
    assert not is_equiv(pred, gold)


def test_grade_reasons():
    assert grade(r"Vậy \boxed{5}", "5").reason == "match"
    assert grade(r"Vậy \boxed{5}", "6").reason == "mismatch"
    g = grade("Đáp án là 5", "5")
    assert (g.correct, g.pred, g.reason) == (False, None, "no_boxed")


def test_extract_gold():
    assert extract_gold("gsm8k", "blah\n#### 1,234") == "1234"
    assert extract_gold("gsm8k", "không có dấu hiệu") is None
    assert extract_gold("math", r"... so the answer is $\boxed{\frac{1}{2}}$.") == r"\frac{1}{2}"
    with pytest.raises(ValueError):
        extract_gold("numina", "x")


def test_normalize_idempotent():
    for s in [r"\dfrac12", "x = 7", r"5\text{ cm}", "1,000", ".5"]:
        once = normalize_answer(s)
        assert normalize_answer(once) == once

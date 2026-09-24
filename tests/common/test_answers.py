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


# ---------------- so tương đương bằng ký hiệu (thêm 23/09/2026)
@pytest.mark.parametrize("a,b", [
    (r"2\sqrt{3}", r"\sqrt{12}"),
    (r"\frac{1}{\sqrt{2}}", r"\frac{\sqrt{2}}{2}"),      # ngoặc lồng nhau
    (r"\sqrt{\frac{4}{9}}", r"\frac{2}{3}"),
    (r"\sqrt[3]{27}", "3"),                               # căn bậc ba
    ("2^3", "8"),
    (r"\frac{\pi}{2}", "1.5707963267948966"),
    (r"\frac{3\sqrt{2}}{6}", r"\frac{1}{\sqrt{2}}"),
    (r"2\pi", "6.283185307179586"),
])
def test_sympy_equiv_accepts_same_value_written_differently(a, b):
    assert is_equiv(a, b) and is_equiv(b, a)              # phải đối xứng


@pytest.mark.parametrize("a,b", [
    (r"2\sqrt{3}", r"3\sqrt{2}"),
    (r"\frac{1}{3}", "0.33"),                             # xấp xỉ KHÔNG được coi là bằng
    ("-2", "2"),
    ("(1,2)", "(2,1)"),                                   # toạ độ: bỏ qua sympy, giữ so chuỗi
    ("x+1", "1+x"),                                       # còn ẩn số thì không kết luận
    ("yes", "no"),
])
def test_sympy_equiv_rejects_different_values(a, b):
    assert not is_equiv(a, b)


def test_sympy_equiv_never_raises_on_garbage():
    from src.common.answers import sympy_equiv
    for a, b in [("", "1"), ("\\", "}"), ("((((", "))))"), ("1/0", "0"), ("a" * 200, "1")]:
        assert sympy_equiv(a, b) in (True, False)


@pytest.mark.parametrize("pred,gold", [
    (r"4 \text{ and } -4", "-4"),        # sympy đọc 'and' như toán tử logic, phải chặn
    (r"2 \text{ or } 3", "3"),
    ("x", "1"),
    ("e", "2.718281828459045"),          # hằng số dạng chữ: không đoán
])
def test_sympy_rejects_text_that_python_would_parse_as_operator(pred, gold):
    assert not is_equiv(pred, gold)


@pytest.mark.parametrize("a,b", [
    (r"2^{-98}", r"\frac{1}{2^{98}}"),
    (r"2\left(1+\sqrt{2}+\sqrt{3}\right)", r"2+2\sqrt{2}+2\sqrt{3}"),
])
def test_sympy_keeps_real_equivalences_from_the_full_pool(a, b):
    assert is_equiv(a, b)

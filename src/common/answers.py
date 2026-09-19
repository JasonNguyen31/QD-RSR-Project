"""
Tách, chuẩn hoá và so khớp đáp án cuối theo quy ước chấm của PRM800K / MATH.

Đây là bản cài đặt lại theo đúng các bước chuẩn hoá của Hendrycks (MATH) và bộ chấm của PRM800K,
KHÔNG phải bản sao nguyên văn. Bộ chấm gốc còn thử tương đương ký hiệu bằng sympy, bản này thì không:
hai biểu thức đúng nhưng viết khác thứ tự (x^2+1 và 1+x^2) sẽ bị chấm sai. Sai lệch này thiên về loại bớt
chuỗi đúng, áp dụng đồng đều cho mọi phương án nên không làm lệch so sánh. Cần nêu trong paper khi mô tả cách chấm.

Quy ước: chỉ \\boxed{...} cuối cùng trong văn bản được coi là đáp án của mô hình (prompt đã yêu cầu như vậy).
Không có \\boxed thì chuỗi bị chấm sai với lý do "no_boxed".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction


# ---------------------------------------------------------------- tách đáp án
def last_boxed(text: str | None) -> str | None:
    """Nội dung của \\boxed{...} hoặc \\fbox{...} cuối cùng, khớp ngoặc lồng nhau. None nếu không có hoặc dở dang."""
    if not text:
        return None
    best, cmd_len = -1, 0
    for cmd in ("\\boxed", "\\fbox"):
        i = text.rfind(cmd)
        if i > best:
            best, cmd_len = i, len(cmd)
    if best < 0:
        return None
    j = best + cmd_len
    while j < len(text) and text[j] == " ":
        j += 1
    if j >= len(text):
        return None
    if text[j] != "{":  # dạng \boxed 5
        m = re.match(r"[^\s$]+", text[j:])
        return m.group(0) if m else None
    depth = 0
    for k in range(j, len(text)):
        c = text[k]
        if k > 0 and text[k - 1] == "\\" and c in "{}":
            continue  # \{ và \} là ký tự, không phải ngoặc nhóm
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                inner = text[j + 1:k].strip()
                return inner or None
    return None  # ngoặc không đóng, thường do chuỗi bị cắt


def extract_gold(source: str, solution: str) -> str | None:
    """Đáp án chuẩn từ lời giải của bộ dữ liệu. GSM8K: sau '####'. MATH: \\boxed cuối cùng."""
    if source == "gsm8k":
        if "####" not in solution:
            return None
        gold = solution.rsplit("####", 1)[1].strip().replace(",", "")
        return gold or None
    if source == "math":
        return last_boxed(solution)
    raise ValueError(f"Không biết nguồn dữ liệu '{source}'")


# ---------------------------------------------------------------- chuẩn hoá (theo Hendrycks math_normalize)
def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return string
    if string != f"{a}/{b}":
        return string
    return "\\frac{" + str(a) + "}{" + str(b) + "}"


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


_TEXT_WRAPPER = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\{([^{}]*)\}")
_THOUSANDS = re.compile(r"-?\d{1,3}(,\d{3})+(\.\d+)?")


def normalize_answer(ans: str | None) -> str:
    if ans is None:
        return ""
    s = str(ans)
    s = s.replace("\n", "").replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    s = _remove_right_units(s)
    s = s.replace("\\%", "").replace("%", "")
    s = _TEXT_WRAPPER.sub(r"\1", s)
    for spacing in ("\\,", "\\;", "\\:", "\\ "):
        s = s.replace(spacing, "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if s.count("=") == 1:  # "x = 5" -> "5" khi vế trái ngắn
        lhs, rhs = s.split("=")
        if len(lhs.strip()) <= 2:
            s = rhs
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    if _THOUSANDS.fullmatch(s):
        s = s.replace(",", "")
    s = _fix_a_slash_b(s)
    if len(s) > 1 and s.endswith("."):
        s = s[:-1]
    return s


# ---------------------------------------------------------------- so khớp
_INT_OR_DEC = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)")
_FRAC = re.compile(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}")


def _to_number(s: str) -> Fraction | None:
    """Số hữu tỉ chính xác nếu chuỗi (đã chuẩn hoá) là số nguyên, số thập phân hoặc \\frac{p}{q}."""
    if not s:
        return None
    if _INT_OR_DEC.fullmatch(s):
        try:
            return Fraction(s)
        except (ValueError, ZeroDivisionError):
            return None
    m = _FRAC.fullmatch(s)
    if m:
        sign, p, q = m.groups()
        if int(q) == 0:
            return None
        val = Fraction(int(p), int(q))
        return -val if sign else val
    return None


def is_equiv(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    a, b = normalize_answer(pred), normalize_answer(gold)
    if not a or not b:
        return False
    if a == b:
        return True
    na, nb = _to_number(a), _to_number(b)
    return na is not None and nb is not None and na == nb


@dataclass(frozen=True)
class Grade:
    correct: bool
    pred: str | None
    reason: str  # "match" | "mismatch" | "no_boxed"


def grade(text: str | None, gold: str | None) -> Grade:
    pred = last_boxed(text)
    if pred is None:
        return Grade(False, None, "no_boxed")
    if is_equiv(pred, gold):
        return Grade(True, pred, "match")
    return Grade(False, pred, "mismatch")

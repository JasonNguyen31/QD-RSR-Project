"""steps: các bước cuối cùng của một chuỗi cho Local Naturalness, dựng từ kết quả cắt bước của GLM-4.5-Air.

Phần thuần: không torch, không gọi mạng. Hai nơi dùng chung module này nên không thể lệch nhau:
    src/tools/audit_steps      đọc steps.glm.jsonl trên Mac, in ra module này làm gì với từng loại chuỗi
    src/stage_b/b1_localnat    chấm LocalNat trên GPU bằng đúng các bước đó

steps.glm.jsonl (src/tools/segment_steps ghi) là nhật ký gốc của các lượt gọi API và KHÔNG bị sửa. Bốn việc dưới
đây được áp LÚC ĐỌC, bằng hàm final_steps, nên chạy lại bao nhiêu lần cũng ra cùng kết quả và không tốn lượt gọi
API nào:

  1. Lùi ranh giới về đầu dòng (snap_to_line_start). segment_steps đặt ranh giới ở ký tự chữ-số đầu tiên của
     nhóm. Khi mô hình bỏ dấu đầu dòng ("1.", "-", "(a)", "###", "Step 2:") khỏi câu nó chép lại, ranh giới rơi
     ngay SAU dấu đó và dấu bị bỏ lại ở cuối bước trước. Ranh giới như vậy được lùi về đầu dòng. Nếu chỗ lùi về
     trùng ranh giới đứng trước (bước trước chỉ gồm mỗi dấu đầu dòng) thì hai bước gộp làm một.
  2. Dời ranh giới về trước khoảng trắng ngang (before_spaces). segment_steps đặt ranh giới SAU dấu cách
     ("x = 10. |Therefore"), trong khi token của Qwen mang dấu cách ở đầu (" Therefore") và token thuộc bước chứa
     ký tự đầu tiên của nó, nên chữ đầu của bước sẽ bị tính cho bước trước. Dời ranh giới về trước dấu cách và
     dấu tab (không dời qua dấu xuống dòng) thì chữ đó rơi đúng vào bước của nó, cùng quy ước với
     signals.sentence_ends. Ranh giới ở đầu dòng không đổi.
  3. Gộp bước quá ngắn (merge_tiny): bước có ít hơn MIN_STEP_CHARS ký tự khác trắng gộp vào bước kế tiếp, cùng
     quy ước với signals.sentence_ends. LocalNat lấy trung bình THEO BƯỚC, nên một bước chỉ có "1." hay "\\]"
     sẽ nặng ngang một bước trăm token.
  4. Chuỗi cắt hỏng (trạng thái json_error, truncated, api_error; khoảng 0,5%), theo thứ tự:
       a. cứu từ phản hồi thô (salvage): json.loads không đọc nổi, nhưng thứ cần lấy chỉ là CHỖ BẮT ĐẦU của từng
          nhóm, nên tách phản hồi theo các khoá "groupN" rồi định vị từng mảnh trong văn bản gốc bằng đúng hàm
          align của segment_steps. Chỉ nhận khi mọi nhóm định vị được và độ dài trả về nằm trong SALVAGE_RATIO
          so với chuỗi gốc; đây vẫn là cách cắt của GLM, chỉ là đọc khoan dung hơn. Cần cột raw (--keep-raw).
       b. cắt dự phòng theo quy tắc (fallback_ends): theo đoạn văn (dòng trống); chuỗi chỉ có một đoạn thì theo
          dòng; chỉ có một dòng thì theo dấu câu. KHÔNG loại chuỗi khỏi kho: loại sẽ làm kho của LocalNat khác
          kho của bảy phương án kia và làm câu chỉ có đúng k ứng viên tụt dưới k.
     Nguồn của từng chuỗi (glm, glm_salvaged, fallback_<quy tắc>) được ghi lại để báo cáo trong paper.

Đổi bất kỳ quy tắc nào ở đây thì tăng STEPS_VERSION: b1_localnat ghi số này vào từng dòng và từ chối chạy bù lên
file chấm bằng phiên bản khác.
"""
from __future__ import annotations

import hashlib
import re
from typing import Mapping, Sequence

from src.stage_b.signals import MIN_STEP_CHARS, sentence_ends
from src.tools.segment_steps import DONE, align

STEPS_VERSION = 1
SALVAGE_RATIO = (0.9, 1.1)       # tỷ lệ độ dài (chữ-số) của phần mô hình trả về trên chuỗi gốc để nhận bản cứu
FALLBACK_RULES = ("paragraph", "line", "sentence")

# Dấu đầu dòng: gạch và chấm đầu dòng, dấu trích, tiêu đề markdown, số hoặc chữ thứ tự ("1.", "2)", "(a)", "iv."),
# nhãn bước ("Step 3:", "Case 2."). Cho phép bọc trong ** hoặc __ và có dấu hai chấm theo sau.
_ORD = r"(?:\d{1,3}|[A-Za-z]|[ivxlIVXL]{1,5})"
_LABEL = r"(?:[Ss]tep|[Cc]ase|[Pp]art|STEP|CASE|PART)[ \t]*\d{1,3}"
_MARK = rf"(?:[-*+•·–—>]|\#{{1,6}}|\(?{_ORD}[.):]|\({_ORD}\)|{_LABEL}[.):\-]?)"
_MARKER_PREFIX = re.compile(rf"[ \t]*(?:\*\*|__)?{_MARK}(?:\*\*|__)?:?[ \t]*")
_PARAGRAPH = re.compile(r"\n(?:[ \t]*\n)+")
_LINE = re.compile(r"\n+")
_GROUP_KEY = re.compile(r'"\s*group[\s_]*\d+\s*"\s*:', re.I)
_SENT_SEP = re.compile(r'"\s*,\s*"')
_TAIL = re.compile(r"[\s,\]\}`]+$")


def text_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def valid_ends(ends, n_chars: int) -> bool:
    """Mốc kết thúc bước hợp lệ: số nguyên, dương, tăng dần, mốc cuối bằng độ dài văn bản."""
    return (isinstance(ends, (list, tuple)) and len(ends) > 0 and all(isinstance(e, int) for e in ends)
            and ends[0] > 0 and ends[-1] == n_chars and all(b > a for a, b in zip(ends, ends[1:])))


def line_start(text: str, pos: int) -> int:
    return text.rfind("\n", 0, pos) + 1


# ============================================================ 1. lùi ranh giới về đầu dòng
def cut_kind(text: str, cut: int) -> str:
    """Một ranh giới nằm ở đâu: 'line' (đầu dòng, kể cả sau phần thụt lề), 'marker' (ngay sau dấu đầu dòng),
    'mid' (giữa dòng, thường là sau một câu)."""
    ls = line_start(text, cut)
    if not text[ls:cut].strip():
        return "line"
    return "marker" if _MARKER_PREFIX.fullmatch(text, ls, cut) else "mid"


def snap_to_line_start(text: str, ends: Sequence[int]) -> tuple[list[int], list[dict]]:
    """Lùi mọi ranh giới đứng ngay sau dấu đầu dòng về đầu dòng đó. Trả về (mốc mới, các lần lùi).

    Mỗi lần lùi ghi: vị trí cũ, vị trí mới, dấu đầu dòng, và dropped = True khi ranh giới bị bỏ hẳn vì chỗ lùi về
    không còn đứng sau ranh giới trước (dấu đầu dòng nằm ở đầu văn bản, hoặc bước trước chỉ gồm mỗi dấu đó).
    """
    cuts: list[int] = []
    moves: list[dict] = []
    for c in ends[:-1]:
        new = line_start(text, c) if cut_kind(text, c) == "marker" else c
        prev = cuts[-1] if cuts else 0
        if new != c:
            moves.append({"from": c, "to": new, "marker": text[new:c].strip(), "dropped": new <= prev})
        if new > prev:
            cuts.append(new)
    return cuts + [len(text)], moves


# ============================================================ 2. dời ranh giới về trước khoảng trắng ngang
def before_spaces(text: str, ends: Sequence[int]) -> tuple[list[int], int]:
    """Dời mỗi ranh giới về trước dãy dấu cách và dấu tab đứng ngay trước nó. Trả về (mốc mới, số ranh giới đã dời).
    Ranh giới nào dời về tới đầu văn bản hoặc trùng ranh giới trước thì bị bỏ."""
    cuts: list[int] = []
    moved = 0
    for c in ends[:-1]:
        new = c
        while new > 0 and text[new - 1] in " \t":
            new -= 1
        moved += new != c
        if new > (cuts[-1] if cuts else 0):
            cuts.append(new)
    return cuts + [len(text)], moved


# ============================================================ 3. gộp bước quá ngắn
def merge_tiny(text: str, ends: Sequence[int], min_chars: int = MIN_STEP_CHARS) -> tuple[list[int], int]:
    """Bước có ít hơn min_chars ký tự khác trắng gộp vào bước kế tiếp; bước cuối quá ngắn thì gộp vào bước trước.
    Trả về (mốc mới, số bước đã gộp)."""
    out: list[int] = []
    start = merged = 0
    for e in ends:
        if len("".join(text[start:e].split())) >= min_chars:
            out.append(e)
            start = e
        else:
            merged += 1
    if not out:
        return [len(text)], max(0, len(ends) - 1)
    if out[-1] != len(text):
        out[-1] = len(text)
    return out, merged


# ============================================================ 4a. cứu phản hồi mà json.loads không đọc nổi
def _unescape(s: str, literal: bool) -> str:
    """Gỡ lối thoát JSON theo từng cặp. literal: chỉ gỡ dấu nháy, mọi gạch chéo khác là ký tự thường (giữ được
    \\nu, \\text, \\frac viết một gạch chéo); không literal: gỡ thêm \\\\, \\n, \\t, \\r như JSON chuẩn."""
    def fix(m: re.Match) -> str:
        ch = m.group(1)
        if ch == '"':
            return '"'
        if literal:
            return m.group(0)
        return {"\\": "\\", "n": "\n", "t": "\t", "r": "\r", "/": "/"}.get(ch, m.group(0))
    return re.sub(r"\\(.)", fix, s, flags=re.S)


def salvage_groups(raw: str) -> dict | None:
    """Đọc khoan dung một phản hồi của mô hình cắt bước: tách theo các khoá "groupN", mỗi mảnh là một nhóm câu.

    Chịu được những thứ làm json.loads hỏng: dấu nháy không thoát trong câu, xuống dòng thật trong chuỗi, dấu phẩy
    thừa, phản hồi bị cắt giữa chừng. Trả về hai bản đọc như segment_steps.parse_groups ('standard', 'literal'),
    hoặc None khi không thấy khoá nhóm nào.
    """
    marks = list(_GROUP_KEY.finditer(raw or ""))
    if not marks:
        return None
    groups = []
    for i, m in enumerate(marks):
        chunk = raw[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(raw)]
        chunk = _TAIL.sub("", chunk.strip().lstrip("[").strip())
        if chunk.startswith('"'):
            chunk = chunk[1:]
        if chunk.endswith('"') and not chunk.endswith('\\"'):
            chunk = chunk[:-1]
        sents = [s for s in _SENT_SEP.split(chunk) if s.strip()]
        if sents:
            groups.append(sents)
    if not groups:
        return None
    return {name: [[_unescape(s, name == "literal") for s in g] for g in groups] for name in ("standard", "literal")}


def salvage(text: str, raw: str | None) -> dict | None:
    """Thử cứu một chuỗi cắt hỏng từ phản hồi thô. Trả về kết quả của align kèm cờ accepted, hoặc None khi phản
    hồi không có gì để đọc."""
    parsed = salvage_groups(raw) if raw else None
    if parsed is None:
        return None
    a = align(text, parsed)
    lo, hi = SALVAGE_RATIO
    a["accepted"] = bool(a["unlocated"] == 0 and lo <= a["length_ratio"] <= hi and valid_ends(a["ends"], len(text)))
    return a


# ============================================================ 4b. cắt dự phòng theo quy tắc
def rule_ends(text: str, rule: str) -> list[int]:
    """Mốc bước theo một quy tắc: paragraph (sau mỗi dòng trống), line (sau mỗi lần xuống dòng), sentence (như
    cột local_nat cũ). Dấu xuống dòng thuộc bước đứng trước; bước quá ngắn được gộp."""
    if rule == "sentence":
        return sentence_ends(text) or [len(text)]
    if rule not in ("paragraph", "line"):
        raise ValueError(f"quy tắc cắt dự phòng không hợp lệ: {rule!r} (có: {FALLBACK_RULES})")
    pat = _PARAGRAPH if rule == "paragraph" else _LINE
    raw = sorted({m.end() for m in pat.finditer(text) if 0 < m.end() < len(text)}) + [len(text)]
    return merge_tiny(text, raw)[0]


def fallback_ends(text: str, order: Sequence[str] = FALLBACK_RULES) -> tuple[list[int], str]:
    """Quy tắc đầu tiên trong order cho ra từ hai bước trở lên. Trả về (mốc, tên quy tắc)."""
    for rule in order:
        ends = rule_ends(text, rule)
        if len(ends) >= 2:
            return ends, rule
    return [len(text)], order[-1]


# ============================================================ gộp cả bốn việc
def final_steps(text: str, row: Mapping | None) -> dict:
    """Các bước cuối cùng của một chuỗi. row là dòng hiện hành của chuỗi trong steps.glm.jsonl (None nếu chưa cắt).

    Trả về ends (mốc kết thúc từng bước, dùng thẳng cho signals.token_steps), source, moves (các lần lùi ranh
    giới), spaces_moved, tiny_merged, salvage (số đo của lần cứu, nếu có thử).
    """
    if not text:
        raise ValueError("chuỗi rỗng không có bước nào")
    if row is not None and row.get("text_md5") not in (None, text_md5(text)):
        raise ValueError(f"{row.get('tid')}: văn bản đã khác lúc cắt bước (text_md5 lệch); steps.glm.jsonl không còn "
                         f"khớp candidates.jsonl")
    out = {"moves": [], "spaces_moved": 0, "tiny_merged": 0, "salvage": None}
    if row is not None and row.get("status") in DONE and valid_ends(row.get("ends"), len(text)):
        ends, source = list(row["ends"]), "glm"
    else:
        s = salvage(text, row.get("raw")) if row is not None else None
        if s is not None:
            out["salvage"] = {k: s[k] for k in ("accepted", "n_groups", "n_steps", "unlocated", "length_ratio")}
        if s is not None and s["accepted"]:
            ends, source = list(s["ends"]), "glm_salvaged"
        else:
            ends, rule = fallback_ends(text)
            source = f"fallback_{rule}"
    if source.startswith("glm"):
        ends, out["moves"] = snap_to_line_start(text, ends)
        ends, out["spaces_moved"] = before_spaces(text, ends)
        ends, out["tiny_merged"] = merge_tiny(text, ends)
    if not valid_ends(ends, len(text)):
        raise ValueError(f"mốc bước không hợp lệ sau khi xử lý (nguồn {source})")
    return {"ends": ends, "source": source, **out}


def current_rows(rows: Sequence[Mapping]) -> dict:
    """Dòng hiện hành của từng chuỗi: steps.glm.jsonl được ghi nối, dòng sau cùng của một tid là kết quả đang dùng."""
    return {r["tid"]: r for r in rows}


def context_share(lengths: Sequence[int], k: int) -> list[float]:
    """Với từng bước từ bước thứ hai trở đi: phần ngữ cảnh đứng trước (tính theo độ dài) mà mô hình còn thấy khi
    chỉ được nhìn k bước liền trước. 1,0 nghĩa là thấy toàn bộ phần trước, tức bước đó được chấm y như GRAPE."""
    out = []
    for i in range(1, len(lengths)):
        before = sum(lengths[:i])
        out.append(sum(lengths[max(0, i - k):i]) / before if before else 1.0)
    return out

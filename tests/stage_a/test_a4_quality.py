import json

import pytest

from src.common.prompts import build_judge_prompt
from src.stage_a.a4_score_quality import combine, parse_judge_json

GOOD = """{"dimensional_evaluation": {
 "factual_accuracy": {"score": 0.9, "reason": "a"},
 "logical_rigor": {"score": 0.8, "reason": "b"},
 "solution_completeness": {"score": 0.7, "reason": "c"},
 "reasoning_efficiency": {"score": 0.6, "reason": "d"},
 "presentation_quality": {"score": 0.5, "reason": "e"}},
 "overall_score": 0.7, "overall_reason": "ok"}"""


def test_parse_clean_json():
    r = parse_judge_json(GOOD)
    assert r["overall_score"] == 0.7 and len(r["dimensions"]) == 5 and r["missing_dimensions"] == []


@pytest.mark.parametrize("wrap", [
    "```json\n{body}\n```",
    "```\n{body}\n```",
    "Here is my evaluation:\n{body}\nHope this helps.",
    "  {body}  ",
])
def test_parse_survives_wrapping(wrap):
    assert parse_judge_json(wrap.format(body=GOOD))["overall_score"] == 0.7


def test_parse_missing_dimension_is_recorded_not_fatal():
    body = GOOD.replace('"presentation_quality": {"score": 0.5, "reason": "e"}', '"x": {"score": 0.5}')
    r = parse_judge_json(body)
    assert r["overall_score"] == 0.7 and r["missing_dimensions"] == ["presentation_quality"]


@pytest.mark.parametrize("bad", [
    "",
    "không phải json",
    '{"dimensional_evaluation": {}}',                 # thiếu overall_score
    '{"overall_score": 1.7}',                          # ngoài khoảng
    '{"overall_score": -0.1}',
])
def test_parse_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_judge_json(bad)


def test_judge_prompt_shape():
    p = build_judge_prompt("Q?", "T", "S")
    assert "Q?" in p and "For your reference" in p and "S" in p
    assert "0.1 increments" in p and "combat score inflation" in p
    assert "{" in p and "{question}" not in p
    assert "For your reference" not in build_judge_prompt("Q?", "T")


def test_combine_alpha_and_missing_judge():
    cands = [{"tid": "t1", "qid": "q", "text": "check therefore " * 20},
             {"tid": "t2", "qid": "q", "text": "word " * 200},
             {"tid": "t3", "qid": "q", "text": "perhaps might " * 5}]
    judge = [{"tid": "t1", "overall_score": 1.0}, {"tid": "t2", "overall_score": 0.0}]
    rows = combine(cands, judge, alpha=0.5)
    by = {r["tid"]: r for r in rows}
    assert by["t3"]["llm_score"] is None and by["t3"]["qual"] is None    # chưa chấm thì để trống
    assert by["t3"]["rule_norm"] is None
    for t in ("t1", "t2"):                                               # qual = trộn hai nửa đã min-max
        assert by[t]["qual"] == pytest.approx(0.5 * by[t]["rule_norm"] + 0.5 * by[t]["llm_norm"])
        assert 0.0 <= by[t]["qual"] <= 1.0
    assert by["t1"]["llm_norm"] == 1.0 and by["t2"]["llm_norm"] == 0.0
    assert by["t2"]["n_words"] == 200

    rows0 = combine(cands, judge, alpha=0.0)                             # alpha=0 thì qual chính là llm_norm
    assert {r["tid"]: r["qual"] for r in rows0}["t1"] == pytest.approx(1.0)


def test_combine_puts_both_halves_on_the_same_scale():
    """Hồi quy 28/09: rule là điểm z trải rộng, judge nằm trong [0,1]. Trộn thẳng thì rule lấn át.
    Sau khi min-max theo câu hỏi, hai nửa phải có cùng biên độ, nên alpha=0,5 mới thật sự là chia đều."""
    cands = [{"tid": f"t{i}", "qid": "q", "text": ("check " * i) + "word " * (10 + 30 * i)} for i in range(4)]
    judge = [{"tid": f"t{i}", "overall_score": s} for i, s in enumerate([1.0, 0.9, 0.9, 0.8])]
    rows = combine(cands, judge, alpha=0.5)
    rn = [r["rule_norm"] for r in rows]
    ln = [r["llm_norm"] for r in rows]
    assert min(rn) == 0.0 and max(rn) == 1.0 and min(ln) == 0.0 and max(ln) == 1.0


def test_combine_constant_half_gives_neutral_score():
    cands = [{"tid": f"t{i}", "qid": "q", "text": "word " * (10 + i)} for i in range(3)]
    judge = [{"tid": f"t{i}", "overall_score": 1.0} for i in range(3)]   # giám khảo cho mọi chuỗi 1,0
    rows = combine(cands, judge, alpha=0.5)
    assert all(r["llm_norm"] == 0.5 for r in rows)


def test_combine_normalises_per_question():
    cands = [{"tid": "a1", "qid": "a", "text": "word " * 10}, {"tid": "a2", "qid": "a", "text": "word " * 20},
             {"tid": "b1", "qid": "b", "text": "word " * 300}, {"tid": "b2", "qid": "b", "text": "word " * 400}]
    judge = [{"tid": t, "overall_score": s} for t, s in [("a1", 0.2), ("a2", 0.3), ("b1", 0.8), ("b2", 0.9)]]
    by = {r["tid"]: r for r in combine(cands, judge, alpha=0.5)}
    assert by["a2"]["llm_norm"] == 1.0 and by["b1"]["llm_norm"] == 0.0      # so trong câu, không so giữa câu


def test_repairs_latex_backslashes_in_reason():
    r"""Giám khảo viết LaTeX vào lý giải làm JSON sai cú pháp; bộ đọc phải tự sửa.

    Lưu ý quan trọng: \f, \t, \b, \n, \r, \u LÀ ký tự thoát hợp lệ của JSON, nên \frac, \times, \boxed
    KHÔNG làm hỏng JSON — chúng chỉ âm thầm biến thành ký tự xuống trang hoặc tab trong phần lý giải,
    ảnh hưởng chữ chứ không ảnh hưởng điểm. Thứ thật sự gây lỗi "Invalid \escape" là \left, \cdot, \sqrt, \pi.
    """
    body = ('{"dimensional_evaluation": {"factual_accuracy": {"score": 0.6, '
            r'"reason": "uses \left( \cdot \right) and \sqrt{2} correctly"}}, '
            r'"overall_score": 0.6, "overall_reason": "\pi is fine"}')
    with pytest.raises(json.JSONDecodeError):
        json.loads(body)                      # JSON gốc thật sự hỏng
    r = parse_judge_json(body)                # nhưng bộ đọc của ta vẫn lấy được điểm
    assert r["overall_score"] == 0.6 and r["dimensions"]["factual_accuracy"] == 0.6


def test_truncated_response_says_so():
    cut = '{\n "dimensional_evaluation": {\n "factual_accuracy": {\n "score": 0.7,\n "reason": "The candidate'
    with pytest.raises(ValueError) as e:
        parse_judge_json(cut)
    assert "CẮT CỤT" in str(e.value) and "judge.max_tokens" in str(e.value)


def test_valid_escapes_untouched():
    from src.stage_a.a4_score_quality import repair_json_escapes

    for keep in (r'"a\nb"', r'"a\u00e9b"', r'"a\tb"', r'"a\\b"', r'"\frac"'):
        assert repair_json_escapes(keep) == keep, "ký tự thoát hợp lệ không được đụng vào"
    for bad, fixed in ((r'"\left"', r'"\\left"'), (r'"\cdot"', r'"\\cdot"'), (r'"\pi"', r'"\\pi"')):
        assert repair_json_escapes(bad) == fixed


def test_accepts_real_newlines_inside_strings():
    """Giám khảo xuống dòng thật giữa phần lý giải; JSON chuẩn coi đó là lỗi, ta thì chấp nhận."""
    body = ('{"dimensional_evaluation": {"factual_accuracy": {"score": 0.8,\n'
            ' "reason": "dòng một\nrồi dòng hai"}},\n "overall_score": 0.8, "overall_reason": "ổn"}')
    with pytest.raises(json.JSONDecodeError):
        json.loads(body)
    assert parse_judge_json(body)["overall_score"] == 0.8


def test_strips_text_around_json():
    body = 'Here is the evaluation:\n{"overall_score": 0.4, "overall_reason": "x"}\nLet me know if you need more.'
    assert parse_judge_json(body)["overall_score"] == 0.4


def test_salvage_from_broken_json():
    """JSON hỏng hẳn nhưng vẫn còn overall_score: lấy điểm và đánh dấu salvaged."""
    broken = '{"dimensional_evaluation": {"x": {"score": 0.9, "reason": "he said "hi" oops"}}, "overall_score": 0.65}'
    r = parse_judge_json(broken)
    assert r["overall_score"] == 0.65 and r["salvaged"] == "regex_overall"


def test_salvage_averages_dimensions_when_cut_before_overall():
    cut = ('{"dimensional_evaluation": {"a": {"score": 0.8, "reason": "x"}, "b": {"score": 0.6, "reason": "y"}, '
           '"c": {"score": 1.0, "reason": "z')
    r = parse_judge_json(cut)
    assert r["overall_score"] == pytest.approx(0.8) and r["salvaged"] == "mean_of_3_dimensions"


def test_salvage_refuses_when_too_little():
    from src.stage_a.a4_score_quality import salvage_score
    assert salvage_score('{"score": 0.5, "reason": "chỉ có một tiêu chí') is None
    with pytest.raises(ValueError):
        parse_judge_json("mô hình trả lời bằng văn xuôi, không có điểm nào")


def test_clean_json_is_not_marked_salvaged():
    assert "salvaged" not in parse_judge_json(GOOD)

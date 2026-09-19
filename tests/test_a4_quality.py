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
    assert by["t1"]["qual"] == pytest.approx(0.5 * by["t1"]["rule_score"] + 0.5 * 1.0)
    assert by["t2"]["n_words"] == 200
    assert by["t1"]["qual"] > by["t2"]["qual"]

    rows0 = combine(cands, judge, alpha=0.0)                             # alpha=0 thì qual chính là llm_score
    assert {r["tid"]: r["qual"] for r in rows0}["t1"] == pytest.approx(1.0)

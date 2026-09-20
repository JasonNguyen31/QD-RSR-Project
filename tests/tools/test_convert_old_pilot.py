import pytest

from src.tools.convert_old_pilot import convert

GSM = [{"qid": f"gsm8k_pilot_{i:03d}", "source": "gsm8k", "orig_index": 100 + i, "question": f"q{i}",
        "answer": str(i), "solution": f"làm...\n#### {i}"} for i in range(5)]
MATH = [{"qid": f"mathpilot_{i:03d}", "source": "math", "question": f"m{i}", "level": f"Level {4 + i % 2}",
         "type": "algebra", "solution": f"vậy $\\boxed{{{i}}}$"} for i in range(4)]


def test_convert_keeps_qids_and_adds_gold():
    rows, warnings = convert(GSM, MATH)
    assert warnings == [] and len(rows) == 9
    assert {r["qid"] for r in rows} == {r["qid"] for r in GSM + MATH}
    by = {r["qid"]: r for r in rows}
    assert by["gsm8k_pilot_003"]["gold"] == "3" and by["gsm8k_pilot_003"]["level"] is None
    assert by["mathpilot_001"]["gold"] == "1" and by["mathpilot_001"]["level"] == 5
    assert len({r["source"] for r in rows[:4]}) == 2            # đã xáo trộn, tiền tố là mẫu trộn
    assert [r["qid"] for r in convert(GSM, MATH)[0]] == [r["qid"] for r in rows]   # tái lập


def test_convert_warns_on_answer_mismatch_and_rejects_bad_rows():
    bad = [dict(GSM[0], answer="999")]
    _, warnings = convert(bad, [])
    assert len(warnings) == 1 and "999" in warnings[0]
    with pytest.raises(SystemExit):
        convert([], [dict(MATH[0], solution="không có đáp án")])
    with pytest.raises(SystemExit):
        convert(GSM, [dict(MATH[0], qid=GSM[0]["qid"])])

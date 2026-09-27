import json

from src.tools.inspect_rsr_ref import load_answers, survey


def _rec(q, a):
    return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def test_load_answers_reads_string_and_list_messages(tmp_path):
    p = tmp_path / "x.json"
    recs = [_rec(" q1 ", "a1"), {"messages": str(_rec("q2", "a2")["messages"])}]
    p.write_text(json.dumps(recs))
    got = load_answers(p)
    assert got == {"q1": ["a1"], "q2": ["a2"]}


def test_survey_counts_exact_matches_and_length_ratio():
    selected = {"q1": ["<think> long long long long </think> x"], "q2": ["short"], "q3": ["<think> y"]}
    teacher = {"q1": ["a b"], "q2": ["short ", "other"], "q4": ["z"]}
    r = survey(selected, teacher)
    assert r["common"] == 2                         # q1, q2
    assert r["selected_from_teacher"] == 1          # chỉ q2 trùng nguyên văn (bỏ khoảng trắng đầu cuối)
    assert r["think_share"] == round(2 / 3, 4)
    assert r["words"]["ratio"] > 1


def test_survey_uses_tokenizer_when_given():
    r = survey({"q": ["a b c d"]}, {"q": ["a b"]}, tokenize=lambda s: s.split())
    assert r["tokens"]["ratio"] == 2.0

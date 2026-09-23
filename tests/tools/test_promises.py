"""Kiểm thử hai công cụ cho hai lời hứa trong paper. Không gọi mạng, không cần GPU."""
import pytest

from src.tools.compare_embedding import auc, build_pairs, chamfer, parse_steps, parse_verdict, spread
from src.tools.judge_stability import sample_chains, summarize


# ---------------- judge_stability
def test_summarize_perfectly_stable():
    rows = [{"tid": "t1", "overall_score": 0.8} for _ in range(5)]
    s = summarize(rows, 5)
    assert s["chains"] == 1 and s["identical_share"] == 1.0
    assert s["std_mean"] == 0.0 and s["span_ge_0.2"] == 0


def test_summarize_flags_wide_span():
    rows = [{"tid": "t1", "overall_score": v} for v in (0.5, 0.9, 0.7)] + \
           [{"tid": "t2", "overall_score": 0.4} for _ in range(3)]
    s = summarize(rows, 3)
    assert s["chains"] == 2 and s["span_max"] == pytest.approx(0.4)
    assert s["span_ge_0.2"] == 1 and s["identical_every_repeat"] == 1


def test_summarize_ignores_single_scored_chain():
    assert summarize([{"tid": "t1", "overall_score": 0.5}], 5)["chains"] == 0


def test_sample_chains_covers_every_group():
    qs = {f"q{i}": {"qid": f"q{i}", "source": "math", "level": 3 + i % 3} for i in range(30)}
    cands = [{"tid": f"q{i}|A|{j}", "qid": f"q{i}"} for i in range(30) for j in range(3)]
    got = sample_chains(cands, qs, 9, seed=1)
    levels = {qs[c["qid"]]["level"] for c in got}
    assert len(got) == 9 and levels == {3, 4, 5}


# ---------------- compare_embedding
@pytest.mark.parametrize("text,expected", [
    ("1. Set up an equation\n2. Solve for x\n3. Check the answer", 3),
    ("- Factor it\n- Substitute\n", 2),
    ("1) A\n2) B", 2),
])
def test_parse_steps_numbered_and_bulleted(text, expected):
    assert len(parse_steps(text)) == expected


def test_parse_steps_falls_back_to_sentences():
    got = parse_steps("First we rewrite the whole expression. Then we substitute the value of x into it.")
    assert len(got) == 2


@pytest.mark.parametrize("text,expected", [
    ('{"verdict": "different", "reason": "x"}', "different"),
    ('```json\n{"verdict":"same","reason":"y"}\n```', "same"),
    ("They follow different strategies.", "different"),
    ("blah", None),
])
def test_parse_verdict(text, expected):
    assert parse_verdict(text) == expected


def test_chamfer_symmetric_and_zero_for_identical():
    import numpy as np
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert chamfer(a, b) == pytest.approx(0.0)
    c = np.array([[-1.0, 0.0]])
    assert chamfer(a, c) == pytest.approx(chamfer(c, a))
    assert chamfer(a, c) > 0


def test_auc_edges():
    assert auc([1.0, 0.0], [1, 0]) == pytest.approx(1.0)
    assert auc([0.0, 1.0], [1, 0]) == pytest.approx(0.0)
    assert auc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)    # bằng nhau thì tính nửa điểm
    assert auc([1.0, 2.0], [1, 1]) is None                  # chỉ có một lớp


def test_spread_iqr():
    s = spread([0.1, 0.2, 0.3, 0.4])
    assert s["n"] == 4 and s["iqr"] == pytest.approx(s["p75"] - s["p25"])


def test_build_pairs_three_per_question_and_skips_small():
    cands = [{"tid": f"q1|A|{i}", "qid": "q1"} for i in range(4)] + \
            [{"tid": "q2|A|0", "qid": "q2"}, {"tid": "q2|A|1", "qid": "q2"}]
    pairs = build_pairs(cands, n_questions=5, seed=0)
    assert len(pairs) == 3 and {p["qid"] for p in pairs} == {"q1"}   # q2 chỉ có 2 ứng viên, bị bỏ

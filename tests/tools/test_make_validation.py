import pytest

from src.tools.make_validation import build_matched, check_clean, stratum_sizes

COMP = [{"source": "math", "level": 4, "n": 800}, {"source": "math", "level": 5, "n": 600},
        {"source": "math", "level": 3, "n": 400}, {"source": "gsm8k", "level": None, "n": 200}]


def _pools():
    math = [{"qid": f"m{i}", "source": "math", "level": 1 + i % 5, "question": f"bài {i}", "gold": "1"}
            for i in range(1000)]
    gsm = [{"qid": f"g{i}", "source": "gsm8k", "level": None, "question": f"gsm {i}", "gold": "2"}
           for i in range(300)]
    return {"math": math, "gsm8k": gsm}


def test_sizes_follow_training_mix_and_sum_to_n():
    sizes = stratum_sizes(COMP, 200)
    assert [k for *_, k in sizes] == [80, 60, 40, 20]
    assert sum(k for *_, k in stratum_sizes(COMP, 37)) == 37


def test_matched_set_has_no_easy_math_and_excludes_train_and_pilot():
    pools = _pools()
    train_qids = {f"m{i}" for i in range(0, 1000, 7)}
    pilot_texts = {"bài 3", "gsm 5"}
    val = build_matched(pools, stratum_sizes(COMP, 200), train_qids, pilot_texts, seed=2042)
    assert len(val) == 200
    assert not any(r["source"] == "math" and r["level"] in (1, 2) for r in val)
    assert not {r["qid"] for r in val} & train_qids
    assert not any(r["question"] in pilot_texts for r in val)
    check_clean(val, train_qids, pilot_texts)


def test_same_seed_same_set():
    pools = _pools()
    a = build_matched(pools, stratum_sizes(COMP, 200), set(), set(), seed=2042)
    b = build_matched(pools, stratum_sizes(COMP, 200), set(), set(), seed=2042)
    assert [r["qid"] for r in a] == [r["qid"] for r in b]


def test_refuses_when_stratum_too_small():
    pools = _pools()
    with pytest.raises(SystemExit):
        build_matched(pools, [("math", 5, 10_000)], set(), set(), seed=1)


def test_check_clean_catches_train_overlap():
    with pytest.raises(SystemExit):
        check_clean([{"qid": "m1", "question": "x"}], {"m1"}, set())

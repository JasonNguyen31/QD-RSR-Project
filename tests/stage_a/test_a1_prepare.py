import random

import pytest

from src.common.io_utils import write_jsonl
from src.stage_a import a1_prepare as a1

COMPOSITION = [
    {"source": "math", "level": 4, "n": 8},
    {"source": "math", "level": 5, "n": 6},
    {"source": "math", "level": 3, "n": 4},
    {"source": "gsm8k", "level": None, "n": 5},
]


def fake_math_rows(n=300):
    rows = []
    for i in range(n):
        level = "Level ?" if i % 50 == 0 else f"Level {1 + i % 5}"
        sol = f"Ta được $\\boxed{{{i}}}$." if i % 37 != 5 else "lời giải không có đáp án"
        rows.append({"problem": f"  Bài   TOÁN số {i} ?", "solution": sol, "level": level, "type": "algebra"})
    return rows


def fake_gsm_rows(n=120):
    return [{"question": f"Bài {k}", "answer": f"tính toán\n#### {k},000"} for k in range(n)]


def test_parse_level():
    assert a1.parse_level("Level 4") == 4
    assert a1.parse_level("Level ?") is None
    assert a1.parse_level("") is None and a1.parse_level(None) is None


def test_norm_matches_old_behavior():
    assert a1.norm("  Bài   TOÁN\n số 1 ") == "bài toán số 1"


def test_dedup_and_qid_stability():
    rows = fake_math_rows()
    banned = {a1.norm(rows[10]["problem"]), a1.norm(rows[11]["problem"])}
    kept, removed = a1.dedup_against(rows, banned)
    assert removed == 2 and len(kept) == 298
    recs, dropped = a1.build_math_records(kept)
    no_gold = len(dropped)
    # câu không có \boxed bị bỏ nhưng vẫn chiếm số thứ tự: qid của câu còn lại không nhảy
    assert no_gold == len([i for i in range(300) if i % 37 == 5 and i not in (10, 11)])
    assert recs[0]["qid"] == "math_00000" and recs[0]["gold"] == "0"
    assert all(d["qid"].startswith("math_") and "tail" in d for d in dropped)
    ids = [int(r["qid"].split("_")[1]) for r in recs]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_gsm_records_and_exclusion():
    recs = a1.build_gsm_records(fake_gsm_rows(), exclude_idx={3, 4})
    assert len(recs) == 118 and recs[3]["qid"] == "gsm8k_00005" and recs[3]["orig_index"] == 5
    assert recs[0]["gold"] == "0000" or recs[0]["gold"] == "0"   # '0,000' -> bỏ dấu phẩy
    assert next(r for r in recs if r["orig_index"] == 7)["gold"] == "7000"


def test_validation_split_reproduces_old_algorithm():
    math_all, _dropped = a1.build_math_records(a1.dedup_against(fake_math_rows(), set())[0])
    gsm_all = a1.build_gsm_records(fake_gsm_rows(), set())
    val, gsm_pool, math_pool = a1.split_validation(gsm_all, math_all, 10, 42)

    # bản cũ: random.seed(42); sample(gsm_pool, N); sample(math_pool, N)
    random.seed(42)
    old_gsm = random.sample(gsm_all, 10)
    old_math = random.sample(math_all, 10)
    assert [r["qid"] for r in val] == [r["qid"] for r in old_gsm + old_math]
    assert len(gsm_pool) == len(gsm_all) - 10 and len(math_pool) == len(math_all) - 10
    assert not {r["qid"] for r in val} & {r["qid"] for r in gsm_pool + math_pool}


def test_sample_train_composition_determinism_no_leak():
    math_all, _dropped = a1.build_math_records(a1.dedup_against(fake_math_rows(), set())[0])
    gsm_all = a1.build_gsm_records(fake_gsm_rows(), set())
    val, gsm_pool, math_pool = a1.split_validation(gsm_all, math_all, 10, 42)
    t1 = a1.sample_train(gsm_pool, math_pool, COMPOSITION, 1042)
    t2 = a1.sample_train(list(reversed(gsm_pool)), list(reversed(math_pool)), COMPOSITION, 1042)
    assert [r["qid"] for r in t1] == [r["qid"] for r in t2]          # không phụ thuộc thứ tự file
    counts = {}
    for r in t1:
        counts[(r["source"], r["level"])] = counts.get((r["source"], r["level"]), 0) + 1
    assert counts == {("math", 4): 8, ("math", 5): 6, ("math", 3): 4, ("gsm8k", None): 5}
    assert len({r["source"] for r in t1[:8]}) == 2                    # tiền tố là mẫu trộn
    a1.check_no_leak(t1, val, set())
    with pytest.raises(SystemExit):
        a1.check_no_leak(t1, t1[:3], set())                           # train giao val
    with pytest.raises(SystemExit):
        a1.check_no_leak(t1, val, {a1.norm(t1[0]["question"])})       # trùng MATH-500
    with pytest.raises(SystemExit):
        a1.sample_train(gsm_pool, math_pool, [{"source": "math", "level": 4, "n": 10_000}], 1)


def test_compare_with_old(tmp_path):
    new = {"gsm8k_pool": [{"qid": "a"}, {"qid": "b"}], "math_pool": [{"qid": "x"}], "validation": [{"qid": "v"}]}
    same, msgs = a1.compare_with_old(tmp_path, new)
    assert same and "chưa có bản cũ" in msgs[0]
    write_jsonl(tmp_path / "gsm8k_pool.jsonl", [{"qid": "a"}, {"qid": "b"}])
    write_jsonl(tmp_path / "math_pool.jsonl", [{"qid": "x"}])
    assert a1.compare_with_old(tmp_path, new)[0]
    write_jsonl(tmp_path / "gsm8k_pool.jsonl", [{"qid": "b"}, {"qid": "a"}])
    same, msgs = a1.compare_with_old(tmp_path, new)
    assert not same and "KHÁC THỨ TỰ" in "".join(msgs)
    write_jsonl(tmp_path / "gsm8k_pool.jsonl", [{"qid": "a"}, {"qid": "c"}])
    same, msgs = a1.compare_with_old(tmp_path, new)
    assert not same and "LỆCH" in "".join(msgs)

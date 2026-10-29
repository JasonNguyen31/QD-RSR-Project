"""Kiểm thử ba quy tắc thêm 27/10/2026 sau phần chẩn đoán diagnose_fq: fq_judge (rank_sum với điểm giám khảo),
fq_veto và fq_veto_judge (veto_topk), và phép đối chứng familiarity_within_teacher (signal_within_teacher)."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.config import load_config
from src.stage_b import b2_select as b2
from test_b2_select import make_workdir

NEW = ("fq_judge", "fq_veto", "fq_veto_judge", "familiarity_within_teacher")


@pytest.fixture
def inp(tmp_path):
    cfg = make_workdir(tmp_path, n_q=40)
    return b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")


def _picks(inp, method):
    cfg = load_config(method=method, student="qwen1_5b")
    qs = inp.questions(cfg.select.pool, cfg.select.rule)
    return cfg, qs, b2.select_all(qs, cfg)


# ---------------------------------------------------------------- veto_topk
def test_veto_count_never_leaves_fewer_than_k():
    third = 1 / 3
    assert [b2.veto_count(n, 3, third) for n in (3, 4, 5, 6, 9, 12)] == [0, 1, 1, 2, 3, 4]
    assert b2.veto_count(4, 3, 0.9) == 1 and b2.veto_count(10, 3, 0.0) == 0


def test_veto_drops_the_worst_quality_then_takes_the_most_familiar():
    fam = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])       # GRAPE sẽ lấy 0, 1, 2
    qual = np.array([0.0, 0.1, 0.9, 0.8, 0.7, 0.6])      # n = 6 -> loại 2 chuỗi Qual thấp nhất: 0 và 1
    p = b2.pick_veto_topk(fam, qual, 3, 1 / 3)
    assert p["idx"] == [2, 3, 4] and p["n_dropped"] == 2 and p["tie"] is False


def test_veto_keeps_grape_choice_when_its_chains_are_not_the_worst():
    fam = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    qual = np.array([0.5, 0.6, 0.7, 0.0, 0.1, 0.9])
    assert b2.pick_veto_topk(fam, qual, 3, 1 / 3)["idx"] == [0, 1, 2]


def test_veto_ties_drop_the_later_chain_in_the_fixed_order():
    fam = np.array([0.1, 0.9, 0.8, 0.7])
    qual = np.array([0.0, 0.0, 0.5, 0.5])                # n = 4 -> loại 1; hoà giữa 0 và 1 -> loại chuỗi 1 (đứng sau)
    p = b2.pick_veto_topk(fam, qual, 3, 1 / 3)
    assert p["idx"] == [0, 2, 3] and p["tie"] is True


def test_veto_with_three_chains_takes_all():
    p = b2.pick_veto_topk(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]), 3, 1 / 3)
    assert p["idx"] == [0, 1, 2] and p["n_dropped"] == 0


# ---------------------------------------------------------------- signal_within_teacher
def test_signal_within_teacher_keeps_the_correct_only_quota():
    teachers = ["qwen72b", "deepseek", "qwen72b", "llama70b", "deepseek", "qwen72b"]
    fam = np.array([0.1, 0.2, 0.9, 0.5, 0.8, 0.3])
    p = b2.pick_signal_within_teacher(teachers, fam, 3, "max")
    assert Counter(teachers[i] for i in p["idx"]) == Counter(teachers[:3])
    assert p["idx"] == [2, 4, 5]


def test_familiarity_within_teacher_matches_correct_only_shares_on_every_question(inp):
    _, qs, fw = _picks(inp, "familiarity_within_teacher")
    _, _, co = _picks(inp, "correct_only")
    for qid, q in qs.items():
        assert Counter(q["teacher"][i] for i in fw[qid]["idx"]) == Counter(q["teacher"][i] for i in co[qid]["idx"])


# ---------------------------------------------------------------- cấu hình và kho giả
def test_new_configs_load_and_stay_out_of_all():
    base = load_config(student="qwen1_5b")
    listed = {m for m, _ in b2.method_list(base, Path(base.root))}
    for m in NEW:
        cfg = load_config(method=m, student="qwen1_5b")
        assert cfg.select.name == m and m not in listed
        assert b2.selection_tag(cfg) == f"{m}.k3"
    assert b2.rank_spec(load_config(method="fq_judge", student="qwen1_5b"))[0] == ["grape", "llm_quality"]


@pytest.mark.parametrize("method", NEW)
def test_every_new_rule_gives_k_distinct_chains_per_question(inp, method):
    cfg, qs, picks = _picks(inp, method)
    assert set(picks) == set(qs)
    for qid, p in picks.items():
        assert len(p["idx"]) == 3 == len(set(p["idx"]))


def test_fq_judge_uses_raw_judge_scores(inp):
    _, qs, picks = _picks(inp, "fq_judge")
    q = next(iter(qs.values()))
    assert b2.signal_values(q, "llm_quality").tolist() == [float(x) for x in q["llm"]]


def test_fq_veto_never_selects_a_vetoed_chain(inp):
    _, qs, picks = _picks(inp, "fq_veto")
    for qid, p in picks.items():
        qual = qs[qid]["qual"]
        n = len(qual)
        d = b2.veto_count(n, 3, 1 / 3)
        if d:
            cutoff = sorted(qual)[d - 1]
            assert all(qual[i] >= cutoff for i in p["idx"])

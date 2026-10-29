"""Kiểm thử hai thí nghiệm thêm 29/10/2026: kho bỏ một mô hình dạy (exclude_teachers) và FQ-Select theo hạn mức
mô hình dạy (veto_within_teacher)."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.config import load_config
from src.common.io_utils import read_json, read_jsonl
from src.stage_b import b2_select as b2
from test_b2_select import make_workdir

NOQWEN = ("correct_only_noqwen", "grape_noqwen", "fq_veto_judge_noqwen")


@pytest.fixture
def inp(tmp_path):
    cfg = make_workdir(tmp_path, n_q=40)
    return b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")


def test_veto_within_teacher_keeps_quota_and_vetoes_inside_each_teacher():
    teachers = ["qwen72b", "deepseek", "qwen72b", "llama70b", "qwen72b", "deepseek", "qwen72b"]
    fam = np.array([0.9, 0.1, 0.8, 0.5, 0.7, 0.6, 0.2])
    veto = np.array([0.0, 0.5, 0.9, 0.5, 0.8, 0.4, 0.7])
    # quota từ 3 chuỗi đầu: qwen72b 2, deepseek 1. Qwen có 4 chuỗi: loại min(4-2, floor(4/3)) = 1, là chuỗi 0 (veto 0)
    p = b2.pick_veto_within_teacher(teachers, fam, veto, 3, 1 / 3)
    assert Counter(teachers[i] for i in p["idx"]) == Counter(teachers[:3])
    assert p["idx"] == [2, 4, 5]
    assert 0 not in p["idx"] and p["n_dropped"] >= 1


def test_without_teachers_drops_chains_and_short_questions():
    q = {"qid": "a", "tids": ["t1", "t2", "t3", "t4"], "teacher": ["qwen72b", "deepseek", "llama70b", "deepseek"],
         "qual": np.array([0.1, 0.2, 0.3, 0.4]), "signals": {"grape": np.array([1.0, 2.0, 3.0, 4.0])},
         "vecs": np.zeros((4, 2)), "llm": [0.1, 0.2, 0.3, 0.4], "note": "x"}
    short = dict(q, qid="b", teacher=["qwen72b", "qwen72b", "deepseek", "llama70b"])
    out = b2.without_teachers({"a": q, "b": short}, ["qwen72b"], 3)
    assert set(out) == {"a"}
    a = out["a"]
    assert a["tids"] == ["t2", "t3", "t4"] and a["teacher"] == ["deepseek", "llama70b", "deepseek"]
    assert a["signals"]["grape"].tolist() == [2.0, 3.0, 4.0] and a["vecs"].shape == (3, 2) and a["note"] == "x"
    assert q["tids"] == ["t1", "t2", "t3", "t4"]          # không sửa kho gốc


@pytest.mark.parametrize("method", NOQWEN + ("fq_veto_judge_within_teacher",))
def test_new_configs_stay_out_of_all(method):
    base = load_config(student="qwen1_5b")
    assert method not in {m for m, _ in b2.method_list(base, Path(base.root))}
    assert load_config(method=method, student="qwen1_5b").select.name == method


def test_noqwen_methods_share_one_question_set_without_qwen(tmp_path):
    cfg = make_workdir(tmp_path, n_q=40)
    inp = b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")
    sets = {}
    for m in NOQWEN:
        c = load_config(method=m, student="qwen1_5b")
        s = b2.run_one(inp, c, tmp_path)
        rows = read_jsonl(tmp_path / f"train.{b2.selection_tag(c)}.jsonl")
        assert all(r["teacher"] != "qwen72b" for r in rows)
        assert s["exclude_teachers"] == ["qwen72b"]
        sets[m] = {r["qid"] for r in rows}
    assert len({frozenset(v) for v in sets.values()}) == 1


def test_within_teacher_variant_matches_correct_only_shares(inp):
    c = load_config(method="fq_veto_judge_within_teacher", student="qwen1_5b")
    qs = inp.questions(c.select.pool, c.select.rule)
    picks = b2.select_all(qs, c)
    co = b2.select_all(qs, load_config(method="correct_only", student="qwen1_5b"))
    for qid, q in qs.items():
        assert Counter(q["teacher"][i] for i in picks[qid]["idx"]) == Counter(q["teacher"][i] for i in co[qid]["idx"])

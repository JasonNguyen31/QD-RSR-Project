"""teacher_pilot: phần tính toán trên dữ liệu giả và một lượt chạy trọn trên kho thu nhỏ."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_b"))

from src.common.config import load_config
from src.common.io_utils import read_json, write_jsonl
from src.tools import teacher_pilot as tp
from test_b2_select import TEACHERS, make_workdir


def test_split_tid_works_from_the_right():
    assert tp.split_tid("math_00001|qwen72b|2") == ("math_00001", "qwen72b", 2)
    assert tp.split_tid("a|b|c|llama70b|0") == ("a|b|c", "llama70b", 0)


def test_question_means_keeps_only_complete_questions_and_first_s_chains():
    fam = {"q1|a|0": -1.0, "q1|a|1": -3.0, "q1|b|0": -2.0, "q2|a|0": -1.0}      # q2 thiếu mô hình b
    qids, mat = tp.question_means(fam, list(fam), ["a", "b"], 2)
    assert qids == ["q1"] and mat.tolist() == [[-2.0, -2.0]]
    _q, mat1 = tp.question_means(fam, list(fam), ["a", "b"], 1)
    assert mat1.tolist() == [[-1.0, -2.0]]


def test_draw_samples_has_no_repeats_and_stratified_keeps_quota():
    rng = np.random.default_rng(0)
    d = tp.draw_samples(50, 10, 200, rng)
    assert d.shape == (200, 10) and all(len(set(r)) == 10 for r in d)
    strata = np.array(["a"] * 30 + ["b"] * 20)
    d = tp.draw_samples(50, 10, 100, rng, strata)
    assert d.shape == (100, 10) and all(len(set(r)) == 10 for r in d)
    assert all((r < 30).sum() == 6 for r in d)                                  # 10 · 30/50 = 6


def test_pick_rate_follows_the_gap_between_teachers():
    rng = np.random.default_rng(1)
    easy = np.column_stack([rng.normal(0, 1, 400), rng.normal(5, 1, 400)])     # cột 1 hơn hẳn
    assert tp.pick_rate(easy, 1, 5, 500, rng)["p"] > 0.99
    hard = np.column_stack([rng.normal(0, 1, 400), rng.normal(0.05, 1, 400)])  # gần như hòa
    assert tp.pick_rate(hard, 1, 5, 500, rng)["p"] < 0.7
    assert tp.pick_rate(easy, 1, 400, 20, rng)["p"] == 1.0                      # m = Q: mẫu là cả kho


def test_smallest_m_requires_all_larger_sizes_to_pass():
    rows = [{"m": 5, "p": 0.99}, {"m": 10, "p": 0.90}, {"m": 20, "p": 0.97}, {"m": 40, "p": 0.99}]
    assert tp.smallest_m(rows, 0.95) == 20
    assert tp.smallest_m(rows, 0.999) is None


def test_analytic_m_is_none_when_target_is_not_ahead():
    assert tp.analytic_m(np.array([-1.0, -2.0, -1.5, -1.0])) is None
    assert tp.analytic_m(np.array([1.0, 2.0, 1.5, 1.2])) > 0


def test_full_run_on_the_miniature_pool(tmp_path):
    make_workdir(tmp_path, n_q=12, seed=3)
    cfg = load_config(student="qwen1_5b")
    # ép llama70b hơn hẳn: tăng grape lên 3 cho mọi chuỗi của nó
    from src.common.io_utils import read_jsonl
    rows = read_jsonl(tmp_path / "fit.qwen1_5b_base.jsonl")
    for r in rows:
        r["grape"] = float(r["grape"]) + (3.0 if "|llama70b|" in r["tid"] else 0.0)
    write_jsonl(tmp_path / "fit.qwen1_5b_base.jsonl", rows)
    res = tp.run(cfg, tmp_path, [3, 6], [1, 3], 200, 0, include_wrong=False, stratify=False)
    assert res["teachers"] == [t["key"] for t in cfg["teachers"]]
    for st in res["settings"]:
        assert st["target"] == "llama70b" and st["rows"][-1]["p"] >= st["rows"][0]["p"] - 0.15
    md = tp.markdown(res, 12)
    assert "llama70b" in md and "m*" in md
    assert tp.main(["--workdir", str(tmp_path), "--outdir", str(tmp_path / "out"), "--sizes", "3", "--chains", "3",
                    "--reps", "50"]) == 0
    assert (tmp_path / "out" / "teacher_pilot.correct.json").exists()


def test_include_wrong_needs_its_file_and_uses_it_when_present(tmp_path):
    make_workdir(tmp_path, n_q=6, seed=4)
    cfg = load_config(student="qwen1_5b")
    with pytest.raises(SystemExit):
        tp.run(cfg, tmp_path, [3], [3], 20, 0, include_wrong=True, stratify=False)
    write_jsonl(tmp_path / "fit.qwen1_5b_base.wrong.jsonl",
                [{"tid": f"math_00000|{t}|{s}", "grape": -9.0} for t in TEACHERS for s in range(3)])
    fam, tids = tp.load_fam(tmp_path, "fit.qwen1_5b_base.jsonl", True)
    assert len(tids) > len(set(t for t in tids if t in fam and fam[t] != -9.0)) - 1 and fam["math_00000|llama70b|2"] == -9.0

"""Kiểm thử FQ-Select (quy tắc rank_sum) và phép đối chứng quality_within_teacher (thêm 20/10/2026)."""
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


def _cfg(method, overrides=None, student="qwen1_5b", ablation=None):
    return load_config(method=method, ablation=ablation, student=student, overrides=overrides)


def _sets(inp, cfg):
    qs = inp.questions(cfg.select.pool, cfg.select.rule)
    picks = b2.select_all(qs, cfg)
    return qs, picks, {q: frozenset(qs[q]["tids"][i] for i in p["idx"]) for q, p in picks.items()}


@pytest.fixture
def inp(tmp_path):
    cfg = make_workdir(tmp_path, n_q=40)
    return b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")


# ---------------------------------------------------------------- thứ hạng chuẩn hoá
def test_norm_ranks_maps_worst_to_zero_best_to_one():
    assert b2.norm_ranks(np.array([3.0, 1.0, 2.0]), "max").tolist() == [1.0, 0.0, 0.5]
    assert b2.norm_ranks(np.array([3.0, 1.0, 2.0]), "min").tolist() == [0.0, 1.0, 0.5]
    assert b2.norm_ranks(np.array([-0.4, -1.2, -0.9, -0.1, -2.0]), "max").tolist() == [0.75, 0.25, 0.5, 1.0, 0.0]
    assert b2.norm_ranks(np.array([7.0]), "max").tolist() == [1.0]


def test_norm_ranks_gives_ties_the_average_rank():
    # hạng 0 và 1 hoà -> 0,5 mỗi chuỗi; chia cho n − 1 = 3
    assert b2.norm_ranks(np.array([1.0, 1.0, 2.0, 3.0]), "max").tolist() == [0.5 / 3, 0.5 / 3, 2 / 3, 1.0]
    assert b2.norm_ranks(np.array([5.0, 5.0, 5.0]), "max").tolist() == [0.5, 0.5, 0.5]
    with pytest.raises(SystemExit):
        b2.norm_ranks(np.array([1.0, np.nan]), "max")


def test_rank_sum_breaks_mathematically_equal_sums_by_the_fixed_order_not_by_float_noise():
    """n = 6 nên hạng chia cho 5: 0,5·(1/5) + 0,5·(4/5) và 0,5·(2/5) + 0,5·(3/5) có thể lệch một ulp. Sau khi làm tròn
    thì hai tổng bằng nhau và chuỗi đứng trước trong thứ tự cố định thắng."""
    q = {"qid": "x", "signals": {"grape": np.array([5.0, 4.0, 1.0, 2.0, 3.0, 0.0])},
         "qual": np.array([4.0, 5.0, 3.0, 2.0, 1.0, 0.0])}
    # tổng hạng: chuỗi 0: 5+4, chuỗi 1: 4+5 (hoà ở đầu), chuỗi 2: 1+3, chuỗi 3: 2+2, chuỗi 4: 3+1 (ba chuỗi hoà = 4)
    p = b2.pick_rank_sum(q, 3, ["grape", "qual"], [0.5, 0.5], ["max", "max"])
    assert p["idx"] == [0, 1, 2] and p["tie"] is True
    q["signals"]["grape"] = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    q["qual"] = np.array([4.0, 5.0, 1.0, 2.0, 3.0, 0.0])
    assert b2.pick_rank_sum(q, 3, ["grape", "qual"], [0.5, 0.5], ["max", "max"])["idx"] == [0, 1, 2]


def test_rank_spec_checks_the_config():
    cfg = _cfg("fq_select")
    assert b2.rank_spec(cfg) == (["grape", "qual"], [0.5, 0.5], ["max", "max"])
    for bad in (["select.rank_weights=[0.6, 0.6]"], ["select.rank_weights=[0.5]"], ["select.signals=[grape, grape]"],
                ["select.rank_weights=[1.5, -0.5]"]):
        with pytest.raises(SystemExit):
            b2.rank_spec(_cfg("fq_select", bad))
    with pytest.raises(KeyError):                      # tín hiệu chưa khai báo chiều tối ưu
        b2.rank_spec(_cfg("fq_select", ["select.signals=[grape, perplexity]"]))


# ---------------------------------------------------------------- FQ-Select trên kho giả
def test_config_and_tag():
    cfg = _cfg("fq_select")
    assert b2.selection_tag(cfg) == "fq_select.k3"
    assert cfg.select.rule == "rank_sum" and cfg.select.pool == "correct" and not cfg.select.get("control")
    base = load_config(student="qwen1_5b")
    assert ("fq_select", None) in b2.method_list(base, Path(base.root))


def test_weight_one_on_grape_reproduces_grape_and_weight_one_on_qual_reproduces_quality_only(inp):
    _, _, grape = _sets(inp, _cfg("grape"))
    _, _, quality = _sets(inp, _cfg("qd_rsr", ablation="quality_only"))
    _, _, fq_g = _sets(inp, _cfg("fq_select", ["select.rank_weights=[1.0, 0.0]"]))
    qs, _, fq_q = _sets(inp, _cfg("fq_select", ["select.rank_weights=[0.0, 1.0]"]))
    assert fq_g == grape
    # Quality-only duyệt tập con và phá hoà theo thứ tự tập con (cộng dấu phẩy động), FQ-Select phá hoà theo thứ tự
    # cố định của chuỗi; nên chỉ so ở những câu không có hai chuỗi cùng Qual ngay ranh giới thứ k.
    clean = [q for q in qs if len(qs[q]["qual"]) == 3 or sorted(qs[q]["qual"])[-3] != sorted(qs[q]["qual"])[-4]]
    assert len(clean) >= 35
    assert all(fq_q[q] == quality[q] for q in clean)


def test_fq_select_keeps_the_k_highest_rank_sums_and_sits_between_its_two_signals(inp):
    qs, picks, fq = _sets(inp, _cfg("fq_select"))
    _, _, grape = _sets(inp, _cfg("grape"))
    _, _, quality = _sets(inp, _cfg("qd_rsr", ablation="quality_only"))
    for qid, p in picks.items():
        q = qs[qid]
        score = np.round(0.5 * b2.norm_ranks(q["signals"]["grape"], "max") + 0.5 * b2.norm_ranks(q["qual"], "max"), 9)
        rest = [i for i in range(len(score)) if i not in p["idx"]]
        assert len(p["idx"]) == 3
        if rest:
            assert min(score[p["idx"]]) >= max(score[rest])
    assert fq != grape and fq != quality                    # thật sự trộn hai tín hiệu trên kho giả
    assert any(fq[q] == grape[q] for q in fq) and any(fq[q] == quality[q] for q in fq)


def test_questions_with_exactly_k_correct_chains_take_them_all(tmp_path):
    cfg = make_workdir(tmp_path, n_q=6, n_right={0: 3, 1: 3, 2: 9, 3: 5, 4: 4, 5: 3})
    inp = b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")
    qs, picks, _ = _sets(inp, _cfg("fq_select"))
    for qid, q in qs.items():
        if len(q["tids"]) == 3:
            assert picks[qid]["idx"] == [0, 1, 2]


def test_end_to_end_and_summary_fields(tmp_path, capsys):
    make_workdir(tmp_path, n_q=30)
    out = tmp_path / "out"
    assert b2.main(["--workdir", str(tmp_path), "--outdir", str(out), "--method", "fq_select"]) == 0
    s = read_json(out / "select.fq_select.k3.json")
    assert s["rule"] == "rank_sum" and s["signals"] == ["grape", "qual"] and s["rank_weights"] == [0.5, 0.5]
    assert s["samples"] == 3 * s["questions"] and s["wrong_kept"] == 0
    assert 0.5 < s["mean_norm_rank"]["grape"] <= 1 and 0.5 < s["mean_norm_rank"]["qual"] <= 1
    assert "fq_select.k3" in capsys.readouterr().out
    rows = read_jsonl(out / "train.fq_select.k3.jsonl")
    assert all(r["weight"] == 1.0 for r in rows)


def test_works_for_the_7b_fit_file_without_local_nat(tmp_path):
    cfg = make_workdir(tmp_path, n_q=10, with_local=False, localnat=False,
                       model=load_config(student="qwen7b").student.base_model_id)
    (tmp_path / "fit.qwen1_5b_base.jsonl").rename(tmp_path / "fit.qwen7b_base.jsonl")
    out = tmp_path / "out"
    assert b2.main(["--workdir", str(tmp_path), "--outdir", str(out), "--student", "qwen7b",
                    "--method", "fq_select"]) == 0
    assert read_json(out / "select.fq_select.k3.json")["student"] == "Qwen/Qwen2.5-7B"


# ---------------------------------------------------------------- đối chứng chất lượng trong từng mô hình dạy
def test_quality_within_teacher_unit():
    teachers = ["deepseek", "qwen72b", "deepseek", "llama70b", "qwen72b", "deepseek"]
    qual = np.array([0.1, 0.2, 0.9, 1.0, 0.8, 0.5])
    # lần bốc của Correct-Only: ba chuỗi đầu = deepseek 2 chỗ, qwen72b 1 chỗ
    p = b2.pick_quality_within_teacher(teachers, qual, 3)
    assert p["idx"] == [2, 4, 5]                      # deepseek: 0,9 và 0,5; qwen72b: 0,8; llama không có chỗ
    assert Counter(teachers[i] for i in p["idx"]) == Counter(teachers[:3])
    tie = b2.pick_quality_within_teacher(["a", "a", "a", "b"], np.array([0.3, 0.7, 0.7, 0.0]), 1)
    assert tie["idx"] == [1] and tie["tie"] is True   # hoà thì chuỗi đứng trước trong thứ tự cố định


def test_control_config_tag_and_kept_out_of_all():
    cfg = _cfg("quality_within_teacher")
    assert b2.selection_tag(cfg) == "quality_within_teacher.k3" and cfg.select.control is True
    base = load_config(student="qwen1_5b")
    assert "quality_within_teacher" not in {m for m, _a in b2.method_list(base, Path(base.root))}


def test_same_teacher_mix_as_correct_only_but_best_quality_inside_each_teacher(inp):
    qs, picks, ctl = _sets(inp, _cfg("quality_within_teacher"))
    _, _, co = _sets(inp, _cfg("correct_only"))
    changed = 0
    for qid, p in picks.items():
        q = qs[qid]
        tid_teacher = dict(zip(q["tids"], q["teacher"]))
        assert Counter(tid_teacher[t] for t in ctl[qid]) == Counter(tid_teacher[t] for t in co[qid])
        for t in set(q["teacher"]):
            mine = [i for i, x in enumerate(q["teacher"]) if x == t]
            got = [i for i in p["idx"] if q["teacher"][i] == t]
            rest = [i for i in mine if i not in got]
            if got and rest:
                assert min(q["qual"][got]) >= max(q["qual"][rest])
        changed += ctl[qid] != co[qid]
        assert sum(q["qual"][list(p["idx"])]) >= sum(q["qual"][i] for i in range(3)) - 1e-12
    assert changed > 0


def test_control_end_to_end_matches_correct_only_teacher_shares(tmp_path):
    make_workdir(tmp_path, n_q=40)
    out = tmp_path / "out"
    for m in ("correct_only", "quality_within_teacher"):
        assert b2.main(["--workdir", str(tmp_path), "--outdir", str(out), "--method", m]) == 0
    a, b = read_json(out / "select.correct_only.k3.json"), read_json(out / "select.quality_within_teacher.k3.json")
    assert a["teacher_share"] == b["teacher_share"] and a["questions"] == b["questions"]
    assert b["control"] is True and b["rule"] == "quality_within_teacher"
    assert b["mean_qual"] > a["mean_qual"]
    assert "signals" not in b and "rank_weights" not in b      # chỉ quy tắc rank_sum mới ghi hai khoá này

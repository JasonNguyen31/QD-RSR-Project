"""Kiểm thử compare_localnat: công cụ so LocalNat ở từng k với GRAPE để chốt k. Dữ liệu giả, đúng tên khoá."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_b"))

from src.common.io_utils import read_jsonl, write_jsonl
from src.tools import compare_localnat as cl
from test_b2_select import make_workdir


def build(tmp, fallback=()):
    """local_nat_k4 chép đúng GRAPE (kịch bản xấu: LocalNat chỉ là GRAPE dưới tên khác); local_nat_k1 xếp NGƯỢC
    GRAPE. Chuỗi có số thứ tự trong fallback mang nguồn bước cắt dự phòng."""
    make_workdir(tmp, n_q=8)
    fit = read_jsonl(tmp / "fit.qwen1_5b_base.jsonl")
    rows = [{"tid": f["tid"], "qid": f["qid"], "local_nat_k1": -f["grape"], "local_nat_k4": f["grape"],
             "steps_source": "fallback_paragraph" if i in fallback else "glm"} for i, f in enumerate(fit)]
    write_jsonl(tmp / "localnat.qwen1_5b_base.glm.jsonl", rows)
    return fit, rows


def test_merge_keeps_shared_chains_and_orders_the_k_columns(tmp_path):
    fit, rows = build(tmp_path)
    merged, cols = cl.merge(fit, rows[:-3] + [{"tid": "ghost|t|0", "qid": "ghost", "local_nat_k1": 0, "local_nat_k4": 0}])
    assert cols == ["local_nat_k1", "local_nat_k4"] and len(merged) == len(fit) - 3       # chuỗi không có trong file fit bị bỏ
    assert all("lark" in r and cl.OLD_LABEL in r and "grape" in r for r in merged)        # ĝ tính lại trên nhóm đã ghép


def test_a_copy_of_grape_is_exposed_and_a_reversed_ranking_is_not(tmp_path):
    fit, rows = build(tmp_path)
    merged, cols = cl.merge(fit, rows)
    groups = cl.by_question(merged, 3, 42)
    m = cl.correlations(groups, cols + ["grape", "rsr"])
    assert m[("local_nat_k4", "grape")] == pytest.approx(1.0) and m[("local_nat_k1", "grape")] == pytest.approx(-1.0)
    same = cl.topk_overlap(groups, "local_nat_k4", "grape", 3)
    assert same["same_set"] == 1.0 and same["sample_share"] == 1.0 and same["questions"] > 0
    assert cl.topk_overlap(groups, "local_nat_k1", "grape", 3)["same_set"] < 1.0
    rsr_self = cl.topk_overlap(groups, "rsr", "rsr", 3)                                    # RSR được đảo dấu nhất quán
    assert rsr_self["same_set"] == 1.0


def test_questions_with_too_few_chains_are_left_out():
    rows = [{"tid": f"q{q}|t|{i}", "qid": f"q{q}", "x": i} for q, n in enumerate([2, 3, 5]) for i in range(n)]
    assert sorted(len(v) for v in cl.by_question(rows, 3, 42).values()) == [3, 5]


def test_fallback_rank_measures_where_rule_cut_chains_land():
    groups = {"q": [{"steps_source": "glm", "s": 1.0}, {"steps_source": "glm", "s": 2.0},
                    {"steps_source": "fallback_line", "s": 3.0}, {"steps_source": "glm_salvaged", "s": 9.0}]}
    r = cl.fallback_rank(groups, "s")
    assert r == {"n": 1, "mean": pytest.approx(2 / 3)}                                     # bản cứu coi như GLM
    assert cl.fallback_rank({"q": groups["q"][:2]}, "s") == {"n": 0, "mean": None}


def test_cli_prints_the_three_sections(tmp_path, capsys):
    build(tmp_path, fallback={0, 1})
    assert cl.main(["--workdir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for piece in ("1. Tương quan hạng", "2. Tập top-3", "3. Chuỗi không do GLM cắt", "fallback_paragraph 2",
                  "local_nat_k1 so với local_nat_k4", "100.0% /   100.0%"):
        assert piece in out, piece
    (tmp_path / "localnat.qwen1_5b_base.glm.jsonl").unlink()
    with pytest.raises(SystemExit, match="Thiếu dữ liệu"):
        cl.main(["--workdir", str(tmp_path)])

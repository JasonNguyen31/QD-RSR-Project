"""Kiểm thử hàm mục tiêu và lambda_scan. So với cách tính độc lập viết bằng vòng lặp thường."""
from itertools import combinations
import math

import numpy as np
import pytest

from src.stage_b.objective import (base_of_subsets, choose, distance_matrix, div_of_subsets, div_scale,
                                   fit_from_rsr, minmax, subsets)
from src.tools.lambda_scan import build_questions, scan, suggest_grid


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def div_by_hand(vecs, s):
    """Div(S) = Σ_t min_{u≠t} ‖e_t − e_u‖, tính thẳng theo định nghĩa."""
    return sum(min(math.dist(vecs[t], vecs[u]) for u in s if u != t) for t in s)


def test_fit_inverts_rsr_so_lowest_rsr_is_best():
    f = fit_from_rsr([3.0, 5.0, 4.0])
    assert f.tolist() == pytest.approx([1.0, 0.0, 0.5])


def test_minmax_constant_group_gives_half():
    assert minmax([2.0, 2.0]).tolist() == [0.5, 0.5]


def test_distance_matrix_matches_euclid():
    rng = np.random.default_rng(0)
    v = np.array([unit(x) for x in rng.normal(size=(5, 8))])
    d = distance_matrix(v)
    for i in range(5):
        for j in range(5):
            assert d[i, j] == pytest.approx(math.dist(v[i], v[j]), abs=1e-9)


def test_subsets_count_is_84_for_nine_choose_three():
    assert len(subsets(9, 3)) == 84 and len(subsets(3, 3)) == 1


def test_div_matches_definition():
    rng = np.random.default_rng(1)
    v = np.array([unit(x) for x in rng.normal(size=(6, 4))])
    sets = subsets(6, 3)
    got = div_of_subsets(distance_matrix(v), sets)
    for s, g in zip(sets, got):
        assert g == pytest.approx(div_by_hand(v, s), abs=1e-9)


def test_div_is_zero_for_duplicates_and_positive_for_spread():
    v = np.array([unit([1, 0]), unit([1, 0]), unit([1, 0]), unit([0, 1])])
    d = distance_matrix(v)
    div = div_of_subsets(d, np.array([[0, 1, 2], [0, 1, 3]]))
    assert div[0] == pytest.approx(0.0) and div[1] > 0


def test_div_scale_subsets_maps_to_unit_interval():
    s = div_scale(np.array([1.0, 3.0, 2.0]), "subsets")
    assert s.tolist() == pytest.approx([0.0, 1.0, 0.5])
    with pytest.raises(ValueError):
        div_scale(np.array([1.0]), "khac")


def test_base_uses_exponents_and_zero_turns_a_signal_off():
    fit, qual = np.array([1.0, 0.5, 0.0]), np.array([0.2, 0.4, 0.8])
    sets = np.array([[0, 1, 2]])
    assert base_of_subsets(fit, qual, sets)[0] == pytest.approx(0.2 + 0.2 + 0.0)
    assert base_of_subsets(fit, qual, sets, a=0.0)[0] == pytest.approx(0.2 + 0.4 + 0.8)   # chỉ Qual


def test_choose_prefers_base_at_zero_and_div_at_large_lambda():
    base, div = np.array([2.0, 1.0]), np.array([0.0, 1.0])
    assert choose(base, div, 0.0) == 0 and choose(base, div, 10.0) == 1


def _toy_questions():
    """Ba câu, mỗi câu 5 ứng viên. Hai ứng viên đầu có Fit·Qual cao nhưng biểu diễn trùng nhau."""
    rows, qual, tids, vecs = [], {}, [], []
    rng = np.random.default_rng(3)
    for q in range(3):
        base = unit(rng.normal(size=6))
        for i in range(5):
            t = f"q{q}|m|{i}"
            rows.append({"tid": t, "rsr": 3.0 + i})           # ứng viên 0 phù hợp nhất
            qual[t] = 1.0 - 0.2 * i
            tids.append(t)
            vecs.append(base if i < 2 else unit(rng.normal(size=6)))
    return rows, qual, tids, np.array(vecs)


def test_build_questions_groups_by_question_and_reports_missing():
    rows, qual, tids, vecs = _toy_questions()
    qual["q0|m|4"] = None
    qs, miss = build_questions(rows, qual, tids[:-1], vecs[:-1])          # bỏ biểu diễn chuỗi cuối
    assert miss == {"qual": 1, "embed": 1}
    assert len(qs["q0"]["tids"]) == 4 and len(qs["q2"]["tids"]) == 4


def test_scan_changes_more_sets_as_lambda_grows():
    rows, qual, tids, vecs = _toy_questions()
    qs, _ = build_questions(rows, qual, tids, vecs)
    r = scan(qs, [0.0, 0.1, 1.0, 100.0], k=3, how="subsets")
    ch = [x["changed"] for x in r["rows"]]
    assert ch[0] == 0.0 and ch == sorted(ch) and ch[-1] > 0
    assert r["rows"][-1]["div_only"] == pytest.approx(1.0)       # λ rất lớn: Div lấn át hoàn toàn


def test_scan_skips_questions_with_only_k_candidates():
    rows, qual, tids, vecs = _toy_questions()
    keep = [r for r in rows if r["tid"].split("|")[2] in {"0", "1", "2"}]
    qs, _ = build_questions(keep, qual, tids, vecs)
    with pytest.raises(ValueError):
        scan(qs, [0.0], k=3, how="subsets")


def test_suggest_grid_picks_first_lambda_reaching_each_level():
    rows = [{"lam": 0.0, "changed": 0.0}, {"lam": 0.1, "changed": 0.05}, {"lam": 0.2, "changed": 0.12},
            {"lam": 0.5, "changed": 0.30}, {"lam": 1.0, "changed": 0.45}]
    assert suggest_grid(rows) == [0.0, 0.2, 0.5]                  # mức 50% không đạt thì bỏ

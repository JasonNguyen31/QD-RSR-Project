import numpy as np
import pytest

from src.stage_a.a5_embed import l2_normalize, pairwise_distances, probe


def test_l2_normalize_handles_zero_vector():
    m = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(m[0], [0.6, 0.8]) and np.allclose(m[1], [0.0, 0.0])


def test_pairwise_distances_count_and_values():
    v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    d = pairwise_distances(v)
    # thứ tự cặp: (0,1), (0,2), (1,2)
    assert len(d) == 3
    assert d[0] == pytest.approx(2 ** 0.5) and d[1] == pytest.approx(0.0) and d[2] == pytest.approx(2 ** 0.5)


def _vecs(spec):
    """spec: tid -> vector 2 chiều."""
    tids = list(spec)
    return tids, l2_normalize(np.array([spec[t] for t in tids], dtype=np.float32))


def test_probe_separates_within_and_across_teacher():
    # cùng mô hình dạy thì gần nhau, khác mô hình dạy thì xa nhau
    tids, v = _vecs({
        "q1|A|0": [1.0, 0.0], "q1|A|1": [1.0, 0.02],
        "q1|B|0": [0.0, 1.0], "q1|B|1": [0.02, 1.0],
    })
    p = probe(tids, v)
    assert p["questions"] == 1
    assert p["within_teacher"]["n"] == 2 and p["across_teacher"]["n"] == 4
    assert p["within_question"]["n"] == 6
    assert p["across_teacher"]["median"] > p["within_teacher"]["median"] * 10


def test_probe_flags_collapsed_distribution():
    """Mọi chuỗi giống hệt nhau: mọi khoảng cách bằng 0, độ trải bằng 0."""
    tids, v = _vecs({f"q1|A|{i}": [1.0, 0.0] for i in range(3)})
    p = probe(tids, v)
    assert p["within_question"]["max"] == 0.0
    assert p["within_question"]["p75"] - p["within_question"]["p25"] == 0.0


def test_probe_skips_single_trajectory_questions():
    tids, v = _vecs({"q1|A|0": [1.0, 0.0], "q2|A|0": [0.0, 1.0]})
    p = probe(tids, v)
    assert p["questions"] == 2 and p["within_question"]["n"] == 0

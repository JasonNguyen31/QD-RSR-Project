import pytest

from src.tools.compare_fit import compare_two, oriented, signal_matrix


def row(tid, rsr=5.0, grape=-1.5, local_nat=-1.6, lark=0.5, n_tokens=100):
    return {"tid": tid, "rsr": rsr, "grape": grape, "local_nat": local_nat,
            "lark": lark, "n_tokens": n_tokens}


def test_rsr_is_inverted_others_are_not():
    """RSR là cực tiểu, nên khi so cùng hướng phải đảo dấu, nếu không kết luận sẽ ngược."""
    r = row("q1|A|0", rsr=5.0, grape=-1.5)
    assert oriented(r, "rsr") == -5.0
    assert oriented(r, "grape") == -1.5


def test_two_students_that_rank_identically():
    a = [row(f"q1|A|{i}", rsr=5.0 + i) for i in range(3)]
    b = [row(f"q1|A|{i}", rsr=10.0 + 2 * i) for i in range(3)]      # khác thang, cùng thứ tự
    r = compare_two(a, b, "rsr")
    assert r["spearman"] == pytest.approx(1.0)
    assert r["pairs"]["rate"] == pytest.approx(1.0)
    assert r["top1"]["rate"] == pytest.approx(1.0)


def test_two_students_that_rank_oppositely():
    a = [row(f"q1|A|{i}", rsr=5.0 + i) for i in range(3)]
    b = [row(f"q1|A|{i}", rsr=9.0 - i) for i in range(3)]
    r = compare_two(a, b, "rsr")
    assert r["spearman"] == pytest.approx(-1.0)
    assert r["pairs"]["rate"] == pytest.approx(0.0)


def test_compare_needs_enough_shared_chains():
    with pytest.raises(SystemExit):
        compare_two([row("q1|A|0")], [row("q1|A|0")], "rsr")


def test_signal_matrix_detects_signals_measuring_the_same_thing():
    """Hai tín hiệu xếp hạng giống nhau cho tương quan 1, ngược nhau cho -1."""
    rows = []
    for i in range(4):
        rows.append(row(f"q1|A|{i}", rsr=5.0 + i, grape=-1.0 - i, lark=0.1 * i))
    m = signal_matrix(rows, ["rsr", "grape", "lark"])
    assert m[("rsr", "grape")] == pytest.approx(1.0)      # RSR tăng, grape giảm; sau khi đảo dấu là cùng chiều
    assert m[("rsr", "lark")] == pytest.approx(-1.0)


def test_signal_matrix_skips_questions_with_too_few_candidates():
    rows = [row("q1|A|0"), row("q1|A|1")]                  # chỉ 2 ứng viên, dưới ngưỡng 3
    assert signal_matrix(rows, ["rsr", "grape"]) == {}

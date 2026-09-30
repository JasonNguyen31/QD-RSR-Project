"""Ba phương án của mô hình 7 tỷ và việc bỏ LocalNat (chốt 30/09/2026)."""
from pathlib import Path

import pytest

from src.common.config import load_config


def test_seven_b_runs_three_methods_one_seed_without_localnat():
    cfg = load_config(student="qwen7b")
    assert cfg.methods == ["qd_rsr", "rsr", "correct_only"]
    assert cfg.project.train_seeds == [42]
    assert cfg.signals.skip_local_nat is True


def test_main_student_unchanged():
    cfg = load_config(student="qwen1_5b")
    assert cfg.project.train_seeds == [42, 43, 44]
    assert cfg.signals.skip_local_nat is False


def test_selection_seed_does_not_depend_on_student():
    """Correct-Only phải chọn đúng cùng một tập ở cả hai mô hình học."""
    assert (load_config(student="qwen1_5b").selection.random_seed
            == load_config(student="qwen7b").selection.random_seed)


def test_seven_b_methods_have_config_files():
    cfg = load_config(student="qwen7b")
    for m in cfg.methods:
        assert (Path(cfg.root) / "configs" / "method" / f"{m}.yaml").exists()


def _stub_b1(monkeypatch):
    """score_one với phần GPU thay bằng số dựng tay, để kiểm tra riêng việc có hay không có LocalNat."""
    from src.stage_b import b1_fit
    calls = {"local": 0}
    monkeypatch.setattr(b1_fit, "build_inputs", lambda tok, q, t, n: ([1, 2, 3], 1, [1], [2, 3], [(0, 2)]))
    monkeypatch.setattr(b1_fit, "token_stats", lambda m, ids, n, c: {
        "ranks": [0, 1], "surprisals": [0.5, 1.0], "sum_sq": [0.5, 0.4], "target_probs": [0.6, 0.4]})

    def fake_local(*_a, **_k):
        calls["local"] += 1
        return -1.2
    monkeypatch.setattr(b1_fit, "local_naturalness", fake_local)
    return b1_fit, calls


def test_score_one_omits_local_nat_field_when_skipped(monkeypatch):
    b1_fit, calls = _stub_b1(monkeypatch)
    v = b1_fit.score_one(None, None, "q", "t", 3072, 100, 4, 512, with_local=False)
    assert "local_nat" not in v and calls["local"] == 0
    assert {"rsr", "grape", "brier"} <= set(v)


def test_score_one_keeps_local_nat_by_default(monkeypatch):
    b1_fit, calls = _stub_b1(monkeypatch)
    v = b1_fit.score_one(None, None, "q", "t", 3072, 100, 4, 512)
    assert v["local_nat"] == -1.2 and calls["local"] == 1


def test_resume_refuses_to_mix_rows_with_and_without_local_nat():
    from src.stage_b.b1_fit import check_local_consistency
    check_local_consistency(None, False, "f")
    check_local_consistency({"local_nat": -1.0}, True, "f")
    check_local_consistency({"rsr": 3.0}, False, "f")
    with pytest.raises(SystemExit):
        check_local_consistency({"local_nat": -1.0}, False, "f")
    with pytest.raises(SystemExit):
        check_local_consistency({"rsr": 3.0}, True, "f")


def test_matrix_drops_local_nat_when_missing():
    from src.tools.compare_fit import SIGNALS
    row = {"tid": "q|t|0", "rsr": 3.0, "grape": -1.0, "lark": 0.4, "n_tokens": 10}
    names = [n for n in SIGNALS if n in row]
    assert "local_nat" not in names and "rsr" in names

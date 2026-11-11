"""Kiểm thử ô Random@GRAPE (thêm 10/11, theo phản biện 07/11): tỷ lệ mô hình dạy của GRAPE, rút ngẫu nhiên trong mỗi mô hình dạy."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.config import load_config
from src.common.io_utils import read_json, read_jsonl
from src.stage_b import b2_select as b2
from test_b2_select import make_workdir

TEACHERS = ["deepseek", "qwen72b", "llama70b", "qwen72b", "deepseek", "llama70b", "qwen72b"]
FAM = np.array([-0.9, -0.5, -0.8, -0.2, -0.95, -0.85, -0.1])           # top-3 theo Fam: chỉ số 6, 3, 1 (cả ba là Qwen)


def test_quota_comes_from_the_signal_but_the_pick_follows_the_fixed_order():
    grape = b2.pick_topk(FAM, 3, "max")["idx"]
    assert grape == [1, 3, 6]
    p = b2.pick_random_at_signal_shares(TEACHERS, FAM, 3, "max")
    assert p["idx"] == [1, 3, 6]                                           # ba chỗ Qwen, Qwen chỉ có ba chuỗi: trùng GRAPE


def test_within_a_teacher_the_choice_ignores_the_signal():
    fam = np.array([-0.9, -0.5, -0.3, -0.2, -0.95, -0.85, -0.1])       # top-3: 6, 3, 2 → 2 chỗ Qwen, 1 chỗ Llama
    p = b2.pick_random_at_signal_shares(TEACHERS, fam, 3, "max")
    assert Counter(TEACHERS[i] for i in p["idx"]) == Counter({"qwen72b": 2, "llama70b": 1})
    assert p["idx"] == [1, 2, 3]                                           # Qwen: hai chuỗi đầu (1, 3), không phải (3, 6)


def test_control_stays_out_of_all():
    base = load_config(student="qwen1_5b")
    assert "random_at_grape" not in {m for m, _a in b2.method_list(base, Path(base.root))}
    cfg = load_config(method="random_at_grape", student="qwen1_5b")
    assert b2.selection_tag(cfg) == "random_at_grape.k3" and cfg.select.control is True


def test_end_to_end_teacher_shares_equal_grape_and_selection_differs(tmp_path):
    make_workdir(tmp_path, n_q=40)
    out = tmp_path / "out"
    for m in ("grape", "random_at_grape"):
        assert b2.main(["--workdir", str(tmp_path), "--outdir", str(out), "--method", m]) == 0
    rows = {m: read_jsonl(out / f"train.{m}.k3.jsonl") for m in ("grape", "random_at_grape")}
    assert len(rows["grape"]) == len(rows["random_at_grape"])
    assert read_json(out / "select.grape.k3.json")["teacher_share"] == \
        read_json(out / "select.random_at_grape.k3.json")["teacher_share"]

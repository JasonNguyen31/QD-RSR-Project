"""Kiểm thử quy tắc chọn theo mô hình dạy (hai phép đối chứng thêm ngày 08/10, không phải phương án)."""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.config import load_config
from src.common.io_utils import read_json, read_jsonl
from src.stage_b import b2_select as b2
from test_b2_select import make_workdir

TEACHERS = ["deepseek", "qwen72b", "llama70b", "qwen72b", "deepseek", "llama70b", "qwen72b"]


def test_preferred_teacher_comes_first_then_the_fixed_order_fills_up():
    p = b2.pick_teacher(TEACHERS, 3, prefer=["qwen72b"])
    assert p == {"idx": [1, 3, 6], "tie": False}                           # đủ ba chuỗi Qwen: lấy cả ba
    p = b2.pick_teacher(TEACHERS[:3], 3, prefer=["qwen72b"])
    assert p["idx"] == [0, 1, 2]                                           # câu chỉ có 3 chuỗi: lấy hết
    p = b2.pick_teacher(["deepseek", "qwen72b", "llama70b", "deepseek"], 3, prefer=["qwen72b"])
    assert p["idx"] == [0, 1, 2]                                           # thiếu Qwen: bù theo thứ tự cố định


def test_avoided_teacher_is_used_only_when_nothing_else_is_left():
    assert b2.pick_teacher(TEACHERS, 3, avoid=["qwen72b"])["idx"] == [0, 2, 4]
    p = b2.pick_teacher(["qwen72b", "deepseek", "qwen72b", "llama70b"], 3, avoid=["qwen72b"])
    assert p["idx"] == [0, 1, 3]                                           # chỉ hai chuỗi ngoài Qwen: bù một chuỗi Qwen đầu tiên


def test_lists_are_checked_against_the_teachers_in_the_config():
    cfg = load_config(method="teacher_qwen_first", student="qwen1_5b")
    assert b2.teacher_lists(cfg) == (["qwen72b"], [])
    for bad in ({"prefer": [], "avoid": []}, {"prefer": ["gpt"]}, {"prefer": ["qwen72b"], "avoid": ["qwen72b"]}):
        broken = {**cfg, "select": {**cfg["select"], "prefer": None, "avoid": None, **bad}}
        with pytest.raises(SystemExit):
            b2.teacher_lists(broken)


def test_controls_stay_out_of_all_and_do_not_depend_on_lambda():
    base = load_config(student="qwen1_5b")
    names = {m for m, _a in b2.method_list(base, Path(base.root))}
    assert "teacher_qwen_first" not in names and "teacher_qwen_last" not in names and "correct_only" in names
    for m in ("teacher_qwen_first", "teacher_qwen_last"):
        assert b2.selection_tag(load_config(method=m, student="qwen1_5b")) == f"{m}.k3"
        assert load_config(method=m, student="qwen1_5b").select.control is True


def test_end_to_end_the_two_controls_pull_the_teacher_share_apart(tmp_path, capsys):
    make_workdir(tmp_path, n_q=40)
    out = tmp_path / "out"
    for m in ("correct_only", "teacher_qwen_first", "teacher_qwen_last"):
        assert b2.main(["--workdir", str(tmp_path), "--outdir", str(out), "--method", m]) == 0
    share = {m: read_json(out / f"select.{m}.k3.json")["teacher_share"].get("qwen72b", 0.0)
             for m in ("correct_only", "teacher_qwen_first", "teacher_qwen_last")}
    assert share["teacher_qwen_last"] < share["correct_only"] < share["teacher_qwen_first"]
    first = read_json(out / "select.teacher_qwen_first.k3.json")
    assert first["rule"] == "teacher" and first["prefer"] == ["qwen72b"] and first["control"] is True
    assert first["questions"] == read_json(out / "select.correct_only.k3.json")["questions"]     # cùng tập câu hỏi
    assert first["samples"] == 3 * first["questions"] and first["wrong_kept"] == 0

    # mỗi câu: số chuỗi Qwen được chọn phải bằng min(3, số chuỗi Qwen đúng có trong kho của câu đó)
    pool = Counter()
    for r in read_jsonl(tmp_path / "fit.qwen1_5b_base.jsonl"):
        pool[(r["qid"], r["tid"].rsplit("|", 2)[1])] += 1
    picked = Counter((r["qid"], r["teacher"]) for r in read_jsonl(out / "train.teacher_qwen_first.k3.jsonl"))
    qids = {q for q, _t in picked}
    assert all(picked[(q, "qwen72b")] == min(3, pool[(q, "qwen72b")]) for q in qids)
    avoided = Counter((r["qid"], r["teacher"]) for r in read_jsonl(out / "train.teacher_qwen_last.k3.jsonl"))
    others = {q: sum(n for (qq, t), n in pool.items() if qq == q and t != "qwen72b") for q in qids}
    assert all(avoided[(q, "qwen72b")] == max(0, 3 - others[q]) for q in qids)
    assert "teacher_qwen_first.k3" in capsys.readouterr().out

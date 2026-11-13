"""Kiểm scripts/pack_hf_generations.py: gộp lời giải với nhãn, bỏ trùng tid giữa các lô, báo lỗi khi thiếu."""
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pack_gen", Path(__file__).resolve().parents[2] / "scripts" / "pack_hf_generations.py")
pg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pg)


def _w(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _traj(tid, qid="q1", teacher="ds", i=0, fr="stop"):
    return {"tid": tid, "qid": qid, "teacher": teacher, "sample_idx": i, "text": f"sol {tid}",
            "finish_reason": fr, "completion_tokens": 10, "latency": 1.0}


def test_rows_join_labels_and_dedup(tmp_path):
    _w(tmp_path / "trajectories.ds.jsonl", [_traj("a"), _traj("b", i=1, fr="length")])
    _w(tmp_path / "trajectories.qw.jsonl", [_traj("c", teacher="qw"), _traj("a")])   # "a" lặp ở lô khác
    _w(tmp_path / "labels.jsonl", [{"tid": "a", "correct": True, "reason": "ok", "pred": "4"},
                                   {"tid": "b", "correct": False, "reason": "truncated", "pred": None},
                                   {"tid": "c", "correct": False, "reason": "wrong", "pred": "5"}])
    rows = pg.build_rows(tmp_path)
    assert [r["tid"] for r in rows] == ["a", "b", "c"]
    assert rows[1]["finish_reason"] == "length" and rows[1]["reason"] == "truncated"
    assert "latency" not in rows[0] and rows[0]["text"] == "sol a" and rows[0]["correct"] is True


def test_missing_label_or_text_is_an_error(tmp_path):
    _w(tmp_path / "trajectories.jsonl", [_traj("a")])
    _w(tmp_path / "labels.jsonl", [{"tid": "a", "correct": True}, {"tid": "z", "correct": False}])
    with pytest.raises(SystemExit, match="không có văn bản"):
        pg.build_rows(tmp_path)
    _w(tmp_path / "labels.jsonl", [])
    with pytest.raises(SystemExit, match="labels.jsonl"):
        pg.build_rows(tmp_path)

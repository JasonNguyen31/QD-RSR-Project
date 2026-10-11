"""Kiểm thử việc b2_select chọn Local Naturalness theo điểm trên bước của GLM (file do b1_localnat ghi)."""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.io_utils import read_json, read_jsonl, write_jsonl
from src.stage_b import b2_select as b2
from src.stage_b import b3_train as b3
from test_b2_select import _cfg, _picked, make_workdir

FIT, LN = "fit.qwen1_5b_base.jsonl", "localnat.qwen1_5b_base.glm.jsonl"


def test_score_file_name_follows_the_fit_file_and_direction_follows_local_nat():
    assert b2.localnat_name(FIT) == LN and b2.localnat_name("fit.qwen7b_base.jsonl") == "localnat.qwen7b_base.glm.jsonl"
    cfg = _cfg("local_naturalness")
    assert cfg.select.signal == "local_nat_k1" and not cfg.select.get("blocked")
    assert b2.direction_of(cfg, "local_nat_k1") == b2.direction_of(cfg, "local_nat_k4") == "max"
    assert b2.direction_of(cfg, "rsr") == "min"                           # các tín hiệu khác không đổi
    with pytest.raises(KeyError):
        b2.direction_of(cfg, "local_nat_kx")


def test_top3_follow_the_chosen_k_and_changing_k_is_one_setting(tmp_path):
    cfg = make_workdir(tmp_path, n_q=20)
    inp = b2.Inputs(cfg, tmp_path, FIT)
    picked = {}
    for k in (1, 4):
        col = f"local_nat_k{k}"
        qs, picked[k], _ = _picked(inp, _cfg("local_naturalness", overrides=[f"select.signal={col}"]))
        for qid, q in qs.items():
            vals = {t: inp.localnat[t][col] for t in q["tids"]}
            assert sorted(vals[t] for t in picked[k][qid]) == sorted(vals.values())[-3:], (k, qid)
    assert picked[1] != picked[4]                                          # hai k cho hai tập khác nhau
    assert _picked(inp, _cfg("local_naturalness"))[1] == picked[1]         # mặc định trong file cấu hình là k = 1


def test_incomplete_or_foreign_or_stale_score_file_is_refused(tmp_path):
    cfg = make_workdir(tmp_path, n_q=8)
    rows = read_jsonl(tmp_path / LN)

    write_jsonl(tmp_path / LN, rows[1:])                                   # thiếu một chuỗi: b1_localnat chưa chấm xong
    inp = b2.Inputs(cfg, tmp_path, FIT)
    with pytest.raises(SystemExit, match="Chưa có điểm 'local_nat_k1'"):
        _picked(inp, _cfg("local_naturalness"))
    assert _picked(inp, _cfg("grape"))[1] and _picked(inp, _cfg())[1]      # phương án khác không cần file này

    write_jsonl(tmp_path / LN, [{k: v for k, v in r.items() if k != "local_nat_k1"} for r in rows])
    with pytest.raises(SystemExit, match="local_nat_k1"):                  # file chấm với bộ k khác
        _picked(b2.Inputs(cfg, tmp_path, FIT), _cfg("local_naturalness"))

    write_jsonl(tmp_path / LN, [{**r, "model": "Qwen/Qwen2.5-7B"} for r in rows])
    with pytest.raises(SystemExit, match="cấu hình đòi"):
        b2.Inputs(cfg, tmp_path, FIT)

    write_jsonl(tmp_path / LN, [{**rows[0], "text_md5": hashlib.md5(b"khac").hexdigest()}] + rows[1:])
    with pytest.raises(SystemExit, match="văn bản khác lúc chấm"):
        b2.Inputs(cfg, tmp_path, FIT)


def test_selection_file_records_where_the_scores_and_steps_came_from(tmp_path):
    work, out = tmp_path / "a", tmp_path / "out"
    work.mkdir()
    make_workdir(work, n_q=30)
    assert b2.main(["--workdir", str(work), "--outdir", str(out), "--method", "local_naturalness"]) == 0
    s = read_json(out / "select.local_naturalness.k3.json")
    assert s["signal"] == "local_nat_k1" and s["blocked"] is None and s["samples"] == 90
    assert s["signal_file"] == LN and s["signal_file_md5"] == hashlib.md5((work / LN).read_bytes()).hexdigest()
    assert s["steps_version"] == [1] and sum(s["steps_source_selected"].values()) == 90
    assert set(s["steps_source_selected"]) <= {"glm", "fallback_paragraph"}
    rows, meta = b3.load_selection(out, "local_naturalness.k3", 3)         # không còn bị khoá: huấn luyện đọc được
    assert len(rows) == 90 and meta["signal"] == "local_nat_k1"

    assert b2.main(["--workdir", str(work), "--outdir", str(out), "--method", "rsr"]) == 0
    assert "signal_file" not in read_json(out / "select.rsr.k3.json")      # phương án khác không mang các khoá này

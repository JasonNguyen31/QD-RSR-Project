"""Kiểm thử b2_select. Dữ liệu giả dựng qua a4.combine (quality.jsonl) và b1_fit.aggregate (file fit), để tên
khoá đúng như file thật: bài kiểm thử dùng dữ liệu giả sai giống mã thì lỗi lọt qua (bẫy hình A-8)."""
import ast
import json
import math
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from src.common.config import load_config
from src.common.io_utils import read_json, read_jsonl, write_jsonl
from src.stage_a.a4_score_quality import combine
from src.stage_b import b2_select as b2
from src.stage_b.b1_fit import aggregate
from src.stage_b.signals import SIGNALS_VERSION, lark_ghat

TEACHERS = ["deepseek", "llama70b", "qwen72b"]
WORDS = "we check since therefore perhaps the sum of two numbers equals verify might value".split()


def _fit_row(tid, qid, model, rng, with_local=True):
    """Một dòng file fit, dựng đúng như b1_fit.main ghi: aggregate() cộng các trường của score_one."""
    n = int(rng.integers(20, 200))
    v = aggregate(rng.integers(0, 300, n).tolist(), rng.uniform(0.1, 4.0, n).tolist(),
                  rng.uniform(0.1, 0.9, n).tolist(), rng.uniform(0.05, 0.95, n).tolist(), 100)
    if with_local:
        v["local_nat"] = float(-rng.uniform(0.5, 3.0))
    v["n_steps"] = int(rng.integers(3, 30))
    v["signals_version"] = SIGNALS_VERSION
    return {"tid": tid, "qid": qid, "model": model, "ts": "2026-10-02T00:00:00+00:00", **v}


def make_workdir(tmp: Path, n_q=12, seed=0, model=None, with_local=True, unjudged=(), n_right=None):
    """Kho thu nhỏ: mỗi câu 9 chuỗi (3 mô hình dạy x 3 mẫu). n_right[q] chuỗi đầu là đúng (mặc định 4 đến 9)."""
    rng = np.random.default_rng(seed)
    cfg = load_config(student="qwen1_5b")
    model = model or cfg.student.base_model_id
    questions, trajs, cands = [], [], []
    for qi in range(n_q):
        qid = f"math_{qi:05d}"
        questions.append({"qid": qid, "source": "math", "level": 4, "question": f"Question {qi}?",
                          "solution": "s", "gold": "1"})
        right = (n_right or {}).get(qi, 4 + qi % 6)
        j = 0
        for t in TEACHERS:
            for s in range(3):
                text = " ".join(rng.choice(WORDS, int(rng.integers(8, 60)))) + f" \\boxed{{{qi}}}"
                row = {"tid": f"{qid}|{t}|{s}", "qid": qid, "teacher": t, "sample_idx": s, "text": text,
                       "finish_reason": "stop", "completion_tokens": len(text.split())}
                trajs.append(row)
                if j < right:
                    cands.append(row)
                j += 1
    judge = [{"tid": c["tid"], "qid": c["qid"], "overall_score": float(rng.choice([0.0, 0.3, 0.6, 0.8, 1.0]))}
             for c in cands if c["tid"] not in set(unjudged)]
    quality = combine(cands, judge, alpha=cfg.quality.alpha)
    fit = [_fit_row(c["tid"], c["qid"], model, rng, with_local) for c in cands]
    vecs = rng.normal(size=(len(cands), 16))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    files = cfg.stage_a_files
    write_jsonl(tmp / files["questions"], questions)
    write_jsonl(tmp / files["trajectories"], trajs)
    write_jsonl(tmp / files["candidates"], cands)
    write_jsonl(tmp / "quality.jsonl", quality)
    write_jsonl(tmp / "fit.qwen1_5b_base.jsonl", fit)
    np.savez_compressed(tmp / "embeddings.npz", tids=np.array([c["tid"] for c in cands]), vectors=vecs)
    return cfg


def _cfg(method=None, ablation=None, overrides=None, student="qwen1_5b"):
    return load_config(method=method or "qd_rsr", ablation=ablation, student=student, overrides=overrides)


def _picked(inp, cfg):
    qs = inp.questions(cfg.select.pool, cfg.select.rule)
    picks = b2.select_all(qs, cfg)
    return qs, {q: {qs[q]["tids"][i] for i in p["idx"]} for q, p in picks.items()}, picks


@pytest.fixture
def inp(tmp_path):
    cfg = make_workdir(tmp_path)
    return b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")


# ---------------------------------------------------------------- tên file, thứ tự cố định
def test_tag_carries_k_and_lambda_only_when_selection_depends_on_it():
    lam = load_config().selection.lambda_div
    assert b2.selection_tag(_cfg("rsr")) == "rsr.k3"
    assert b2.selection_tag(_cfg("qd_rsr")) == f"qd_rsr.k3.lam{lam:g}"
    assert b2.selection_tag(_cfg(overrides=["selection.lambda_div=0.15"])) == "qd_rsr.k3.lam0.15"
    assert b2.selection_tag(_cfg(ablation="fit_quality")) == "fit_quality.k3"
    assert b2.selection_tag(_cfg(ablation="diversity_only")) == "diversity_only.k3"     # mọi λ > 0 như nhau
    assert b2.selection_tag(_cfg(overrides=["selection.k=1"])) == "qd_rsr.k1"           # k = 1 thì Div = 0
    assert b2.selection_tag(_cfg("local_naturalness")) == "local_naturalness.k3.tam"    # đang bị khoá


def test_order_uses_hashlib_never_the_builtin_hash():
    """hash() của chuỗi đổi theo PYTHONHASHSEED ở mỗi lần chạy; dùng nó thì tập chọn không tái lập được."""
    tree = ast.parse(Path(b2.__file__).read_text(encoding="utf-8"))
    calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "hash" not in calls
    import hashlib
    assert b2.order_key(42, "math_00001|deepseek|0") == hashlib.sha256(b"42|math_00001|deepseek|0").hexdigest()


def test_selection_is_identical_under_different_python_hash_seeds(tmp_path):
    """Chạy b2_select ở hai tiến trình với PYTHONHASHSEED khác nhau: mọi file train phải giống từng byte."""
    work = tmp_path / "a"
    work.mkdir()
    make_workdir(work, n_q=25)
    root = Path(b2.__file__).resolve().parents[2]
    digests = []
    for seed in ("1", "987654"):
        out = tmp_path / f"out{seed}"
        for extra in (["--method", "correct_only"], ["--method", "no_filter"], ["--method", "qd_rsr"],
                      ["--ablation", "quality_only"], ["--method", "token_length"]):
            subprocess.run([sys.executable, "-m", "src.stage_b.b2_select", "--workdir", str(work), "--outdir",
                            str(out)] + extra, cwd=root, env={**os.environ, "PYTHONHASHSEED": seed},
                           check=True, capture_output=True)
        digests.append({p.name: read_json(p)["md5"] for p in sorted(out.glob("select.*.json"))})
    assert len(digests[0]) == 5 and digests[0] == digests[1]


def test_order_key_is_stable_and_depends_on_seed():
    assert b2.order_key(42, "q|a|0") == b2.order_key(42, "q|a|0")
    assert b2.order_key(42, "q|a|0") != b2.order_key(43, "q|a|0")


def test_every_config_file_declares_a_valid_rule():
    base = load_config(student="qwen1_5b")
    todo = b2.method_list(base, Path(base.root))
    assert len(todo) == 14                                    # 8 phương án, 6 biến thể
    names = set()
    for m, a in todo:
        cfg = load_config(method=m, ablation=a, student="qwen1_5b")
        assert cfg.select.rule in ("random", "topk", "objective")
        assert cfg.select.pool in ("correct",) + b2.EXTENDED_POOLS
        assert cfg.select.name == (a or m)                    # tên file train khớp tên file cấu hình
        names.add(b2.selection_tag(cfg))
    assert len(names) == 14
    seven = load_config(student="qwen7b")
    assert [m for m, _ in b2.method_list(seven, Path(seven.root))] == seven.methods


def test_diversity_only_does_not_inherit_lambda_zero():
    """Nếu λ chọn được là 0, biến thể Diversity-only vẫn phải tối đa hoá Div chứ không trả tập bất kỳ."""
    cfg = _cfg(ablation="diversity_only", overrides=[])
    assert cfg.selection.lambda_div > 0 and cfg.selection.a == 0 and cfg.selection.b == 0


# ---------------------------------------------------------------- kho ứng viên
def test_pool_drops_unjudged_chains_and_questions_falling_below_three(tmp_path):
    # câu 0 có đúng 4 chuỗi đúng; bỏ điểm giám khảo của 2 chuỗi thì còn 2, câu bị loại
    drop = ["math_00000|deepseek|0", "math_00000|deepseek|1", "math_00003|deepseek|0"]
    cfg = make_workdir(tmp_path, unjudged=drop)
    inp = b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")
    assert inp.info["excluded_no_qual"] == 3
    assert inp.info["dropped_qids"] == ["math_00000"] and inp.info["questions"] == 11
    assert all(t not in drop for q in inp.pool.values() for t in (c["tid"] for c in q))


def test_candidate_missing_from_quality_file_is_an_error_not_a_silent_drop(tmp_path):
    cfg = make_workdir(tmp_path)
    rows = read_jsonl(tmp_path / "quality.jsonl")[1:]
    write_jsonl(tmp_path / "quality.jsonl", rows)
    with pytest.raises(SystemExit, match="quality.jsonl"):
        b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")


def test_usable_chain_without_fit_row_stops_even_for_correct_only(tmp_path):
    cfg = make_workdir(tmp_path)
    write_jsonl(tmp_path / "fit.qwen1_5b_base.jsonl", read_jsonl(tmp_path / "fit.qwen1_5b_base.jsonl")[1:])
    inp = b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")
    with pytest.raises(SystemExit, match="file fit thiếu 1"):
        inp.questions("correct", "random")


def test_inputs_refuse_fit_file_of_another_model_or_old_definition(tmp_path):
    cfg = make_workdir(tmp_path, model="Qwen/Qwen2.5-7B")
    with pytest.raises(SystemExit, match="Sai file fit"):
        b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")
    cfg = make_workdir(tmp_path)
    rows = [{**r, "signals_version": 1} for r in read_jsonl(tmp_path / "fit.qwen1_5b_base.jsonl")]
    write_jsonl(tmp_path / "fit.qwen1_5b_base.jsonl", rows)
    with pytest.raises(SystemExit, match="định nghĩa tín hiệu cũ"):
        b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")


# ---------------------------------------------------------------- chiều tối ưu: chỗ dễ nhầm nhất
def test_rsr_takes_the_lowest_and_every_other_signal_the_highest(inp):
    for method, col, lowest in (("rsr", "rsr", True), ("grape", "grape", False),
                                ("local_naturalness", "local_nat", False), ("token_length", "n_tokens", False)):
        qs, picked, _ = _picked(inp, _cfg(method))
        for qid, q in qs.items():
            vals = {t: inp.fit[t][col] for t in q["tids"]}
            ranked = sorted(vals.values(), reverse=not lowest)[:3]
            assert sorted((vals[t] for t in picked[qid]), reverse=not lowest) == ranked, (method, qid)


def test_fit_only_ablation_selects_exactly_the_rsr_sets(inp):
    assert _picked(inp, _cfg(ablation="fit_only"))[1] == _picked(inp, _cfg("rsr"))[1]


def test_quality_only_takes_highest_qual(inp):
    qs, picked, _ = _picked(inp, _cfg(ablation="quality_only"))
    for qid, q in qs.items():
        qual = {t: inp.quality[t]["qual"] for t in q["tids"]}
        assert sorted(qual[t] for t in picked[qid]) == sorted(qual.values())[-3:]


# ---------------------------------------------------------------- hàm mục tiêu
def _f_by_hand(q, s, a, b, lam):
    """F(S) tính thẳng theo định nghĩa trong paper, không dùng objective.py."""
    rsr = q["signals"]["rsr"]
    lo, hi = rsr.min(), rsr.max()
    fit = [1 - (r - lo) / (hi - lo) for r in rsr]
    base = sum((fit[i] ** a if a else 1.0) * (q["qual"][i] ** b if b else 1.0) for i in s)
    div = sum(min(math.dist(q["vecs"][i], q["vecs"][j]) for j in s if j != i) for i in s)
    return base + lam * div


@pytest.mark.parametrize("lam", [0.0, 0.15, 0.4, 1.0])
def test_objective_returns_the_exact_optimum_over_all_subsets(inp, lam):
    cfg = _cfg(overrides=[f"selection.lambda_div={lam}"])
    qs, _, picks = _picked(inp, cfg)
    for qid, q in qs.items():
        best = max(_f_by_hand(q, s, 1, 1, lam) for s in combinations(range(len(q["tids"])), 3))
        assert _f_by_hand(q, picks[qid]["idx"], 1, 1, lam) == pytest.approx(best, abs=1e-9)


def test_lambda_zero_equals_fit_quality_and_huge_lambda_equals_diversity_only(inp):
    assert _picked(inp, _cfg(overrides=["selection.lambda_div=0"]))[1] == _picked(inp, _cfg(ablation="fit_quality"))[1]
    assert (_picked(inp, _cfg(overrides=["selection.lambda_div=1000"]))[1]
            == _picked(inp, _cfg(ablation="diversity_only"))[1])


def test_k1_objective_is_the_best_fit_times_qual_product(inp):
    qs, _, picks = _picked(inp, _cfg(overrides=["selection.k=1"]))
    for qid, q in qs.items():
        prod = q["fit"] * q["qual"]
        assert prod[picks[qid]["idx"][0]] == pytest.approx(prod.max())


def test_greedy_is_a_side_analysis_and_agrees_with_exact_when_diversity_barely_matters(inp):
    qs, _, picks = _picked(inp, _cfg(overrides=["selection.lambda_div=1e-9"]))
    flags = [p["greedy_same"] for p in picks.values() if "greedy_same" in p and not p["tie"]]
    assert flags and all(flags)                               # λ gần 0: tham lam là top-k theo Fit·Qual
    q = next(iter(qs.values()))
    got = b2.greedy_set(q["fit"] * q["qual"], b2.distance_matrix(q["vecs"]), 3, 0.4)
    assert len(set(got)) == 3 and int(np.argmax(q["fit"] * q["qual"])) in got
    assert "greedy_same" not in next(iter(_picked(inp, _cfg(ablation="fit_quality"))[2].values()))


# ---------------------------------------------------------------- phá hoà và rút ngẫu nhiên
def test_ties_follow_the_hashed_order_not_the_alphabetical_one():
    seed, tids = 42, [f"q|{t}|{s}" for t in TEACHERS for s in range(3)]
    order = sorted(tids, key=lambda t: b2.order_key(seed, t))
    p = b2.pick_topk(np.ones(9), 3, "max")
    assert p["tie"] and p["idx"] == [0, 1, 2]                 # 3 chuỗi đầu của thứ tự đã xếp sẵn
    assert order[:3] != sorted(tids)[:3]                      # và thứ tự đó không phải bảng chữ cái
    assert not b2.pick_topk(np.arange(9.0), 3, "max")["tie"]


def test_random_methods_do_not_depend_on_the_student(tmp_path):
    """Correct-Only phải cho đúng cùng tập ở 1,5B và 7B: đổi toàn bộ file fit, tập chọn không đổi."""
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir(), b_dir.mkdir()
    cfg = make_workdir(a_dir, seed=0)
    make_workdir(b_dir, seed=0)
    rng = np.random.default_rng(99)
    rows = read_jsonl(b_dir / "fit.qwen1_5b_base.jsonl")
    write_jsonl(b_dir / "fit.qwen1_5b_base.jsonl",
                [_fit_row(r["tid"], r["qid"], r["model"], rng) for r in rows])
    ia, ib = (b2.Inputs(cfg, d, "fit.qwen1_5b_base.jsonl") for d in (a_dir, b_dir))
    assert _picked(ia, _cfg("correct_only"))[1] == _picked(ib, _cfg("correct_only"))[1]
    assert _picked(ia, _cfg("rsr"))[1] != _picked(ib, _cfg("rsr"))[1]


def test_no_filter_draws_from_wrong_chains_too_and_keeps_the_common_question_set(tmp_path):
    cfg = make_workdir(tmp_path, n_q=40, unjudged=["math_00001|deepseek|0"])
    traj = read_jsonl(tmp_path / "trajectories.jsonl")
    traj[-1]["text"] = "   "                                  # chuỗi rỗng của câu cuối
    write_jsonl(tmp_path / "trajectories.jsonl", traj)
    inp = b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")
    qs, picked, picks = _picked(inp, _cfg("no_filter"))
    assert set(picked) == set(inp.pool)
    all_tids = {t for s in picked.values() for t in s}
    assert any(t not in inp.correct_tids for t in all_tids)   # có chuỗi sai được rút
    assert "math_00001|deepseek|0" not in {t for q in qs.values() for t in q["tids"]}
    assert traj[-1]["tid"] not in {t for q in qs.values() for t in q["tids"]}
    s = b2.summarize(qs, picks, 3)
    assert s["wrong_kept"] == sum(t not in inp.correct_tids for t in all_tids) and s["mean_div"] is None
    # ghép cặp có chủ đích: câu nào No-Filter không rút trúng chuỗi sai thì trùng hẳn Correct-Only
    correct = _picked(inp, _cfg("correct_only"))[1]
    clean = [q for q, s_ in picked.items() if s_ <= inp.correct_tids]
    assert clean and all(picked[q] == correct[q] for q in clean)


def test_no_prefilter_stops_instead_of_quietly_becoming_qd_rsr(inp):
    with pytest.raises(SystemExit, match="backfill_wrong"):
        _picked(inp, _cfg(ablation="no_prefilter"))


# ---------------------------------------------------------------- LARK
def test_lark_ghat_is_recomputed_on_the_usable_pool_and_weights_average_one(tmp_path):
    n_right = {0: 3, 1: 6}                                    # câu 0 có đúng k ứng viên
    cfg = make_workdir(tmp_path, n_right=n_right, unjudged=["math_00001|deepseek|0"])
    inp = b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")
    qs, picked, picks = _picked(inp, _cfg("lark"))
    q = qs["math_00001"]
    assert len(q["tids"]) == 5                                # chuỗi thiếu Qual không được góp vào ĝ
    _rho, g = lark_ghat([inp.fit[t]["mean_surprisal"] for t in q["tids"]], [inp.fit[t]["brier"] for t in q["tids"]])
    assert q["signals"]["lark"].tolist() == pytest.approx(g)
    assert {q["tids"][i] for i in np.argsort(g)[-3:]} == picked["math_00001"]
    for p in picks.values():
        assert sum(p["weights"]) == pytest.approx(3.0) and min(p["weights"]) >= 0
    assert picks["math_00000"]["weights"] == pytest.approx([1.0, 1.0, 1.0])     # đúng k ứng viên: trọng số đều
    w = picks["math_00001"]["weights"]
    assert max(w) > 1.0 > min(w)                              # còn lại: trọng số mềm thật sự
    assert all(p["weights"] == [1.0, 1.0, 1.0] for p in _picked(inp, _cfg("rsr"))[2].values())


def test_local_naturalness_fails_clearly_on_a_fit_file_without_it(tmp_path):
    cfg = make_workdir(tmp_path, with_local=False)
    inp = b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")
    with pytest.raises(SystemExit, match="local_nat"):
        _picked(inp, _cfg("local_naturalness"))
    assert _picked(inp, _cfg("rsr"))[1]                       # các phương án khác vẫn chạy


# ---------------------------------------------------------------- chạy trọn vẹn
def test_all_writes_one_shared_question_set_with_k_rows_each(tmp_path, capsys):
    work, out = tmp_path / "a", tmp_path / "out"
    work.mkdir()
    make_workdir(work, n_q=30, unjudged=["math_00002|deepseek|1"])
    code = b2.main(["--workdir", str(work), "--outdir", str(out), "--all", "--expect-questions", "30"])
    printed = capsys.readouterr().out
    assert code == 1 and "CHƯA CHẠY ĐƯỢC no_prefilter" in printed          # chưa chấm bù chuỗi sai
    assert "CHƯA CHẠY ĐƯỢC local_naturalness.k3.tam" in printed            # bị khoá cho tới khi làm đúng bài gốc
    files = sorted(out.glob("train.*.jsonl"))
    assert len(files) == 12
    qsets = set()
    for f in files:
        rows = read_jsonl(f)
        by_q = {}
        for r in rows:
            by_q.setdefault(r["qid"], []).append(r["tid"])
            assert r["question"].startswith("Question") and r["text"] and r["weight"] >= 0
        assert all(len(set(v)) == 3 for v in by_q.values()) and len(rows) == 90
        qsets.add(frozenset(by_q))
    assert len(qsets) == 1
    s = read_json(out / "select.rsr.k3.json")
    assert s["questions"] == 30 and s["samples"] == 90 and s["rule"] == "topk" and len(s["md5"]) == 32
    fit_only = {(r["qid"], r["tid"]) for r in read_jsonl(out / "train.fit_only.k3.jsonl")}
    assert fit_only == {(r["qid"], r["tid"]) for r in read_jsonl(out / "train.rsr.k3.jsonl")}

    assert b2.main(["--outdir", str(out), "--report", "--versus", "qwen7b"]) == 0
    table = capsys.readouterr().out
    assert "| Toàn kho ứng viên |" in table and "| rsr.k3 |" in table


def test_blocked_method_needs_an_explicit_flag_and_gets_a_tam_name(tmp_path):
    work, out = tmp_path / "a", tmp_path / "out"
    work.mkdir()
    make_workdir(work)
    with pytest.raises(SystemExit, match="đang bị khoá"):
        b2.main(["--workdir", str(work), "--outdir", str(out), "--method", "local_naturalness"])
    assert not list(out.glob("train.*"))
    assert b2.main(["--workdir", str(work), "--outdir", str(out), "--method", "local_naturalness",
                    "--allow-blocked"]) == 0
    assert [p.name for p in out.glob("train.*")] == ["train.local_naturalness.k3.tam.jsonl"]
    assert read_json(out / "select.local_naturalness.k3.tam.json")["blocked"]
    b2.main(["--workdir", str(work), "--outdir", str(out), "--method", "rsr"])
    assert read_json(out / "select.rsr.k3.json")["blocked"] is None


def test_rerun_is_byte_identical(tmp_path):
    work = tmp_path / "a"
    work.mkdir()
    make_workdir(work)
    md5 = []
    for name in ("o1", "o2"):
        b2.main(["--workdir", str(work), "--outdir", str(tmp_path / name), "--method", "qd_rsr"])
        md5.append(read_json(next((tmp_path / name).glob("select.qd_rsr.*.json")))["md5"])
    assert md5[0] == md5[1]


def test_expect_questions_mismatch_stops(tmp_path):
    make_workdir(tmp_path)
    with pytest.raises(SystemExit, match="mong đợi 99"):
        b2.main(["--workdir", str(tmp_path), "--outdir", str(tmp_path / "o"), "--method", "rsr",
                 "--expect-questions", "99"])


def test_all_refuses_overrides_that_would_unpin_the_ablations(tmp_path):
    for extra in (["--override", "selection.lambda_div=0.15"], ["--lambda-grid"]):
        with pytest.raises(SystemExit):
            b2.main(["--workdir", str(tmp_path), "--all"] + extra)
    with pytest.raises(SystemExit):
        b2.main(["--workdir", str(tmp_path), "--ablation", "fit_only", "--lambda-grid"])


def test_lambda_grid_writes_one_file_per_grid_value(tmp_path):
    work, out = tmp_path / "a", tmp_path / "out"
    work.mkdir()
    cfg = make_workdir(work)
    assert b2.main(["--workdir", str(work), "--outdir", str(out), "--method", "qd_rsr", "--lambda-grid"]) == 0
    tags = {json.loads(p.read_text(encoding="utf-8"))["lambda_div"] for p in out.glob("select.qd_rsr.*.json")}
    assert tags == set(float(x) for x in cfg.selection.lambda_grid)


def test_overlap_counts_shared_samples_and_identical_sets():
    a = {"q1": {"a", "b", "c"}, "q2": {"a", "b", "c"}}
    b = {"q1": {"a", "b", "c"}, "q2": {"a", "b", "d"}}
    o = b2.overlap(a, b)
    assert o["sample_share"] == pytest.approx(5 / 6) and o["same_set"] == 0.5
    assert b2.overlap(a, b, only={"q2"})["same_set"] == 0.0

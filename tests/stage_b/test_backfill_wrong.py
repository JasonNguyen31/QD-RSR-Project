"""Kiểm thử công cụ chấm bù chuỗi sai và việc b2_select đọc kho mở rộng. Không gọi mạng, không cần GPU:
giám khảo, bộ mã hoá và b1_fit được thay bằng bản giả, còn phần ghép Qual dùng đúng a4.combine."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.common.config import load_config
from src.common.io_utils import read_jsonl, write_jsonl
from src.stage_a import a4_score_quality as a4
from src.stage_a import a5_embed as a5
from src.stage_b import b1_fit
from src.stage_b import b2_select as b2
from src.tools import backfill_wrong as bw
from test_b2_select import _cfg, _fit_row, _picked, make_workdir

PROTECTED = ["quality.jsonl", "judge.jsonl", "embeddings.npz", "fit.qwen1_5b_base.jsonl", "candidates.jsonl"]


def _md5s(wd: Path) -> dict:
    return {n: hashlib.md5((wd / n).read_bytes()).hexdigest() for n in PROTECTED}


class FakeJudge:
    """Thay OpenRouterJudge: điểm thấp, mỗi lượt tốn 0,001 đô. Chuỗi có chữ 'UNJUDGEABLE' thì hỏng."""

    def __init__(self):
        self.ok = self.failed = self.truncated = self.parse_retries = 0
        self.seconds, self.cost, self.errors, self.prompts = 0.0, 0.0, {}, []

    def score(self, prompt):
        self.prompts.append(prompt)
        if "UNJUDGEABLE" in prompt:
            self.failed += 1
            raise ValueError("giám khảo lặp vô hạn")
        self.ok += 1
        self.cost += 0.001
        return {"overall_score": 0.2, "overall_reason": "wrong answer"}


@pytest.fixture
def work(tmp_path, monkeypatch):
    """Kho thu nhỏ đã có judge.jsonl của lô trước, và ba bước chấm được thay bằng bản giả."""
    cfg = make_workdir(tmp_path, n_q=20, unjudged=["math_00002|deepseek|0"])
    quality = read_jsonl(tmp_path / "quality.jsonl")
    write_jsonl(tmp_path / "judge.jsonl", [{"tid": r["tid"], "qid": r["qid"], "overall_score": r["llm_score"]}
                                           for r in quality if r["llm_score"] is not None])
    judge = FakeJudge()
    monkeypatch.setattr(a4, "make_judge", lambda cfg, model_id: judge)
    rng = np.random.default_rng(7)

    def fake_encode(texts, model_id, device, max_length, batch_size):
        v = rng.normal(size=(len(texts), 16)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    monkeypatch.setattr(a5, "encode", fake_encode)

    def fake_b1(argv):
        out = argv[argv.index("--out") + 1]
        src = argv[argv.index("--override") + 1].split("=")[1]
        wd = Path(argv[argv.index("--workdir") + 1])
        write_jsonl(wd / out, [_fit_row(c["tid"], c["qid"], cfg.student.base_model_id, rng, with_local=False)
                               for c in read_jsonl(wd / src)])
        return 0
    monkeypatch.setattr(b1_fit, "main", fake_b1)
    return tmp_path, cfg, judge


def _run(wd, step, *extra):
    return bw.main(["--workdir", str(wd), "--step", step, *extra])


def _backfill(wd):
    for step, extra in (("prepare", ["--prev-cost-usd", "9.4"]), ("judge", ["--max-cost", "5"]), ("embed", []),
                        ("fit", []), ("combine", [])):
        assert _run(wd, step, *extra) == 0


# ---------------------------------------------------------------- kho mở rộng là gì
def test_wrong_chains_are_exactly_the_non_candidates_of_usable_questions(work):
    wd, cfg, _ = work
    assert _run(wd, "prepare") == 0
    wrong = read_jsonl(wd / "candidates.wrong.jsonl")
    cands = {c["tid"] for c in read_jsonl(wd / "candidates.jsonl")}
    traj = read_jsonl(wd / "trajectories.jsonl")
    inp = b2.Inputs(cfg, wd, "fit.qwen1_5b_base.jsonl")
    expected = sorted(t["tid"] for t in traj if t["qid"] in inp.pool and t["tid"] not in cands)
    assert [c["tid"] for c in wrong] == expected and expected
    assert all(c["text"] for c in wrong)
    # và đúng bằng phần chuỗi sai của kho 'all' mà b2_select tự dựng
    qs = inp.questions("all", "random")
    assert {t for q in qs.values() for t, ok in zip(q["tids"], q["correct"]) if not ok} == set(expected)


def test_prepare_refuses_to_reuse_a_list_that_no_longer_matches(work):
    wd, _, _ = work
    _run(wd, "prepare")
    write_jsonl(wd / "candidates.wrong.jsonl", read_jsonl(wd / "candidates.wrong.jsonl")[1:])
    with pytest.raises(SystemExit, match="KHÁC danh sách"):
        _run(wd, "prepare")


# ---------------------------------------------------------------- chi phí
def test_cost_estimate_uses_previous_cost_per_chain_and_length_ratio():
    e = bw.estimate_cost(9.0, 15000, prev_chars=[1000] * 10, new_chars=[2000] * 2000)
    assert e["per_chain"] == pytest.approx(0.0006) and e["plain"] == pytest.approx(1.2)
    assert e["length_ratio"] == pytest.approx(2.0) and e["scaled"] == pytest.approx(2.4)
    shorter = bw.estimate_cost(9.0, 15000, [1000] * 10, [500] * 2000)
    assert shorter["scaled"] == pytest.approx(shorter["plain"])        # không hạ ước tính khi chuỗi ngắn hơn
    with pytest.raises(ValueError):
        bw.estimate_cost(0.0, 100, [1], [1])


def test_prepare_prints_estimate_only_when_given_the_previous_cost(work, capsys):
    wd, _, judge = work
    _run(wd, "prepare")
    assert "CHƯA ƯỚC ĐƯỢC CHI PHÍ" in capsys.readouterr().out
    _run(wd, "prepare", "--prev-cost-usd", "9.4")
    assert "ƯỚC TÍNH cho" in capsys.readouterr().out and judge.prompts == []      # chưa gọi giám khảo lần nào


def test_judge_step_requires_a_budget(work):
    wd, _, judge = work
    _run(wd, "prepare")
    with pytest.raises(SystemExit, match="--max-cost"):
        _run(wd, "judge")
    assert judge.prompts == []


def test_judge_uses_the_same_prompt_builder_with_reference_solution(work):
    wd, _, judge = work
    _run(wd, "prepare")
    _run(wd, "judge", "--max-cost", "5")
    wrong = read_jsonl(wd / "candidates.wrong.jsonl")
    assert len(judge.prompts) == len(wrong)
    q = {r["qid"]: r for r in read_jsonl(wd / "questions.jsonl")}
    c = wrong[0]
    from src.common.prompts import build_judge_prompt
    assert build_judge_prompt(q[c["qid"]]["question"], c["text"], q[c["qid"]]["solution"]) in judge.prompts
    row = read_jsonl(wd / "judge.wrong.jsonl")[0]
    assert row["model"] == load_config().judge.model_id and row["use_reference"] is True
    n = len(judge.prompts)
    _run(wd, "judge", "--max-cost", "5")                                 # chạy lại không chấm lại chuỗi đã có
    assert len(judge.prompts) == n


def test_fit_command_is_base_16bit_without_localnat_and_a_separate_file():
    cfg = load_config(student="qwen1_5b")
    argv = bw.fit_argv(Path("/w"), "qwen1_5b", b2.extended_files(cfg), ["a.b=1"])
    assert "--base" in argv and "--skip-localnat" in argv and "--load-4bit" not in argv
    out = argv[argv.index("--out") + 1]
    assert out == "fit.qwen1_5b_base.wrong.jsonl" and out != f"fit.{b2.student_tag(cfg)}.jsonl"
    assert "stage_a_files.candidates=candidates.wrong.jsonl" in argv and "a.b=1" in argv
    # b1_fit thật phải nhận đúng các cờ này
    for flag in ("--base", "--skip-localnat", "--out", "--override", "--workdir", "--student"):
        assert flag in Path(b1_fit.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------- ghép và an toàn dữ liệu
def test_backfill_never_touches_the_existing_files(work):
    wd, _, _ = work
    before = _md5s(wd)
    _backfill(wd)
    assert _md5s(wd) == before
    for name in ("candidates.wrong.jsonl", "judge.wrong.jsonl", "embeddings.wrong.npz",
                 "fit.qwen1_5b_base.wrong.jsonl", "quality.all.jsonl", "quality.all_complete.jsonl"):
        assert (wd / name).exists()


def test_extended_quality_renormalises_within_the_larger_group(work):
    wd, cfg, _ = work
    _backfill(wd)
    old = {r["tid"]: r for r in read_jsonl(wd / "quality.jsonl")}
    new = {r["tid"]: r for r in read_jsonl(wd / "quality.all.jsonl")}
    wrong = {c["tid"] for c in read_jsonl(wd / "candidates.wrong.jsonl")}
    assert wrong <= set(new) and "math_00002|deepseek|0" not in new          # chuỗi đúng thiếu Qual vẫn ở ngoài
    shared = [t for t in new if t in old]
    assert all(new[t]["llm_score"] == old[t]["llm_score"] for t in shared)   # điểm giám khảo thô không đổi
    assert all(new[t]["rule_score"] == old[t]["rule_score"] for t in shared)  # điểm quy tắc trên thang kho gốc
    touched = {new[t]["qid"] for t in wrong}
    assert any(abs(new[t]["qual"] - old[t]["qual"]) > 1e-6 for t in shared if new[t]["qid"] in touched)
    untouched = [t for t in shared if new[t]["qid"] not in touched]
    assert untouched and all(new[t]["qual"] == old[t]["qual"] for t in untouched)   # câu không có chuỗi sai: y nguyên
    by_q = {}
    for r in new.values():
        by_q.setdefault(r["qid"], []).append(r)
    for rows in by_q.values():                                               # min-max trên cả nhóm mở rộng
        assert min(r["rule_norm"] for r in rows) == 0.0 and max(r["rule_norm"] for r in rows) == 1.0


def test_questions_without_wrong_chains_are_selected_exactly_as_in_the_correct_pool(work):
    """Hồi quy cho lỗi đo được 02/10: chuẩn hoá lại điểm quy tắc trên kho mở rộng làm 97/1.236 câu KHÔNG có
    chuỗi sai nào vẫn đổi tập chọn. Kho mở rộng chỉ được làm đổi những câu có chuỗi sai."""
    wd, cfg, _ = work
    _backfill(wd)
    inp = b2.Inputs(cfg, wd, "fit.qwen1_5b_base.jsonl")
    touched = {c["qid"] for c in read_jsonl(wd / "candidates.wrong.jsonl")}
    untouched = set(inp.pool) - touched
    assert untouched and touched
    for lam in (0.0, 0.4, 1.0):
        over = [f"selection.lambda_div={lam}"]
        base = _picked(inp, _cfg(overrides=over))[1]
        ext = _picked(inp, _cfg(ablation="no_prefilter", overrides=over))[1]
        assert all(base[q] == ext[q] for q in untouched), lam
    assert any(base[q] != ext[q] for q in touched)            # còn câu có chuỗi sai thì được phép đổi


def test_all_complete_pool_drops_truncated_chains_everywhere(work):
    wd, cfg, _ = work
    traj = read_jsonl(wd / "trajectories.jsonl")
    cands = {c["tid"] for c in read_jsonl(wd / "candidates.jsonl")}
    cut = [t["tid"] for t in traj if t["tid"] not in cands][:15]
    for t in traj:
        if t["tid"] in cut:
            t["finish_reason"] = "length"
    write_jsonl(wd / "trajectories.jsonl", traj)
    _backfill(wd)
    full = {r["tid"] for r in read_jsonl(wd / "quality.all.jsonl")}
    complete = {r["tid"] for r in read_jsonl(wd / "quality.all_complete.jsonl")}
    assert set(cut) <= full and not set(cut) & complete and complete < full
    inp = b2.Inputs(cfg, wd, "fit.qwen1_5b_base.jsonl")
    in_all = {t for q in inp.questions("all", "objective").values() for t in q["tids"]}
    in_complete = {t for q in inp.questions("all_complete", "objective").values() for t in q["tids"]}
    assert in_all - in_complete == set(cut) & in_all and set(cut) & in_all
    # bộ đếm chuỗi bị cắt trong tóm tắt
    qs, _picked_sets, picks = _picked(inp, _cfg(ablation="no_prefilter"))
    s = b2.summarize(qs, picks, 3)
    chosen = {qs[q]["tids"][i] for q, p in picks.items() for i in p["idx"]}
    assert s["truncated_kept"] == len(chosen & set(cut)) <= s["wrong_kept"]
    # đổi kho của biến thể sang all_complete thì không còn chuỗi bị cắt nào được chọn
    cfg2 = _cfg(ablation="no_prefilter", overrides=["select.pool=all_complete"])
    qs2, picked2, picks2 = _picked(inp, cfg2)
    assert b2.summarize(qs2, picks2, 3)["truncated_kept"] == 0 and set(picked2) == set(inp.pool)
    # No-Filter trên all_complete cũng không rút trúng chuỗi bị cắt
    nf = _picked(inp, _cfg("no_filter", overrides=["select.pool=all_complete"]))[1]
    assert not {t for s_ in nf.values() for t in s_} & set(cut)


def test_combine_refuses_when_embeddings_or_fit_are_incomplete(work):
    wd, _, _ = work
    _run(wd, "prepare")
    _run(wd, "judge", "--max-cost", "5")
    with pytest.raises(SystemExit, match="Chưa đủ để ghép"):
        _run(wd, "combine")


# ---------------------------------------------------------------- b2_select trên kho mở rộng
def test_no_prefilter_runs_on_the_extended_pool_and_can_pick_wrong_chains(work):
    wd, cfg, _ = work
    _backfill(wd)
    inp = b2.Inputs(cfg, wd, "fit.qwen1_5b_base.jsonl")
    qs, picked, picks = _picked(inp, _cfg(ablation="no_prefilter"))
    assert set(picked) == set(inp.pool) and all(len(s) == 3 for s in picked.values())
    sizes = {q: len(v["tids"]) for q, v in qs.items()}
    assert max(sizes.values()) == 9 and all(sizes[q] >= len(inp.pool[q]) for q in sizes)
    new = {r["tid"]: r for r in read_jsonl(wd / "quality.all.jsonl")}
    q = next(v for v in qs.values() if not all(v["correct"]))
    assert q["qual"].tolist() == pytest.approx([new[t]["qual"] for t in q["tids"]])   # Qual của kho mở rộng
    assert q["fit"].min() == 0.0 and q["fit"].max() == 1.0                            # Fit min-max trên 9 chuỗi
    s = b2.summarize(qs, picks, 3)
    assert s["wrong_kept"] >= 0 and s["mean_div"] is not None
    # kho chuỗi đúng không bị ảnh hưởng: QD-RSR vẫn chọn như khi chưa chấm bù
    assert all(all(v["correct"]) for v in inp.questions("correct", "objective").values())


def test_unjudgeable_wrong_chain_is_dropped_from_the_extended_pool_only(work):
    wd, cfg, _ = work
    _run(wd, "prepare")
    wrong = read_jsonl(wd / "candidates.wrong.jsonl")
    wrong[0]["text"] = "UNJUDGEABLE " + wrong[0]["text"]
    write_jsonl(wd / "candidates.wrong.jsonl", wrong)
    for step, extra in (("judge", ["--max-cost", "5"]), ("embed", []), ("fit", []), ("combine", [])):
        _run(wd, step, *extra)
    bad = wrong[0]["tid"]
    assert {r["tid"]: r for r in read_jsonl(wd / "quality.all.jsonl")}[bad]["qual"] is None
    inp = b2.Inputs(cfg, wd, "fit.qwen1_5b_base.jsonl")
    ext = inp.questions("all", "objective")
    assert bad not in {t for q in ext.values() for t in q["tids"]}
    assert bad in {t for q in inp.questions("all", "random").values() for t in q["tids"]}   # No-Filter không đổi


def test_no_filter_selection_is_unchanged_by_the_backfill(work):
    wd, cfg, _ = work
    before = _picked(b2.Inputs(cfg, wd, "fit.qwen1_5b_base.jsonl"), _cfg("no_filter"))[1]
    _backfill(wd)
    assert _picked(b2.Inputs(cfg, wd, "fit.qwen1_5b_base.jsonl"), _cfg("no_filter"))[1] == before


def test_stale_extended_quality_is_detected(work):
    wd, cfg, _ = work
    _backfill(wd)
    rows = read_jsonl(wd / "quality.jsonl")
    target = next(r for r in rows if r["llm_score"] is not None)
    target["llm_score"] = 0.123
    write_jsonl(wd / "quality.jsonl", rows)
    inp = b2.Inputs(cfg, wd, "fit.qwen1_5b_base.jsonl")
    with pytest.raises(SystemExit, match="lệch điểm giám khảo"):
        inp.questions("all", "objective")

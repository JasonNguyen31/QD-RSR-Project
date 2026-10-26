"""Kiểm thử check_grape_format: kiểm định xem đổi khuôn chấm GRAPE có làm đổi tập top-3 hay không.

Phần chạy mô hình (make_gpu_scorer) cần GPU và dùng lại b1_fit.token_stats, b1_fit.aggregate, b1_fit.build_inputs đã
có bài kiểm thử riêng, nên ở đây được thay bằng bộ chấm giả. Mọi bài chạy offline, không cần torch.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_b"))

from src.common.config import load_config
from src.common.io_utils import JsonlWriter, read_json, read_jsonl, write_jsonl
from src.stage_b import b2_select as b2
from src.tools import check_grape_format as cg
from test_b2_select import make_workdir

K = load_config()["selection"]["k"]
SEED = load_config()["selection"]["random_seed"]
DATA_SEED = load_config()["project"]["data_seed"]


def pool_of(tmp, n_q=12):
    cfg = make_workdir(tmp, n_q=n_q)
    fit = {r["tid"]: r for r in read_jsonl(tmp / "fit.qwen1_5b_base.jsonl")}
    cands = read_jsonl(tmp / cfg.stage_a_files["candidates"])
    quality = {r["tid"]: r for r in read_jsonl(tmp / "quality.jsonl")}
    return cfg, cands, fit, quality


# ---------------------------------------------------------------- khuôn
def test_template_needs_one_question_then_one_answer_at_the_end():
    assert cg.parse_template(cg.PLAIN_TEMPLATE) == "Question: {question}\nAnswer: "
    for bad in ("{question}", "{answer}", "{answer} {question}", "Q {question} {question} A {answer}",
                "Q {question} A {answer} thừa"):
        with pytest.raises(SystemExit):
            cg.parse_template(bad)


def test_trailing_space_before_the_answer_moves_to_the_answer():
    """'Answer:' rồi ' The', không phải 'Answer: ' rồi 'The': token của Qwen mang dấu cách ở đầu."""
    prompt, lead = cg.plain_split(cg.PLAIN_TEMPLATE, "What is 1+1?")
    assert prompt == "Question: What is 1+1?\nAnswer:" and lead == " "
    assert cg.plain_split("Q: {question}\nA:{answer}", "x") == ("Q: x\nA:", "")
    assert cg.plain_split("Q: {question}\nA:\n{answer}", "x") == ("Q: x\nA:\n", "")      # xuống dòng giữ ở phần đề


def test_question_text_is_not_touched_by_template_markers():
    """Đề bài chứa {answer} hay {question} không được bị thay thế thêm lần nữa."""
    prompt, _ = cg.plain_split(cg.PLAIN_TEMPLATE, "Find {answer} when {question} is 2")
    assert prompt == "Question: Find {answer} when {question} is 2\nAnswer:"


class RecordingTok:
    """Mỗi từ một token; ghi lại các chuỗi đã được mã hoá để kiểm tra chỗ cắt giữa đề và chuỗi."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False                  # Qwen không dùng BOS; phải giống b1_fit
        self.calls.append(text)
        return {"input_ids": list(range(len(text.split())))}


def test_plain_ids_encode_prompt_and_answer_separately_like_b1():
    tok = RecordingTok()
    got = cg.plain_token_ids(tok, "What is 1+1 ?", "so it is 2 .", 3072, cg.PLAIN_TEMPLATE)
    assert tok.calls == ["Question: What is 1+1 ?\nAnswer:", " so it is 2 ."]
    p_ids, r_ids = got
    assert len(p_ids) == 6 and len(r_ids) == 5


def test_plain_ids_truncate_at_the_same_total_length_as_b1():
    tok = RecordingTok()
    p_ids, r_ids = cg.plain_token_ids(tok, "q", " ".join(["w"] * 100), 40, cg.PLAIN_TEMPLATE)
    assert len(p_ids) + len(r_ids) == 40
    assert cg.plain_token_ids(tok, "q", " ".join(["w"] * 100), len(p_ids) + 1, cg.PLAIN_TEMPLATE) is None   # còn 1 token
    assert cg.plain_token_ids(tok, "q", "", 3072, cg.PLAIN_TEMPLATE) is None                                 # chuỗi rỗng


# ---------------------------------------------------------------- tập top-k
def test_top_set_is_b2_topk_for_a_maximised_signal_with_the_same_tie_break():
    s = np.array([0.3, 0.9, 0.9, 0.9, 0.1])
    assert cg.top_set(s, 3) == frozenset(b2.pick_topk(s, 3, "max")["idx"]) == frozenset({1, 2, 3})
    t = np.array([0.5, 0.5, 0.5, 0.5, 0.5])                 # hoà hết: thứ tự cố định giữ k chuỗi đầu
    assert cg.top_set(t, 3) == frozenset({0, 1, 2})
    with pytest.raises(SystemExit):
        cg.top_set([0.1, float("nan"), 0.3, 0.2], 3)


def test_wilson_interval():
    lo, hi = cg.wilson(160, 200)
    assert lo < 0.8 < hi and 0.73 < lo < 0.75 and 0.84 < hi < 0.86       # 160/200: [73,9%; 85,0%]
    lo, hi = cg.wilson(200, 200)
    assert hi == pytest.approx(1.0) and lo > 0.98
    assert cg.wilson(0, 0) is None


def make_rows():
    """q1: hai cách chấm xếp giống hệt. q2: B đổi một chuỗi trong top-3. q3: B đảo hẳn thứ tự."""
    return {
        "q1": {"tids": list("abcd"), "teacher": ["x", "x", "y", "y"], "n_tokens": [10, 20, 30, 40],
               "a": [4.0, 3.0, 2.0, 1.0], "b": [40.0, 30.0, 20.0, 10.0]},
        "q2": {"tids": list("abcd"), "teacher": ["x", "x", "y", "y"], "n_tokens": [10, 20, 30, 40],
               "a": [4.0, 3.0, 2.0, 1.0], "b": [4.0, 3.0, 1.0, 2.0]},
        "q3": {"tids": list("abcd"), "teacher": ["x", "x", "y", "y"], "n_tokens": [10, 20, 30, 40],
               "a": [4.0, 3.0, 2.0, 1.0], "b": [1.0, 2.0, 3.0, 4.0]},
    }


def test_compare_counts_identical_sets_shared_chains_and_rank_agreement():
    c = cg.compare_scores(make_rows(), "a", "b", 3)
    assert (c["same"], c["questions"]) == (2 - 1, 3)                       # chỉ q1 cùng tập; q2 và q3 khác
    assert c["rate"] == pytest.approx(1 / 3)
    assert c["mean_shared"] == pytest.approx((3 + 2 + 2) / 3)              # q3: top-3 của B là b, c, d: chung b, c
    assert c["spearman"] == pytest.approx((1.0 + 0.8 - 1.0) / 3)
    assert [x["same"] for x in c["per_question"]] == [True, False, False]
    # A chọn a,b,c ở cả ba câu; B chọn a,b,c / a,b,d / b,c,d
    assert c["teacher_share_a"] == {"x": pytest.approx(6 / 9), "y": pytest.approx(3 / 9)}
    assert c["teacher_share_b"] == {"x": pytest.approx(5 / 9), "y": pytest.approx(4 / 9)}
    assert c["mean_tokens_a"] == pytest.approx(20.0)
    assert c["mean_tokens_b"] == pytest.approx((60 + 70 + 90) / 9)


def test_identical_scores_give_one_hundred_percent():
    rows = make_rows()
    c = cg.compare_scores(rows, "a", "a", 3)
    assert c["rate"] == 1.0 and c["mean_shared"] == 3 and c["spearman"] == pytest.approx(1.0)


def test_the_decision_rule_is_eighty_percent_and_inclusive():
    assert cg.THRESHOLD == 0.80
    assert cg.decide(0.80) == "keep" and cg.decide(0.81) == "keep"
    assert cg.decide(0.7999) == "rerun" and cg.decide(0.0) == "rerun"
    assert cg.decide(158 / 200) == "rerun" and cg.decide(160 / 200) == "keep"    # 79% giữ, 80% thì giữ


# ---------------------------------------------------------------- kho và mẫu
def test_pool_has_the_same_chains_in_the_same_order_as_b2(tmp_path):
    cfg, cands, fit, quality = pool_of(tmp_path)
    by_q, info = cg.build_pool(cands, fit, quality, int(cfg.data.min_correct_per_question), SEED)
    inp = b2.Inputs(cfg, tmp_path, "fit.qwen1_5b_base.jsonl")
    ref = inp.questions("correct", "topk")
    assert set(by_q) == set(ref)
    for qid, chains in by_q.items():
        assert [c["tid"] for c in chains] == ref[qid]["tids"]                       # cùng thứ tự cố định
        assert [c["current"] for c in chains] == pytest.approx(list(ref[qid]["signals"]["grape"]))
        assert [c["teacher"] for c in chains] == ref[qid]["teacher"]
    assert info["pool_source"].startswith("usable_pool")


def test_pool_without_quality_still_works_and_says_so(tmp_path):
    cfg, cands, fit, quality = pool_of(tmp_path)
    with_q, _ = cg.build_pool(cands, fit, quality, 3, SEED)
    no_q, info = cg.build_pool(cands, fit, None, 3, SEED)
    assert {q: [c["tid"] for c in v] for q, v in no_q.items()} == {q: [c["tid"] for c in v] for q, v in with_q.items()}
    assert "KHÔNG có quality.jsonl" in info["pool_source"]


def test_pool_stops_when_the_fit_file_misses_a_chain(tmp_path):
    cfg, cands, fit, quality = pool_of(tmp_path)
    fit.pop(cands[0]["tid"])
    with pytest.raises(SystemExit, match="không có dòng trong file fit"):
        cg.build_pool(cands, fit, quality, 3, SEED)


def test_the_sample_only_draws_questions_with_more_than_k_chains_and_is_reproducible(tmp_path):
    cfg, cands, fit, quality = pool_of(tmp_path, n_q=12)
    by_q, _ = cg.build_pool(cands, fit, quality, 3, SEED)
    big = {q for q, v in by_q.items() if len(v) > K}
    assert 0 < len(big) < len(by_q) or len(big) == len(by_q)
    a = cg.sample_questions(by_q, K, 5, DATA_SEED)
    assert a == cg.sample_questions(by_q, K, 5, DATA_SEED) and len(a) == 5 and set(a) <= big
    assert cg.sample_questions(by_q, K, 10 ** 6, DATA_SEED) == sorted(big)           # xin quá nhiều thì lấy hết
    assert cg.sample_questions(by_q, K, 5, DATA_SEED + 1) != a


# ---------------------------------------------------------------- file điểm
def test_resume_refuses_another_template_model_or_sample_size():
    want = cg.settings_of("m", cg.PLAIN_TEMPLATE, 3072, 200)
    cg.check_resume(None, want, "f")
    cg.check_resume(dict(want, tid="t", grape_plain=-1.0), want, "f")
    for key, value in (("template", "Q: {question}\nA: {answer}"), ("model", "other"), ("max_len", 2048),
                       ("sample_n", 100)):
        with pytest.raises(SystemExit, match="cài đặt khác"):
            cg.check_resume(dict(want, **{key: value}), want, "f")


def fake_scorer(flip=()):
    """grape_chat bằng đúng cột grape của file fit (sàn nhiễu 100%); grape_plain đổi thứ tự ở các câu trong flip."""
    state = {}

    def scorer(question, text):
        v = state[text]
        return {"grape_plain": v[1], "n_tokens_plain": 10, "grape_chat": v[0], "n_tokens_chat": 11}
    scorer.state = state
    return scorer


def score_sample(tmp, by_q, qids, flip_questions, path, n_q=None):
    """Ghi file điểm bằng score_all. Ở câu trong flip_questions, grape_plain đảo thứ tự so với cột grape hiện tại."""
    questions = {q["qid"]: q for q in read_jsonl(tmp / "questions.jsonl")}
    sc = fake_scorer()
    for q in qids:
        cur = [c["current"] for c in by_q[q]]
        plain = [-x for x in cur] if q in flip_questions else [x - 1.0 for x in cur]     # trừ đều: giữ nguyên thứ hạng
        for c, a, b in zip(by_q[q], cur, plain):
            sc.state[c["text"]] = (a, b)
    meta = cg.settings_of(load_config(student="qwen1_5b").student.base_model_id, cg.PLAIN_TEMPLATE, 3072, len(qids))
    todo = [(q, c) for q in qids for c in by_q[q]]
    with JsonlWriter(path) as w:
        return cg.score_all(todo, questions, sc, w, meta, progress=lambda it, **_: it)


def test_score_all_writes_one_row_per_chain_and_counts_failures(tmp_path):
    cfg, cands, fit, quality = pool_of(tmp_path)
    by_q, _ = cg.build_pool(cands, fit, quality, 3, SEED)
    qids = cg.sample_questions(by_q, K, 4, DATA_SEED)
    path = tmp_path / "g.jsonl"
    assert score_sample(tmp_path, by_q, qids, set(), path) == 0
    rows = read_jsonl(path)
    assert len(rows) == sum(len(by_q[q]) for q in qids)
    assert {"tid", "qid", "grape_plain", "grape_chat", "n_tokens_plain", "n_tokens_chat", "model", "template",
            "max_len", "sample_n", "ts"} <= set(rows[0])
    questions = {q["qid"]: q for q in read_jsonl(tmp_path / "questions.jsonl")}
    seen = []
    def none_scorer(question, text):
        seen.append(text)
        return None
    todo = [(qids[0], by_q[qids[0]][0])]
    with JsonlWriter(tmp_path / "h.jsonl") as w:
        assert cg.score_all(todo, questions, none_scorer, w, {}, progress=lambda it, **_: it) == 1
    assert read_jsonl(tmp_path / "h.jsonl") == []                                   # chuỗi không chấm được không ghi dòng


def test_report_counts_flipped_questions_and_decides(tmp_path):
    cfg, cands, fit, quality = pool_of(tmp_path, n_q=12)
    by_q, info = cg.build_pool(cands, fit, quality, 3, SEED)
    qids = cg.sample_questions(by_q, K, 20, DATA_SEED)
    flipped = set(qids[:2])
    path = tmp_path / "g.jsonl"
    score_sample(tmp_path, by_q, qids, flipped, path)
    rep = cg.build_report(by_q, qids, {r["tid"]: r for r in read_jsonl(path)}, K, info)
    p = rep["current_vs_plain"]
    assert p["questions"] == len(qids) == rep["scored"] and rep["skipped"] == []
    assert rep["current_vs_chat"]["rate"] == 1.0                                    # sàn nhiễu: cùng khuôn, cùng tập
    assert p["rate"] == rep["chat_vs_plain"]["rate"]                                # A' bằng A ở dữ liệu giả này
    # các câu đảo thứ tự có thể vẫn trùng tập nếu chỉ có k + 1 chuỗi; ít nhất mọi câu không đảo phải trùng
    assert all(x["same"] for x in p["per_question"] if x["qid"] not in flipped)
    assert p["same"] >= len(qids) - len(flipped)
    assert rep["decision"] == cg.decide(p["rate"])
    n_big = sum(len(v) > K for v in by_q.values())
    assert rep["questions_with_more_than_k"] == n_big
    assert rep["pool_estimate_same_set"] == pytest.approx(((len(by_q) - n_big) + p["rate"] * n_big) / len(by_q))


def test_report_skips_questions_with_a_missing_score_and_stops_when_nothing_is_scored(tmp_path):
    cfg, cands, fit, quality = pool_of(tmp_path)
    by_q, info = cg.build_pool(cands, fit, quality, 3, SEED)
    qids = cg.sample_questions(by_q, K, 4, DATA_SEED)
    path = tmp_path / "g.jsonl"
    score_sample(tmp_path, by_q, qids, set(), path)
    scores = {r["tid"]: r for r in read_jsonl(path)}
    scores.pop(by_q[qids[0]][0]["tid"])                                             # một chuỗi của câu đầu chưa chấm
    rep = cg.build_report(by_q, qids, scores, K, info)
    assert rep["skipped"] == [qids[0]] and rep["scored"] == len(qids) - 1
    with pytest.raises(SystemExit, match="Chưa có câu nào"):
        cg.build_report(by_q, qids, {}, K, info)


def test_train_consistency_catches_a_pool_that_differs_from_the_one_used_for_training(tmp_path):
    cfg, cands, fit, quality = pool_of(tmp_path)
    by_q, _ = cg.build_pool(cands, fit, quality, 3, SEED)
    qids = cg.sample_questions(by_q, K, 6, DATA_SEED)
    train = [{"qid": q, "tid": t} for q in qids for t in cg.current_tids(by_q[q], K)]
    assert cg.train_consistency(train, by_q, qids, K) == {"questions": 6, "same": 6}
    train[0] = {"qid": train[0]["qid"], "tid": "khác"}
    assert cg.train_consistency(train, by_q, qids, K) == {"questions": 6, "same": 5}
    assert cg.train_consistency([], by_q, qids, K) == {"questions": 0, "same": 0}   # không có file train: không đếm


# ---------------------------------------------------------------- dòng lệnh, chế độ báo cáo
def test_main_report_reads_scores_without_a_gpu_and_writes_the_json(tmp_path, capsys):
    cfg, cands, fit, quality = pool_of(tmp_path)
    by_q, _ = cg.build_pool(cands, fit, quality, 3, SEED)
    qids = cg.sample_questions(by_q, K, 5, DATA_SEED)
    score_sample(tmp_path, by_q, qids, set(), tmp_path / "grape_format.qwen1_5b_base.jsonl")
    assert cg.main(["--workdir", str(tmp_path), "--n-questions", "5", "--report"]) == 0
    out = capsys.readouterr().out
    assert "Quy tắc chốt trước" in out and "GIỮ cách chấm hiện tại" in out and "sàn nhiễu" in out
    rep = read_json(tmp_path / "grape_format_check.qwen1_5b_base.json")
    assert rep["decision"] == "keep" and rep["scored"] == len(qids) and rep["threshold"] == 0.8
    json.dumps(rep)                                                                 # ghi được ra JSON, không có kiểu lạ


def test_main_report_warns_when_the_noise_floor_is_not_near_one_hundred_percent(tmp_path, capsys):
    cfg, cands, fit, quality = pool_of(tmp_path)
    by_q, _ = cg.build_pool(cands, fit, quality, 3, SEED)
    qids = cg.sample_questions(by_q, K, 8, DATA_SEED)
    path = tmp_path / "grape_format.qwen1_5b_base.jsonl"
    score_sample(tmp_path, by_q, qids, set(), path)
    rows = read_jsonl(path)
    for r in rows:                                                                  # làm hỏng grape_chat: sàn nhiễu tụt
        r["grape_chat"] = -r["grape_chat"]
    write_jsonl(path, rows)
    cg.main(["--workdir", str(tmp_path), "--n-questions", "8", "--report"])
    assert "CẢNH BÁO: sàn nhiễu" in capsys.readouterr().out


def test_main_refuses_a_score_file_made_with_a_different_sample_size(tmp_path):
    cfg, cands, fit, quality = pool_of(tmp_path)
    by_q, _ = cg.build_pool(cands, fit, quality, 3, SEED)
    qids = cg.sample_questions(by_q, K, 5, DATA_SEED)
    score_sample(tmp_path, by_q, qids, set(), tmp_path / "grape_format.qwen1_5b_base.jsonl")
    with pytest.raises(SystemExit, match="cài đặt khác"):
        cg.main(["--workdir", str(tmp_path), "--n-questions", "3", "--report"])


def test_main_stops_on_a_fit_file_of_another_model(tmp_path):
    make_workdir(tmp_path, model="Qwen/Qwen2.5-1.5B-Instruct")
    with pytest.raises(SystemExit, match="Sai file fit"):
        cg.main(["--workdir", str(tmp_path), "--report"])

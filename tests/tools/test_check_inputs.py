"""Ba phép kiểm đầu vào trước b2_select. Dữ liệu giả dựng qua a4.combine để có đúng tên khoá của file thật."""
import json

from src.stage_a.a4_score_quality import combine
from src.tools import check_inputs as ci

WORDS = ["We check the answer. Therefore it is 4.",
         "Perhaps we might try. Since x is 2, the answer is 4.",
         "Compute directly. The answer is 4.",
         "Let us verify: 2 + 2 = 4. Therefore 4.",
         "First, note the sum.\nThen add.\n\nSo the answer is 4."]


def _quality(judge_scores_by_q):
    cands, judge = [], []
    for qid, scores in judge_scores_by_q.items():
        for i, s in enumerate(scores):
            tid = f"{qid}|deepseek|{i}"
            cands.append({"tid": tid, "qid": qid, "text": WORDS[i % len(WORDS)] * (i + 1)})
            judge.append({"tid": tid, "qid": qid, "overall_score": s})
    return cands, judge, combine(cands, judge, alpha=0.5)


def test_constant_judge_means_qual_follows_rule_score():
    _c, _j, q = _quality({"q1": [1.0] * 5, "q2": [1.0, 0.2, 0.2, 0.9, 0.1], "q3": [0.8, 0.8, 0.8]})
    s = ci.ceiling_stats(q, k=3)
    assert s["questions"] == 3 and s["multi"] == 2          # q3 có đúng k ứng viên, không tính
    assert s["const"] == 1 and s["const_top"] == 1
    assert s["same_as_rule"] >= 1                           # ở q1 Qual chỉ còn điểm quy tắc


def test_chains_without_judge_score_are_ignored():
    _c, _j, q = _quality({"q1": [1.0, 0.5, 0.5, 0.2]})
    q.append({"tid": "q1|llama|9", "qid": "q1", "rule_score": 5.0, "llm_score": None,
              "rule_norm": None, "llm_norm": None, "qual": None, "n_words": 3})
    assert ci.ceiling_stats(q, k=3)["chains"] == 4


def test_top_k_breaks_ties_by_tid():
    rows = [{"tid": t, "v": 1.0} for t in ("c", "a", "b", "d")]
    assert ci.top_k(rows, "v", 2) == frozenset({"a", "b"})


def test_step_stats_counts_sentences_and_single_step_chains():
    s = ci.step_stats(["One. Two words here. Three.\n\nFour is last.", "no punctuation at all"])
    assert s["n_steps"][1] == 1 and s["one_step"] == 1
    assert s["n_steps"][0] >= 3


def test_show_split_marks_boundaries():
    out = ci.show_split("First sentence. Second one.\nThird line here.")
    assert "‖" in out and "Third" in out


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_commands_run_on_files_with_real_key_names(tmp_path, capsys):
    cands, judge, q = _quality({"q1": [1.0] * 4, "q2": [0.9, 0.3, 0.5, 0.7]})
    judge[0]["salvaged"] = "regex_overall"
    _write(tmp_path / "candidates.jsonl", cands)
    _write(tmp_path / "judge.jsonl", judge)
    _write(tmp_path / "quality.jsonl", q)
    _write(tmp_path / "questions.jsonl", [{"qid": "q1", "question": "2+2?", "gold": "4"},
                                          {"qid": "q2", "question": "2+2?", "gold": "4"}])
    _write(tmp_path / "fit.test.jsonl", [{"tid": c["tid"], "n_steps": 2} for c in cands])
    for argv in (["ceiling"], ["salvaged", "--show", "1"], ["steps", "--show", "2", "--fit", "fit.test.jsonl"]):
        assert ci.main(argv + ["--workdir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "[1]" in out and "[2]" in out and "[3]" in out and "regex_overall" in out

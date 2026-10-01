"""Hai phép đo thêm ngày 01/10: so 4-bit với 16-bit (rút mẫu theo câu) và độ nghiêng độ dài của Qual."""
from src.stage_a.a4_score_quality import combine
from src.stage_b.b1_fit import sample_questions
from src.tools.inspect_qual import within_question_corr


def _cands(nq=10, per=4):
    return [{"tid": f"math_{q:05d}|deepseek|{i}", "qid": f"math_{q:05d}", "text": "x"}
            for q in range(nq) for i in range(per)]


def test_sample_keeps_every_candidate_of_each_sampled_question():
    out = sample_questions(_cands(), 3, seed=42)
    qids = {c["qid"] for c in out}
    assert len(qids) == 3 and len(out) == 12


def test_sample_is_reproducible_and_seed_dependent():
    a = sample_questions(_cands(50), 5, seed=42)
    assert a == sample_questions(_cands(50), 5, seed=42)
    assert {c["qid"] for c in a} != {c["qid"] for c in sample_questions(_cands(50), 5, seed=7)}


def test_sample_larger_than_pool_returns_all():
    assert len(sample_questions(_cands(2), 10, seed=42)) == 8


def test_within_question_corr_on_real_quality_keys():
    """Dữ liệu dựng qua a4.combine để có đúng tên khoá của quality.jsonl."""
    texts = ["Short.", "A bit longer answer here.", "Much longer answer with many more words in it.",
             "The longest answer of all, with a great many words to make it clearly the longest one."]
    cands = [{"tid": f"q1|deepseek|{i}", "qid": "q1", "text": t} for i, t in enumerate(texts)]
    judge = [{"tid": c["tid"], "qid": "q1", "overall_score": s} for c, s in zip(cands, [0.1, 0.4, 0.6, 0.9])]
    rows = combine(cands, judge, alpha=0.5)
    v, nq = within_question_corr(rows, "llm_score", "n_words")
    assert nq == 1 and v > 0.99


def test_within_question_corr_skips_rows_without_qual():
    rows = [{"tid": f"q1|a|{i}", "qual": None, "n_words": i} for i in range(4)]
    assert within_question_corr(rows, "qual", "n_words") == (None, 0)

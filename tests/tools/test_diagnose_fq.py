"""Kiểm thử diagnose_fq trên các câu hỏi giả có đúng dạng mà b2_select.assemble trả về. Offline, không cần torch."""
import numpy as np
import pytest

from src.tools import diagnose_fq as d


def make_q(qid, teachers, fam, qual, ntok):
    tids = [f"{qid}|{t}|{i}" for i, t in enumerate(teachers)]
    return {"qid": qid, "tids": tids, "teacher": list(teachers), "n_tokens": list(ntok),
            "qual": np.array(qual, dtype=float), "signals": {"grape": np.array(fam, dtype=float)}}


@pytest.fixture
def data():
    # Câu A: Fam và Qual gần như ngược chiều (Spearman -0,8); qwen quen nhất nhưng Qual thấp nhất.
    qa = make_q("A", ["qwen72b", "qwen72b", "deepseek", "llama70b"], [-0.1, -0.2, -0.5, -0.6],
                [0.1, 0.2, 0.9, 0.8], [300, 350, 900, 700])
    # Câu B chỉ có 3 chuỗi: không có lựa chọn, phải bị bỏ qua.
    qb = make_q("B", ["qwen72b", "deepseek", "llama70b"], [-0.1, -0.3, -0.2], [0.5, 0.4, 0.3], [1, 2, 3])
    qs = {"A": qa, "B": qb}
    quality = {t: {"rule_score": float(q["qual"][i]), "llm_score": float(q["qual"][i])}
               for q in qs.values() for i, t in enumerate(q["tids"])}
    return qs, quality


def test_within_corr_skips_small_and_detects_opposition(data):
    qs, quality = data
    c = d.within_corr(qs, quality)
    assert c[("fam", "qual")]["n"] == 1
    assert c[("fam", "qual")]["median"] == pytest.approx(-0.8)
    assert c[("fam", "qual")]["neg"] == 1.0
    assert c[("qual", "len")]["median"] > 0.7          # Qual cao đi với chuỗi dài


def test_teacher_ranks(data):
    qs, quality = data
    r = d.teacher_ranks(qs, quality)
    assert r["qwen72b"]["n"] == 2 and "deepseek" in r and r["qwen72b"]["fam"] > r["deepseek"]["fam"]
    assert r["qwen72b"]["qual"] < r["deepseek"]["qual"]
    assert r["qwen72b"]["fam"] == pytest.approx((1.0 + 2 / 3) / 2)


def test_swap(data):
    qs, quality = data
    idx = d.chain_index(qs, quality)
    ta, tb = qs["A"]["tids"], qs["B"]["tids"]
    grape = {"A": {ta[0], ta[1], ta[2]}, "B": set(tb)}
    fq = {"A": {ta[0], ta[2], ta[3]}, "B": set(tb)}
    s = d.swap(grape, fq, idx)
    assert s["questions"] == 1 and s["identical"] == 0          # câu B bị bỏ qua vì chỉ có 3 chuỗi
    assert s["dropped"]["n"] == 1 and s["dropped"]["teacher"] == {"qwen72b": 1}
    assert s["added"]["teacher"] == {"llama70b": 1} and s["added"]["len"] == 700
    assert s["share_a"]["qwen72b"] == 2 and s["share_b"]["qwen72b"] == 1
    assert s["added"]["qual"] > s["dropped"]["qual"]

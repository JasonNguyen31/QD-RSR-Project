"""Kiểm thử length_split: tách chênh lệch Acc@4 do lượt chạm trần khỏi phần còn lại, chỉ đọc file đánh giá.

Dữ liệu giả ghi đúng dạng c1_evaluate ghi: <tên>/seed<seed>/summary.json và <bộ>.jsonl, mỗi dòng là một lượt sinh
với qid, sample, correct, n_tokens, stopped. Mọi bài chạy offline, không cần torch.
"""
import random

import pytest

from src.common.config import load_config
from src.common.io_utils import write_json, write_jsonl
from src.tools import length_split as ls
from src.tools.eval_table import MUST_MATCH, find_runs

CFG = load_config()
K = int(CFG["eval"]["n_samples"])
CAP = 100                                                  # trần nhỏ cho dữ liệu giả


def settings(**over):
    st = {k: CFG["eval"][k] for k in MUST_MATCH}
    st["max_new_tokens"] = CAP
    st.update(over)
    return st


def gen(correct, tokens=50, stopped=True):
    return (correct, tokens, stopped)


OK, BAD = gen(True), gen(False)
CUT = gen(False, CAP, False)                               # chạm trần, sai (chưa kịp viết \boxed)
CUT_OK = gen(True, CAP, False)                             # chạm trần nhưng vẫn có đáp án đúng


def write_run(root, tag, seed, per_q, bench="math500", **over):
    """per_q: qid -> danh sách K lượt (đúng, token, đã dừng)."""
    d = root / tag / f"seed{seed}"
    rows = [{"qid": q, "sample": i, "correct": c, "pred": None, "gold": "1", "n_tokens": t, "stopped": st, "text": ""}
            for q, gens in per_q.items() for i, (c, t, st) in enumerate(gens)]
    write_jsonl(d / f"{bench}.jsonl", rows)
    write_json(d / "summary.json", {"tag": tag, "seed": seed, "settings": settings(**over),
                                    "benchmarks": {bench: {"acc@4": 0.0, "n_questions": len(per_q)}}})


# A (fq_select) và B (quality_only): q1 đến q4 không lượt nào chạm trần; q5 chỉ B bị cắt; q6 cả hai bị cắt
A = {"q1": [OK] * 4, "q2": [OK] * 3 + [BAD], "q3": [OK] * 2 + [BAD] * 2, "q4": [BAD] * 4,
     "q5": [OK] * 3 + [BAD], "q6": [OK] * 2 + [BAD, CUT]}
B = {"q1": [OK] * 4, "q2": [OK] * 3 + [BAD], "q3": [OK] + [BAD] * 3, "q4": [BAD] * 4,
     "q5": [OK] + [BAD] + [CUT] * 2, "q6": [CUT] * 3 + [BAD]}


def split(a, b, seeds=(42,), tmp=None):
    """Đi qua file thật: ghi, đọc lại bằng find_runs rồi tách."""
    write_run(tmp, "a.k3", 42, a)
    write_run(tmp, "b.k3", 42, b)
    return ls.run_pair(find_runs(tmp), "a.k3", "b.k3", "math500", list(seeds), K, 500)


# ---------------------------------------------------------------- chạm trần và đọc file
def test_a_generation_is_capped_only_if_it_did_not_stop_and_used_the_whole_budget():
    assert ls.is_capped({"stopped": False, "n_tokens": CAP}, CAP)
    assert ls.is_capped({"stopped": False, "n_tokens": CAP + 5}, CAP)
    assert not ls.is_capped({"stopped": True, "n_tokens": CAP}, CAP)          # dừng đúng ở trần: không phải bị cắt
    assert not ls.is_capped({"stopped": False, "n_tokens": CAP - 1}, CAP)     # chưa đủ trần
    # đúng định nghĩa hit_cap_share của c1_evaluate
    assert CFG["eval"]["max_new_tokens"] == 3072


def test_load_generations_drops_incomplete_questions_and_extra_samples(tmp_path):
    rows = [{"qid": "q1", "sample": i, "correct": i % 2 == 0, "n_tokens": 10, "stopped": True} for i in range(K)]
    rows += [{"qid": "q2", "sample": i, "correct": True, "n_tokens": 10, "stopped": True} for i in range(K - 1)]   # thiếu
    rows += [{"qid": "q1", "sample": K, "correct": False, "n_tokens": 10, "stopped": True}]                    # thừa
    write_jsonl(tmp_path / "x.jsonl", rows)
    got = ls.load_generations(tmp_path / "x.jsonl", K, CAP)
    assert list(got) == ["q1"] and got["q1"]["correct"] == [True, False, True, False]


# ---------------------------------------------------------------- tách
def test_split_matches_a_hand_computation(tmp_path):
    (r,) = split(A, B, tmp=tmp_path)
    assert r["questions"] == 6
    assert (r["clean"]["n"], r["cut"]["n"]) == (4, 2)                          # q5 và q6 có lượt chạm trần ở A hoặc B
    assert r["clean"]["diff"] == pytest.approx(0.0625)                         # chỉ q3: 0,5 so 0,25, chia 4 câu
    assert r["cut"]["diff"] == pytest.approx(0.5)                              # q5: 0,75 so 0,25; q6: 0,5 so 0
    assert r["all"]["diff"] == pytest.approx(1.25 / 6)
    assert r["clean"]["contribution"] == pytest.approx(4 / 6 * 0.0625)
    assert r["cut"]["contribution"] == pytest.approx(2 / 6 * 0.5)
    assert r["clean"]["contribution"] + r["cut"]["contribution"] == pytest.approx(r["all"]["diff"])
    assert r["clean"]["acc_a"] == pytest.approx((1 + 0.75 + 0.5 + 0) / 4)


def test_generation_level_numbers(tmp_path):
    (r,) = split(A, B, tmp=tmp_path)
    assert r["gen_a"]["capped"] == 1 and r["gen_a"]["generations"] == 6 * K
    assert r["gen_b"]["capped"] == 5 and r["gen_b"]["capped_share"] == pytest.approx(5 / (6 * K))
    assert r["gen_b"]["capped_correct_share"] == 0.0                           # mọi lượt chạm trần đều sai
    assert r["gen_b"]["mean_tokens"] > r["gen_a"]["mean_tokens"]               # B dài hơn A


def test_a_capped_generation_that_is_still_correct_is_reported(tmp_path):
    a = {"q1": [OK] * 4, "q2": [OK] * 4}
    b = {"q1": [CUT_OK, OK, OK, OK], "q2": [CUT, OK, OK, OK]}
    (r,) = split(a, b, tmp=tmp_path)
    assert r["gen_b"]["capped"] == 2 and r["gen_b"]["capped_correct_share"] == pytest.approx(0.5)
    assert r["cut"]["n"] == 2 and r["clean"]["n"] == 0
    assert r["clean"]["diff"] is None and r["clean"]["contribution"] == 0.0     # nhóm rỗng đóng góp 0


def test_no_capped_generation_means_all_questions_are_clean(tmp_path):
    a = {"q1": [OK] * 4, "q2": [OK] * 2 + [BAD] * 2, "q3": [BAD] * 4}
    b = {"q1": [OK] * 3 + [BAD], "q2": [OK] * 2 + [BAD] * 2, "q3": [BAD] * 4}
    (r,) = split(a, b, tmp=tmp_path)
    assert (r["clean"]["n"], r["cut"]["n"]) == (3, 0) and r["cut"]["diff"] is None
    assert r["clean"]["diff"] == pytest.approx(r["all"]["diff"]) == pytest.approx(0.25 / 3)
    assert r["gen_a"]["capped_correct_share"] is None                          # không có lượt chạm trần để tính


def test_the_two_parts_always_add_up_to_the_total():
    """Với mọi cách rải lượt chạm trần và đáp án đúng, tổng hai phần đóng góp bằng Δ của mọi câu."""
    rng = random.Random(7)
    for _ in range(25):
        qs = [f"q{i}" for i in range(rng.randint(2, 30))]
        def make():
            return {q: [rng.choice([OK, BAD, CUT, CUT_OK]) if rng.random() < 0.3 else rng.choice([OK, BAD])
                        for _ in range(K)] for q in qs}
        ga, gb = ({0: {q: _conv(v) for q, v in make().items()}}, {0: {q: _conv(v) for q, v in make().items()}})
        r = ls.split_pair(ga, gb, [0], n_boot=20)
        assert r["clean"]["n"] + r["cut"]["n"] == r["all"]["n"] == len(qs)
        assert r["clean"]["contribution"] + r["cut"]["contribution"] == pytest.approx(r["all"]["diff"])


def _conv(gens):
    return {"correct": [g[0] for g in gens], "tokens": [g[1] for g in gens],
            "capped": [(not g[2]) and g[1] >= CAP for g in gens]}


def test_the_confidence_interval_is_the_one_eval_table_uses(tmp_path):
    from src.tools.eval_table import paired_bootstrap
    (r,) = split(A, B, tmp=tmp_path)
    acc = lambda g: {q: sum(x[0] for x in v) / K for q, v in g.items()}
    ref = paired_bootstrap(acc(A), acc(B), 500)
    assert (r["all"]["lo"], r["all"]["hi"]) == pytest.approx((ref["lo"], ref["hi"]))


# ---------------------------------------------------------------- nhiều seed
def test_pooling_seeds_marks_a_question_as_cut_if_any_seed_was_capped(tmp_path):
    a1 = {"q1": [OK] * 4, "q2": [OK] * 4}
    b1 = {"q1": [OK] * 4, "q2": [OK] * 4}
    b2 = {"q1": [OK] * 3 + [CUT], "q2": [OK] * 4}                              # chỉ seed 43 bị cắt ở q1
    write_run(tmp_path, "a.k3", 42, a1), write_run(tmp_path, "b.k3", 42, b1)
    write_run(tmp_path, "a.k3", 43, a1), write_run(tmp_path, "b.k3", 43, b2)
    res = ls.run_pair(find_runs(tmp_path), "a.k3", "b.k3", "math500", None, K, 50)
    assert [r["seeds"] for r in res] == [[42], [43], [42, 43]]                 # từng seed, rồi trung bình
    assert res[0]["cut"]["n"] == 0 and res[1]["cut"]["n"] == 1
    pooled = res[2]
    assert pooled["cut"]["n"] == 1 and pooled["clean"]["n"] == 1               # q1 bị cắt ở seed 43 nên cả hai seed coi là cắt
    assert pooled["all"]["diff"] == pytest.approx((0.0 + 0.125) / 2)           # q1: seed 42 hơn 0, seed 43 hơn 0,25; trung bình 0,125


def test_seeds_argument_restricts_and_a_missing_benchmark_file_is_skipped(tmp_path, capsys):
    for s in (42, 43):
        write_run(tmp_path, "a.k3", s, A), write_run(tmp_path, "b.k3", s, B)
    (tmp_path / "b.k3" / "seed43" / "math500.jsonl").unlink()
    res = ls.run_pair(find_runs(tmp_path), "a.k3", "b.k3", "math500", None, K, 50)
    assert [r["seeds"] for r in res] == [[42]] and "bỏ seed 43" in capsys.readouterr().out
    res = ls.run_pair(find_runs(tmp_path), "a.k3", "b.k3", "math500", [42], K, 50)
    assert [r["seeds"] for r in res] == [[42]]
    with pytest.raises(SystemExit, match="không có seed chung"):
        ls.run_pair(find_runs(tmp_path), "a.k3", "b.k3", "math500", [99], K, 50)


# ---------------------------------------------------------------- từ chối so những thứ không so được
def test_refuses_runs_evaluated_with_different_settings(tmp_path):
    write_run(tmp_path, "a.k3", 42, A)
    write_run(tmp_path, "b.k3", 42, B, temperature=0.9)
    with pytest.raises(SystemExit, match="cài đặt khác nhau"):
        ls.run_pair(find_runs(tmp_path), "a.k3", "b.k3", "math500", None, K, 50)


def test_unknown_model_is_named_in_the_error(tmp_path):
    write_run(tmp_path, "a.k3", 42, A)
    with pytest.raises(SystemExit, match="Không có kết quả của khong_co.k3"):
        ls.run_pair(find_runs(tmp_path), "a.k3", "khong_co.k3", "math500", None, K, 50)


def test_summary_without_the_cap_is_refused(tmp_path):
    write_run(tmp_path, "a.k3", 42, A), write_run(tmp_path, "b.k3", 42, B)
    for tag in ("a.k3", "b.k3"):
        p = tmp_path / tag / "seed42" / "summary.json"
        s = __import__("json").loads(p.read_text(encoding="utf-8"))
        s["settings"].pop("max_new_tokens")
        write_json(p, s)
    with pytest.raises(SystemExit, match="max_new_tokens"):
        ls.run_pair(find_runs(tmp_path), "a.k3", "b.k3", "math500", None, K, 50)


# ---------------------------------------------------------------- dòng lệnh
def test_main_prints_both_default_pairs_and_the_identity(tmp_path, capsys):
    for tag, g in (("fq_select.k3", A), ("quality_only.k3", B), ("correct_only.k3", A)):
        write_run(tmp_path, tag, 42, g)
    assert ls.main(["--root", str(tmp_path), "--boot", "50"]) == 0
    out = capsys.readouterr().out
    assert "fq_select.k3 so với quality_only.k3" in out and "quality_only.k3 so với correct_only.k3" in out
    assert "Câu không bị cắt" in out and "Đóng góp vào Δ tất cả" in out and "Đúng trong lượt chạm trần" in out
    assert "báo lỗi" not in out                                                # hai phần cộng đúng bằng tổng
    assert "+20.8" in out                                                      # Δ tất cả của hai phép so đầu: 1,25 / 6


def test_main_accepts_explicit_pairs_and_benchmarks(tmp_path, capsys):
    write_run(tmp_path, "a.k3", 42, A), write_run(tmp_path, "b.k3", 42, B)
    assert ls.main(["--root", str(tmp_path), "--pair", "a.k3", "b.k3", "--benchmarks", "math500", "--boot", "50"]) == 0
    assert "a.k3 so với b.k3, MATH-500" in capsys.readouterr().out


def test_main_without_results_stops_with_the_path(tmp_path):
    with pytest.raises(SystemExit, match="Không thấy summary.json"):
        ls.main(["--root", str(tmp_path)])

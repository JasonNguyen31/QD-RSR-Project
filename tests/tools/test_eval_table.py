"""Kiểm thử eval_table: bảng so sánh dựng từ summary.json và <bộ>.jsonl của c1_evaluate (dữ liệu giả, đúng tên khoá)."""
import statistics

import pytest

from src.common.config import load_config
from src.common.io_utils import write_json, write_jsonl
from src.tools import eval_table as et

CFG = load_config(student="qwen1_5b")
TIER1 = list(CFG.eval.tiers[0])
K = int(CFG.eval.n_samples)
SETTINGS = {key: CFG.eval[key] for key in ("n_samples", "temperature", "top_p", "top_k", "max_new_tokens", "seed",
                                           "stage_tokens", "gen_batch", "bbh_per_task")}


def write_run(root, tag, seed, hits, settings=None, dirname=None):
    """hits: {bộ: [số lượt đúng của từng câu]}. Ghi <bộ>.jsonl và summary.json như c1_evaluate."""
    d = root / tag / (dirname or f"seed{seed}")
    d.mkdir(parents=True)
    summary = {"tag": tag, "seed": seed, "settings": settings or SETTINGS, "eval_commit": "abc1234", "benchmarks": {}}
    for bench, per_q in hits.items():
        rows = [{"qid": f"{bench}_{i:03d}", "sample": s, "correct": s < n, "pred": "1" if s < n else None, "gold": "1",
                 "n_tokens": 100, "stopped": True, "text": "x"} for i, n in enumerate(per_q) for s in range(K)]
        write_jsonl(d / f"{bench}.jsonl", rows)
        summary["benchmarks"][bench] = {
            "acc@4": sum(per_q) / (K * len(per_q)), "pass@4": sum(n > 0 for n in per_q) / len(per_q),
            "n_questions": len(per_q), "n_generations": K * len(per_q), "mean_new_tokens": 100.0,
            "stopped_share": 1.0, "hit_cap_share": 0.0, "no_boxed_share": sum(K - n for n in per_q) / (K * len(per_q))}
    write_json(d / "summary.json", summary)
    return d


@pytest.fixture
def root(tmp_path):
    base = {b: [4, 2, 0, 1, 3, 0, 4, 2] for b in TIER1}
    better = {b: [4, 3, 1, 2, 4, 1, 4, 3] for b in TIER1}
    write_run(tmp_path, "correct_only.k3", 42, base)
    write_run(tmp_path, "correct_only.k3", 43, {b: [4, 2, 0, 1, 3, 0, 4, 4] for b in TIER1})
    write_run(tmp_path, "rsr.k3", 42, better)
    write_run(tmp_path, "qd_rsr.k3.lam0.4", 42, {TIER1[0]: better[TIER1[0]]})          # thiếu ba bộ
    write_run(tmp_path, "correct_only.k3", 42, {TIER1[0]: [4] * 8}, dirname="seed42.limit64")   # phép đo tốc độ
    return tmp_path


def test_runs_are_found_by_tag_and_seed_and_speed_runs_are_ignored(root):
    runs = et.find_runs(root)
    assert set(runs) == {"correct_only.k3", "rsr.k3", "qd_rsr.k3.lam0.4"} and sorted(runs["correct_only.k3"]) == [42, 43]
    assert runs["correct_only.k3"][42]["benchmarks"][TIER1[0]]["acc@4"] == pytest.approx(16 / 32)   # không phải bản .limit64


def test_rows_follow_the_method_order_of_the_paper_and_lambda_order():
    tags = ["qd_rsr.k3.lam1", "lark.k3", "fit_quality.k3", "qd_rsr.k3.lam0.15", "no_filter.k3", "rsr.k3", "zzz.k3"]
    assert sorted(tags, key=et.sort_key) == ["no_filter.k3", "rsr.k3", "lark.k3", "qd_rsr.k3.lam0.15", "qd_rsr.k3.lam1",
                                             "fit_quality.k3", "zzz.k3"]


def test_cells_are_mean_and_sample_std_over_seeds():
    assert et.cell([0.5]) == "50.0" and et.cell([]) == "—"
    assert et.cell([0.50, 0.54]) == f"52.0 ± {statistics.stdev([50.0, 54.0]):.1f}"
    assert et.mean_std([1.0, 3.0]) == (2.0, pytest.approx(2 ** 0.5))                   # chia n − 1


def test_tier1_average_needs_all_four_benchmarks(root):
    runs = et.find_runs(root)
    assert et.tier1_mean(runs["rsr.k3"], None, TIER1, "acc@4") == [pytest.approx(22 / 32)]
    assert et.tier1_mean(runs["qd_rsr.k3.lam0.4"], None, TIER1, "acc@4") == []
    assert len(et.tier1_mean(runs["correct_only.k3"], [43], TIER1, "acc@4")) == 1


def test_paired_bootstrap_sees_a_consistent_gain_and_not_a_noisy_one(root):
    runs = et.find_runs(root)
    b = TIER1[0]
    ref = et.pooled_question_acc(runs["correct_only.k3"], [42], b, K)
    new = et.pooled_question_acc(runs["rsr.k3"], [42], b, K)
    assert len(ref) == 8 and ref[f"{b}_000"] == 1.0 and ref[f"{b}_002"] == 0.0
    r = et.paired_bootstrap(new, ref, n_boot=500)
    assert r["n"] == 8 and r["diff"] == pytest.approx(6 / 32) and r["lo"] > 0                # 6/8 câu cùng tăng
    noisy = {q: (v + (0.25 if i % 2 else -0.25)) for i, (q, v) in enumerate(sorted(ref.items()))}
    r2 = et.paired_bootstrap(noisy, ref, n_boot=500)
    assert r2["lo"] < 0 < r2["hi"] and et.paired_bootstrap(new, ref, 500) == r               # cùng seed thì cùng kết quả
    pooled = et.pooled_question_acc(runs["correct_only.k3"], None, b, K)
    assert pooled[f"{b}_007"] == pytest.approx((2 / 4 + 4 / 4) / 2)                          # trung bình qua hai seed
    assert et.paired_bootstrap({"a": 1.0}, {"a": 0.0}) is None


def test_cli_prints_three_tables_and_flags_what_cannot_be_compared(root, capsys):
    odd = dict(SETTINGS, temperature=1.0)
    write_run(root, "lark.k3", 42, {b: [1] * 8 for b in TIER1}, settings=odd)
    assert et.main(["--root", str(root), "--versus", "correct_only.k3", "--boot", "200"]) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("| ")]
    order = [ln.split("|")[1].strip() for ln in lines if "k3" in ln][:4]
    assert order == ["correct_only.k3", "rsr.k3", "lark.k3", "qd_rsr.k3.lam0.4"]
    assert "### Acc@4" in out and "### Pass@4" in out and "### Chẩn đoán" in out and "TB tầng 1" in out
    assert "53.1 ± " in out                                                                 # hai seed của correct_only
    assert "### Chênh lệch Acc@4 so với correct_only.k3" in out and "+18.8 [" in out and " *" in out
    assert "CÀI ĐẶT ĐÁNH GIÁ KHÁC NHAU" in out and "qd_rsr.k3.lam0.4: thiếu kết quả" in out
    assert "Số seed khác nhau" in out and "Một lượt sinh đúng thêm" in out

    assert et.main(["--root", str(root), "--seeds", "43", "--format", "tsv"]) == 0
    out = capsys.readouterr().out
    assert "correct_only.k3\t" in out and "rsr.k3" not in out and "| " not in out            # chỉ mô hình có seed 43
    with pytest.raises(SystemExit, match="Không thấy summary.json"):
        et.main(["--root", str(root / "nowhere")])
    with pytest.raises(SystemExit, match="--versus"):
        et.main(["--root", str(root), "--versus", "missing.k3"])

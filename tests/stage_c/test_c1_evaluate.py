"""Kiểm thử phần không cần torch của c1_evaluate: tầng, giai đoạn sinh, chấm BBH, tóm tắt chỉ số, đọc dữ liệu.
Phần sinh (generate_staged, chạy lại, seed theo khối) đã chạy thử trên mô hình Qwen2 tí hon khi viết công cụ."""
import ast
import json
from pathlib import Path

import pytest

from src.common.config import load_config
from src.stage_c import c0_prepare_eval as c0, c1_evaluate as c1


def cfg_with(tmp_path, *extra):
    return load_config(method="correct_only", student="qwen1_5b",
                       overrides=[f"paths.data_eval={tmp_path}/eval", f"paths.data_raw={tmp_path}/raw", *extra])


@pytest.fixture()
def evaldir(tmp_path):
    """data/eval giả ghi bằng chính các hàm của c0_prepare_eval, kèm manifest và tập kiểm định."""
    (tmp_path / "eval").mkdir()
    (tmp_path / "raw").mkdir()
    math = c0.build_math500([{"problem": f"Problem {i}?", "answer": str(i), "subject": "Algebra" if i % 2 else "Geometry",
                              "unique_id": f"u{i}"} for i in range(6)])
    bbh = c0.build_bbh({"snarks": [{"input": f"s{i}", "target": "(A)"} for i in range(9)],
                        "navigate": [{"input": f"n{i}", "target": "Yes"} for i in range(7)]})
    manifest = {"benchmarks": {}}
    for name, rows in (("math500", math), ("bbh", bbh)):
        path = tmp_path / "eval" / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        manifest["benchmarks"][name] = {"file": path.name, "n": len(rows), "md5": c0.file_md5(path)}
    (tmp_path / "eval" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "raw" / "validation_matched.jsonl").write_text("".join(
        json.dumps({"qid": f"math_{i:05d}", "source": "math", "level": 4, "question": f"V{i}?", "solution": "s", "gold": str(i)}) + "\n"
        for i in range(3)), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------- cấu hình đã chốt
def test_tiers_are_the_ones_agreed_on_0310():
    """Cố ý ghi cứng: tầng 1 là bốn bộ của bài RSR; không bộ nào trong sáu bộ bị bỏ."""
    cfg = load_config(method="correct_only", student="qwen1_5b")
    assert c1.tier_benchmarks(cfg, 1) == ["math500", "amc23", "aime24", "aime25"]
    assert c1.tier_benchmarks(cfg, 2) == ["math500", "amc23", "aime24", "aime25", "gsm8k"]
    assert sorted(c1.tier_benchmarks(cfg, 3)) == sorted(cfg["eval"]["benchmarks"])
    with pytest.raises(SystemExit):
        c1.tier_benchmarks(cfg, 4)
    ev = cfg["eval"]
    assert (ev["n_samples"], ev["temperature"], ev["top_p"], ev["max_new_tokens"]) == (4, 0.6, 0.95, 3072)
    plan = c1.stage_plan(ev["stage_tokens"], ev["gen_batch"], ev["max_new_tokens"])
    assert plan[-1][0] == 3072 and [c for c, _ in plan] == sorted(c for c, _ in plan)
    assert [b for _, b in plan] == sorted((b for _, b in plan), reverse=True)        # lượt càng dài, lô càng nhỏ


def test_stage_plan_always_ends_at_the_cap():
    assert c1.stage_plan([512, 1536], [64, 32, 16], 3072) == [(512, 64), (1536, 32), (3072, 16)]
    assert c1.stage_plan([], [8], 3072) == [(3072, 8)]                                # không chia giai đoạn
    assert c1.stage_plan([4096, 512], [64, 32, 16], 3072) == [(512, 64), (3072, 32)]  # mốc vượt trần bị bỏ
    with pytest.raises(SystemExit, match="gen_batch"):
        c1.stage_plan([512, 1536], [64, 32], 3072)


def test_work_is_cut_into_fixed_chunks_in_question_order():
    qs = [{"qid": f"q{i}"} for i in range(70)]
    items = c1.work_items(qs, 4)
    assert items[:5] == [("q0", 0), ("q0", 1), ("q0", 2), ("q0", 3), ("q1", 0)] and len(items) == 280
    parts = c1.chunks(items, 256)
    assert [len(p) for p in parts] == [256, 24] and sum(parts, []) == items
    assert load_config(method="correct_only", student="qwen1_5b")["eval"]["chunk"] >= 256


# ---------------------------------------------------------------- chấm
@pytest.mark.parametrize("pred, gold, ok", [
    ("B", "(B)", True), ("(b)", "(B)", True), ("\\text{(B)}", "(B)", True), ("(C)", "(B)", False),
    ("no", "No", True), ("\\text{No}.", "No", True), ("Yes", "No", False),
    ("valid", "invalid", False), ("Invalid", "invalid", True), ("True", "False", False),
    ("] ]", "] ]", True), ("\\] \\]", "] ]", True), ("] )", "] ]", False), (") ] }", ") ] }", True),
    ("syndrome, therefrom", "syndrome therefrom", True), ("therefrom syndrome", "syndrome therefrom", False),
    ("8.0", "8", True), ("24", "24", True), (None, "Yes", False),
])
def test_bbh_answers_are_compared_after_normalisation(pred, gold, ok):
    assert c1.bbh_equal(pred, gold) is ok


def test_grade_answer_uses_the_last_boxed_for_every_benchmark():
    assert c1.grade_answer("bbh", "First \\boxed{(A)} but finally \\boxed{B}", "(B)") == (True, "B")
    assert c1.grade_answer("bbh", "The answer is (B).", "(B)") == (False, None)              # không có \boxed là sai
    assert c1.grade_answer("aime24", "so \\boxed{25}", "025") == (True, "25")
    assert c1.grade_answer("gsm8k", "thus \\boxed{\\$1,007}", "1007") == (True, "\\$1,007")
    assert c1.grade_answer("math500", "\\boxed{\\frac{1}{2}}", "0.5")[0] is True


# ---------------------------------------------------------------- tóm tắt
def result_rows(pattern):
    """pattern: qid -> 4 giá trị đúng/sai."""
    return [{"qid": q, "sample": s, "correct": c, "pred": "1" if c else None, "gold": "1",
             "n_tokens": 3072 if (q, s) == ("q0", 0) else 100, "stopped": (q, s) != ("q0", 0), "text": "t"}
            for q, v in pattern.items() for s, c in enumerate(v)]


def test_summary_reports_acc_and_pass_with_token_statistics():
    cfg = load_config(method="correct_only", student="qwen1_5b")
    qs = [{"qid": "q0", "subset": "A"}, {"qid": "q1", "subset": "A"}, {"qid": "q2", "subset": "B"}]
    rows = result_rows({"q0": [False, True, False, False], "q1": [True] * 4, "q2": [False] * 4})
    s = c1.summarise("math500", qs, rows, cfg)
    assert s["acc@4"] == pytest.approx((0.25 + 1 + 0) / 3) and s["pass@4"] == pytest.approx(2 / 3)
    assert (s["n_questions"], s["n_generations"]) == (3, 12)
    assert s["hit_cap_share"] == pytest.approx(1 / 12) and s["stopped_share"] == pytest.approx(11 / 12)
    assert s["mean_new_tokens"] == pytest.approx((3072 + 11 * 100) / 12) and s["no_boxed_share"] == pytest.approx(7 / 12)
    assert s["by_subset"]["A"]["acc@4"] == pytest.approx(0.625) and s["by_subset"]["B"]["pass@4"] == 0.0
    one = c1.summarise("amc23", [{"qid": "q1", "subset": None}], rows, cfg)                  # dòng của câu khác bị bỏ qua
    assert one["acc@4"] == 1.0 and "by_subset" not in one
    with pytest.raises(ValueError, match="chưa đủ 4 lượt"):
        c1.summarise("math500", qs, rows[:-1], cfg)


# ---------------------------------------------------------------- đọc dữ liệu
def test_questions_come_from_verified_files_and_bbh_is_sampled(evaldir):
    cfg = cfg_with(evaldir, "eval.bbh_per_task=5")
    assert [q["qid"] for q in c1.load_questions(cfg, "math500")] == [f"math500_{i:05d}" for i in range(6)]
    assert len(c1.load_questions(cfg, "math500", limit=2)) == 2
    bbh = c1.load_questions(cfg, "bbh")
    assert len(bbh) == 10 and {q["subset"] for q in bbh} == {"snarks", "navigate"}
    assert bbh == c1.load_questions(cfg, "bbh")                                              # cùng seed, cùng mẫu
    val = c1.load_questions(cfg, "val")
    assert [q["gold"] for q in val] == ["0", "1", "2"] and val[0]["benchmark"] == "val" and val[0]["subset"] == "math"
    with pytest.raises(SystemExit, match="không có bộ"):
        c1.load_questions(cfg, "aime24")
    with open(evaldir / "eval" / "math500.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(SystemExit, match="md5 khác manifest"):
        c1.load_questions(cfg, "math500")


def test_plan_prints_the_workload_without_a_gpu(evaldir, capsys):
    ov = ["--override", f"paths.data_eval={evaldir}/eval", "--override", f"paths.data_raw={evaldir}/raw",
          "--override", "eval.tiers=[[math500],[bbh]]", "--override", "eval.bbh_per_task=5"]
    assert c1.main(["--method", "correct_only", "--tier", "2", "--plan"] + ov) == 0
    out = capsys.readouterr().out
    assert "math500 6 câu, bbh 10 câu" in out and "tổng 64 lượt sinh" in out
    assert c1.main(["--method", "correct_only", "--benchmarks", "val", "--plan"] + ov) == 0
    assert "val 3 câu" in capsys.readouterr().out


def test_module_imports_no_gpu_library_at_top_level():
    tree = ast.parse(Path(c1.__file__).read_text(encoding="utf-8"))
    top = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
           for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
           for a in (n.names if isinstance(n, ast.Import) else [None])}
    assert not top & {"torch", "transformers", "peft", "datasets", "bitsandbytes"}

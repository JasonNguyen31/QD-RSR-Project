"""Chỉ kiểm tra phần dựng SVG, không cần thư viện cairo, nên chạy được ở máy nào cũng được."""
import xml.etree.ElementTree as ET

import pytest

from src.tools import make_figures as mf


def _stats(**groups):
    return {"min_correct": 3, "per_group": {
        k: {"questions": sum(v), "correct_count_hist": {str(i): n for i, n in enumerate(v)}}
        for k, v in groups.items()}}


def test_pipeline_is_valid_svg_with_all_boxes():
    svg = mf.build_pipeline()
    ET.fromstring(svg)                                   # cú pháp XML hợp lệ
    for word in ("Stage A:", "Stage B:", "Stage C:", "Fit(t, m)", "exact search", "5,685 samples", ">LoRA<",
                 "next method", "next student", "filtered candidates", "cached Qual(t)"):
        assert word in svg, word
    assert "QLoRA" not in svg        # 1,5B dùng LoRA 16-bit, chỉ 7B dùng QLoRA (chốt 02/10); xem phần Thiết lập


def test_distribution_uses_shares_not_counts():
    """Nhóm 100 câu và nhóm 20 câu có cùng hình dạng phải cho cùng tỷ lệ."""
    big = [0] * 9 + [100]
    small = [0] * 9 + [20]
    series = mf.distribution_series(_stats(gsm8k=big, **{"math-L5": small}))
    (_, _, n1, v1), (_, _, n2, v2) = series
    assert (n1, n2) == (100, 20) and v1 == v2 and v1[-1] == pytest.approx(100.0)


def test_distribution_orders_groups_easy_to_hard_and_skips_missing():
    series = mf.distribution_series(_stats(**{"math-L5": [1] * 10, "gsm8k": [1] * 10}))
    assert [s[0] for s in series] == ["GSM8K", "MATH level 5"]


def test_distribution_svg_marks_dropped_region():
    svg = mf.build_distribution(_stats(gsm8k=[0, 1, 0, 2, 0, 0, 0, 1, 4, 92]))
    ET.fromstring(svg)
    assert "fewer than 3 correct" in svg and "(n = 100)" in svg


def test_distribution_rejects_empty_stats():
    with pytest.raises(SystemExit):
        mf.distribution_series({"per_group": {}})


def test_selection_not_ready_yet():
    with pytest.raises(SystemExit) as e:
        mf.build_selection()
    assert "Giai đoạn B" in str(e.value)


# ---------------- sáu hình cho báo cáo (Notion mục T4)
def _stage_a(tmp_path):
    """Dựng một bộ dữ liệu nhỏ đủ để vẽ: 2 câu × 3 mô hình dạy × 2 mẫu."""
    import json

    from src.common.io_utils import write_json, write_jsonl

    wd = tmp_path / "stage_a"
    wd.mkdir()
    qs = [{"qid": "gsm8k_1", "source": "gsm8k", "question": "?", "gold": "1"},
          {"qid": "math_1", "source": "math", "level": 5, "question": "?", "gold": "1"}]
    write_jsonl(wd / "questions.jsonl", qs)
    labels, trajs, quality = [], [], []
    for q in qs:
        for teacher in ("deepseek", "llama70b", "qwen72b"):
            for i in range(2):
                tid = f"{q['qid']}|{teacher}|{i}"
                ok = not (teacher == "deepseek" and i == 1 and q["qid"] == "math_1")
                labels.append({"tid": tid, "qid": q["qid"], "teacher": teacher, "correct": ok,
                               "pred": "1", "reason": "match" if ok else "truncated"})
                trajs.append({"tid": tid, "qid": q["qid"], "teacher": teacher, "text": "x",
                              "completion_tokens": 400 + i * 50, "prompt_id": "rsr"})
                if ok:
                    quality.append({"tid": tid, "qid": q["qid"], "rule_score": 0.1 * i,
                                    "llm_score": 0.9 if i else 1.0, "qual": 0.5, "n_words": 50 + i})
    write_jsonl(wd / "labels.jsonl", labels)
    write_jsonl(wd / "trajectories.deepseek.jsonl", [t for t in trajs if t["teacher"] == "deepseek"])
    write_jsonl(wd / "trajectories.llama70b.jsonl", [t for t in trajs if t["teacher"] == "llama70b"])
    write_jsonl(wd / "trajectories.qwen72b.jsonl", [t for t in trajs if t["teacher"] == "qwen72b"])
    write_jsonl(wd / "quality.jsonl", quality)
    write_json(wd / "embed_probe.json", {
        "within_teacher": {"median": 0.308}, "across_teacher": {"median": 0.465},
        "by_group": {"gsm8k": {"median": 0.348}, "math-L5": {"median": 0.460}}})
    # Đúng tên khoá b1_fit ghi từ 29/09: brier và mean_surprisal, không có lark (ĝ tính theo câu hỏi).
    write_jsonl(wd / "fit.test.jsonl", [
        {"tid": f"math_1|deepseek|{i}", "qid": "math_1", "rsr": 5.0 + i, "grape": -1.0 - i,
         "local_nat": -1.1 - i, "brier": 0.3 + 0.05 * i, "mean_surprisal": 1.0 + i, "n_tokens": 100 + i,
         "n_steps": 5, "signals_version": 2} for i in range(4)])
    write_jsonl(wd / "fit.v1.jsonl", [
        {"tid": f"math_1|deepseek|{i}", "rsr": 5.0 + i, "grape": -1.0 - i, "local_nat": -1.1 - i,
         "lark": 0.1 * i, "n_tokens": 100 + i} for i in range(4)])
    return wd


@pytest.mark.parametrize("fn,needle", [
    ("build_teachers", "Tỷ lệ chuỗi đúng"),
    ("build_truncation", "Tỷ lệ chuỗi bị cắt"),
    ("build_lengths", "Độ dài chuỗi đúng"),
    ("build_judge", "Phân bố điểm giám khảo"),
    ("build_distance", "Khoảng cách biểu diễn"),
])
def test_report_figures_render_from_real_files(tmp_path, fn, needle):
    from src.common.config import load_config
    from src.tools import make_figures as mf

    svg = getattr(mf, fn)(load_config(), str(_stage_a(tmp_path)))
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert needle in svg


def test_correlation_figure_renders(tmp_path):
    from src.common.config import load_config
    from src.tools import make_figures as mf

    svg = mf.build_correlation(load_config(), str(_stage_a(tmp_path)), "fit.test.jsonl")
    assert "Tương quan hạng" in svg and "RSR" in svg and "LARK" in svg and "LocalNat" in svg


def test_correlation_figure_refuses_a_file_with_old_signal_definitions(tmp_path):
    from src.common.config import load_config
    from src.tools import make_figures as mf

    with pytest.raises(SystemExit):
        mf.build_correlation(load_config(), str(_stage_a(tmp_path)), "fit.v1.jsonl")


def test_truncation_figure_shows_the_teacher_that_was_cut(tmp_path):
    """Trong bộ giả chỉ DeepSeek bị cắt, nên cột của nó phải khác 0 còn hai mô hình kia bằng 0."""
    from src.common.config import load_config
    from src.tools import make_figures as mf

    svg = mf.build_truncation(load_config(), str(_stage_a(tmp_path)))
    assert ">25<" in svg or ">50<" in svg          # DeepSeek bị cắt 1/2 chuỗi ở math-L5


def test_report_group_lists_every_report_figure():
    from src.tools.make_figures import FIGURES, REPORT

    assert set(REPORT) <= set(FIGURES) and len(REPORT) == 6


def test_distance_figure_reads_the_keys_a5_actually_writes(tmp_path):
    """Hồi quy: hình từng đọc nhầm tên khoá nên hai cột đầu ra 0 mà không báo lỗi."""
    from src.common.config import load_config
    from src.tools import make_figures as mf

    svg = mf.build_distance(load_config(), str(_stage_a(tmp_path)))
    assert ">0,308<" in svg and ">0,465<" in svg


def test_distance_figure_refuses_probe_without_teacher_split(tmp_path):
    import json
    from src.common.config import load_config
    from src.tools import make_figures as mf

    wd = _stage_a(tmp_path)
    (wd / "embed_probe.json").write_text(json.dumps({"by_group": {"gsm8k": {"median": 0.3}}}))
    with pytest.raises(SystemExit):
        mf.build_distance(load_config(), str(wd))


# ---------------------------------------------------------------- hình kết quả (thêm 12/10)
def _eval_run(root, tag, seed, correct):
    """Một lần đánh giá giả: 4 câu x 4 lượt trên math500; `correct[q]` là số lượt đúng của câu q."""
    import json
    d = root / tag / f"seed{seed}"
    d.mkdir(parents=True)
    rows = [{"qid": f"q{q}", "sample": s, "correct": s < c} for q, c in enumerate(correct) for s in range(4)]
    (d / "math500.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    acc = sum(correct) / (4 * len(correct))
    (d / "summary.json").write_text(json.dumps({"settings": {"n_samples": 4},
                                                "benchmarks": {"math500": {"acc@4": acc, "pass@4": 1.0}}}),
                                    encoding="utf-8")


def _select(seldir, tag, qwen):
    import json
    seldir.mkdir(parents=True, exist_ok=True)
    (seldir / f"select.{tag}.json").write_text(json.dumps(
        {"tag": tag, "teacher_share": {"qwen72b": qwen, "deepseek": 1 - qwen, "llama70b": 0.0}}), encoding="utf-8")


def _results_fixture(tmp_path):
    res, sel = tmp_path / "eval", tmp_path / "stage_b"
    for sd in (42, 43):
        _eval_run(res, "correct_only.k3", sd, [2, 2, 2, 2])
        _eval_run(res, "grape.k3", sd, [4, 4, 2, 2])
    _eval_run(res, "teacher_qwen_last.k3", 42, [0, 0, 2, 2])
    _eval_run(res, "unknown_set.k3", 42, [4, 4, 4, 4])         # không có trong RESULT_SETS thì không vẽ
    _select(sel, "correct_only.k3", 0.287)
    _select(sel, "grape.k3", 0.753)
    _select(sel, "teacher_qwen_last.k3", 0.004)
    return res, sel


def test_result_rows_read_acc_share_and_paired_difference(tmp_path):
    res, sel = _results_fixture(tmp_path)
    rows = {r["tag"]: r for r in mf.result_rows(res, sel, n_boot=200)}
    assert set(rows) == {"correct_only.k3", "grape.k3", "teacher_qwen_last.k3"}
    g, t, b = rows["grape.k3"], rows["teacher_qwen_last.k3"], rows["correct_only.k3"]
    assert (g["seeds"], t["seeds"]) == (2, 1)
    assert g["acc"] == pytest.approx(75.0) and g["share"] == pytest.approx(75.3)
    assert g["diff"] == pytest.approx(25.0) and g["lo"] <= g["diff"] <= g["hi"]
    assert t["diff"] == pytest.approx(-25.0) and t["group"] == "teacher"
    assert b["diff"] is None


def test_result_rows_respect_seed_filter(tmp_path):
    res, sel = _results_fixture(tmp_path)
    rows = {r["tag"]: r for r in mf.result_rows(res, sel, seeds=[43], n_boot=50)}
    assert "teacher_qwen_last.k3" not in rows and rows["grape.k3"]["seeds"] == 1


def test_result_rows_refuse_missing_select_file(tmp_path):
    res, sel = _results_fixture(tmp_path)
    (sel / "select.grape.k3.json").unlink()
    with pytest.raises(SystemExit, match="select.grape.k3.json"):
        mf.result_rows(res, sel, n_boot=50)


def test_spearman_matches_known_values():
    assert mf.spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert mf.spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert mf.spearman_rho([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833, abs=1e-6)


@pytest.mark.parametrize("lang,needles", [("en", ["Prefer Qwen-72B", "mean of 2+ seeds", "teacher-identity control"]),
                                          ("vi", ["Ưu tiên Qwen-72B", "trung bình từ 2 seed", "mức Correct-Only"])])
def test_teacher_share_figure_uses_circles_only(lang, needles):
    rows = [{"tag": "correct_only.k3", "en": "Correct-Only", "vi": "Correct-Only", "group": "free", "seeds": 2,
             "acc": 47.6, "share": 28.7},
            {"tag": "teacher_qwen_first.k3", "en": "Prefer Qwen-72B", "vi": "Ưu tiên Qwen-72B", "group": "teacher",
             "seeds": 1, "acc": 54.9, "share": 80.3},
            {"tag": "fit_quality.k3", "en": "QD-RSR (λ = 0)", "vi": "QD-RSR (λ = 0)", "group": "student", "seeds": 2,
             "acc": 45.5, "share": 5.8}]
    svg = mf.build_teacher_share(rows, lang)
    ET.fromstring(svg)
    for w in needles:
        assert w in svg, w
    assert "<polygon" not in svg and "<path" not in svg   # Jason 12/10: chỉ chấm tròn, không hình thoi
    assert svg.count('fill="#ffffff" stroke="#eb6834"') == 1  # tập 1 seed là chấm rỗng


def test_forest_figure_explains_symbols_and_signs():
    rows = [{"tag": "grape.k3", "vi": "GRAPE", "group": "student", "seeds": 2, "diff": 6.8, "lo": 5.1, "hi": 8.6},
            {"tag": "rsr.k3", "vi": "RSR", "group": "student", "seeds": 2, "diff": -2.0, "lo": -3.5, "hi": -0.4}]
    svg = mf.build_forest(rows)
    ET.fromstring(svg)
    for w in ("+6,8", "−2,0", "khoảng tin cậy 95%", "không cắt vạch 0"):
        assert w in svg, w


def test_results_group_lists_every_result_figure():
    assert set(mf.RESULTS) <= set(mf.FIGURES)
    assert mf.FIGURES["teacher_share"][0] == "fig3_teacher_share"     # tên file trong main.tex

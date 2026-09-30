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
    for word in ("Stage A:", "Stage B:", "Stage C:", "Fit(t, m)", "exact search", "5,685 samples", "QLoRA",
                 "next method", "next student", "filtered candidates", "cached Qual(t)"):
        assert word in svg, word


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

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
    for word in ("Stage A:", "Stage B:", "Stage C:", "Fit(t, m)", "exact search", "5,670 samples", "QLoRA",
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

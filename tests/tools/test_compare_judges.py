import pytest

from src.tools.compare_judges import pair_agreement, rankdata, spearman, top1_agreement


def test_rankdata_handles_ties():
    assert rankdata([10, 20, 30]) == [1.0, 2.0, 3.0]
    assert rankdata([5, 5, 9]) == [1.5, 1.5, 3.0]


def test_spearman_edges():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2], [1, 2]) is None            # dưới 3 điểm thì không tính
    assert spearman([1, 1, 1], [1, 2, 3]) is None      # một bên hằng số


def test_pair_agreement_counts_ties_separately():
    by_q = {"q1": [(0.9, 0.8), (0.5, 0.4), (0.7, 0.9)], "q2": [(0.5, 0.5), (0.6, 0.5)]}
    r = pair_agreement(by_q)
    # q1: (0.9,0.5)->cùng chiều, (0.9,0.7)->cùng chiều, (0.5,0.7)->ngược chiều. q2: B hoà nên bỏ
    assert r["same"] == 2 and r["diff"] == 1 and r["tie_b"] == 1
    assert r["rate"] == pytest.approx(2 / 3)


def test_top1_agreement_allows_tie_at_top():
    by_q = {"q1": [(0.9, 0.7), (0.5, 0.4)],          # cùng chọn chuỗi đầu
            "q2": [(0.9, 0.3), (0.5, 0.8)],          # chọn khác nhau
            "q3": [(0.9, 0.6), (0.5, 0.6)],          # B hoà, chuỗi A chọn vẫn cao nhất theo B
            "q4": [(0.9, 0.7)]}                       # chỉ một chuỗi, bỏ qua
    r = top1_agreement(by_q)
    assert r["questions"] == 3 and r["hit"] == 2 and r["rate"] == pytest.approx(2 / 3)


def test_self_preference_table(tmp_path, capsys):
    """Giám khảo cùng họ nâng điểm mô hình dạy cùng họ; bảng phải cho thấy điều đó."""
    from src.common.io_utils import write_jsonl
    from src.tools.compare_judges import report_self_preference

    tids = [f"q{i}|{t}|0" for i in range(6) for t in ("qwen72b", "llama70b")]
    fair = [{"tid": t, "overall_score": 0.7, "model": "gemini"} for t in tids]
    biased = [{"tid": t, "overall_score": 0.95 if "qwen" in t else 0.7, "model": "qwen-judge"} for t in tids]
    write_jsonl(tmp_path / "a.jsonl", fair)
    write_jsonl(tmp_path / "b.jsonl", biased)
    report_self_preference(tmp_path, ["a.jsonl", "b.jsonl"])
    out = capsys.readouterr().out
    assert "gemini" in out and "qwen-judge" in out
    assert "0.950" in out and "0.700" in out


def test_self_preference_needs_enough_overlap(tmp_path, capsys):
    from src.common.io_utils import write_jsonl
    from src.tools.compare_judges import report_self_preference

    rows = [{"tid": "q1|qwen72b|0", "overall_score": 0.8, "model": "m"}]
    write_jsonl(tmp_path / "a.jsonl", rows)
    write_jsonl(tmp_path / "b.jsonl", rows)
    report_self_preference(tmp_path, ["a.jsonl", "b.jsonl"])
    assert "chưa đủ để xét" in capsys.readouterr().out

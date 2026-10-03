"""Kiểm thử phần không cần torch của diagnose_stop: tìm \\boxed, kiểu kết thúc chuỗi, đổi vị trí ký tự sang vị trí
token, đường xác suất dừng, và các dòng báo cáo. Phần cần mô hình (hàng nhúng, chấm xác suất, sinh thử) đã được
chạy thử trên một mô hình Qwen2 tí hon với transformers 5.5.0 và peft 0.21.0 khi viết công cụ."""
import ast
import math
from pathlib import Path

import pytest

from src.tools import diagnose_stop as ds


# ---------------------------------------------------------------- tìm \boxed và phân loại cách kết thúc
def test_boxed_spans_match_nested_braces_and_skip_unclosed_ones():
    text = "so \\boxed{\\frac{1}{2}} then \\boxed {x^{2}} and \\boxed 5 and \\boxed{\\{a\\}} end"
    spans = ds.boxed_spans(text)
    assert [c for _s, _e, c in spans] == ["\\frac{1}{2}", "x^{2}", "\\{a\\}"]
    for s, e, _c in spans:
        assert text[s:].startswith("\\boxed") and text[e - 1] == "}"
    assert ds.boxed_spans("no answer here") == []
    assert [c for _s, _e, c in ds.boxed_spans("\\boxed{3} and then \\boxed{4 + \\frac{1}{")] == ["3"]   # cái sau bị cắt


@pytest.mark.parametrize("text, kind", [
    ("Thus \\boxed{5}", "answer_last"),
    ("Thus $\\boxed{5}$.", "answer_last"),
    ("\\[\n\\boxed{5}\n\\]\n", "answer_last"),
    ("\\begin{align*} x &= \\boxed{5} \\end{align*}", "answer_last"),
    ("**Answer:** \\boxed{5}**", "answer_last"),
    ("\\boxed{5}. This is the final answer because the sum is odd.", "text_after"),
    ("\\boxed{4} is wrong, so the answer is \\boxed{5}\n\nLet me verify again", "text_after"),
    ("I could not finish", "no_boxed"),
])
def test_ending_kind_separates_answer_last_from_text_after_the_answer(text, kind):
    assert ds.ending_kind(text) == kind
    assert kind in ds.KIND_LABEL


# ---------------------------------------------------------------- vị trí ký tự -> vị trí token
def test_token_at_char_is_the_shortest_prefix_that_covers_the_position():
    pieces = ["So ", "the ", "answer ", "is ", "\\boxed{", "12", "}", " and", " more"]
    text = "".join(pieces)
    prefix_len = lambda m: len("".join(pieces[:m]))
    end = ds.boxed_spans(text)[0][1]
    j = ds.token_at_char(prefix_len, len(pieces), end)
    assert j == 7 and "".join(pieces[:j]).endswith("\\boxed{12}")
    assert ds.token_at_char(prefix_len, len(pieces), 0) == 0
    assert ds.token_at_char(prefix_len, len(pieces), len(text)) == len(pieces)
    assert ds.token_at_char(prefix_len, len(pieces), 1) == 1               # giữa token đầu thì phải lấy cả token


# ---------------------------------------------------------------- đường xác suất dừng
def test_stop_curve_sums_the_tracked_tokens_and_starts_at_the_first_new_token():
    lp = [[math.log(0.5), math.log(0.1)]] * 3 + [[math.log(0.2), math.log(0.05)], [math.log(0.01), math.log(0.01)],
                                                [math.log(0.7), math.log(0.2)]]
    curve = ds.stop_curve(lp, n_prompt=4)          # 4 token đề, 2 token mới: 3 vị trí kể cả sau token cuối
    assert curve == pytest.approx([0.25, 0.02, 0.9])


def test_exit_profile_splits_before_exit_window_and_after():
    curve = [0.01, 0.02, 0.6, 0.3, 0.05, 0.2, 0.15]
    p = ds.exit_profile(curve, j_exit=2, window=1)
    assert p == {"before": 0.02, "at_exit": 0.6, "after": 0.2, "after_above_0.1": 2}
    assert ds.exit_profile(curve, None) == {"before": 0.6, "at_exit": None, "after": None, "after_above_0.1": None}
    assert ds.exit_profile(curve, j_exit=6, window=40)["after"] is None      # \boxed ở sát cuối: không còn phần sau


def test_repeat_share_is_zero_for_distinct_lines_and_high_for_loops():
    assert ds.repeat_share("line one is here\nline two is here\nline three is here\nline four is here") == 0.0
    assert ds.repeat_share("\n".join(["The answer is 5 indeed."] * 9 + ["Something else entirely."])) == pytest.approx(0.8)
    assert ds.repeat_share("short\nshort\nshort\nshort\nshort") == 0.0       # dòng quá ngắn không tính


def test_describe_and_pick_samples_are_deterministic():
    d = ds.describe([0.1, 0.9, 0.5, 0.7])
    assert d["n"] == 4 and d["mean"] == pytest.approx(0.55) and d["median"] == pytest.approx(0.6)
    assert d["share_ge_half"] == 0.75 and ds.describe([])["mean"] is None
    rows = [{"tid": f"q{i}|deepseek|0"} for i in range(50)]
    a = ds.pick_samples(rows, 10, 42)
    assert a == ds.pick_samples(list(reversed(rows)), 10, 42) and len(a) == 10
    assert a != ds.pick_samples(rows, 10, 43)


# ---------------------------------------------------------------- mô tả một lượt sinh và các dòng báo cáo
def _generation(text_tokens, stop, curve, qid="math_00001"):
    """Dựng bản ghi của một lượt sinh đúng như generation_report dựng (mỗi phần tử của text_tokens là một token)."""
    text = "".join(text_tokens)
    spans = ds.boxed_spans(text)
    j = ds.token_at_char(lambda m: len("".join(text_tokens[:m])), len(text_tokens), spans[0][1]) if spans else None
    return {"qid": qid, "stop_token": stop, **ds.describe_generation(text_tokens, stop, text, curve, j),
            "special_tokens_inside": {}, "curve_above_0.01": [], "question": "Question?", "text": text}


def test_generation_record_reports_answers_and_what_follows_the_first_boxed():
    toks = ["x ", "\\boxed{", "4", "}", " but ", "wait ", "\\boxed{", "5", "}", " again ", "\\boxed{", "5", "}"]
    r = _generation(toks, None, [0.0] * 4 + [0.3] + [0.02] * 9)
    assert (r["stopped"], r["n_new"], r["n_boxed"], r["distinct_answers"]) == (False, 13, 3, 2)
    assert r["first_equals_last"] is False and r["tokens_after_first_boxed"] == 9
    assert r["after_first_boxed"].startswith(" but wait") and r["p_stop"]["at_exit"] == 0.3
    done = _generation(["x ", "\\boxed{", "4", "}"], "<|im_end|>", [0.0, 0.0, 0.0, 0.0, 0.9])
    assert done["stopped"] and done["tokens_after_first_boxed"] == 0 and done["p_stop_final"] == 0.9
    none = _generation(["no ", "answer"], None, [0.1, 0.2, 0.05])
    assert none["n_boxed"] == 0 and none["first_equals_last"] is None and none["p_stop"]["before"] == 0.2


def test_generation_lines_separate_stopped_from_running_turns():
    recs = [_generation(["x ", "\\boxed{", "4", "}"], "<|im_end|>", [0.0, 0.0, 0.0, 0.0, 0.9], "q_stop"),
            _generation(["x ", "\\boxed{", "4", "}", " but ", "\\boxed{", "5", "}"] + [" more"] * 50, None,
                        [0.0] * 4 + [0.3] + [0.02] * 54, "q_run"),
            _generation(["no ", "answer"], None, [0.1, 0.2, 0.05], "q_nobox")]
    lines, summary = ds.generation_lines(recs, {"n": 16, "stopped": 7}, 3072)
    text = "\n".join(lines)
    assert "dừng 1/3 (run.json ghi 7/16)" in text and "<|im_end|> ×1" in text
    assert summary["stopped"] == 1 and summary["stopped_by"] == {"<|im_end|>": 1}
    assert summary["loose_with_boxed"] == 1 and summary["loose_first_differs_from_last"] == 1
    assert summary["stopped_within_exit_window"] == 1
    assert "q_run" in text and "q_nobox" in text and "KHÔNG có \\boxed" in text
    assert not any(ln.startswith("  --- ") and "q_stop" in ln for ln in lines)      # lượt đã dừng không in chi tiết


def test_chain_end_lines_summarise_base_against_tuned():
    def rec(teacher, kind, base_p, tuned_p, top1):
        side = lambda p, first: {"p": {"<|im_end|>": p, "<|endoftext|>": 0.01}, "rank": 1 if first else 3,
                                 "top": [["<|im_end|>" if first else "\n\n", max(p, 0.4)]], "eot_is_top1": first,
                                 "sampled_stop": p, "max_stop_inside": 0.001}
        return {"tid": "t", "teacher": teacher, "kind": kind, "n_labels": 100,
                "base": side(base_p, False), "tuned": side(tuned_p, top1)}
    recs = [rec("deepseek", "answer_last", 0.0, 0.9, True), rec("deepseek", "answer_last", 0.0, 0.7, True),
            rec("llama70b", "text_after", 0.0, 0.2, False), rec("qwen72b", "text_after", 0.0, 0.2, False)]
    lines, summary = ds.chain_end_lines(recs, "<|im_end|>", ["<|endoftext|>"], (0.6, 0.95))
    assert summary["base"]["p_eot"]["mean"] == 0.0 and summary["tuned"]["p_eot"]["mean"] == pytest.approx(0.5)
    assert summary["tuned"]["eot_top1_share"] == 0.5 and summary["rivals"] == [("\n\n", 2)]
    text = "\n".join(lines)
    assert "đáp án ở cuối (2): p 0.800" in text and "còn lời sau đáp án (2): p 0.200" in text
    assert "deepseek (2)" in text and "T=0.6, top-p 0.95" in text
    assert ds.chain_end_lines([], "<|im_end|>", [], (0.6, 0.95))[1] == {}


def test_embedding_lines_print_every_named_token_for_tied_and_untied_models():
    tokens = {"<|im_end|>": {"id": 151645, "norm": 0.4, "norm_percentile_among_ordinary": 0.0,
                             "cos_ordinary_mean": 0.9, "cos_other_added_mean": 0.99, "cos_unused_mean": 0.99},
              "<|endoftext|>": {"id": 151643, "norm": 1.2, "norm_percentile_among_ordinary": 0.6,
                                "cos_ordinary_mean": 0.1, "cos_other_added_mean": 0.2, "cos_unused_mean": 0.2}}
    side = {"groups": {"ordinary": {"rows": 151643, "median_norm": 1.1}, "other_added": {"rows": 20, "median_norm": 0.4},
                       "unused": {"rows": 271, "median_norm": 0.4}},
            "tokens": tokens, "cos_between": {"<|im_end|> ~ <|endoftext|>": 0.2}}
    tied = ds.embedding_lines({"tied_input_output": True, "matrix_rows": 151936, "tokenizer_size": 151665, "output": side})
    assert "DÙNG CHUNG" in tied[0] and sum("<|im_end|>" in ln for ln in tied) == 2 and len(tied) == 6
    both = ds.embedding_lines({"tied_input_output": False, "matrix_rows": 152064, "tokenizer_size": 151665,
                               "output": side, "input": side})
    assert "hai ma trận riêng" in both[0] and len(both) == 11
    no_unused = {**side, "groups": {**side["groups"], "unused": {"rows": 0, "median_norm": None}}}
    assert ds.embedding_lines({"tied_input_output": True, "matrix_rows": 9, "tokenizer_size": 9, "output": no_unused})


def test_module_imports_no_gpu_library_at_top_level():
    tree = ast.parse(Path(ds.__file__).read_text(encoding="utf-8"))
    top = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
           for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
           for a in (n.names if isinstance(n, ast.Import) else [None])}
    assert not top & {"torch", "transformers", "peft", "safetensors", "bitsandbytes"}


def test_chain_end_lines_work_when_the_end_token_is_the_only_stop_token():
    """Sau 03/10 token kết thúc là eos_token, nên không còn token dừng thứ hai để in cột riêng."""
    side = lambda p: {"p": {"<|endoftext|>": p}, "rank": 1, "top": [["<|endoftext|>", p]], "eot_is_top1": True,
                      "sampled_stop": 1.0, "max_stop_inside": 0.01}
    recs = [{"tid": "t", "teacher": "deepseek", "kind": "answer_last", "n_labels": 50, "base": side(0.96),
             "tuned": side(0.99)}] * 3
    lines, summary = ds.chain_end_lines(recs, "<|endoftext|>", [], (0.6, 0.95))
    assert summary["tuned"]["p_eot"]["mean"] == pytest.approx(0.99) and summary["tuned"]["p_other_mean"] == {}
    assert summary["rivals"] == [] and not any("không đứng đầu, token" in ln for ln in lines)
    assert any("bản nền" in ln and "0.960" in ln for ln in lines)

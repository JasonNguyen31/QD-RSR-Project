"""Kiểm thử src/stage_b/steps: các bước cuối cùng của LocalNat dựng từ kết quả cắt bước của GLM-4.5-Air.

Không gọi mạng, không cần torch. Phản hồi hỏng được dựng tay theo đúng các kiểu làm json.loads hỏng: dấu nháy
không thoát, xuống dòng thật trong chuỗi, bị cắt giữa chừng.
"""
import json

import pytest

from src.stage_b import steps as st
from src.stage_b.signals import token_steps
from src.tools import segment_steps as ss

TEXT = ("We solve it in three steps.\n\n"
        "1. First compute 2 + 3 = 5.\n"
        "2. Then double it to get 10.\n\n"
        "- Check: 10 / 2 = 5, as needed.\n"
        "**Step 3:** conclude from the check.\n"
        "So x = 10. Therefore the answer is \\boxed{10}.")


def at(fragment: str) -> int:
    return TEXT.index(fragment)


def parts(text, ends):
    return [text[a:b] for a, b in zip([0] + list(ends[:-1]), ends)]


def glm_row(text, ends, status="ok", **extra):
    return {"tid": "q|t|0", "status": status, "ends": list(ends), "text_md5": st.text_md5(text), **extra}


RAW_ENDS = [at("First"), at("Then"), at("Check"), at("conclude"), at("Therefore"), len(TEXT)]


# ---------------------------------------------------------------- lùi ranh giới về đầu dòng
@pytest.mark.parametrize("fragment, kind", [("First", "marker"), ("Then", "marker"), ("Check", "marker"),
                                            ("conclude", "marker"), ("Therefore", "mid"), ("1. First", "line"),
                                            ("- Check", "line"), ("So x", "line")])
def test_cut_kinds(fragment, kind):
    assert st.cut_kind(TEXT, at(fragment)) == kind


@pytest.mark.parametrize("line", ["(a) Start", "a) Start", "iv. Start", "### Start", "> Start", "• Start",
                                  "**2.** Start", "Case 2: Start", "  3) Start", "**Step 1**: Start"])
def test_many_marker_shapes_are_recognised(line):
    text = "Intro sentence here.\n" + line + " of the step."
    assert st.cut_kind(text, text.index("Start")) == "marker"
    ends, moves = st.snap_to_line_start(text, [text.index("Start"), len(text)])
    assert ends == [text.index("\n") + 1, len(text)] and len(moves) == 1


@pytest.mark.parametrize("text, fragment", [("x = 5. Therefore y = 2.", "Therefore"),
                                            ("Intro.\nSince 12. is odd we stop.", "is odd"),
                                            ("Intro.\nThus the answer follows.", "the answer")])
def test_a_boundary_after_real_words_is_not_moved(text, fragment):
    cut = text.index(fragment)
    assert st.cut_kind(text, cut) == "mid"
    assert st.snap_to_line_start(text, [cut, len(text)]) == ([cut, len(text)], [])


def test_boundaries_after_list_markers_move_to_the_line_start():
    ends, moves = st.snap_to_line_start(TEXT, RAW_ENDS)
    got = parts(TEXT, ends)
    assert "".join(got) == TEXT and len(got) == len(RAW_ENDS)             # vẫn phủ kín, không mất bước nào
    assert got[1].startswith("1. First") and got[2].startswith("2. Then") and got[3].startswith("- Check")
    assert got[4].startswith("**Step 3:** conclude") and got[5].startswith("Therefore")   # ranh giới giữa dòng giữ nguyên
    assert not got[0].rstrip().endswith("1.")                              # dấu đầu dòng không còn bị bỏ lại ở bước trước
    assert [m["marker"] for m in moves] == ["1.", "2.", "-", "**Step 3:**"] and not any(m["dropped"] for m in moves)


def test_marker_only_step_and_marker_at_text_start_are_merged_not_left_empty():
    text = "1. Start here.\n2. Go on.\n3. Finish."
    # mô hình tách "2." thành một bước riêng: ranh giới ở đầu dòng 2 và ngay sau dấu "2."
    ends, moves = st.snap_to_line_start(text, [text.index("Start"), text.index("2."), text.index("Go"), len(text)])
    assert parts(text, ends) == ["1. Start here.\n", "2. Go on.\n3. Finish."]
    assert [m["dropped"] for m in moves] == [True, True]                   # dấu ở đầu văn bản, và bước chỉ có dấu


def test_mid_line_boundary_moves_before_the_space_so_the_first_word_stays_in_its_step():
    """Token của Qwen mang dấu cách ở đầu và thuộc bước chứa ký tự đầu của nó: ranh giới phải đứng TRƯỚC dấu cách."""
    text = "So x = 10.  Therefore y = 5.\n\tIndented line here.\nPlain line."
    raw = [text.index("Therefore"), text.index("Indented"), text.index("Plain"), len(text)]
    ends, moved = st.before_spaces(text, raw)
    assert parts(text, ends) == ["So x = 10.", "  Therefore y = 5.\n", "\tIndented line here.\n", "Plain line."]
    assert moved == 2                                                      # ranh giới ở đầu dòng không đổi
    # tokenizer giả kiểu Qwen: dấu xuống dòng là token riêng, chữ mang dấu cách hoặc tab đứng trước nó
    spaced = [(m.start(), m.end()) for m in __import__("re").finditer(r"\n+|[ \t]*[^\s]+", text)]
    first = [text[spaced[a][0]:spaced[a][1]].strip() for a, _b in token_steps(spaced, text, ends=ends)]
    assert first == ["So", "Therefore", "Indented", "Plain"]
    assert [text[spaced[a][0]:spaced[a][1]].strip() for a, _b in token_steps(spaced, text, ends=raw)][1] != "Therefore"
    assert st.before_spaces("  Lead in. Next.", [2, 11, 16]) == ([10, 16], 2)   # dời về đầu văn bản thì bỏ


def test_tiny_steps_merge_forward_and_a_tiny_tail_merges_back():
    text = "Ok.\nFirst real step here.\n\\]\nSecond real step.\n.."
    raw = [text.index("First"), text.index("\\]"), text.index("Second"), text.index(".."), len(text)]
    ends, merged = st.merge_tiny(text, raw)
    assert parts(text, ends) == ["Ok.\nFirst real step here.\n", "\\]\nSecond real step.\n.."] and merged == 3
    assert st.merge_tiny("ab", [1, 2]) == ([2], 1)


# ---------------------------------------------------------------- cứu phản hồi mà json.loads không đọc nổi
SOL = ('He said "wait" and paused.\nThen \\frac{1}{2} + \\frac{1}{2} = 1.\n\n'
       "So the total is 1.\nThe answer is \\boxed{1}.")
GROUPS = [['He said "wait" and paused.', "Then \\frac{1}{2} + \\frac{1}{2} = 1."],
          ["So the total is 1.", "The answer is \\boxed{1}."]]


def good_json(groups=GROUPS):
    return json.dumps({"sentence_groups": {f"group{i + 1}": g for i, g in enumerate(groups)}}, indent=2)


def test_unescaped_quotes_break_json_but_not_the_salvage():
    raw = good_json().replace('\\"wait\\"', '"wait"')                      # mô hình quên thoát dấu nháy trong câu
    assert ss.parse_groups(raw) is None
    s = st.salvage(SOL, raw)
    assert s["accepted"] and s["unlocated"] == 0 and parts(SOL, s["ends"])[1].startswith("So the total")


def test_raw_newline_inside_a_string_and_single_backslash_latex():
    raw = good_json().replace("\\\\", "\\").replace("paused.\",\n", "paused.\n\",\n")
    s = st.salvage(SOL, raw)
    assert s["accepted"] and len(s["ends"]) == 2


def test_truncated_response_is_read_but_refused_when_it_covers_too_little():
    raw = good_json()
    cut_late = raw[:raw.index("boxed") + 8]                                # cụt ở gần cuối: mọi nhóm vẫn có mặt
    assert ss.parse_groups(cut_late) is None and st.salvage(SOL, cut_late)["accepted"]
    long_sol = SOL + "\n\n" + " ".join(f"Extra sentence number {i} goes here." for i in range(12))
    s = st.salvage(long_sol, raw)                                          # phản hồi chỉ phủ nửa đầu của chuỗi
    assert s["length_ratio"] < st.SALVAGE_RATIO[0] and not s["accepted"]


def test_rewritten_text_and_missing_group_keys_are_not_salvaged():
    rewritten = good_json([GROUPS[0], ["A completely different closing paragraph that was never written."]])
    assert not st.salvage(SOL, rewritten.replace('\\"', '"'))["accepted"]
    assert st.salvage_groups("I cannot split this solution.") is None and st.salvage(SOL, None) is None


# ---------------------------------------------------------------- cắt dự phòng
def test_fallback_prefers_paragraphs_then_lines_then_sentences():
    paras = "Step one is here.\nStill step one.\n\nStep two is here.\n\n\nStep three."
    ends, rule = st.fallback_ends(paras)
    assert rule == "paragraph" and parts(paras, ends) == ["Step one is here.\nStill step one.\n\n",
                                                           "Step two is here.\n\n\n", "Step three."]
    lines = "Line one of it.\nLine two of it.\nLine three."
    assert st.fallback_ends(lines) == ([16, 32, len(lines)], "line")
    one_line = "First we add. Then we stop."
    ends, rule = st.fallback_ends(one_line)
    assert rule == "sentence" and len(ends) == 2
    assert st.fallback_ends("Just one sentence without a break") == ([33], "sentence")
    with pytest.raises(ValueError):
        st.rule_ends(paras, "tokens")


# ---------------------------------------------------------------- final_steps
def test_final_steps_for_each_kind_of_row():
    ok = st.final_steps(TEXT, glm_row(TEXT, RAW_ENDS))
    assert ok["source"] == "glm" and len(ok["moves"]) == 4 and parts(TEXT, ok["ends"])[1].startswith("1. First")
    assert ok["spaces_moved"] == 1 and parts(TEXT, ok["ends"])[-1] == " Therefore the answer is \\boxed{10}."
    partial = st.final_steps(TEXT, glm_row(TEXT, RAW_ENDS, status="partial"))
    assert partial["ends"] == ok["ends"]

    raw = good_json().replace('\\"wait\\"', '"wait"')
    saved = st.final_steps(SOL, {"tid": "x", "status": "json_error", "text_md5": st.text_md5(SOL), "raw": raw})
    assert saved["source"] == "glm_salvaged" and saved["salvage"]["accepted"] and len(saved["ends"]) == 2

    no_raw = st.final_steps(SOL, {"tid": "x", "status": "truncated", "text_md5": st.text_md5(SOL)})
    assert no_raw["source"] == "fallback_paragraph" and no_raw["salvage"] is None and len(no_raw["ends"]) == 2
    bad_raw = st.final_steps(SOL, {"tid": "x", "status": "json_error", "text_md5": st.text_md5(SOL), "raw": "nothing"})
    assert bad_raw["source"] == "fallback_paragraph"
    assert st.final_steps(SOL, {"tid": "x", "status": "api_error"})["source"] == "fallback_paragraph"
    assert st.final_steps(SOL, None)["source"] == "fallback_paragraph"     # chưa cắt: người gọi tự quyết có dùng không
    broken_ends = st.final_steps(SOL, glm_row(SOL, [10, 5, len(SOL)]))     # dòng ok nhưng mốc hỏng: coi như cắt hỏng
    assert broken_ends["source"] == "fallback_paragraph"


def test_final_steps_refuses_a_text_that_changed_since_segmentation():
    with pytest.raises(ValueError, match="text_md5"):
        st.final_steps(TEXT + " extra", glm_row(TEXT, RAW_ENDS))
    with pytest.raises(ValueError):
        st.final_steps("", None)


@pytest.mark.parametrize("text, row", [(TEXT, glm_row(TEXT, RAW_ENDS)), (SOL, None),
                                       (TEXT, glm_row(TEXT, [at("First"), len(TEXT)]))])
def test_final_ends_always_partition_the_text_and_feed_token_steps(text, row):
    f = st.final_steps(text, row)
    assert st.valid_ends(f["ends"], len(text)) and "".join(parts(text, f["ends"])) == text
    offsets = [(i, i + 1) for i in range(len(text))]                       # tokenizer giả: mỗi ký tự một token
    assert token_steps(offsets, text, ends=f["ends"]) == list(zip([0] + f["ends"][:-1], f["ends"]))


def test_processing_is_deterministic_and_never_edits_the_log_row():
    row = glm_row(TEXT, RAW_ENDS)
    before = json.dumps(row, sort_keys=True)
    assert st.final_steps(TEXT, row) == st.final_steps(TEXT, row)
    assert json.dumps(row, sort_keys=True) == before


def test_current_rows_keeps_the_last_line_of_each_chain():
    rows = [{"tid": "a", "status": "json_error"}, {"tid": "b", "status": "ok"}, {"tid": "a", "status": "ok"}]
    assert st.current_rows(rows)["a"]["status"] == "ok" and len(st.current_rows(rows)) == 2


def test_context_share_tells_how_much_of_the_past_a_step_sees():
    lengths = [100, 100, 100, 100, 100, 100]
    assert st.context_share(lengths, 1) == pytest.approx([1.0, 0.5, 1 / 3, 0.25, 0.2])
    assert st.context_share(lengths, 4) == pytest.approx([1.0, 1.0, 1.0, 1.0, 0.8])       # gần như toàn bộ: sát GRAPE
    assert st.context_share([50], 1) == []

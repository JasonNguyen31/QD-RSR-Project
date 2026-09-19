from src.tools.check_grader import compare_extraction, compare_matching


def row(tid, pred, gold, ok, text=None):
    r = {"tid": tid, "pred_answer": pred, "gold_answer": gold, "is_correct": ok}
    if text is not None:
        r["trajectory"] = text
    return r


def test_matching_agreement_and_disagreement():
    rows = [
        row("t1", "9", "9", True),
        row("t2", "8", "9", False),
        row("t3", r"\frac{1}{2}", "0.5", True),     # dạng khác nhau, bộ mới vẫn cho khớp
        row("t4", "x+1", "1+x", True),              # bộ mới KHÔNG có sympy nên coi là lệch
        row("t5", None, "9", False),                # cũ không tách được
        row("t6", "9", None, None),                 # thiếu trường
    ]
    tally, diffs = compare_matching(rows)
    assert tally["khớp"] == 3 and tally["LỆCH"] == 1
    assert tally["bộ chấm cũ không tách được"] == 1 and tally["thiếu trường"] == 1
    assert diffs[0]["tid"] == "t4" and diffs[0]["cũ"] is True and diffs[0]["mới"] is False


def test_extraction_counts():
    rows = [
        row("a", "9", "9", True, r"vậy $\boxed{9}$."),
        row("b", "9", "9", True, "vậy đáp án là #### 9"),      # prompt cũ, không có boxed
        row("c", "7", "7", True, r"$\boxed{8}$"),              # lệch thật
        row("d", "7", "7", True, r"\boxed{"),                  # mới không tách được
    ]
    tally, diffs = compare_extraction(rows)
    assert tally["khớp"] == 1 and tally["chuỗi không có \\boxed"] == 1
    assert tally["LỆCH"] == 1 and tally["mới không tách được"] == 1
    assert {d["tid"] for d in diffs} == {"c", "d"}

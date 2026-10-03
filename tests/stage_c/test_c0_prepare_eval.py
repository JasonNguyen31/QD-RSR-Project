"""Kiểm thử phần không cần mạng của c0_prepare_eval. Dữ liệu giả mang đúng tên cột của từng nguồn thật:
GSM8K (question, answer), MATH-500 (problem, answer, subject, unique_id), file amc23 và aime24 của kho Qwen2.5-Math
(id, problem, question, answer, url), và BBH (examples: input, target)."""
import ast
import json
from pathlib import Path

import pytest

from src.common.config import load_config
from src.stage_c import c0_prepare_eval as c0


def aime_rows(n=6, style="hf"):
    rows = []
    for i in range(n):
        part = "I" if i < n // 2 else "II"
        text = f"Problem {i}: let $x_{i}$ satisfy a condition number {i * 7 + 3}; find the remainder when it is divided by 1000."
        if style == "hf":      # kiểu Maxwell-Jia/AIME_2024: tên cột viết hoa, đáp án là số
            rows.append({"ID": f"2024-{part}-{i + 1}", "Problem": text, "Solution": "...", "Answer": 20 + i})
        else:                  # kiểu kho Qwen2.5-Math: đáp án là chuỗi ba chữ số
            rows.append({"id": 60 + i, "problem": text, "question": text, "solution": "...", "answer": f"{20 + i:03d}",
                         "url": f"https://artofproblemsolving.com/wiki/index.php/2024_AIME_{part}_Problems/Problem_{i + 1}"})
    return rows


# ---------------------------------------------------------------- nguồn đã chốt
def test_sources_are_the_ones_decided_on_0310():
    """Cố ý ghi cứng: đổi nguồn hay số câu là đổi bộ đánh giá, phải sửa cả trang Cấu hình đã chốt."""
    src = load_config()["eval_data"]["sources"]
    assert set(src) == set(c0.BENCHMARKS) == set(load_config()["eval"]["benchmarks"])
    assert {b: src[b]["n"] for b in src} == {"gsm8k": 1319, "math500": 500, "amc23": 40, "aime24": 30, "aime25": 30, "bbh": 6511}
    assert (src["gsm8k"]["repo"], src["gsm8k"]["config"], src["gsm8k"]["split"]) == ("openai/gsm8k", "main", "test")
    assert src["math500"]["repo"] == "HuggingFaceH4/MATH-500" and src["aime24"]["repo"] == "Maxwell-Jia/AIME_2024"
    assert src["aime25"]["configs"] == ["AIME2025-I", "AIME2025-II"]
    for b in ("amc23", "bbh"):                                   # nguồn GitHub phải ghim commit đầy đủ và md5
        assert len(src[b]["commit"]) == 40 and len(src[b]["md5"]) == 32
    assert len(src["aime24"]["cross_check"]["commit"]) == 40 and len(c0.BBH_TASKS) == 27 == len(set(c0.BBH_TASKS))


# ---------------------------------------------------------------- dựng dòng chuẩn
def test_pick_ignores_case_and_names_the_columns_it_saw():
    assert c0.pick({"Problem": "p", "Answer": 3}, ["problem", "question"]) == "p"
    assert c0.pick({"question": "q"}, ["problem", "question"]) == "q"
    with pytest.raises(KeyError, match="de_bai"):
        c0.pick({"de_bai": "x"}, ["problem", "question"])


def test_gold_is_cleaned_without_losing_information():
    assert c0.clean_gold(27.0) == "27" and c0.clean_gold(-1.0) == "-1" and c0.clean_gold(2.5) == "2.5"
    assert c0.clean_gold("025") == "025" and c0.clean_gold(204) == "204" and c0.clean_gold(" (A) ") == "(A)"


def test_gsm8k_qids_cannot_collide_with_training_qids():
    rows = c0.build_gsm8k([{"question": "How many?", "answer": "3 + 4 = <<3+4=7>>7\n#### 1,007"}])
    assert rows == [{"qid": "gsm8k_test_00000", "benchmark": "gsm8k", "subset": None, "question": "How many?",
                     "gold": "1007", "source_id": "0"}]


def test_math500_amc_and_bbh_rows_keep_subset_and_source():
    m = c0.build_math500([{"problem": "P?", "solution": "s", "answer": "\\frac{1}{2}", "subject": "Algebra", "level": 3,
                           "unique_id": "test/algebra/1.json"}])
    assert (m[0]["qid"], m[0]["subset"], m[0]["gold"], m[0]["source_id"]) == ("math500_00000", "Algebra", "\\frac{1}{2}", "test/algebra/1.json")
    a = c0.build_amc23([{"id": 0, "problem": "P?", "question": "P?", "answer": 27.0,
                         "url": "https://artofproblemsolving.com/wiki/index.php/2023_AMC_12B_Problems/Problem_1"}])
    assert (a[0]["subset"], a[0]["gold"]) == ("12B", "27") and not c0.check_rows("amc23", a, 1)
    b = c0.build_bbh({"snarks": [{"input": "Which is sarcastic?\nOptions:\n(A) x\n(B) y", "target": "(B)"}],
                      "navigate": [{"input": "Turn left.", "target": "No"}, {"input": "Turn right.", "target": "Yes"}]})
    assert [r["qid"] for r in b] == ["bbh_navigate_0000", "bbh_navigate_0001", "bbh_snarks_0000"]
    assert b[2]["gold"] == "(B)" and b[2]["subset"] == "snarks" and "Options:" in b[2]["question"]


def test_aime_rows_from_both_column_styles_and_from_split_configs():
    hf = c0.build_aime("aime24", aime_rows(6, "hf"))
    assert [r["subset"] for r in hf] == ["I"] * 3 + ["II"] * 3 and hf[0]["gold"] == "20" and hf[0]["source_id"] == "2024-I-1"
    qw = c0.build_aime("aime24_ref", aime_rows(6, "qwen"))
    assert qw[0]["gold"] == "020" and qw[0]["subset"] is None            # mã câu là số, không đọc được đề I hay II
    split = c0.build_aime("aime25", [{"question": "Q1", "answer": "5"}, {"question": "Q2", "answer": "6"}], ["I", "II"])
    assert [r["subset"] for r in split] == ["I", "II"] and split[1]["qid"] == "aime25_00001"
    assert c0.aime_part("AIME2025-II") == "II" and c0.aime_part("2024_AIME_I_Problems") == "I" and c0.aime_part(60) is None


# ---------------------------------------------------------------- các phép kiểm
def test_check_rows_reports_counts_duplicates_and_bad_golds():
    rows = c0.build_aime("aime24", aime_rows(6, "hf"))
    assert c0.check_rows("aime24", rows, 6) == []
    assert "đã ghim 30" in c0.check_rows("aime24", rows, 30)[0]
    dup = rows + [{**rows[0], "qid": "aime24_00099"}]
    assert any("lặp lại" in e for e in c0.check_rows("aime24", dup, 7))
    assert any("không phải số nguyên" in e for e in c0.check_rows("aime24", [{**rows[0], "gold": "\\frac{1}{2}"}], 1))
    assert any("ngoài khoảng" in e for e in c0.check_rows("aime24", [{**rows[0], "gold": "1000"}], 1))
    assert any("thiếu đề hoặc thiếu đáp án" in e for e in c0.check_rows("gsm8k", [{**rows[0], "gold": None}], 1))
    bbh = c0.build_bbh({"t": [{"input": "same", "target": "Yes"}, {"input": "same", "target": "No"}]})
    assert c0.check_rows("bbh", bbh, 2) == [] and "1 câu lặp sẵn" in c0.describe_rows("bbh", bbh)   # BBH gốc có câu lặp


def test_cross_check_matches_shuffled_sources_and_stops_on_a_different_answer():
    primary = c0.build_aime("aime24", list(reversed(aime_rows(6, "hf"))))
    ref = c0.build_aime("aime24_ref", aime_rows(6, "qwen"))
    ok = c0.cross_check(primary, ref)
    assert ok == {"errors": [], "notes": [], "matched": 6}                # '020' và '20' là một đáp án
    fixed = [dict(r) for r in primary]
    fixed[0]["question"] = fixed[0]["question"].replace("remainder", "largest possible remainder over all valid cases")
    noted = c0.cross_check(fixed, ref)
    assert noted["errors"] == [] and [n["qid"] for n in noted["notes"]] == [fixed[0]["qid"]]   # đề được sửa: chỉ ghi chú
    note = noted["notes"][0]
    assert note["primary_only"] == "over all valid cases" and note["reference_only"] == ""   # đoạn chèn dài nhất
    assert note["primary_chars"] > note["reference_chars"] and note["source_id"] == fixed[0]["source_id"]
    wrong = [dict(r) for r in primary]
    wrong[2]["gold"] = "999"
    assert any("đáp án '999'" in e for e in c0.cross_check(wrong, ref)["errors"])
    assert any("nguồn đối chiếu có 5" in e for e in c0.cross_check(primary, ref[:5])["errors"])
    alien = primary[:1] + [{**primary[1], "question": "zzzz qqqq 9999 completely unrelated"}]
    assert any("không tìm thấy câu tương ứng" in e for e in c0.cross_check(alien, ref)["errors"])


def test_leak_check_uses_the_same_normalisation_as_a1_prepare():
    rows = {"gsm8k": c0.build_gsm8k([{"question": "Tom has  3 apples.\nHow many?", "answer": "#### 3"},
                                     {"question": "Another question?", "answer": "#### 4"}])}
    protected = {"questions.jsonl": {c0.norm("tom has 3 apples. how many?")}, "validation.jsonl": {c0.norm("unrelated")}}
    assert c0.leak_check(rows, protected) == {"questions.jsonl": ["gsm8k_test_00000"]}
    assert c0.leak_check(rows, {"validation.jsonl": {c0.norm("unrelated")}}) == {}


def test_bbh_sample_is_fixed_by_seed_and_takes_small_tasks_whole():
    files = {"big": [{"input": f"q{i}", "target": "Yes"} for i in range(200)],
             "small": [{"input": f"s{i}", "target": "No"} for i in range(30)]}
    rows = c0.build_bbh(files)
    s = c0.sample_per_subset(rows, 50, 42)
    assert len(s) == 80 and sum(r["subset"] == "small" for r in s) == 30
    assert s == c0.sample_per_subset(list(rows), 50, 42) and [r["qid"] for r in s] == sorted(r["qid"] for r in s)
    assert {r["qid"] for r in s} != {r["qid"] for r in c0.sample_per_subset(rows, 50, 43)}
    assert {r["qid"] for r in c0.sample_per_subset(rows, 25, 42)} <= {r["qid"] for r in s}      # mẫu nhỏ nằm trong mẫu lớn


def test_verify_detects_missing_and_modified_files(tmp_path):
    assert "không thấy" in c0.verify(tmp_path)[0]
    f = tmp_path / "amc23.jsonl"
    f.write_text('{"qid": "amc23_00000"}\n', encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"benchmarks": {
        "amc23": {"file": "amc23.jsonl", "n": 1, "md5": c0.file_md5(f)},
        "bbh": {"file": "bbh.jsonl", "n": 2, "md5": "0" * 32}}}), encoding="utf-8")
    assert c0.verify(tmp_path) == ["bbh: thiếu file bbh.jsonl"]
    f.write_text('{"qid": "amc23_00001"}\n', encoding="utf-8")
    assert any("amc23: md5" in e for e in c0.verify(tmp_path))
    assert c0.combined_md5({"b": "2", "a": "1"}) == c0.combined_md5({"a": "1", "b": "2"}) != c0.combined_md5({"a": "2", "b": "1"})


def test_module_imports_no_network_library_at_top_level():
    tree = ast.parse(Path(c0.__file__).read_text(encoding="utf-8"))
    top = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
           for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
           for a in (n.names if isinstance(n, ast.Import) else [None])}
    assert not top & {"datasets", "huggingface_hub", "torch", "transformers"}


def test_aime_gold_drops_the_degree_sign_and_records_the_original():
    """opencompass/AIME2025 ghi đáp án câu II-5 là '336^\\circ'; đáp án của đề thi là số nguyên 336."""
    assert c0.aime_gold("336^\\circ") == "336" and c0.aime_gold("336^{\\circ}") == "336" and c0.aime_gold("336°") == "336"
    assert c0.aime_gold("$104$") == "104" and c0.aime_gold("025") == "025" and c0.aime_gold(204) == "204"
    rows = c0.build_aime("aime25", [{"question": "Q1", "answer": "468"}, {"question": "Find the arcs in degrees.", "answer": "336^\\circ"}],
                         ["II", "II"])
    assert "gold_raw" not in rows[0] and (rows[1]["gold"], rows[1]["gold_raw"]) == ("336", "336^\\circ")
    assert c0.check_rows("aime25", rows, 2) == []


def test_diff_excerpt_shows_where_two_versions_of_a_problem_differ():
    a = "Find the number of rectangles inside a regular dodecagon. [asy] draw(dir(120)--dir(330)); [/asy]"
    b = "Find the number of rectangles inside a regular dodecagon."
    assert c0.diff_excerpt(a, b) == ("[asy] draw(dir(120)--dir(330)); [/asy]", "")
    assert c0.diff_excerpt(b, b) == ("", "")


def test_broken_latex_commands_are_restored_only_when_unambiguous():
    """Maxwell-Jia/AIME_2024 câu II-8 ghi '$' + tab + 'frac{m}{n}$' thay cho '$\\\\tfrac{m}{n}$' (lỗi thoát ký tự ở nguồn)."""
    fixed, cmds = c0.repair_latex("written as $\tfrac{m}{n}$, and $a\times b$ with $\x0crac{1}{2}$ and $\x08inom{4}{2}$")
    assert fixed == "written as $\\tfrac{m}{n}$, and $a\\times b$ with $\\frac{1}{2}$ and $\\binom{4}{2}$"
    assert sorted(cmds) == ["\\binom", "\\frac", "\\tfrac", "\\times"]
    assert c0.repair_latex("x \neq y and $\rho$; an intact $\\rho$ stays") == ("x \\neq y and $\\rho$; an intact $\\rho$ stays", ["\\rho", "\\neq"])
    legit = "It produces three\tdresses for every four.\nA.\t67.332\nB.\t67.473"       # tab thật trong lời văn của MATH
    assert c0.repair_latex(legit) == (legit, []) and c0.control_chars(legit) == 3
    assert c0.repair_latex("the\ttotal is\ntwo\nequal parts") == ("the\ttotal is\ntwo\nequal parts", [])
    rows = c0.build_aime("aime24", [{"ID": "2024-II-8", "Problem": "Torus $T$ ... written as $\tfrac{m}{n}$. Find $m+n$.", "Answer": 127}])
    assert rows[0]["question"].endswith("written as $\\tfrac{m}{n}$. Find $m+n$.") and rows[0]["latex_fixed"] == ["\\tfrac"]
    assert c0.check_rows("aime24", rows, 1) == []
    stuck = c0.build_aime("aime24", [{"ID": "2024-I-1", "Problem": "A tab\tthat is not a command.", "Answer": 5}])
    assert "latex_fixed" not in stuck[0] and any("ký tự điều khiển" in e for e in c0.check_rows("aime24", stuck, 1))
    math = c0.build_math500([{"problem": legit, "answer": "67.473", "subject": "Prealgebra", "unique_id": "u"}])
    assert math[0]["question"] == legit and c0.check_rows("math500", math, 1) == []      # MATH-500 giữ nguyên như nguồn

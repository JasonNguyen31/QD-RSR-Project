"""Kiểm thử audit_steps: công cụ đếm xem src/stage_b/steps làm gì với file cắt bước, trước khi chấm trên GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_b"))

from src.common.io_utils import read_jsonl, write_jsonl
from src.stage_b import steps as st
from src.tools import audit_steps as au
from test_b2_select import make_workdir

BODY = ("We proceed in steps.\n\n1. First we add the numbers together.\n2. Then we double the sum we found.\n\n"
        "So the result is ten. Therefore the answer is \\boxed{%d}.")


def build(tmp):
    """Kho giả mà mỗi chuỗi có nhiều dòng. Chuỗi 0: json_error không có phản hồi thô; chuỗi 1: chưa cắt; còn lại
    được cắt ở ba chỗ, hai chỗ đứng ngay sau dấu đầu dòng và một chỗ ở giữa dòng."""
    make_workdir(tmp, n_q=4)
    cands = read_jsonl(tmp / "candidates.jsonl")
    for i, c in enumerate(cands):
        c["text"] = BODY % i
    write_jsonl(tmp / "candidates.jsonl", cands)
    rows = []
    for i, c in enumerate(cands):
        text = c["text"]
        base = {"tid": c["tid"], "qid": c["qid"], "text_md5": st.text_md5(text), "n_chars": len(text)}
        if i == 0:
            rows.append({**base, "status": "json_error", "finish_reason": "stop", "completion_tokens": 50})
        elif i > 1:
            ends = [text.index("First"), text.index("Then"), text.index("Therefore"), len(text)]
            rows.append({**base, "status": "ok", "ends": ends, "n_steps": 4})
    write_jsonl(tmp / "steps.glm.jsonl", rows)
    return cands, rows


def test_analyse_counts_boundaries_sources_and_context(tmp_path):
    cands, rows = build(tmp_path)
    a = au.analyse(cands, st.current_rows(rows), k_select=3)
    n_ok = len(cands) - 2
    assert a["with_row"] == len(cands) - 1 and a["done"] == n_ok and a["status"] == {"ok": n_ok, "json_error": 1}
    assert a["kinds"] == {"marker": 2 * n_ok, "mid": n_ok} and a["moves"] == 2 * n_ok and a["moved_chains"] == n_ok
    assert a["markers"] == {"1.": n_ok, "2.": n_ok} and a["dropped"] == 0 and a["spaces"] == n_ok
    assert a["source"] == {"glm": n_ok, "fallback_paragraph": 1} and a["no_raw"] == 1 and len(a["broken"]) == 1
    assert set(a["before"]) == {4} and a["after"].count(4) == n_ok        # lùi ranh giới không làm mất bước nào
    assert a["rules"]["paragraph"]["glm"] == 3 * n_ok and 0 < a["rules"]["line"]["hit"] <= a["rules"]["line"]["glm"]
    assert a["ctx"][4]["full"] == a["ctx"][4]["steps"] > a["ctx"][1]["full"]   # 4 bước: k = 4 luôn thấy toàn bộ phần trước


def test_broken_chain_in_a_question_with_exactly_k_candidates_is_reported(tmp_path):
    cands, rows = build(tmp_path)
    qid = cands[0]["qid"]
    assert au.analyse(cands, st.current_rows(rows), 3, per_q={qid: 3})["broken_in_exact_k"] == 1
    assert au.analyse(cands, st.current_rows(rows), 3, per_q={qid: 5})["broken_in_exact_k"] == 0


def test_cli_prints_the_five_sections_and_writes_nothing(tmp_path, capsys):
    build(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert au.main(["--workdir", str(tmp_path), "--show", "2", "--list-broken"]) == 0
    out = capsys.readouterr().out
    for piece in ("1. Ranh giới", "2. Số bước", "3. Chuỗi cắt hỏng: 1", "4. Quy tắc cắt dự phòng", "5. Ngữ cảnh",
                  "chưa cắt 1", "LÙI về đầu dòng", "không có gì để cứu", "trước: ", "sau:   "):
        assert piece in out, piece
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    assert au.main(["--workdir", str(tmp_path), "--pilot", "5"]) == 0
    assert "5 chuỗi dùng được" in capsys.readouterr().out

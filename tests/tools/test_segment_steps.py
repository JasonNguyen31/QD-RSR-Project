"""Kiểm thử công cụ cắt bước bằng GLM-4.5-Air. Không gọi mạng: mô hình được thay bằng bản giả trả về đúng các
kiểu phản hồi hay gặp (JSON sạch, LaTeX hỏng dấu gạch chéo, bọc trong khối mã, bỏ sót câu, bị cắt)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_b"))

from src.common.api import ChatResult
from src.common.io_utils import read_jsonl
from src.stage_b.signals import token_steps
from src.tools import segment_steps as ss
from test_b2_select import make_workdir

TEXT = ("First, compute \\frac{1}{2} + \\frac{1}{3}. We need a common denominator.\n"
        "This gives \\frac{5}{6}. \\[ \\text{check: } 3 + 2 = 5 \\]\n\n"
        "Therefore the answer is \\boxed{\\frac{5}{6}}.")
G1 = ["First, compute \\frac{1}{2} + \\frac{1}{3}.", "We need a common denominator."]
G2 = ["This gives \\frac{5}{6}.", "\\[ \\text{check: } 3 + 2 = 5 \\]"]
G3 = ["Therefore the answer is \\boxed{\\frac{5}{6}}."]


def clean(groups):
    """Phản hồi JSON đúng chuẩn (json.dumps tự nhân đôi dấu gạch chéo)."""
    return json.dumps({"sentence_groups": {f"group{i + 1}": g for i, g in enumerate(groups)}})


def broken(groups):
    """Mô hình chép LaTeX với MỘT dấu gạch chéo: \\frac, \\text, \\boxed thành lối thoát JSON sai nghĩa."""
    return clean(groups).replace("\\\\", "\\")


def steps_of(text, ends):
    return [text[a:b] for a, b in zip([0] + ends[:-1], ends)]


# ---------------------------------------------------------------- đọc JSON
def test_clean_json_and_fenced_json_give_the_same_groups():
    a = ss.parse_groups(clean([G1, G2, G3]))
    b = ss.parse_groups("Here is the split:\n```json\n" + clean([G1, G2, G3]) + "\n```")
    assert a["standard"] == [G1, G2, G3] == b["standard"]


def test_single_backslash_latex_is_recovered_by_the_literal_reading():
    p = ss.parse_groups(broken([G1, G2, G3]))
    assert p["literal"] == [G1, G2, G3]                         # đọc theo kiểu ký tự thường: nguyên văn
    assert p["standard"] is not None and p["standard"] != [G1, G2, G3]   # đọc chuẩn: \f, \t, \b thành ký tự điều khiển
    assert "\x0c" in p["standard"][0][0]


def test_correctly_escaped_pair_is_not_double_fixed():
    """Hai gạch chéo liền nhau rồi một ký tự không phải lối thoát, như \\\\( : không được sửa thành lỗi."""
    raw = json.dumps({"sentence_groups": {"group1": ["Let \\( x = 2 \\)."]}})
    assert ss.parse_groups(raw)["standard"] == [["Let \\( x = 2 \\)."]]
    assert ss.parse_groups(raw)["literal"] == [["Let \\( x = 2 \\)."]]


def test_other_shapes_and_garbage():
    assert ss.parse_groups(json.dumps({"sentence_groups": [G1, G2]}))["standard"] == [G1, G2]
    assert ss.parse_groups(json.dumps({"sentence_groups": {"group1": "One sentence."}}))["standard"] == [["One sentence."]]
    assert ss.parse_groups('{"sentence_groups": {"group1": ["cut off in the mid') is None     # bị cắt
    assert ss.parse_groups("I cannot do that.") is None


# ---------------------------------------------------------------- áp ranh giới lên văn bản gốc
@pytest.mark.parametrize("make", [clean, broken])
def test_steps_partition_the_original_text_at_group_starts(make):
    a = ss.align(TEXT, ss.parse_groups(make([G1, G2, G3])))
    assert a["ends"][-1] == len(TEXT) and a["ends"] == sorted(set(a["ends"])) and a["n_steps"] == 3
    s1, s2, s3 = steps_of(TEXT, a["ends"])
    assert s1.startswith("First, compute") and s2.startswith("This gives") and s3.startswith("Therefore")
    assert s1 + s2 + s3 == TEXT                                   # chấm trên văn bản GỐC, không mất ký tự nào
    assert "\\]" in s2                                            # khối công thức nằm trọn trong bước của nó
    assert a["unlocated"] == 0 and a["verbatim_share"] == 1.0 and a["length_ratio"] == pytest.approx(1.0)


def test_leading_latex_delimiter_stays_with_its_step():
    text = "We start. \\[ x = 1 \\] is given.\n$y = 2$ follows from it."
    groups = [["We start."], ["\\[ x = 1 \\] is given."], ["$y = 2$ follows from it."]]
    a = ss.align(text, ss.parse_groups(clean(groups)))
    s = steps_of(text, a["ends"])
    assert s[1].startswith("\\[") and s[2].startswith("$y")


def test_skipped_sentence_is_still_covered_and_shows_in_the_metrics():
    dropped = [G1[:1], G2, G3]                                    # mô hình bỏ câu "We need a common denominator."
    a = ss.align(TEXT, ss.parse_groups(clean(dropped)))
    assert "".join(steps_of(TEXT, a["ends"])) == TEXT and a["n_steps"] == 3
    assert "common denominator" in steps_of(TEXT, a["ends"])[0]   # câu bị bỏ vẫn thuộc bước đầu
    assert a["length_ratio"] < 0.9 and a["verbatim_share"] == 1.0


def test_rewritten_group_is_merged_and_counted():
    rewritten = [G1, ["Completely different words that are not in the solution at all."], G3]
    a = ss.align(TEXT, ss.parse_groups(clean(rewritten)))
    assert a["unlocated"] == 1 and a["n_steps"] == 2 and a["verbatim_share"] < 1.0
    assert "".join(steps_of(TEXT, a["ends"])) == TEXT


def test_slightly_damaged_group_start_is_found_by_fuzzy_match():
    damaged = [G1, ["Ths gives \\frac{5}{6}.", G2[1]], G3]        # mô hình gõ sai một chữ ở đầu nhóm
    a = ss.align(TEXT, ss.parse_groups(clean(damaged)))
    assert a["unlocated"] == 0 and a["n_steps"] == 3
    assert steps_of(TEXT, a["ends"])[1].lstrip().startswith("This gives")


def test_external_step_ends_feed_token_steps():
    a = ss.align(TEXT, ss.parse_groups(clean([G1, G2, G3])))
    offsets = [(i, i + 1) for i in range(len(TEXT))]              # tokenizer giả: mỗi ký tự một token
    steps = token_steps(offsets, TEXT, ends=a["ends"])
    assert steps == list(zip([0] + a["ends"][:-1], a["ends"]))
    assert len(token_steps(offsets, TEXT)) > len(steps)           # cắt theo dấu câu cho nhiều bước hơn
    for bad in ([], [5, 5, len(TEXT)], [10, 4, len(TEXT)], [3, len(TEXT) - 1]):
        with pytest.raises(ValueError):
            token_steps(offsets, TEXT, ends=bad)


# ---------------------------------------------------------------- lời nhắc
def test_placeholder_prompt_file_is_refused(tmp_path):
    from src.common.config import load_config
    cfg = load_config()
    with pytest.raises(SystemExit, match="chưa có lời nhắc"):
        ss.load_prompt(Path(cfg.root) / cfg.segmenter.prompt_file)     # file trong repo chỉ là chỗ giữ chỗ
    p = tmp_path / "p.txt"
    p.write_text("split {solution} only", encoding="utf-8")
    with pytest.raises(SystemExit):
        ss.load_prompt(p)                                              # thiếu {problem}
    p.write_text('problem: {problem}\nsolution: {solution}\nE.g. {"sentence_groups": {"group1": []}}', encoding="utf-8")
    out = ss.build_prompt(ss.load_prompt(p), "What is {x}?", "It is {y}.")
    assert "What is {x}?" in out and "It is {y}." in out and '{"sentence_groups"' in out   # ngoặc nhọn không bị format


# ---------------------------------------------------------------- chạy trọn vẹn với mô hình giả
class FakeClient:
    """Cắt mỗi chuỗi thành hai nhóm tại dấu cách ở giữa. Chuỗi có tid trong `bad` thì trả phản hồi hỏng."""

    def __init__(self, bad=(), cost=0.001):
        self.calls, self.bad, self.cost, self.bodies = [], set(bad), cost, []

    def chat(self, model, messages, temperature, top_p, max_tokens, provider_order=None, extra_body=None):
        prompt = messages[0]["content"]
        self.calls.append((model, temperature, max_tokens))
        self.bodies.append(extra_body)
        solution = prompt.split("solution: ", 1)[1].split("\nEND", 1)[0]
        if any(b in prompt for b in self.bad):
            return ChatResult("not json at all", "stop", 100, 5, self.cost, 0.1, model)
        mid = solution.find(" ", len(solution) // 2)
        text = clean([[solution[:mid]], [solution[mid + 1:]]])
        return ChatResult(text, "stop", 100, 120, self.cost, 0.1, model)


@pytest.fixture
def work(tmp_path, monkeypatch):
    make_workdir(tmp_path, n_q=10)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("problem: {problem}\nsolution: {solution}\nEND", encoding="utf-8")
    client = FakeClient()
    monkeypatch.setattr("src.common.api.make_openrouter_client", lambda cfg, tracker=None: client)
    args = ["--workdir", str(tmp_path), "--override", f"segmenter.prompt_file={prompt}"]
    return tmp_path, client, args


def test_pilot_segments_a_fixed_sample_with_thinking_off_and_resumes(work, capsys):
    wd, client, args = work
    assert ss.main(args + ["--pilot", "12"]) == 0
    rows = read_jsonl(wd / ss.OUT_NAME)
    assert len(rows) == 12 == len(client.calls) and all(r["status"] == "ok" and r["n_steps"] == 2 for r in rows)
    assert all(b == {"reasoning": {"enabled": False}} for b in client.bodies)       # tắt chế độ suy nghĩ
    assert all(c == ("z-ai/glm-4.5-air", 0.0, 8192) for c in client.calls)
    assert all("raw" in r and r["ends"][-1] == r["n_chars"] for r in rows)          # lô thử giữ phản hồi thô
    first = [r["tid"] for r in rows]
    assert ss.main(args + ["--pilot", "12"]) == 0 and len(client.calls) == 12       # chạy lại không gọi thêm
    assert ss.main(args + ["--pilot", "15"]) == 0 and len(client.calls) == 15       # mẫu lớn hơn chứa mẫu nhỏ
    assert [r["tid"] for r in read_jsonl(wd / ss.OUT_NAME)][:12] == first
    assert "giữ nguyên câu chữ" in capsys.readouterr().out


def test_bad_responses_are_recorded_and_retried_next_time(work):
    wd, client, args = work
    ss.main(args + ["--pilot", "6"])
    bad_tid = read_jsonl(wd / ss.OUT_NAME)[0]["tid"]
    (wd / ss.OUT_NAME).unlink()
    q = {c["tid"]: c for c in read_jsonl(wd / "candidates.jsonl")}[bad_tid]["text"][:30]
    client.bad = {q}
    ss.main(args + ["--pilot", "6"])
    status = {r["tid"]: r["status"] for r in read_jsonl(wd / ss.OUT_NAME)}
    assert status[bad_tid] == "json_error" and sum(v == "ok" for v in status.values()) == 5
    client.bad, n = set(), len(client.calls)
    ss.main(args + ["--pilot", "6"])
    assert len(client.calls) == n + 1                                                # chỉ gọi lại chuỗi hỏng
    assert {r["tid"]: r["status"] for r in read_jsonl(wd / ss.OUT_NAME)}[bad_tid] == "ok"


def test_full_run_needs_a_budget_and_stops_at_it(work, capsys):
    wd, client, args = work
    with pytest.raises(SystemExit, match="--max-cost"):
        ss.main(args + ["--all"])
    client.cost = 0.5
    ss.main(args + ["--all", "--max-cost", "2", "--workers", "1"])
    assert 4 <= len(client.calls) <= 5 and "DỪNG vì chạm trần" in capsys.readouterr().out
    assert ss.main(args + ["--report"]) == 0

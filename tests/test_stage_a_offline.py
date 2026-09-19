"""Kiểm thử a2 + a3 không cần mạng: dùng client giả thay OpenRouter."""
import json
from pathlib import Path

import pytest

from src.common.api import ChatResult, EmptyResponse, UsageTracker
from src.common.config import load_config, parse_override
from src.common.io_utils import JsonlWriter, read_jsonl, split_tid, traj_id
from src.stage_a import a2_generate, a3_filter


class FakeClient:
    """Đóng vai ChatClient. Hành vi quyết định theo (model_id, đề bài) để kết quả tái lập được."""

    def __init__(self, questions, fail_once=None, cost=0.0, truncate_tid=None):
        self.tracker = UsageTracker()
        self.gold = {q["question"]: q["gold"] for q in questions}
        self.fail_once = set(fail_once or [])
        self.cost = cost
        self.calls = 0
        self.seen_messages = []

    def chat(self, model, messages, temperature, top_p, max_tokens, provider_order=None):
        self.calls += 1
        self.seen_messages.append(messages)
        question = messages[-1]["content"]
        key = (model, question)
        if key in self.fail_once:
            self.fail_once.discard(key)
            self.tracker.add_final_failure()
            raise EmptyResponse("giả lập phản hồi rỗng")
        gold = self.gold[question]
        answer = gold if "llama" in model or "qwen" in model else "999"   # deepseek luôn sai trong bài thử này
        res = ChatResult(text=f"Ta có ... \\boxed{{{answer}}}", finish_reason="stop", prompt_tokens=10,
                         completion_tokens=50, cost=self.cost or None, latency_s=0.01, served_model=model)
        self.tracker.add_ok(res)
        return res


def make_questions(n=4):
    qs = []
    for i in range(n):
        qs.append({"qid": f"math-train-{i}", "source": "math", "level": 4, "question": f"Tính {i}+{i}?",
                   "solution": "", "gold": str(2 * i)})
    qs.append({"qid": "gsm8k-train-0", "source": "gsm8k", "level": None, "question": "Có 3 quả, thêm 4 quả?",
               "solution": "", "gold": "7"})
    return qs


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def workdir(tmp_path, cfg):
    wd = tmp_path / "run"
    with JsonlWriter(wd / cfg["stage_a_files"]["questions"]) as w:
        for q in make_questions():
            w.append(q)
    return wd


def test_config_merge_and_directions(cfg):
    assert cfg.selection.k == 3 and cfg.generation.temperature == 0.8
    from src.common.config import signal_direction
    assert signal_direction(cfg, "rsr") == "min"
    assert all(signal_direction(cfg, n) == "max" for n in ("grape", "local_nat", "lark", "token_length"))
    assert abs(sum(cfg.quality.rule_weights.values()) - 1.0) < 1e-9
    assert parse_override("selection.k=1") == {"selection": {"k": 1}}
    assert load_config(overrides=["selection.k=1"]).selection.k == 1
    with pytest.raises(AttributeError):
        cfg.selection.khong_ton_tai


def test_prompt_is_boxed_and_system_role(cfg, workdir):
    qs = read_jsonl(workdir / cfg["stage_a_files"]["questions"])
    client = FakeClient(qs)
    teachers = a2_generate.select_teachers(cfg, ["llama70b"])
    a2_generate.run_generation(cfg, qs[:1], teachers, client, workdir, show_progress=False)
    msgs = client.seen_messages[0]
    assert msgs[0] == {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}"}
    assert msgs[1]["role"] == "user" and "####" not in msgs[0]["content"]


def test_generate_resume_and_filter(cfg, workdir):
    files = cfg["stage_a_files"]
    qs = read_jsonl(workdir / files["questions"])
    teachers = a2_generate.select_teachers(cfg, None)
    # qwen72b lỗi đúng MỘT lượt (mẫu đầu tiên) ở câu math-train-1, hai mẫu còn lại vẫn thành công
    client = FakeClient(qs, fail_once=[("qwen/qwen-2.5-72b-instruct", "Tính 1+1?")])
    a2_generate.run_generation(cfg, qs, teachers, client, workdir, show_progress=False, workers=2)
    trajs = read_jsonl(workdir / files["trajectories"])
    fails = read_jsonl(workdir / files["failures"])
    total = len(qs) * 3 * 3
    assert len(fails) == 1 and len(trajs) == total - 1
    assert fails[0]["tid"] == "math-train-1|qwen72b|0" and fails[0]["error_type"] == "EmptyResponse"

    # a3 phải dừng vì thiếu chuỗi
    with pytest.raises(SystemExit) as e:
        a3_filter.main(["--workdir", str(workdir)])
    assert "không đủ" in str(e.value)

    # chạy bù: chỉ sinh đúng những tid thiếu
    before = client.calls
    a2_generate.run_generation(cfg, qs, teachers, client, workdir, show_progress=False)
    assert client.calls - before == 1                       # đúng một tid còn thiếu
    assert len(read_jsonl(workdir / files["trajectories"])) == total
    # chạy lần ba: không còn việc
    before = client.calls
    a2_generate.run_generation(cfg, qs, teachers, client, workdir, show_progress=False)
    assert client.calls == before

    # a3
    assert a3_filter.main(["--workdir", str(workdir)]) == 0
    labels = {lb["tid"]: lb for lb in read_jsonl(workdir / files["labels"])}
    assert len(labels) == total
    assert all(not lb["correct"] for lb in labels.values() if lb["teacher"] == "deepseek")
    assert all(lb["correct"] for lb in labels.values() if lb["teacher"] != "deepseek")
    stats = json.loads((workdir / files["stats"]).read_text(encoding="utf-8"))
    assert stats["kept_questions"] == len(qs)              # mỗi câu có 6 chuỗi đúng >= 3
    assert stats["per_teacher"]["deepseek"]["accuracy"] == 0.0
    assert stats["per_group"]["math-L4"]["correct_count_hist"]["6"] == 4
    cands = read_jsonl(workdir / files["candidates"])
    assert len(cands) == len(qs) * 6 and all("text" in c for c in cands)
    assert all(split_tid(c["tid"])[1] != "deepseek" for c in cands)


def test_truncated_counts_as_wrong(cfg, workdir):
    files = cfg["stage_a_files"]
    qs = read_jsonl(workdir / files["questions"])
    rows = [
        {"tid": traj_id("math-train-1", "llama70b", 0), "qid": "math-train-1", "teacher": "llama70b", "sample_idx": 0,
         "text": "... \\boxed{2}", "finish_reason": "length", "completion_tokens": 2048},
        {"tid": traj_id("math-train-1", "llama70b", 1), "qid": "math-train-1", "teacher": "llama70b", "sample_idx": 1,
         "text": "... \\boxed{2}", "finish_reason": "stop", "completion_tokens": 100},
        {"tid": traj_id("math-train-1", "llama70b", 2), "qid": "math-train-1", "teacher": "llama70b", "sample_idx": 2,
         "text": "không có đáp án", "finish_reason": "stop", "completion_tokens": 100},
    ]
    labels = a3_filter.label_trajectories(qs, rows)
    assert [(lb["correct"], lb["reason"]) for lb in labels] == [(False, "truncated"), (True, "match"), (False, "no_boxed")]


def test_budget_guard(cfg, workdir):
    files = cfg["stage_a_files"]
    qs = read_jsonl(workdir / files["questions"])
    teachers = a2_generate.select_teachers(cfg, None)
    client = FakeClient(qs, cost=0.5)
    s = a2_generate.run_generation(cfg, qs, teachers, client, workdir, max_cost=2.0, workers=1, show_progress=False)
    assert s["budget_hit"] and 4 <= client.calls <= 6, client.calls
    assert client.tracker.cost >= 2.0


def test_dry_run_makes_no_calls(cfg, workdir, capsys):
    assert a2_generate.main(["--workdir", str(workdir), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "cần sinh 45" in out and "\\boxed{}" in out
    assert not (workdir / cfg["stage_a_files"]["trajectories"]).exists()


def test_writer_survives_truncated_last_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"a": 2', encoding="utf-8")       # dòng cuối bị cắt giữa chừng
    assert read_jsonl(p) == [{"a": 1}]
    with JsonlWriter(p) as w:
        w.append({"a": 3})
    text = p.read_text(encoding="utf-8")
    assert text.count("\n") == 3
    with pytest.raises(ValueError):                             # dòng hỏng ở giữa file thì phải báo lỗi
        read_jsonl(p)

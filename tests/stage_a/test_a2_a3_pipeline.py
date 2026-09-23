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
        content = messages[-1]["content"]
        question = next((q for q in self.gold if q in content), content)
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

    # Đọc tên tín hiệu từ chính cấu hình thay vì viết cứng, để đổi tên khoá không làm hỏng bài kiểm thử.
    # Điều cần giữ là quy tắc, không phải danh sách tên: CHỈ rsr là min, mọi tín hiệu khác đều max.
    names = list(cfg["signal_direction"])
    assert "rsr" in names and len(names) >= 5
    assert signal_direction(cfg, "rsr") == "min"
    assert [n for n in names if signal_direction(cfg, n) == "min"] == ["rsr"]
    assert any("local" in n for n in names), "thiếu khoá cho Local Naturalness trong signal_direction"
    with pytest.raises(KeyError):
        signal_direction(cfg, "tin_hieu_khong_ton_tai")
    assert abs(sum(cfg.quality.rule_weights.values()) - 1.0) < 1e-9
    assert parse_override("selection.k=1") == {"selection": {"k": 1}}
    assert load_config(overrides=["selection.k=1"]).selection.k == 1
    with pytest.raises(AttributeError):
        cfg.selection.khong_ton_tai


def test_default_prompt_follows_rsr_generation_recipe(cfg, workdir):
    """Bài gốc (A.2) nối chỉ dẫn vào cuối đề bài, vai trò user, không có system prompt."""
    assert cfg.generation.prompt_id == "rsr"
    qs = read_jsonl(workdir / cfg["stage_a_files"]["questions"])
    client = FakeClient(qs)
    teachers = a2_generate.select_teachers(cfg, ["llama70b"])
    a2_generate.run_generation(cfg, qs[:1], teachers, client, workdir, show_progress=False)
    msgs = client.seen_messages[0]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert msgs[0]["content"].endswith("Return your final response within \\boxed{}.")
    assert qs[0]["question"] in msgs[0]["content"] and "####" not in msgs[0]["content"]
    row = read_jsonl(workdir / cfg["stage_a_files"]["trajectories"])[0]
    assert row["prompt_id"] == "rsr"


def test_legacy_boxed_prompt_still_available():
    from src.common.prompts import build_generation_messages
    msgs = build_generation_messages("2+2?", "boxed")
    assert msgs[0]["role"] == "system" and msgs[1] == {"role": "user", "content": "2+2?"}
    import pytest as _pytest
    with _pytest.raises(KeyError):
        build_generation_messages("2+2?", "khong_ton_tai")


def test_generate_resume_and_filter(cfg, workdir):
    files = cfg["stage_a_files"]
    qs = read_jsonl(workdir / files["questions"])
    teachers = a2_generate.select_teachers(cfg, None)
    # qwen72b lỗi đúng MỘT lượt (mẫu đầu tiên) ở câu math-train-1, hai mẫu còn lại vẫn thành công.
    # Lấy tên mô hình từ CẤU HÌNH chứ không viết cứng: đổi nhà cung cấp hay biến thể mô hình trong
    # models.yaml là chuyện bình thường, và không được làm hỏng bài kiểm thử.
    qwen_id = next(t["model_id"] for t in teachers if t["key"] == "qwen72b")
    client = FakeClient(qs, fail_once=[(qwen_id, "Tính 1+1?")])
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
    assert "cần sinh 45" in out and "Return your final response within \\boxed{}." in out
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


def test_shard_writes_separate_files_and_sees_other_shards(cfg, workdir):
    """Chạy song song nhiều tiến trình: mỗi lô ghi file riêng, nhưng vẫn thấy phần các lô khác đã sinh."""
    files = cfg["stage_a_files"]
    qs = read_jsonl(workdir / files["questions"])
    client = FakeClient(qs)

    a2_generate.run_generation(cfg, qs, a2_generate.select_teachers(cfg, ["llama70b"]), client, workdir,
                               shard="llama70b", show_progress=False)
    a2_generate.run_generation(cfg, qs, a2_generate.select_teachers(cfg, ["qwen72b"]), client, workdir,
                               shard="qwen72b", show_progress=False)

    assert not (workdir / files["trajectories"]).exists()          # không có file chính
    n = len(qs) * 3
    assert len(read_jsonl(workdir / "trajectories.llama70b.jsonl")) == n
    assert len(read_jsonl(workdir / "trajectories.qwen72b.jsonl")) == n

    # lô thứ ba thấy được 2 lô trước, nên chỉ sinh phần của chính nó
    before = client.calls
    a2_generate.run_generation(cfg, qs, a2_generate.select_teachers(cfg, ["deepseek"]), client, workdir,
                               shard="deepseek", show_progress=False)
    assert client.calls - before == n

    # chạy lại một lô cũ thì không còn việc
    before = client.calls
    a2_generate.run_generation(cfg, qs, a2_generate.select_teachers(cfg, ["llama70b"]), client, workdir,
                               shard="llama70b", show_progress=False)
    assert client.calls == before

    # a3 đọc gộp cả ba lô
    assert a3_filter.main(["--workdir", str(workdir)]) == 0
    assert len(read_jsonl(workdir / files["labels"])) == n * 3


def test_a3_detects_overlapping_shards(cfg, workdir):
    """Hai lô cùng nhận một mô hình dạy sẽ sinh trùng tid; a3 phải bắt được thay vì đếm gấp đôi."""
    files = cfg["stage_a_files"]
    qs = read_jsonl(workdir / files["questions"])
    client = FakeClient(qs)
    rows = [{"tid": traj_id(qs[0]["qid"], "llama70b", 0), "qid": qs[0]["qid"], "teacher": "llama70b",
             "sample_idx": 0, "text": "x", "finish_reason": "stop", "completion_tokens": 5}]
    from src.common.io_utils import write_jsonl
    write_jsonl(workdir / "trajectories.a.jsonl", rows)
    write_jsonl(workdir / "trajectories.b.jsonl", rows)
    with pytest.raises(SystemExit) as e:
        a3_filter.main(["--workdir", str(workdir), "--allow-incomplete"])
    assert "trùng nhau" in str(e.value)


def test_key_limit_detection():
    """Lỗi 403 do khoá chạm hạn mức phải được nhận ra, để dừng cả lô thay vì ghi hàng nghìn dòng lỗi."""
    from src.stage_a.a2_generate import is_key_limit

    class PermissionDeniedError(Exception):
        pass

    assert is_key_limit(PermissionDeniedError("Error code: 403 - {'message': 'Key limit exceeded (total limit)'}"))
    assert not is_key_limit(PermissionDeniedError("403 forbidden region"))
    assert not is_key_limit(RuntimeError("Key limit exceeded"))       # sai loại lỗi thì không tính


def test_trajectory_files_lists_only_existing(cfg, tmp_path):
    from src.stage_a.a2_generate import all_trajectory_files
    files = cfg["stage_a_files"]
    (tmp_path / "trajectories.deepseek.jsonl").write_text("", encoding="utf-8")
    got = [p.name for p in all_trajectory_files(tmp_path, files)]
    assert got == ["trajectories.deepseek.jsonl"]                     # không liệt kê trajectories.jsonl không tồn tại

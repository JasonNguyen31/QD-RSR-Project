"""Kiểm thử phần không cần torch của b3_train: định dạng mẫu, token kết thúc lượt, chia lô, tốc độ học, việc
đọc tập chọn. Phần toán của mất mát cần torch nên nằm trong `python -m src.stage_b.b3_train --selftest`."""
import ast
import math
from pathlib import Path

import pytest

from src.common.config import load_config
from src.common.io_utils import read_json, read_jsonl, write_json, write_jsonl
from src.common.prompts import SYSTEM_PROMPT_TRAIN
from src.stage_b import b2_select as b2
from src.stage_b import b3_train as b3
from test_b2_select import make_workdir

SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2}


class FakeTok:
    """Tokenizer giả: mỗi ký tự một token, hai token đặc biệt và khuôn hội thoại kiểu ChatML như của Qwen."""
    eos_token_id = 0
    pad_token_id = 0
    end = "<|im_end|>\n"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}{self.end}" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for name, sid in SPECIAL.items():
                if text.startswith(name, i):
                    ids.append(sid)
                    i += len(name)
                    break
            else:
                ids.append(10 + ord(text[i]))
                i += 1
        return {"input_ids": ids}


class NoEndTok(FakeTok):
    end = "\n"                     # khuôn không gắn token nào sau câu trả lời


# ---------------------------------------------------------------- token kết thúc lượt và định dạng mẫu
def test_end_of_turn_token_comes_from_the_chat_template():
    assert b3.end_of_turn_ids(FakeTok()) == [SPECIAL["<|im_end|>"]]
    assert b3.end_of_turn_ids(NoEndTok()) == [NoEndTok.eos_token_id]      # không có thì lùi về eos


def test_sample_is_prompt_then_chain_then_end_of_turn_with_loss_only_on_the_answer():
    tok, eot = FakeTok(), [2]
    s = b3.encode_sample(tok, "What is 1+1?", "It is \\boxed{2}", 500, eot)
    prompt = tok.apply_chat_template(b3.prompt_messages("What is 1+1?"), add_generation_prompt=True)
    p_ids, r_ids = tok(prompt)["input_ids"], tok("It is \\boxed{2}")["input_ids"]
    assert s["ids"] == p_ids + r_ids + eot                       # phần đề và phần chuỗi đúng như b1_fit dựng
    assert s["labels"] == [b3.IGNORE] * len(p_ids) + r_ids + eot  # đề không tính mất mát, token kết thúc lượt có
    assert s["n_labels"] == len(r_ids) + 1 and s["n_prompt"] == len(p_ids) and not s["truncated"]
    assert SYSTEM_PROMPT_TRAIN in prompt and prompt.endswith("<|im_start|>assistant\n")


def test_truncated_sample_gets_no_end_of_turn_token():
    tok, eot = FakeTok(), [2]
    full = b3.encode_sample(tok, "q", "x" * 50, 10_000, eot)
    cut = b3.encode_sample(tok, "q", "x" * 50, full["n_prompt"] + 20, eot)
    assert cut["truncated"] and len(cut["ids"]) == full["n_prompt"] + 20 and cut["ids"][-1] != 2
    exact = b3.encode_sample(tok, "q", "x" * 50, full["n_prompt"] + 51, eot)
    assert not exact["truncated"] and exact["ids"][-1] == 2          # vừa khít thì vẫn có token kết thúc lượt
    assert b3.encode_sample(tok, "q", "x" * 50, full["n_prompt"] + 1, eot) is None    # không còn chỗ cho chuỗi


# ---------------------------------------------------------------- chia lô
def test_every_sample_appears_once_per_epoch_in_random_effective_batches():
    lengths = [(i * 37) % 91 + 5 for i in range(101)]
    windows = b3.epoch_windows(lengths, 42, 0, 2, 8)
    flat = [j for w in windows for mb in w for j in mb]
    assert sorted(flat) == list(range(101))
    assert [sum(len(mb) for mb in w) for w in windows] == [16] * 6 + [5]
    assert all(len(mb) <= 2 for w in windows for mb in w)
    assert windows == b3.epoch_windows(lengths, 42, 0, 2, 8)                     # xác định
    assert windows != b3.epoch_windows(lengths, 42, 1, 2, 8)                     # epoch sau xáo khác
    assert windows != b3.epoch_windows(lengths, 43, 0, 2, 8)                     # seed khác xáo khác


def test_length_sorting_never_changes_which_samples_share_an_effective_batch():
    """Xếp theo độ dài chỉ được đổi cách chia lô nhỏ BÊN TRONG lô hiệu dụng, không đổi thành phần của nó."""
    a = b3.epoch_windows([(i * 37) % 91 for i in range(64)], 7, 2, 2, 8)
    b = b3.epoch_windows([(i * 11) % 53 for i in range(64)], 7, 2, 2, 8)
    assert [sorted(j for mb in w for j in mb) for w in a] == [sorted(j for mb in w for j in mb) for w in b]
    lengths = [(i * 37) % 91 for i in range(64)]
    for w in a:
        order = [lengths[j] for mb in w for j in mb]
        assert order == sorted(order)                             # trong lô hiệu dụng, mẫu xếp theo độ dài


def test_padding_masks_and_window_denominator():
    samples = [{"ids": [5, 6, 7, 8], "labels": [-100, 6, 7, 8], "n_labels": 3, "weight": 1.0},
               {"ids": [5, 9], "labels": [-100, 9], "n_labels": 1, "weight": 2.5}]
    batch = b3.pad_batch(samples, pad_id=0)
    assert batch["input_ids"] == [[5, 6, 7, 8], [5, 9, 0, 0]] and batch["attention_mask"][1] == [1, 1, 0, 0]
    assert batch["labels"][1] == [-100, 9, -100, -100] and batch["weights"] == [1.0, 2.5]
    assert b3.window_denominator(samples) == pytest.approx(3 * 1.0 + 1 * 2.5)


def test_learning_rate_follows_linear_warmup_then_cosine():
    total, warmup = 200, 6
    assert b3.lr_factor(0, total, warmup) == 0.0 and b3.lr_factor(3, total, warmup) == pytest.approx(0.5)
    assert b3.lr_factor(warmup, total, warmup) == 1.0
    mid = warmup + (total - warmup) // 2
    assert b3.lr_factor(mid, total, warmup) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.5)))
    assert b3.lr_factor(total, total, warmup) == pytest.approx(0.0, abs=1e-12)
    values = [b3.lr_factor(s, total, warmup) for s in range(warmup, total + 1)]
    assert values == sorted(values, reverse=True)


def test_step_counts_come_from_the_config():
    cfg = load_config(student="qwen1_5b")
    t = cfg.training
    total, warmup = b3.total_steps(5685, cfg)
    assert total == math.ceil(5685 / (t.batch_size * t.grad_accum)) * t.num_epochs
    assert warmup == math.ceil(total * t.warmup_ratio)
    for key in ("weight_decay", "max_grad_norm", "loss_chunk", "save_every_steps", "stop_check_samples"):
        assert key in t, key


# ---------------------------------------------------------------- đọc tập chọn
@pytest.fixture
def seldir(tmp_path):
    work = tmp_path / "a"
    work.mkdir()
    make_workdir(work, n_q=20)
    out = tmp_path / "sel"
    for extra in (["--method", "correct_only"], ["--method", "lark"],
                  ["--method", "local_naturalness", "--allow-blocked"]):
        assert b2.main(["--workdir", str(work), "--outdir", str(out)] + extra) == 0
    return out


def test_selection_is_loaded_with_weights_and_checked_against_its_md5(seldir):
    rows, meta = b3.load_selection(seldir, "correct_only.k3", 3)
    assert len(rows) == 60 == meta["samples"] and all(r["weight"] == 1.0 for r in rows)
    lark, _ = b3.load_selection(seldir, "lark.k3", 3)
    assert sum(r["weight"] for r in lark) == pytest.approx(60, abs=1e-3) and len({r["weight"] for r in lark}) > 3
    path = seldir / "train.correct_only.k3.jsonl"
    write_jsonl(path, read_jsonl(path)[:-1] + [{**read_jsonl(path)[-1], "text": "sửa tay"}])
    with pytest.raises(SystemExit, match="md5"):
        b3.load_selection(seldir, "correct_only.k3", 3)


def test_blocked_or_missing_or_wrong_k_selection_is_refused(seldir):
    with pytest.raises(SystemExit, match="bị khoá"):
        b3.load_selection(seldir, "local_naturalness.k3.tam", 3)
    with pytest.raises(SystemExit, match="Không thấy tập chọn"):
        b3.load_selection(seldir, "rsr.k3", 3)
    with pytest.raises(SystemExit, match="k = 1"):
        b3.load_selection(seldir, "correct_only.k3", 1)


def test_plan_runs_without_torch_and_run_dir_is_named_by_tag_and_seed(seldir, capsys):
    assert b3.main(["--method", "correct_only", "--seldir", str(seldir), "--plan"]) == 0
    line = capsys.readouterr().out
    assert "correct_only.k3" in line and "60 mẫu" in line
    cfg = load_config(method="correct_only", student="qwen1_5b")
    assert b3.run_dir(cfg, "correct_only.k3", 42).parts[-3:] == ("qwen1_5b_base", "correct_only.k3", "seed42")
    assert b3.run_dir(cfg, "correct_only.k3", 42, smoke=True).name == "smoke"


def test_selection_made_for_another_student_is_refused(seldir):
    meta = read_json(seldir / "select.correct_only.k3.json")
    write_json(seldir / "select.correct_only.k3.json", {**meta, "student": "Qwen/Qwen2.5-7B"})
    with pytest.raises(SystemExit, match="dành cho"):
        b3.main(["--method", "correct_only", "--seldir", str(seldir), "--plan"])


def test_module_imports_no_gpu_library_at_top_level():
    """--plan, các bài kiểm thử này và máy Mac không có GPU đều phải nạp được module."""
    tree = ast.parse(Path(b3.__file__).read_text(encoding="utf-8"))
    top = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
           for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
           for a in (n.names if isinstance(n, ast.Import) else [None])}
    assert not top & {"torch", "transformers", "peft", "safetensors", "bitsandbytes"}


# ---------------------------------------------------------------- phép thử dừng (phần không cần torch)
def test_stop_tokens_are_end_of_turn_then_eos_when_different():
    assert b3.stop_token_ids(FakeTok(), [2]) == [2, 0]                  # <|im_end|> rồi eos_token
    assert b3.stop_token_ids(FakeTok(), [0]) == [0]                     # trùng nhau thì chỉ một token


def test_stop_check_items_are_distinct_questions_in_an_order_fixed_by_seed():
    rows = [{"qid": f"q{i // 3}", "question": f"Question {i // 3}?"} for i in range(30)]
    items = b3.stop_check_items(rows, 4, 42)
    assert len(items) == 4 == len({it["qid"] for it in items})
    assert all(it["question"] == f"Question {it['qid'][1:]}?" for it in items)
    assert items == b3.stop_check_items(list(reversed(rows)), 4, 42)    # không phụ thuộc thứ tự dòng trong file
    assert items[:2] == b3.stop_check_items(rows, 2, 42)
    assert [it["qid"] for it in items] != [it["qid"] for it in b3.stop_check_items(rows, 4, 43)]
    assert len(b3.stop_check_items(rows, 99, 42)) == 10                 # không đủ câu thì lấy hết


def test_stop_summary_counts_stops_by_token_and_lengths_before_the_stop():
    box = [10 + ord(c) for c in "\\boxed{2}"]
    rows = [box + [2, 0, 0],            # dừng ở <|im_end|>, phần sau là đệm
            [50, 51] + box + [0, 0],    # dừng ở eos_token
            [50] * 6,                   # không dừng, không có \boxed
            box + [50] * 4]             # không dừng dù đã có \boxed
    decode = lambda ids: "".join(chr(i - 10) for i in ids)
    s = b3.summarise_stop(rows, [2, 0], decode, 16, {2: "<|im_end|>", 0: "<|endoftext|>"}.get)
    assert (s["n"], s["stopped"], s["with_boxed"], s["max_new_tokens"]) == (4, 2, 3, 16)
    assert s["stopped_share"] == 0.5 and s["stopped_by"] == {"<|im_end|>": 1, "<|endoftext|>": 1}
    assert s["new_tokens"] == [len(box), len(box) + 2, 6, len(box) + 4]
    assert s["mean_new_tokens"] == sum(s["new_tokens"]) / 4
    assert b3.split_at_stop([5, 6, 2, 0], [2, 0]) == ([5, 6], 2) and b3.split_at_stop([5, 6], [2, 0]) == ([5, 6], None)


# ---------------------------------------------------------------- token kết thúc (chốt 03/10) và chạy thử ngắn
def test_end_token_is_the_tokenizer_eos_by_decision_of_0310():
    """Cố ý ghi cứng: bản nền Qwen chưa học <|im_end|>, nên token kết thúc là eos_token (<|endoftext|>)."""
    cfg = load_config(method="correct_only", student="qwen1_5b")
    assert cfg.training.end_token == "eos"
    assert load_config(method="qd_rsr", student="qwen7b").training.end_token == "eos"
    assert b3.end_token_ids(FakeTok(), "eos") == [FakeTok.eos_token_id]
    assert b3.end_token_ids(FakeTok(), "chat_template") == [SPECIAL["<|im_end|>"]]      # cách cũ, để tái lập
    with pytest.raises(SystemExit, match="không hợp lệ"):
        b3.end_token_ids(FakeTok(), "im_end")

    class NoEos(FakeTok):
        eos_token_id = None
    with pytest.raises(SystemExit, match="eos_token"):
        b3.end_token_ids(NoEos(), "eos")
    assert b3.stop_token_ids(FakeTok(), [FakeTok.eos_token_id]) == [FakeTok.eos_token_id]   # một token dừng duy nhất


def test_end_token_label_survives_padding_although_pad_is_the_same_token():
    """pad_token của Qwen cũng là <|endoftext|>: phần đệm phải bị che theo vị trí, không theo id của token."""
    tok = FakeTok()
    assert tok.pad_token_id == tok.eos_token_id
    eot = b3.end_token_ids(tok, "eos")
    short = {**b3.encode_sample(tok, "q", "ab \\boxed{1}", 500, eot), "weight": 1.0}
    long = {**b3.encode_sample(tok, "q", "a much longer chain \\boxed{1}", 500, eot), "weight": 1.0}
    batch = b3.pad_batch([short, long], tok.pad_token_id)
    n = len(short["ids"])
    assert batch["input_ids"][0][n - 1] == eot[0] and batch["labels"][0][n - 1] == eot[0]   # nhãn kết thúc còn nguyên
    assert set(batch["labels"][0][n:]) == {b3.IGNORE} and set(batch["attention_mask"][0][n:]) == {0}
    assert batch["labels"][1][-1] == eot[0]
    assert sum(lab != b3.IGNORE for lab in batch["labels"][0]) == short["n_labels"]         # mẫu số không lệch


def test_untrained_end_token_row_is_recognised_from_its_direction():
    dead = {"norm": 0.414, "unused_rows": 271, "unused_median_norm": 0.414, "cos_unused_mean": 1.0}     # <|im_end|>
    alive = {"norm": 1.151, "unused_rows": 271, "unused_median_norm": 0.414, "cos_unused_mean": -0.381} # <|endoftext|>
    unknown = {"norm": 1.0, "unused_rows": 0, "unused_median_norm": None, "cos_unused_mean": None}
    assert b3.end_token_is_untrained(dead)
    assert not b3.end_token_is_untrained(alive) and not b3.end_token_is_untrained(unknown)


def test_pilot_takes_whole_questions_in_an_order_fixed_by_seed(seldir, capsys):
    rows, _ = b3.load_selection(seldir, "correct_only.k3", 3)
    sub = b3.pilot_rows(rows, 20, 42)
    per_q = {q: sum(r["qid"] == q for r in sub) for q in {r["qid"] for r in sub}}
    assert set(per_q.values()) == {3} and 20 <= len(sub) < 23                    # trọn câu hỏi, vừa đủ n
    assert sub == [r for r in rows if r["qid"] in per_q]                         # giữ thứ tự dòng của file train
    assert sub == b3.pilot_rows(rows, 20, 42)
    assert {r["tid"] for r in sub} == {r["tid"] for r in b3.pilot_rows(list(reversed(rows)), 20, 42)}
    assert {r["qid"] for r in sub} != {r["qid"] for r in b3.pilot_rows(rows, 20, 43)}
    assert len(b3.pilot_rows(rows, 10_000, 42)) == len(rows)
    cfg = load_config(method="correct_only", student="qwen1_5b")
    assert b3.run_dir(cfg, "correct_only.k3", 42, pilot=500).name == "pilot500_seed42"
    assert b3.run_dir(cfg, "correct_only.k3", 42).name == "seed42"               # lần chạy thật không đổi chỗ
    assert b3.main(["--method", "correct_only", "--seldir", str(seldir), "--plan", "--pilot", "20"]) == 0
    line = capsys.readouterr().out
    assert "21 mẫu (chạy thử ngắn)" in line and "token kết thúc: eos" in line
    assert math.ceil(21 / 16) * cfg.training.num_epochs == b3.total_steps(21, cfg)[0]
    with pytest.raises(SystemExit):
        b3.main(["--method", "correct_only", "--seldir", str(seldir), "--pilot", "20", "--smoke"])

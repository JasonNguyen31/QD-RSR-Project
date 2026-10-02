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

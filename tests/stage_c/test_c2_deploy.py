"""Kiểm thử c2_deploy phần không cần mô hình, llama.cpp hay Ollama thật (Ollama thay bằng hàm giả)."""
import json
import re

import pytest

from src.common.config import load_config
from src.common.io_utils import read_jsonl, write_jsonl
from src.common.prompts import SYSTEM_PROMPT_TRAIN
from src.stage_c import c2_deploy as c2


def _cfg(method="grape", student="qwen1_5b"):
    return load_config(method=method, student=student)


# ---------------------------------------------------------------- khuôn và Modelfile
def test_ollama_template_renders_exactly_the_python_qwen_prompt():
    for system in (SYSTEM_PROMPT_TRAIN, ""):
        for q in ("What is 1 + 1?", "Find $x$ if \\frac{x}{2} = 3.\nShow work."):
            assert c2.render_ollama_template(c2.OLLAMA_TEMPLATE, system, q) == c2.qwen_prompt(system, q)


def test_qwen_prompt_has_the_layout_of_the_qwen25_chat_template():
    assert c2.qwen_prompt("S", "Q") == ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n"
                                        "<|im_start|>assistant\n")


def test_modelfile_carries_the_eval_settings_and_the_training_system_prompt():
    text = c2.modelfile_text("model-Q4_K_M.gguf", _cfg(), 3400)
    assert text.startswith("FROM ./model-Q4_K_M.gguf\n")
    assert f'SYSTEM """{SYSTEM_PROMPT_TRAIN}"""' in text
    assert f'TEMPLATE """{c2.OLLAMA_TEMPLATE}"""' in text
    for line in ("PARAMETER temperature 0.6", "PARAMETER top_p 0.95", "PARAMETER top_k 0", "PARAMETER min_p 0.0",
                 "PARAMETER repeat_penalty 1.0", "PARAMETER num_predict 3072", "PARAMETER num_ctx 3400",
                 'PARAMETER stop "<|endoftext|>"', 'PARAMETER stop "<|im_end|>"'):
        assert line in text.splitlines()


def test_sampling_options_override_ollama_defaults():
    o = c2.sampling_options(_cfg(), 4000, 7)
    assert o["repeat_penalty"] == 1.0 and o["top_k"] == 0 and o["min_p"] == 0.0
    assert (o["temperature"], o["top_p"], o["num_predict"], o["num_ctx"], o["seed"]) == (0.6, 0.95, 3072, 4000, 7)
    assert o["stop"] == ["<|endoftext|>", "<|im_end|>"]


def test_names_and_paths():
    cfg = _cfg("fq_select")
    name = c2.ollama_model_name(cfg, "fq_select.k3", 42, "Q4_K_M")
    assert name == "qdrsr-fq-select-qwen1_5b-s42:q4_k_m"
    assert re.fullmatch(r"[a-z0-9_.\-]+:[a-z0-9_.\-]+", name)
    assert c2.gguf_name("Q4_K_M") == "model-Q4_K_M.gguf" and c2.gguf_name("f16") == "model-f16.gguf"
    assert c2.results_dir(cfg, "fq_select.k3", 42, "Q4_K_M").name == "seed42.q4_k_m"
    assert c2.results_dir(cfg, "fq_select.k3", 42, "Q4_K_M", 50).name == "seed42.q4_k_m.limit50"
    assert c2.deploy_dir(cfg, "fq_select.k3", 42).parts[-4:] == ("deploy", "qwen1_5b_base", "fq_select.k3", "seed42")
    assert c2.default_merge_dtype(cfg) == "float32"
    assert c2.default_merge_dtype(_cfg(student="qwen7b")) == "bfloat16"


# ---------------------------------------------------------------- Ollama giả
class FakeOllama:
    """Trả lời /api/generate theo đề bài: đề có chữ RIGHT thì trả lời đúng, chữ LONG thì chạm trần."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, payload=None, timeout=0):
        if url.endswith("/api/ps"):
            return {"models": [{"name": "m:q4_k_m", "size": 2 * 2 ** 30, "size_vram": 2 * 2 ** 30}]}
        self.calls.append(payload)
        assert payload["raw"] is True and payload["stream"] is False
        p = payload["prompt"]
        if "LONG" in p:
            return {"response": "and so on " * 10, "done_reason": "length", "eval_count": 3072,
                    "eval_duration": 3072 * 10 ** 7, "prompt_eval_count": 40, "prompt_eval_duration": 10 ** 7}
        ans = "\\boxed{2}" if "RIGHT" in p else "\\boxed{5}"
        return {"response": f"Step 1. {ans}", "done_reason": "stop", "eval_count": 100, "eval_duration": 10 ** 9,
                "prompt_eval_count": 40, "prompt_eval_duration": 10 ** 7, "load_duration": 2 * 10 ** 9}


class FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return c2.qwen_prompt(messages[0]["content"], messages[1]["content"])

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


QUESTIONS = [{"qid": "q1", "question": "RIGHT 1+1?", "gold": "2", "subset": "x"},
             {"qid": "q2", "question": "wrong 1+1?", "gold": "2", "subset": "x"},
             {"qid": "q3", "question": "LONG 1+1?", "gold": "2", "subset": "x"}]


def test_to_row_marks_truncated_generations_as_not_stopped():
    g = {"text": "x \\boxed{2}", "done_reason": "stop", "eval_count": 10}
    assert c2.to_row("q", 0, "2", g, 3072) == {"qid": "q", "sample": 0, "correct": True, "pred": "2", "gold": "2",
                                               "n_tokens": 10, "stopped": True, "done_reason": "stop",
                                               "text": "x \\boxed{2}"}
    assert c2.to_row("q", 0, "2", {**g, "done_reason": "length", "eval_count": 3072}, 3072)["stopped"] is False


def test_eval_writes_c1_rows_scores_and_resumes(tmp_path):
    fake, out = FakeOllama(), tmp_path / "math500.jsonl"
    res = c2.step_eval(_cfg(), "m:q4_k_m", "http://x", QUESTIONS, FakeTok(), 4000, out, workers=3, post=fake,
                       log=lambda *_: None)
    rows = read_jsonl(out)
    assert len(rows) == 12 and len(fake.calls) == 12
    assert set(rows[0]) == {"qid", "sample", "correct", "pred", "gold", "n_tokens", "stopped", "done_reason", "text"}
    assert res["acc@4"] == pytest.approx(1 / 3) and res["pass@4"] == pytest.approx(1 / 3)
    assert res["hit_cap_share"] == pytest.approx(1 / 3) and res["n_questions"] == 3
    seeds = sorted(c["options"]["seed"] for c in fake.calls)
    assert len(set(seeds)) == 12                                        # mỗi lượt một seed riêng
    assert all(c["options"]["repeat_penalty"] == 1.0 for c in fake.calls)
    # chạy lại sau khi mất 5 dòng cuối: chỉ gọi lại đúng 5 lượt
    write_jsonl(out, rows[:7])
    fake2 = FakeOllama()
    res2 = c2.step_eval(_cfg(), "m:q4_k_m", "http://x", QUESTIONS, FakeTok(), 4000, out, workers=2, post=fake2,
                        log=lambda *_: None)
    assert len(fake2.calls) == 5 and len(read_jsonl(out)) == 12 and res2["acc@4"] == pytest.approx(1 / 3)


def test_bench_reports_speed_and_memory():
    b = c2.step_bench(_cfg(), "m:q4_k_m", "http://x", QUESTIONS[:2], FakeTok(), 4000, 2, post=FakeOllama(),
                      log=lambda *_: None)
    assert b["n"] == 2 and b["gen_tok_per_s_median"] == 100.0 and b["prompt_tok_per_s_median"] == 4000.0
    assert b["ollama_size_mb"] == 2048.0 and b["first_load_s"] == 2.0


def test_compare_is_paired_by_question(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [{"qid": f"q{i}", "sample": s, "correct": i % 2 == 0} for i in range(20) for s in range(4)])
    write_jsonl(b, [{"qid": f"q{i}", "sample": s, "correct": True} for i in range(20) for s in range(4)])
    d = c2.compare(a, b, 4)
    assert d["n"] == 20 and d["diff"] == pytest.approx(-0.5) and d["lo"] <= -0.5 <= d["hi"]
    assert d["ours_acc4"] == pytest.approx(0.5) and d["reference_acc4"] == 1.0


def test_plan_needs_nothing(capsys):
    assert c2.main(["--method", "grape", "--seed", "42", "--plan"]) == 0
    out = capsys.readouterr().out
    assert "qdrsr-grape-qwen1_5b-s42:q4_k_m" in out and "float32" in out


def test_merge_refuses_without_a_finished_training_run(tmp_path, monkeypatch):
    # thư mục đóng gói trỏ vào thư mục tạm: không được đụng outputs/deploy thật (trên máy đã có bản merge thật thì
    # bước merge sẽ bỏ qua và bài này không còn kiểm được gì; sửa 20/10)
    monkeypatch.setattr(c2, "deploy_dir", lambda cfg, tag, seed: tmp_path / "deploy")
    with pytest.raises(SystemExit):
        c2.main(["--method", "grape", "--seed", "42", "--step", "merge", "--run-dir", str(tmp_path / "run")])
    assert not (tmp_path / "deploy" / "deploy.json").exists()


# ---------------------------------------------------------------- sửa 20/10 sau lần chạy thật trên Mac (Ollama 0.40.6)
PS_OUT = """ 55552 ollama serve
5620848 /opt/homebrew/Cellar/ollama/0.40.6/libexec/lib/ollama/llama-server --model /Users/x/.ollama/models/blobs/sha256-9b02 --port 51136 --host 127.0.0.1 -c 13820 -np 4 --flash-attn auto
 1234 /usr/bin/grep ollama
"""


def test_parse_runners_finds_llama_server_of_new_ollama():
    r = c2.parse_runners(PS_OUT)
    assert r == {"runner_rss_mb": round(5620848 / 1024, 1), "runner_slots": 4, "runner_ctx": 13820}
    old = " 900000 /Applications/Ollama.app/Contents/Resources/ollama runner --model /m --ctx-size 4096\n"
    assert c2.parse_runners(old)["runner_rss_mb"] == round(900000 / 1024, 1)
    assert c2.parse_runners(" 10 ollama serve\n")["runner_rss_mb"] is None


QWEN_JINJA = """{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
{%- else %}
    {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}
"""


def test_jinja_template_from_gguf_is_checked_by_rendering_it():
    tok = FakeTok()
    assert c2.template_renders_training_prompt(c2.OLLAMA_TEMPLATE, tok)
    assert c2.template_renders_training_prompt(QWEN_JINJA, tok)
    assert not c2.template_renders_training_prompt(QWEN_JINJA.replace("assistant\\n", "assistant"), tok)
    assert not c2.template_renders_training_prompt("{{ .Prompt }}", tok)

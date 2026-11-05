"""Định dạng prompt của mô hình học (thêm 04/11/2026 cho Llama-3.2-1B)."""
import pytest

from src.common.config import load_config
from src.common.prompts import (PROMPT_FORMATS, SYSTEM_PROMPT_TRAIN, prompt_format_of, render_prompt,
                                set_prompt_format)
from src.stage_b import b2_select as b2


class Tok:
    bos_token = "<s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


class NoBosTok(Tok):
    bos_token = None


def test_default_is_chat_template_so_old_runs_are_unchanged():
    t = Tok()
    assert prompt_format_of(t) == "chat_template"
    assert render_prompt(t, "2+2?") == f"<system>{SYSTEM_PROMPT_TRAIN}<user>2+2?<assistant>"


def test_plain_format_puts_bos_inside_the_string_and_ends_at_the_answer():
    t = set_prompt_format(Tok(), "plain")
    out = render_prompt(t, "2+2?")
    assert out == f"<s>{SYSTEM_PROMPT_TRAIN}\n\nQuestion: 2+2?\n\nAnswer:\n"
    assert out.startswith("<s>") and "<system>" not in out


def test_plain_format_without_bos_token_adds_nothing():
    assert render_prompt(set_prompt_format(NoBosTok(), "plain"), "q").startswith(SYSTEM_PROMPT_TRAIN)


def test_unknown_format_is_refused():
    with pytest.raises(SystemExit):
        set_prompt_format(Tok(), "alpaca")
    assert PROMPT_FORMATS == ("chat_template", "plain")


def test_format_is_per_tokenizer_object():
    a, b = set_prompt_format(Tok(), "plain"), Tok()
    assert prompt_format_of(a) == "plain" and prompt_format_of(b) == "chat_template"


def test_llama_configs_load_and_stay_out_of_all():
    cfg = load_config(student="llama1b")
    assert cfg["student"]["prompt_format"] == "plain"
    assert b2.student_tag(cfg) == "llama1b_base"
    assert cfg["training"]["load_in_4bit"] is False
    for m, who in (("teacher_llama_first", "llama70b"), ("teacher_deepseek_first", "deepseek")):
        c = load_config(method=m, student="llama1b")
        assert b2.teacher_lists(c) == ([who], []) and c.select.control is True
        assert b2.selection_tag(c) == f"{m}.k3"
    names = {m for m, _a in b2.method_list(cfg, __import__("pathlib").Path(cfg.root))}
    assert not names & {"teacher_llama_first", "teacher_deepseek_first"}


def test_llama_student_skips_localnat_and_lists_only_what_needs_no_it():
    cfg = load_config(student="llama1b")
    assert cfg.signals.skip_local_nat is True and cfg.methods == ["correct_only", "grape"]


def test_new_controls_run_end_to_end_on_a_fit_without_localnat(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_b"))
    from test_b2_select import make_workdir
    from src.common.io_utils import read_json
    make_workdir(tmp_path, n_q=30, with_local=False, localnat=False)
    out = tmp_path / "out"
    share = {}
    for m, who in (("correct_only", "llama70b"), ("teacher_llama_first", "llama70b"),
                   ("teacher_deepseek_first", "deepseek")):
        assert b2.main(["--workdir", str(tmp_path), "--outdir", str(out), "--method", m]) == 0
        share[m] = read_json(out / f"select.{m}.k3.json")["teacher_share"].get(who, 0.0)
    assert share["teacher_llama_first"] > share["correct_only"]
    assert share["teacher_deepseek_first"] > 0.34

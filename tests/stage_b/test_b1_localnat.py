"""Kiểm thử b1_localnat: chấm LocalNat trên bước của GLM cho nhiều k trong một lượt.

Phần thuần (gộp mục trùng giữa các k, chia lô theo ngân sách bộ nhớ, chặn chạy bù lẫn cài đặt) không cần torch.
Phần chạy mô hình dùng một mô hình Qwen2 TÍ HON khởi tạo ngẫu nhiên trên CPU và một tokenizer giả: không tải gì
từ mạng. Máy không có torch hoặc transformers thì các bài đó tự bỏ qua; trên GPU thật thì dùng
python -m src.stage_b.b1_localnat --student qwen1_5b --base --selftest.
"""
import hashlib
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.io_utils import read_jsonl, write_jsonl
from src.stage_b import b1_localnat as ln
from src.stage_b import steps as st
from src.stage_b.signals import local_items, mean_over_steps
from test_b2_select import make_workdir

STEPS = [(0, 5), (5, 12), (12, 20), (20, 31), (31, 40), (40, 52)]         # sáu bước, như trung vị của GLM


# ---------------------------------------------------------------- phần thuần
def test_items_shared_between_k_values_are_scored_once():
    items, index = ln.unique_items(STEPS, [1, 4])
    assert len(items) == 10 < 2 * len(STEPS)                              # bước 0 và 1 giống nhau ở k = 1 và k = 4
    for k in (1, 4):
        assert [items[j] for j in index[k]] == local_items(STEPS, k)      # từng k vẫn nhận đúng mục của nó
    assert index[1][:2] == index[4][:2] and index[1][2:] != index[4][2:]
    one, ix = ln.unique_items(STEPS[:1], [1, 4])
    assert one == [(0, 0, 5)] and ix == {1: [0], 4: [0]}                  # chuỗi một bước: một mục, bằng GRAPE


def test_combine_is_the_mean_over_steps_per_k():
    items, index = ln.unique_items(STEPS, [1, 4])
    means = [-(j + 1) / 10 for j in range(len(items))]
    got = ln.combine(means, index)
    assert got[1] == pytest.approx(mean_over_steps([means[j] for j in index[1]])) and got[1] != got[4]
    assert ln.column(4) == "local_nat_k4" and ln.parse_ks([4, 1, 4]) == [1, 4]
    with pytest.raises(SystemExit):
        ln.parse_ks([0, 4])


def test_batches_respect_the_count_cap_and_the_memory_budget():
    lengths = [3000, 2900, 400, 380, 350, 120, 90, 60]
    cells = 2 * 3000 * 3000
    batches = ln.plan_batches(lengths, max_batch=4, cells=cells)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))      # mỗi mục đúng một lần
    assert batches[0] == [0, 1]                                                    # hai mục dài: ngân sách chỉ cho 2
    assert all(len(b) <= 4 and len(b) * max(lengths[i] for i in b) ** 2 <= cells for b in batches)
    assert ln.plan_batches([5000], 32, cells=1.0) == [[0]]                         # lô nào cũng có ít nhất một mục
    assert ln.attention_cells(12) > ln.attention_cells(28)                         # nhiều đầu attention thì lô nhỏ hơn


def test_resume_refuses_to_mix_settings():
    row = {"model": "m", "steps_version": st.STEPS_VERSION, "local_nat_k1": -1.0, "local_nat_k4": -0.9}
    ln.check_resume(None, [1, 4], "m", "f")
    ln.check_resume(row, [1, 4], "m", "f")
    for ks, model, r in [([1, 2, 4], "m", row), ([1, 4], "other", row), ([1, 4], "m", {**row, "steps_version": -1})]:
        with pytest.raises(SystemExit, match="cài đặt khác"):
            ln.check_resume(r, ks, model, "f")


# ---------------------------------------------------------------- kho giả có bước
def split_rows(tmp, broken=(), skip=()):
    """Ghi steps.glm.jsonl cho kho giả: mỗi chuỗi hai bước, cắt ở dấu cách gần giữa. broken: json_error không có
    phản hồi thô; skip: chưa cắt."""
    rows = []
    for i, c in enumerate(read_jsonl(tmp / "candidates.jsonl")):
        if i in skip:
            continue
        text = c["text"]
        base = {"tid": c["tid"], "qid": c["qid"], "text_md5": st.text_md5(text), "n_chars": len(text)}
        if i in broken:
            rows.append({**base, "status": "json_error", "finish_reason": "stop"})
        else:
            rows.append({**base, "status": "ok", "ends": [text.index(" ", len(text) // 2) + 1, len(text)], "n_steps": 2})
    write_jsonl(tmp / "steps.glm.jsonl", rows)
    return rows


def test_plan_counts_sources_and_stops_on_missing_or_too_many_fallbacks(tmp_path, capsys):
    make_workdir(tmp_path, n_q=6)
    n = len(read_jsonl(tmp_path / "candidates.jsonl"))
    args = ["--workdir", str(tmp_path), "--plan"]
    split_rows(tmp_path)
    assert ln.main(args) == 0
    out = capsys.readouterr().out
    assert f"có bước {n}, chưa cắt 0" in out and f"glm {n}" in out and f"{2 * n} mục chấm" in out   # 2 bước: không mục trùng

    split_rows(tmp_path, skip={0, 1})
    with pytest.raises(SystemExit, match="chưa được cắt bước"):
        ln.main(args)
    assert ln.main(args + ["--allow-missing"]) == 0                       # chấm phần đã có khi được cho phép

    split_rows(tmp_path, broken={0, 1, 2})                                 # 3 chuỗi hỏng trên vài chục: vượt 2%
    with pytest.raises(SystemExit, match="cắt dự phòng"):
        ln.main(args)
    assert ln.main(args + ["--max-fallback-share", "0.5"]) == 0
    assert "fallback_" in capsys.readouterr().out


# ---------------------------------------------------------------- phần chạy mô hình, trên CPU
class FakeTok:
    """Kiểu Qwen: dấu xuống dòng là token riêng; mỗi cụm ký tự khác trắng mang dấu cách hoặc tab đứng trước nó."""

    def __init__(self, vocab):
        self.vocab = vocab

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages) + " ASSISTANT:"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\n+|[ \t]*[^\s]+", text)]
        ids = [2 + int(hashlib.md5(text[a:b].strip().encode()).hexdigest(), 16) % (self.vocab - 2) for a, b in spans]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = tf.Qwen2Config(vocab_size=211, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                         pad_token_id=0, eos_token_id=1)
    return torch, tf.Qwen2ForCausalLM(cfg).eval()


def naive_means(torch, model, p_ids, r_ids, steps, k):
    """Chạy riêng từng mục bằng lượt forward đầy đủ, không lô, không đệm."""
    out = []
    for c, s, e in local_items(steps, k):
        x = torch.tensor([p_ids + r_ids[c:e]])
        first = len(p_ids) + (s - c)
        with torch.no_grad():
            lg = model(x).logits[0, first - 1:x.shape[1] - 1].float()
        out.append(float(torch.log_softmax(lg, -1).gather(-1, x[0, first:].unsqueeze(-1)).mean()))
    return out


def random_chain(seed=3, n=140, n_cuts=8):
    rng = random.Random(seed)
    p_ids = [rng.randrange(2, 211) for _ in range(17)]
    r_ids = [rng.randrange(2, 211) for _ in range(n)]
    cuts = sorted(rng.sample(range(5, n - 5), n_cuts))
    return p_ids, r_ids, list(zip([0] + cuts, cuts + [n]))


def test_one_pass_for_several_k_matches_scoring_every_item_alone(tiny):
    torch, model = tiny
    p_ids, r_ids, steps = random_chain()
    ks = [1, 2, 4]
    items, index = ln.unique_items(steps, ks)
    means, splits = ln.step_logprob_means(model, p_ids, r_ids, items)
    assert splits == 0 and len(items) < len(ks) * len(steps)
    scores = ln.combine(means, index)
    for k in ks:
        ref = naive_means(torch, model, p_ids, r_ids, steps, k)
        assert [means[j] for j in index[k]] == pytest.approx(ref, abs=1e-4)
        assert scores[k] == pytest.approx(mean_over_steps(ref), abs=1e-4)
    assert len({round(scores[k], 6) for k in ks}) == len(ks)              # k khác nhau thì điểm khác nhau


def test_batching_and_memory_budget_do_not_change_the_scores(tiny):
    _torch, model = tiny
    p_ids, r_ids, steps = random_chain(seed=5)
    items, _ = ln.unique_items(steps, [1, 4])
    ref, _ = ln.step_logprob_means(model, p_ids, r_ids, items)
    for kw in ({"max_batch": 1}, {"max_batch": 3}, {"attn_gb": 1e-9}, {"head_chunk": 7}):
        got, _ = ln.step_logprob_means(model, p_ids, r_ids, items, **kw)
        assert got == pytest.approx(ref, abs=1e-4)


def test_full_context_recombined_by_token_equals_the_whole_chain_log_probability(tiny):
    """Khi k lớn hơn số bước, mỗi bước thấy toàn bộ phần trước; gộp điểm các bước theo số token phải ra đúng log
    xác suất trung bình của cả chuỗi (GRAPE). Phép này bắt mọi lỗi lệch chỉ số token giữa các bước."""
    torch, model = tiny
    p_ids, r_ids, steps = random_chain(seed=9)
    k = len(steps) + 3
    items, index = ln.unique_items(steps, [k])
    means, _ = ln.step_logprob_means(model, p_ids, r_ids, items)
    n_tok = [e - s for s, e in steps]
    by_token = sum(means[j] * n for j, n in zip(index[k], n_tok)) / sum(n_tok)
    x = torch.tensor([p_ids + r_ids])
    with torch.no_grad():
        lg = model(x).logits[0, len(p_ids) - 1:-1].float()
    grape = float(torch.log_softmax(lg, -1).gather(-1, x[0, len(p_ids):].unsqueeze(-1)).mean())
    assert by_token == pytest.approx(grape, abs=1e-4)
    single, _ = ln.step_logprob_means(model, p_ids, r_ids, [(0, 0, len(r_ids))])
    assert single[0] == pytest.approx(grape, abs=1e-4)                    # chuỗi một bước: LocalNat bằng GRAPE


def test_score_chain_uses_the_given_step_ends_and_truncates_like_b1_fit(tiny):
    _torch, model = tiny
    tok = FakeTok(211)
    text = "alpha beta gamma delta.\n\nepsilon zeta eta theta iota.\nkappa lambda mu nu xi omicron."
    ends = [text.index("epsilon"), text.index("kappa"), len(text)]
    row = ln.score_chain(model, tok, "What is it?", text, ends, [1, 4], max_len=3072)
    assert row["n_steps"] == 3 == row["n_steps_text"] and row["step_tokens"] == [5, 6, 6] and row["n_tokens"] == 17
    assert row["n_items"] == 4 and len(row["step_logp_k1"]) == 3 and row["local_nat_k1"] != row["local_nat_k4"]
    assert row["step_logp_k1"][:2] == row["step_logp_k4"][:2]              # hai bước đầu có cùng ngữ cảnh ở mọi k
    one = ln.score_chain(model, tok, "What is it?", text, [len(text)], [1, 4], max_len=3072)
    assert one["n_steps"] == 1 and one["local_nat_k1"] == one["local_nat_k4"]
    n_prompt = len(tok(tok.apply_chat_template([{"content": ln.SYSTEM_PROMPT_TRAIN}, {"content": "What is it?"}]))["input_ids"])
    cut = ln.score_chain(model, tok, "What is it?", text, ends, [1, 4], max_len=n_prompt + 6)
    assert cut["n_tokens"] == 6 and cut["n_steps"] == 2 < cut["n_steps_text"]   # bước cuối rơi ngoài độ dài tối đa
    assert ln.score_chain(model, tok, "What is it?", text, ends, [1, 4], max_len=n_prompt + 1) is None


def test_full_run_writes_one_row_per_chain_resumes_and_keeps_the_fit_file_untouched(tiny, tmp_path, monkeypatch, capsys):
    torch, model = tiny
    cfg = make_workdir(tmp_path, n_q=4)
    rows = split_rows(tmp_path, broken={0})
    fit_before = (tmp_path / "fit.qwen1_5b_base.jsonl").read_bytes()
    loads = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("src.stage_b.b1_fit.load_student", lambda mid, four: (loads.append(mid), (model, FakeTok(211)))[1])
    args = ["--workdir", str(tmp_path), "--student", "qwen1_5b", "--base", "--max-fallback-share", "0.5"]

    assert ln.main(args + ["--limit", "5"]) == 0
    out_path = tmp_path / "localnat.qwen1_5b_base.glm.jsonl"
    assert len(read_jsonl(out_path)) == 5
    assert ln.main(args) == 0                                              # chạy lại: làm tiếp, không chấm lại
    got = read_jsonl(out_path)
    assert [r["tid"] for r in got] == [r["tid"] for r in rows] and loads == [cfg.student.base_model_id] * 2
    assert got[0]["steps_source"].startswith("fallback") and {r["steps_source"] for r in got[1:]} == {"glm"}
    assert all(r["model"] == cfg.student.base_model_id and r["steps_version"] == st.STEPS_VERSION
               and "local_nat_k1" in r and "local_nat_k4" in r and len(r["step_tokens"]) == r["n_steps"] for r in got)
    assert (tmp_path / "fit.qwen1_5b_base.jsonl").read_bytes() == fit_before
    assert "local_nat_k4 trên" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="cài đặt khác"):                  # đổi bộ k rồi chạy bù lên file cũ
        ln.main(args + ["--local-k", "1", "2"])

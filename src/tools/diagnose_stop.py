"""diagnose_stop: vì sao mô hình sau tinh chỉnh không dừng ở token kết thúc lượt. Chạy trên máy có GPU.

    python -m src.tools.diagnose_stop --student qwen1_5b --method correct_only --seed 42
    python -m src.tools.diagnose_stop --student qwen1_5b --method correct_only --seed 42 --skip-generate   # vài phút
    python -m src.tools.diagnose_stop --student qwen1_5b --method correct_only --seed 42 --n-gen 48        # ước lượng chặt hơn
    python -m src.tools.diagnose_stop --student qwen1_5b --method correct_only --seed 42 --pilot 500       # adapter của lần chạy thử ngắn

Đọc  outputs/models/<student>_base/<tên>/seed<seed>/adapter   (adapter LoRA mà b3_train đã ghi)
     data/stage_b/<student>_base/train.<tên>.jsonl            (tập chọn đã dùng để huấn luyện)
Ghi  outputs/results/diagnose_stop/<student>_base/<tên>/<tên thư mục lần chạy>/     (seed42, pilot500_seed42, ...)
        report.txt          đúng nội dung in ra màn hình, để dán vào chat
        summary.json        mọi con số của báo cáo
        chain_ends.jsonl    một dòng mỗi chuỗi huấn luyện được đo ở phần 2
        generations.jsonl   một dòng mỗi lượt sinh thử ở phần 3, kèm toàn văn

Ba phép đo, mỗi phép trả lời một câu hỏi:

  1. HÀNG NHÚNG. LoRA chỉ cập nhật các lớp chiếu, không cập nhật ma trận nhúng và lớp đầu ra. Logit của token
     kết thúc lượt là tích vô hướng giữa trạng thái ẩn cuối và hàng của token đó trong lớp đầu ra. Nếu bản nền
     chưa từng học token này thì hàng của nó giống các hàng chưa bao giờ được dùng (các id nằm ngoài bộ từ vựng
     của tokenizer), và LoRA phải tự xoay trạng thái ẩn về một hướng mà mô hình chưa từng dùng. Phép đo so chuẩn
     và hướng của hàng <|im_end|> với ba nhóm: token thường, token thêm vào khác, và hàng chưa từng dùng.

  2. CUỐI CHUỖI HUẤN LUYỆN. Đưa lại chính các chuỗi đã huấn luyện vào mô hình (không sinh), đọc xác suất của
     token kết thúc lượt và của eos_token tại đúng vị trí mà nhãn là token kết thúc lượt, khi tắt LoRA (bản nền)
     và khi bật LoRA (sau huấn luyện). Đây là chỗ mô hình đã thấy đáp án ba lần: nếu ở đây xác suất vẫn thấp thì
     việc học token này chưa xong, không phải do mô hình tự sinh lạc hướng. Kèm "xác suất dừng khi lấy mẫu": xác
     suất rút trúng một token dừng sau khi áp nhiệt độ và top-p của cài đặt đánh giá, là con số ứng trực tiếp
     với tỷ lệ dừng của phép thử.

  3. SINH THỬ LẠI. Sinh lại đúng các lượt của phép thử dừng trong b3_train (cùng đề, cùng seed, cùng lô), rồi
     chấm từng vị trí của văn bản đã sinh: xác suất dừng ngay sau \\boxed đầu tiên, và ở phần còn lại. In phần
     văn bản mà các lượt không dừng viết sau \\boxed đầu tiên.

Mọi xác suất ghi "T = 1" là xác suất gốc của mô hình, chưa áp nhiệt độ.

Token kết thúc được đọc từ run.json của chính lần chạy đang chẩn đoán (lần chạy đó đã học token nào thì đo token
đó); run.json không ghi thì lấy theo training.end_token của cấu hình. Nhờ vậy công cụ đo được cả adapter cũ học
<|im_end|> lẫn adapter mới học <|endoftext|>.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import statistics
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Mapping, Sequence

from src.common.answers import normalize_answer
from src.common.config import load_config, path_of, resolve_path
from src.common.io_utils import ensure_dir, read_json, write_json, write_jsonl
from src.stage_b import b3_train as b3
from src.stage_b.b2_select import selection_tag, student_tag

EXIT_WINDOW = 40          # số token sau \boxed đầu tiên được coi là "lối ra" tự nhiên của lượt sinh
SHOW_AFTER, SHOW_LAST = 400, 200
KIND_LABEL = {"answer_last": "đáp án ở cuối", "text_after": "còn lời sau đáp án", "no_boxed": "không có \\boxed"}


# ============================================================ phần không cần torch
def boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """(vị trí đầu, vị trí ngay sau ngoặc đóng, nội dung) của mọi \\boxed{...} hoàn chỉnh, khớp ngoặc lồng nhau."""
    spans, i = [], 0
    while True:
        i = text.find("\\boxed", i)
        if i < 0:
            return spans
        j = i + len("\\boxed")
        while j < len(text) and text[j] == " ":
            j += 1
        if j >= len(text) or text[j] != "{":
            i = j
            continue
        depth, end = 0, None
        for k in range(j, len(text)):
            c = text[k]
            if text[k - 1] == "\\" and c in "{}":
                continue                       # \{ và \} là ký tự, không phải ngoặc nhóm
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = k + 1
                    break
        if end is None:
            return spans                       # ngoặc không đóng: chuỗi bị cắt giữa đáp án
        spans.append((i, end, text[j + 1:end - 1].strip()))
        i = end


def ending_kind(text: str) -> str:
    """Chuỗi kết thúc thế nào: đáp án \\boxed nằm ở cuối, hay sau nó còn lời văn."""
    spans = boxed_spans(text)
    if not spans:
        return "no_boxed"
    tail = re.sub(r"\\[A-Za-z]+\*?(\{[^{}]*\})?", " ", text[spans[-1][1]:])     # bỏ lệnh LaTeX như \end{align*}
    return "text_after" if re.search(r"[A-Za-z]{2,}", tail) else "answer_last"


def token_at_char(prefix_len: Callable[[int], int], n_tokens: int, char_pos: int) -> int:
    """Số token ít nhất j sao cho văn bản giải mã từ j token đầu dài ít nhất char_pos ký tự (tìm nhị phân)."""
    lo, hi = 0, n_tokens
    while lo < hi:
        mid = (lo + hi) // 2
        if prefix_len(mid) >= char_pos:
            hi = mid
        else:
            lo = mid + 1
    return lo


def repeat_share(text: str, min_len: int = 12) -> float:
    """Tỷ lệ dòng lặp lại trong một đoạn văn: 0 là không dòng nào trùng, gần 1 là quay vòng một vài dòng."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= min_len]
    return 0.0 if len(lines) < 4 else 1.0 - len(set(lines)) / len(lines)


def pick_samples(rows: Sequence[Mapping], n: int, seed: int) -> list:
    """n chuỗi huấn luyện theo thứ tự cố định sha256 của seed|diagnose|tid (không phụ thuộc PYTHONHASHSEED)."""
    return sorted(rows, key=lambda r: hashlib.sha256(f"{seed}|diagnose|{r['tid']}".encode()).hexdigest())[:n]


def describe(values: Sequence[float]) -> dict:
    """Trung bình, trung vị, phân vị 10 và 90, và tỷ lệ giá trị từ 0,5 trở lên."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "share_ge_half": None}
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"n": len(xs), "mean": sum(xs) / len(xs), "median": statistics.median(xs), "p10": q(0.10), "p90": q(0.90),
            "share_ge_half": sum(x >= 0.5 for x in xs) / len(xs)}


def stop_curve(logp_rows: Sequence[Sequence[float]], n_prompt: int) -> list[float]:
    """Xác suất dừng theo từng vị trí của phần đã sinh. Phần tử j là xác suất token mới thứ j (đếm từ 0) là một
    token dừng; phần tử cuối là xác suất dừng ngay sau token cuối cùng. Độ dài bằng số token mới cộng 1."""
    return [sum(math.exp(v) for v in row) for row in logp_rows[n_prompt - 1:]]


def exit_profile(curve: Sequence[float], j_exit: int | None, window: int = EXIT_WINDOW) -> dict:
    """Xác suất dừng lớn nhất ở ba đoạn: trước lối ra, trong cửa sổ lối ra, và sau đó. j_exit là số token tính
    đến hết \\boxed đầu tiên; None nếu lượt sinh không có \\boxed hoàn chỉnh."""
    top = lambda xs: max(xs) if xs else None
    if j_exit is None:
        return {"before": top(curve), "at_exit": None, "after": None, "after_above_0.1": None}
    after = curve[j_exit + window + 1:]
    return {"before": top(curve[:j_exit]), "at_exit": top(curve[j_exit:j_exit + window + 1]), "after": top(after),
            "after_above_0.1": sum(p > 0.1 for p in after)}


def describe_generation(kept: Sequence[int], stop, text: str, curve: Sequence[float], j_exit: int | None) -> dict:
    """Mô tả một lượt sinh: có dừng không, mấy \\boxed, mấy đáp án khác nhau, văn bản viết sau \\boxed đầu tiên."""
    spans = boxed_spans(text)
    answers = [normalize_answer(c) for _s, _e, c in spans]
    after = text[spans[0][1]:] if spans else ""
    return {"stopped": stop is not None, "n_new": len(kept), "n_boxed": len(spans),
            "tokens_after_first_boxed": None if j_exit is None else len(kept) - j_exit,
            "distinct_answers": len(set(answers)), "first_equals_last": (answers[0] == answers[-1]) if answers else None,
            "repeat_share": round(repeat_share(after), 3), "chars_after_first_boxed": len(after),
            "after_first_boxed": after[:SHOW_AFTER], "last_chars": text[-SHOW_LAST:],
            "p_stop": {k: (None if v is None else round(v, 4) if isinstance(v, float) else v)
                       for k, v in exit_profile(curve, j_exit).items()},
            "p_stop_final": round(curve[-1], 4) if curve else None}


# ============================================================ phần cần torch
def load_trained(cfg: Mapping, adapter_dir: Path, device: str):
    """Nạp mô hình học ở đúng độ chính xác mà b3_train đã huấn luyện, rồi gắn adapter đã lưu."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t, model_id = cfg["training"], cfg["student"]["base_model_id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map=device, dtype=torch.bfloat16)
    if device != "cpu":
        kw["attn_implementation"] = "sdpa"
    if t["load_in_4bit"]:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_id, **kw), str(adapter_dir))
    model.eval()
    return model, tok


def sampled_stop_prob(logits_row, stop_ids: Sequence[int], temperature: float, top_p: float) -> float:
    """Xác suất rút trúng một token dừng khi lấy mẫu với nhiệt độ rồi top-p, theo đúng thứ tự của generate:
    chia logit cho nhiệt độ, giữ các token xác suất cao nhất cho tới khi tổng vượt top_p, rồi chuẩn hoá lại."""
    import torch
    p = torch.softmax(logits_row.float() / temperature, dim=-1)
    sp, si = torch.sort(p, descending=True)
    keep = (torch.cumsum(sp, 0) - sp) < top_p
    kept = torch.zeros_like(p, dtype=torch.bool)
    kept[si[keep]] = True
    return float(sum(p[s] for s in stop_ids if bool(kept[s])) / p[kept].sum())


def next_token_logprobs(model, ids: Sequence[int], track: Sequence[int], chunk: int, detail_at: int | None = None,
                        sampling: tuple[float, float] | None = None, name=str):
    """log p(token kế tiếp = t | các token từ đầu đến vị trí i), cho mọi vị trí i và mọi t trong track (T = 1).

    Trả về (danh sách T dòng, mỗi dòng len(track) số; chi tiết tại vị trí detail_at hoặc None). Lớp đầu ra tính
    theo từng đoạn vị trí như b3_train, nên ma trận logit của cả chuỗi không bao giờ được dựng cùng lúc."""
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.no_grad():
        x = torch.tensor([list(ids)], device=base.lm_head.weight.device)
        hidden = base.model(input_ids=x).last_hidden_state[0]
        cols = torch.tensor(list(track), device=hidden.device)
        out, detail = [], None
        for i in range(0, hidden.shape[0], chunk):
            logits = base.lm_head(hidden[i:i + chunk]).float()
            logp = torch.log_softmax(logits, dim=-1)
            out.extend(logp[:, cols].tolist())
            if detail_at is not None and i <= detail_at < i + chunk:
                row, lrow = logits[detail_at - i], logp[detail_at - i]
                top_p, top_i = torch.topk(lrow, 5)
                detail = {"rank": int((row > row[cols[0]]).sum()) + 1,
                          "top": [[name(int(t)), round(math.exp(float(v)), 4)] for v, t in zip(top_p, top_i)],
                          "top1_id": int(top_i[0]),
                          "sampled_stop": sampled_stop_prob(row, list(track), *sampling) if sampling else None}
    return out, detail


def embedding_report(model, tok, named: Mapping[str, int], rows_per_chunk: int = 8192) -> dict:
    """Chuẩn và hướng của hàng nhúng của các token có tên, so với ba nhóm hàng: token thường, token thêm vào
    khác, và hàng chưa từng dùng (id từ len(tokenizer) trở lên: không văn bản nào chứa chúng)."""
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    w_in, w_out = base.get_input_embeddings().weight, base.lm_head.weight
    tied = w_in.data_ptr() == w_out.data_ptr() or (w_in.shape == w_out.shape and bool(torch.equal(w_in, w_out)))
    n_rows, n_tok = int(w_out.shape[0]), len(tok)
    added = sorted(int(i) for i in tok.added_tokens_decoder)
    first_added = min(added) if added else n_tok
    groups = {"ordinary": list(range(0, first_added)),
              "other_added": [i for i in added if i not in set(named.values()) and i < n_rows],
              "unused": list(range(n_tok, n_rows))}

    def analyse(w) -> dict:
        with torch.no_grad():
            norms = torch.cat([w[i:i + rows_per_chunk].float().norm(dim=1) for i in range(0, n_rows, rows_per_chunk)])
            means = {}
            for g, idx in groups.items():
                if idx:
                    ix = torch.tensor(idx, device=w.device)
                    total = sum(w[ix[i:i + rows_per_chunk]].float().sum(0) for i in range(0, len(idx), rows_per_chunk))
                    means[g] = total / len(idx)
            ordinary = norms[:first_added]
            res = {"groups": {g: {"rows": len(idx), "median_norm": float(norms[torch.tensor(idx, device=w.device)]
                                                                         .median()) if idx else None}
                              for g, idx in groups.items()}, "tokens": {}}
            cos = torch.nn.functional.cosine_similarity
            for label, tid in named.items():
                row = w[tid].float()
                res["tokens"][label] = {
                    "id": tid, "norm": float(norms[tid]),
                    "norm_percentile_among_ordinary": float((ordinary < norms[tid]).float().mean()),
                    **{f"cos_{g}_mean": float(cos(row, m, dim=0)) for g, m in means.items()}}
            labels = list(named)
            res["cos_between"] = {f"{a} ~ {b}": float(cos(w[named[a]].float(), w[named[b]].float(), dim=0))
                                  for i, a in enumerate(labels) for b in labels[i + 1:]}
        return res

    report = {"tied_input_output": bool(tied), "matrix_rows": n_rows, "tokenizer_size": n_tok,
              "output": analyse(w_out)}
    if not tied:
        report["input"] = analyse(w_in)
    return report


def chain_end_report(model, tok, rows: Sequence[Mapping], cfg: Mapping, eot_ids: Sequence[int], stops: Sequence[int],
                     n: int, seed: int) -> list[dict]:
    """Phần 2: xác suất dừng tại cuối các chuỗi huấn luyện, khi tắt và khi bật LoRA."""
    t, ev = cfg["training"], cfg["eval"]
    sampling = (float(ev["temperature"]), float(ev["top_p"]))
    name, recs = tok.convert_ids_to_tokens, []
    show = lambda i: tok.decode([i])                # chữ thật của token (xuống dòng, dấu cách), để đọc được token đứng đầu
    for r in pick_samples(rows, n, seed):
        s = b3.encode_sample(tok, r["question"], r["text"], int(t["max_seq_len"]), eot_ids)
        if s is None or s["truncated"]:
            continue                                # mẫu bị cắt không mang token kết thúc lượt
        end = len(s["ids"]) - 2                     # vị trí mà token kế tiếp là token kết thúc lượt
        rec = {"tid": r["tid"], "teacher": r.get("teacher"), "kind": ending_kind(r["text"]), "n_labels": s["n_labels"]}
        for label, ctx in (("base", model.disable_adapter()), ("tuned", nullcontext())):
            with ctx:
                lp, d = next_token_logprobs(model, s["ids"], stops, int(t["loss_chunk"]), end, sampling, show)
            inside = [sum(math.exp(v) for v in row) for row in lp[s["n_prompt"] - 1:end]]     # trước khi hết chuỗi
            rec[label] = {"p": {name(tid): math.exp(v) for tid, v in zip(stops, lp[end])}, "rank": d["rank"],
                          "top": d["top"], "eot_is_top1": d["top1_id"] == eot_ids[0],
                          "sampled_stop": d["sampled_stop"], "max_stop_inside": max(inside) if inside else 0.0}
        recs.append(rec)
    return recs


def generation_report(model, tok, rows: Sequence[Mapping], cfg: Mapping, stops: Sequence[int], n: int, seed: int,
                      max_new_tokens: int | None) -> list[dict]:
    """Phần 3: sinh lại các lượt của phép thử dừng, rồi chấm xác suất dừng tại từng vị trí của văn bản đã sinh."""
    items = b3.stop_check_items(rows, n, seed)
    new_rows = b3.generate_for_stop_check(model, tok, [it["question"] for it in items], cfg, stops, seed,
                                          max_new_tokens)
    model.eval()                                    # generate_for_stop_check trả mô hình về chế độ huấn luyện
    decode = lambda ids: tok.decode(ids, skip_special_tokens=True)
    special = set(int(i) for i in tok.added_tokens_decoder)
    recs = []
    for it, row in zip(items, new_rows):
        kept, stop = b3.split_at_stop(row, stops)
        text = decode(kept)
        prompt = tok.apply_chat_template(b3.prompt_messages(it["question"]), tokenize=False, add_generation_prompt=True)
        p_ids = list(tok(prompt, add_special_tokens=False)["input_ids"])
        lp, _ = next_token_logprobs(model, p_ids + kept, stops, int(cfg["training"]["loss_chunk"]))
        curve = stop_curve(lp, len(p_ids))
        spans = boxed_spans(text)
        j_exit = token_at_char(lambda m: len(decode(kept[:m])), len(kept), spans[0][1]) if spans else None
        rec = {"qid": it["qid"], "stop_token": None if stop is None else tok.convert_ids_to_tokens(stop),
               **describe_generation(kept, stop, text, curve, j_exit),
               "special_tokens_inside": dict(Counter(tok.convert_ids_to_tokens(t) for t in kept if t in special)),
               "curve_above_0.01": [[j, round(p, 4)] for j, p in enumerate(curve) if p > 0.01],
               "question": it["question"], "text": text}
        recs.append(rec)
    return recs


# ============================================================ báo cáo
def _f(x, digits: int = 3) -> str:
    return "  -  " if x is None else f"{x:.{digits}f}"


def _pct(x) -> str:
    return "  - " if x is None else f"{100 * x:.0f}%"


def embedding_lines(emb: Mapping) -> list[str]:
    out = [f"  lớp nhúng và lớp đầu ra {'DÙNG CHUNG một ma trận' if emb['tied_input_output'] else 'là hai ma trận riêng'}"
           f" | {emb['matrix_rows']} hàng, tokenizer có {emb['tokenizer_size']} token"]
    for side in ("output", "input"):
        if side not in emb:
            continue
        a = emb[side]
        g = a["groups"]
        out.append(f"  [{'lớp đầu ra' if side == 'output' else 'lớp nhúng'}] chuẩn trung vị: token thường "
                   f"{_f(g['ordinary']['median_norm'])} ({g['ordinary']['rows']} hàng) | token thêm vào khác "
                   f"{_f(g['other_added']['median_norm'])} ({g['other_added']['rows']}) | hàng chưa từng dùng "
                   f"{_f(g['unused']['median_norm'])} ({g['unused']['rows']})")
        out.append(f"    {'token':<16}{'chuẩn':>8}{'phân vị':>9}   cos với trung bình: {'thường':>7}{'thêm vào':>10}"
                   f"{'chưa dùng':>11}")
        for label, v in a["tokens"].items():
            out.append(f"    {label:<16}{v['norm']:>8.3f}{_pct(v['norm_percentile_among_ordinary']):>9}"
                       f"{'':>23}{_f(v.get('cos_ordinary_mean')):>7}{_f(v.get('cos_other_added_mean')):>10}"
                       f"{_f(v.get('cos_unused_mean')):>11}")
        out.append("    cos giữa các token: " + " | ".join(f"{k}: {_f(v)}" for k, v in a["cos_between"].items()))
    return out


def chain_end_lines(recs: Sequence[Mapping], eot: str, others: Sequence[str], sampling: tuple[float, float]) -> tuple:
    """(các dòng báo cáo, phần tóm tắt bằng số) cho phần 2."""
    if not recs:
        return ["  không có chuỗi nào để đo"], {}
    summary, out = {}, []
    out.append(f"  {'':<22}{'p(' + eot + ')':>24}" + "".join(f"{'p(' + o + ')':>18}" for o in others)
               + f"{eot + ' đứng đầu':>20}{'dừng khi lấy mẫu':>18}")
    out.append(f"  {'':<22}{'tb / trung vị / p10':>24}" + "".join(f"{'tb':>18}" for _ in others)
               + f"{'':>20}{f'T={sampling[0]}, top-p {sampling[1]}':>18}")
    for label, title in (("base", "bản nền (tắt LoRA)"), ("tuned", "sau huấn luyện")):
        d = describe([r[label]["p"][eot] for r in recs])
        top1 = sum(r[label]["eot_is_top1"] for r in recs) / len(recs)
        samp = describe([r[label]["sampled_stop"] for r in recs])
        oth = {o: describe([r[label]["p"][o] for r in recs])["mean"] for o in others}
        summary[label] = {"p_eot": d, "eot_top1_share": top1, "sampled_stop_mean": samp["mean"], "p_other_mean": oth,
                          "median_rank": statistics.median(r[label]["rank"] for r in recs),
                          "max_stop_inside_mean": describe([r[label]["max_stop_inside"] for r in recs])["mean"]}
        out.append(f"  {title:<22}{_f(d['mean']) + ' / ' + _f(d['median']) + ' / ' + _f(d['p10']):>24}"
                   + "".join(f"{_f(oth[o]):>18}" for o in others) + f"{_pct(top1):>20}{_f(samp['mean']):>18}")
    t = summary["tuned"]
    out.append(f"  sau huấn luyện: thứ hạng trung vị của {eot} là {t['median_rank']:.0f}; tỷ lệ chuỗi có p({eot}) từ 0,5 "
               f"trở lên: {_pct(t['p_eot']['share_ge_half'])}; xác suất dừng lớn nhất Ở GIỮA chuỗi (dừng non), trung "
               f"bình: {_f(t['max_stop_inside_mean'])}")
    for key, title in (("kind", "theo kiểu kết thúc"), ("teacher", "theo mô hình dạy")):
        parts = []
        for val, n in Counter(r[key] for r in recs).most_common():
            sub = [r for r in recs if r[key] == val]
            parts.append(f"{KIND_LABEL.get(val, val)} ({n}): p {_f(describe([r['tuned']['p'][eot] for r in sub])['mean'])}"
                         f", đứng đầu {_pct(sum(r['tuned']['eot_is_top1'] for r in sub) / n)}")
        summary[f"by_{key}"] = parts
        out.append(f"  {title}: " + " | ".join(parts))
    rivals = Counter(r["tuned"]["top"][0][0] for r in recs if not r["tuned"]["eot_is_top1"])
    summary["rivals"] = rivals.most_common(8)
    if rivals:
        out.append(f"  khi {eot} không đứng đầu, token đứng đầu là: "
                   + ", ".join(f"{rival!r} ×{c}" for rival, c in rivals.most_common(8)))
    return out, summary


def generation_lines(recs: Sequence[Mapping], recorded: Mapping | None, max_new: int) -> tuple:
    """(các dòng báo cáo, phần tóm tắt bằng số) cho phần 3."""
    stopped = [r for r in recs if r["stopped"]]
    loose = [r for r in recs if not r["stopped"]]
    by = Counter(r["stop_token"] for r in stopped)
    summary = {"n": len(recs), "stopped": len(stopped), "stopped_by": dict(by),
               "stopped_new_tokens": [r["n_new"] for r in stopped],
               "stopped_p_stop_final_mean": describe([r["p_stop_final"] for r in stopped])["mean"],
               "loose_with_boxed": sum(r["n_boxed"] > 0 for r in loose),
               "loose_first_differs_from_last": sum(r["first_equals_last"] is False for r in loose),
               "loose_p_stop_at_exit": [r["p_stop"]["at_exit"] for r in loose],
               "loose_p_stop_after": [r["p_stop"]["after"] for r in loose]}
    at_exit = [r for r in stopped if r["tokens_after_first_boxed"] is not None
               and r["tokens_after_first_boxed"] <= EXIT_WINDOW]
    summary["stopped_with_boxed"] = sum(r["n_boxed"] > 0 for r in stopped)
    summary["stopped_within_exit_window"] = len(at_exit)
    rec_txt = f" (run.json ghi {recorded['stopped']}/{recorded['n']})" if recorded else ""
    out = [f"  dừng {len(stopped)}/{len(recs)}{rec_txt} | dừng bởi: "
           + (", ".join(f"{k} ×{v}" for k, v in by.items()) or "không lượt nào"),
           f"  lượt dừng: số token {sorted(summary['stopped_new_tokens'])}; {summary['stopped_with_boxed']} lượt có "
           f"\\boxed, {len(at_exit)} lượt dừng trong {EXIT_WINDOW} token sau \\boxed đầu; xác suất dừng tại chỗ đã dừng "
           f"(T = 1), trung bình {_f(summary['stopped_p_stop_final_mean'])}",
           f"  lượt không dừng: {len(loose)}, trong đó {summary['loose_with_boxed']} có \\boxed; \\boxed cuối khác "
           f"\\boxed đầu ở {summary['loose_first_differs_from_last']} lượt (bộ chấm lấy \\boxed cuối)"]
    for i, r in enumerate(recs):
        if r["stopped"]:
            continue
        p = r["p_stop"]
        extra = f" | token đặc biệt trong văn bản: {r['special_tokens_inside']}" if r["special_tokens_inside"] else ""
        if not r["n_boxed"]:
            out.append(f"  --- lượt {i} | {r['qid']} | {r['n_new']} token (trần {max_new}) | KHÔNG có \\boxed | p(dừng) "
                       f"lớn nhất cả lượt {_f(p['before'])}{extra}")
        else:
            out.append(f"  --- lượt {i} | {r['qid']} | {r['n_new']} token (trần {max_new}), trong đó "
                       f"{r['tokens_after_first_boxed']} token sau \\boxed đầu | {r['n_boxed']} \\boxed, "
                       f"{r['distinct_answers']} đáp án khác nhau | tỷ lệ dòng lặp {r['repeat_share']:.2f}")
            out.append(f"      p(dừng) lớn nhất: trước \\boxed đầu {_f(p['before'])} | trong {EXIT_WINDOW} token sau "
                       f"\\boxed đầu {_f(p['at_exit'])} | về sau {_f(p['after'])} (số vị trí trên 0,1: "
                       f"{p['after_above_0.1']}){extra}")
            out.append(f"      sau \\boxed đầu ({r['chars_after_first_boxed']} ký tự): {r['after_first_boxed']!r}")
        out.append(f"      cuối lượt: {r['last_chars']!r}")
    return out, summary


# ============================================================ dòng lệnh
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="qwen1_5b")
    ap.add_argument("--method", help="tên file trong configs/method/")
    ap.add_argument("--ablation", help="tên file trong configs/ablation/")
    ap.add_argument("--seed", type=int, help="seed của lần huấn luyện cần chẩn đoán; mặc định seed đầu")
    ap.add_argument("--seldir", help="thư mục tập chọn, mặc định data/stage_b/<student>_base")
    ap.add_argument("--run-dir", help="thư mục lần chạy (chứa adapter/ và run.json), mặc định theo b3_train")
    ap.add_argument("--pilot", type=int, default=0, metavar="N", help="chẩn đoán lần chạy thử ngắn b3_train --pilot N")
    ap.add_argument("--n-train", type=int, default=200, help="số chuỗi huấn luyện đo ở phần 2")
    ap.add_argument("--n-gen", type=int, help="số đề sinh thử ở phần 3; mặc định bằng phép thử dừng của b3_train")
    ap.add_argument("--max-new-tokens", type=int, help="trần token khi sinh thử; mặc định eval.max_new_tokens")
    ap.add_argument("--skip-generate", action="store_true", help="bỏ phần 3 (phần lâu nhất)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)
    if not (args.method or args.ablation):
        ap.error("cần --method hoặc --ablation")

    import torch

    cfg = load_config(method=args.method or "qd_rsr", ablation=args.ablation, student=args.student,
                      overrides=args.override)
    seed = args.seed if args.seed is not None else int(cfg["project"]["train_seeds"][0])
    tag = selection_tag(cfg)
    sel_dir = resolve_path(cfg, args.seldir) if args.seldir else path_of(cfg, "data_stage_b") / student_tag(cfg)
    rows, _meta = b3.load_selection(sel_dir, tag, int(cfg["selection"]["k"]))
    run = resolve_path(cfg, args.run_dir) if args.run_dir else b3.run_dir(cfg, tag, seed, pilot=args.pilot)
    if not (run / "adapter").exists():
        raise SystemExit(f"Không thấy adapter ở {run / 'adapter'}. Lần huấn luyện này chưa chạy xong trên máy này.")
    run_info = read_json(run / "run.json") if (run / "run.json").exists() else {}
    recorded = run_info.get("stop_check")
    out_dir = ensure_dir(path_of(cfg, "out_results") / "diagnose_stop" / student_tag(cfg) / tag / run.name)

    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(cfg["hardware"]["torch_memory_fraction"]))
    t0 = time.perf_counter()
    model, tok = load_trained(cfg, run / "adapter", args.device)
    eot_ids = [int(i) for i in (run_info.get("end_of_turn_ids")
                                or b3.end_token_ids(tok, str(cfg["training"]["end_token"])))]
    stops = b3.stop_token_ids(tok, eot_ids)
    name = tok.convert_ids_to_tokens
    eot, others = name(eot_ids[0]), [name(s) for s in stops[1:]]
    ev = cfg["eval"]
    sampling = (float(ev["temperature"]), float(ev["top_p"]))
    lines: list[str] = []

    def emit(new: Sequence[str]) -> None:
        for ln in new:
            print(ln, flush=True)
        lines.extend(new)

    emit([f"[chẩn đoán dừng] {tag} | {cfg['student']['base_model_id']} | seed {seed} | adapter {run / 'adapter'}",
          f"  token kết thúc mà lần chạy này đã học: {eot} (id {eot_ids[0]}"
          f"{', theo run.json' if run_info.get('end_of_turn_ids') else ', theo cấu hình'}) | eos_token: {tok.eos_token} "
          f"(id {tok.eos_token_id}) | pad_token: {tok.pad_token} | token làm việc sinh dừng: {[name(s) for s in stops]}"])
    summary: dict = {"tag": tag, "seed": seed, "student": cfg["student"]["base_model_id"], "end_of_turn": eot,
                     "stops": [name(s) for s in stops], "recorded_stop_check": recorded}

    emit(["", "[1] Hàng nhúng của các token đặc biệt trong bản nền (LoRA không đụng tới hai ma trận này)"])
    named = {name(s): s for s in stops}
    extra = b3.end_of_turn_ids(tok) + [tok.convert_tokens_to_ids("<|im_start|>")]     # token của khuôn hội thoại
    for tid in extra:
        if isinstance(tid, int) and tid >= 0 and tid not in named.values() and tid != tok.unk_token_id:
            named[name(tid)] = tid
    summary["embedding"] = embedding_report(model, tok, named)
    emit(embedding_lines(summary["embedding"]))

    emit(["", f"[2] Cuối chuỗi huấn luyện, không sinh: xác suất của token kế tiếp tại chỗ nhãn là {eot} (T = 1)"])
    ends = chain_end_report(model, tok, rows, cfg, eot_ids, stops, args.n_train, seed)
    kinds = Counter(ending_kind(r["text"]) for r in rows)
    summary["ending_kinds_all_rows"] = dict(kinds)
    emit([f"  {len(ends)} chuỗi không bị cắt, rút cố định từ {len(rows)} mẫu của {tag} | kiểu kết thúc trên cả "
          f"{len(rows)} mẫu: " + ", ".join(f"{KIND_LABEL[k]} {_pct(c / len(rows))}" for k, c in kinds.most_common())])
    end_lines, summary["chain_ends"] = chain_end_lines(ends, eot, others, sampling)
    emit(end_lines)
    write_jsonl(out_dir / "chain_ends.jsonl", ends)

    if not args.skip_generate:
        n_gen = args.n_gen or (recorded or {}).get("n") or int(cfg["training"]["stop_check_samples"])
        max_new = int(args.max_new_tokens or ev["max_new_tokens"])
        emit(["", f"[3] Sinh thử lại {n_gen} đề như b3_train (T = {sampling[0]}, top-p {sampling[1]}, trần {max_new} "
                  f"token), rồi chấm xác suất dừng tại từng vị trí của văn bản đã sinh (T = 1)"])
        gens = generation_report(model, tok, rows, cfg, stops, n_gen, seed, args.max_new_tokens)
        gen_lines, summary["generation"] = generation_lines(gens, recorded, max_new)
        emit(gen_lines)
        write_jsonl(out_dir / "generations.jsonl", gens)

    emit(["", f"[xong] {time.perf_counter() - t0:.0f} giây | kết quả ở {out_dir}"])
    write_json(out_dir / "summary.json", summary)
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""b3_train: tinh chỉnh mô hình học trên một tập chọn của b2_select.

    python -m src.stage_b.b3_train --selftest                                      # kiểm phần toán, cần torch, không cần dữ liệu
    python -m src.stage_b.b3_train --student qwen1_5b --method correct_only --plan # không cần GPU: số mẫu, số bước
    python -m src.stage_b.b3_train --student qwen1_5b --method correct_only --smoke # 32 mẫu, 2 bước, thử sinh; vài phút
    python -m src.stage_b.b3_train --student qwen1_5b --method correct_only --seed 42
    python -m src.stage_b.b3_train --student qwen1_5b --ablation fit_quality --seed 42
    python -m src.stage_b.b3_train --student qwen1_5b --method qd_rsr --seed 42 --override selection.lambda_div=0.15
    python -m src.stage_b.b3_train --student qwen1_5b --method correct_only --seed 42 --resume   # chạy tiếp sau khi đứt
    python -m src.stage_b.b3_train --student qwen1_5b --method correct_only --seed 42 --pilot 500 # chạy thử ngắn, khoảng 20 phút

Đọc  data/stage_b/<student>_base/train.<tên>.jsonl và select.<tên>.json (tên do b2_select.selection_tag đặt)
Ghi  outputs/models/<student>_base/<tên>/seed<seed>/
        adapter/          trọng số LoRA cuối cùng
        run.json          cấu hình đã dùng, số mẫu, số token, thời gian, token/giây, đỉnh bộ nhớ, phép thử dừng
        train_log.jsonl   một dòng mỗi bước tối ưu: mất mát, tốc độ học, token/giây
        ckpt/             điểm lưu để chạy tiếp, xoá khi chạy xong
     --pilot N ghi vào .../<tên>/pilot<N>_seed<seed>/ và không đụng tới thư mục của lần chạy thật.

Các quy ước đã chốt mà mã này thực thi:
  - Định dạng mẫu giống hệt lúc b1_fit chấm tín hiệu: [system] câu SYSTEM_PROMPT_TRAIN, [user] đề bài, rồi chuỗi
    suy luận ở vai trò assistant. Sau chuỗi là TOKEN KẾT THÚC (chốt 02/10: có học). Mất mát chỉ tính trên chuỗi
    và token kết thúc, không tính trên đề. Mẫu phải cắt vì vượt training.max_seq_len thì KHÔNG gắn token kết thúc
    (không dạy mô hình dừng giữa chừng).
  - Token kết thúc là eos_token của tokenizer, với Qwen bản nền là <|endoftext|> (training.end_token: eos, chốt
    03/10). Trước đó mã lấy token mà khuôn hội thoại đặt sau câu trả lời, tức <|im_end|>. Lần chạy thử
    Correct-Only seed 42 và diagnose_stop cho thấy cách đó không dùng được với bản nền: hàng của <|im_end|>
    trong lớp đầu ra trùng với 271 hàng chưa từng dùng (chuẩn 0,414, cos 1,000), LoRA không cập nhật lớp đầu ra,
    nên sau ba epoch xác suất của <|im_end|> ở cuối chuỗi vẫn là 0 và chỉ 7/16 lượt sinh thử dừng (đều nhờ
    <|endoftext|> còn sót). Bản nền chưa huấn luyện đã cho <|endoftext|> xác suất 0,961 ở đúng vị trí đó.
    Phần đề vẫn dùng khuôn hội thoại như cũ, nên tín hiệu của b1_fit không phải chấm lại. Trước khi huấn luyện,
    mã kiểm hàng của token kết thúc trong lớp đầu ra và từ chối chạy nếu nó trùng hướng với các hàng chưa từng
    dùng (end_token_row), để lỗi này không lặp lại với mô hình học khác.
  - pad_token của Qwen cũng là <|endoftext|>. Phần đệm được che theo VỊ TRÍ (pad_batch), không theo id của
    token, nên nhãn của token kết thúc không bị che. Bộ ghép lô nào che nhãn theo id của pad_token sẽ xoá mất
    nhãn này và mô hình lại không dừng.
  - Mọi phương án đi chung một nhánh tính mất mát. Mất mát của một lô hiệu dụng là trung bình theo token có trọng
    số: Σ_i w_i Σ_t CE(i,t) / Σ_i w_i n_i, với w_i là cột weight của file train (1 cho mọi phương án, trọng số
    mềm cho LARK) và n_i là số token được tính của mẫu i. Với w_i = 1, đây đúng là mất mát chuẩn của tinh chỉnh
    có giám sát (trung bình trên mọi token của lô).
    Đối chiếu mã LARK chính thức ngày 03/10 (github.com/Tianrun-Yu/LARK, commit 38cfd4f): công thức này trùng
    với lark/train/train_full.py, bản tinh chỉnh toàn bộ tham số mà tài liệu của họ ghi là dùng cho kết quả
    chính: trọng số của mẫu được gán cho từng token rồi chia cho tổng trọng số của các token có nhãn. Bản LoRA
    của họ (lark/train/train_lora.py) làm khác: lấy trung bình theo token trong từng mẫu trước, rồi mới lấy
    trung bình có trọng số giữa các mẫu. Khác biệt còn lại so với train_full.py: họ chia mẫu số trong từng lô
    nhỏ (một chuỗi đã ghép nhiều mẫu), còn ở đây chia trên cả lô hiệu dụng.
  - Mẫu số lấy trên CẢ lô hiệu dụng (batch_size x grad_accum mẫu), tính trước khi chạy, nên tích luỹ gradient cho
    đúng gradient của một lô lớn thật. Hệ quả: cách chia 16 mẫu của một lô hiệu dụng thành các lô nhỏ không làm
    đổi gradient. Mã dùng điều đó để ghép các mẫu gần độ dài vào cùng lô nhỏ, bớt phần đệm mà không đổi gì về
    mặt tối ưu: mỗi lô hiệu dụng vẫn là 16 mẫu rút ngẫu nhiên.
  - Độ chính xác đọc từ cấu hình của từng mô hình học (training.load_in_4bit). 1,5 tỷ: trọng số bfloat16 không
    lượng tử, KHÔNG gọi prepare_model_for_kbit_training (hàm đó nâng cả mô hình lên 32-bit). 7 tỷ: lượng tử nf4
    như b1_fit --load-4bit, rồi hạ các tham số 32-bit xuống bfloat16. Đây là hai đường đã đo bằng check_vram.
  - Thứ tự mẫu chỉ phụ thuộc seed và số thứ tự epoch; hạt giống của dropout đặt lại ở mỗi bước tối ưu theo seed
    và số bước. Nhờ vậy chạy tiếp từ điểm lưu đi đúng con đường của lần chạy không đứt.
  - Từ chối tập chọn có trường blocked (ví dụ LocalNat bản xấp xỉ), và từ chối file train có md5 khác md5 ghi
    trong select.<tên>.json (file chép giữa hai máy bị hỏng hoặc không cùng lần chọn).

Phép thử dừng: sau khi huấn luyện, mô hình sinh thử cho vài đề bài với cài đặt đánh giá. Tỷ lệ lượt sinh dừng ở
token kết thúc được ghi vào run.json. Nếu mô hình không dừng thì mỗi câu đánh giá sẽ sinh đủ
eval.max_new_tokens, nên phải biết điều này TRƯỚC khi chạy cả loạt.

Chạy thử ngắn (--pilot N): huấn luyện đủ số epoch trên khoảng N mẫu (trọn câu hỏi, rút cố định theo seed), rồi
làm phép thử dừng trên ĐÚNG các đề của lần chạy thật. Dùng để kiểm một thay đổi của cách huấn luyện trong vài chục
phút trước khi bỏ ra gần 4 giờ cho một lần chạy đủ. Kết quả của nó không phải số liệu của phương án.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from src.common.config import load_config, path_of, resolve_path
from src.common.io_utils import JsonlWriter, ensure_dir, now_iso, read_json, read_jsonl, write_json
from src.common.prompts import SYSTEM_PROMPT_TRAIN
from src.stage_b.b2_select import selection_tag, student_tag

IGNORE = -100                    # nhãn bị bỏ qua khi tính mất mát (đề bài và phần đệm)
PROBE = "\u2400PROBE\u2400"      # nội dung giả để dò phần đuôi mà khuôn hội thoại gắn sau câu trả lời
SMOKE_SAMPLES, SMOKE_NEW_TOKENS = 32, 256


# ============================================================ phần không cần torch
def prompt_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT_TRAIN}, {"role": "user", "content": question}]


def end_of_turn_ids(tok) -> list[int]:
    """Token mà khuôn hội thoại đặt ngay sau câu trả lời của assistant (với Qwen là <|im_end|>).

    Lấy từ khuôn chứ không viết cứng, để đổi mô hình học thì không phải sửa mã. Khuôn không gắn gì thì dùng
    eos_token của tokenizer.
    """
    full = tok.apply_chat_template(prompt_messages("x") + [{"role": "assistant", "content": PROBE}], tokenize=False)
    tail = full.split(PROBE)[-1].strip() if PROBE in full else ""
    ids = tok(tail, add_special_tokens=False)["input_ids"] if tail else []
    if not ids:
        if tok.eos_token_id is None:
            raise SystemExit("Tokenizer không có token kết thúc lượt lẫn eos_token; không dạy mô hình dừng được.")
        ids = [tok.eos_token_id]
    return list(ids[:1])         # chỉ token đầu của phần đuôi: các ký tự xuống dòng sau đó không cần học


END_TOKEN_MODES = ("eos", "chat_template")
UNTRAINED_COS = 0.95             # cos với trung bình các hàng chưa từng dùng từ mức này trở lên là "chưa được học"


def end_token_ids(tok, mode: str) -> list[int]:
    """Token gắn sau mỗi chuỗi huấn luyện, theo training.end_token.

    "eos"            eos_token của tokenizer (Qwen bản nền: <|endoftext|>). Cách đã chốt 03/10.
    "chat_template"  token mà khuôn hội thoại đặt sau câu trả lời (<|im_end|>). Chỉ dùng được khi mô hình đã học
                     token đó, tức bản Instruct; giữ lại để tái lập lần chạy thử đầu tiên.
    """
    if mode == "eos":
        if tok.eos_token_id is None:
            raise SystemExit("Tokenizer không có eos_token; không dạy mô hình dừng được.")
        return [int(tok.eos_token_id)]
    if mode == "chat_template":
        return end_of_turn_ids(tok)
    raise SystemExit(f"training.end_token = '{mode}' không hợp lệ. Hiện có: {list(END_TOKEN_MODES)}")


def end_token_is_untrained(row: Mapping, limit: float = UNTRAINED_COS) -> bool:
    """Hàng của token kết thúc có trùng hướng với các hàng chưa từng dùng không (kết quả của end_token_row)."""
    cos = row.get("cos_unused_mean")
    return cos is not None and cos >= limit


def pilot_rows(rows: Sequence[Mapping], n: int, seed: int) -> list:
    """Khoảng n mẫu cho lần chạy thử ngắn: lấy trọn từng câu hỏi theo thứ tự cố định sha256 của seed|pilot|qid
    cho tới khi đủ n mẫu, giữ nguyên thứ tự dòng của file train."""
    counts = Counter(r["qid"] for r in rows)
    chosen, total = set(), 0
    for qid in sorted(counts, key=lambda q: hashlib.sha256(f"{seed}|pilot|{q}".encode()).hexdigest()):
        if total >= n:
            break
        chosen.add(qid)
        total += counts[qid]
    return [r for r in rows if r["qid"] in chosen]


def encode_sample(tok, question: str, text: str, max_len: int, eot_ids: Sequence[int]) -> dict | None:
    """Một mẫu huấn luyện: ids, nhãn (đề bài mang IGNORE), số token được tính, có bị cắt không.

    Phần đề và phần chuỗi được mã hoá đúng như b1_fit.build_inputs, để mô hình học gặp lúc tinh chỉnh đúng
    cái nó đã được chấm tín hiệu.
    """
    prompt = tok.apply_chat_template(prompt_messages(question), tokenize=False, add_generation_prompt=True)
    p_ids = list(tok(prompt, add_special_tokens=False)["input_ids"])
    r_ids = list(tok(text, add_special_tokens=False)["input_ids"])
    room = max_len - len(p_ids)
    if room < 2 or len(r_ids) < 2:
        return None
    truncated = len(r_ids) + len(eot_ids) > room
    tail = r_ids[:room] if truncated else r_ids + list(eot_ids)
    return {"ids": p_ids + tail, "labels": [IGNORE] * len(p_ids) + tail, "n_labels": len(tail),
            "n_prompt": len(p_ids), "truncated": truncated}


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Hệ số tốc độ học ở bước thứ step (đếm từ 0): tăng tuyến tính rồi giảm theo cosine về 0.
    Cùng công thức với get_cosine_schedule_with_warmup của transformers."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def _rng(*parts) -> random.Random:
    """Bộ sinh số ngẫu nhiên xác định từ các phần của khoá. Dùng hashlib, không dùng hash() của Python."""
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16], 16))


def epoch_windows(lengths: Sequence[int], seed: int, epoch: int, batch_size: int, grad_accum: int) -> list:
    """Chia một epoch thành các lô hiệu dụng, mỗi lô là danh sách các lô nhỏ (danh sách chỉ số mẫu).

    Mỗi lô hiệu dụng là batch_size x grad_accum mẫu rút ngẫu nhiên không lặp. BÊN TRONG một lô hiệu dụng, các
    mẫu được xếp theo độ dài rồi mới cắt thành lô nhỏ, để hai mẫu cùng lô nhỏ dài gần bằng nhau. Việc này không
    đổi gradient của lô hiệu dụng (xem đầu file), chỉ bớt phần đệm.
    """
    idx = list(range(len(lengths)))
    _rng("order", seed, epoch).shuffle(idx)
    eff = batch_size * grad_accum
    windows = []
    for i in range(0, len(idx), eff):
        group = sorted(idx[i:i + eff], key=lambda j: (lengths[j], j))
        windows.append([group[k:k + batch_size] for k in range(0, len(group), batch_size)])
    return windows


def window_denominator(samples: Sequence[Mapping]) -> float:
    """Mẫu số của một lô hiệu dụng: Σ w_i · n_i."""
    return float(sum(s["weight"] * s["n_labels"] for s in samples))


def pad_batch(samples: Sequence[Mapping], pad_id: int) -> dict:
    """Đệm bên phải tới độ dài của mẫu dài nhất. Trả về danh sách thuần, chưa phải tensor."""
    width = max(len(s["ids"]) for s in samples)
    return {"input_ids": [s["ids"] + [pad_id] * (width - len(s["ids"])) for s in samples],
            "attention_mask": [[1] * len(s["ids"]) + [0] * (width - len(s["ids"])) for s in samples],
            "labels": [s["labels"] + [IGNORE] * (width - len(s["labels"])) for s in samples],
            "weights": [float(s["weight"]) for s in samples]}


def load_selection(sel_dir: Path, tag: str, k: int) -> tuple[list, dict]:
    """Đọc file train của một tập chọn và kiểm nó đúng là file mà b2_select đã ghi."""
    meta_path, train_path = sel_dir / f"select.{tag}.json", sel_dir / f"train.{tag}.jsonl"
    if not meta_path.exists() or not train_path.exists():
        raise SystemExit(f"Không thấy tập chọn '{tag}' trong {sel_dir}. Chạy b2_select trước, rồi chép cả thư mục "
                         f"sang máy huấn luyện.")
    meta = read_json(meta_path)
    if meta.get("blocked"):
        raise SystemExit(f"Tập chọn '{tag}' đang bị khoá, không dùng để huấn luyện: {meta['blocked']}")
    md5 = hashlib.md5(train_path.read_bytes()).hexdigest()
    if md5 != meta["md5"]:
        raise SystemExit(f"{train_path.name} có md5 {md5}, khác md5 {meta['md5']} ghi trong {meta_path.name}. File "
                         f"chép giữa hai máy bị hỏng hoặc không cùng một lần chọn: chép lại cả hai file.")
    rows = read_jsonl(train_path)
    per_q = Counter(r["qid"] for r in rows)
    if len(rows) != meta["samples"] or set(per_q.values()) != {k}:
        raise SystemExit(f"{train_path.name}: {len(rows)} mẫu, số mẫu mỗi câu {sorted(set(per_q.values()))}; "
                         f"cấu hình đòi k = {k} và {meta['samples']} mẫu.")
    return rows, meta


def stop_token_ids(tok, eot_ids: Sequence[int]) -> list[int]:
    """Các token làm việc sinh dừng: token kết thúc đã học, rồi eos_token của tokenizer nếu là token khác."""
    return [eot_ids[0]] + ([tok.eos_token_id] if tok.eos_token_id not in (None, eot_ids[0]) else [])


def stop_check_items(rows: Sequence[Mapping], n: int, seed: int) -> list[dict]:
    """n đề bài dùng cho phép thử dừng: mỗi câu hỏi một lần, theo thứ tự cố định sha256 của seed|qid."""
    seen, items = set(), []
    for r in sorted(rows, key=lambda r: hashlib.sha256(f"{seed}|{r['qid']}".encode()).hexdigest()):
        if len(items) >= n:
            break
        if r["qid"] not in seen:
            seen.add(r["qid"])
            items.append({"qid": r["qid"], "question": r["question"]})
    return items


def split_at_stop(row: Sequence[int], stops: Sequence[int]) -> tuple[list[int], int | None]:
    """(phần token trước token dừng đầu tiên, token dừng đó). Không có token dừng thì trả cả dòng và None."""
    cut = next((p for p, t in enumerate(row) if t in stops), None)
    return (list(row), None) if cut is None else (list(row[:cut]), row[cut])


def summarise_stop(new_rows: Sequence[Sequence[int]], stops: Sequence[int], decode, max_new_tokens: int,
                   name=str) -> dict:
    """Tóm tắt phép thử dừng. stopped_by cho biết lượt dừng là do token nào; new_tokens là số token của từng lượt."""
    lengths, by, boxed = [], Counter(), 0
    for row in new_rows:
        kept, stop = split_at_stop(row, stops)
        lengths.append(len(kept))
        if stop is not None:
            by[name(stop)] += 1
        boxed += "\\boxed" in decode(kept)
    m, stopped = len(lengths), sum(by.values())
    return {"n": m, "stopped": stopped, "stopped_share": stopped / m if m else None, "with_boxed": boxed,
            "mean_new_tokens": sum(lengths) / m if m else None, "max_new_tokens": int(max_new_tokens),
            "stopped_by": dict(by), "new_tokens": lengths}


def run_dir(cfg: Mapping, tag: str, seed: int, smoke: bool = False, pilot: int = 0) -> Path:
    name = "smoke" if smoke else f"pilot{pilot}_seed{seed}" if pilot else f"seed{seed}"
    return path_of(cfg, "out_models") / student_tag(cfg) / tag / name


def total_steps(n_samples: int, cfg: Mapping) -> tuple[int, int]:
    """(số bước tối ưu cả lần chạy, số bước khởi động)."""
    t = cfg["training"]
    per_epoch = math.ceil(n_samples / (int(t["batch_size"]) * int(t["grad_accum"])))
    total = per_epoch * int(t["num_epochs"])
    return total, math.ceil(total * float(t["warmup_ratio"]))


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


# ============================================================ phần cần torch
def load_model(cfg: Mapping, seed: int):
    """Nạp mô hình học và gắn LoRA, theo đúng đường đã đo bằng check_vram cho từng độ chính xác."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t, model_id = cfg["training"], cfg["student"]["base_model_id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map="cuda", attn_implementation="sdpa", dtype=torch.bfloat16)
    if t["load_in_4bit"]:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(t["gradient_checkpointing"]))
        for prm in model.parameters():           # prepare_... nâng các lớp không lượng tử lên 32-bit; hạ lại
            if prm.dtype == torch.float32:
                prm.data = prm.data.to(torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    torch.manual_seed(seed)                      # khởi tạo LoRA phụ thuộc seed
    model = get_peft_model(model, LoraConfig(
        r=int(t["lora_r"]), lora_alpha=int(t["lora_alpha"]), lora_dropout=float(t["lora_dropout"]),
        task_type="CAUSAL_LM", target_modules=list(t["target_modules"])))
    if t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()                                # gradient checkpointing chỉ có tác dụng ở chế độ huấn luyện
    return model, tok


def end_token_row(model, n_tok: int, end_id: int) -> dict:
    """Hàng của token kết thúc trong lớp đầu ra, so với các hàng chưa từng dùng (id từ len(tokenizer) trở lên:
    không văn bản nào chứa chúng). LoRA không cập nhật lớp đầu ra, nên một token mà bản nền chưa học thì không
    thể được phát ra dù huấn luyện bao lâu. Mô hình không có hàng thừa thì không so được, các trường để None."""
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    w = base.lm_head.weight
    info = {"norm": None, "unused_rows": 0, "unused_median_norm": None, "cos_unused_mean": None}
    if w.dim() != 2 or not w.is_floating_point() or end_id >= w.shape[0]:
        return info
    with torch.no_grad():
        row = w[end_id].float()
        info["norm"] = float(row.norm())
        if w.shape[0] > n_tok:
            unused = w[n_tok:].float()
            info.update(unused_rows=int(unused.shape[0]), unused_median_norm=float(unused.norm(dim=1).median()),
                        cos_unused_mean=float(torch.nn.functional.cosine_similarity(row, unused.mean(0), dim=0)))
    return info


def weighted_chunked_loss(model, input_ids, attention_mask, labels, weights, chunk: int = 256):
    """TỔNG có trọng số của mất mát từng token: Σ_i w_i Σ_t CE(i, t). Người gọi chia cho mẫu số của lô hiệu dụng.

    Lớp đầu ra và mất mát tính theo từng đoạn vị trí, mỗi đoạn bọc trong checkpoint, như check_vram đã đo: ma
    trận logit của cả chuỗi 3.072 token không bao giờ được dựng cùng lúc. Đoạn nào không có nhãn (toàn đề bài
    hoặc phần đệm) thì bỏ qua.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    lm_head = base.lm_head
    w = weights.to(torch.float32).unsqueeze(1)
    total = torch.zeros((), dtype=torch.float32, device=hidden.device)
    width = input_ids.shape[1]

    def one(h, tgt, ww):
        ce = F.cross_entropy(lm_head(h).float().flatten(0, 1), tgt.flatten(), reduction="none", ignore_index=IGNORE)
        return (ce.view(tgt.shape) * ww).sum()

    for i in range(0, width - 1, chunk):
        j = min(i + chunk, width - 1)
        tgt = labels[:, i + 1:j + 1]
        if bool((tgt != IGNORE).any()):
            total = total + checkpoint(one, hidden[:, i:j], tgt, w, use_reentrant=False)
    return total


def to_tensors(batch: Mapping, device: str):
    import torch
    return (torch.tensor(batch["input_ids"], device=device), torch.tensor(batch["attention_mask"], device=device),
            torch.tensor(batch["labels"], device=device), torch.tensor(batch["weights"], device=device))


def save_ckpt(model, opt, sched, state: Mapping, path: Path) -> None:
    """Ghi vào thư mục tạm rồi đổi tên, để mất điện giữa chừng không để lại điểm lưu hỏng."""
    import torch
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(str(tmp / "adapter"))
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), **state}, tmp / "state.pt")
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def load_ckpt(model, opt, sched, path: Path) -> dict:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(str(path / "adapter" / "adapter_model.safetensors")))
    state = torch.load(path / "state.pt", map_location="cuda", weights_only=False)
    opt.load_state_dict(state.pop("opt"))
    sched.load_state_dict(state.pop("sched"))
    return state


def generate_for_stop_check(model, tok, questions: Sequence[str], cfg: Mapping, stops: Sequence[int], seed: int,
                            max_new_tokens: int | None = None, batch: int = 8) -> list[list[int]]:
    """Sinh thử cho từng đề bài với cài đặt đánh giá. Trả về token mới của từng đề, kể cả token dừng và phần đệm
    sau nó. Tách riêng để diagnose_stop sinh lại đúng những lượt này (cùng seed, cùng lô, cùng cài đặt)."""
    import torch

    ev = cfg["eval"]
    side, tok.padding_side = tok.padding_side, "left"
    model.eval()
    model.config.use_cache = True
    torch.manual_seed(seed)
    new_rows: list[list[int]] = []
    try:
        for i in range(0, len(questions), batch):
            prompts = [tok.apply_chat_template(prompt_messages(q), tokenize=False, add_generation_prompt=True)
                       for q in questions[i:i + batch]]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=float(ev["temperature"]),
                                     top_p=float(ev["top_p"]), top_k=max(0, int(ev["top_k"])),
                                     max_new_tokens=int(max_new_tokens or ev["max_new_tokens"]),
                                     eos_token_id=list(stops), pad_token_id=tok.pad_token_id)
            new_rows.extend(out[:, enc["input_ids"].shape[1]:].tolist())
    finally:
        tok.padding_side = side
        model.config.use_cache = False
        model.train()
    return new_rows


def check_stop(model, tok, rows: Sequence[Mapping], cfg: Mapping, eot_ids: Sequence[int], n: int, seed: int,
               max_new_tokens: int | None = None, batch: int = 8) -> dict:
    """Sinh thử cho n đề bài với cài đặt đánh giá, đo tỷ lệ lượt sinh dừng ở token kết thúc lượt."""
    stops = stop_token_ids(tok, eot_ids)
    items = stop_check_items(rows, n, seed)
    new_rows = generate_for_stop_check(model, tok, [it["question"] for it in items], cfg, stops, seed,
                                       max_new_tokens, batch)
    return summarise_stop(new_rows, stops, lambda ids: tok.decode(ids, skip_special_tokens=True),
                          int(max_new_tokens or cfg["eval"]["max_new_tokens"]), tok.convert_ids_to_tokens)


def train(cfg: Mapping, tag: str, rows: Sequence[Mapping], meta: Mapping, seed: int, out: Path, resume: bool,
          smoke: bool, n_stop: int, pilot: int = 0) -> dict:
    import torch

    t = cfg["training"]
    frac = float(cfg["hardware"]["torch_memory_fraction"])
    torch.cuda.set_per_process_memory_fraction(frac)     # vượt thì báo hết bộ nhớ, không âm thầm tràn sang RAM
    torch.cuda.reset_peak_memory_stats()
    model, tok = load_model(cfg, seed)
    eot_ids = end_token_ids(tok, str(t["end_token"]))
    eot_row = end_token_row(model, len(tok), eot_ids[0])
    if end_token_is_untrained(eot_row):
        raise SystemExit(
            f"Token kết thúc {tok.convert_ids_to_tokens(eot_ids)} chưa được mô hình này học: hàng của nó trong lớp "
            f"đầu ra có cos {eot_row['cos_unused_mean']:.3f} với trung bình của {eot_row['unused_rows']} hàng chưa "
            f"từng dùng (chuẩn {eot_row['norm']:.3f} so với {eot_row['unused_median_norm']:.3f}). LoRA không cập "
            f"nhật lớp đầu ra nên mô hình sẽ không dừng được. Đổi training.end_token (hiện là '{t['end_token']}').")
    max_len = int(t["max_seq_len"])

    samples, skipped = [], 0
    for r in rows[:SMOKE_SAMPLES] if smoke else pilot_rows(rows, pilot, seed) if pilot else rows:
        s = encode_sample(tok, r["question"], r["text"], max_len, eot_ids)
        if s is None:
            skipped += 1
            continue
        samples.append({**s, "weight": float(r["weight"]), "tid": r["tid"]})
    if skipped:
        raise SystemExit(f"{skipped} mẫu không còn chỗ cho chuỗi sau phần đề (max_seq_len {max_len}). Kiểm tra dữ liệu.")
    lengths = [len(s["ids"]) for s in samples]
    n_trunc = sum(s["truncated"] for s in samples)
    bs, accum, epochs = int(t["batch_size"]), int(t["grad_accum"]), 1 if smoke else int(t["num_epochs"])
    per_epoch = math.ceil(len(samples) / (bs * accum))
    total = 2 if smoke else per_epoch * epochs
    warmup = 0 if smoke else math.ceil(total * float(t["warmup_ratio"]))
    print(f"[b3] {tag} | {cfg['student']['base_model_id']} | {'QLoRA 4-bit' if t['load_in_4bit'] else 'LoRA 16-bit'} "
          f"| seed {seed} | {len(samples)} mẫu, {sum(lengths)} token, bị cắt {n_trunc} | lô {bs} x {accum}, "
          f"{total} bước tối ưu, khởi động {warmup} | token kết thúc {tok.convert_ids_to_tokens(eot_ids)}"
          + (f" | CHẠY THỬ NGẮN trên {len(samples)} mẫu" if pilot else ""))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(t["learning_rate"]), weight_decay=float(t["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, warmup))
    state = {"gstep": 0, "epoch": 0, "window": 0, "tokens": 0, "seconds": 0.0, "epoch_loss": []}
    ckpt = out / "ckpt"
    if resume and ckpt.exists():
        state = load_ckpt(model, opt, sched, ckpt)
        print(f"[b3] chạy tiếp từ bước {state['gstep']} (epoch {state['epoch'] + 1}, lô {state['window']})")
    log = JsonlWriter(out / "train_log.jsonl")
    speeds, loss_num, loss_den, warned = [], 0.0, 0.0, False
    gstep, start = state["gstep"], time.perf_counter()

    for epoch in range(state["epoch"], epochs):
        windows = epoch_windows(lengths, seed, epoch, bs, accum)
        for wi in range(state["window"] if epoch == state["epoch"] else 0, len(windows)):
            if gstep >= total:
                break
            window = windows[wi]
            members = [samples[j] for mb in window for j in mb]
            denom = window_denominator(members)
            torch.manual_seed(int(hashlib.sha256(f"{seed}|step|{gstep}".encode()).hexdigest()[:8], 16))
            t0, num = time.perf_counter(), 0.0
            for mb in window:
                ids, mask, labels, weights = to_tensors(pad_batch([samples[j] for j in mb], tok.pad_token_id), "cuda")
                part = weighted_chunked_loss(model, ids, mask, labels, weights, int(t["loss_chunk"]))
                (part / denom).backward()
                num += float(part.detach())
            torch.nn.utils.clip_grad_norm_(params, float(t["max_grad_norm"]))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            ntok = sum(len(s["ids"]) for s in members)
            gstep += 1
            state["tokens"] += ntok
            loss_num, loss_den = loss_num + num, loss_den + denom
            speed = ntok / secs
            speeds.append(speed)
            log.append({"step": gstep, "epoch": epoch + 1, "loss": num / denom, "lr": sched.get_last_lr()[0],
                       "tokens": ntok, "tok_per_s": round(speed, 1), "seconds": round(secs, 2)})
            if gstep % 10 == 0 or gstep == total or smoke:
                done = time.perf_counter() - start
                eta = done / (gstep - state["gstep"]) * (total - gstep) / 3600
                print(f"[b3] bước {gstep}/{total} | epoch {epoch + 1} | mất mát {num / denom:.4f} | "
                      f"{speed:.0f} token/giây | còn khoảng {eta:.1f} giờ", flush=True)
            if len(speeds) > 30 and not warned:
                ref = sorted(speeds[:20])[10]
                recent = sorted(speeds[-10:])[5]
                if recent < 0.5 * ref:
                    warned = True
                    print(f"[b3] CẢNH BÁO: tốc độ tụt còn {recent:.0f} token/giây so với {ref:.0f} lúc đầu. Thường là "
                          f"bộ nhớ GPU tràn sang RAM hệ thống: đóng các ứng dụng khác đang dùng GPU.", flush=True)
            if not smoke and gstep % int(t["save_every_steps"]) == 0 and gstep < total:
                save_ckpt(model, opt, sched, {**state, "gstep": gstep, "epoch": epoch, "window": wi + 1,
                                              "seconds": state["seconds"] + time.perf_counter() - start}, ckpt)
        if loss_den:
            state["epoch_loss"].append(loss_num / loss_den)
            loss_num = loss_den = 0.0
    log.close()

    seconds = state["seconds"] + time.perf_counter() - start
    model.save_pretrained(str(out / "adapter"))
    stop = check_stop(model, tok, rows, cfg, eot_ids, n_stop, seed, SMOKE_NEW_TOKENS if smoke else None) \
        if n_stop else None
    import peft
    import transformers
    result = {
        "tag": tag, "status": "smoke" if smoke else "pilot" if pilot else "done",
        "student": cfg["student"]["base_model_id"],
        "seed": seed, "finished": now_iso(), "git_commit": git_commit(Path(cfg["root"])),
        "train_file": meta["train_file"], "train_md5": meta["md5"], "selection": {
            key: meta.get(key) for key in ("name", "pool", "rule", "signal", "weights", "a", "b", "lambda_div", "k")},
        "training": {key: t[key] for key in (
            "method", "load_in_4bit", "lora_r", "lora_alpha", "lora_dropout", "target_modules", "learning_rate",
            "lr_scheduler", "warmup_ratio", "num_epochs", "max_seq_len", "batch_size", "grad_accum",
            "gradient_checkpointing", "weight_decay", "max_grad_norm", "loss_chunk")},
        "samples": len(samples), "truncated_samples": n_trunc, "tokens_per_epoch": sum(lengths),
        "label_tokens_per_epoch": sum(s["n_labels"] for s in samples), "optimizer_steps": gstep,
        "end_token_mode": str(t["end_token"]), "end_token_row": eot_row,
        "end_of_turn_token": tok.convert_ids_to_tokens(eot_ids), "end_of_turn_ids": list(eot_ids),
        "epoch_loss": state["epoch_loss"], "seconds": round(seconds, 1), "hours": round(seconds / 3600, 3),
        "tok_per_s": round(state["tokens"] / seconds, 1) if seconds else None,
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
        "memory_fraction_cap": frac, "stop_check": stop,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }
    write_json(out / "run.json", result)
    shutil.rmtree(ckpt, ignore_errors=True)
    return result


# ============================================================ tự kiểm phần toán (cần torch, không cần dữ liệu)
def selftest() -> int:
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace

    torch.manual_seed(0)
    vocab, hid = 37, 12

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.lin = torch.nn.Embedding(vocab, hid), torch.nn.Linear(hid, hid)

        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(last_hidden_state=torch.tanh(self.lin(self.emb(input_ids))))

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.lm_head = Inner(), torch.nn.Linear(hid, vocab, bias=False)

    net = Tiny()
    rng = random.Random(1)
    samples = []
    for i in range(16):
        n_p, n_r = rng.randint(2, 6), rng.randint(3, 20)
        ids = [rng.randrange(vocab) for _ in range(n_p + n_r)]
        samples.append({"ids": ids, "labels": [IGNORE] * n_p + ids[n_p:], "n_labels": n_r,
                        "weight": rng.choice([0.2, 1.0, 1.7, 3.0])})

    def direct(group):
        """Tính thẳng theo định nghĩa trên cả nhóm, không chia đoạn, không chia lô nhỏ."""
        ids, mask, labels, w = to_tensors(pad_batch(group, 0), "cpu")
        logits = net.lm_head(net.model(ids, mask).last_hidden_state)
        ce = F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), reduction="none",
                             ignore_index=IGNORE).view(labels[:, 1:].shape)
        return (ce * w.unsqueeze(1)).sum() / window_denominator(group)

    def accumulated(group, bs, chunk):
        """Cách b3_train làm: chia lô nhỏ, tính theo đoạn, chia cho mẫu số của cả lô hiệu dụng, cộng gradient."""
        net.zero_grad()
        denom, total = window_denominator(group), 0.0
        for k in range(0, len(group), bs):
            ids, mask, labels, w = to_tensors(pad_batch(group[k:k + bs], 0), "cpu")
            part = weighted_chunked_loss(net, ids, mask, labels, w, chunk) / denom
            part.backward()
            total += float(part.detach())
        return total, net.lm_head.weight.grad.clone()

    net.zero_grad()
    ref = direct(samples)
    ref.backward()
    ref_grad = net.lm_head.weight.grad.clone()
    uniform = [{**s, "weight": 1.0} for s in samples]
    ids, mask, labels, _w = to_tensors(pad_batch(uniform, 0), "cpu")
    standard = F.cross_entropy(net.lm_head(net.model(ids, mask).last_hidden_state)[:, :-1].flatten(0, 1),
                               labels[:, 1:].flatten(), ignore_index=IGNORE)
    by_len = sorted(samples, key=lambda s: len(s["ids"]))
    checks = []
    for bs, chunk in ((2, 1), (2, 5), (4, 256), (16, 3)):
        val, grad = accumulated(samples, bs, chunk)
        checks.append((f"lô nhỏ {bs}, đoạn {chunk}: mất mát tích luỹ bằng mất mát của cả lô hiệu dụng",
                       abs(val - float(ref)) < 1e-5))
        checks.append((f"lô nhỏ {bs}, đoạn {chunk}: gradient tích luỹ bằng gradient của cả lô hiệu dụng",
                       torch.allclose(grad, ref_grad, atol=1e-6)))
    val, grad = accumulated(by_len, 2, 4)
    checks.append(("xếp mẫu theo độ dài trong lô hiệu dụng không đổi gradient", torch.allclose(grad, ref_grad, atol=1e-6)))
    val_u, _ = accumulated(uniform, 2, 4)
    checks.append(("trọng số đều: bằng mất mát chuẩn (trung bình trên mọi token có nhãn)",
                   abs(val_u - float(standard)) < 1e-5))
    heavy = [{**s, "weight": s["weight"] * 7.0} for s in samples]
    checks.append(("nhân mọi trọng số với một hằng số không đổi mất mát",
                   abs(accumulated(heavy, 2, 4)[0] - float(ref)) < 1e-5))
    checks.append(("trọng số khác nhau cho mất mát khác trọng số đều", abs(float(ref) - val_u) > 1e-4))
    checks.append(("tốc độ học: 0 ở bước đầu, đỉnh sau khởi động, về 0 ở bước cuối",
                   lr_factor(0, 100, 10) == 0 and lr_factor(10, 100, 10) == 1.0 and lr_factor(100, 100, 10) < 1e-9))
    with torch.no_grad():                        # hàng 30 trở lên coi như chưa từng dùng; hàng 5 bị gán trùng hướng
        net.lm_head.weight[30:] = net.lm_head.weight[30:].mean(0) + 0.01 * torch.randn(vocab - 30, hid)
        net.lm_head.weight[5] = 0.4 * net.lm_head.weight[30:].mean(0)
    dead, alive = end_token_row(net, 30, 5), end_token_row(net, 30, 6)
    checks.append(("token kết thúc trùng hướng với các hàng chưa từng dùng thì bị từ chối",
                   end_token_is_untrained(dead) and dead["unused_rows"] == vocab - 30))
    checks.append(("token kết thúc đã được học thì được chấp nhận; mô hình không có hàng thừa thì không so",
                   not end_token_is_untrained(alive) and not end_token_is_untrained(end_token_row(net, vocab, 5))))
    ok = True
    for name, passed in checks:
        ok &= bool(passed)
        print(f"  {'ĐẠT ' if passed else 'HỎNG'}  {name}")
    print(f"[b3] tự kiểm: {sum(bool(p) for _n, p in checks)}/{len(checks)}")
    return 0 if ok else 1


# ============================================================ dòng lệnh
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="qwen1_5b")
    ap.add_argument("--method", help="tên file trong configs/method/")
    ap.add_argument("--ablation", help="tên file trong configs/ablation/ (đè lên phương án qd_rsr)")
    ap.add_argument("--seed", type=int, help="mặc định là seed đầu của project.train_seeds")
    ap.add_argument("--seldir", help="thư mục tập chọn, mặc định data/stage_b/<student>_base")
    ap.add_argument("--plan", action="store_true", help="chỉ in số mẫu và số bước, không cần GPU")
    ap.add_argument("--smoke", action="store_true", help=f"chạy thử {SMOKE_SAMPLES} mẫu, 2 bước, rồi sinh thử")
    ap.add_argument("--pilot", type=int, default=0, metavar="N",
                    help="chạy thử ngắn: đủ số epoch trên khoảng N mẫu, phép thử dừng trên các đề của lần chạy thật")
    ap.add_argument("--resume", action="store_true", help="chạy tiếp từ điểm lưu nếu có")
    ap.add_argument("--force", action="store_true", help="chạy lại dù lần chạy này đã xong")
    ap.add_argument("--check-stop", type=int, help="số đề bài sinh thử sau huấn luyện (0 để bỏ qua)")
    ap.add_argument("--selftest", action="store_true", help="kiểm phần toán của mất mát, không cần dữ liệu")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.method or args.ablation):
        ap.error("cần --method hoặc --ablation (hoặc --selftest)")

    cfg = load_config(method=args.method or "qd_rsr", ablation=args.ablation, student=args.student,
                      overrides=args.override)
    seed = args.seed if args.seed is not None else int(cfg["project"]["train_seeds"][0])
    tag, k = selection_tag(cfg), int(cfg["selection"]["k"])
    sel_dir = resolve_path(cfg, args.seldir) if args.seldir else path_of(cfg, "data_stage_b") / student_tag(cfg)
    rows, meta = load_selection(sel_dir, tag, k)
    if meta["student"] != cfg["student"]["base_model_id"]:
        raise SystemExit(f"Tập chọn này dành cho {meta['student']}, không phải {cfg['student']['base_model_id']}.")

    if args.pilot and args.smoke:
        ap.error("--pilot và --smoke là hai chế độ khác nhau, chỉ dùng một")
    if args.pilot < 0:
        ap.error("--pilot cần một số mẫu dương")
    used = pilot_rows(rows, args.pilot, seed) if args.pilot else rows
    steps, warmup = total_steps(len(used), cfg)
    if args.plan:
        chain_tokens = sum(r["n_tokens"] or 0 for r in used)
        t = cfg["training"]
        print(f"[b3] {tag} | {len(used)} mẫu{' (chạy thử ngắn)' if args.pilot else ''} | lô {t['batch_size']} x "
              f"{t['grad_accum']} | {t['num_epochs']} epoch | "
              f"{steps} bước tối ưu, khởi động {warmup} | {chain_tokens} token chuỗi mỗi epoch (chưa tính đề bài) | "
              f"tổng trọng số {sum(r['weight'] for r in used):.1f} | token kết thúc: {cfg['training']['end_token']}")
        return 0

    out = run_dir(cfg, tag, seed, args.smoke, args.pilot)
    done = out / "run.json"
    if done.exists() and not args.smoke and not args.force and read_json(done).get("status") == "done":
        print(f"[b3] lần chạy này đã xong: {out}. Thêm --force nếu muốn chạy lại.")
        return 0
    if (out / "ckpt").exists() and not args.resume and not args.force and not args.pilot:
        raise SystemExit(f"Có điểm lưu dở ở {out / 'ckpt'}. Thêm --resume để chạy tiếp, hoặc --force để làm lại từ đầu.")
    if args.force or args.smoke or args.pilot:
        shutil.rmtree(out, ignore_errors=True)
    ensure_dir(out)
    n_stop = args.check_stop if args.check_stop is not None else (4 if args.smoke else int(cfg["training"]["stop_check_samples"]))
    r = train(cfg, tag, rows, meta, seed, out, args.resume and not args.pilot, args.smoke, n_stop, args.pilot)
    line = (f"[b3] xong {tag} seed {seed}: {r['hours']:.2f} giờ, {r['tok_per_s']} token/giây, đỉnh bộ nhớ cấp phát "
            f"{r['peak_allocated_gb']} GB, giữ chỗ {r['peak_reserved_gb']} GB")
    if r["stop_check"]:
        s = r["stop_check"]
        line += (f" | sinh thử {s['n']} đề: {s['stopped']} lượt dừng đúng {s['stopped_by']}, {s['with_boxed']} có "
                 f"\\boxed, trung bình {s['mean_new_tokens']:.0f} token (trần {s['max_new_tokens']})")
    print(line)
    print(f"[b3] kết quả ở {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

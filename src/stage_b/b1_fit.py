"""b1_fit: tính bốn tín hiệu nội tại của mô hình học cho từng chuỗi ứng viên.

    python -m src.stage_b.b1_fit --selftest                       # kiểm chứng cài đặt, không cần dữ liệu
    python -m src.stage_b.b1_fit --workdir data/pilot/run2_boxed --limit 200
    python -m src.stage_b.b1_fit --workdir data/stage_a --student qwen1_5b
    python -m src.stage_b.b1_fit --workdir data/stage_a --student qwen1_5b --base   # bản Base thay vì Instruct

Đọc  <workdir>/candidates.jsonl và questions.jsonl
Ghi  <workdir>/fit.<student>.jsonl   một dòng mỗi chuỗi, ghi dần nên chạy bù được

Bốn tín hiệu, tính trong CÙNG MỘT lượt forward vì cả bốn đều đọc từ một bộ logits:

    RSR        tỷ lệ tổng thứ hạng token đã cắt ngưỡng trên tổng độ bất ngờ      CỰC TIỂU
    GRAPE      log xác suất trung bình của chuỗi                                 cực đại
    LocalNat   như GRAPE nhưng chỉ cho mô hình thấy W token ngữ cảnh gần nhất     cực đại
    LARK       điểm Brier trung bình, đo mức mô hình còn chưa chắc chắn          cực đại

CHỈ RSR LÀ CỰC TIỂU. Đây là chỗ dễ cài nhầm nhất trong cả dự án.

Chuỗi được chấm ở đúng định dạng huấn luyện: system prompt huấn luyện, câu hỏi ở vai người dùng, chuỗi
suy luận ở vai trợ lý. Chỉ các token của chuỗi suy luận được tính điểm, phần câu hỏi thì không.

Chạy trên máy có GPU NVIDIA. Mô hình nạp ở 16-bit, không lượng tử, vì lượng tử 4-bit làm nhiễu xác suất
từng token mà thứ hạng token dựa vào, và thứ hạng chính là đại lượng RSR đo.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import JsonlWriter, load_done_keys, now_iso, read_jsonl
from src.common.prompts import SYSTEM_PROMPT_TRAIN

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it


# ============================================================ phần toán thuần
# Tách riêng khỏi phần GPU để kiểm thử được offline: hàm dưới đây chỉ nhận các mảng số đã tính sẵn.

def aggregate(ranks: Sequence[int], surprisals: Sequence[float], sum_sq: Sequence[float],
              target_probs: Sequence[float], rank_clip: int = 100) -> dict:
    """Gộp số liệu từng token thành bốn tín hiệu của cả chuỗi.

    ranks         thứ hạng của token đúng trong phân phối dự đoán, 0 là token được đoán chắc nhất
    surprisals    -log p của token đúng
    sum_sq        tổng bình phương xác suất trên toàn bộ từ vựng, dùng cho điểm Brier
    target_probs  xác suất của token đúng
    """
    n = len(ranks)
    if n == 0 or n != len(surprisals) or n != len(sum_sq) or n != len(target_probs):
        raise ValueError(f"bốn mảng phải cùng độ dài và khác rỗng: {n}, {len(surprisals)}, "
                         f"{len(sum_sq)}, {len(target_probs)}")

    clipped = sum(min(r + 1, rank_clip) for r in ranks)   # thứ hạng đếm từ 1 cho khớp bài gốc
    total_surprisal = sum(surprisals)
    # Chuỗi mà mô hình đoán được hoàn hảo cho tổng độ bất ngờ gần 0; chặn dưới để không chia cho 0.
    rsr = clipped / max(total_surprisal, 1e-6)

    # Điểm Brier của một token: tổng trên từ vựng của (p_v - 1 nếu v đúng)^2 = 1 - 2*p_đúng + tổng p^2.
    brier = [1.0 - 2.0 * p + s for p, s in zip(target_probs, sum_sq)]

    return {
        "rsr": rsr,
        "grape": -total_surprisal / n,                    # log xác suất trung bình
        "lark": sum(brier) / n,
        "mean_rank": sum(ranks) / n,
        "mean_clipped_rank": clipped / n,
        "mean_surprisal": total_surprisal / n,
        "n_tokens": n,
    }


def perplexity(mean_logprob: float) -> float:
    return math.exp(-mean_logprob)


# ============================================================ phần chạy trên GPU
def load_student(model_id: str, load_4bit: bool, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    kw = dict(device_map=device, attn_implementation="sdpa")
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()                       # chỉ đọc chuỗi, không huấn luyện
    model.config.use_cache = False
    return model, tok


def build_inputs(tok, question: str, trajectory: str, max_len: int) -> tuple:
    """Dựng chuỗi token theo ĐÚNG định dạng huấn luyện, và đánh dấu phần nào được tính điểm.

    Chấm ở định dạng huấn luyện là quan trọng: Fit phải đo mức phù hợp trong đúng hoàn cảnh mà mô hình
    học sẽ gặp khi tinh chỉnh, không phải ở một định dạng khác.
    """
    import torch

    messages = [{"role": "system", "content": SYSTEM_PROMPT_TRAIN},
                {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(trajectory, add_special_tokens=False)["input_ids"]

    ids = (p_ids + r_ids)[:max_len]
    n_prompt = min(len(p_ids), len(ids))
    if len(ids) - n_prompt < 2:                      # không còn token nào của chuỗi để chấm
        return None, 0
    return torch.tensor([ids], device="cuda"), n_prompt


def token_stats(model, ids, n_prompt: int, chunk: int = 512) -> dict:
    """Một lượt forward, rồi rút số liệu từng token theo từng đoạn.

    Vì sao chia đoạn: ma trận logit có kích thước (độ dài chuỗi) x (152.000 mục từ vựng). Phép xếp hạng
    phải so logit của token đúng với TOÀN BỘ từ vựng, tạo ra một mảng nữa cùng kích thước. Làm một lần cho
    cả chuỗi tốn gần 8GB; chia đoạn 512 vị trí đưa xuống khoảng 1GB.
    """
    import torch

    with torch.no_grad():
        logits = model(ids).logits[0]                # (độ dài chuỗi, từ vựng)
    ranks, surprisals, sum_sq, tprobs = [], [], [], []
    # Vị trí i dự đoán token i+1, nên chỉ chấm từ n_prompt-1 tới áp chót.
    start, end = max(n_prompt - 1, 0), ids.shape[1] - 1
    for a in range(start, end, chunk):
        b = min(a + chunk, end)
        part = logits[a:b].float()                   # 32-bit cho phần tính xác suất, đoạn nhỏ nên rẻ
        tgt = ids[0, a + 1:b + 1]
        logp = torch.log_softmax(part, dim=-1)
        tgt_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        probs = logp.exp()
        ranks.extend((part > part.gather(-1, tgt.unsqueeze(-1))).sum(-1).tolist())
        surprisals.extend((-tgt_logp).tolist())
        sum_sq.extend((probs * probs).sum(-1).tolist())
        tprobs.extend(tgt_logp.exp().tolist())
        del part, logp, probs, tgt_logp
    del logits
    return {"ranks": ranks, "surprisals": surprisals, "sum_sq": sum_sq, "target_probs": tprobs}


def local_naturalness(model, ids, n_prompt: int, window: int) -> float:
    """Log xác suất trung bình khi mô hình CHỈ thấy `window` token gần nhất.

    Ý tưởng của Local Naturalness: một chuỗi có thể có xác suất toàn cục thấp chỉ vì nó dài, trong khi
    từng bước vẫn tự nhiên. Đo trong ngữ cảnh cục bộ tách được hai điều đó.
    Cài đặt: cắt chuỗi thành các cửa sổ không chồng nhau, mỗi cửa sổ tự làm ngữ cảnh cho chính nó.
    """
    import torch

    total, n = 0.0, 0
    start, end = max(n_prompt - 1, 0), ids.shape[1] - 1
    for a in range(start, end, window):
        b = min(a + window, end)
        piece = ids[:, a:b + 1]
        if piece.shape[1] < 2:
            continue
        with torch.no_grad():
            lg = model(piece).logits[0, :-1].float()
        tgt = piece[0, 1:]
        lp = torch.log_softmax(lg, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total += float(lp.sum())
        n += int(tgt.numel())
        del lg, lp
    return total / max(n, 1)


def score_one(model, tok, question: str, trajectory: str, max_len: int, rank_clip: int,
              window: int, chunk: int) -> dict | None:
    ids, n_prompt = build_inputs(tok, question, trajectory, max_len)
    if ids is None:
        return None
    stats = token_stats(model, ids, n_prompt, chunk)
    out = aggregate(stats["ranks"], stats["surprisals"], stats["sum_sq"], stats["target_probs"], rank_clip)
    out["local_nat"] = local_naturalness(model, ids, n_prompt, window)
    return out


# ============================================================ kiểm chứng cài đặt
SELFTEST_TEXTS = {
    "tự nhiên": "The capital of France is Paris. It is one of the largest cities in Europe.",
    "lặp vô nghĩa": "the the the the the the the the the the the the the the the the the the",
    "xáo trộn": "Paris largest cities the is of France one capital Europe in It the is.",
    "lời giải đúng": "We need 3 times 4. Since 3 times 4 equals 12, the answer is 12.",
    "lời giải sai": "We need 3 times 4. Since 3 times 4 equals 47, the answer is 47.",
}


def selftest(model, tok, rank_clip: int, window: int, chunk: int) -> int:
    """Chạy bốn phép kiểm chứng trên văn bản có tính chất biết trước.

    Đây là câu trả lời cho câu hỏi "dựa vào đâu mà tin cài đặt RSR đúng". Không phép nào cần dữ liệu thật.
    """
    q = "Answer the question."
    r = {name: score_one(model, tok, q, text, 2048, rank_clip, window, chunk)
         for name, text in SELFTEST_TEXTS.items()}

    print(f"\n{'văn bản':<16}{'RSR':>10}{'thứ hạng TB':>14}{'bất ngờ TB':>13}{'GRAPE':>10}"
          f"{'LocalNat':>11}{'LARK':>9}")
    for name, v in r.items():
        print(f"{name:<16}{v['rsr']:>10.2f}{v['mean_rank']:>14.1f}{v['mean_surprisal']:>13.3f}"
              f"{v['grape']:>10.3f}{v['local_nat']:>11.3f}{v['lark']:>9.3f}")

    checks = [
        ("văn bản xáo trộn có thứ hạng token cao hơn văn bản tự nhiên",
         r["xáo trộn"]["mean_rank"] > r["tự nhiên"]["mean_rank"]),
        ("văn bản xáo trộn bất ngờ hơn văn bản tự nhiên",
         r["xáo trộn"]["mean_surprisal"] > r["tự nhiên"]["mean_surprisal"]),
        ("lời giải sai bất ngờ hơn lời giải đúng",
         r["lời giải sai"]["mean_surprisal"] > r["lời giải đúng"]["mean_surprisal"]),
        (f"thứ hạng đã cắt không vượt ngưỡng {rank_clip}",
         all(v["mean_clipped_rank"] <= rank_clip + 1e-9 for v in r.values())),
        ("chuỗi lặp vô nghĩa dễ đoán nên thứ hạng thấp",
         r["lặp vô nghĩa"]["mean_rank"] < r["xáo trộn"]["mean_rank"]),
        ("chấm lại cùng một chuỗi cho kết quả giống hệt",
         abs(score_one(model, tok, q, SELFTEST_TEXTS["tự nhiên"], 2048, rank_clip, window, chunk)["rsr"]
             - r["tự nhiên"]["rsr"]) < 1e-6),
    ]
    print()
    bad = 0
    for name, ok in checks:
        bad += not ok
        print(f"  {'ĐẠT ' if ok else 'HỎNG'}  {name}")
    print(f"\n{len(checks) - bad}/{len(checks)} phép kiểm chứng đạt")
    if bad:
        print("Có phép HỎNG: cài đặt tín hiệu sai, KHÔNG chạy trên dữ liệu thật trước khi sửa.")
    return bad


# ============================================================ dòng lệnh
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--student", default="qwen1_5b", help="tên file trong configs/student/")
    ap.add_argument("--base", action="store_true", help="dùng bản Base thay vì Instruct")
    ap.add_argument("--limit", type=int, help="chỉ chấm N chuỗi đầu, dùng để thử")
    ap.add_argument("--selftest", action="store_true", help="chỉ chạy kiểm chứng, không cần dữ liệu")
    ap.add_argument("--load-4bit", action="store_true",
                    help="nạp mô hình ở 4-bit. Làm nhiễu thứ hạng token nên chỉ dùng khi 16-bit không vừa")
    ap.add_argument("--chunk", type=int, default=512, help="số vị trí xếp hạng mỗi lần")
    ap.add_argument("--out", help="tên file đầu ra, mặc định fit.<student>.jsonl")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(student=args.student, overrides=args.override)
    st = cfg["student"]
    model_id = st["base_model_id"] if args.base else st["model_id"]
    rank_clip = cfg["selection"]["rsr_rank_cap"]
    window = cfg["signals"]["local_window"]
    max_len = cfg["training"]["max_seq_len"]

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("Không thấy GPU. Lệnh này chạy trên máy Windows có card NVIDIA.")
    print(f"[b1] mô hình học {model_id}, {'4-bit' if args.load_4bit else '16-bit'}, "
          f"ngưỡng cắt thứ hạng {rank_clip}, cửa sổ cục bộ {window}")
    model, tok = load_student(model_id, args.load_4bit)
    print(f"[b1] nạp xong, chiếm {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    if args.selftest:
        return selftest(model, tok, rank_clip, window, args.chunk)

    wd: Path = resolve_path(cfg, args.workdir)
    files = cfg["stage_a_files"]
    cands = read_jsonl(wd / files["candidates"])
    questions = {q["qid"]: q for q in read_jsonl(wd / files["questions"])}
    if not cands:
        raise SystemExit(f"Không có ứng viên trong {wd}. Chạy a3_filter trước.")
    if args.limit:
        cands = cands[: args.limit]

    tag = args.student + ("_base" if args.base else "")
    path = wd / (args.out or f"fit.{tag}.jsonl")
    done = load_done_keys(path, lambda r: r["tid"])
    todo = [c for c in cands if c["tid"] not in done]
    print(f"[b1] {len(cands)} chuỗi, đã có {len(done)}, cần chấm {len(todo)}")

    failed = 0
    with JsonlWriter(path) as w:
        for c in tqdm(todo, desc="chấm tín hiệu"):
            q = questions.get(c["qid"])
            if q is None:
                failed += 1
                continue
            try:
                v = score_one(model, tok, q["question"], c["text"], max_len, rank_clip, window, args.chunk)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                failed += 1
                continue
            if v is None:
                failed += 1
                continue
            w.append({"tid": c["tid"], "qid": c["qid"], "model": model_id, "ts": now_iso(), **v})

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"[b1] xong, đỉnh bộ nhớ {peak:.2f} GB" + (f", {failed} chuỗi không chấm được" if failed else ""))

    rows = read_jsonl(path)
    if rows:
        rsr = sorted(r["rsr"] for r in rows)
        print(f"[b1] RSR trên {len(rows)} chuỗi: nhỏ nhất {rsr[0]:.2f}, trung vị {rsr[len(rsr) // 2]:.2f}, "
              f"lớn nhất {rsr[-1]:.2f}")
        print("[b1] Nhắc lại: RSR càng THẤP càng phù hợp. Ba tín hiệu còn lại càng CAO càng tốt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""b1_fit: tính bốn tín hiệu nội tại của mô hình học cho từng chuỗi ứng viên.

    python -m src.stage_b.b1_fit --selftest                       # kiểm chứng cài đặt, không cần dữ liệu
    python -m src.stage_b.b1_fit --workdir data/pilot/run2_boxed --limit 200
    python -m src.stage_b.b1_fit --workdir data/stage_a --student qwen1_5b
    python -m src.stage_b.b1_fit --workdir data/stage_a --student qwen1_5b --base   # bản Base thay vì Instruct
    python -m src.stage_b.b1_fit --workdir data/stage_a --student qwen7b --base --load-4bit   # 7B: tự bỏ LocalNat

Đọc  <workdir>/candidates.jsonl và questions.jsonl
Ghi  <workdir>/fit.<student>.jsonl   một dòng mỗi chuỗi, ghi dần nên chạy bù được

Bốn tín hiệu (đối chiếu bài gốc ngày 29/09/2026, xem src/stage_b/signals.py):

    RSR        tổng thứ hạng token đã cắt ngưỡng chia tổng độ bất ngờ (Yang 2026)        CỰC TIỂU
    GRAPE      log xác suất trung bình từng token của cả chuỗi (Zhang 2025; cách hiểu    cực đại
               của Just 2025 phương trình 1 và LARK phụ lục C.2)
    LocalNat   trung bình theo CÂU của log xác suất từng câu, khi mô hình chỉ thấy đề     cực đại
               bài và tối đa k = 4 câu liền trước (Just 2025 phương trình 2)
    LARK       ĝ = ℓ/Σℓ · (2ρ̂ − Σρ̂ℓ/Σℓ), ρ̂ = Brier/ℓ, tổng trên ứng viên cùng câu hỏi     cực đại
               (Yu 2026 phương trình 7). b1 chỉ ghi Brier và ℓ; ĝ tính theo nhóm ở
               signals.add_lark, vì nó cần đủ các ứng viên của câu hỏi.

RSR, GRAPE và phần của LARK đọc từ CÙNG MỘT lượt forward. LocalNat cần các lượt riêng, mỗi câu một mục,
chạy theo lô. CHỈ RSR LÀ CỰC TIỂU. Đây là chỗ dễ cài nhầm nhất trong cả dự án.

Bỏ LocalNat: đặt signals.skip_local_nat: true (configs/student/qwen7b.yaml đã đặt, chốt 30/09) hoặc thêm cờ
--skip-localnat. Khi đó dòng kết quả KHÔNG có trường local_nat (không phải null), nên compare_fit và
make_figures tự bỏ LocalNat khỏi ma trận. Một file fit không được trộn dòng có và không có LocalNat.

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
from src.stage_b.signals import SIGNALS_VERSION, local_items, mean_over_steps, token_steps

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
        "brier": sum(brier) / n,                          # Brier_k của LARK; ĝ tính theo nhóm ở signals.add_lark
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

    Trả về (ids trên GPU, số token phần đề, ids phần đề, ids phần chuỗi đã cắt, các bước theo câu).
    """
    import torch

    messages = [{"role": "system", "content": SYSTEM_PROMPT_TRAIN},
                {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    enc = tok(trajectory, add_special_tokens=False, return_offsets_mapping=True)
    r_ids, offsets = enc["input_ids"], enc["offset_mapping"]

    keep = max(0, min(len(r_ids), max_len - len(p_ids)))
    r_ids, offsets = r_ids[:keep], offsets[:keep]
    if len(r_ids) < 2:                               # không còn token nào của chuỗi để chấm
        return None, 0, None, None, None
    steps = token_steps(offsets, trajectory)
    return torch.tensor([p_ids + r_ids], device="cuda"), len(p_ids), p_ids, r_ids, steps


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


def local_naturalness(model, p_ids: Sequence[int], r_ids: Sequence[int], steps, k: int,
                      batch: int = 32, head_chunk: int = 512) -> float:
    """Local Naturalness đúng phương trình 2 của Just 2025.

    Mỗi câu thành một mục: [đề bài] + [tối đa k câu liền trước] + [câu đang chấm]; chỉ token của câu đang
    chấm được tính. Điểm câu là log xác suất trung bình của nó; điểm chuỗi là trung bình theo câu.

    Cách chạy cho nhanh (29/09, bản đầu mất 1,6 giây mỗi chuỗi):
      - lô 32 mục, nên phần lớn chuỗi chỉ cần một lượt forward cho mọi câu;
      - dựng tensor trên CPU rồi chuyển sang GPU một lần, không tạo tensor GPU cho từng mục;
      - chỉ tính lớp đầu ra trên các vị trí cần chấm, gom theo đoạn head_chunk vị trí, cộng dồn vào từng
        mục bằng index_add_, và chỉ đồng bộ về CPU một lần ở cuối.
    Đệm bên phải, nên vị trí của token thật vẫn là 0, 1, 2... như khi chạy riêng từng mục. Kết quả phải khớp
    local_naturalness_naive (kiểm chứng trong --selftest).
    """
    import torch

    items = local_items(steps, k)
    pad = model.config.pad_token_id
    if pad is None:
        pad = model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else 0
    body, head = model.model, model.lm_head
    dev = head.weight.device
    sums = torch.zeros(len(items), device=dev, dtype=torch.float32)
    counts = torch.zeros(len(items), device=dev, dtype=torch.float32)
    n_prompt = len(p_ids)
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        seqs = [list(p_ids) + list(r_ids[c:e]) for c, s, e in chunk]
        width = max(len(x) for x in seqs)
        ids_cpu = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask_cpu = torch.zeros((len(seqs), width), dtype=torch.long)
        rows, cols, owner = [], [], []
        for j, ((c, s, e), x) in enumerate(zip(chunk, seqs)):
            ids_cpu[j, :len(x)] = torch.tensor(x, dtype=torch.long)
            mask_cpu[j, :len(x)] = 1
            first = n_prompt + (s - c)               # vị trí token đầu của câu trong mục
            rows.extend([j] * (len(x) - first))
            cols.extend(range(first - 1, len(x) - 1))  # vị trí t dự đoán token t+1
            owner.extend([i + j] * (len(x) - first))
        ids, mask = ids_cpu.to(dev), mask_cpu.to(dev)
        rows_t = torch.tensor(rows, device=dev)
        cols_t = torch.tensor(cols, device=dev)
        owner_t = torch.tensor(owner, device=dev)
        with torch.no_grad():
            hidden = body(input_ids=ids, attention_mask=mask).last_hidden_state
            h = hidden[rows_t, cols_t]               # (số token cần chấm, chiều ẩn)
            tgt = ids[rows_t, cols_t + 1]
            del hidden
            for a in range(0, h.shape[0], head_chunk):
                lg = head(h[a:a + head_chunk]).float()
                lp = lg.gather(-1, tgt[a:a + head_chunk].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, dim=-1)
                sums.index_add_(0, owner_t[a:a + head_chunk], lp)
                del lg, lp
            counts.index_add_(0, owner_t, torch.ones_like(owner_t, dtype=torch.float32))
        del h, tgt, ids, mask
    return mean_over_steps((sums / counts).tolist())


def local_naturalness_naive(model, p_ids: Sequence[int], r_ids: Sequence[int], steps, k: int) -> float:
    """Cùng đại lượng, chạy riêng từng câu bằng lượt forward đầy đủ. Chỉ dùng trong --selftest để đối chiếu."""
    import torch

    means = []
    for c, s, e in local_items(steps, k):
        x = torch.tensor([list(p_ids) + list(r_ids[c:e])], device="cuda")
        first = len(p_ids) + (s - c)
        with torch.no_grad():
            lg = model(x).logits[0, first - 1:x.shape[1] - 1].float()
        lp = torch.log_softmax(lg, dim=-1).gather(-1, x[0, first:].unsqueeze(-1))
        means.append(float(lp.mean()))
    return mean_over_steps(means)


def score_one(model, tok, question: str, trajectory: str, max_len: int, rank_clip: int,
              local_k: int, chunk: int, with_local: bool = True) -> dict | None:
    ids, n_prompt, p_ids, r_ids, steps = build_inputs(tok, question, trajectory, max_len)
    if ids is None:
        return None
    stats = token_stats(model, ids, n_prompt, chunk)
    out = aggregate(stats["ranks"], stats["surprisals"], stats["sum_sq"], stats["target_probs"], rank_clip)
    if with_local:
        out["local_nat"] = local_naturalness(model, p_ids, r_ids, steps, local_k)
    out["n_steps"] = len(steps)
    out["signals_version"] = SIGNALS_VERSION
    return out


# ============================================================ kiểm chứng cài đặt
SELFTEST_TEXTS = {
    "tự nhiên": "The capital of France is Paris. It is one of the largest cities in Europe.",
    "lặp vô nghĩa": "the the the the the the the the the the the the the the the the the the",
    "xáo trộn": "Paris largest cities the is of France one capital Europe in It the is.",
    "lời giải đúng": "We need 3 times 4. Since 3 times 4 equals 12, the answer is 12.",
    "lời giải sai": "We need 3 times 4. Since 3 times 4 equals 47, the answer is 47.",
    "một câu": "The answer is 12 because 3 times 4 equals 12",
    "nhiều câu": ("Let x be the number of apples. Tom has 3 more apples than Ann. Ann has 5 apples. "
                  "So Tom has 5 + 3 = 8 apples.\n\nTogether they have 5 + 8 = 13 apples. "
                  "We check: 13 - 5 = 8, which matches. Therefore the answer is \\boxed{13}."),
}


def selftest(model, tok, rank_clip: int, local_k: int, chunk: int) -> int:
    """Chạy bốn phép kiểm chứng trên văn bản có tính chất biết trước.

    Đây là câu trả lời cho câu hỏi "dựa vào đâu mà tin cài đặt RSR đúng". Không phép nào cần dữ liệu thật.
    """
    q = "Answer the question."
    r = {name: score_one(model, tok, q, text, 3072, rank_clip, local_k, chunk)
         for name, text in SELFTEST_TEXTS.items()}

    print(f"\n{'văn bản':<16}{'RSR':>10}{'thứ hạng TB':>14}{'bất ngờ TB':>13}{'GRAPE':>10}"
          f"{'LocalNat':>11}{'Brier':>9}{'số câu':>8}")
    for name, v in r.items():
        print(f"{name:<16}{v['rsr']:>10.2f}{v['mean_rank']:>14.1f}{v['mean_surprisal']:>13.3f}"
              f"{v['grape']:>10.3f}{v['local_nat']:>11.3f}{v['brier']:>9.3f}{v['n_steps']:>8}")

    # Đối chiếu LocalNat chạy theo lô với cách chạy từng câu, trên chuỗi nhiều câu.
    long_text = SELFTEST_TEXTS["nhiều câu"]
    _ids, _n, p_ids, r_ids, steps = build_inputs(tok, q, long_text, 3072)
    batched = local_naturalness(model, p_ids, r_ids, steps, local_k)
    naive = local_naturalness_naive(model, p_ids, r_ids, steps, local_k)
    # Một câu duy nhất: ngữ cảnh chỉ có đề bài, nên LocalNat phải bằng GRAPE.
    one = score_one(model, tok, q, SELFTEST_TEXTS["một câu"], 3072, rank_clip, local_k, chunk)
    print(f"\nLocalNat theo lô {batched:.4f}, từng câu {naive:.4f}, {len(steps)} câu")
    print(f"Chuỗi một câu: LocalNat {one['local_nat']:.4f}, GRAPE {one['grape']:.4f}")

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
         abs(score_one(model, tok, q, SELFTEST_TEXTS["tự nhiên"], 3072, rank_clip, local_k, chunk)["rsr"]
             - r["tự nhiên"]["rsr"]) < 1e-6),
        ("LocalNat chạy theo lô khớp cách chạy từng câu (lệch dưới 0,02)",
         abs(batched - naive) < 0.02),
        ("chuỗi một câu: LocalNat bằng GRAPE (lệch dưới 0,02)",
         one["n_steps"] == 1 and abs(one["local_nat"] - one["grape"]) < 0.02),
        ("chuỗi nhiều câu được cắt thành nhiều bước", len(steps) >= 6),
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
def check_local_consistency(first_row: Mapping | None, with_local: bool, name: str) -> None:
    """Chặn việc chạy bù làm file fit lẫn dòng có và không có LocalNat."""
    if first_row is None:
        return
    has_local = "local_nat" in first_row
    if has_local != with_local:
        was = "có" if has_local else "không có"
        now = "tính" if with_local else "bỏ"
        raise SystemExit(f"{name} {was} LocalNat, nhưng lần chạy này {now} LocalNat. Chạy bù sẽ làm file lẫn "
                         f"hai loại dòng. Giữ đúng cài đặt cũ, hoặc ghi ra file khác bằng --out.")


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
    ap.add_argument("--skip-localnat", action="store_true",
                    help="không tính LocalNat (mặc định lấy signals.skip_local_nat trong cấu hình)")
    ap.add_argument("--out", help="tên file đầu ra, mặc định fit.<student>.jsonl")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(student=args.student, overrides=args.override)
    st = cfg["student"]
    model_id = st["base_model_id"] if args.base else st["model_id"]
    rank_clip = cfg["selection"]["rsr_rank_cap"]
    local_k = cfg["signals"]["local_steps"]
    max_len = cfg["training"]["max_seq_len"]
    with_local = not (args.skip_localnat or cfg["signals"].get("skip_local_nat", False))

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("Không thấy GPU. Lệnh này chạy trên máy Windows có card NVIDIA.")
    print(f"[b1] mô hình học {model_id}, {'4-bit' if args.load_4bit else '16-bit'}, "
          f"ngưỡng cắt thứ hạng {rank_clip}, "
          + (f"LocalNat {local_k} câu liền trước" if with_local else "BỎ LocalNat")
          + f", độ dài tối đa {max_len}")
    model, tok = load_student(model_id, args.load_4bit)
    print(f"[b1] nạp xong, chiếm {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    if args.selftest:
        return selftest(model, tok, rank_clip, local_k, args.chunk)

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
    if path.exists():
        first = next(iter(read_jsonl(path)), None)
        if first is not None and int(first.get("signals_version", 1)) < SIGNALS_VERSION:
            raise SystemExit(
                f"{path.name} được tạo bằng định nghĩa tín hiệu cũ (LARK là Brier, LocalNat là khối 256 token).\n"
                f"Chạy bù lên file này sẽ bỏ qua mọi chuỗi. Đổi tên file cũ trước, ví dụ:\n"
                f"    mv {path} {path.with_suffix('.v1.jsonl')}")
        check_local_consistency(first, with_local, path.name)
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
                v = score_one(model, tok, q["question"], c["text"], max_len, rank_clip, local_k, args.chunk,
                              with_local)
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
        print("[b1] Cột brier chưa phải LARK: ĝ tính theo câu hỏi bằng signals.add_lark (compare_fit tự làm).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

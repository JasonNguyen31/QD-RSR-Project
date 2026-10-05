"""b1_localnat: chấm Local Naturalness đúng bài gốc, trên các bước do GLM-4.5-Air cắt, cho nhiều k trong một lượt.

    python -m src.stage_b.b1_localnat --plan                                  # không cần GPU: đếm chuỗi, nguồn bước, số mục
    python -m src.stage_b.b1_localnat --student qwen1_5b --base --selftest    # cần GPU và mô hình, không cần dữ liệu
    python -m src.stage_b.b1_localnat --student qwen1_5b --base --limit 50    # chấm thử 50 chuỗi đầu (vẫn ghi vào file chính)
    python -m src.stage_b.b1_localnat --student qwen1_5b --base               # chấm k = 1 và k = 4, chạy lại thì làm tiếp

Đọc  <workdir>/candidates.jsonl, questions.jsonl, quality.jsonl, steps.glm.jsonl
Ghi  <workdir>/localnat.<student>_base.glm.jsonl   một dòng mỗi chuỗi: local_nat_k1, local_nat_k4, nguồn bước,
                                                   log xác suất trung bình và số token của từng bước

Vì sao là một bước riêng mà không nằm trong b1_fit:
  - fit.<student>_base.jsonl là file mà mọi tập chọn đã dựa vào; chấm LocalNat trên bước mới không cần và không
    được đụng tới nó. RSR, GRAPE, LARK không đổi, chỉ LocalNat cần các lượt forward riêng.
  - File này và src/stage_b/steps.py đều là file MỚI, không sửa module nào mà b3_train hay c1_evaluate nạp, nên
    cập nhật mã trên máy GPU giữa một chuỗi lệnh đang chạy không làm đổi hành vi của chuỗi lệnh đó.

Định nghĩa (Just và cộng sự, arXiv 2510.03988, phương trình 2; bản mới nhất gọi là LALP): chuỗi được chia thành
các bước s_1..s_p, mỗi bước là một NHÓM CÂU do GLM-4.5-Air cắt. Điểm của bước i là log xác suất trung bình từng
token của s_i khi mô hình chỉ thấy đề bài và k bước liền trước; điểm của chuỗi là trung bình cộng THEO BƯỚC. Mã
hoá đề bài và chuỗi giống hệt b1_fit.build_inputs, tức đúng định dạng huấn luyện.

Các bước lấy từ src/stage_b/steps.final_steps (lùi ranh giới về đầu dòng, gộp bước quá ngắn, cứu hoặc cắt dự phòng
cho chuỗi cắt hỏng); src/tools/audit_steps in ra hàm đó làm gì trên dữ liệu thật mà không cần GPU.

Nhiều k trong một lượt: mục chấm của bước i với k bước ngữ cảnh là (bước i−k .. i−1 làm ngữ cảnh, bước i được
chấm). Với i ≤ k thì mục này giống hệt nhau ở mọi k lớn hơn, nên các mục trùng chỉ chạy một lần (unique_items).
Với trung vị 6 bước, chấm k = 1 và k = 4 cần 10 mục thay vì 12.

Cỡ lô theo ngân sách bộ nhớ: mục chấm bây giờ có thể dài gần bằng cả chuỗi (k = 4 với bước là nhóm câu), nên lô
32 mục cố định như b1_fit có thể tràn 12 GB. plan_batches xếp mục theo độ dài và giới hạn (số mục) x (độ dài)²;
hết bộ nhớ thì chia đôi lô và chạy lại, không bỏ chuỗi.
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import JsonlWriter, now_iso, read_jsonl
from src.common.prompts import SYSTEM_PROMPT_TRAIN
from src.stage_b import steps as st
from src.stage_b.signals import SIGNALS_VERSION, local_items, mean_over_steps, token_steps
from src.tools.segment_steps import OUT_NAME as STEPS_FILE, usable_chains

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it

DEFAULT_KS = (1, 4)
MAX_BATCH = 32
ATTN_GB = 1.5                     # ngân sách cho ma trận attention của một lô, tính dư: 2 byte x số đầu x lô x dài²


# ============================================================ phần thuần (kiểm thử được không cần torch)
def column(k: int) -> str:
    return f"local_nat_k{int(k)}"


def parse_ks(values: Sequence[int]) -> list[int]:
    ks = sorted({int(v) for v in values})
    if not ks or ks[0] < 1:
        raise SystemExit("--local-k cần các số nguyên từ 1 trở lên, ví dụ: --local-k 1 4")
    return ks


def unique_items(steps: Sequence[tuple[int, int]], ks: Sequence[int]) -> tuple[list[tuple[int, int, int]], dict]:
    """Các mục chấm KHÁC NHAU cần chạy cho mọi k, và với từng k: mục nào ứng với bước nào.

    Trả về (danh sách mục (ngữ cảnh bắt đầu, bước bắt đầu, bước kết thúc), {k: [chỉ số mục của bước 0, 1, ...]}).
    """
    items: list[tuple[int, int, int]] = []
    seen: dict[tuple[int, int, int], int] = {}
    index: dict[int, list[int]] = {}
    for k in ks:
        index[k] = []
        for it in local_items(steps, k):
            if it not in seen:
                seen[it] = len(items)
                items.append(it)
            index[k].append(seen[it])
    return items, index


def plan_batches(lengths: Sequence[int], max_batch: int, cells: float) -> list[list[int]]:
    """Chia các mục thành lô: xếp theo độ dài giảm dần, mỗi lô không quá max_batch mục và (số mục) x (độ dài của
    mục dài nhất)² không vượt cells. Lô nào cũng có ít nhất một mục."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches, i = [], 0
    while i < len(order):
        width = lengths[order[i]]
        n = max(1, min(max_batch, int(cells // max(1, width * width))))
        batches.append(order[i:i + n])
        i += n
    return batches


def attention_cells(n_heads: int, attn_gb: float = ATTN_GB) -> float:
    return attn_gb * 1024 ** 3 / (2.0 * max(1, int(n_heads)))


def combine(means: Sequence[float], index: Mapping[int, Sequence[int]]) -> dict:
    """Điểm LocalNat của chuỗi cho từng k: trung bình theo bước của điểm từng mục."""
    return {k: mean_over_steps([means[j] for j in idx]) for k, idx in index.items()}


def prepare(chains: Sequence[Mapping], rows: Mapping[str, Mapping]) -> tuple[dict, list]:
    """Các bước cuối cùng của mọi chuỗi đã cắt, và danh sách chuỗi chưa có dòng nào trong file cắt bước."""
    final, missing = {}, []
    for c in chains:
        row = rows.get(c["tid"])
        if row is None:
            missing.append(c["tid"])
        else:
            final[c["tid"]] = st.final_steps(c["text"], row)
    return final, missing


def check_resume(first: Mapping | None, ks: Sequence[int], model_id: str, name: str) -> None:
    """Chặn việc chạy bù làm một file lẫn dòng của mô hình khác, bộ k khác hoặc quy tắc dựng bước khác."""
    if first is None:
        return
    have = sorted(int(key[len("local_nat_k"):]) for key in first if key.startswith("local_nat_k"))
    problems = []
    if have != list(ks):
        problems.append(f"k là {have}, lần này {list(ks)}")
    if first.get("model") != model_id:
        problems.append(f"mô hình là {first.get('model')}, lần này {model_id}")
    if first.get("steps_version") != st.STEPS_VERSION:
        problems.append(f"quy tắc dựng bước phiên bản {first.get('steps_version')}, mã hiện là {st.STEPS_VERSION}")
    if problems:
        raise SystemExit(f"{name} được chấm với cài đặt khác: " + "; ".join(problems)
                         + ". Chạy bù sẽ làm file lẫn hai loại dòng. Đổi tên file cũ, hoặc ghi ra file khác bằng --out.")


# ============================================================ phần chạy mô hình
def encode(tok, question: str, text: str, ends: Sequence[int], max_len: int):
    """Mã hoá đúng như b1_fit.build_inputs, nhưng chia bước theo ends. Trả về (ids đề, ids chuỗi, các bước theo
    token) hoặc None khi không còn token nào của chuỗi để chấm."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT_TRAIN}, {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_ids = list(tok(prompt, add_special_tokens=False)["input_ids"])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    r_ids, offsets = list(enc["input_ids"]), list(enc["offset_mapping"])
    keep = max(0, min(len(r_ids), max_len - len(p_ids)))
    r_ids, offsets = r_ids[:keep], offsets[:keep]
    if len(r_ids) < 2:
        return None
    return p_ids, r_ids, token_steps(offsets, text, ends)


def _score_batch(model, p_ids: Sequence[int], r_ids: Sequence[int], batch: Sequence[tuple[int, int, int]],
                 head_chunk: int) -> list[float]:
    """Log xác suất trung bình của bước được chấm trong từng mục của một lô. Đệm bên phải, như b1_fit."""
    import torch

    pad = model.config.pad_token_id
    if pad is None:
        pad = model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else 0
    body, head = model.model, model.lm_head
    dev = head.weight.device
    n_prompt = len(p_ids)
    seqs = [list(p_ids) + list(r_ids[c:e]) for c, _s, e in batch]
    width = max(len(x) for x in seqs)
    ids_cpu = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask_cpu = torch.zeros((len(seqs), width), dtype=torch.long)
    rows, cols, owner = [], [], []
    for j, ((c, s, _e), x) in enumerate(zip(batch, seqs)):
        ids_cpu[j, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask_cpu[j, :len(x)] = 1
        first = n_prompt + (s - c)                       # vị trí token đầu của bước được chấm trong mục
        rows.extend([j] * (len(x) - first))
        cols.extend(range(first - 1, len(x) - 1))        # vị trí t dự đoán token t+1
        owner.extend([j] * (len(x) - first))
    ids, mask = ids_cpu.to(dev), mask_cpu.to(dev)
    rows_t, cols_t, owner_t = (torch.tensor(v, device=dev) for v in (rows, cols, owner))
    sums = torch.zeros(len(batch), device=dev, dtype=torch.float32)
    counts = torch.zeros(len(batch), device=dev, dtype=torch.float32)
    with torch.no_grad():
        hidden = body(input_ids=ids, attention_mask=mask).last_hidden_state
        h = hidden[rows_t, cols_t]
        tgt = ids[rows_t, cols_t + 1]
        del hidden
        for a in range(0, h.shape[0], head_chunk):
            lg = head(h[a:a + head_chunk]).float()
            lp = lg.gather(-1, tgt[a:a + head_chunk].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, dim=-1)
            sums.index_add_(0, owner_t[a:a + head_chunk], lp)
            del lg, lp
        counts.index_add_(0, owner_t, torch.ones_like(owner_t, dtype=torch.float32))
    return (sums / counts).tolist()


def step_logprob_means(model, p_ids: Sequence[int], r_ids: Sequence[int], items: Sequence[tuple[int, int, int]],
                       max_batch: int = MAX_BATCH, head_chunk: int = 512, attn_gb: float = ATTN_GB) -> tuple[list, int]:
    """Điểm của từng mục chấm, theo đúng thứ tự của items. Trả về (điểm, số lần phải chia đôi lô vì hết bộ nhớ)."""
    import torch

    lengths = [len(p_ids) + (e - c) for c, _s, e in items]
    heads = getattr(model.config, "num_attention_heads", 16)
    out: list[float | None] = [None] * len(items)
    splits = 0
    queue = plan_batches(lengths, max_batch, attention_cells(heads, attn_gb))
    while queue:
        batch = queue.pop(0)
        try:
            vals = _score_batch(model, p_ids, r_ids, [items[i] for i in batch], head_chunk)
        except torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            splits += 1
            half = len(batch) // 2
            queue[:0] = [batch[:half], batch[half:]]
            continue
        for i, v in zip(batch, vals):
            out[i] = v
    return out, splits


def score_chain(model, tok, question: str, text: str, ends: Sequence[int], ks: Sequence[int], max_len: int,
                max_batch: int = MAX_BATCH, attn_gb: float = ATTN_GB) -> dict | None:
    enc = encode(tok, question, text, ends, max_len)
    if enc is None:
        return None
    p_ids, r_ids, steps = enc
    items, index = unique_items(steps, ks)
    means, splits = step_logprob_means(model, p_ids, r_ids, items, max_batch, attn_gb=attn_gb)
    row = {column(k): v for k, v in combine(means, index).items()}
    row.update({f"step_logp_k{k}": [round(means[j], 5) for j in index[k]] for k in ks})
    row.update(n_steps=len(steps), n_steps_text=len(ends), n_tokens=len(r_ids), step_tokens=[e - s for s, e in steps],
               n_items=len(items), oom_splits=splits)
    return row


# ============================================================ kiểm chứng trên GPU, không cần dữ liệu
def selftest(model, tok, max_len: int, chunk: int) -> int:
    """So với cài đặt đã kiểm của b1_fit trên cùng một chuỗi. Sai số cho phép 0,02 là mức nhiễu bfloat16 mà
    b1_fit --selftest vẫn dùng."""
    from src.stage_b import b1_fit as b1
    from src.stage_b.signals import sentence_ends

    q, text = "Answer the question.", b1.SELFTEST_TEXTS["nhiều câu"]
    sent = sentence_ends(text)
    p_ids, r_ids, steps = encode(tok, q, text, sent, max_len)
    ids, n_prompt, p_ref, r_ref, steps_ref = b1.build_inputs(tok, q, text, max_len)
    stats = b1.token_stats(model, ids, n_prompt, chunk)
    grape = b1.aggregate(stats["ranks"], stats["surprisals"], stats["sum_sq"], stats["target_probs"])["grape"]

    ks = [1, 4]
    items, index = unique_items(steps, ks)
    means, _ = step_logprob_means(model, p_ids, r_ids, items)
    multi = combine(means, index)
    small, _ = step_logprob_means(model, p_ids, r_ids, items, max_batch=2)
    old = {k: b1.local_naturalness(model, p_ids, r_ids, steps, k) for k in ks}
    naive = {k: b1.local_naturalness_naive(model, p_ids, r_ids, steps, k) for k in ks}

    big = len(steps) + 5                                   # k lớn hơn số bước: mỗi bước thấy toàn bộ phần trước
    it_all, ix_all = unique_items(steps, [big])
    m_all, _ = step_logprob_means(model, p_ids, r_ids, it_all)
    n_tok = [e - s for s, e in steps]
    by_token = sum(m_all[j] * n for j, n in zip(ix_all[big], n_tok)) / sum(n_tok)

    probe = "So x = 10. Therefore the answer is 12.\nNext line starts here."
    probe_ends = st.before_spaces(probe, [probe.index("Therefore"), probe.index("Next"), len(probe)])[0]
    _p, probe_ids, probe_steps = encode(tok, q, probe, probe_ends, max_len)
    heads = [tok.decode(probe_ids[a:a + 2]).strip() for a, _b in probe_steps]

    one = score_chain(model, tok, q, text, [len(text)], ks, max_len)
    cut = text.index("\n\n") + 2
    two = score_chain(model, tok, q, text, [cut, len(text)], ks, max_len)

    print(f"\n{len(steps)} bước theo dấu câu, {len(items)} mục khác nhau cho k = 1 và 4 (thay vì {2 * len(steps)})")
    for k in ks:
        print(f"k = {k}: lượt gộp {multi[k]:.4f} | b1_fit theo lô {old[k]:.4f} | từng mục {naive[k]:.4f}")
    print(f"GRAPE {grape:.4f} | ngữ cảnh đầy đủ, gộp theo token {by_token:.4f} | một bước {one[column(1)]:.4f} | "
          f"hai bước k = 1: {two[column(1)]:.4f}")

    tol = 0.02
    checks = [
        ("mã hoá giống hệt b1_fit.build_inputs (đề, chuỗi, các bước)",
         p_ids == list(p_ref) and r_ids == list(r_ref) and steps == steps_ref),
        ("các mục trùng giữa k = 1 và k = 4 chỉ chạy một lần", len(items) < 2 * len(steps)),
        ("lượt gộp nhiều k khớp b1_fit.local_naturalness ở từng k", all(abs(multi[k] - old[k]) < tol for k in ks)),
        ("lượt gộp nhiều k khớp cách chạy riêng từng mục", all(abs(multi[k] - naive[k]) < tol for k in ks)),
        ("lô nhỏ và lô lớn cho cùng điểm", max(abs(a - b) for a, b in zip(means, small)) < tol),
        ("k = 1 và k = 4 cho điểm khác nhau trên chuỗi nhiều bước", abs(multi[1] - multi[4]) > 1e-4),
        ("ngữ cảnh đầy đủ, gộp lại theo token, bằng GRAPE", abs(by_token - grape) < tol),
        ("chuỗi một bước: LocalNat bằng GRAPE ở mọi k",
         one["n_steps"] == 1 and all(abs(one[column(k)] - grape) < tol for k in ks)),
        ("mốc bước đưa từ ngoài vào được dùng đúng (hai bước)", two["n_steps"] == 2 and two["n_items"] == 2),
        (f"chữ đầu của mỗi bước thuộc đúng bước đó với tokenizer thật (đầu các bước: {heads})",
         len(probe_steps) == 3 and heads[1].startswith("Therefore") and heads[2].startswith("Next")),
    ]
    bad = 0
    print()
    for name, ok in checks:
        bad += not ok
        print(f"  {'ĐẠT ' if ok else 'HỎNG'}  {name}")
    print(f"\n{len(checks) - bad}/{len(checks)} phép kiểm chứng đạt")
    if bad:
        print("Có phép HỎNG: KHÔNG chấm dữ liệu thật trước khi sửa.")
    return bad


# ============================================================ dòng lệnh
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--student", default="qwen1_5b", help="tên file trong configs/student/")
    ap.add_argument("--base", action="store_true", help="dùng bản Base thay vì Instruct (như b1_fit)")
    ap.add_argument("--steps", default=STEPS_FILE, help="file kết quả của segment_steps")
    ap.add_argument("--local-k", type=int, nargs="+", default=list(DEFAULT_KS), metavar="K",
                    help="các số bước ngữ cảnh cần chấm trong cùng một lượt (mặc định 1 4)")
    ap.add_argument("--plan", action="store_true", help="in khối lượng và nguồn bước rồi thoát, không cần GPU")
    ap.add_argument("--selftest", action="store_true", help="kiểm chứng trên GPU, không cần dữ liệu")
    ap.add_argument("--limit", type=int, help="chỉ chấm N chuỗi đầu, để thử")
    ap.add_argument("--allow-missing", action="store_true", help="bỏ qua chuỗi chưa được cắt bước thay vì dừng")
    ap.add_argument("--max-fallback-share", type=float, default=0.02,
                    help="dừng nếu tỷ lệ chuỗi phải cắt dự phòng vượt mức này (mặc định 2%%)")
    ap.add_argument("--load-4bit", action="store_true", help="nạp mô hình ở 4-bit (chỉ khi 16-bit không vừa)")
    ap.add_argument("--max-batch", type=int, default=MAX_BATCH)
    ap.add_argument("--attn-gb", type=float, default=ATTN_GB, help="ngân sách bộ nhớ attention của một lô")
    ap.add_argument("--out", help="tên file đầu ra, mặc định localnat.<student>[_base].glm.jsonl")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(student=args.student, overrides=args.override)
    model_id = cfg["student"]["base_model_id"] if args.base else cfg["student"]["model_id"]
    max_len = int(cfg["training"]["max_seq_len"])
    ks = parse_ks(args.local_k)

    if args.selftest:
        import torch
        from src.stage_b.b1_fit import load_student
        if not torch.cuda.is_available():
            raise SystemExit("Không thấy GPU. Lệnh này chạy trên máy Windows có card NVIDIA.")
        model, tok = load_student(model_id, args.load_4bit)
        return selftest(model, tok, max_len, int(cfg["signals"]["rank_chunk"]))

    wd: Path = resolve_path(cfg, args.workdir)
    files = cfg["stage_a_files"]
    quality = {r["tid"]: r for r in read_jsonl(wd / "quality.jsonl")}
    chains = usable_chains(read_jsonl(wd / files["candidates"]), quality)
    if not chains:
        raise SystemExit(f"Không có ứng viên dùng được trong {wd}.")
    if not (wd / args.steps).exists():
        raise SystemExit(f"Không thấy {wd / args.steps}. Chép file cắt bước từ máy Mac sang trước.")
    final, missing = prepare(chains, st.current_rows(read_jsonl(wd / args.steps)))
    sources = Counter(f["source"] for f in final.values())
    fallback = sum(n for s, n in sources.items() if s.startswith("fallback"))
    n_steps = [len(f["ends"]) for f in final.values()]
    n_items = sum(len(unique_items([(i, i + 1) for i in range(n)], ks)[0]) for n in n_steps)
    print(f"[localnat] {len(chains)} chuỗi dùng được; có bước {len(final)}, chưa cắt {len(missing)} | nguồn bước: "
          + ", ".join(f"{s} {n}" for s, n in sorted(sources.items()))
          + (f" | số bước trung vị {statistics.median(n_steps):.0f}" if n_steps else "")
          + f" | k = {ks}: {n_items} mục chấm, nếu chấm riêng từng k là {len(ks) * sum(n_steps)}")
    if missing and not args.allow_missing:
        raise SystemExit(f"[localnat] DỪNG: {len(missing)} chuỗi chưa được cắt bước (ví dụ {missing[:3]}). Chạy xong "
                         f"segment_steps rồi chép lại {args.steps}, hoặc thêm --allow-missing để chấm phần đã có.")
    if final and fallback / len(final) > args.max_fallback_share:
        raise SystemExit(f"[localnat] DỪNG: {fallback}/{len(final)} chuỗi phải cắt dự phòng, vượt mức "
                         f"{args.max_fallback_share:.1%}. Xem python -m src.tools.audit_steps --list-broken.")
    if args.plan:
        return 0

    import torch
    from src.stage_b.b1_fit import load_student
    if not torch.cuda.is_available():
        raise SystemExit("Không thấy GPU. Lệnh này chạy trên máy Windows có card NVIDIA.")
    questions = {q["qid"]: q["question"] for q in read_jsonl(wd / files["questions"])}
    tag = args.student + ("_base" if args.base else "")
    path = wd / (args.out or f"localnat.{tag}.glm.jsonl")
    done_rows = read_jsonl(path) if path.exists() else []
    check_resume(done_rows[0] if done_rows else None, ks, model_id, path.name)
    done = {r["tid"] for r in done_rows}
    todo = [c for c in chains if c["tid"] in final and c["tid"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[localnat] mô hình học {model_id}, {'4-bit' if args.load_4bit else '16-bit'}, độ dài tối đa {max_len}; "
          f"đã có {len(done)}, cần chấm {len(todo)} | ghi vào {path.name}")
    model, tok = load_student(model_id, args.load_4bit)
    print(f"[localnat] nạp xong, chiếm {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    failed, splits, t0 = [], 0, time.perf_counter()
    with JsonlWriter(path) as w:
        for c in tqdm(todo, desc="chấm LocalNat"):
            f = final[c["tid"]]
            try:
                v = score_chain(model, tok, questions[c["qid"]], c["text"], f["ends"], ks, max_len, args.max_batch,
                                args.attn_gb)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                v = None
            if v is None:
                failed.append(c["tid"])
                continue
            splits += v["oom_splits"]
            w.append({"tid": c["tid"], "qid": c["qid"], "model": model_id, "ts": now_iso(), **v,
                      "steps_source": f["source"], "steps_version": st.STEPS_VERSION,
                      "signals_version": SIGNALS_VERSION, "text_md5": st.text_md5(c["text"])})
    took = time.perf_counter() - t0
    rows = read_jsonl(path)
    print(f"[localnat] xong {len(todo) - len(failed)} chuỗi trong {took / 60:.1f} phút, đỉnh bộ nhớ "
          f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB, chia đôi lô {splits} lần"
          + (f"; {len(failed)} chuỗi KHÔNG chấm được, ví dụ {failed[:3]}" if failed else ""))
    if rows:
        for k in ks:
            vals = [r[column(k)] for r in rows]
            print(f"[localnat] {column(k)} trên {len(rows)} chuỗi: trung bình {statistics.mean(vals):.4f}, "
                  f"trung vị {statistics.median(vals):.4f}")
        cut = sum(r["n_steps"] < r["n_steps_text"] for r in rows)
        print(f"[localnat] {cut} chuỗi mất bước cuối vì vượt {max_len} token. Điểm càng CAO càng tự nhiên. So với GRAPE: "
              f"python -m src.tools.compare_localnat")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

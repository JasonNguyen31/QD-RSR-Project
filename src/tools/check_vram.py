"""Đo bộ nhớ GPU thật sự cần, trước khi chạy b1 và b3. Chạy trên máy Windows.

    python -m src.tools.check_vram                        # Qwen2.5-1.5B, cả hai phép đo
    python -m src.tools.check_vram --model Qwen/Qwen2.5-7B-Instruct --skip-train
    python -m src.tools.check_vram --seq 3072 --batch 1 2 4

Hai phép đo, tương ứng hai bước dùng GPU:
  1. **Đọc chuỗi ở 16-bit** (b1_fit): chỉ forward, không gradient. Cần cho RSR, GRAPE, LocalNat, LARK.
     Dùng 16-bit chứ không lượng tử 4-bit, vì lượng tử làm nhiễu thứ hạng token mà RSR dựa vào.
  2. **Một bước huấn luyện QLoRA 4-bit** (b3_train): có gradient và trạng thái bộ tối ưu, tốn nhất.

Nguyên tắc đã chốt: không để chiếm quá 10GB dù card có 12GB. Chạy sát trần dễ đổ giữa chừng.
Công cụ dùng dữ liệu giả, không cần candidates.jsonl, nên chạy được ngay.
"""
from __future__ import annotations

import argparse
import gc
from typing import Sequence

LIMIT_GB = 10.0


def gb(x: int) -> float:
    return x / 1024 ** 3


def reset(torch) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure_forward(model_id: str, seq: int, batch: int) -> dict:
    """Bước b1: đọc chuỗi, lấy logits, không tính gradient."""
    import torch
    from transformers import AutoModelForCausalLM

    reset(torch)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    after_load = torch.cuda.max_memory_allocated()
    ids = torch.randint(0, model.config.vocab_size, (batch, seq), device="cuda")
    with torch.no_grad():
        out = model(ids)
        # RSR cần thứ hạng token, tức phải xếp hạng trên toàn bộ từ vựng: đây là chỗ tốn bộ nhớ nhất
        logits = out.logits[:, :-1]
        target = ids[:, 1:].unsqueeze(-1)
        rank = (logits > logits.gather(-1, target)).sum(-1)
        del out, logits, rank
    peak = torch.cuda.max_memory_allocated()
    del model, ids
    reset(torch)
    return {"load": gb(after_load), "peak": gb(peak)}


def chunked_loss(model, ids, chunk: int = 256):
    """Mất mát cắt thành từng đoạn vị trí, để không dựng cả ma trận logit cùng lúc.

    Vấn đề: logit có kích thước (độ dài chuỗi) x (152.000 mục từ vựng). Ở 3.072 token, riêng bản 32-bit
    đã là 1,9GB, cộng gradient của nó và các bản sao trung gian thì vượt xa card 12GB.
    Cách chữa: lấy trạng thái ẩn trước, rồi tính lớp đầu ra và mất mát cho từng đoạn 256 vị trí, mỗi đoạn
    bọc trong checkpoint nên logit KHÔNG được giữ lại mà tính lại khi lan truyền ngược. Bộ nhớ khi đó tỷ lệ
    với kích thước đoạn chứ không với độ dài chuỗi. Đánh đổi là chậm hơn khoảng 20%.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids).last_hidden_state
    lm_head = base.lm_head
    total, n = 0.0, 0
    for i in range(0, ids.shape[1] - 1, chunk):
        j = min(i + chunk, ids.shape[1] - 1)

        def one(h, tgt):
            return F.cross_entropy(lm_head(h).float().flatten(0, 1), tgt.flatten(), reduction="sum")

        total = total + checkpoint(one, hidden[:, i:j], ids[:, i + 1:j + 1], use_reentrant=False)
        n += (j - i) * ids.shape[0]
    return total / max(n, 1)


def measure_train_step(model_id: str, seq: int, batch: int, grad_ckpt: bool, upcast: bool = True,
                       loss_mode: str = "default", diag: bool = False) -> dict:
    """Bước b3: một bước huấn luyện QLoRA 4-bit, gồm forward, backward và cập nhật tham số.

    upcast=False giữ lớp đầu ra ở 16-bit thay vì để prepare_model_for_kbit_training nâng lên 32-bit.
    Lớp đầu ra của Qwen2.5-1.5B có 152k x 1536 tham số, và ma trận logit là 152k cho MỖI vị trí chuỗi,
    nên nâng lên 32-bit làm tốn gấp đôi ở đúng chỗ tốn nhất. Đây thường là thủ phạm gây hết bộ nhớ.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    reset(torch)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map="cuda",
                                                 attn_implementation="sdpa", dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)
    if not upcast:
        # Phải hạ TOÀN BỘ tham số 32-bit xuống 16-bit, không chỉ lớp đầu ra: prepare_model_for_kbit_training
        # nâng cả các lớp chuẩn hoá, nên nếu chỉ hạ lớp đầu ra thì đầu vào 32-bit gặp trọng số 16-bit và
        # báo lỗi "expected mat1 and mat2 to have the same dtype".
        for prm in model.parameters():
            if prm.dtype == torch.float32:
                prm.data = prm.data.to(torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False          # bộ nhớ đệm khoá-giá trị vô dụng khi huấn luyện, chỉ tốn chỗ
    # BẮT BUỘC: gradient checkpointing trong transformers chỉ có tác dụng khi mô hình ở chế độ huấn luyện.
    # from_pretrained trả về mô hình ở chế độ đánh giá, nên thiếu dòng này thì cờ bật mà cơ chế không chạy,
    # và bộ nhớ tăng tuyến tính theo độ dài chuỗi vì giữ lại trạng thái trung gian của toàn bộ 28 lớp.
    model.train()
    after_load = torch.cuda.max_memory_allocated()

    if diag:
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        inner = base.model
        on = getattr(inner, "gradient_checkpointing", None)
        on = f"{on} (chế độ huấn luyện: {model.training})"
        norm_dtype = next((m.weight.dtype for n, m in inner.named_modules()
                           if "norm" in n.lower() and hasattr(m, "weight")), None)
        hid_dtype = next((m.weight.dtype for n, m in inner.named_modules()
                          if n.endswith("embed_tokens") and hasattr(m, "weight")), None)
        print(f"   [chẩn đoán] gradient checkpointing: {on} | attention: "
              f"{getattr(base.config, '_attn_implementation', '?')} | kiểu lớp chuẩn hoá: {norm_dtype} "
              f"| kiểu lớp nhúng: {hid_dtype}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    ids = torch.randint(0, model.config.vocab_size, (batch, seq), device="cuda")
    import time

    def one_step():
        loss = chunked_loss(model, ids) if loss_mode == "chunked" else model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    loss = one_step()                      # bước đầu gồm cả thời gian khởi tạo, không tính vào
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(2):
        loss = one_step()
    torch.cuda.synchronize()
    secs = (time.perf_counter() - t0) / 2
    peak = torch.cuda.max_memory_allocated()
    del model, opt, ids, loss
    reset(torch)
    return {"load": gb(after_load), "peak": gb(peak), "secs": secs,
            "tok_per_s": batch * seq / secs}


def verdict(peak: float) -> str:
    if peak <= LIMIT_GB:
        return "ĐẠT"
    return "VƯỢT TRẦN" if peak <= 11.5 else "KHÔNG CHẠY ĐƯỢC"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seq", type=int, default=3072)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="tắt gradient checkpointing: nhanh hơn nhưng tốn bộ nhớ hơn nhiều")
    ap.add_argument("--sweep", type=int, nargs="+",
                    help="quét nhiều độ dài chuỗi để tìm ngưỡng chạy được, ví dụ --sweep 1024 1536 2048 3072")
    ap.add_argument("--no-upcast", action="store_true",
                    help="hạ mọi tham số 32-bit xuống 16-bit sau prepare_model_for_kbit_training")
    ap.add_argument("--loss", choices=["default", "chunked"], default="default",
                    help="cách tính hàm mất mát khi đo bước huấn luyện")
    args = ap.parse_args(argv)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("Không thấy GPU. Lệnh này chạy trên máy Windows có card NVIDIA.")
    total = gb(torch.cuda.get_device_properties(0).total_memory)
    print(f"[bộ nhớ] {torch.cuda.get_device_name(0)}, tổng {total:.1f} GB, trần tự đặt {LIMIT_GB} GB")
    print(f"[bộ nhớ] {args.model}, độ dài chuỗi {args.seq}\n")
    if not args.sweep:
        print(f"cấu hình huấn luyện: {'16-bit' if args.no_upcast else '32-bit'}, "
              f"mất mát {'theo đoạn' if args.loss == 'chunked' else 'thường'}\n")
    print(f"{'phép đo':<30}{'lô':>4}{'đỉnh':>9}{'giây/bước':>12}{'token/giây':>13}{'kết luận':>17}")

    if args.sweep:
        combos = [("32-bit + thường", True, "default"), ("32-bit + đoạn", True, "chunked"),
                  ("16-bit + thường", False, "default"), ("16-bit + đoạn", False, "chunked")]
        print(f"\n{'độ dài':<9}" + "".join(f"{c[0]:>20}" for c in combos))
        first = True
        for sq in args.sweep:
            cells = []
            for _, upcast, mode in combos:
                try:
                    r = measure_train_step(args.model, sq, 1, not args.no_grad_ckpt,
                                           upcast=upcast, loss_mode=mode, diag=first)
                    first = False
                    cells.append(f"{r['peak']:.2f}G")
                except torch.cuda.OutOfMemoryError:
                    reset(torch)
                    cells.append("hết bộ nhớ")
                except RuntimeError as exc:
                    reset(torch)
                    cells.append(f"lỗi: {str(exc)[:28]}")
            print(f"{sq:<9}" + "".join(f"{c:>20}" for c in cells))
        print(f"\nTrần tự đặt {LIMIT_GB} GB. Lưu ý: trên Windows, driver NVIDIA có thể tràn sang RAM hệ thống")
        print("thay vì báo hết bộ nhớ, nên con số trên 12GB vẫn chạy được nhưng CHẬM KINH KHỦNG. Phải để dưới trần.")
        print("Dòng chẩn đoán ở trên cho biết gradient checkpointing có thật sự bật và các lớp đang ở kiểu nào.")
        print("Nếu cắt ngưỡng: 2.560 mất 0,92% chuỗi và 5 câu hỏi; 2.048 mất 2,55% chuỗi và 21 câu hỏi.")
        return 0

    rows = []
    for b in args.batch:
        if not args.skip_forward:
            try:
                r = measure_forward(args.model, args.seq, b)
                rows.append(("b1: đọc chuỗi, 16-bit", b, r))
            except torch.cuda.OutOfMemoryError:
                rows.append(("b1: đọc chuỗi, 16-bit", b, None))
        if not args.skip_train:
            try:
                r = measure_train_step(args.model, args.seq, b, not args.no_grad_ckpt,
                                        upcast=not args.no_upcast, loss_mode=args.loss,
                                        diag=(b == args.batch[0]))
                rows.append(("b3: một bước QLoRA 4-bit", b, r))
            except torch.cuda.OutOfMemoryError:
                rows.append(("b3: một bước QLoRA 4-bit", b, None))

    for name, b, r in rows:
        if r is None:
            print(f"{name:<30}{b:>4}{'—':>9}{'—':>12}{'—':>13}{'HẾT BỘ NHỚ':>17}")
        else:
            secs = r.get("secs")
            tps = r.get("tok_per_s")
            print(f"{name:<30}{b:>4}{r['peak']:>8.2f}G"
                  f"{(f'{secs:.2f}' if secs else '—'):>12}{(f'{tps:,.0f}' if tps else '—'):>13}"
                  f"{verdict(r['peak']):>17}")
    if any(r and r.get("tok_per_s") for _, _, r in rows):
        print("\nChọn lô lớn nhất còn ĐẠT mà token/giây vẫn tăng. Nếu token/giây ngừng tăng thì lô lớn hơn")
        print("chỉ tốn bộ nhớ chứ không nhanh thêm, vì đã chạm giới hạn băng thông chứ không phải tính toán.")

    print(f"\nCách đọc: cột đỉnh là mức cao nhất trong một bước. Dưới {LIMIT_GB} GB là an toàn.")
    print("Nếu b3 vượt trần: giảm kích thước lô xuống 1 và tăng số bước tích luỹ gradient để giữ")
    print("kích thước lô hiệu dụng, kết quả huấn luyện không đổi, chỉ chậm hơn.")
    print("Nếu b1 vượt trần: chia chuỗi thành từng đoạn khi tính thứ hạng token, hoặc hạ xuống lô 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

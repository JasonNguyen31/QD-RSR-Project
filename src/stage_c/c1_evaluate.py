"""c1_evaluate: đánh giá một mô hình đã tinh chỉnh trên các bộ đánh giá. Chạy trên máy có GPU, không cần mạng.

    python -m src.stage_c.c1_evaluate --student qwen1_5b --method correct_only --seed 42            # tầng 1
    python -m src.stage_c.c1_evaluate --student qwen1_5b --method correct_only --seed 42 --tier 3   # cả ba tầng
    python -m src.stage_c.c1_evaluate ... --benchmarks math500 --limit 64     # đo tốc độ trên 64 câu, kết quả ghi riêng
    python -m src.stage_c.c1_evaluate ... --benchmarks val                    # tập kiểm định chọn lambda
    python -m src.stage_c.c1_evaluate ... --plan                              # in khối lượng, không cần GPU

Đọc  data/eval/<bộ>.jsonl và manifest.json (c0_prepare_eval ghi; md5 phải khớp, lệch là dừng)
     outputs/models/<student>_base/<tên>/seed<seed>/adapter và run.json (b3_train ghi)
Ghi  outputs/results/eval/<student>_base/<tên>/seed<seed>/
        <bộ>.jsonl      một dòng mỗi lượt sinh: qid, sample, correct, pred, gold, n_tokens, stopped, text
        summary.json    chỉ số từng bộ, số token, tốc độ sinh, cài đặt, commit

Cách đánh giá theo Phụ lục A.4 bài RSR, đã chốt 30/09: mỗi câu sinh eval.n_samples lượt độc lập ở nhiệt độ 0,6 và
top-p 0,95; lượt chạm eval.max_new_tokens bị cắt và chấm như bình thường. Đề bài đi vào mô hình theo đúng khuôn lúc
huấn luyện (b3_train.prompt_messages). Việc sinh dừng ở token kết thúc mà lần huấn luyện đã học (đọc từ run.json).

Ba tầng (chốt 03/10): tầng 1 gồm bốn bộ của bài RSR và chạy cho mọi mô hình; tầng 2 là GSM8K; tầng 3 là BBH, rút
eval.bbh_per_task câu mỗi tác vụ theo seed cố định. --tier N chạy các tầng từ 1 đến N.

SINH THEO GIAI ĐOẠN. Một lô sinh chỉ xong khi lượt dài nhất của nó xong, nên một lượt chạy tới trần giữ cả lô lại
(đo ở phép thử dừng: lô 8 lượt mất đủ 3.072 bước vì một lượt không ra đáp án). Ở đây mọi lượt được sinh tới mốc
đầu của eval.stage_tokens với lô lớn; lượt nào chưa dừng thì được gom lại và sinh tiếp tới mốc sau với lô nhỏ hơn.
Lượt sinh tiếp nhận nguyên phần đã viết làm ngữ cảnh, nên phân phối của văn bản sinh ra không đổi; chỉ dòng số ngẫu
nhiên khác so với sinh một mạch.

Adapter KHÔNG được gộp vào trọng số nền. Trọng số nền ở bfloat16 chỉ có 8 bit phần định trị, cộng phần hiệu chỉnh
nhỏ của LoRA vào đó sẽ làm tròn mất một phần của nó; giữ adapter rời thì phép tính giống hệt lúc huấn luyện.

Chạy lại được: mỗi khối eval.chunk lượt xong là ghi ngay; chạy lại lệnh thì bỏ qua các lượt đã có. Seed của từng
khối cố định theo vị trí của khối, nên chạy một mạch hay chạy nối đều cho cùng kết quả. Khối lớn thì các giai đoạn
sau gom được đủ lượt cho một lô đầy; khối nhỏ thì ghi file thường xuyên hơn.

Đo ngày 04/10 trên RTX 3060 (64 câu MATH-500, lô 64/32/16, khối 256): 224 token/giây, khoảng 90 mili giây mỗi bước
sinh bất kể cỡ lô. Thời gian mỗi bước do phần cố định chi phối, nên tốc độ gần như tỷ lệ với số lượt còn đang sinh
trong lô: lô càng đầy càng nhanh. Dòng tiến độ in số mili giây mỗi bước của từng giai đoạn để chỉnh eval.gen_batch.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from src.common.answers import grade, is_equiv, last_boxed
from src.common.config import load_config, path_of, resolve_path
from src.common.io_utils import JsonlWriter, ensure_dir, iter_jsonl, now_iso, read_json, read_jsonl, write_json
from src.stage_b import b3_train as b3
from src.stage_b.b2_select import selection_tag, student_tag
from src.stage_c import metrics
from src.stage_c.c0_prepare_eval import file_md5, sample_per_subset

VAL = "val"                      # tập kiểm định chọn lambda, đọc từ data/raw chứ không từ data/eval


# ============================================================ phần thuần
def tier_benchmarks(cfg: Mapping, tier: int) -> list[str]:
    """Các bộ của tầng 1 đến tầng đã chọn, theo thứ tự chạy."""
    tiers = cfg["eval"]["tiers"]
    if not 1 <= tier <= len(tiers):
        raise SystemExit(f"--tier phải từ 1 đến {len(tiers)}")
    return [b for t in tiers[:tier] for b in t]


def load_questions(cfg: Mapping, bench: str, limit: int | None = None) -> list[dict]:
    """Câu hỏi của một bộ. BBH được rút mẫu cố định; --limit lấy các câu đầu sau bước đó."""
    if bench == VAL:
        path = path_of(cfg, "data_raw") / cfg["data"]["val_matched"]["file"]
        rows = [{"qid": r["qid"], "benchmark": VAL, "subset": r.get("source"), "question": r["question"],
                 "gold": r["gold"]} for r in iter_jsonl(path)]
    else:
        eval_dir = path_of(cfg, "data_eval")
        if not (eval_dir / "manifest.json").exists():
            raise SystemExit(f"Không thấy {eval_dir / 'manifest.json'}. Chạy c0_prepare_eval rồi chép data/eval sang máy này.")
        manifest = read_json(eval_dir / "manifest.json")["benchmarks"]
        if bench not in manifest:
            raise SystemExit(f"manifest.json không có bộ '{bench}'. Có: {sorted(manifest)} và '{VAL}'.")
        path = eval_dir / manifest[bench]["file"]
        if file_md5(path) != manifest[bench]["md5"]:
            raise SystemExit(f"{path.name}: md5 khác manifest. Chạy python -m src.stage_c.c0_prepare_eval --verify")
        rows = read_jsonl(path)
        if bench == "bbh":
            rows = sample_per_subset(rows, int(cfg["eval"]["bbh_per_task"]), int(cfg["eval"]["seed"]))
    return rows[:limit] if limit else rows


def work_items(questions: Sequence[Mapping], n_samples: int) -> list[tuple[str, int]]:
    """Mọi lượt sinh cần có: (qid, số thứ tự lượt), theo thứ tự câu hỏi của file."""
    return [(q["qid"], s) for q in questions for s in range(n_samples)]


def chunks(items: Sequence, size: int) -> list[list]:
    """Cắt danh sách lượt sinh thành các khối. Khối là đơn vị ghi file và đơn vị chạy lại (eval.chunk)."""
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def stage_plan(stage_tokens: Sequence[int], batches: Sequence[int], max_new_tokens: int) -> list[tuple[int, int]]:
    """Các giai đoạn sinh: (mốc token cộng dồn, cỡ lô). Mốc cuối luôn là max_new_tokens."""
    caps = sorted({int(c) for c in stage_tokens if 0 < int(c) < max_new_tokens}) + [int(max_new_tokens)]
    if len(batches) < len(caps):
        raise SystemExit(f"eval.gen_batch cần {len(caps)} cỡ lô cho các mốc {caps}, mới có {list(batches)}")
    return [(c, int(b)) for c, b in zip(caps, batches)]


_TEXT_WRAP = re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf|mbox)\s*\{([^{}]*)\}")
_NOT_BRACKET = re.compile(r"[^()\[\]{}<>]")


def bbh_normalize(s: str) -> str:
    """Đáp án BBH về dạng so được: bỏ lớp bọc \\text{}, dấu đô la, dấu chấm cuối; '(B)' và 'B' là một; không phân
    biệt hoa thường; dấu phẩy và khoảng trắng liền nhau coi như một khoảng trắng."""
    s = _TEXT_WRAP.sub(r"\1", s.strip()).replace("$", "").strip().rstrip(".").strip()
    m = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?", s)
    if m:
        return m.group(1).lower()
    return re.sub(r"[\s,]+", " ", s).strip().lower()


def bbh_equal(pred: str | None, gold: str) -> bool:
    """So đáp án BBH. Đáp án chỉ gồm dấu ngoặc (tác vụ dyck_languages) thì so đúng dãy ngoặc; còn lại so chuỗi đã
    chuẩn hoá, rồi so như số nếu cả hai là số."""
    if pred is None:
        return False
    if gold.strip() and re.fullmatch(r"[\s()\[\]{}<>]+", gold):
        return _NOT_BRACKET.sub("", _TEXT_WRAP.sub(r"\1", pred)) == _NOT_BRACKET.sub("", gold)
    return bbh_normalize(pred) == bbh_normalize(gold) or is_equiv(pred, gold)


def grade_answer(bench: str, text: str, gold: str) -> tuple[bool, str | None]:
    """(đúng hay sai, nội dung \\boxed cuối cùng). Bộ toán dùng bộ chấm của Giai đoạn A; BBH dùng bbh_equal."""
    if bench == "bbh":
        pred = last_boxed(text)
        return bbh_equal(pred, gold), pred
    g = grade(text, gold)
    return g.correct, g.pred


def summarise(bench: str, questions: Sequence[Mapping], rows: Sequence[Mapping], cfg: Mapping) -> dict:
    """Chỉ số của một bộ từ các dòng kết quả. Câu nào chưa đủ n_samples lượt thì dừng, không tính trên phần thiếu."""
    k = int(cfg["eval"]["n_samples"])
    by: dict[str, list] = {q["qid"]: [None] * k for q in questions}
    used = [r for r in rows if r["qid"] in by and r["sample"] < k]
    for r in used:
        by[r["qid"]][r["sample"]] = bool(r["correct"])
    missing = [q for q, v in by.items() if None in v]
    if missing:
        raise ValueError(f"{bench}: {len(missing)} câu chưa đủ {k} lượt, ví dụ {missing[0]}")
    names = list(cfg["eval"]["report"])
    out = dict(metrics.report(by, names))
    cap = int(cfg["eval"]["max_new_tokens"])
    out.update(n_questions=len(by), n_generations=len(used),
               mean_new_tokens=sum(r["n_tokens"] for r in used) / len(used),
               stopped_share=sum(bool(r["stopped"]) for r in used) / len(used),
               hit_cap_share=sum((not r["stopped"]) and r["n_tokens"] >= cap for r in used) / len(used),
               no_boxed_share=sum(r["pred"] is None for r in used) / len(used))
    subsets = Counter(q["subset"] for q in questions)
    if len(subsets) > 1:
        sub_of = {q["qid"]: q["subset"] for q in questions}
        out["by_subset"] = {str(s): metrics.report({q: v for q, v in by.items() if sub_of[q] == s}, names)
                            for s in sorted(subsets, key=str)}
    return out


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


# ============================================================ phần cần torch
def load_model(cfg: Mapping, adapter_dir: Path, device: str):
    """Mô hình nền ở đúng độ chính xác lúc huấn luyện, gắn adapter rời (không gộp, xem phần đầu file)."""
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
    model.config.use_cache = True
    return model, tok


def _generate_batch(model, tok, seqs: Sequence[Sequence[int]], max_new: int, stops: Sequence[int], ev: Mapping,
                    seed: int) -> tuple[list[list[int]], int]:
    """Sinh tiếp cho một lô chuỗi token (đệm trái). Trả về (token mới của từng dòng, số lần phải chia đôi lô vì
    hết bộ nhớ)."""
    import torch

    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, width - len(s):] = 1
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), do_sample=True,
                                 temperature=float(ev["temperature"]), top_p=float(ev["top_p"]),
                                 top_k=max(0, int(ev["top_k"])), max_new_tokens=int(max_new),
                                 eos_token_id=list(stops), pad_token_id=tok.pad_token_id)
        return out[:, width:].tolist(), 0
    except torch.OutOfMemoryError:
        if len(seqs) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(seqs) // 2
        a, ra = _generate_batch(model, tok, seqs[:half], max_new, stops, ev, seed)
        b, rb = _generate_batch(model, tok, seqs[half:], max_new, stops, ev, seed + 1)
        return a + b, ra + rb + 1


def generate_staged(model, tok, prompts: Sequence[Sequence[int]], plan: Sequence[tuple[int, int]], stops: Sequence[int],
                    ev: Mapping, seed: int, log=None) -> tuple[list[dict], dict]:
    """Sinh cho mọi đề theo từng giai đoạn. Trả về (mỗi đề: token đã sinh và có dừng hay không; số liệu của khối).
    stats["stages"] ghi cho từng giai đoạn: số lượt vào, số bước sinh, số token, số giây."""
    state = [{"tokens": [], "stopped": False} for _ in prompts]
    stats = {"steps": 0, "oom_splits": 0, "per_stage": [], "stages": []}
    prev = 0
    for stage, (cap, batch) in enumerate(plan):
        todo = sorted((i for i, s in enumerate(state) if not s["stopped"]),
                      key=lambda i: -(len(prompts[i]) + len(state[i]["tokens"])))
        stats["per_stage"].append(len(todo))
        rec = {"cap": cap, "batch": batch, "rows": len(todo), "steps": 0, "tokens": 0, "seconds": 0.0}
        n_batches = -(-len(todo) // batch)
        for b in range(0, len(todo), batch):
            idx = todo[b:b + batch]
            t0 = time.perf_counter()
            new_rows, splits = _generate_batch(model, tok, [list(prompts[i]) + state[i]["tokens"] for i in idx],
                                               cap - prev, stops, ev, seed * 100_003 + stage * 1_009 + b)
            took, steps, kept_total = time.perf_counter() - t0, max(len(r) for r in new_rows), 0
            stats["oom_splits"] += splits
            for i, row in zip(idx, new_rows):
                kept, stop = b3.split_at_stop(row, stops)
                state[i]["tokens"] += kept
                state[i]["stopped"] = stop is not None
                kept_total += len(kept)
            rec["steps"] += steps
            rec["tokens"] += kept_total
            rec["seconds"] += took
            if log:
                log(f"[c1]   giai đoạn {stage + 1} (tới {cap} token), lô {b // batch + 1}/{n_batches}: {len(idx)} lượt, "
                    f"{steps} bước trong {took:.0f} giây ({1000 * took / max(steps, 1):.0f} ms mỗi bước), "
                    f"{sum(not state[i]['stopped'] for i in idx)} lượt chưa dừng"
                    + (f", phải chia lô {splits} lần vì hết bộ nhớ" if splits else ""))
        stats["steps"] += rec["steps"]
        stats["stages"].append(rec)
        prev = cap
    return state, stats


def evaluate_benchmark(model, tok, cfg: Mapping, bench: str, questions: Sequence[Mapping], out_path: Path,
                       stops: Sequence[int], log=print) -> dict:
    """Sinh và chấm một bộ, ghi từng khối vào out_path. Trả về số liệu thời gian của phần vừa chạy."""
    ev = cfg["eval"]
    plan = stage_plan(ev["stage_tokens"], ev["gen_batch"], int(ev["max_new_tokens"]))
    by_qid = {q["qid"]: q for q in questions}
    done = {(r["qid"], r["sample"]) for r in iter_jsonl(out_path)} if out_path.exists() else set()
    todo = [(c, items) for c, items in enumerate(chunks(work_items(questions, int(ev["n_samples"])), int(ev["chunk"])))
            if any(it not in done for it in items)]
    timing = {"seconds": 0.0, "new_tokens": 0, "generations": 0, "oom_splits": 0, "steps": 0, "stages": []}
    if not todo:
        return timing
    prompt_ids = {}
    for q in questions:
        text = tok.apply_chat_template(b3.prompt_messages(q["question"]), tokenize=False, add_generation_prompt=True)
        prompt_ids[q["qid"]] = tok(text, add_special_tokens=False)["input_ids"]
    with JsonlWriter(out_path) as writer:
        for n_done, (c, items) in enumerate(todo, 1):
            items = [it for it in items if it not in done]
            t0 = time.perf_counter()
            state, stats = generate_staged(model, tok, [prompt_ids[q] for q, _s in items], plan, stops, ev,
                                           int(ev["seed"]) * 7919 + c, log)
            took = time.perf_counter() - t0
            for (qid, sample), st in zip(items, state):
                text = tok.decode(st["tokens"], skip_special_tokens=True)
                correct, pred = grade_answer(bench, text, by_qid[qid]["gold"])
                writer.append({"qid": qid, "sample": sample, "correct": correct, "pred": pred,
                               "gold": by_qid[qid]["gold"], "n_tokens": len(st["tokens"]), "stopped": st["stopped"],
                               "text": text})
            new = sum(len(s["tokens"]) for s in state)
            timing["seconds"] += took
            timing["new_tokens"] += new
            timing["generations"] += len(items)
            timing["oom_splits"] += stats["oom_splits"]
            timing["steps"] += stats["steps"]
            for j, rec in enumerate(stats["stages"]):               # cộng dồn số liệu từng giai đoạn qua các khối
                if j == len(timing["stages"]):
                    timing["stages"].append(dict(rec))
                else:
                    for key in ("rows", "steps", "tokens", "seconds"):
                        timing["stages"][j][key] += rec[key]
            left = (len(todo) - n_done) * timing["seconds"] / n_done
            log(f"[c1] {bench}: khối {n_done}/{len(todo)} | {len(items)} lượt, số lượt vào từng giai đoạn "
                f"{stats['per_stage']}, {new} token trong {took:.0f} giây | "
                f"{timing['new_tokens'] / timing['seconds']:.0f} token/giây | còn khoảng {left / 60:.0f} phút")
    return timing


# ============================================================ dòng lệnh
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="qwen1_5b")
    ap.add_argument("--method", help="tên file trong configs/method/")
    ap.add_argument("--ablation", help="tên file trong configs/ablation/")
    ap.add_argument("--seed", type=int, help="seed của lần huấn luyện; mặc định seed đầu")
    ap.add_argument("--tier", type=int, default=1, help="chạy các tầng từ 1 đến N (mặc định 1)")
    ap.add_argument("--benchmarks", nargs="+", help=f"chọn bộ cụ thể thay cho --tier; '{VAL}' là tập kiểm định")
    ap.add_argument("--limit", type=int, help="chỉ lấy N câu đầu mỗi bộ (đo tốc độ); kết quả ghi vào thư mục riêng")
    ap.add_argument("--run-dir", help="thư mục lần huấn luyện, mặc định theo b3_train")
    ap.add_argument("--plan", action="store_true", help="in khối lượng công việc rồi thoát, không cần GPU")
    ap.add_argument("--force", action="store_true", help="xoá kết quả cũ của các bộ được chọn rồi chạy lại")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)
    if not (args.method or args.ablation):
        ap.error("cần --method hoặc --ablation")

    cfg = load_config(method=args.method or "qd_rsr", ablation=args.ablation, student=args.student,
                      overrides=args.override)
    ev = cfg["eval"]
    seed = args.seed if args.seed is not None else int(cfg["project"]["train_seeds"][0])
    tag = selection_tag(cfg)
    benches = list(args.benchmarks) if args.benchmarks else tier_benchmarks(cfg, args.tier)
    data = {b: load_questions(cfg, b, args.limit) for b in benches}
    k = int(ev["n_samples"])
    plan = stage_plan(ev["stage_tokens"], ev["gen_batch"], int(ev["max_new_tokens"]))
    print(f"[c1] {tag} seed {seed} | " + ", ".join(f"{b} {len(q)} câu" for b, q in data.items())
          + f" | {k} lượt mỗi câu, tổng {sum(len(q) for q in data.values()) * k} lượt sinh"
          + f" | giai đoạn (mốc token, cỡ lô): {plan}")
    if args.plan:
        return 0

    run = resolve_path(cfg, args.run_dir) if args.run_dir else b3.run_dir(cfg, tag, seed)
    info = read_json(run / "run.json") if (run / "run.json").exists() else {}
    if not (run / "adapter").exists() or (info.get("status") != "done" and not args.run_dir):
        raise SystemExit(f"Lần huấn luyện ở {run} chưa xong (cần adapter/ và run.json có status done).")
    out_dir = ensure_dir(path_of(cfg, "out_results") / "eval" / student_tag(cfg) / tag
                         / (run.name + (f".limit{args.limit}" if args.limit else "")))

    import torch

    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(cfg["hardware"]["torch_memory_fraction"]))
        torch.cuda.reset_peak_memory_stats()
    model, tok = load_model(cfg, run / "adapter", args.device)
    eot_ids = [int(i) for i in (info.get("end_of_turn_ids")
                                or b3.end_token_ids(tok, str(cfg["training"]["end_token"])))]
    stops = b3.stop_token_ids(tok, eot_ids)

    spath = out_dir / "summary.json"
    summary = read_json(spath) if spath.exists() else {"benchmarks": {}}
    summary.update(tag=tag, seed=seed, student=cfg["student"]["base_model_id"], adapter=str(run / "adapter"),
                   train_commit=info.get("git_commit"), eval_commit=git_commit(Path(cfg.root)),
                   stop_tokens=tok.convert_ids_to_tokens(stops),
                   settings={key: ev[key] for key in ("n_samples", "temperature", "top_p", "top_k", "max_new_tokens",
                                                      "seed", "stage_tokens", "gen_batch", "chunk", "bbh_per_task")})
    for bench, questions in data.items():
        path = out_dir / f"{bench}.jsonl"
        if args.force and path.exists():
            path.unlink()
        timing = evaluate_benchmark(model, tok, cfg, bench, questions, path, stops)
        res = summarise(bench, questions, read_jsonl(path), cfg)
        prev = {} if args.force else summary["benchmarks"].get(bench, {})
        res["seconds"] = round(prev.get("seconds", 0.0) + timing["seconds"], 1)
        res["tokens_per_second"] = round(timing["new_tokens"] / timing["seconds"], 1) if timing["seconds"] \
            else prev.get("tokens_per_second")
        res["oom_splits"] = timing["oom_splits"]
        if timing["stages"]:
            res["stages"] = [{**r, "seconds": round(r["seconds"], 1),
                              "ms_per_step": round(1000 * r["seconds"] / max(r["steps"], 1), 1)} for r in timing["stages"]]
            print("[c1]   theo giai đoạn: " + " | ".join(
                f"tới {r['cap']} (lô {r['batch']}): {r['rows']} lượt, {r['steps']} bước, {r['seconds'] / 60:.1f} phút, "
                f"{r['ms_per_step']:.0f} ms mỗi bước" for r in res["stages"]))
        res["finished"] = now_iso()
        summary["benchmarks"][bench] = res
        if args.device == "cuda":
            summary["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        write_json(spath, summary)
        print(f"[c1] {bench}: " + ", ".join(f"{n} {100 * res[n]:.1f}" for n in ev["report"])
              + f" | {res['n_questions']} câu | trung bình {res['mean_new_tokens']:.0f} token, dừng "
              f"{100 * res['stopped_share']:.0f}%, chạm trần {100 * res['hit_cap_share']:.0f}%, không có \\boxed "
              f"{100 * res['no_boxed_share']:.0f}% | {res['seconds'] / 60:.1f} phút"
              + (f", {res['tokens_per_second']:.0f} token/giây" if res["tokens_per_second"] else "")
              + (f" | đỉnh bộ nhớ giữ chỗ {summary['peak_reserved_gb']} GB" if "peak_reserved_gb" in summary else ""))
    print(f"[c1] kết quả ở {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

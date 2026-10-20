"""c2_deploy: đóng gói một mô hình đã tinh chỉnh để chạy trên máy cá nhân (Ollama, LM Studio) và đo nó. Giai đoạn D.

    python -m src.stage_c.c2_deploy --student qwen1_5b --method grape --seed 42 --step merge
    python -m src.stage_c.c2_deploy ... --step gguf --llama-cpp ~/llama.cpp       # GGUF f16 rồi lượng tử Q4_K_M
    python -m src.stage_c.c2_deploy ... --step ollama                             # tạo mô hình trong Ollama
    python -m src.stage_c.c2_deploy ... --step bench --n 20                       # tốc độ, bộ nhớ, kích thước file
    python -m src.stage_c.c2_deploy ... --step eval --limit 50                    # thử MATH-500 trên 50 câu
    python -m src.stage_c.c2_deploy ... --step eval                               # MATH-500 đủ 500 câu x 4 lượt
    python -m src.stage_c.c2_deploy ... --step all --llama-cpp ~/llama.cpp        # năm bước liền, bước nào xong thì bỏ qua
    python -m src.stage_c.c2_deploy ... --plan                                    # in đường dẫn và tên, không chạy gì

Đọc  outputs/models/<student>_base/<tên>/seed<seed>/adapter và run.json (b3_train ghi; chép từ Windows sang)
     data/eval/math500.jsonl và manifest.json (c0_prepare_eval ghi)
Ghi  outputs/deploy/<student>_base/<tên>/seed<seed>/
        merged/                       trọng số nền đã cộng LoRA, kèm tokenizer (định dạng Hugging Face)
        model-f16.gguf                bản GGUF chưa lượng tử (16-bit), làm mốc tốc độ và kích thước
        model-<lượng tử>.gguf         bản lượng tử, mặc định Q4_K_M; đây là file mở được trên Ollama và LM Studio
        Modelfile                     khuôn hội thoại, system prompt và tham số sinh y như lúc huấn luyện và đánh giá
        deploy.json                   mọi số đo: kích thước file, tốc độ, bộ nhớ, tên mô hình trong Ollama
     outputs/results/deploy/<student>_base/<tên>/seed<seed>.<lượng tử>/math500.jsonl và summary.json
        cùng định dạng dòng với c1_evaluate (qid, sample, correct, pred, gold, n_tokens, stopped, text), nên so được
        với kết quả bản gốc bằng phép so ghép cặp theo câu hỏi (--reference; mặc định là file math500.jsonl của
        c1_evaluate cho đúng tập chọn và seed này, nếu máy có).

Năm bước:
  merge   nạp mô hình nền và adapter, cộng LoRA vào trọng số (merge_and_unload) rồi lưu. Mặc định cộng và lưu ở
          float32 cho mô hình 3 tỷ trở xuống: c1_evaluate ghi rằng cộng LoRA vào trọng số bfloat16 làm tròn mất một
          phần hiệu chỉnh; bản float32 giữ đủ, rồi GGUF f16 (10 bit phần định trị) giữ nhiều hơn bfloat16 (7 bit).
          Mô hình 7 tỷ ở float32 cần khoảng 30 GB nên mặc định là bfloat16; hơn nữa adapter 7B được huấn luyện trên
          trọng số nền lượng tử NF4, cộng vào bản nền 16-bit là lệch so với lúc huấn luyện. Ghi rõ trong Hạn chế.
  gguf    llama.cpp: convert_hf_to_gguf.py đổi merged/ sang GGUF f16, llama-quantize lượng tử hoá.
  ollama  ghi Modelfile và gọi `ollama create`. Khuôn trong Modelfile được kiểm là cho ra đúng chuỗi đề bài mà
          tokenizer dựng lúc huấn luyện (b3_train.prompt_messages), và system prompt Ollama nhận được đúng nguyên văn.
  bench   gửi --n đề MATH-500 đầu tiên, từng đề một (không song song), đo tốc độ đọc đề và tốc độ sinh mà Ollama
          báo, bộ nhớ Ollama báo cho mô hình (/api/ps) và bộ nhớ thường trú của tiến trình runner, kích thước file.
  eval    chấm lại MATH-500 trên bản lượng tử: 4 lượt mỗi câu, cùng nhiệt độ, top-p, trần token, bộ chấm như
          c1_evaluate. Ghi từng lượt ngay khi xong, chạy lại thì làm tiếp.

Gọi Ollama ở chế độ raw: chuỗi đề bài do chính tokenizer dựng (giống hệt c1_evaluate), nên khuôn của Modelfile không
ảnh hưởng số đo. Tham số mặc định của Ollama khác cài đặt đánh giá (top_k 40, repeat_penalty 1.1, có thể có min_p)
nên mọi lần gọi đều đặt lại: top_k 0 (tắt), min_p 0, repeat_penalty 1, nhiệt độ và top-p theo eval. Token dừng:
<|endoftext|> (token kết thúc mà mô hình đã học, chốt 03/10) và <|im_end|>.

Không tái lập từng bit: llama.cpp sinh song song theo lô và lượng tử hoá làm đổi phân phối, nên seed chỉ giữ cho lần
chạy lặp lại gần giống. Phép so với bản gốc là phép so trung bình trên 500 câu, không phải so từng lượt.

Cần: transformers, peft, torch (bước merge, chạy trên CPU); bản sao llama.cpp có convert_hf_to_gguf.py và lệnh
llama-quantize (brew install llama.cpp, hoặc build/bin trong bản sao); Ollama đang chạy (ollama serve hoặc ứng dụng).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Mapping, Sequence

from src.common.config import load_config, path_of, resolve_path
from src.common.io_utils import JsonlWriter, ensure_dir, iter_jsonl, now_iso, read_json, read_jsonl, write_json
from src.common.prompts import SYSTEM_PROMPT_TRAIN
from src.stage_b import b3_train as b3
from src.stage_b.b2_select import selection_tag, student_tag
from src.stage_c import c1_evaluate as c1

STEPS = ("merge", "gguf", "ollama", "bench", "eval")
STOP_STRINGS = ("<|endoftext|>", "<|im_end|>")
BENCH = "math500"
DEFAULT_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not DEFAULT_URL.startswith("http"):
    DEFAULT_URL = "http://" + DEFAULT_URL

# Khuôn hội thoại Qwen2.5 có system prompt, viết theo cú pháp khuôn của Ollama. qwen_prompt() là cùng khuôn đó viết
# bằng Python; bài kiểm thử giữ hai bản khớp nhau, còn bước ollama kiểm chúng khớp tokenizer thật.
OLLAMA_TEMPLATE = ("{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
                   "<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n")


def qwen_prompt(system: str, question: str) -> str:
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return f"{head}<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def render_ollama_template(template: str, system: str, prompt: str) -> str:
    """Dựng khuôn OLLAMA_TEMPLATE như Ollama làm (chỉ đủ cho đúng khuôn này: một khối if .System, hai biến)."""
    m = re.fullmatch(r"\{\{- if \.System \}\}(.*?)\{\{ end \}\}(.*)", template, re.S)
    if not m:
        raise ValueError("khuôn không đúng dạng mong đợi")
    sys_part, rest = m.groups()
    out = (sys_part if system else "") + rest
    return out.replace("{{ .System }}", system).replace("{{ .Prompt }}", prompt)


# ============================================================ tên và đường dẫn
def deploy_dir(cfg: Mapping, tag: str, seed: int) -> Path:
    return Path(cfg["root"]) / "outputs" / "deploy" / student_tag(cfg) / tag / f"seed{seed}"


def results_dir(cfg: Mapping, tag: str, seed: int, quant: str, limit: int | None = None) -> Path:
    name = f"seed{seed}.{quant.lower()}" + (f".limit{limit}" if limit else "")
    return path_of(cfg, "out_results") / "deploy" / student_tag(cfg) / tag / name


def gguf_name(quant: str) -> str:
    return "model-f16.gguf" if quant.upper() == "F16" else f"model-{quant.upper()}.gguf"


def ollama_model_name(cfg: Mapping, tag: str, seed: int, quant: str) -> str:
    """Ví dụ qdrsr-grape-qwen1_5b-s42:q4_k_m. Tên Ollama chỉ nhận chữ thường, số, '_', '-', '.'."""
    name = tag.split(".k")[0].replace("_", "-")
    return f"qdrsr-{name}-{cfg['student']['key']}-s{seed}:{quant.lower()}"


def default_merge_dtype(cfg: Mapping) -> str:
    return "float32" if float(cfg["student"]["params_b"]) <= 3 else "bfloat16"


def sampling_options(cfg: Mapping, num_ctx: int, seed: int) -> dict:
    """Tham số sinh gửi cho Ollama, theo đúng eval của base.yaml; đặt lại các mặc định của Ollama khác cài đặt đó."""
    ev = cfg["eval"]
    top_k = int(ev["top_k"])
    return {"temperature": float(ev["temperature"]), "top_p": float(ev["top_p"]), "top_k": top_k if top_k > 0 else 0,
            "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "num_predict": int(ev["max_new_tokens"]), "num_ctx": int(num_ctx), "seed": int(seed),
            "stop": list(STOP_STRINGS)}


def modelfile_text(gguf: str, cfg: Mapping, num_ctx: int) -> str:
    """Modelfile cho người dùng trò chuyện trực tiếp: cùng khuôn, system prompt và tham số sinh như lúc đánh giá."""
    o = sampling_options(cfg, num_ctx, int(cfg["eval"]["seed"]))
    lines = [f"FROM ./{gguf}", f'TEMPLATE """{OLLAMA_TEMPLATE}"""', f'SYSTEM """{SYSTEM_PROMPT_TRAIN}"""']
    for key in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "num_predict", "num_ctx"):
        lines.append(f"PARAMETER {key} {o[key]}")
    lines += [f'PARAMETER stop "{s}"' for s in STOP_STRINGS]
    return "\n".join(lines) + "\n"


# ============================================================ Ollama qua HTTP (không cần thư viện thêm)
def http_json(url: str, payload: Mapping | None = None, timeout: float = 900.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Ollama trả lỗi {e.code} ở {url}: {e.read().decode('utf-8', 'replace')[:300]}") from None
    except urllib.error.URLError as e:
        raise SystemExit(f"Không gọi được Ollama ở {url}: {e}. Ollama đã chạy chưa (mở ứng dụng Ollama hoặc "
                         f"`ollama serve` ở một cửa sổ khác)?") from None


def generate(base_url: str, model: str, prompt: str, options: Mapping, post: Callable = http_json) -> dict:
    """Một lượt sinh ở chế độ raw. Trả về văn bản và các số đo Ollama báo (thời gian tính bằng giây)."""
    r = post(f"{base_url}/api/generate", {"model": model, "prompt": prompt, "raw": True, "stream": False,
                                          "options": dict(options), "keep_alive": "30m"})
    ns = 1e-9
    return {"text": r.get("response", ""), "done_reason": r.get("done_reason"),
            "eval_count": int(r.get("eval_count") or 0), "eval_s": (r.get("eval_duration") or 0) * ns,
            "prompt_count": int(r.get("prompt_eval_count") or 0), "prompt_s": (r.get("prompt_eval_duration") or 0) * ns,
            "load_s": (r.get("load_duration") or 0) * ns, "total_s": (r.get("total_duration") or 0) * ns}


def to_row(qid: str, sample: int, gold: str, g: Mapping, cap: int) -> dict:
    """Một dòng kết quả cùng định dạng c1_evaluate. 'stopped' là dừng vì token kết thúc, không phải vì chạm trần."""
    correct, pred = c1.grade_answer(BENCH, g["text"], gold)
    stopped = g["done_reason"] == "stop" and g["eval_count"] < cap
    return {"qid": qid, "sample": sample, "correct": correct, "pred": pred, "gold": gold, "n_tokens": g["eval_count"],
            "stopped": stopped, "done_reason": g["done_reason"], "text": g["text"]}


def parse_runners(ps_out: str, gguf_blob: str | None = None) -> dict:
    """Từ bản in `ps -axo rss=,command=`: bộ nhớ thường trú (MB) của tiến trình chạy mô hình, số lượt song song (-np)
    và độ dài ngữ cảnh tổng (-c). Ollama cũ chạy mô hình bằng `ollama runner`, Ollama từ 0.40 bằng `llama-server`
    nằm trong thư mục cài Ollama. Trên Mac, bộ nhớ GPU dùng chung cũng có thể bị tính vào số thường trú; vì vậy ghi
    kèm số Ollama tự báo (/api/ps) và nói rõ cả hai trong bài."""
    rss, slots, ctx = [], None, None
    for line in ps_out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        cmd = parts[1]
        if "ollama" not in cmd or not ("runner" in cmd or "llama-server" in cmd):
            continue
        if gguf_blob and "--model" in cmd and gguf_blob not in cmd:
            continue
        rss.append(int(parts[0]))
        m = re.search(r"(?:^|\s)-np\s+(\d+)", cmd)
        slots = int(m.group(1)) if m else slots
        m = re.search(r"(?:^|\s)-c\s+(\d+)", cmd)
        ctx = int(m.group(1)) if m else ctx
    return {"runner_rss_mb": round(sum(rss) / 1024, 1) if rss else None, "runner_slots": slots, "runner_ctx": ctx}


def runner_info(gguf_blob: str | None = None) -> dict:
    try:
        out = subprocess.run(["ps", "-axo", "rss=,command="], capture_output=True, text=True, check=True).stdout
    except Exception:
        return parse_runners("")
    return parse_runners(out, gguf_blob)


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# ============================================================ các bước
def step_merge(cfg: Mapping, run: Path, out: Path, dtype: str, log=print) -> dict:
    merged = out / "merged"
    if (merged / "config.json").exists():
        log(f"[c2] merge: đã có {merged}, bỏ qua")
        return {}
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg["student"]["base_model_id"]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"     # bản cũ chỉ biết torch_dtype và lờ đi dtype
    base = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", **{key: getattr(torch, dtype)})
    if base.dtype != getattr(torch, dtype):
        raise SystemExit(f"Mô hình nền nạp ở {base.dtype}, không phải {dtype} như yêu cầu.")
    model = PeftModel.from_pretrained(base, str(run / "adapter")).merge_and_unload()
    tmp = out / "merged.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    tmp.rename(merged)
    log(f"[c2] merge: xong {merged} ({dtype}) trong {time.time() - t0:.0f} giây")
    return {"merge_dtype": dtype, "merged_bytes": dir_bytes(merged)}


def find_quantize(llama_cpp: Path | None) -> str:
    if llama_cpp and (llama_cpp / "build" / "bin" / "llama-quantize").exists():
        return str(llama_cpp / "build" / "bin" / "llama-quantize")
    found = shutil.which("llama-quantize")
    if not found:
        raise SystemExit("Không thấy lệnh llama-quantize. Cài bằng `brew install llama.cpp`, hoặc build llama.cpp rồi "
                         "trỏ --llama-cpp tới thư mục đó.")
    return found


def step_gguf(out: Path, quant: str, llama_cpp: Path | None, log=print) -> dict:
    merged, f16, q = out / "merged", out / gguf_name("F16"), out / gguf_name(quant)
    if not (merged / "config.json").exists():
        raise SystemExit(f"Chưa có {merged}. Chạy --step merge trước.")
    if not f16.exists():
        if llama_cpp is None or not (llama_cpp / "convert_hf_to_gguf.py").exists():
            raise SystemExit("Cần --llama-cpp trỏ tới bản sao llama.cpp có convert_hf_to_gguf.py "
                             "(git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp).")
        t0 = time.time()
        tmp = out / (f16.name + ".tmp")
        subprocess.run([sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(tmp),
                        "--outtype", "f16"], check=True)
        tmp.rename(f16)
        log(f"[c2] gguf: {f16.name} {f16.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f} giây")
    if quant.upper() != "F16" and not q.exists():
        t0 = time.time()
        tmp = out / (q.name + ".tmp")
        subprocess.run([find_quantize(llama_cpp), str(f16), str(tmp), quant.upper()], check=True)
        tmp.rename(q)
        log(f"[c2] gguf: {q.name} {q.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f} giây")
    return {"files_bytes": {p.name: p.stat().st_size for p in sorted(out.glob("*.gguf"))}}


def prompt_of(tok, question: str) -> str:
    return tok.apply_chat_template(b3.prompt_messages(question), tokenize=False, add_generation_prompt=True)


def max_prompt_tokens(tok, questions: Sequence[Mapping]) -> int:
    return max(len(tok(prompt_of(tok, q["question"]), add_special_tokens=False)["input_ids"]) for q in questions)


def check_template(tok) -> None:
    probe = "What is 1 + 1?"
    want = prompt_of(tok, probe)
    if qwen_prompt(SYSTEM_PROMPT_TRAIN, probe) != want or \
            render_ollama_template(OLLAMA_TEMPLATE, SYSTEM_PROMPT_TRAIN, probe) != want:
        raise SystemExit(f"Khuôn của Modelfile không cho ra đúng chuỗi đề bài mà tokenizer dựng lúc huấn luyện. "
                         f"Tokenizer dựng: {want!r}")


def template_renders_training_prompt(template: str, tok) -> bool:
    """Khuôn Ollama lưu có cho ra đúng chuỗi đề bài lúc huấn luyện không. Ollama cũ giữ khuôn Go của Modelfile;
    Ollama từ 0.40 chạy bằng llama-server và dùng khuôn Jinja nằm sẵn trong file GGUF (khuôn của tokenizer Qwen),
    bỏ qua TEMPLATE của Modelfile. Với khuôn Jinja thì dựng thử bằng jinja2 với đúng system prompt và đề bài."""
    probe = "What is 1 + 1?"
    want = prompt_of(tok, probe)
    if template == OLLAMA_TEMPLATE:
        return render_ollama_template(template, SYSTEM_PROMPT_TRAIN, probe) == want
    if "{%" not in template:
        return False
    import jinja2
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, **_k: json.dumps(x, ensure_ascii=False)
    got = env.from_string(template).render(messages=b3.prompt_messages(probe), add_generation_prompt=True, tools=None)
    return got == want


def step_ollama(cfg: Mapping, out: Path, quant: str, name: str, url: str, num_ctx: int, tok, log=print) -> dict:
    gguf = out / gguf_name(quant)
    if not gguf.exists():
        raise SystemExit(f"Chưa có {gguf}. Chạy --step gguf trước.")
    check_template(tok)
    (out / "Modelfile").write_text(modelfile_text(gguf.name, cfg, num_ctx), encoding="utf-8")
    if shutil.which("ollama") is None:
        raise SystemExit("Không thấy lệnh ollama. Cài Ollama (brew install ollama, hoặc ứng dụng từ ollama.com).")
    subprocess.run(["ollama", "create", name, "-f", "Modelfile"], cwd=out, check=True)
    shown = http_json(f"{url}/api/show", {"model": name})
    system_ok = (shown.get("system") or "").strip() == SYSTEM_PROMPT_TRAIN
    template = shown.get("template") or ""
    template_ok = template_renders_training_prompt(template, tok)
    kind = "Modelfile" if template == OLLAMA_TEMPLATE else ("Jinja trong GGUF" if "{%" in template else "khác")
    log(f"[c2] ollama: đã tạo {name}; system prompt {'khớp' if system_ok else 'KHÁC'}, khuôn ({kind}) "
        f"{'cho ra đúng đề bài lúc huấn luyện' if template_ok else 'KHÔNG cho ra đúng đề bài lúc huấn luyện'}"
        + ("" if system_ok and template_ok else f" (Ollama nhận system={shown.get('system')!r}). Số đo của bước bench "
                                                f"và eval KHÔNG bị ảnh hưởng vì gọi ở chế độ raw; chỉ lúc trò chuyện "
                                                f"trực tiếp mới cần khuôn này."))
    return {"ollama_model": name, "modelfile_system_ok": system_ok, "modelfile_template_ok": template_ok,
            "ollama_template_kind": kind}


def step_bench(cfg: Mapping, name: str, url: str, questions: Sequence[Mapping], tok, num_ctx: int, n: int,
               post: Callable = http_json, log=print) -> dict:
    """Từng đề một, đo tốc độ Ollama báo. Lượt đầu nạp mô hình nên thời gian nạp ghi riêng."""
    runs = []
    for i, q in enumerate(questions[:n]):
        g = generate(url, name, prompt_of(tok, q["question"]), sampling_options(cfg, num_ctx, 1000 + i), post)
        runs.append(g)
        log(f"[c2] bench {i + 1}/{n}: {g['eval_count']} token, {g['eval_count'] / max(g['eval_s'], 1e-9):.1f} token/giây")
    ps = post(f"{url}/api/ps")
    model_mem = next((m for m in ps.get("models", []) if name in (m.get("name"), m.get("model"))), {})
    gen = [g["eval_count"] / g["eval_s"] for g in runs if g["eval_s"] > 0]
    pre = [g["prompt_count"] / g["prompt_s"] for g in runs if g["prompt_s"] > 0]
    mb = lambda x: round(x / 2 ** 20, 1) if x else None
    return {"n": len(runs), "finished": now_iso(), "num_ctx": num_ctx,
            "gen_tok_per_s_median": round(statistics.median(gen), 1) if gen else None,
            "gen_tok_per_s_mean": round(statistics.mean(gen), 1) if gen else None,
            "prompt_tok_per_s_median": round(statistics.median(pre), 1) if pre else None,
            "mean_new_tokens": round(statistics.mean(g["eval_count"] for g in runs), 1) if runs else None,
            "first_load_s": round(runs[0]["load_s"], 2) if runs else None,
            "ollama_size_mb": mb(model_mem.get("size")), "ollama_size_vram_mb": mb(model_mem.get("size_vram")),
            **runner_info()}


def step_eval(cfg: Mapping, name: str, url: str, questions: Sequence[Mapping], tok, num_ctx: int, out_path: Path,
              workers: int, post: Callable = http_json, log=print) -> dict:
    """Sinh và chấm MATH-500 trên bản lượng tử; ghi từng lượt, chạy lại thì bỏ qua các lượt đã có."""
    ev = cfg["eval"]
    k, cap = int(ev["n_samples"]), int(ev["max_new_tokens"])
    items = c1.work_items(questions, k)
    done = {(r["qid"], r["sample"]) for r in iter_jsonl(out_path)} if out_path.exists() else set()
    by_qid = {q["qid"]: q for q in questions}
    todo = [(i, it) for i, it in enumerate(items) if it not in done]
    t0, n_tok, n_done = time.time(), 0, 0
    if todo:
        prompts = {q["qid"]: prompt_of(tok, q["question"]) for q in questions}
        log(f"[c2] eval: còn {len(todo)}/{len(items)} lượt, {workers} lượt gửi cùng lúc")

        def one(i: int, it: tuple) -> dict:
            qid, sample = it
            g = generate(url, name, prompts[qid], sampling_options(cfg, num_ctx, int(ev["seed"]) * 100003 + i), post)
            return to_row(qid, sample, by_qid[qid]["gold"], g, cap)

        with JsonlWriter(out_path) as writer, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for f in as_completed([pool.submit(one, i, it) for i, it in todo]):
                row = f.result()
                writer.append(row)
                n_done += 1
                n_tok += row["n_tokens"]
                if n_done % 50 == 0 or n_done == len(todo):
                    took = max(time.time() - t0, 1e-9)
                    log(f"[c2] eval: {n_done}/{len(todo)} lượt | {n_tok / took:.0f} token/giây cả máy | "
                        f"còn khoảng {(len(todo) - n_done) * took / n_done / 60:.0f} phút")
    res = c1.summarise(BENCH, questions, read_jsonl(out_path), cfg)
    took = time.time() - t0
    res["seconds_this_run"] = round(took, 1)
    res["tokens_per_second_all_workers"] = round(n_tok / took, 1) if n_done and took > 0 else None
    return res


def compare(ours: Path, reference: Path, n_samples: int) -> dict | None:
    """Chênh lệch Acc@4 ghép cặp theo câu hỏi: bản lượng tử trừ bản gốc (thang 0 đến 1), khoảng tin cậy 95%."""
    from src.tools.eval_table import paired_bootstrap, question_acc
    a, b = question_acc(ours, n_samples), question_acc(reference, n_samples)
    d = paired_bootstrap(a, b)
    if d is None:
        return None
    common = sorted(set(a) & set(b))
    return {**d, "ours_acc4": sum(a[q] for q in common) / len(common),
            "reference_acc4": sum(b[q] for q in common) / len(common), "reference": str(reference)}


# ============================================================ dòng lệnh
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="qwen1_5b")
    ap.add_argument("--method", help="tên file trong configs/method/")
    ap.add_argument("--ablation", help="tên file trong configs/ablation/")
    ap.add_argument("--seed", type=int, help="seed của lần huấn luyện; mặc định seed đầu")
    ap.add_argument("--step", choices=STEPS + ("all",), help="bước cần chạy")
    ap.add_argument("--quant", default="Q4_K_M", help="kiểu lượng tử của llama-quantize, hoặc F16 (mặc định Q4_K_M)")
    ap.add_argument("--llama-cpp", help="thư mục bản sao llama.cpp (có convert_hf_to_gguf.py)")
    ap.add_argument("--merge-dtype", choices=("float32", "bfloat16"), help="mặc định float32 nếu mô hình ≤ 3 tỷ")
    ap.add_argument("--ollama-url", default=DEFAULT_URL)
    ap.add_argument("--n", type=int, default=20, help="số đề cho bước bench")
    ap.add_argument("--limit", type=int, help="bước eval: chỉ lấy N câu đầu (kết quả ghi vào thư mục riêng)")
    ap.add_argument("--workers", type=int, default=4, help="bước eval: số lượt gửi Ollama cùng lúc; Ollama chỉ chạy "
                                                           "song song khi OLLAMA_NUM_PARALLEL ít nhất bằng số này")
    ap.add_argument("--reference", help="file math500.jsonl của bản gốc để so; mặc định đường dẫn của c1_evaluate")
    ap.add_argument("--run-dir", help="thư mục lần huấn luyện, mặc định theo b3_train")
    ap.add_argument("--plan", action="store_true", help="in đường dẫn và tên rồi thoát")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)
    if not (args.method or args.ablation):
        ap.error("cần --method hoặc --ablation")
    if not (args.step or args.plan):
        ap.error("cần --step hoặc --plan")

    cfg = load_config(method=args.method or "qd_rsr", ablation=args.ablation, student=args.student,
                      overrides=args.override)
    seed = args.seed if args.seed is not None else int(cfg["project"]["train_seeds"][0])
    tag, quant = selection_tag(cfg), args.quant.upper()
    run = resolve_path(cfg, args.run_dir) if args.run_dir else b3.run_dir(cfg, tag, seed)
    out = deploy_dir(cfg, tag, seed)
    name = ollama_model_name(cfg, tag, seed, quant)
    res_dir = results_dir(cfg, tag, seed, quant, args.limit)
    dtype = args.merge_dtype or default_merge_dtype(cfg)
    reference = (resolve_path(cfg, args.reference) if args.reference else
                 path_of(cfg, "out_results") / "eval" / student_tag(cfg) / tag / run.name / f"{BENCH}.jsonl")
    print(f"[c2] {tag} seed {seed} | adapter {run / 'adapter'} | đóng gói vào {out} | lượng tử {quant}, cộng LoRA ở "
          f"{dtype} | Ollama: {name} | chấm lại ghi vào {res_dir} | so với {reference}"
          + ("" if reference.exists() else " (chưa có trên máy này)"))
    if args.plan:
        return 0

    steps = list(STEPS) if args.step == "all" else [args.step]
    if "merge" in steps and not (out / "merged" / "config.json").exists():
        info = read_json(run / "run.json") if (run / "run.json").exists() else {}
        if not (run / "adapter").exists() or info.get("status") != "done":
            raise SystemExit(f"Lần huấn luyện ở {run} chưa xong hoặc chưa chép sang máy này (cần adapter/ và run.json "
                             f"có status done).")
    ensure_dir(out)
    meta_path = out / "deploy.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    meta.update(tag=tag, seed=seed, student=cfg["student"]["base_model_id"], adapter=str(run / "adapter"),
                quant=quant, ollama_model=name, system_prompt=SYSTEM_PROMPT_TRAIN)

    tok = questions = None
    num_ctx = 0
    if any(s in steps for s in ("ollama", "bench", "eval")):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["student"]["base_model_id"])
        check_template(tok)
        questions = c1.load_questions(cfg, BENCH, args.limit)
        num_ctx = max_prompt_tokens(tok, questions) + int(cfg["eval"]["max_new_tokens"]) + 16
        meta["num_ctx"] = num_ctx

    for s in steps:
        if s == "merge":
            meta.update(step_merge(cfg, run, out, dtype))
            if (run / "adapter").exists():
                meta["adapter_bytes"] = dir_bytes(run / "adapter")
        elif s == "gguf":
            meta.update(step_gguf(out, quant, Path(os.path.expanduser(args.llama_cpp)) if args.llama_cpp else None))
        elif s == "ollama":
            meta.update(step_ollama(cfg, out, quant, name, args.ollama_url, num_ctx, tok))
        elif s == "bench":
            b = step_bench(cfg, name, args.ollama_url, questions, tok, num_ctx, args.n)
            meta.setdefault("bench", {})[quant] = b
            print(f"[c2] bench {quant}: sinh {b['gen_tok_per_s_median']} token/giây (trung vị {b['n']} đề), đọc đề "
                  f"{b['prompt_tok_per_s_median']} token/giây | Ollama báo {b['ollama_size_mb']} MB (trên GPU "
                  f"{b['ollama_size_vram_mb']} MB) | tiến trình chạy mô hình {b['runner_rss_mb']} MB thường trú, "
                  f"{b['runner_slots']} lượt song song, ngữ cảnh tổng {b['runner_ctx']}")
        elif s == "eval":
            ensure_dir(res_dir)
            res = step_eval(cfg, name, args.ollama_url, questions, tok, num_ctx, res_dir / f"{BENCH}.jsonl", args.workers)
            v = compare(res_dir / f"{BENCH}.jsonl", reference, int(cfg["eval"]["n_samples"])) if reference.exists() else None
            write_json(res_dir / "summary.json", {
                "tag": tag, "seed": seed, "quant": quant, "ollama_model": name, "finished": now_iso(),
                "settings": sampling_options(cfg, num_ctx, int(cfg["eval"]["seed"])), "benchmarks": {BENCH: res},
                "versus_reference": v})
            line = (f"[c2] eval {quant}: Acc@4 {100 * res['acc@4']:.1f}, Pass@4 {100 * res['pass@4']:.1f} | "
                    f"{res['n_questions']} câu | trung bình {res['mean_new_tokens']:.0f} token, chạm trần "
                    f"{100 * res['hit_cap_share']:.0f}%, không có \\boxed {100 * res['no_boxed_share']:.0f}%")
            if v:
                line += (f" | bản gốc {100 * v['reference_acc4']:.1f}, chênh {100 * v['diff']:+.1f} "
                         f"(từ {100 * v['lo']:+.1f} đến {100 * v['hi']:+.1f}) trên {v['n']} câu")
            print(line)
            meta.setdefault("eval", {})[res_dir.name] = {"acc@4": res["acc@4"], "pass@4": res["pass@4"],
                                                         "versus_reference": v, "results": str(res_dir)}
        write_json(meta_path, meta)
    print(f"[c2] số đo ghi ở {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

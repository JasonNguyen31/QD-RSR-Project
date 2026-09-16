"""B3: Sinh chuoi suy luan pilot tu 3 mo hinh day qua OpenRouter."""
import json, os, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

TEACHERS = {
    "deepseek-v3":   "deepseek/deepseek-chat",
    "qwen2.5-72b":   "qwen/qwen-2.5-72b-instruct",
    "llama3.3-70b":  "meta-llama/llama-3.3-70b-instruct",
}
N_SAMPLES   = 3          # so chuoi moi mo hinh day sinh cho moi cau
TEMPERATURE = 0.8
TOP_P       = 0.95
MAX_TOKENS  = 1024
N_WORKERS   = 8

IN_FILE  = Path("data/pilot/questions.jsonl")
OUT_FILE = Path("data/pilot/trajectories.jsonl")

PROMPT = (
    "Solve the following math problem. Think step by step, "
    "then give the final numeric answer on the last line "
    "in the form: #### <answer>\n\nProblem: {q}"
)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

lock  = threading.Lock()
stats = {"calls": 0, "errors": 0, "in_tok": 0, "out_tok": 0, "seconds": 0.0}

@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=2, min=2, max=60))
def call_model(model_id: str, question: str):
    return client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": PROMPT.format(q=question)}],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

def one_job(item, teacher, model_id, k):
    t0 = time.time()
    try:
        r = call_model(model_id, item["question"])
        dt = time.time() - t0
        u = r.usage
        with lock:
            stats["calls"]   += 1
            stats["seconds"] += dt
            stats["in_tok"]  += getattr(u, "prompt_tokens", 0) or 0
            stats["out_tok"] += getattr(u, "completion_tokens", 0) or 0
        return {
            "tid": f'{item["qid"]}__{teacher}__{k}',
            "qid": item["qid"], "teacher": teacher, "sample_idx": k,
            "question": item["question"], "gold_answer": item["answer"],
            "trajectory": r.choices[0].message.content,
            "latency_sec": round(dt, 2),
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        }
    except Exception as e:
        with lock:
            stats["errors"] += 1
        print(f"\n[LOI] {item['qid']} {teacher} #{k}: {type(e).__name__}: {str(e)[:100]}")
        return None

def main():
    items = [json.loads(l) for l in IN_FILE.open(encoding="utf-8")]

    done = set()
    if OUT_FILE.exists():
        for l in OUT_FILE.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["tid"])
            except Exception:
                pass
        print(f"Da co san {len(done)} chuoi, se bo qua.")

    jobs = [
        (it, tname, mid, k)
        for it in items
        for tname, mid in TEACHERS.items()
        for k in range(N_SAMPLES)
        if f'{it["qid"]}__{tname}__{k}' not in done
    ]
    print(f"Can sinh {len(jobs)} chuoi (tong dang le {len(items)*len(TEACHERS)*N_SAMPLES}).")
    if not jobs:
        print("Khong con gi de sinh.")
        return

    t_start = time.time()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("a", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(one_job, *j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Sinh chuoi"):
            rec = fut.result()
            if rec:
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()

    wall = time.time() - t_start
    n = max(stats["calls"], 1)
    print("\n=== SO LIEU DO DUOC ===")
    print(f"Goi thanh cong  : {stats['calls']}")
    print(f"Loi (sau retry) : {stats['errors']}")
    print(f"Tong thoi gian  : {wall/60:.1f} phut")
    print(f"Do tre trung binh moi luot: {stats['seconds']/n:.1f} giay")
    print(f"Token vao : {stats['in_tok']:,}  (tb {stats['in_tok']/n:.0f}/luot)")
    print(f"Token ra  : {stats['out_tok']:,}  (tb {stats['out_tok']/n:.0f}/luot)")
    print(f"\nUoc tinh cho 18.000 chuoi that: x{18000/n:.1f} lan lo pilot nay")

if __name__ == "__main__":
    main()

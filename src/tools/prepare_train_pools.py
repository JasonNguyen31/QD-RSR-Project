"""B2 (phan con lai): Chuan bi kho cau hoi cho B9 + tap kiem dinh."""
import json, random, re
from pathlib import Path
from datasets import load_dataset

SEED = 42
N_VAL = 200
OUT = Path("data/raw")

def norm(s: str) -> str:
    """Chuan hoa de doi chieu trung lap: bo khoang trang va ky tu khong phai chu so."""
    return re.sub(r"\s+", " ", s.strip()).lower()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    # --- 1. MATH-500: chi de lay danh sach cau can loai ---
    print("Tai MATH-500 (bo danh gia, chi dung de doi chieu)...")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    banned = {norm(r["problem"]) for r in math500}
    print(f"  MATH-500 co {len(math500)} cau, {len(banned)} cau hoi duy nhat")

    # --- 2. MATH train, loai cau trung MATH-500 ---
    print("\nTai MATH (tap train)...")
    math = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
    configs = ["algebra", "counting_and_probability", "geometry",
               "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    rows = []
    for c in configs:
        d = load_dataset("EleutherAI/hendrycks_math", c, split="train")
        for r in d:
            rows.append({"problem": r["problem"], "solution": r["solution"],
                         "level": r.get("level", ""), "type": c})
    print(f"  Tong MATH train: {len(rows)} cau")

    kept, removed = [], 0
    for r in rows:
        if norm(r["problem"]) in banned:
            removed += 1
        else:
            kept.append(r)
    print(f"  Da loai {removed} cau trung MATH-500")
    print(f"  Con lai: {len(kept)} cau")

    with (OUT / "math_pool.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(kept):
            f.write(json.dumps({
                "qid": f"math_{i:05d}", "source": "math",
                "question": r["problem"], "solution": r["solution"],
                "level": r["level"], "type": r["type"],
            }, ensure_ascii=False) + "\n")

    # --- 3. GSM8K pool, loai 100 cau da dung cho pilot ---
    print("\nTai GSM8K (tap train)...")
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    pilot_idx = set()
    pf = Path("data/pilot/questions.jsonl")
    if pf.exists():
        pilot_idx = {json.loads(l)["orig_index"] for l in pf.open(encoding="utf-8")}
        print(f"  Loai {len(pilot_idx)} cau da dung cho lo pilot")

    with (OUT / "gsm8k_pool.jsonl").open("w", encoding="utf-8") as f:
        n = 0
        for k in range(len(gsm)):
            if k in pilot_idx:
                continue
            r = gsm[k]
            f.write(json.dumps({
                "qid": f"gsm8k_{k:05d}", "source": "gsm8k", "orig_index": k,
                "question": r["question"],
                "answer": r["answer"].split("####")[-1].strip().replace(",", ""),
                "solution": r["answer"],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"  GSM8K pool: {n} cau")

    # --- 4. Tap kiem dinh 200 cau, tach RIENG khoi hai pool tren ---
    print(f"\nTach tap kiem dinh {N_VAL} cau (100 GSM8K + 100 MATH)...")
    gsm_pool = [json.loads(l) for l in (OUT / "gsm8k_pool.jsonl").open(encoding="utf-8")]
    math_pool = [json.loads(l) for l in (OUT / "math_pool.jsonl").open(encoding="utf-8")]

    val_gsm = random.sample(gsm_pool, N_VAL // 2)
    val_math = random.sample(math_pool, N_VAL // 2)
    val_ids = {r["qid"] for r in val_gsm + val_math}

    with (OUT / "validation.jsonl").open("w", encoding="utf-8") as f:
        for r in val_gsm + val_math:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Ghi lai hai pool da tru tap kiem dinh
    for name, pool in [("gsm8k_pool", gsm_pool), ("math_pool", math_pool)]:
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in pool:
                if r["qid"] not in val_ids:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== KET QUA ===")
    for p in ["gsm8k_pool.jsonl", "math_pool.jsonl", "validation.jsonl"]:
        n = sum(1 for _ in (OUT / p).open(encoding="utf-8"))
        print(f"  {p:22s}: {n:>6,} cau")
    print(f"\nDa loai {removed} cau MATH trung MATH-500.")
    print(f"Tap kiem dinh {N_VAL} cau khong trung pilot, khong trung pool, khong trung MATH-500.")

if __name__ == "__main__":
    main()

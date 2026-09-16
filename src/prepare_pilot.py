"""B2: Chuan bi lo pilot 100 cau GSM8K."""
import json, random
from pathlib import Path
from datasets import load_dataset

SEED = 42
N_PILOT = 100
OUT = Path("data/pilot/questions.jsonl")

def extract_answer(sol: str) -> str:
    """Dap an GSM8K nam sau dau #### o cuoi loi giai."""
    return sol.split("####")[-1].strip().replace(",", "")

def main():
    print("Dang tai GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"Tong so cau trong tap train: {len(ds)}")

    random.seed(SEED)
    idx = random.sample(range(len(ds)), N_PILOT)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, k in enumerate(idx):
            row = ds[k]
            f.write(json.dumps({
                "qid": f"gsm8k_pilot_{i:03d}",
                "source": "gsm8k",
                "orig_index": k,
                "question": row["question"],
                "answer": extract_answer(row["answer"]),
                "solution": row["answer"],
            }, ensure_ascii=False) + "\n")

    print(f"Da ghi {N_PILOT} cau vao {OUT} (seed={SEED})")

    with OUT.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    print("\n--- Cau dau tien de kiem tra ---")
    print("Cau hoi:", first["question"][:150])
    print("Dap an :", first["answer"])

if __name__ == "__main__":
    main()

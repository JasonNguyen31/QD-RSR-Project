"""Kiem tra ham so khop co that su hoat dong khong."""
import re
from datasets import load_dataset

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()

math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
banned = {norm(r["problem"]) for r in math500}
print(f"MATH-500: {len(banned)} cau hoi duy nhat\n")

configs = ["algebra", "counting_and_probability", "geometry",
           "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

for split in ["test", "train"]:
    hit = total = 0
    for c in configs:
        d = load_dataset("EleutherAI/hendrycks_math", c, split=split)
        for r in d:
            total += 1
            if norm(r["problem"]) in banned:
                hit += 1
    print(f"MATH {split:5s}: {hit:>4} / {total:>5} cau trung MATH-500")

print("\nKY VONG: test phai trung gan 500 cau, train phai trung 0 cau.")
print("Neu test cung ra 0 -> ham so khop HONG, phai sua truoc khi chay B9.")

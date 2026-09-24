"""Phân tích thành phần Qual: hai nửa của nó có bổ trợ hay triệt tiêu nhau.

    python -m src.tools.inspect_qual --workdir data/stage_a

Xuất phát từ quan sát ngày 22/09: ba chuỗi có điểm quy tắc cao nhất đều là chuỗi DeepSeek trên 1.100 từ
mà giám khảo chấm 0,0 đến 0,1. Nếu hiện tượng đó phổ biến thì với alpha = 0,5, hai nửa triệt tiêu nhau và
Qual mất khả năng phân biệt. Lệnh này đo mức độ phổ biến, và thử vài cách chuẩn hoá khác nhau.

Không gọi mạng, chỉ đọc quality.jsonl và candidates.jsonl.
"""
from __future__ import annotations

import argparse
import statistics
from typing import Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl, split_tid
from src.tools.compare_judges import rankdata, spearman


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def minmax(xs: Sequence[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [0.5] * len(xs) if hi - lo < 1e-12 else [(x - lo) / (hi - lo) for x in xs]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    args = ap.parse_args(argv)

    cfg = load_config()
    wd = resolve_path(cfg, args.workdir)
    rows = [r for r in read_jsonl(wd / "quality.jsonl") if r.get("llm_score") is not None]
    if not rows:
        raise SystemExit("Không thấy quality.jsonl có điểm giám khảo.")

    rule = [r["rule_score"] for r in rows]
    llm = [r["llm_score"] for r in rows]
    words = [r["n_words"] for r in rows]

    print(f"[Qual] {len(rows)} chuỗi có đủ hai nguồn\n")
    print("1. HAI NỬA CÓ CÙNG CHIỀU KHÔNG (tương quan hạng trên toàn bộ)")
    print(f"   điểm quy tắc  và điểm giám khảo : {spearman(rule, llm):+.3f}")
    print(f"   điểm quy tắc  và số từ          : {spearman(rule, words):+.3f}")
    print(f"   điểm giám khảo và số từ         : {spearman(llm, words):+.3f}")
    print("   Tương quan âm giữa hai nửa nghĩa là chúng kéo ngược nhau, và với alpha = 0,5 thì triệt tiêu.")
    print("   Tương quan cao giữa điểm quy tắc và số từ nghĩa là nửa này chủ yếu đang đo ĐỘ DÀI.")

    hi_rule = sorted(rows, key=lambda r: -r["rule_score"])[: len(rows) // 10]
    share = sum(1 for r in hi_rule if r["llm_score"] <= 0.5) / len(hi_rule)
    print(f"\n2. NHÓM 10% ĐIỂM QUY TẮC CAO NHẤT ({len(hi_rule)} chuỗi)")
    print(f"   điểm giám khảo trung bình : {statistics.mean(r['llm_score'] for r in hi_rule):.3f} "
          f"(toàn bộ: {statistics.mean(llm):.3f})")
    print(f"   số từ trung bình          : {statistics.mean(r['n_words'] for r in hi_rule):.0f} "
          f"(toàn bộ: {statistics.mean(words):.0f})")
    print(f"   tỷ lệ bị giám khảo chấm <= 0,5 : {share:.1%}")

    print("\n3. TRONG TỪNG CÂU HỎI: Qual có phân biệt được các ứng viên không")
    by_q: dict[str, list] = {}
    for r in rows:
        by_q.setdefault(split_tid(r["tid"])[0], []).append(r)
    multi = [v for v in by_q.values() if len(v) >= 2]
    variants = {
        "alpha 0,5 (hiện tại)": lambda rr: 0.5,
        "alpha 0,7 (nặng quy tắc)": lambda rr: 0.7,
        "alpha 0,3 (nặng giám khảo)": lambda rr: 0.3,
    }
    print(f"   {'cách tính':<28}{'khoảng biến thiên TB':>22}{'số câu Qual hoà nhau':>24}")
    for name, af in variants.items():
        spans, ties = [], 0
        for v in multi:
            a = af(v)
            q = [a * x + (1 - a) * y for x, y in zip(minmax([r["rule_score"] for r in v]),
                                                     minmax([r["llm_score"] for r in v]))]
            spans.append(max(q) - min(q))
            ties += len(set(round(x, 3) for x in q)) < len(q)
        print(f"   {name:<28}{statistics.mean(spans):>22.3f}{f'{ties}/{len(multi)}':>24}")

    # cắt z-score để giá trị ngoại lai không ép các chuỗi còn lại về 0
    spans_clip = []
    for v in multi:
        a = 0.5
        q = [a * x + (1 - a) * y for x, y in zip(minmax([clip(r["rule_score"], -3, 3) for r in v]),
                                                 minmax([r["llm_score"] for r in v]))]
        spans_clip.append(max(q) - min(q))
    print(f"   {'alpha 0,5 + cắt z ở ±3':<28}{statistics.mean(spans_clip):>22.3f}")
    print("\n   Khoảng biến thiên càng lớn thì Qual càng phân biệt được các ứng viên trong cùng câu hỏi.")
    print("   Số câu hoà nhau càng nhiều thì Qual càng vô dụng cho việc chọn lọc ở những câu đó.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

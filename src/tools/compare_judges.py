"""So hai mô hình giám khảo: chúng có xếp hạng chuỗi giống nhau không, và chênh bao nhiêu tiền.

    python -m src.tools.compare_judges data/pilot/run2_boxed judge_flash.jsonl judge_lite.jsonl

Vì sao đo thứ hạng chứ không đo điểm: hàm mục tiêu chuẩn hoá min-max TRONG TỪNG CÂU HỎI rồi so các chuỗi
với nhau. Giám khảo rẻ hơn chỉ cần xếp hạng giống giám khảo đắt là đủ, không cần chấm cùng điểm số.

Ba phép đo, quan trọng dần:
  1. Tương quan hạng Spearman trên toàn bộ chuỗi — cái nhìn tổng quát, nhưng KHÔNG phải điều quyết định.
  2. Tỷ lệ đồng thuận trên các cặp chuỗi CÙNG một câu hỏi — đây mới là việc mà b2_select thực sự làm.
  3. Tỷ lệ chọn cùng chuỗi tốt nhất cho mỗi câu hỏi — sát nhất với hệ quả cuối cùng.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl


def rankdata(values: Sequence[float]) -> list[float]:
    """Hạng trung bình cho giá trị bằng nhau, giống scipy.stats.rankdata."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else None


def pair_agreement(by_q: Mapping[str, list[tuple[float, float]]]) -> dict:
    """Với mỗi cặp chuỗi cùng câu hỏi: hai giám khảo có xếp cùng chiều không."""
    same = diff = tie_a = tie_b = 0
    for pairs in by_q.values():
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                (a1, b1), (a2, b2) = pairs[i], pairs[j]
                if a1 == a2:
                    tie_a += 1
                    continue
                if b1 == b2:
                    tie_b += 1
                    continue
                same += (a1 > a2) == (b1 > b2)
                diff += (a1 > a2) != (b1 > b2)
    total = same + diff
    return {"same": same, "diff": diff, "tie_a": tie_a, "tie_b": tie_b,
            "rate": same / total if total else None, "comparable": total}


def top1_agreement(by_q: Mapping[str, list[tuple[float, float]]]) -> dict:
    hit = n = 0
    for pairs in by_q.values():
        if len(pairs) < 2:
            continue
        n += 1
        best_a = max(range(len(pairs)), key=lambda i: pairs[i][0])
        best_b = max(range(len(pairs)), key=lambda i: pairs[i][1])
        # coi là trùng nếu chuỗi mà A chọn cũng đạt điểm cao nhất theo B (cho phép hoà)
        hit += pairs[best_a][1] == pairs[best_b][1]
    return {"hit": hit, "questions": n, "rate": hit / n if n else None}


def report_self_preference(wd, file_names: Sequence[str]) -> None:
    """Điểm trung bình mà mỗi giám khảo cho từng mô hình dạy, tính trên đúng những chuỗi mọi giám khảo cùng chấm.

    Vì sao cần: giám khảo cùng họ kiến trúc với một mô hình dạy có thể ưu ái chuỗi của mô hình đó. Nếu giám khảo
    Qwen chấm chuỗi Qwen cao hơn hẳn so với cách các giám khảo khác nhìn cùng những chuỗi ấy, trong khi hai mô
    hình dạy còn lại không đổi, thì đó là dấu hiệu thiên lệch cùng họ. So cột theo hàng, không so số tuyệt đối.
    """
    from src.common.io_utils import split_tid

    loaded = []
    for name in file_names:
        rows = read_jsonl(wd / name)
        if rows:
            loaded.append((rows[0].get("model", name), {r["tid"]: r["overall_score"] for r in rows}))
    if len(loaded) < 2:
        return
    common = set.intersection(*(set(d) for _, d in loaded))
    if len(common) < 10:
        print(f"\n4. Thiên lệch cùng họ: chỉ {len(common)} chuỗi được mọi giám khảo cùng chấm, chưa đủ để xét.")
        return

    teachers = sorted({split_tid(t)[1] for t in common})
    print(f"\n4. Điểm trung bình theo mô hình dạy, trên {len(common)} chuỗi mọi giám khảo cùng chấm")
    width = max(len(m) for m, _ in loaded) + 2
    print(f"{'giám khảo':<{width}}" + "".join(f"{t:>14}" for t in teachers) + f"{'chênh lệch':>13}")
    for model, scores in loaded:
        means = []
        for t in teachers:
            vals = [scores[tid] for tid in common if split_tid(tid)[1] == t]
            means.append(sum(vals) / len(vals) if vals else float("nan"))
        spread = max(means) - min(means)
        print(f"{model:<{width}}" + "".join(f"{m:>14.3f}" for m in means) + f"{spread:>13.3f}")
    print("   Đọc theo HÀNG: mỗi giám khảo xếp ba mô hình dạy theo thứ tự nào. Nếu một giám khảo đảo thứ tự")
    print("   so với các giám khảo khác, và mô hình được nâng lên đúng là mô hình cùng họ với nó, thì cần nêu")
    print("   trong phần Hạn chế. Chênh lệch nhỏ ở mọi hàng nghĩa là không có dấu hiệu thiên lệch.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir")
    ap.add_argument("file_a", help="file điểm của giám khảo thứ nhất, ví dụ judge_cmp_flash.jsonl")
    ap.add_argument("file_b", help="file điểm của giám khảo thứ hai, ví dụ judge_cmp_lite.jsonl")
    ap.add_argument("more", nargs="*", default=[],
                    help="thêm file giám khảo khác, chỉ dùng cho bảng thiên lệch cùng họ ở mục 4")
    args = ap.parse_args(argv)

    cfg = load_config()
    wd = resolve_path(cfg, args.workdir)
    A = {r["tid"]: r for r in read_jsonl(wd / args.file_a)}
    B = {r["tid"]: r for r in read_jsonl(wd / args.file_b)}
    shared = sorted(set(A) & set(B))
    if len(shared) < 3:
        raise SystemExit(f"Chỉ có {len(shared)} chuỗi được chấm bởi cả hai, không đủ để so. "
                         f"Chạy a4 với cùng --group, --limit-questions và --sample-seed cho cả hai.")

    a_name = next(iter(A.values())).get("model", args.file_a)
    b_name = next(iter(B.values())).get("model", args.file_b)
    sa = [A[t]["overall_score"] for t in shared]
    sb = [B[t]["overall_score"] for t in shared]
    print(f"A = {a_name}\nB = {b_name}\nSố chuỗi cả hai cùng chấm: {len(shared)}")
    print(f"\nĐiểm trung bình: A {sum(sa) / len(sa):.3f}, B {sum(sb) / len(sb):.3f}")
    print(f"Số giá trị khác nhau: A {len(set(sa))}, B {len(set(sb))}")

    rho = spearman(sa, sb)
    print(f"\n1. Tương quan hạng Spearman toàn cục: {rho:.3f}" if rho is not None else "\n1. Không đủ dữ liệu")

    by_q: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for t in shared:
        by_q[A[t]["qid"]].append((A[t]["overall_score"], B[t]["overall_score"]))
    multi = {q: v for q, v in by_q.items() if len(v) >= 2}
    print(f"\n2. Cặp chuỗi cùng câu hỏi ({len(multi)} câu có từ 2 chuỗi trở lên)")
    pa = pair_agreement(multi)
    if pa["rate"] is None:
        print("   Không có cặp nào so được. Chạy lại a4 với --limit-questions để có nhiều chuỗi cùng câu.")
    else:
        print(f"   Đồng thuận {pa['same']}/{pa['comparable']} = {pa['rate']:.1%}  "
              f"(A hoà {pa['tie_a']} cặp, B hoà {pa['tie_b']} cặp, các cặp hoà không tính)")

    t1 = top1_agreement(multi)
    if t1["rate"] is not None:
        print(f"\n3. Chọn cùng chuỗi tốt nhất: {t1['hit']}/{t1['questions']} = {t1['rate']:.1%} số câu hỏi")

    report_self_preference(wd, [args.file_a, args.file_b] + list(args.more))

    print("\nCách quyết: nếu đồng thuận cặp từ 75% trở lên và Spearman từ 0,7 trở lên thì dùng giám khảo rẻ,")
    print("vì chênh lệch còn lại nhỏ hơn nhiễu do lấy mẫu, mà tiết kiệm được phần lớn chi phí.")
    print("Nếu thấp hơn thì hai giám khảo nhìn chuỗi khác nhau, nên trả thêm tiền cho bản đắt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

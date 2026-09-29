"""So hai file điểm của b1, và đo tương quan giữa bốn tín hiệu nội tại.

    python -m src.tools.compare_fit data/pilot/run2_boxed fit.qwen1_5b.jsonl fit.qwen1_5b_base.jsonl
    python -m src.tools.compare_fit data/pilot/run2_boxed fit.qwen1_5b.jsonl --matrix

Hai câu hỏi khác nhau:

  1. **Hai mô hình học có xếp hạng chuỗi giống nhau không** (đưa hai file vào). Đây là cách chốt câu hỏi
     Base hay Instruct: nếu thứ tự gần như giống nhau thì chọn bản nào cũng được, và nên theo bài gốc.
     Đo trên các cặp chuỗi CÙNG câu hỏi, vì đó đúng là việc mà b2_select làm.

  2. **Bốn tín hiệu có đang đo cùng một thứ không** (cờ --matrix). Phần Giới thiệu của paper khẳng định
     nhóm tín hiệu nội tại hội tụ về một đại lượng; đây là cách kiểm chứng khẳng định đó trên dữ liệu
     của chính nhóm. Cũng chính là Hình P4 trong kế hoạch hình vẽ.

RSR là cực tiểu, nên khi so hướng "càng phù hợp càng cao" thì phải đảo dấu. Công cụ tự làm việc đó.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl, split_tid
from src.stage_b.signals import prepare_fit_rows
from src.tools.compare_judges import pair_agreement, spearman, top1_agreement

# Tên tín hiệu và hướng: True nghĩa là càng cao càng phù hợp với mô hình học.
SIGNALS = {"rsr": False, "grape": True, "local_nat": True, "lark": True, "n_tokens": True}


def oriented(row: Mapping, name: str) -> float:
    """Giá trị đã đưa về cùng hướng càng cao càng phù hợp, để so sánh không bị ngược dấu."""
    v = float(row[name])
    return v if SIGNALS[name] else -v


def compare_two(a: Sequence[Mapping], b: Sequence[Mapping], signal: str) -> dict:
    ma = {r["tid"]: r for r in a}
    mb = {r["tid"]: r for r in b}
    shared = sorted(set(ma) & set(mb))
    if len(shared) < 3:
        raise SystemExit(f"Chỉ có {len(shared)} chuỗi được cả hai chấm, không đủ để so.")
    sa = [oriented(ma[t], signal) for t in shared]
    sb = [oriented(mb[t], signal) for t in shared]
    by_q: dict[str, list] = defaultdict(list)
    for t in shared:
        by_q[split_tid(t)[0]].append((oriented(ma[t], signal), oriented(mb[t], signal)))
    multi = {q: v for q, v in by_q.items() if len(v) >= 2}
    return {"n": len(shared), "questions": len(multi), "spearman": spearman(sa, sb),
            "pairs": pair_agreement(multi), "top1": top1_agreement(multi)}


def signal_matrix(rows: Sequence[Mapping], names: Sequence[str]) -> dict:
    """Tương quan hạng giữa các tín hiệu, tính TRONG TỪNG câu hỏi rồi lấy trung bình.

    Tính trên toàn bộ dữ liệu sẽ bị chi phối bởi khác biệt giữa các câu hỏi, chứ không phản ánh việc
    các tín hiệu có xếp hạng ứng viên của cùng một câu giống nhau hay không.
    """
    by_q: dict[str, list] = defaultdict(list)
    for r in rows:
        by_q[split_tid(r["tid"])[0]].append(r)
    out: dict[tuple, list] = defaultdict(list)
    for rs in by_q.values():
        if len(rs) < 3:
            continue
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                s = spearman([oriented(r, x) for r in rs], [oriented(r, y) for r in rs])
                if s is not None:
                    out[(x, y)].append(s)
    return {k: sum(v) / len(v) for k, v in out.items() if v}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir")
    ap.add_argument("file_a")
    ap.add_argument("file_b", nargs="?")
    ap.add_argument("--signal", default="rsr", choices=list(SIGNALS))
    ap.add_argument("--matrix", action="store_true", help="in ma trận tương quan giữa các tín hiệu")
    args = ap.parse_args(argv)

    cfg = load_config()
    wd = resolve_path(cfg, args.workdir)
    a, stale_a = prepare_fit_rows(read_jsonl(wd / args.file_a))
    if not a:
        raise SystemExit(f"Không đọc được {wd / args.file_a}")
    if stale_a:
        print(f"[chú ý] {args.file_a} tạo trước 29/09: bỏ {', '.join(stale_a)} vì định nghĩa cũ sai. "
              f"Chạy lại b1_fit để có đủ bốn tín hiệu.\n")

    if args.matrix or not args.file_b:
        names = [n for n in SIGNALS if n in a[0]]
        m = signal_matrix(a, names)
        print(f"[tín hiệu] tương quan hạng trung bình trong từng câu hỏi, {len(a)} chuỗi")
        print(f"   (mọi tín hiệu đã đưa về hướng càng cao càng phù hợp; RSR đã đảo dấu)\n")
        w = max(len(n) for n in names) + 2
        print(" " * w + "".join(f"{n:>13}" for n in names))
        for x in names:
            cells = []
            for y in names:
                if x == y:
                    cells.append(f"{'1.00':>13}")
                else:
                    v = m.get((x, y)) or m.get((y, x))
                    cells.append(f"{v:>13.2f}" if v is not None else f"{'—':>13}")
            print(f"{x:<{w}}" + "".join(cells))
        print("\n   Giá trị gần 1 nghĩa là hai tín hiệu xếp hạng ứng viên gần như giống nhau, tức chúng đang")
        print("   đo cùng một thứ. Đây là kiểm chứng cho luận điểm ở phần Giới thiệu của paper.")
        if not args.file_b:
            return 0

    b, _stale_b = prepare_fit_rows(read_jsonl(wd / args.file_b))
    if not b:
        raise SystemExit(f"Không đọc được {wd / args.file_b}")
    if args.signal not in a[0] or args.signal not in b[0]:
        raise SystemExit(f"Một trong hai file không có tín hiệu {args.signal} theo định nghĩa hiện hành.")
    r = compare_two(a, b, args.signal)
    ma_name = a[0].get("model", args.file_a)
    mb_name = b[0].get("model", args.file_b)
    print(f"\n[so hai mô hình học] tín hiệu {args.signal}, {r['n']} chuỗi cả hai cùng chấm")
    print(f"   A = {ma_name}\n   B = {mb_name}")
    print(f"\n   tương quan hạng toàn cục        : {r['spearman']:.3f}")
    p = r["pairs"]
    if p["rate"] is not None:
        print(f"   đồng thuận cặp cùng câu hỏi    : {p['same']}/{p['comparable']} = {p['rate']:.1%}"
              f"   ({r['questions']} câu có từ 2 chuỗi)")
    t = r["top1"]
    if t["rate"] is not None:
        print(f"   chọn cùng chuỗi phù hợp nhất   : {t['hit']}/{t['questions']} = {t['rate']:.1%}")
    print("\n   Đồng thuận cặp từ 85% trở lên: hai bản xếp hạng gần như giống nhau, chọn bản nào cũng được,")
    print("   nên theo bài gốc là Base. Dưới 70%: hai bản nhìn chuỗi khác nhau, phải chọn có cân nhắc và")
    print("   nêu rõ trong paper vì nó ảnh hưởng tới toàn bộ kết quả.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

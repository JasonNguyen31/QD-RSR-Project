"""So sánh hai lô sinh chuỗi và đo phân bố độ dài, để chốt prompt, max_tokens và max_seq_len.

    python -m src.tools.compare_runs data/pilot/run2_boxed data/pilot/run3_rsrprompt

Không gọi mạng, chỉ đọc trajectories.jsonl và labels.jsonl có sẵn.

Ba câu hỏi được trả lời cùng lúc:
  1. Hai prompt có cho tỷ lệ đúng khác nhau không, khi so TRÊN CÙNG TẬP CÂU HỎI (giao của hai lô).
     So thẳng số tổng của hai lô là sai, vì lô này có GSM8K còn lô kia không.
  2. Chuỗi bị cắt mất bao nhiêu, mô hình dạy nào bị nặng nhất, ở mức độ khó nào.
  3. Chuỗi ĐÚNG dài bao nhiêu token, để biết max_tokens và max_seq_len đặt bao nhiêu là đủ.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl


def pct(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(int(p * len(s)), len(s) - 1)]


def load_run(cfg, path_str: str) -> dict:
    wd = resolve_path(cfg, path_str)
    files = cfg["stage_a_files"]
    qs = {q["qid"]: q for q in read_jsonl(wd / files["questions"])}
    from src.stage_a.a2_generate import all_trajectory_files
    trajs = {t["tid"]: t for p in all_trajectory_files(wd, files) for t in read_jsonl(p)}
    labels = read_jsonl(wd / files["labels"])
    if not labels:
        raise SystemExit(f"{wd} chưa có labels.jsonl. Chạy a3_filter trước (có thể kèm --allow-incomplete).")
    return {"name": path_str, "questions": qs, "trajectories": trajs, "labels": labels}


def group_of(q: Mapping) -> str:
    return q["source"] if q.get("level") in (None, "") else f"{q['source']}-L{q['level']}"


def report_shared(a: dict, b: dict) -> None:
    shared = sorted(set(a["questions"]) & set(b["questions"]))
    print(f"\n{'='*78}\n1. SO TRÊN CÙNG TẬP CÂU HỎI: {len(shared)} câu có ở cả hai lô\n{'='*78}")
    if not shared:
        print("Hai lô không có câu hỏi chung, không so được.")
        return
    groups = sorted({group_of(a["questions"][q]) for q in shared})
    print(f"{'nhóm':<10}{'câu':>5} | {'lô A đúng':>10}{'cắt':>6} | {'lô B đúng':>10}{'cắt':>6} | {'chênh':>7}")
    for g in groups + ["TẤT CẢ"]:
        qids = [q for q in shared if g == "TẤT CẢ" or group_of(a["questions"][q]) == g]
        row = [f"{g:<10}{len(qids):>5} |"]
        accs = []
        for run in (a, b):
            lb = [l for l in run["labels"] if l["qid"] in set(qids)]
            if not lb:
                row.append(f"{'-':>10}{'-':>6} |")
                accs.append(None)
                continue
            acc = sum(l["correct"] for l in lb) / len(lb)
            cut = sum(l["reason"] == "truncated" for l in lb)
            accs.append(acc)
            row.append(f"{acc:>9.1%}{cut:>6} |")
        diff = f"{(accs[1] - accs[0]) * 100:+.1f}đ" if None not in accs else "-"
        print(" ".join(row) + f"{diff:>7}")
    print("\nChênh dưới 3 điểm phần trăm coi như hoà, vì mỗi nhóm chỉ vài trăm chuỗi.")


def report_truncation(run: dict) -> None:
    print(f"\n{'='*78}\n2. CHUỖI BỊ CẮT — lô {run['name']}\n{'='*78}")
    by: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "cut": 0, "ok": 0})
    for l in run["labels"]:
        g = group_of(run["questions"][l["qid"]])
        s = by[(l["teacher"], g)]
        s["n"] += 1
        s["cut"] += l["reason"] == "truncated"
        s["ok"] += bool(l["correct"])
    print(f"{'mô hình dạy':<12}{'nhóm':<10}{'chuỗi':>7}{'bị cắt':>8}{'tỷ lệ cắt':>11}{'đúng':>8}")
    for (t, g), s in sorted(by.items()):
        print(f"{t:<12}{g:<10}{s['n']:>7}{s['cut']:>8}{s['cut'] / s['n']:>11.1%}{s['ok'] / s['n']:>8.1%}")
    total = len(run["labels"])
    cut = sum(l["reason"] == "truncated" for l in run["labels"])
    print(f"\nTổng: {cut}/{total} = {cut / total:.1%} chuỗi bị cắt và bị tính là sai.")
    print("Bài gốc RSR lấy mẫu lại tối đa 10 lần nên tỷ lệ cắt cuối của họ dưới 1%.")


def report_lengths(run: dict, train_cap: int) -> None:
    print(f"\n{'='*78}\n3. ĐỘ DÀI CHUỖI ĐÚNG — lô {run['name']}\n{'='*78}")
    ok = {l["tid"] for l in run["labels"] if l["correct"]}
    by_t: dict[str, list[int]] = defaultdict(list)
    for tid in ok:
        t = run["trajectories"].get(tid)
        if t and t.get("completion_tokens"):
            by_t[t["teacher"]].append(int(t["completion_tokens"]))
    allv = [v for vs in by_t.values() for v in vs]
    if not allv:
        print("Không có dữ liệu độ dài.")
        return
    print(f"{'mô hình dạy':<12}{'chuỗi':>7}{'trung vị':>10}{'p75':>7}{'p90':>7}{'p99':>7}{'max':>7}"
          f"{'vượt ' + str(train_cap):>11}")
    for t, vs in sorted(by_t.items()):
        over = sum(v > train_cap for v in vs)
        print(f"{t:<12}{len(vs):>7}{pct(vs, .5):>10}{pct(vs, .75):>7}{pct(vs, .90):>7}{pct(vs, .99):>7}"
              f"{max(vs):>7}{over / len(vs):>10.1%}")
    over = sum(v > train_cap for v in allv)
    print(f"\nTất cả: {len(allv)} chuỗi đúng, trung vị {pct(allv, .5)}, p95 {pct(allv, .95)}, max {max(allv)}")
    if over:
        print(f"{over}/{len(allv)} = {over / len(allv):.1%} chuỗi đúng dài hơn {train_cap} token, "
              f"tức phải nâng max_seq_len hoặc chấp nhận bỏ nhóm chuỗi dài này khi huấn luyện.")
    else:
        print(f"Không có chuỗi đúng nào dài hơn {train_cap} token, nên giữ max_seq_len = {train_cap} "
              f"KHÔNG làm mất chuỗi nào đang có.")
    print(f"\nCẢNH BÁO VỀ CÁCH ĐỌC: phân bố này BỊ CẮT CỤT ở ngưỡng max_tokens lúc sinh. Mọi chuỗi muốn dài hơn")
    print(f"ngưỡng đó đã bị cắt và bị tính là sai, nên không nằm trong bảng. Vì vậy 'không có chuỗi nào vượt")
    print(f"{train_cap}' là hệ quả tất yếu của cách sinh, KHÔNG phải bằng chứng rằng {train_cap} là đủ.")
    print("Muốn biết ngưỡng bao nhiêu mới đủ thì phải sinh lại một lô nhỏ với max_tokens cao hơn rồi đo lại.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", help="lô A, ví dụ data/pilot/run2_boxed")
    ap.add_argument("run_b", nargs="?", help="lô B, ví dụ data/pilot/run3_rsrprompt")
    ap.add_argument("--train-cap", type=int, help="ngưỡng huấn luyện, mặc định training.max_seq_len")
    args = ap.parse_args(argv)

    cfg = load_config()
    cap = args.train_cap or cfg["training"]["max_seq_len"]
    a = load_run(cfg, args.run_a)
    print(f"Lô A = {args.run_a}: {len(a['questions'])} câu, {len(a['labels'])} chuỗi")
    if args.run_b:
        b = load_run(cfg, args.run_b)
        print(f"Lô B = {args.run_b}: {len(b['questions'])} câu, {len(b['labels'])} chuỗi")
        report_shared(a, b)
        for run in (a, b):
            report_truncation(run)
            report_lengths(run, cap)
    else:
        report_truncation(a)
        report_lengths(a, cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

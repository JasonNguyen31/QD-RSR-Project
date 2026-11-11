"""Số liệu cho phản biện bài báo ngày 07/11 (mục B1.3a và B1.3d). Chỉ đọc dữ liệu, không ghi file nào.

    python scripts/review_stats.py --workdir data/stage_a --student qwen1_5b

In ba bảng:
  1. Điểm giám khảo (nửa judge của Qual) theo mô hình dạy, trên kho 1.895 câu: trung bình, tỷ lệ đạt điểm tối đa.
  2. Bước loại của FQ-Select v2 (fq_veto_judge): mỗi mô hình dạy chiếm bao nhiêu trong kho, trong tập V_q bị loại,
     và trong tập được chọn. Dòng cuối so tỷ lệ Qwen của tập chọn với file select.fq_veto_judge.k3.json để chắc
     phép dựng lại khớp với lần chọn thật.
  3. Nhãn chiến lược của 180 cặp (embed_cmp_labels.jsonl của src.tools.compare_embedding), tách cặp cùng mô hình dạy
     và cặp khác mô hình dạy.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, path_of, resolve_path  # noqa: E402
from src.common.io_utils import read_jsonl  # noqa: E402
from src.stage_b import b2_select as b2  # noqa: E402


def dropped_set(primary, veto, k, drop, primary_dir, veto_dir):
    """Dựng lại tập bị loại đúng như b2.pick_veto_topk, và kiểm tra tập chọn trùng với hàm gốc."""
    v = np.asarray(veto, dtype=float)
    sv = v if veto_dir == "max" else -v
    n_drop = b2.veto_count(len(v), k, drop)
    worst_first = sorted(range(len(v)), key=lambda i: (sv[i], -i))
    dropped = set(worst_first[:n_drop])
    keep = [i for i in range(len(v)) if i not in dropped]
    p = b2.pick_topk(np.asarray(primary, dtype=float)[keep], k, primary_dir)
    chosen = sorted(keep[j] for j in p["idx"])
    ref = b2.pick_veto_topk(primary, veto, k, drop, primary_dir, veto_dir)["idx"]
    if chosen != ref:
        raise SystemExit("Dựng lại bước loại không khớp b2.pick_veto_topk; dừng.")
    return dropped, chosen


def pct(c: Counter, t: str) -> str:
    n = sum(c.values())
    return f"{100 * c[t] / n:5.1f}%" if n else "   - "


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--student", default="qwen1_5b")
    args = ap.parse_args(argv)

    cfg = load_config(method="fq_veto_judge", student=args.student)
    wd = resolve_path(cfg, args.workdir)
    inp = b2.Inputs(cfg, wd, f"fit.{b2.student_tag(cfg)}.jsonl")
    qs = inp.questions("correct", "veto_topk")
    k = int(cfg["selection"]["k"])
    teachers = sorted({t for q in qs.values() for t in q["teacher"]})
    print(f"{len(qs)} câu, {sum(len(q['tids']) for q in qs.values())} lời giải đúng\n")

    # 1. điểm giám khảo theo mô hình dạy
    scores = defaultdict(list)
    for q in qs.values():
        for t, s in zip(q["teacher"], b2.signal_values(q, "llm_quality")):
            scores[t].append(float(s))
    top = max(max(v) for v in scores.values())
    print(f"1. Điểm giám khảo theo mô hình dạy (điểm tối đa quan sát được: {top:g})")
    print(f"   {'mô hình dạy':<12}{'số lời giải':>12}{'trung bình':>12}{'đạt tối đa':>12}")
    for t in teachers:
        v = np.array(scores[t])
        print(f"   {t:<12}{len(v):>12}{v.mean():>12.3f}{100 * (v == top).mean():>11.1f}%")
    allv = np.concatenate([np.array(v) for v in scores.values()])
    print(f"   {'tất cả':<12}{len(allv):>12}{allv.mean():>12.3f}{100 * (allv == top).mean():>11.1f}%\n")

    # 2. bước loại của v2
    spec = cfg["select"]
    drop = float(spec["veto_drop"])
    pdir, vdir = b2.direction_of(cfg, spec["signal"]), b2.direction_of(cfg, spec["veto_signal"])
    pool, removed, chosen = Counter(), Counter(), Counter()
    n_with_removal = 0
    for q in qs.values():
        d, c = dropped_set(b2.signal_values(q, spec["signal"]), b2.signal_values(q, spec["veto_signal"]),
                           k, drop, pdir, vdir)
        pool.update(q["teacher"])
        removed.update(q["teacher"][i] for i in d)
        chosen.update(q["teacher"][i] for i in c)
        n_with_removal += bool(d)
    print(f"2. FQ-Select v2: tỷ lệ mô hình dạy (V_q gồm {sum(removed.values())} lời giải bị loại, ở "
          f"{n_with_removal}/{len(qs)} câu có loại)")
    print(f"   {'mô hình dạy':<12}{'trong kho':>12}{'trong V_q':>12}{'được chọn':>12}")
    for t in teachers:
        print(f"   {t:<12}{pct(pool, t):>12}{pct(removed, t):>12}{pct(chosen, t):>12}")
    sel = path_of(cfg, "data_stage_b") / b2.student_tag(cfg) / "select.fq_veto_judge.k3.json"
    if sel.exists():
        print(f"   đối chiếu: tỷ lệ mô hình dạy trong {sel.name}: {json.loads(sel.read_text())['teacher_share']}")
    print()

    # 4. B2.6: còn bao nhiêu chỗ để chọn trong một mô hình dạy (thêm 10/11)
    nt = Counter()
    forced, slots, free_slots = 0, 0, 0
    for q in qs.values():
        cnt = Counter(q["teacher"])
        for t in teachers:
            nt[cnt.get(t, 0)] += 1
        quota = Counter(q["teacher"][:k])                 # số chỗ của mỗi mô hình dạy trong lần bốc Correct-Only
        forced += all(cnt[t] == c for t, c in quota.items())
        slots += sum(quota.values())
        free_slots += sum(c for t, c in quota.items() if cnt[t] > c)
    n_pairs = len(qs) * len(teachers)
    print("4. Số lời giải đúng n_t của mỗi mô hình dạy trong một câu (B2.6)")
    print("   " + "   ".join(f"n_t = {v}: {100 * nt[v] / n_pairs:.1f}%" for v in sorted(nt)))
    print(f"   Câu mà FWT/QWT buộc phải trùng Correct-Only (mỗi mô hình dạy có đúng bằng số chỗ): "
          f"{forced}/{len(qs)} ({100 * forced / len(qs):.1f}%)")
    print(f"   Chỗ còn quyền chọn (mô hình dạy có nhiều lời giải hơn số chỗ): {free_slots}/{slots} "
          f"({100 * free_slots / slots:.1f}%)")
    sdir = path_of(cfg, "data_stage_b") / b2.student_tag(cfg)
    base = sdir / "train.correct_only.k3.jsonl"
    for name in ("familiarity_within_teacher", "quality_within_teacher"):
        other = sdir / f"train.{name}.k3.jsonl"
        if base.exists() and other.exists():
            o = b2.overlap(b2._tid_sets(other), b2._tid_sets(base))
            print(f"   {name} so với Correct-Only: {100 * o['sample_share']:.1f}% mẫu chung, "
                  f"{100 * o['same_set']:.1f}% câu chọn đúng cùng tập")
    print()

    # 3. nhãn chiến lược theo cặp
    path = wd / "embed_cmp_labels.jsonl"
    if not path.exists():
        print(f"3. Không có {path}; bỏ qua bảng nhãn chiến lược.")
        return 0
    from src.tools.compare_embedding import parse_verdict
    teacher_of = {c["tid"]: c["teacher"] for c in inp.cands}
    teacher_of.update({r["tid"]: r["teacher"] for r in read_jsonl(wd / cfg["stage_a_files"]["candidates"])})
    tab = defaultdict(Counter)
    model = Counter()
    for r in read_jsonl(path):
        a, b = r["key"].split("__")
        ta, tb = teacher_of.get(a), teacher_of.get(b)
        kind = "?" if ta is None or tb is None else ("cùng mô hình dạy" if ta == tb else "khác mô hình dạy")
        tab[kind][parse_verdict(r["text"]) or "không đọc được"] += 1
        model[r.get("model")] += 1
    print(f"3. Nhãn chiến lược theo cặp (mô hình gán nhãn: {dict(model)})")
    print(f"   {'loại cặp':<18}{'số cặp':>8}{'khác chiến lược':>18}{'không đọc được':>16}")
    for kind in sorted(tab):
        c = tab[kind]
        n = sum(c.values())
        print(f"   {kind:<18}{n:>8}{c['different']:>10} ({100 * c['different'] / n:4.1f}%){c['không đọc được']:>12}")
    total = sum(tab.values(), Counter())
    n = sum(total.values())
    print(f"   {'tất cả':<18}{n:>8}{total['different']:>10} ({100 * total['different'] / n:4.1f}%)"
          f"{total['không đọc được']:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

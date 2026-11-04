"""teacher_pilot: cần bao nhiêu câu hỏi để chọn đúng mô hình dạy theo Fam trung bình? (Hướng 2, thêm 04/11/2026)

    python -m src.tools.teacher_pilot                                  # mặc định: Fam của Qwen2.5-1.5B, chuỗi đúng
    python -m src.tools.teacher_pilot --sizes 5 10 20 40 80 160 --chains 1 3
    python -m src.tools.teacher_pilot --include-wrong                  # thêm chuỗi sai (điều kiện thật của một lượt thử)
    python -m src.tools.teacher_pilot --stratify                       # chia mẫu theo độ khó (level) của câu hỏi

Đọc  data/stage_a/{candidates,questions,quality,fit.<student>_base}.jsonl (và *.wrong.jsonl nếu có cờ). Ghi
     outputs/results/teacher_pilot/<student>/teacher_pilot.json và teacher_pilot.md. Không GPU, không torch, không mạng.

Câu hỏi. Quy tắc "mô hình dạy nào có Fam trung bình cao nhất" chọn Qwen2.5-72B trên cả kho (1.895 câu). Một người dùng
mới không có kho đó: họ chỉ nhờ mỗi mô hình dạy sinh `s` chuỗi cho `m` câu hỏi thử, chấm Fam, lấy trung bình. Cần m bằng
bao nhiêu để quy tắc này chọn ĐÚNG mô hình mà cả kho chọn?

Cách đo. Với mỗi câu và mỗi mô hình dạy, Fam của câu = trung bình Fam của các chuỗi có sample_idx < s. Chỉ giữ câu mà MỌI
mô hình dạy đều có ít nhất một chuỗi (complete-case), để so sánh giữa các mô hình dạy dùng cùng một tập câu. Mô hình đích
= mô hình có trung bình cao nhất trên toàn bộ các câu đó. Rồi lặp R lần: rút m câu không hoàn lại, lấy mô hình có trung
bình mẫu cao nhất, đếm tỷ lệ trùng mô hình đích. m* = m nhỏ nhất đạt P >= 0,95 (và 0,99).

Giới hạn cần nhớ khi đọc. (1) Đây là xác suất chọn lại đúng kết quả của CHÍNH kho này, mẫu rút từ kho nên chưa phải từ một
tập câu hỏi mới; với kho 1.895 câu, m lớn gần Q thì P tiến về 1 vì mẫu trùng gần hết kho (không cần coi là bằng chứng).
Đọc các m nhỏ hơn khoảng một phần năm kho. (2) Fam trung bình chỉ phân định được mô hình dạy, không nói gì về việc
mô hình đó có THẮNG ở MATH-500 hay không; câu đó đã trả lời bằng Prefer-Teacher. (3) Chuỗi đúng ở đây đã qua bước lọc
đáp án; một lượt thử thật chưa biết đáp án nên nên đọc thêm kết quả với --include-wrong. (4) Khi hai mô hình hòa
điểm tuyệt đối, np.argmax lấy mô hình đứng trước trong danh sách; với Fam là số thực, hòa gần như không xảy ra.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.common.config import load_config, path_of, resolve_path
from src.common.io_utils import read_jsonl
from src.stage_b import b2_select as b2

SIGNAL = "grape"                   # cột Fam trong file fit
P_LEVELS = (0.95, 0.99)


# ============================================================ phần không cần dữ liệu thật
def split_tid(tid: str) -> tuple[str, str, int]:
    """'qid|teacher|sample' -> (qid, teacher, sample). qid có thể chứa '|' nên tách từ bên phải."""
    qid, teacher, s = tid.rsplit("|", 2)
    return qid, teacher, int(s)


def question_means(fam: Mapping[str, float], tids: Sequence[str], teachers: Sequence[str], s: int):
    """Ma trận (số câu, số mô hình dạy) trung bình Fam trên các chuỗi có sample_idx < s, chỉ giữ câu đủ mọi mô hình dạy.

    Trả về (qids theo thứ tự, ma trận). Chuỗi không có điểm Fam trong `fam` bị bỏ qua."""
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for t in tids:
        if t not in fam:
            continue
        qid, teacher, idx = split_tid(t)
        if idx < s and teacher in teachers:
            acc[qid][teacher].append(float(fam[t]))
    qids = sorted(q for q, by in acc.items() if all(by.get(tc) for tc in teachers))
    mat = np.array([[np.mean(acc[q][tc]) for tc in teachers] for q in qids], dtype=float)
    return qids, mat.reshape(len(qids), len(teachers))


def draw_samples(n: int, m: int, reps: int, rng: np.random.Generator, strata: np.ndarray | None = None) -> np.ndarray:
    """reps hàng, mỗi hàng m chỉ số câu rút không hoàn lại từ n câu. strata: nhãn nhóm của từng câu; nếu có thì chia
    phần m theo tỷ lệ nhóm (làm tròn lớn nhất, bù cho đủ m)."""
    if not 1 <= m <= n:
        raise ValueError(f"m={m} ngoài khoảng [1, {n}]")
    if strata is None:
        return np.argpartition(rng.random((reps, n)), m - 1, axis=1)[:, :m]
    labels, counts = np.unique(strata, return_counts=True)
    quota = np.floor(m * counts / n).astype(int)
    for i in np.argsort(-(m * counts / n - quota))[: m - quota.sum()]:
        quota[i] += 1
    quota = np.minimum(quota, counts)
    out = []
    for lab, q in zip(labels, quota):
        idx = np.flatnonzero(strata == lab)
        if q:
            out.append(idx[np.argpartition(rng.random((reps, len(idx))), q - 1, axis=1)[:, :q]] if q < len(idx)
                       else np.tile(idx, (reps, 1)))
    return np.concatenate(out, axis=1)


def pick_rate(mat: np.ndarray, target: int, m: int, reps: int, rng: np.random.Generator,
              strata: np.ndarray | None = None, chunk: int = 1000) -> dict:
    """Tỷ lệ lượt thử chọn đúng mô hình `target`, cùng phân bố số lần chọn mỗi mô hình."""
    wins = np.zeros(mat.shape[1], dtype=int)
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        idx = draw_samples(len(mat), m, r, rng, strata)
        picked = mat[idx].mean(axis=1).argmax(axis=1)             # (r, m, T) -> (r, T) -> (r,)
        wins += np.bincount(picked, minlength=mat.shape[1])
        done += r
    return {"p": float(wins[target] / reps), "wins": wins.tolist()}


def smallest_m(rows: Sequence[Mapping], level: float) -> int | None:
    """m nhỏ nhất mà từ đó trở đi (mọi m lớn hơn trong lưới) P >= level. None nếu không có."""
    ok = None
    for r in sorted(rows, key=lambda r: -r["m"]):
        if r["p"] >= level:
            ok = r["m"]
        else:
            break
    return ok


def analytic_m(diff: np.ndarray, z: float = 1.645) -> float | None:
    """Số câu ước theo xấp xỉ chuẩn để hiệu trung bình (đích trừ đối thủ) dương với xác suất ~95% một phía:
    m ≈ (z · sd / mean)^2. Chỉ để đối chiếu thô với mô phỏng."""
    mean, sd = float(diff.mean()), float(diff.std(ddof=1))
    return None if mean <= 0 else (z * sd / mean) ** 2


# ============================================================ đọc kho
def load_fam(work: Path, fit_name: str, with_wrong: bool) -> tuple[dict, list]:
    fam, tids = {}, []
    for r in read_jsonl(work / fit_name):
        fam[r["tid"]] = float(r[SIGNAL])
    tids += list(fam)
    if with_wrong:
        wrong = work / fit_name.replace(".jsonl", ".wrong.jsonl")
        if not wrong.exists():
            raise SystemExit(f"Cờ --include-wrong cần {wrong.name}, máy này không có.")
        w = {r["tid"]: float(r[SIGNAL]) for r in read_jsonl(wrong)}
        fam.update(w)
        tids += list(w)
    return fam, tids


def run(cfg: Mapping, work: Path, sizes: Sequence[int], chains: Sequence[int], reps: int, seed: int,
        include_wrong: bool, stratify: bool) -> dict:
    files = cfg["stage_a_files"]
    cands = read_jsonl(work / files["candidates"])
    quality = {r["tid"]: r for r in read_jsonl(work / "quality.jsonl")}
    pool, _info = b2.usable_pool(cands, quality, int(cfg["data"]["min_correct_per_question"]))
    kept = set(pool)
    fit_name = f"fit.{b2.student_tag(cfg)}.jsonl"
    fam, all_tids = load_fam(work, fit_name, include_wrong)
    teachers = [t["key"] for t in cfg["teachers"]]
    usable = {c["tid"] for q in pool.values() for c in q}
    tids = [t for t in all_tids if split_tid(t)[0] in kept and (include_wrong or t in usable)]
    level_of = {q["qid"]: q.get("level") for q in read_jsonl(work / files["questions"])}
    rng = np.random.default_rng(seed)
    out = {"student": cfg["student"]["key"], "include_wrong": include_wrong, "stratify": stratify, "reps": reps,
           "seed": seed, "teachers": teachers, "settings": []}
    for s in chains:
        qids, mat = question_means(fam, tids, teachers, s)
        means = mat.mean(axis=0)
        target = int(means.argmax())
        strata = np.array([str(level_of.get(q)) for q in qids]) if stratify else None
        rows = []
        for m in sizes:
            if m <= len(qids):
                rows.append({"m": m, **pick_rate(mat, target, m, reps, rng, strata)})
        diffs = {teachers[j]: analytic_m(mat[:, target] - mat[:, j]) for j in range(len(teachers)) if j != target}
        out["settings"].append({
            "chains_per_question": s, "n_questions": len(qids), "full_means": dict(zip(teachers, means.tolist())),
            "target": teachers[target], "rows": rows,
            "m_star": {str(p): smallest_m(rows, p) for p in P_LEVELS},
            "analytic_m_vs": diffs, "sd_of_means": dict(zip(teachers, mat.std(axis=0, ddof=1).tolist()))})
    return out


def markdown(res: Mapping, q_total: int) -> str:
    lines = [f"Mô hình học {res['student']}, {'cả chuỗi sai' if res['include_wrong'] else 'chuỗi đúng'}, "
             f"{res['reps']:,} lượt rút mỗi ô, {'chia theo level' if res['stratify'] else 'rút ngẫu nhiên đơn giản'}.", ""]
    for st in res["settings"]:
        s, n = st["chains_per_question"], st["n_questions"]
        means = ", ".join(f"{t} {v:.4f}" for t, v in st["full_means"].items())
        lines += [f"**s = {s} chuỗi mỗi mô hình dạy mỗi câu** | {n} câu đủ mọi mô hình dạy | trung bình cả kho: {means} | "
                  f"đích: {st['target']}", "",
                  "| m câu | chuỗi phải sinh (T·m·s) | so với kho (T·Q·3) | P(chọn đúng) | số lần chọn mỗi mô hình |",
                  "| ---: | ---: | ---: | ---: | --- |"]
        T = len(res["teachers"])
        for r in st["rows"]:
            cost = T * r["m"] * s
            lines.append(f"| {r['m']} | {cost} | {cost / (T * q_total * 3):.1%} | {r['p']:.4f} | {r['wins']} |")
        ms = st["m_star"]
        lines += ["", f"m* (P ≥ 0,95) = {ms['0.95']}; m* (P ≥ 0,99) = {ms['0.99']}. "
                      f"Xấp xỉ chuẩn (95% một phía) so với từng đối thủ: "
                      + ", ".join(f"{k} {v:.0f}" if v else f"{k} n/a" for k, v in st["analytic_m_vs"].items()), ""]
    lines.append(f"Thứ tự cột 'số lần chọn': {res['teachers']}.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", default="qwen1_5b")
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160, 320])
    ap.add_argument("--chains", type=int, nargs="+", default=[1, 3], help="s: số chuỗi mỗi mô hình dạy mỗi câu hỏi thử")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-wrong", action="store_true")
    ap.add_argument("--stratify", action="store_true")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(student=args.student, overrides=args.override)
    work = resolve_path(cfg, args.workdir) if args.workdir else path_of(cfg, "data_stage_a")
    res = run(cfg, work, sorted(set(args.sizes)), sorted(set(args.chains)), args.reps, args.seed,
              args.include_wrong, args.stratify)
    q_total = max(st["n_questions"] for st in res["settings"])
    outdir = resolve_path(cfg, args.outdir) if args.outdir else path_of(cfg, "out_results") / "teacher_pilot" / args.student
    outdir.mkdir(parents=True, exist_ok=True)
    tag = ("wrong" if args.include_wrong else "correct") + ("_strat" if args.stratify else "")
    (outdir / f"teacher_pilot.{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    md = markdown(res, q_total)
    (outdir / f"teacher_pilot.{tag}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

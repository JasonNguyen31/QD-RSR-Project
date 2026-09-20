"""a3_filter: gán nhãn đúng/sai, lọc sơ bộ, thống kê (Giai đoạn A, bước [2]).

    python -m src.stage_a.a3_filter --workdir data/pilot/run2_boxed
    python -m src.stage_a.a3_filter --workdir data/stage_a

Đọc  <workdir>/questions.jsonl, <workdir>/trajectories.jsonl
Ghi  <workdir>/labels.jsonl      MỌI chuỗi (không kèm văn bản): tid, đúng/sai, đáp án tách được, lý do
     <workdir>/candidates.jsonl  chuỗi ĐÚNG của các câu hỏi được giữ lại (kèm văn bản). Đây là đầu vào của a4, a5, b1.
                                 Là file dẫn xuất, xoá đi chạy lại a3 là có lại.
     <workdir>/kept_qids.json    danh sách câu hỏi được giữ (có ít nhất min_correct chuỗi đúng), cố định cho mọi phương án
     <workdir>/stats.json        thống kê, cũng là số liệu cho Hình 2 của paper

No-Filter chọn từ mọi chuỗi của các câu hỏi được giữ, nên cần đọc thêm trajectories.jsonl (nối qua tid).

Quy ước chấm: chỉ \\boxed{} cuối cùng; chuỗi bị cắt (finish_reason=length) luôn tính là sai dù có \\boxed,
vì chuỗi cụt không thể làm mẫu huấn luyện hoàn chỉnh.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from src.common.answers import grade
from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl, write_json, write_jsonl


def group_of(q: Mapping) -> str:
    return q["source"] if q.get("level") in (None, "") else f"{q['source']}-L{q['level']}"


def label_trajectories(questions: Sequence[Mapping], trajectories: Sequence[Mapping]) -> list[dict]:
    qmap = {q["qid"]: q for q in questions}
    unknown = {t["qid"] for t in trajectories} - set(qmap)
    if unknown:
        raise SystemExit(f"{len(unknown)} qid trong trajectories.jsonl không có trong questions.jsonl, "
                         f"ví dụ {sorted(unknown)[:3]}. Sai workdir?")
    labels = []
    for t in trajectories:
        q = qmap[t["qid"]]
        truncated = t.get("finish_reason") == "length"
        g = grade(t["text"], q["gold"])
        labels.append({
            "tid": t["tid"], "qid": t["qid"], "teacher": t["teacher"], "sample_idx": t["sample_idx"],
            "correct": bool(g.correct and not truncated),
            "reason": "truncated" if truncated else g.reason,
            "pred": g.pred, "completion_tokens": t.get("completion_tokens"),
        })
    return labels


def compute_stats(questions: Sequence[Mapping], labels: Sequence[Mapping], expected_chains: int,
                  min_correct: int) -> tuple[dict, list[str], list[str]]:
    qmap = {q["qid"]: q for q in questions}
    by_q: dict[str, list[Mapping]] = defaultdict(list)
    for lb in labels:
        by_q[lb["qid"]].append(lb)

    incomplete = sorted(qid for qid in qmap if len(by_q.get(qid, [])) != expected_chains)
    n_correct = {qid: sum(lb["correct"] for lb in by_q.get(qid, [])) for qid in qmap}
    kept = sorted(qid for qid in qmap if n_correct[qid] >= min_correct)
    kept_set = set(kept)

    groups = sorted({group_of(q) for q in questions})
    per_group: dict[str, dict] = {}
    for g in groups:
        qids = [qid for qid in qmap if group_of(qmap[qid]) == g]
        chains = [lb for qid in qids for lb in by_q.get(qid, [])]
        hist = Counter(n_correct[qid] for qid in qids)
        per_group[g] = {
            "questions": len(qids),
            "chains": len(chains),
            "accuracy": round(sum(lb["correct"] for lb in chains) / len(chains), 4) if chains else None,
            "kept": sum(1 for qid in qids if qid in kept_set),
            "dropped": sum(1 for qid in qids if n_correct[qid] < min_correct),
            "correct_count_hist": {str(i): hist.get(i, 0) for i in range(expected_chains + 1)},
        }

    per_teacher: dict[str, dict] = {}
    for teacher in sorted({lb["teacher"] for lb in labels}):
        chains = [lb for lb in labels if lb["teacher"] == teacher]
        toks = [lb["completion_tokens"] for lb in chains if lb.get("completion_tokens")]
        per_teacher[teacher] = {
            "chains": len(chains),
            "accuracy": round(sum(lb["correct"] for lb in chains) / len(chains), 4),
            "mean_tokens": round(statistics.mean(toks), 1) if toks else None,
        }

    reasons = Counter(lb["reason"] for lb in labels)
    total = len(labels)
    stats = {
        "questions": len(qmap), "chains": total, "expected_chains_per_question": expected_chains,
        "incomplete_questions": len(incomplete),
        "accuracy": round(sum(lb["correct"] for lb in labels) / total, 4) if total else None,
        "reasons": dict(reasons),
        "min_correct": min_correct, "kept_questions": len(kept), "dropped_questions": len(qmap) - len(kept),
        "per_group": per_group, "per_teacher": per_teacher,
    }
    return stats, kept, incomplete


def print_stats(stats: Mapping) -> None:
    print(f"\n[a3] {stats['questions']} câu hỏi, {stats['chains']} chuỗi, tỷ lệ đúng chung {stats['accuracy']:.1%}")
    print(f"[a3] lý do: {stats['reasons']}")
    print(f"[a3] giữ {stats['kept_questions']} câu (>= {stats['min_correct']} chuỗi đúng), "
          f"loại {stats['dropped_questions']} câu")
    print(f"\n{'mô hình dạy':<12}{'chuỗi':>8}{'đúng':>9}{'token TB':>10}")
    for k, v in stats["per_teacher"].items():
        print(f"{k:<12}{v['chains']:>8}{v['accuracy']:>9.1%}{v['mean_tokens'] or 0:>10.0f}")
    print(f"\n{'nhóm':<10}{'câu':>6}{'đúng':>9}{'giữ':>6}{'loại':>6}   số chuỗi đúng mỗi câu (0..N)")
    for g, v in stats["per_group"].items():
        hist = " ".join(f"{n:>3}" for n in v["correct_count_hist"].values())
        acc = f"{v['accuracy']:.1%}" if v["accuracy"] is not None else "-"
        print(f"{g:<10}{v['questions']:>6}{acc:>9}{v['kept']:>6}{v['dropped']:>6}   {hist}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", help="thư mục làm việc, mặc định paths.data_stage_a")
    ap.add_argument("--min-correct", type=int, help="mặc định data.min_correct_per_question")
    ap.add_argument("--expected-chains", type=int,
                    help="số chuỗi mong đợi mỗi câu, mặc định số mô hình dạy x samples_per_teacher")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="vẫn chạy khi có câu hỏi thiếu chuỗi (mặc định dừng để sinh bù bằng a2 trước)")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.override)
    files = cfg["stage_a_files"]
    workdir: Path = resolve_path(cfg, args.workdir or cfg["paths"]["data_stage_a"])
    min_correct = args.min_correct or cfg["data"]["min_correct_per_question"]
    expected = args.expected_chains or len(cfg["teachers"]) * cfg["generation"]["samples_per_teacher"]

    questions = read_jsonl(workdir / files["questions"])
    from src.stage_a.a2_generate import all_trajectory_files
    traj_files = all_trajectory_files(workdir, files)
    trajectories = [t for p in traj_files for t in read_jsonl(p)]
    if len(traj_files) > 1:
        print(f"[a3] đọc gộp {len(traj_files)} file chuỗi: {', '.join(p.name for p in traj_files)}")
    if not questions or not trajectories:
        raise SystemExit(f"Thiếu dữ liệu trong {workdir} (questions={len(questions)}, trajectories={len(trajectories)}).")
    tids = [t["tid"] for t in trajectories]
    if len(tids) != len(set(tids)):
        from collections import Counter as _C
        dup = [t for t, n in _C(tids).items() if n > 1][:3]
        raise SystemExit(f"Có tid trùng nhau giữa các file chuỗi, ví dụ {dup}. Nếu chạy song song nhiều lô thì "
                         f"mỗi lô phải nhận các mô hình dạy KHÁC nhau qua --teachers.")

    labels = label_trajectories(questions, trajectories)
    stats, kept, incomplete = compute_stats(questions, labels, expected, min_correct)
    print_stats(stats)

    if incomplete and not args.allow_incomplete:
        raise SystemExit(f"\n[a3] DỪNG: {len(incomplete)} câu hỏi không đủ {expected} chuỗi (ví dụ {incomplete[:3]}). "
                         f"Chạy lại a2_generate để sinh bù (xem failures.jsonl), hoặc dùng --allow-incomplete.")

    kept_set = set(kept)
    correct = {lb["tid"] for lb in labels if lb["correct"]}
    candidates = [t for t in trajectories if t["tid"] in correct and t["qid"] in kept_set]

    write_jsonl(workdir / files["labels"], labels)
    write_jsonl(workdir / files["candidates"], candidates)
    write_json(workdir / files["kept_qids"], {"min_correct": min_correct, "qids": kept})
    write_json(workdir / files["stats"], stats)
    print(f"\n[a3] đã ghi: {files['labels']} ({len(labels)}), {files['candidates']} ({len(candidates)}), "
          f"{files['kept_qids']} ({len(kept)}), {files['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

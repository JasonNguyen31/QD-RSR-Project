"""backfill_wrong: chấm bù các chuỗi SAI để biến thể no_prefilter (QD-RSR không lọc đáp án) chạy được.

a4, a5 và b1 chỉ chấm candidates.jsonl, tức chuỗi đúng. Biến thể không lọc cần Qual, biểu diễn và tín hiệu của
mô hình học cho cả chuỗi sai của các câu hỏi được giữ. Công cụ này chấm đúng phần còn thiếu, bằng chính mã,
mô hình, lời nhắc và cài đặt của lần chấm trước (run_judge của a4, encode của a5, b1_fit), rồi ghi ra FILE
RIÊNG. Nó không sửa quality.jsonl, judge.jsonl, embeddings.npz hay file fit hiện có.

    # máy Mac (có trajectories*.jsonl và khoá API)
    python -m src.tools.backfill_wrong --step prepare --prev-cost-usd 9.4     # không gọi mạng, in ước tính chi phí
    python -m src.tools.backfill_wrong --step judge --max-cost 3              # giám khảo, chạy bù được
    python -m src.tools.backfill_wrong --step embed                           # BGE-M3
    # chép candidates.wrong.jsonl sang Windows (so md5), rồi trên máy Windows
    python -m src.tools.backfill_wrong --step fit                             # b1_fit bản nền, 16-bit, bỏ LocalNat
    # chép fit.qwen1_5b_base.wrong.jsonl về Mac (so md5), rồi
    python -m src.tools.backfill_wrong --step combine                         # ghi quality.all.jsonl, kiểm đủ ba nguồn
    python -m src.tools.backfill_wrong --step status                          # tiến độ từng file, lúc nào cũng chạy được

File ghi vào <workdir> (tên lấy từ b2_select.extended_files):
    candidates.wrong.jsonl           chuỗi sai của các câu được giữ, kèm văn bản (đầu vào của ba bước chấm)
    judge.wrong.jsonl                điểm giám khảo của chuỗi sai
    embeddings.wrong.npz             biểu diễn của chuỗi sai
    fit.<student>_base.wrong.jsonl   tín hiệu của mô hình học cho chuỗi sai, không có LocalNat
    quality.all.jsonl                Qual của kho mở rộng "all": mọi chuỗi đúng dùng được cộng mọi chuỗi sai
    quality.all_complete.jsonl       Qual của kho "all_complete": như trên nhưng bỏ chuỗi bị cắt

Vì sao Qual phải là file riêng cho từng kho mở rộng chứ không chỉ thêm dòng cho chuỗi sai: cả hai nửa của Qual
được min-max trong từng câu hỏi, nên thêm chuỗi sai vào một câu làm đổi Qual của cả các chuỗi đúng trong câu
đó. Điểm giám khảo thô, biểu diễn và tín hiệu của mô hình học thì tính riêng cho từng chuỗi, nên chỉ cần chấm
thêm phần chuỗi sai.

Điểm quy tắc của chuỗi sai được chấm trên THANG CỦA KHO GỐC (rule_scores_with_reference, tham chiếu là
candidates.jsonl), không chuẩn hoá lại trên kho mở rộng. Bản đầu (02/10) chuẩn hoá lại trên cả kho mở rộng,
làm trung bình và độ lệch chuẩn đổi, kéo theo điểm quy tắc của mọi chuỗi đúng đổi: trên dữ liệu thật, 97 trong
1.236 câu KHÔNG có chuỗi sai nào vẫn bị đổi tập chọn. Với cách hiện tại, câu không có chuỗi sai có Qual giống
hệt quality.jsonl, và bước combine dừng nếu không đúng như vậy.

Chuỗi sai là gì: mọi chuỗi của các câu hỏi dùng được mà không nằm trong candidates.jsonl (sai đáp án, bị cắt,
thiếu \\boxed), trừ chuỗi rỗng. Cùng một hàm (b2_select.all_chains_pool) định nghĩa kho này cho b2_select,
nên hai nơi không thể lệch nhau.

Ước tính chi phí: judge.jsonl không ghi chi phí từng lượt, nên phải đưa tổng chi phí của lô giám khảo trước
(lịch sử OpenRouter) qua --prev-cost-usd. Chi phí mỗi chuỗi = tổng đó chia số chuỗi đã chấm trong judge.jsonl.
Chuỗi sai dài hơn chuỗi đúng, nên công cụ in thêm ước tính đã nhân tỷ lệ độ dài lời nhắc; con số này là cận
trên, vì phần đầu ra của giám khảo không dài theo chuỗi. Bước judge bắt buộc có --max-cost.
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl, write_jsonl
from src.common.prompts import build_judge_prompt
from src.common.rule_score import rule_scores_with_reference
from src.stage_b.b2_select import (EXTENDED_POOLS, all_chains_pool, extended_files, extended_quality_file,
                                   is_truncated, usable_pool)


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ============================================================ kho mở rộng
def wrong_chains(trajectories: Sequence[Mapping], cands: Sequence[Mapping], quality: Mapping[str, Mapping],
                 min_correct: int) -> tuple[list, dict]:
    """Chuỗi sai của các câu hỏi dùng được, theo thứ tự tid. Trả về (chuỗi, thông tin kho)."""
    pool, info = usable_pool(cands, quality, min_correct)
    chains, empty = all_chains_pool(trajectories, set(pool), info["excluded_tids"])
    correct = {c["tid"] for c in cands}
    wrong = sorted((c for v in chains.values() for c in v if c["tid"] not in correct), key=lambda c: c["tid"])
    return wrong, {"questions": len(pool), "usable_correct": info["usable_candidates"], "empty": empty,
                   "pool": pool, "questions_with_wrong": len({c["qid"] for c in wrong})}


def extended_candidates(pool: Mapping[str, Sequence[Mapping]], wrong: Sequence[Mapping]) -> list:
    """Mọi chuỗi của kho mở rộng: chuỗi đúng dùng được rồi tới chuỗi sai."""
    return [c for q in sorted(pool) for c in pool[q]] + list(wrong)


def estimate_cost(prev_cost: float, n_prev: int, prev_chars: Sequence[int], new_chars: Sequence[int]) -> dict:
    """Ước chi phí giám khảo cho lô mới từ chi phí mỗi chuỗi của lô trước.

    plain    số chuỗi mới nhân chi phí mỗi chuỗi của lô trước
    scaled   plain nhân tỷ lệ độ dài lời nhắc trung bình (mới trên cũ). Cận trên: chỉ phần đầu vào dài theo chuỗi.
    """
    if n_prev <= 0 or prev_cost <= 0:
        raise ValueError("cần tổng chi phí dương và ít nhất một chuỗi đã chấm ở lô trước")
    per = prev_cost / n_prev
    ratio = statistics.mean(new_chars) / statistics.mean(prev_chars) if new_chars and prev_chars else 1.0
    plain = per * len(new_chars)
    return {"per_chain": per, "n_new": len(new_chars), "length_ratio": ratio, "plain": plain,
            "scaled": plain * max(ratio, 1.0)}


def judge_prompt_chars(chains: Sequence[Mapping], questions: Mapping[str, Mapping]) -> list[int]:
    return [len(build_judge_prompt(questions[c["qid"]]["question"], c["text"], questions[c["qid"]].get("solution")))
            for c in chains]


def fit_argv(workdir: Path, student: str, names: Mapping[str, str], overrides: Sequence[str]) -> list[str]:
    """Dòng lệnh của b1_fit cho chuỗi sai: bản nền, 16-bit như file fit chính, bỏ LocalNat, file ra RIÊNG.

    Gói lại ở đây để không ai phải gõ tay --override stage_a_files.candidates: gõ thiếu --out thì b1_fit sẽ ghi
    nối chuỗi sai vào file fit chính.
    """
    argv = ["--workdir", str(workdir), "--student", student, "--base", "--skip-localnat", "--out", names["fit"],
            "--override", f"stage_a_files.candidates={names['wrong']}"]
    for o in overrides:
        argv += ["--override", o]
    return argv


# ============================================================ các bước
def load_stage_a(cfg: Mapping, wd: Path) -> tuple[list, dict, dict]:
    files = cfg["stage_a_files"]
    cands = read_jsonl(wd / files["candidates"])
    questions = {q["qid"]: q for q in read_jsonl(wd / files["questions"])}
    quality = {r["tid"]: r for r in read_jsonl(wd / "quality.jsonl")}
    if not cands or not questions or not quality:
        raise SystemExit(f"Thiếu dữ liệu trong {wd}: candidates {len(cands)}, questions {len(questions)}, "
                         f"quality {len(quality)}.")
    return cands, questions, quality


def step_prepare(cfg: Mapping, wd: Path, names: Mapping[str, str], prev_cost: float | None) -> int:
    from src.stage_a.a2_generate import all_trajectory_files
    cands, questions, quality = load_stage_a(cfg, wd)
    paths = all_trajectory_files(wd, cfg["stage_a_files"])
    if not paths:
        raise SystemExit("Bước prepare cần trajectories*.jsonl, máy này không có. Chạy trên máy giữ dữ liệu gốc.")
    trajectories = [t for p in paths for t in read_jsonl(p)]
    wrong, info = wrong_chains(trajectories, cands, quality, int(cfg["data"]["min_correct_per_question"]))
    if not wrong:
        raise SystemExit("Không có chuỗi sai nào trong các câu được giữ; không có gì để chấm bù.")
    out = wd / names["wrong"]
    if out.exists() and [r["tid"] for r in read_jsonl(out)] != [c["tid"] for c in wrong]:
        raise SystemExit(f"{out.name} đã có nhưng KHÁC danh sách chuỗi sai tính được bây giờ (kho đã đổi?). Các file "
                         f"chấm bù cũ không còn khớp: xoá {out.name} cùng các file *.wrong.* rồi chạy lại.")
    write_jsonl(out, wrong)

    by_reason = Counter("bị cắt" if c.get("finish_reason") == "length" else "sai đáp án hoặc thiếu \\boxed"
                        for c in wrong)
    by_teacher = Counter(c["teacher"] for c in wrong)
    print(f"[bù] {info['questions']} câu dùng được, {info['usable_correct']} chuỗi đúng dùng được, "
          f"{len(wrong)} chuỗi sai thuộc {info['questions_with_wrong']} câu"
          + (f" (bỏ {info['empty']} chuỗi rỗng)" if info["empty"] else ""))
    print("[bù] theo loại: " + ", ".join(f"{k} {v}" for k, v in sorted(by_reason.items()))
          + " | theo mô hình dạy: " + ", ".join(f"{k} {v}" for k, v in sorted(by_teacher.items())))
    print(f"[bù] đã ghi {out.name} ({len(wrong)} dòng, md5 {md5_of(out)}); chép file này sang máy chạy bước fit")

    prev = [r["tid"] for r in read_jsonl(wd / "judge.jsonl")]
    done = {c["tid"]: c for c in cands}
    prev_chars = judge_prompt_chars([done[t] for t in prev if t in done], questions)
    new_chars = judge_prompt_chars(wrong, questions)
    if not prev_chars:
        raise SystemExit("judge.jsonl rỗng hoặc không khớp candidates.jsonl: không có lô giám khảo trước để so.")
    ratio = statistics.mean(new_chars) / statistics.mean(prev_chars)
    print(f"\n[bù] giám khảo {cfg['judge']['model_id']}, max_tokens {cfg['judge']['max_tokens']}, nhiệt độ "
          f"{cfg['judge']['temperature']}, kèm lời giải tham chiếu (như lô trước)")
    print(f"[bù] lời nhắc trung bình: lô trước {statistics.mean(prev_chars):.0f} ký tự ({len(prev_chars)} chuỗi), "
          f"chuỗi sai {statistics.mean(new_chars):.0f} ký tự, tỷ lệ {ratio:.2f}")
    if prev_cost is None:
        print("[bù] CHƯA ƯỚC ĐƯỢC CHI PHÍ: judge.jsonl không ghi chi phí từng lượt. Lấy tổng chi phí của lô giám khảo "
              "trước trong lịch sử OpenRouter rồi chạy lại với --prev-cost-usd <số đô>.")
        return 0
    e = estimate_cost(prev_cost, len(prev_chars), prev_chars, new_chars)
    print(f"[bù] chi phí mỗi chuỗi của lô trước: {prev_cost:.2f} / {len(prev_chars)} = {e['per_chain']:.6f} đô")
    print(f"[bù] ƯỚC TÍNH cho {e['n_new']} chuỗi sai: {e['plain']:.2f} đô nếu chi phí mỗi chuỗi như cũ; "
          f"{e['scaled']:.2f} đô nếu tăng đúng theo độ dài lời nhắc (cận trên)")
    print(f"[bù] gợi ý: --max-cost {e['scaled'] * 1.25:.1f}. Phản hồi bị cắt ở lô trước từng phải chấm lại với "
          f"--override judge.max_tokens=16000; nếu lặp lại thì tốn thêm một ít.")
    return 0


def read_wrong(wd: Path, names: Mapping[str, str]) -> list:
    wrong = read_jsonl(wd / names["wrong"])
    if not wrong:
        raise SystemExit(f"Không thấy {names['wrong']} trong {wd}. Chạy --step prepare trước (máy Mac), rồi chép sang.")
    return wrong


def step_judge(cfg: Mapping, wd: Path, names: Mapping[str, str], max_cost: float | None, workers: int) -> int:
    from src.stage_a import a4_score_quality as a4
    if max_cost is None:
        raise SystemExit("Bước judge tiêu tiền: bắt buộc có --max-cost <số đô>. Xem ước tính ở --step prepare.")
    wrong = read_wrong(wd, names)
    questions = {q["qid"]: q for q in read_jsonl(wd / cfg["stage_a_files"]["questions"])}
    a4.run_judge(cfg, wrong, questions, wd, names["judge"], cfg["judge"]["model_id"], workers,
                 use_reference=True, max_cost=max_cost)
    done = {r["tid"] for r in read_jsonl(wd / names["judge"])}
    left = sum(c["tid"] not in done for c in wrong)
    print(f"[bù] {names['judge']}: {len(wrong) - left}/{len(wrong)} chuỗi sai có điểm"
          + (f"; còn {left}, chạy lại cùng lệnh để chấm bù (lô trước phải thêm --override judge.max_tokens=16000 "
             f"cho các phản hồi bị cắt)" if left else ""))
    return 0


def step_embed(cfg: Mapping, wd: Path, names: Mapping[str, str], device: str | None, batch_size: int) -> int:
    import numpy as np
    from src.stage_a import a5_embed as a5
    wrong = read_wrong(wd, names)
    emb = cfg["embedding"]
    vecs = a5.encode([c["text"] for c in wrong], emb["model_id"], device or emb["device"], emb["max_length"],
                     batch_size)
    main = np.load(wd / "embeddings.npz")
    if vecs.shape[1] != main["vectors"].shape[1]:
        raise SystemExit(f"Số chiều lệch: chuỗi sai {vecs.shape[1]}, embeddings.npz {main['vectors'].shape[1]}.")
    np.savez_compressed(wd / names["embeddings"], tids=np.array([c["tid"] for c in wrong]), vectors=vecs)
    print(f"[bù] đã ghi {names['embeddings']}: {vecs.shape[0]} vector, {vecs.shape[1]} chiều")
    return 0


def step_fit(wd: Path, names: Mapping[str, str], student: str, overrides: Sequence[str]) -> int:
    from src.stage_b import b1_fit
    read_wrong(wd, names)
    code = b1_fit.main(fit_argv(wd, student, names, overrides))
    path = wd / names["fit"]
    if path.exists():
        print(f"[bù] {names['fit']}: {len(read_jsonl(path))} dòng, md5 {md5_of(path)}; chép về máy chạy b2_select")
    return code


def coverage(wd: Path, names: Mapping[str, str], wrong: Sequence[Mapping]) -> dict:
    import numpy as np
    tids = {c["tid"] for c in wrong}
    judged = {r["tid"] for r in read_jsonl(wd / names["judge"])} & tids
    fitted = {r["tid"] for r in read_jsonl(wd / names["fit"])} & tids
    embedded = set()
    if (wd / names["embeddings"]).exists():
        embedded = {str(t) for t in np.load(wd / names["embeddings"])["tids"]} & tids
    return {"total": len(tids), "judge": len(judged), "embeddings": len(embedded), "fit": len(fitted)}


def step_status(wd: Path, names: Mapping[str, str]) -> int:
    wrong = read_wrong(wd, names)
    c = coverage(wd, names, wrong)
    print(f"[bù] {names['wrong']}: {c['total']} chuỗi sai, md5 {md5_of(wd / names['wrong'])}")
    for key in ("judge", "embeddings", "fit"):
        print(f"[bù] {names[key]:<34} {c[key]}/{c['total']}")
    for pool in EXTENDED_POOLS:
        q = wd / extended_quality_file(pool)
        print(f"[bù] {q.name:<34} " + (f"{len(read_jsonl(q))} dòng" if q.exists() else "chưa có, chạy --step combine"))
    return 0


def step_combine(cfg: Mapping, wd: Path, names: Mapping[str, str]) -> int:
    from src.stage_a.a4_score_quality import combine
    cands, _questions, quality = load_stage_a(cfg, wd)
    wrong = read_wrong(wd, names)
    pool, _info = usable_pool(cands, quality, int(cfg["data"]["min_correct_per_question"]))
    stray = [c["tid"] for c in wrong if c["qid"] not in pool]
    if stray:
        raise SystemExit(f"{names['wrong']} có {len(stray)} chuỗi thuộc câu không còn trong kho dùng được (ví dụ "
                         f"{stray[:3]}): kho đã đổi sau bước prepare, chạy lại từ prepare.")
    c = coverage(wd, names, wrong)
    lacking = [f"{names[k]} thiếu {c['total'] - c[k]}" for k in ("embeddings", "fit") if c[k] < c["total"]]
    if lacking:
        raise SystemExit("Chưa đủ để ghép: " + "; ".join(lacking) + " chuỗi. Biểu diễn và tín hiệu phải đủ cho mọi "
                         "chuỗi sai (chỉ điểm giám khảo được phép thiếu, chuỗi đó sẽ bị loại khỏi kho mở rộng).")
    judge_rows = read_jsonl(wd / "judge.jsonl") + read_jsonl(wd / names["judge"])
    reference = [c["text"] for c in cands]          # thang điểm quy tắc của kho gốc, đúng thứ tự a4 đã dùng
    for pool_name in EXTENDED_POOLS:
        part = [c for c in wrong if not (pool_name == "all_complete" and is_truncated(c))]
        chains = extended_candidates(pool, part)
        rule = rule_scores_with_reference([c["text"] for c in chains], reference)
        rows = combine(chains, judge_rows, cfg["quality"]["alpha"], rule=rule)
        by_tid = {r["tid"]: r for r in rows}
        drift = [t for t, r in by_tid.items() if t in quality and (
            r["llm_score"] != quality[t].get("llm_score") or r["rule_score"] != quality[t].get("rule_score"))]
        if drift:
            raise SystemExit(f"Điểm giám khảo hoặc điểm quy tắc của {len(drift)} chuỗi đúng khác quality.jsonl (ví dụ "
                             f"{drift[:3]}): quality.jsonl cũ hơn judge.jsonl hoặc candidates.jsonl, chạy "
                             f"a4_score_quality --rule-only trước.")
        touched = {c["qid"] for c in part}
        leaked = [t for t, r in by_tid.items() if r["qid"] not in touched and r["qual"] != quality[t].get("qual")]
        if leaked:
            raise SystemExit(f"Qual của {len(leaked)} chuỗi thuộc câu KHÔNG có chuỗi sai lại khác quality.jsonl (ví dụ "
                             f"{leaked[:3]}). Kho mở rộng không được làm đổi các câu đó; đừng dùng file này.")
        out = wd / extended_quality_file(pool_name)
        write_jsonl(out, rows)

        part_tids = {c["tid"] for c in part}
        no_qual = sum(1 for t in part_tids if by_tid[t]["qual"] is None)
        moved = sum(1 for t, r in by_tid.items() if t in quality and r["qual"] is not None
                    and abs(r["qual"] - quality[t]["qual"]) > 1e-9)
        w_llm = [by_tid[t]["llm_score"] for t in part_tids if by_tid[t]["llm_score"] is not None]
        c_llm = [r["llm_score"] for t, r in by_tid.items() if t not in part_tids and r["llm_score"] is not None]
        print(f"[bù] {out.name}: {len(rows)} chuỗi ({len(rows) - len(part)} đúng, {len(part)} sai thuộc "
              f"{len(touched)} câu), md5 {md5_of(out)}")
        print(f"     chuỗi sai giám khảo không chấm được: {no_qual} (b2_select loại khỏi kho này)")
        if w_llm:
            print(f"     điểm giám khảo trung bình: chuỗi sai {statistics.mean(w_llm):.3f}, chuỗi đúng "
                  f"{statistics.mean(c_llm):.3f}")
        print(f"     Qual của chuỗi đúng đổi ở {moved} chuỗi, tất cả thuộc {len(touched)} câu có chuỗi sai; "
              f"{len(pool) - len(touched)} câu còn lại giống hệt quality.jsonl")
    print("[bù] xong. Chuỗi sai phải có điểm giám khảo thấp hơn rõ chuỗi đúng; nếu không thì giám khảo không phân "
          "biệt được đáp án sai và biến thể không lọc mất ý nghĩa.")
    print("[bù] Chạy: python -m src.stage_b.b2_select --ablation no_prefilter")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=["prepare", "judge", "embed", "fit", "combine", "status"])
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--student", default="qwen1_5b", help="mô hình học cho bước fit; biến thể này chỉ chạy trên 1,5B")
    ap.add_argument("--prev-cost-usd", type=float, help="prepare: tổng chi phí (đô) của lô giám khảo trước")
    ap.add_argument("--max-cost", type=float, help="judge: dừng khi chi phí cộng dồn (đô) chạm mức này; bắt buộc")
    ap.add_argument("--workers", type=int, default=4, help="judge: số luồng, mặc định như a4")
    ap.add_argument("--device", help="embed: mặc định embedding.device")
    ap.add_argument("--batch-size", type=int, default=8, help="embed: mặc định như a5")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(student=args.student, overrides=args.override)
    wd = resolve_path(cfg, args.workdir)
    names = extended_files(cfg)
    if args.step == "prepare":
        return step_prepare(cfg, wd, names, args.prev_cost_usd)
    if args.step == "judge":
        return step_judge(cfg, wd, names, args.max_cost, args.workers)
    if args.step == "embed":
        return step_embed(cfg, wd, names, args.device, args.batch_size)
    if args.step == "fit":
        return step_fit(wd, names, args.student, args.override)
    if args.step == "combine":
        return step_combine(cfg, wd, names)
    return step_status(wd, names)


if __name__ == "__main__":
    raise SystemExit(main())

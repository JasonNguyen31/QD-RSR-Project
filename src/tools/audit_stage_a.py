"""Kiểm tra toàn vẹn dữ liệu Giai đoạn A: các file có khớp nhau không.

    python -m src.tools.audit_stage_a --workdir data/stage_a

Khác với tests/: bài kiểm thử kiểm tra MÃ, còn lệnh này kiểm tra DỮ LIỆU đã sinh ra. Loại lỗi nó bắt là
loại nguy hiểm nhất: mã đúng nhưng file lệch nhau vì chạy nửa chừng, ví dụ embeddings.npz còn là bản cũ
sau khi candidates.jsonl đã đổi.

Chạy trước khi sang Giai đoạn B, và chạy lại mỗi khi có bước nào của Giai đoạn A chạy lại.
Trả mã thoát khác 0 nếu có phép kiểm tra nào hỏng, để dùng được trong script.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl, split_tid
from src.stage_a.a2_generate import all_trajectory_files


class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((ok, name, detail))
        return ok

    def skip(self, name: str, detail: str) -> None:
        """Phép kiểm tra không áp dụng được vì thiếu file. Khác với HỎNG: đây không phải lỗi."""
        self.rows.append((None, name, detail))

    def report(self) -> int:
        w = max(len(n) for _, n, _ in self.rows) + 2
        bad = sum(1 for ok, _, _ in self.rows if ok is False)
        skipped = sum(1 for ok, _, _ in self.rows if ok is None)
        for ok, name, detail in self.rows:
            tag = "BỎ QUA" if ok is None else ("ĐẠT  " if ok else "HỎNG ")
            print(f"  {tag} {name:<{w}}{detail}")
        done = len(self.rows) - skipped
        print(f"\n{done - bad}/{done} phép kiểm tra đạt" + (f", {skipped} bỏ qua" if skipped else ""))
        if bad:
            print("Có phép kiểm tra HỎNG: chạy lại bước tương ứng trước khi sang Giai đoạn B.")
        return bad


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--validation", default="data/raw/validation.jsonl")
    args = ap.parse_args(argv)

    cfg = load_config()
    wd: Path = resolve_path(cfg, args.workdir)
    files = cfg["stage_a_files"]
    k = cfg["selection"]["k"]
    min_correct = cfg["data"]["min_correct_per_question"]
    n_teachers, n_samples = len(cfg["teachers"]), cfg["generation"]["samples_per_teacher"]
    a = Audit()
    print(f"[kiểm tra] {wd}\n")

    # ---- questions
    qs = read_jsonl(wd / files["questions"])
    qids = [q["qid"] for q in qs]
    a.check(len(qids) == len(set(qids)), "qid không trùng nhau", f"{len(qids)} câu hỏi")
    a.check(all(q.get("gold") for q in qs), "mọi câu hỏi có đáp án chuẩn")
    groups = Counter(f"{q['source']}" + (f"-L{q['level']}" if q.get("level") else "") for q in qs)
    a.check(True, "thành phần theo nhóm", dict(groups))

    # ---- trajectories và labels
    # Máy chạy Giai đoạn B chỉ cần candidates, quality và embeddings, nên thường KHÔNG có file chuỗi và
    # file nhãn. Thiếu chúng là bình thường, không phải lỗi, nên bỏ qua thay vì báo hỏng.
    traj_files = all_trajectory_files(wd, files)
    trajs = [t for p in traj_files for t in read_jsonl(p)]
    tids = [t["tid"] for t in trajs]
    if trajs:
        a.check(len(tids) == len(set(tids)), "tid không trùng giữa các lô",
                f"{len(traj_files)} file, {len(tids)} chuỗi")
        want = len(qs) * n_teachers * n_samples
        a.check(len(tids) == want, "đủ số chuỗi", f"{len(tids)}/{want}")
        per_q = Counter(split_tid(t)[0] for t in tids)
        short = [q for q in qids if per_q[q] != n_teachers * n_samples]
        a.check(not short, "mọi câu hỏi đủ chuỗi", f"{len(short)} câu thiếu" if short else "")
        prompts = {t.get("prompt_id") for t in trajs}
        a.check(len(prompts) == 1, "cùng một prompt cho mọi chuỗi", str(prompts))
    else:
        a.skip("bốn phép trên file chuỗi", "không có trajectories.*.jsonl — bình thường ở máy Giai đoạn B")

    labels = read_jsonl(wd / files["labels"])
    if labels:
        lab = {l["tid"]: l for l in labels}
        if trajs:
            a.check(set(lab) == set(tids), "nhãn phủ đúng tập chuỗi", f"{len(lab)} nhãn")
        bad_trunc = [l for l in labels if l["reason"] == "truncated" and l["correct"]]
        a.check(not bad_trunc, "chuỗi bị cắt luôn tính là sai", f"{len(bad_trunc)} vi phạm" if bad_trunc else "")
        bad_reason = [l for l in labels if l["correct"] != (l["reason"] == "match")]
        a.check(not bad_reason, "cờ đúng/sai khớp với lý do", f"{len(bad_reason)} vi phạm" if bad_reason else "")
    else:
        a.skip("ba phép trên file nhãn", "không có labels.jsonl — bình thường ở máy Giai đoạn B")

    # ---- ứng viên phải là chuỗi đúng: chỉ kiểm được khi có nhãn
    lab = {l["tid"]: l for l in labels}

    # ---- candidates và kept_qids
    cands = read_jsonl(wd / files["candidates"])
    ctids = [c["tid"] for c in cands]
    a.check(len(ctids) == len(set(ctids)), "ứng viên không trùng", f"{len(ctids)} ứng viên")
    if labels:
        a.check(all(lab[t]["correct"] for t in ctids if t in lab), "mọi ứng viên đều là chuỗi đúng")
    import json
    # a3 ghi {"min_correct": ..., "qids": [...]}, không phải một danh sách trần. Đọc nhầm thì set() lấy ra
    # tên hai khoá chứ không phải mã câu hỏi, và mọi phép kiểm tra phía sau đều báo hỏng oan.
    raw = json.loads((wd / files["kept_qids"]).read_text(encoding="utf-8"))
    kept = set(raw["qids"] if isinstance(raw, dict) else raw)
    per_qc = Counter(c["qid"] for c in cands)
    a.check(set(per_qc) == kept, "ứng viên đúng bằng tập câu được giữ", f"{len(kept)} câu")
    thin = [q for q in kept if per_qc[q] < min_correct]
    a.check(not thin, f"mọi câu được giữ có ít nhất {min_correct} ứng viên",
            f"{len(thin)} câu thiếu" if thin else "")

    # ---- quality
    qual = read_jsonl(wd / "quality.jsonl")
    qmap = {r["tid"]: r for r in qual}
    a.check(set(qmap) == set(ctids), "điểm chất lượng phủ đúng tập ứng viên", f"{len(qmap)} dòng")
    a.check(all(r["rule_score"] is not None for r in qual), "mọi ứng viên có điểm quy tắc")
    mismatch = [r for r in qual if (r["qual"] is None) != (r["llm_score"] is None)]
    a.check(not mismatch, "qual trống đúng khi thiếu điểm giám khảo", f"{len(mismatch)} vi phạm" if mismatch else "")
    no_judge = [r for r in qual if r["llm_score"] is None]
    a.check(True, "ứng viên thiếu điểm giám khảo",
            f"{len(no_judge)} = {len(no_judge) / max(len(qual), 1):.2%}")

    # ---- embeddings
    npz = wd / "embeddings.npz"
    if npz.exists():
        import numpy as np
        d = np.load(npz, allow_pickle=False)
        etids, vecs = [str(x) for x in d["tids"]], d["vectors"]
        a.check(etids == ctids, "vector khớp ĐÚNG THỨ TỰ với ứng viên",
                f"{len(etids)} vector" if etids == ctids else
                f"{len(etids)} vector so với {len(ctids)} ứng viên, ĐÃ LỆCH")
        norms = np.linalg.norm(vecs, axis=1)
        a.check(bool(np.all(np.isfinite(vecs))), "vector không có giá trị lỗi")
        a.check(bool(np.allclose(norms, 1.0, atol=1e-3)), "vector đã chuẩn hoá L2",
                f"chuẩn nhỏ nhất {norms.min():.4f}, lớn nhất {norms.max():.4f}")
    else:
        a.check(False, "có file embeddings.npz", "không thấy, chạy a5_embed")

    # ---- không rò rỉ sang tập kiểm định
    val = resolve_path(cfg, args.validation)
    if val.exists():
        vq = {q["qid"] for q in read_jsonl(val)}
        overlap = vq & set(qids)
        a.check(not overlap, "không giao với tập kiểm định",
                f"{len(overlap)} câu trùng" if overlap else f"{len(vq)} câu kiểm định")

    # ---- tập thật sự dùng được ở Giai đoạn B
    lost = Counter(r["qid"] for r in qual if r["qual"] is None)
    usable = [q for q in kept if per_qc[q] - lost[q] >= k]
    print(f"\n  Tập dùng cho Giai đoạn B: {len(usable)} câu hỏi, {len(usable) * k} mẫu mỗi phương án")
    print(f"  (bỏ {len(kept) - len(usable)} câu tụt dưới {k} ứng viên khi loại chuỗi thiếu điểm giám khảo)\n")
    return a.report()


if __name__ == "__main__":
    raise SystemExit(main())

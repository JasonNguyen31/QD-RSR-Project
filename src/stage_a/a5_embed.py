"""
a5_embed: tính biểu diễn vector cho từng chuỗi, và đo xem cách biểu diễn có phân biệt được chuỗi hay không.

    python -m src.stage_a.a5_embed --workdir data/pilot/run2_boxed --probe
    python -m src.stage_a.a5_embed --workdir data/pilot/run2_boxed

Đọc  <workdir>/candidates.jsonl
Ghi  <workdir>/embeddings.npz   mảng vector đã chuẩn hoá L2, kèm danh sách tid theo đúng thứ tự
     <workdir>/embed_probe.json kết quả phép đo phân bố khoảng cách (chỉ khi có --probe)

Chạy trên máy Mac, thiết bị MPS.

**Phép đo để trả lời câu hỏi còn mở ở mục I.5.** Feng và cộng sự đo được biểu diễn toàn chuỗi chỉ chọn đúng
cặp khác chiến lược 40% số lần, so với 53% của biểu diễn theo bước, và phân bố điểm dồn hết vào khoảng
0,00-0,04. Con số đó đo trên Long-CoT. Với chuỗi ngắn của đề tài thì chưa biết. Cờ --probe in ra phân bố
khoảng cách trong cùng câu hỏi để xem có đủ trải rộng hay không. Đọc kết quả:
  - khoảng cách trong cùng câu hỏi dồn cục quanh một giá trị  -> biểu diễn toàn chuỗi KHÔNG phân biệt được,
    phải chuyển sang biểu diễn theo bước trước khi chấm 18.000 chuỗi
  - trải rộng, và khoảng cách giữa các mô hình dạy lớn hơn rõ so với trong cùng mô hình dạy -> dùng được

Phép đo này cũng trả lời câu hỏi thứ hai: ba chuỗi từ cùng một mô hình dạy có quá giống nhau không, tức
nhiệt độ 0,8 có đang tạo ra đủ khác biệt hay không (mục I.5, dòng tham số sinh).
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl, split_tid, write_json


def l2_normalize(mat):
    import numpy as np

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0        # chuỗi rỗng thì giữ vector 0 thay vì chia cho 0
    return mat / norms


def pairwise_distances(vecs) -> list[float]:
    """Khoảng cách Euclid giữa mọi cặp trong một nhóm nhỏ. Vector đã chuẩn hoá nên giá trị nằm trong [0, 2]."""
    import numpy as np

    n = len(vecs)
    return [float(np.linalg.norm(vecs[i] - vecs[j])) for i in range(n) for j in range(i + 1, n)]


def probe(tids: Sequence[str], vecs, by_question: bool = True) -> dict:
    """Phân bố khoảng cách, tách theo ba nhóm để so sánh với nhau."""
    import numpy as np

    idx = {t: i for i, t in enumerate(tids)}
    groups: dict[str, list[str]] = defaultdict(list)
    for t in tids:
        qid, _, _ = split_tid(t)
        groups[qid].append(t)

    within_teacher, across_teacher, all_within_q = [], [], []
    for qid, ts in groups.items():
        if len(ts) < 2:
            continue
        vs = np.stack([vecs[idx[t]] for t in ts])
        all_within_q.extend(pairwise_distances(vs))
        by_teacher: dict[str, list[str]] = defaultdict(list)
        for t in ts:
            by_teacher[split_tid(t)[1]].append(t)
        for tt in by_teacher.values():
            if len(tt) >= 2:
                within_teacher.extend(pairwise_distances(np.stack([vecs[idx[t]] for t in tt])))
        keys = sorted(by_teacher)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                for ta in by_teacher[keys[a]]:
                    for tb in by_teacher[keys[b]]:
                        across_teacher.append(float(np.linalg.norm(vecs[idx[ta]] - vecs[idx[tb]])))

    def summarize(xs: Sequence[float]) -> dict:
        if not xs:
            return {"n": 0}
        xs = sorted(xs)
        q = lambda p: xs[min(int(p * len(xs)), len(xs) - 1)]  # noqa: E731
        return {"n": len(xs), "min": round(xs[0], 4), "p25": round(q(0.25), 4),
                "median": round(q(0.5), 4), "p75": round(q(0.75), 4), "max": round(xs[-1], 4),
                "mean": round(statistics.mean(xs), 4),
                "sd": round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0}

    return {"questions": len(groups),
            "within_question": summarize(all_within_q),
            "within_teacher": summarize(within_teacher),
            "across_teacher": summarize(across_teacher)}


def print_probe(p: Mapping) -> None:
    print(f"\n[a5] phân bố khoảng cách trên {p['questions']} câu hỏi (vector đã chuẩn hoá, giá trị trong [0, 2])")
    print(f"{'nhóm':<22}{'số cặp':>8}{'nhỏ nhất':>10}{'p25':>8}{'trung vị':>10}{'p75':>8}{'lớn nhất':>10}{'độ lệch':>9}")
    for name, label in [("within_question", "trong cùng câu hỏi"), ("within_teacher", "cùng mô hình dạy"),
                        ("across_teacher", "khác mô hình dạy")]:
        s = p[name]
        if not s.get("n"):
            continue
        print(f"{label:<22}{s['n']:>8}{s['min']:>10.3f}{s['p25']:>8.3f}{s['median']:>10.3f}"
              f"{s['p75']:>8.3f}{s['max']:>10.3f}{s['sd']:>9.3f}")

    wt, at = p["within_teacher"], p["across_teacher"]
    if wt.get("n") and at.get("n"):
        print(f"\n[a5] trung vị khác mô hình dạy / cùng mô hình dạy = {at['median'] / max(wt['median'], 1e-9):.2f} lần")
        print("     tỷ lệ này càng lớn thì ba mô hình dạy càng tạo ra chuỗi khác nhau, đúng điều kiện cần cho Div.")
    wq = p["within_question"]
    if wq.get("n"):
        spread = wq["p75"] - wq["p25"]
        print(f"[a5] độ trải giữa p25 và p75 trong cùng câu hỏi = {spread:.3f}")
        print("     nếu con số này rất nhỏ thì biểu diễn toàn chuỗi không phân biệt được các chuỗi,")
        print("     phải chuyển sang biểu diễn theo từng bước TRƯỚC khi chấm 18.000 chuỗi.")


def encode(texts: Sequence[str], model_id: str, device: str, max_length: int, batch_size: int):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"[a5] nạp {model_id} trên thiết bị {device}...")
    model = SentenceTransformer(model_id, device=device)
    model.max_seq_length = max_length
    vecs = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=False)
    return l2_normalize(np.asarray(vecs, dtype=np.float32))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", help="mặc định paths.data_stage_a")
    ap.add_argument("--probe", action="store_true", help="in phân bố khoảng cách và ghi embed_probe.json")
    ap.add_argument("--limit", type=int, help="chỉ tính N chuỗi đầu, dùng để thử")
    ap.add_argument("--device", help="mặc định embedding.device (mps trên Mac, cpu nếu không có)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="embeddings.npz")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    import numpy as np

    cfg = load_config(overrides=args.override)
    workdir: Path = resolve_path(cfg, args.workdir or cfg["paths"]["data_stage_a"])
    cands = read_jsonl(workdir / cfg["stage_a_files"]["candidates"])
    if not cands:
        raise SystemExit(f"Không có ứng viên trong {workdir}. Chạy a3_filter trước.")
    if args.limit:
        cands = cands[: args.limit]

    emb_cfg = cfg["embedding"]
    vecs = encode([c["text"] for c in cands], emb_cfg["model_id"], args.device or emb_cfg["device"],
                  emb_cfg["max_length"], args.batch_size)
    tids = [c["tid"] for c in cands]
    np.savez_compressed(workdir / args.out, tids=np.array(tids), vectors=vecs)
    print(f"[a5] đã ghi {workdir / args.out}: {vecs.shape[0]} vector, {vecs.shape[1]} chiều")

    if args.probe:
        p = probe(tids, vecs)
        print_probe(p)
        write_json(workdir / "embed_probe.json", {"model": emb_cfg["model_id"], "n": len(tids), **p})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

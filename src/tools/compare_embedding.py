"""So embedding toàn chuỗi với embedding theo từng bước, tái lập phép đo của Feng 2025 ở quy mô nhỏ.

    python -m src.tools.compare_embedding --workdir data/stage_a --questions 60 --max-cost 1

Paper đã hứa phép đo này ở §3.5 (Notion R2). Feng đo được embedding toàn chuỗi chỉ chọn đúng cặp khác
chiến lược 40% số lần, so với 53% của embedding theo bước, nhưng đo trên chuỗi dài hàng nghìn token.
Chuỗi của đề tài ngắn hơn nhiều, nên phải đo lại trên chính dữ liệu của nhóm.

Bốn bước:
  1. Rút Q câu hỏi, mỗi câu lấy ba ứng viên, tạo ba cặp mỗi câu.
  2. Nhờ mô hình ngôn ngữ tóm tắt mỗi chuỗi thành 3-5 bước, và gán nhãn từng cặp là cùng hay khác chiến lược.
     Nhãn này đóng vai trò chuẩn đối chiếu, giống cách Feng dùng mô hình phân loại chiến lược.
  3. Tính hai khoảng cách cho mỗi cặp: toàn chuỗi (vector cả chuỗi) và theo bước (khoảng cách Chamfer
     giữa hai tập vector bước, tức trung bình khoảng cách tới bước gần nhất, lấy hai chiều).
  4. So hai cách trên ba chỉ số: phân biệt được cặp khác chiến lược không (AUC), chọn đúng cặp khác nhất
     trong ba cặp của một câu không, và phân bố khoảng cách có đủ trải rộng không.

Ghi  <workdir>/embed_cmp_steps.jsonl   tóm tắt từng bước, ghi dần
     <workdir>/embed_cmp_labels.jsonl  nhãn cùng/khác chiến lược, ghi dần
     <workdir>/embed_cmp.json          kết quả để trích vào paper
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import JsonlWriter, now_iso, read_jsonl, write_json
from src.common.prompts import build_strategy_prompt, build_summarize_prompt
from src.stage_a.a4_score_quality import make_judge

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it

_STEP = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*(.+)$")


def parse_steps(text: str, max_steps: int = 6) -> list[str]:
    """Tách danh sách bước từ phản hồi. Bỏ dòng rỗng và dòng dẫn nhập."""
    steps = [m.group(1).strip() for line in (text or "").splitlines() if (m := _STEP.match(line))]
    if not steps:                       # mô hình trả văn xuôi: cắt theo câu
        steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if len(s.strip()) > 15]
    return steps[:max_steps]


def parse_verdict(text: str) -> str | None:
    s = re.sub(r"^\s*```(?:json)?|```\s*$", "", (text or "").strip())
    try:
        v = json.loads(s[s.find("{"): s.rfind("}") + 1]).get("verdict", "")
    except Exception:  # noqa: BLE001
        v = ""
    v = str(v).strip().lower()
    if v in ("same", "different"):
        return v
    low = (text or "").lower()
    if "different" in low and "same" not in low:
        return "different"
    if "same" in low and "different" not in low:
        return "same"
    return None


def chamfer(a, b) -> float:
    """Khoảng cách Chamfer giữa hai tập vector bước: trung bình khoảng cách tới bước gần nhất, hai chiều.

    Dạng bất đối xứng của Feng bắt được trường hợp một lời giải có bước mà lời giải kia hoàn toàn không có.
    """
    import numpy as np

    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) / 2)


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Diện tích dưới đường ROC, tính bằng thống kê Mann-Whitney, xử lý cả giá trị bằng nhau."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def spread(xs: Sequence[float]) -> dict:
    xs = sorted(xs)
    q = lambda p: xs[min(int(p * len(xs)), len(xs) - 1)]  # noqa: E731
    return {"n": len(xs), "p25": round(q(0.25), 4), "median": round(q(0.5), 4),
            "p75": round(q(0.75), 4), "iqr": round(q(0.75) - q(0.25), 4)}


def build_pairs(cands: Sequence[Mapping], n_questions: int, seed: int) -> list[dict]:
    by_q: dict[str, list] = defaultdict(list)
    for c in cands:
        by_q[c["qid"]].append(c)
    rng = random.Random(seed)
    qids = [q for q in sorted(by_q) if len(by_q[q]) >= 3]
    rng.shuffle(qids)
    out = []
    for qid in qids[:n_questions]:
        three = rng.sample(by_q[qid], 3)
        for i in range(3):
            for j in range(i + 1, 3):
                out.append({"qid": qid, "a": three[i], "b": three[j]})
    return out


def run_llm(cfg, model_id, items, key_fn, prompt_fn, path: Path, workers: int, max_cost, desc: str):
    """Chạy một lượt gọi mô hình cho từng phần tử chưa có trong file, ghi dần."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    existing = {r["key"]: r for r in read_jsonl(path)}
    todo = [it for it in items if key_fn(it) not in existing]
    print(f"[{desc}] {len(items)} mục, đã có {len(existing)}, cần gọi {len(todo)}")
    if todo:
        judge = make_llm(cfg, model_id)
        with JsonlWriter(path) as w:
            def work(it):
                if max_cost is not None and getattr(judge, "cost", 0.0) >= max_cost:
                    return "ngân sách"
                try:
                    text = judge.raw(prompt_fn(it))
                except Exception as exc:  # noqa: BLE001
                    return f"{key_fn(it)}: {type(exc).__name__}"
                w.append({"key": key_fn(it), "text": text, "model": model_id, "ts": now_iso()})
                return None
            ex = ThreadPoolExecutor(max_workers=workers)
            futs = [ex.submit(work, it) for it in todo]
            errs = [e for f in tqdm(as_completed(futs), total=len(futs), desc=desc) if (e := f.result())]
            ex.shutdown(wait=True)
        if errs:
            print(f"[{desc}] {len(errs)} lỗi, ví dụ {errs[:2]}", file=sys.stderr)
        print(f"[{desc}] chi phí {getattr(judge, 'cost', 0.0):.4f} USD")
    return {r["key"]: r["text"] for r in read_jsonl(path)}


def make_llm(cfg, model_id):
    """Bọc lại giám khảo để lấy văn bản thô, vì hai prompt này không trả JSON điểm số."""
    base = make_judge(cfg, model_id)

    class Raw:
        cost = 0.0

        def raw(self, prompt: str) -> str:
            res = base._client.chat(model=model_id, messages=[{"role": "user", "content": prompt}],
                                    temperature=0.0, top_p=1.0,
                                    max_tokens=cfg["judge"].get("max_tokens", 3000))
            self.cost = getattr(base._client.tracker, "cost", 0.0)
            return res.text

    return Raw()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--questions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", help="mặc định judge.model_id")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cost", type=float)
    ap.add_argument("--device")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    import numpy as np

    cfg = load_config(overrides=args.override)
    wd: Path = resolve_path(cfg, args.workdir)
    files = cfg["stage_a_files"]
    cands = {c["tid"]: c for c in read_jsonl(wd / files["candidates"])}
    questions = {q["qid"]: q for q in read_jsonl(wd / files["questions"])}
    pairs = build_pairs(list(cands.values()), args.questions, args.seed)
    if not pairs:
        raise SystemExit("Không dựng được cặp nào.")
    tids = sorted({p[k]["tid"] for p in pairs for k in ("a", "b")})
    model_id = args.model or cfg["judge"]["model_id"]
    print(f"[so embedding] {len({p['qid'] for p in pairs})} câu hỏi, {len(tids)} chuỗi, {len(pairs)} cặp")

    steps_text = run_llm(cfg, model_id, [cands[t] for t in tids], lambda c: c["tid"],
                         lambda c: build_summarize_prompt(c["text"]),
                         wd / "embed_cmp_steps.jsonl", args.workers, args.max_cost, "tóm tắt bước")
    labels_text = run_llm(cfg, model_id, pairs, lambda p: f"{p['a']['tid']}__{p['b']['tid']}",
                          lambda p: build_strategy_prompt(questions[p["qid"]]["question"],
                                                          p["a"]["text"], p["b"]["text"]),
                          wd / "embed_cmp_labels.jsonl", args.workers, args.max_cost, "gán nhãn cặp")

    steps = {t: parse_steps(steps_text[t]) for t in tids if t in steps_text}
    steps = {t: v for t, v in steps.items() if v}
    print(f"[so embedding] tóm tắt được {len(steps)}/{len(tids)} chuỗi, "
          f"trung bình {statistics.mean(len(v) for v in steps.values()):.1f} bước")

    from src.stage_a.a5_embed import encode
    emb_cfg = cfg["embedding"]
    device = args.device or emb_cfg["device"]
    whole = encode([cands[t]["text"] for t in tids], emb_cfg["model_id"], device, emb_cfg["max_length"], 8)
    whole = {t: whole[i] for i, t in enumerate(tids)}
    flat = [(t, s) for t in steps for s in steps[t]]
    step_vecs = encode([s for _, s in flat], emb_cfg["model_id"], device, emb_cfg["max_length"], 16)
    by_tid: dict[str, list] = defaultdict(list)
    for (t, _), v in zip(flat, step_vecs):
        by_tid[t].append(v)
    step_emb = {t: np.stack(v) for t, v in by_tid.items()}

    rows = []
    for p in pairs:
        ta, tb = p["a"]["tid"], p["b"]["tid"]
        key = f"{ta}__{tb}"
        v = parse_verdict(labels_text.get(key, ""))
        if v is None or ta not in step_emb or tb not in step_emb:
            continue
        rows.append({"qid": p["qid"], "pair": key, "label": 1 if v == "different" else 0,
                     "whole": float(np.linalg.norm(whole[ta] - whole[tb])),
                     "step": chamfer(step_emb[ta], step_emb[tb])})
    if not rows:
        raise SystemExit("Không có cặp nào đủ dữ liệu.")

    n_diff = sum(r["label"] for r in rows)
    print(f"\n[so embedding] {len(rows)} cặp dùng được, {n_diff} cặp khác chiến lược ({n_diff / len(rows):.1%})")

    out = {"pairs": len(rows), "different": n_diff, "model": model_id}
    print(f"\n{'cách đo':<22}{'AUC':>8}{'chọn đúng cặp khác nhất':>26}{'p25':>8}{'trung vị':>10}{'p75':>8}{'IQR':>8}")
    for name, field in [("toàn chuỗi", "whole"), ("theo bước", "step")]:
        a = auc([r[field] for r in rows], [r["label"] for r in rows])
        by_q: dict[str, list] = defaultdict(list)
        for r in rows:
            by_q[r["qid"]].append(r)
        hit = tot = 0
        for rs in by_q.values():
            if len(rs) < 2 or sum(x["label"] for x in rs) != 1:
                continue            # chỉ xét câu có ĐÚNG một cặp khác chiến lược
            tot += 1
            hit += max(rs, key=lambda x: x[field])["label"] == 1
        sp = spread([r[field] for r in rows])
        rate = hit / tot if tot else None
        out[field] = {"auc": round(a, 4) if a is not None else None,
                      "top_pair_hit": hit, "top_pair_total": tot,
                      "top_pair_rate": round(rate, 4) if rate is not None else None, **sp}
        print(f"{name:<22}{a if a is None else round(a,3):>8}"
              f"{(f'{hit}/{tot} = {rate:.1%}' if rate is not None else '-'):>26}"
              f"{sp['p25']:>8.3f}{sp['median']:>10.3f}{sp['p75']:>8.3f}{sp['iqr']:>8.3f}")

    print("\nCách đọc: AUC 0,5 là đoán mò, càng gần 1 càng phân biệt tốt. Cột chọn đúng cặp khác nhất tương")
    print("ứng với con số 40% so với 53% của Feng. IQR nhỏ nghĩa là mọi cặp gần như cùng một khoảng cách,")
    print("tức cách đo không phân giải được. Nếu theo bước KHÔNG hơn rõ rệt thì giữ toàn chuỗi và trích số này.")
    write_json(wd / "embed_cmp.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

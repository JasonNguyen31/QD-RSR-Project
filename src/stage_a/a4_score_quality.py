"""a4_score_quality: chấm Qual(t) = alpha * rule_score(t) + (1 - alpha) * llm_score(t).

    python -m src.stage_a.a4_score_quality --workdir data/pilot/run2_boxed --rule-only
    python -m src.stage_a.a4_score_quality --workdir data/pilot/run2_boxed --limit 30
    python -m src.stage_a.a4_score_quality --workdir data/pilot/run2_boxed --judge-model gemini-3.5-flash-lite \
        --out judge_lite.jsonl --limit 30

Đọc  <workdir>/candidates.jsonl, <workdir>/questions.jsonl
Ghi  <workdir>/judge.jsonl    một dòng mỗi chuỗi, ghi ngay khi chấm xong (chạy lại là bỏ qua phần đã có)
     <workdir>/quality.jsonl  tid, rule_score, llm_score, qual  (tính lại từ đầu mỗi lần chạy, rất nhanh)

Chạy trên máy Mac: phần rule thuần CPU, phần giám khảo thuần gọi mạng.

Hai lưu ý về cách tính:
  - rule_score dùng z-score trên TOÀN BỘ tập nên phải chấm cả tập một lượt. Nếu chấm từng phần rồi ghép,
    kết quả sẽ khác. Vì vậy quality.jsonl luôn được tính lại toàn bộ, còn judge.jsonl mới là thứ ghi dần.
  - qual ở đây CHƯA chuẩn hoá về [0, 1]. Bước min-max theo từng câu hỏi diễn ra ở b2_select.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import JsonlWriter, load_done_keys, now_iso, read_jsonl, write_jsonl
from src.common.prompts import JUDGE_CRITERIA, build_judge_prompt
from src.common.rule_score import rule_scores

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_judge_json(text: str) -> dict:
    """Đọc JSON giám khảo trả về. Chịu được dấu ```json bọc ngoài và chữ thừa hai đầu."""
    s = _FENCE.sub("", (text or "").strip())
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i < 0 or j <= i:
            raise ValueError(f"Không tìm thấy JSON trong phản hồi: {s[:150]!r}") from None
        obj = json.loads(s[i:j + 1])

    if "overall_score" not in obj:
        raise ValueError(f"Thiếu overall_score. Các khoá có: {sorted(obj)}")
    score = float(obj["overall_score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"overall_score = {score}, ngoài khoảng [0, 1]")
    dims = obj.get("dimensional_evaluation") or {}
    missing = [c for c in JUDGE_CRITERIA if c not in dims]
    return {
        "overall_score": score,
        "overall_reason": str(obj.get("overall_reason", ""))[:1000],
        "dimensions": {c: float(dims[c]["score"]) for c in JUDGE_CRITERIA if c in dims and "score" in dims[c]},
        "missing_dimensions": missing,   # giữ lại để biết giám khảo có bỏ sót tiêu chí nào không
    }


class GeminiJudge:
    """Gọi Gemini, cùng cách gọi với test_api.py đã chạy được. Đếm lượt và thời gian để ước chi phí."""

    def __init__(self, model_id: str, api_key_env: str, max_attempts: int = 4):
        from google import genai  # nhập muộn để phần rule chạy được khi chưa cài thư viện

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Thiếu biến môi trường {api_key_env}. Đặt trong .env ở gốc repo.")
        self._client = genai.Client(api_key=key)
        self.model_id = model_id
        self.max_attempts = max_attempts
        self._lock = threading.Lock()
        self.ok = 0
        self.failed = 0
        self.seconds = 0.0
        self.errors: dict[str, int] = {}

    def score(self, prompt: str) -> dict:
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.monotonic()
            try:
                r = self._client.models.generate_content(model=self.model_id, contents=prompt)
                out = parse_judge_json(getattr(r, "text", "") or "")
                with self._lock:
                    self.ok += 1
                    self.seconds += time.monotonic() - t0
                return out
            except Exception as exc:  # noqa: BLE001
                last = exc
                with self._lock:
                    self.seconds += time.monotonic() - t0
                    name = type(exc).__name__
                    self.errors[name] = self.errors.get(name, 0) + 1
                if attempt < self.max_attempts:
                    time.sleep(min(2 ** attempt, 20))
        with self._lock:
            self.failed += 1
        raise last  # type: ignore[misc]


def run_judge(cfg: Mapping, cands: Sequence[Mapping], questions: Mapping[str, Mapping], workdir: Path,
              out_name: str, model_id: str, workers: int, use_reference: bool, show_progress: bool = True) -> dict:
    path = workdir / out_name
    done = load_done_keys(path, lambda r: r["tid"])
    todo = [c for c in cands if c["tid"] not in done]
    print(f"[a4] giám khảo {model_id}: {len(cands)} chuỗi, đã có {len(done)}, cần chấm {len(todo)}")
    if not todo:
        return {"judged": 0}

    judge = GeminiJudge(model_id, cfg["judge"]["api_key_env"])
    with JsonlWriter(path) as w:
        def work(c):
            q = questions[c["qid"]]
            prompt = build_judge_prompt(q["question"], c["text"],
                                        q.get("solution") if use_reference else None)
            try:
                res = judge.score(prompt)
            except Exception as exc:  # noqa: BLE001
                return c["tid"], None, f"{type(exc).__name__}: {str(exc)[:200]}"
            w.append({"tid": c["tid"], "qid": c["qid"], "model": model_id,
                      "use_reference": use_reference, **res, "ts": now_iso()})
            return c["tid"], res["overall_score"], None

        ex = ThreadPoolExecutor(max_workers=workers)
        futures = [ex.submit(work, c) for c in todo]
        bar = tqdm(as_completed(futures), total=len(futures), desc="chấm giám khảo") if show_progress \
            else as_completed(futures)
        failures = []
        try:
            for fut in bar:
                tid, score, err = fut.result()
                if err:
                    failures.append((tid, err))
        except KeyboardInterrupt:
            print("\n[a4] nhận Ctrl+C, dừng sau khi các lượt đang dở xong.", file=sys.stderr)
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    print(f"[a4] xong {judge.ok} lượt, lỗi {judge.failed}, trung bình {judge.seconds / max(judge.ok, 1):.1f}s mỗi lượt")
    if judge.errors:
        print(f"[a4] lỗi gặp phải: {judge.errors}")
    for tid, err in failures[:5]:
        print(f"      {tid}: {err}")
    if failures:
        print(f"[a4] {len(failures)} chuỗi chưa chấm được, chạy lại cùng lệnh để chấm bù.")
    return {"judged": judge.ok, "failed": judge.failed, "seconds": judge.seconds}


def combine(cands: Sequence[Mapping], judge_rows: Sequence[Mapping], alpha: float) -> list[dict]:
    """Ghép rule và judge. Chuỗi chưa có điểm giám khảo thì llm_score = None và qual = None."""
    scores, feats = rule_scores([c["text"] for c in cands])
    jmap = {r["tid"]: r["overall_score"] for r in judge_rows}
    out = []
    for c, rs, f in zip(cands, scores, feats):
        llm = jmap.get(c["tid"])
        out.append({
            "tid": c["tid"], "qid": c["qid"],
            "rule_score": round(rs, 6),
            "llm_score": llm,
            "qual": round(alpha * rs + (1 - alpha) * llm, 6) if llm is not None else None,
            "n_words": int(f["elaborated"]),
        })
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", help="mặc định paths.data_stage_a")
    ap.add_argument("--rule-only", action="store_true", help="chỉ tính điểm quy tắc, không gọi giám khảo")
    ap.add_argument("--limit", type=int, help="chỉ chấm N chuỗi đầu, dùng để thử và ước chi phí")
    ap.add_argument("--judge-model", help="mặc định judge.model_id; dùng để so với bản Flash-Lite")
    ap.add_argument("--out", default="judge.jsonl", help="tên file điểm giám khảo trong workdir")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-reference", action="store_true",
                    help="không đưa lời giải chuẩn vào prompt (bỏ reference-guided grading)")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.override)
    files = cfg["stage_a_files"]
    workdir: Path = resolve_path(cfg, args.workdir or cfg["paths"]["data_stage_a"])
    alpha = cfg["quality"]["alpha"]

    cands = read_jsonl(workdir / files["candidates"])
    questions = {q["qid"]: q for q in read_jsonl(workdir / files["questions"])}
    if not cands:
        raise SystemExit(f"Không có ứng viên trong {workdir / files['candidates']}. Chạy a3_filter trước.")
    missing = {c["qid"] for c in cands} - set(questions)
    if missing:
        raise SystemExit(f"{len(missing)} qid của ứng viên không có trong questions.jsonl, ví dụ {sorted(missing)[:3]}")

    subset = cands[: args.limit] if args.limit else cands
    if not args.rule_only:
        run_judge(cfg, subset, questions, workdir, args.out,
                  args.judge_model or cfg["judge"]["model_id"], args.workers, not args.no_reference)

    judge_rows = read_jsonl(workdir / args.out)
    rows = combine(cands, judge_rows, alpha)      # rule tính trên TOÀN BỘ ứng viên, không phải subset
    write_jsonl(workdir / "quality.jsonl", rows)

    have = [r for r in rows if r["qual"] is not None]
    rs = [r["rule_score"] for r in rows]
    print(f"\n[a4] đã ghi quality.jsonl: {len(rows)} chuỗi, {len(have)} chuỗi có đủ cả hai nguồn (alpha={alpha})")
    print(f"[a4] rule_score: nhỏ nhất {min(rs):.3f}, lớn nhất {max(rs):.3f}")
    if have:
        ls = [r["llm_score"] for r in have]
        print(f"[a4] llm_score : nhỏ nhất {min(ls):.2f}, lớn nhất {max(ls):.2f}, "
              f"trung bình {sum(ls) / len(ls):.3f}")
        print("[a4] nếu llm_score dồn hết vào một hai giá trị thì giám khảo không phân biệt được, "
              "cần xem lại prompt hoặc đổi mô hình.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

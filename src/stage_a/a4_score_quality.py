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

from src.common.api import make_openrouter_client
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
# Giám khảo hay viết công thức LaTeX (\frac, \boxed, \times) vào trong chuỗi JSON. Dấu gạch chéo ngược đó
# không phải ký tự thoát hợp lệ của JSON nên json.loads từ chối. Ký tự thoát hợp lệ: " \ / b f n r t u.
_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


_RE_OVERALL = re.compile(r'"overall_score"\s*:\s*([01](?:\.\d+)?)')
_RE_DIM = re.compile(r'"score"\s*:\s*([01](?:\.\d+)?)')


def salvage_score(s: str) -> tuple[float, str] | None:
    """Lớp cứu hộ cuối khi JSON hỏng hẳn: chỉ cần lấy được ĐIỂM, phần lý giải bỏ đi cũng không sao.

    Hai mức, thử lần lượt:
      1. tìm thẳng "overall_score" bằng biểu thức chính quy
      2. nếu phản hồi bị cắt trước khi tới overall_score, lấy trung bình các điểm tiêu chí đọc được
    Mức 2 KHÔNG giống cách giám khảo tự tổng hợp (họ không dùng trọng số đều), nên bản ghi được đánh dấu
    salvaged để về sau lọc ra kiểm tra hoặc loại khỏi phân tích nếu cần.
    """
    m = _RE_OVERALL.search(s)
    if m:
        return float(m.group(1)), "regex_overall"
    dims = [float(x) for x in _RE_DIM.findall(s)]
    if len(dims) >= 3:                      # ít nhất ba trên năm tiêu chí mới đủ tin
        return sum(dims) / len(dims), f"mean_of_{len(dims)}_dimensions"
    return None


def _between_braces(s: str) -> str | None:
    """Phần từ dấu ngoặc nhọn đầu tới dấu ngoặc nhọn cuối, để bỏ chữ thừa hai đầu phản hồi."""
    i, j = s.find("{"), s.rfind("}")
    return s[i:j + 1] if i >= 0 and j > i else None


def repair_json_escapes(s: str) -> str:
    """Nhân đôi các dấu gạch chéo ngược không hợp lệ, để LaTeX trong lý giải không làm hỏng cả lượt chấm."""
    return _BAD_ESCAPE.sub(r"\\\\", s)


def parse_judge_json(text: str) -> dict:
    """Đọc JSON giám khảo trả về. Chịu được dấu ```json bọc ngoài và chữ thừa hai đầu."""
    s = _FENCE.sub("", (text or "").strip())
    obj = None
    # strict=False cho phép ký tự điều khiển (xuống dòng thật) nằm trong chuỗi JSON. Giám khảo hay xuống dòng
    # giữa phần lý giải, và JSON chuẩn coi đó là lỗi. Nới chỗ này không ảnh hưởng tới việc đọc điểm số.
    for candidate in (s, repair_json_escapes(s)):
        for chunk in (candidate, _between_braces(candidate)):
            if chunk is None:
                continue
            try:
                obj = json.loads(chunk, strict=False)
                break
            except json.JSONDecodeError:
                continue
        if obj is not None:
            break
    if obj is None:
        rescued = salvage_score(s)
        if rescued is not None:
            score, how = rescued
            return {"overall_score": score, "overall_reason": "", "dimensions": {},
                    "missing_dimensions": list(JUDGE_CRITERIA), "salvaged": how}
        hint = (" Phản hồi có vẻ bị CẮT CỤT: nâng judge.max_tokens trong configs/models.yaml."
                if s.lstrip().startswith("{") and not s.rstrip().endswith("}") else "")
        raise ValueError(f"Không đọc được JSON ({len(s)} ký tự).{hint} Đầu phản hồi: {s[:120]!r}")

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


class OpenRouterJudge:
    """Giám khảo gọi qua OpenRouter, dùng lại ChatClient nên có sẵn thử lại, chờ tăng dần và đếm chi phí."""

    def __init__(self, cfg: Mapping, model_id: str):
        self.model_id = model_id
        self.jcfg = cfg["judge"]
        self._client = make_openrouter_client(cfg)
        self._lock = threading.Lock()
        self.ok = 0
        self.failed = 0
        self.seconds = 0.0
        self.cost = 0.0
        self.truncated = 0
        self.parse_retries = 0
        self.errors: dict[str, int] = {}

    def score(self, prompt: str) -> dict:
        t0 = time.monotonic()
        try:
            # Phản hồi JSON dở dang thường là sự cố tức thời phía nhà cung cấp: trên lô thử 200 chuỗi,
            # phần lớn lượt hỏng dài chỉ 400-800 ký tự, không phải do chạm trần token. Thử lại một lần.
            for attempt in (1, 2):
                res = self._client.chat(model=self.model_id, messages=[{"role": "user", "content": prompt}],
                                        temperature=self.jcfg.get("temperature", 0.0), top_p=1.0,
                                        max_tokens=self.jcfg.get("max_tokens", 3000))
                if res.finish_reason == "length":
                    with self._lock:
                        self.truncated += 1
                try:
                    out = parse_judge_json(res.text)
                    break
                except ValueError:
                    if attempt == 2:
                        raise
                    with self._lock:
                        self.parse_retries += 1
            with self._lock:
                self.ok += 1
                self.seconds += time.monotonic() - t0
                if res.cost:
                    self.cost += res.cost
            return out
        except Exception as exc:
            with self._lock:
                self.failed += 1
                self.seconds += time.monotonic() - t0
                n = type(exc).__name__
                self.errors[n] = self.errors.get(n, 0) + 1
            raise


class GeminiJudge:
    """Gọi Gemini, cùng cách gọi với test_api.py đã chạy được. Đếm lượt và thời gian để ước chi phí."""

    def __init__(self, model_id: str, api_key_env: str, max_attempts: int = 5):
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
                    # 429 của Google là hết hạn mức chứ không phải nghẽn tức thời, nên chờ lâu hơn hẳn
                    quota = "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc)
                    time.sleep(min(15 * attempt, 60) if quota else min(2 ** attempt, 20))
        with self._lock:
            self.failed += 1
        raise last  # type: ignore[misc]


def make_judge(cfg: Mapping, model_id: str):
    """Chọn đường gọi giám khảo theo judge.provider."""
    provider = cfg["judge"].get("provider", "openrouter")
    if provider == "openrouter":
        return OpenRouterJudge(cfg, model_id)
    if provider == "gemini":
        return GeminiJudge(model_id, cfg["judge"]["api_key_env"])
    raise SystemExit(f"judge.provider phải là openrouter hoặc gemini, nhận được {provider!r}")


def run_judge(cfg: Mapping, cands: Sequence[Mapping], questions: Mapping[str, Mapping], workdir: Path,
              out_name: str, model_id: str, workers: int, use_reference: bool, show_progress: bool = True,
              max_cost: float | None = None) -> dict:
    path = workdir / out_name
    done = load_done_keys(path, lambda r: r["tid"])
    todo = [c for c in cands if c["tid"] not in done]
    print(f"[a4] giám khảo {model_id}: {len(cands)} chuỗi, đã có {len(done)}, cần chấm {len(todo)}")
    if not todo:
        return {"judged": 0}

    judge = make_judge(cfg, model_id)
    with JsonlWriter(path) as w:
        budget_hit = threading.Event()

        def work(c):
            if budget_hit.is_set():
                return c["tid"], None, None
            if max_cost is not None and getattr(judge, "cost", 0.0) >= max_cost:
                budget_hit.set()
                return c["tid"], None, None
            q = questions[c["qid"]]
            prompt = build_judge_prompt(q["question"], c["text"],
                                        q.get("solution") if use_reference else None)
            try:
                res = judge.score(prompt)
            except Exception as exc:  # noqa: BLE001
                from src.stage_a.a2_generate import is_key_limit
                if is_key_limit(exc):
                    budget_hit.set()
                    print("\n[a4] DỪNG: khoá OpenRouter chạm hạn mức của khoá. Nâng Credit limit tại "
                          "https://openrouter.ai/settings/keys rồi chạy lại cùng lệnh.", file=sys.stderr)
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

    if max_cost is not None and budget_hit.is_set():
        print(f"[a4] ĐÃ DỪNG do chạm ngân sách --max-cost {max_cost} USD. Nâng mức rồi chạy lại để chấm tiếp.")
    salvaged = [r for r in read_jsonl(path) if r.get("salvaged")]
    if salvaged:
        print(f"[a4] {len(salvaged)} chuỗi phải dùng lớp cứu hộ để lấy điểm (JSON hỏng). Các bản ghi này có "
              f"trường 'salvaged' để lọc ra kiểm tra về sau.")
    if getattr(judge, "parse_retries", 0):
        print(f"[a4] {judge.parse_retries} lượt phải chấm lại vì JSON hỏng ở lần đầu.")
    if getattr(judge, "truncated", 0):
        print(f"[a4] {judge.truncated} phản hồi bị cắt ở judge.max_tokens. Nâng giá trị đó trong configs/models.yaml.")
    cost = getattr(judge, "cost", 0.0)
    print(f"[a4] xong {judge.ok} lượt, lỗi {judge.failed}, trung bình {judge.seconds / max(judge.ok, 1):.1f}s mỗi lượt"
          + (f", chi phí {cost:.4f} USD (ước cho 18.000 chuỗi: {cost / max(judge.ok, 1) * 18000:.1f} USD)"
             if cost else ""))
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
    ap.add_argument("--limit", type=int, help="chỉ chấm N chuỗi, dùng để thử và ước chi phí")
    ap.add_argument("--group", help="chỉ lấy chuỗi thuộc nhóm này, ví dụ math-L5. Lấy 30 chuỗi ĐẦU của "
                                    "candidates.jsonl sẽ dính toàn GSM8K dễ, chấm ra điểm cao là bình thường "
                                    "chứ không phải lạm phát điểm. Muốn đo giám khảo có phân biệt được hay không "
                                    "thì phải chấm trên nhóm khó.")
    ap.add_argument("--sample-seed", type=int, default=42, help="seed rút mẫu khi dùng --limit")
    ap.add_argument("--limit-questions", type=int,
                    help="rút N CÂU HỎI và chấm TOÀN BỘ chuỗi của chúng. Dùng khi cần so hai giám khảo: "
                         "việc chọn lọc xếp hạng các chuỗi TRONG CÙNG một câu hỏi, nên phải có nhiều chuỗi "
                         "cùng câu mới so được thứ hạng. --limit rút lẻ từng chuỗi nên không làm được việc này.")
    ap.add_argument("--judge-model", help="mặc định judge.model_id; dùng để so với bản Flash-Lite")
    ap.add_argument("--out", default="judge.jsonl", help="tên file điểm giám khảo trong workdir")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cost", type=float, help="dừng khi chi phí cộng dồn (USD) chạm mức này")
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

    subset = cands
    if args.group:
        from src.stage_a.a3_filter import group_of
        subset = [c for c in subset if group_of(questions[c["qid"]]) == args.group]
        groups = sorted({group_of(q) for q in questions.values()})
        if not subset:
            raise SystemExit(f"Không có chuỗi nào thuộc nhóm {args.group!r}. Các nhóm có: {groups}")
        print(f"[a4] lọc nhóm {args.group}: còn {len(subset)} chuỗi")
    if args.limit_questions:
        import random as _random
        qids = sorted({c["qid"] for c in subset})
        keep = set(_random.Random(args.sample_seed).sample(qids, min(args.limit_questions, len(qids))))
        subset = [c for c in subset if c["qid"] in keep]
        print(f"[a4] rút {len(keep)} câu hỏi, tổng {len(subset)} chuỗi (seed {args.sample_seed})")
    if args.limit and len(subset) > args.limit:
        import random as _random
        subset = _random.Random(args.sample_seed).sample(subset, args.limit)
        print(f"[a4] rút ngẫu nhiên {args.limit} chuỗi (seed {args.sample_seed})")
    if not args.rule_only:
        run_judge(cfg, subset, questions, workdir, args.out,
                  args.judge_model or cfg["judge"]["model_id"], args.workers, not args.no_reference,
                  max_cost=args.max_cost)

    judge_rows = read_jsonl(workdir / args.out)
    rows = combine(cands, judge_rows, alpha)      # rule tính trên TOÀN BỘ ứng viên, không phải subset
    write_jsonl(workdir / "quality.jsonl", rows)

    have = [r for r in rows if r["qual"] is not None]
    rs = [r["rule_score"] for r in rows]
    print(f"\n[a4] đã ghi quality.jsonl: {len(rows)} chuỗi, {len(have)} chuỗi có đủ cả hai nguồn (alpha={alpha})")
    print(f"[a4] rule_score: nhỏ nhất {min(rs):.3f}, lớn nhất {max(rs):.3f}")
    if have:
        from collections import Counter
        ls = [r["llm_score"] for r in have]
        hist = Counter(round(v, 1) for v in ls)
        print(f"[a4] llm_score : nhỏ nhất {min(ls):.2f}, lớn nhất {max(ls):.2f}, "
              f"trung bình {sum(ls) / len(ls):.3f}, số giá trị khác nhau {len(set(ls))}")
        print("[a4] phân bố: " + "  ".join(f"{k:.1f}:{n}" for k, n in sorted(hist.items())))
        top = hist.most_common(1)[0]
        if top[1] / len(ls) > 0.7:
            print(f"[a4] CẢNH BÁO: {top[1]}/{len(ls)} chuỗi cùng nhận điểm {top[0]:.1f}. Giám khảo gần như không "
                  f"phân biệt được, nên thành phần llm_score sẽ đóng góp rất ít vào Qual.")
            print("     Trước khi kết luận, hãy chấm lại trên nhóm khó bằng --group math-L5: điểm cao đồng loạt "
                  "trên GSM8K là hợp lý vì bài dễ, không phải lỗi giám khảo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

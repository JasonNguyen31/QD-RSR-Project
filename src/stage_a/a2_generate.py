"""a2_generate: sinh chuỗi suy luận từ các mô hình dạy qua OpenRouter (chạy trên máy Mac, thuần gọi mạng).

Từ thư mục gốc repo:

    python -m src.stage_a.a2_generate --workdir data/pilot/run2_boxed --dry-run
    python -m src.stage_a.a2_generate --workdir data/pilot/run2_boxed --limit-questions 2
    python -m src.stage_a.a2_generate --workdir data/stage_a --max-cost 12

Đọc  <workdir>/questions.jsonl
Ghi  <workdir>/trajectories.jsonl   mỗi chuỗi thành công một dòng, ghi ngay khi có
     <workdir>/failures.jsonl       lượt thất bại sau khi hết số lần thử (KHÔNG tính là đã làm)
     <workdir>/generation_runs.jsonl  một dòng mỗi lần chạy: mô hình, tham số, thời điểm (đưa vào paper)

Chạy lại đúng lệnh cũ sẽ bỏ qua chuỗi đã có (khoá tid) và chỉ sinh bù phần còn thiếu, kể cả các lượt từng lỗi.
"""
from __future__ import annotations

import argparse
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Sequence

from src.common.api import ChatClient, make_openrouter_client
from src.common.config import load_config, resolve_path
from src.common.io_utils import JsonlWriter, load_done_keys, now_iso, read_jsonl, traj_id
from src.common.prompts import build_generation_messages, describe_prompt

try:
    from tqdm import tqdm
except ImportError:  # tqdm không bắt buộc
    def tqdm(it, **_kw):  # type: ignore
        return it


def shard_names(files: Mapping, shard: str | None) -> tuple[str, str]:
    """Tên file chuỗi và file lỗi cho một lô.

    Chạy song song NHIỀU tiến trình cùng ghi vào một file jsonl là KHÔNG an toàn: một dòng chuỗi suy luận
    thường vượt 4096 byte nên hệ điều hành không đảm bảo ghi nối nguyên vẹn, các dòng có thể lồng vào nhau.
    Vì vậy mỗi tiến trình phải ghi ra file riêng qua --shard, rồi a3_filter đọc gộp tất cả.
    """
    if not shard:
        return files["trajectories"], files["failures"]
    return (files["trajectories"].replace(".jsonl", f".{shard}.jsonl"),
            files["failures"].replace(".jsonl", f".{shard}.jsonl"))


def all_trajectory_files(workdir: Path, files: Mapping) -> list[Path]:
    """File chuỗi chính cộng mọi file lô, để việc chạy bù thấy được phần các lô khác đã sinh."""
    base = files["trajectories"]
    stem = base.replace(".jsonl", "")
    return sorted(set([workdir / base] + list(workdir.glob(f"{stem}.*.jsonl"))))


def select_teachers(cfg: Mapping, keys: Sequence[str] | None) -> list[dict]:
    teachers = [dict(t) for t in cfg["teachers"]]
    if keys:
        unknown = set(keys) - {t["key"] for t in teachers}
        if unknown:
            raise SystemExit(f"Không có mô hình dạy {sorted(unknown)}. Hiện có: {[t['key'] for t in teachers]}")
        teachers = [t for t in teachers if t["key"] in keys]
    return teachers


def build_tasks(questions: Sequence[Mapping], teachers: Sequence[Mapping], n_samples: int, done: set[str]):
    """Mọi mẫu của một câu hỏi đứng liền nhau, nên nếu dừng giữa chừng (hết ngân sách) thì phần đã xong
    là các câu hỏi trọn vẹn chứ không phải một mẩu của mọi câu."""
    tasks = []
    for q in questions:
        for t in teachers:
            for i in range(n_samples):
                tid = traj_id(q["qid"], t["key"], i)
                if tid not in done:
                    tasks.append((q, t, i, tid))
    return tasks


def run_generation(
    cfg: Mapping,
    questions: Sequence[Mapping],
    teachers: Sequence[Mapping],
    client: ChatClient,
    workdir: Path,
    *,
    max_cost: float | None = None,
    workers: int | None = None,
    shard: str | None = None,
    show_progress: bool = True,
) -> dict:
    files = cfg["stage_a_files"]
    gen = cfg["generation"]
    prompt_id = gen["prompt_id"]
    n_samples = gen["samples_per_teacher"]
    workers = workers or gen["max_workers"]

    traj_name, fail_name = shard_names(files, shard)
    traj_path, fail_path = workdir / traj_name, workdir / fail_name
    done: set[str] = set()
    for p in all_trajectory_files(workdir, files):
        done |= load_done_keys(p, lambda r: r["tid"])
    tasks = build_tasks(questions, teachers, n_samples, done)

    print(f"[a2] {len(questions)} câu hỏi x {len(teachers)} mô hình dạy x {n_samples} mẫu = "
          f"{len(questions) * len(teachers) * n_samples} chuỗi cần có; đã có {len(done)} (tính cả các lô khác); "
          f"cần sinh {len(tasks)}")
    if shard:
        print(f"[a2] lô '{shard}': ghi vào {traj_name}")
    if not tasks:
        return {"tasks": 0}

    with JsonlWriter(workdir / files["runs"]) as runs_w:
        runs_w.append({
            "ts_start": now_iso(), "argv": sys.argv, "n_tasks": len(tasks), "workers": workers,
            "prompt_id": prompt_id, "prompt_shape": describe_prompt(prompt_id),
            "temperature": gen["temperature"], "top_p": gen["top_p"], "max_tokens": gen["max_tokens"],
            "teachers": [{"key": t["key"], "model_id": t["model_id"], "provider_order": t.get("provider_order") or []}
                         for t in teachers],
        })

    stop = threading.Event()
    budget_hit = threading.Event()
    per_teacher: dict[str, dict] = defaultdict(lambda: {"ok": 0, "failed": 0, "skipped": 0, "tokens": 0,
                                                        "latency": 0.0, "truncated": 0})
    warned_cost = False

    with JsonlWriter(traj_path) as traj_w, JsonlWriter(fail_path) as fail_w:

        def work(task):
            q, t, i, tid = task
            if stop.is_set():
                return t["key"], "skipped", None
            if max_cost is not None and client.tracker.cost >= max_cost:
                budget_hit.set()
                stop.set()
                return t["key"], "skipped", None
            try:
                res = client.chat(
                    model=t["model_id"],
                    messages=build_generation_messages(q["question"], prompt_id),
                    temperature=gen["temperature"], top_p=gen["top_p"], max_tokens=gen["max_tokens"],
                    provider_order=t.get("provider_order") or None,
                )
            except Exception as exc:  # noqa: BLE001 - mọi lỗi API đều ghi lại rồi đi tiếp
                fail_w.append({"tid": tid, "qid": q["qid"], "teacher": t["key"], "sample_idx": i,
                               "error_type": type(exc).__name__, "error": str(exc)[:300], "ts": now_iso()})
                return t["key"], "failed", None
            traj_w.append({
                "tid": tid, "qid": q["qid"], "teacher": t["key"], "sample_idx": i,
                "text": res.text, "finish_reason": res.finish_reason,
                "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                "cost": res.cost, "latency_s": round(res.latency_s, 2),
                "model_id": t["model_id"], "served_model": res.served_model,
                "prompt_id": prompt_id, "ts": now_iso(),
            })
            return t["key"], "ok", res

        ex = ThreadPoolExecutor(max_workers=workers)
        futures = [ex.submit(work, task) for task in tasks]
        bar = tqdm(as_completed(futures), total=len(futures), desc="sinh chuỗi", disable=not show_progress) \
            if show_progress else as_completed(futures)
        completed = 0
        try:
            for fut in bar:
                key, status, res = fut.result()
                s = per_teacher[key]
                s[status] += 1
                if res is not None:
                    s["tokens"] += res.completion_tokens
                    s["latency"] += res.latency_s
                    s["truncated"] += int(res.finish_reason == "length")
                completed += 1
                if (max_cost is not None and not warned_cost and completed >= 10
                        and client.tracker.cost_reports == 0):
                    warned_cost = True
                    print("\n[a2] CẢNH BÁO: đặt --max-cost nhưng OpenRouter chưa trả chi phí nên chốt ngân sách "
                          "chưa có tác dụng. Kiểm tra openrouter.request_usage_cost trong configs/models.yaml.",
                          file=sys.stderr)
        except KeyboardInterrupt:
            print("\n[a2] nhận Ctrl+C, đang dừng (chờ các lượt gọi đang dở xong)...", file=sys.stderr)
            stop.set()
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    summary = {"tasks": len(tasks), "per_teacher": {k: dict(v) for k, v in per_teacher.items()},
               "usage": client.tracker.snapshot(), "budget_hit": budget_hit.is_set()}
    print_summary(summary)
    return summary


def print_summary(summary: Mapping) -> None:
    print("\n[a2] Tổng kết")
    print(f"{'mô hình dạy':<12}{'ok':>7}{'lỗi':>7}{'bỏ qua':>8}{'token TB':>10}{'trễ TB(s)':>11}{'bị cắt':>8}")
    for key, s in summary["per_teacher"].items():
        n = max(s["ok"], 1)
        print(f"{key:<12}{s['ok']:>7}{s['failed']:>7}{s['skipped']:>8}{s['tokens'] / n:>10.0f}"
              f"{s['latency'] / n:>11.1f}{s['truncated']:>8}")
    u = summary["usage"]
    print(f"token vào {u['prompt_tokens']:,}, token ra {u['completion_tokens']:,}, "
          f"chi phí {u['cost_usd']} USD ({u['cost_reports']}/{u['ok']} lượt có báo chi phí)")
    if u["attempt_errors"]:
        print(f"lỗi trong các lần thử (kể cả lần đã thử lại thành công): {u['attempt_errors']}")
    if summary["budget_hit"]:
        print("ĐÃ DỪNG do chạm ngân sách --max-cost. Nâng mức rồi chạy lại cùng lệnh để sinh tiếp.")
    if any(s["failed"] for s in summary["per_teacher"].values()):
        print("Có lượt thất bại (xem failures.jsonl). Chạy lại cùng lệnh để sinh bù.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", help="thư mục làm việc, mặc định paths.data_stage_a")
    ap.add_argument("--teachers", nargs="*", help="chỉ chạy các mô hình dạy này (theo key), mặc định tất cả")
    ap.add_argument("--limit-questions", type=int, help="chỉ lấy N câu hỏi đầu, dùng để chạy thử")
    ap.add_argument("--max-cost", type=float, help="dừng khi chi phí cộng dồn (USD) chạm mức này")
    ap.add_argument("--workers", type=int, help="số luồng, mặc định generation.max_workers")
    ap.add_argument("--shard", help="tên lô, ví dụ deepseek. Ghi ra trajectories.<lô>.jsonl để chạy song song "
                                    "nhiều tiến trình an toàn. a3_filter tự đọc gộp mọi lô.")
    ap.add_argument("--override", action="append", default=[], help="ví dụ generation.temperature=0.6")
    ap.add_argument("--dry-run", action="store_true", help="chỉ đếm việc và in mẫu prompt, không gọi API")
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.override)
    workdir = resolve_path(cfg, args.workdir or cfg["paths"]["data_stage_a"])
    q_path = workdir / cfg["stage_a_files"]["questions"]
    questions = read_jsonl(q_path)
    if not questions:
        raise SystemExit(f"Không đọc được câu hỏi nào từ {q_path}. Chạy a1_prepare trước hoặc kiểm tra --workdir.")
    if args.limit_questions:
        questions = questions[: args.limit_questions]
    dup = len(questions) - len({q["qid"] for q in questions})
    if dup:
        raise SystemExit(f"{q_path} có {dup} qid trùng nhau, sửa trước khi sinh.")
    teachers = select_teachers(cfg, args.teachers)

    if args.dry_run:
        gen = cfg["generation"]
        done = set()
        for p in all_trajectory_files(workdir, cfg["stage_a_files"]):
            done |= load_done_keys(p, lambda r: r["tid"])
        tasks = build_tasks(questions, teachers, gen["samples_per_teacher"], done)
        print(f"[dry-run] workdir={workdir}\n[dry-run] {len(questions)} câu hỏi, mô hình dạy: "
              f"{[t['model_id'] for t in teachers]}\n[dry-run] đã có {len(done)} chuỗi, cần sinh {len(tasks)}")
        print("[dry-run] tin nhắn mẫu cho câu đầu tiên:")
        print(f"[dry-run] prompt_id={gen['prompt_id']}")
        for m in build_generation_messages(questions[0]["question"], gen["prompt_id"]):
            print(f"  [{m['role']}] {m['content'][:200]}")
        return 0

    client = make_openrouter_client(cfg)
    run_generation(cfg, questions, teachers, client, workdir, max_cost=args.max_cost, workers=args.workers,
                   shard=args.shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

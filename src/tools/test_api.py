"""Kiểm tra bốn mô hình còn gọi được. Chạy trước mỗi giai đoạn lớn (bài học ở Notion mục J2).

    python -m src.tools.test_api                 # ba mô hình dạy + giám khảo
    python -m src.tools.test_api --skip-judge    # chỉ ba mô hình dạy
    python -m src.tools.test_api --judge-compare # thêm mô hình so sánh (Flash-Lite)

Khác test_api.py cũ: ba mô hình dạy được gọi qua ĐÚNG lớp ChatClient và ĐÚNG system prompt \\boxed{} mà a2 sẽ dùng,
nên nếu lỗi thì lỗi ở đây chứ không phải giữa lúc sinh 18.000 chuỗi. Đồng thời in ra chi phí OpenRouter trả về
để xác nhận cờ request_usage_cost trong configs/models.yaml hoạt động.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from src.common.api import ChatClient, make_openrouter_client
from src.common.config import load_config
from src.common.prompts import build_generation_messages

# Không được thêm yêu cầu về định dạng vào câu hỏi: nó sẽ đè lên system prompt \\boxed{} và làm phép thử sai.
QUESTION = "What is 17 * 23?"


def check_teachers(cfg, client: ChatClient) -> int:
    client.max_attempts = 2  # kiểm tra nhanh, không cần kiên nhẫn
    bad = 0
    for t in cfg["teachers"]:
        try:
            res = client.chat(model=t["model_id"], messages=build_generation_messages(QUESTION, cfg["generation"]["prompt_id"]),
                              temperature=cfg["generation"]["temperature"], top_p=cfg["generation"]["top_p"],
                              max_tokens=400, provider_order=t.get("provider_order") or None)
            has_box = "\\boxed" in res.text
            cost = f"{res.cost:.6f} USD" if res.cost is not None else "không trả chi phí"
            print(f"[OK]  {t['key']:<10} {res.latency_s:>5.1f}s  {cost:<20} có \\boxed: {'có' if has_box else 'KHÔNG'}"
                  f"  -> {res.text.strip()[:50]!r}")
            if not has_box:
                print("      cảnh báo: mô hình không tuân theo prompt \\boxed{} với câu hỏi đơn giản này")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"[LỖI] {t['key']:<10} {type(exc).__name__}: {str(exc)[:160]}")
    snap = client.tracker.snapshot()
    if snap["ok"] and snap["cost_reports"] == 0:
        print("\nOpenRouter KHÔNG trả chi phí: --max-cost của a2 sẽ không có tác dụng. Thử đặt "
              "openrouter.request_usage_cost=false nếu các lượt trên toàn lỗi BadRequest, "
              "còn nếu chỉ thiếu chi phí thì cờ này chưa được nhà cung cấp hỗ trợ.")
    return bad


def check_judge(cfg, model_ids: Sequence[str]) -> int:
    from google import genai  # cùng cách gọi với test_api.py cũ đã chạy được

    env = cfg["judge"]["api_key_env"]
    key = os.environ.get(env)
    if not key:
        print(f"[LỖI] giám khảo: thiếu biến môi trường {env}")
        return len(model_ids)
    g = genai.Client(api_key=key)
    bad = 0
    for mid in model_ids:
        try:
            r = g.models.generate_content(model=mid, contents=QUESTION)
            print(f"[OK]  {mid:<24} -> {r.text.strip()[:50]!r}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"[LỖI] {mid:<24} {type(exc).__name__}: {str(exc)[:160]}")
    return bad


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--judge-compare", action="store_true", help="thử thêm judge.compare_model_id")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.override)
    bad = check_teachers(cfg, make_openrouter_client(cfg))
    if not args.skip_judge:
        ids = [cfg["judge"]["model_id"]] + ([cfg["judge"]["compare_model_id"]] if args.judge_compare else [])
        bad += check_judge(cfg, ids)
    print("\nTẤT CẢ ĐỀU GỌI ĐƯỢC" if bad == 0 else f"\nCÓ {bad} MÔ HÌNH LỖI")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

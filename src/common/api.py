"""Gọi mô hình dạy qua OpenRouter (giao diện tương thích OpenAI), có thử lại và đếm chi phí.

Sửa hai lỗi đã gặp ở mã pilot cũ (Notion mục R5):
  1. Phản hồi rỗng của Qwen2.5-72B (choices = None hoặc content = None) nay được coi là lỗi có thể thử lại,
     thay vì làm văng NoneType.
  2. Bộ đếm chỉ tăng SAU KHI phản hồi đã được kiểm tra hợp lệ. Mã cũ tăng trước khi đọc choices[0]
     nên lượt lỗi bị đếm nhầm là thành công.

Gemini (giám khảo) chưa có ở đây, thêm cùng a4_quality.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from json import JSONDecodeError

from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from tenacity import RetryCallState, Retrying, retry_if_exception, wait_random_exponential


class EmptyResponse(RuntimeError):
    """Nhà cung cấp trả 200 nhưng không có nội dung dùng được."""


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    cost: float | None       # đô la, None nếu OpenRouter không trả
    latency_s: float
    served_model: str | None


class UsageTracker:
    """Bộ đếm dùng chung cho mọi luồng."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ok = 0
        self.failed = 0                        # lượt gọi thất bại sau khi hết số lần thử
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0
        self.cost_reports = 0                  # số lượt có trả về chi phí
        self.attempt_errors: dict[str, int] = {}

    def add_ok(self, res: ChatResult) -> None:
        with self._lock:
            self.ok += 1
            self.prompt_tokens += res.prompt_tokens
            self.completion_tokens += res.completion_tokens
            if res.cost is not None:
                self.cost += res.cost
                self.cost_reports += 1

    def add_attempt_error(self, exc: BaseException) -> None:
        name = type(exc).__name__
        with self._lock:
            self.attempt_errors[name] = self.attempt_errors.get(name, 0) + 1

    def add_final_failure(self) -> None:
        with self._lock:
            self.failed += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "ok": self.ok,
                "failed": self.failed,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost, 4),
                "cost_reports": self.cost_reports,
                "attempt_errors": dict(self.attempt_errors),
            }


_NEVER_RETRY = (AuthenticationError, PermissionDeniedError, NotFoundError)
_RETRYABLE = (EmptyResponse, APIConnectionError, RateLimitError, InternalServerError, BadRequestError,
              JSONDecodeError, APIError)
# APITimeoutError là lớp con của APIConnectionError nên đã nằm trong nhóm trên.
# JSONDecodeError thêm 19/09 sau lô pilot run3: nhà cung cấp thỉnh thoảng trả về thân phản hồi không phải JSON
# hợp lệ, SDK ném thẳng JSONDecodeError. Trước đây lỗi này KHÔNG được thử lại nên mất luôn chuỗi (3 lượt ở run3).
# APIError là lớp cha của phần lớn lỗi SDK, đặt cuối để hứng các lỗi tạm thời chưa liệt kê ở trên.


def _is_retryable(exc: BaseException) -> bool:
    return not isinstance(exc, _NEVER_RETRY) and isinstance(exc, _RETRYABLE)


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float = 180,
        max_attempts: int = 6,
        request_usage_cost: bool = True,
        tracker: UsageTracker | None = None,
        sleep: Callable[[float], None] = time.sleep,
        _sdk_client: object | None = None,
    ):
        # max_retries=0: chỉ tenacity được thử lại, tránh SDK thử ngầm rồi nhân đôi số lượt gọi
        self._sdk = _sdk_client or OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        self.max_attempts = max_attempts
        self.request_usage_cost = request_usage_cost
        self.tracker = tracker or UsageTracker()
        self._sleep = sleep

    # ---- một lượt gọi, chưa thử lại
    def _once(self, model: str, messages: Sequence[Mapping], temperature: float, top_p: float,
              max_tokens: int, provider_order: Sequence[str] | None) -> ChatResult:
        extra: dict = {}
        if self.request_usage_cost:
            extra["usage"] = {"include": True}
        if provider_order:
            extra["provider"] = {"order": list(provider_order)}

        t0 = time.monotonic()
        r = self._sdk.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra or None,
        )
        latency = time.monotonic() - t0

        choices = getattr(r, "choices", None)
        if not choices:
            raise EmptyResponse(f"không có choices (error={getattr(r, 'error', None)!r})")
        choice = choices[0]
        msg = getattr(choice, "message", None)
        text = getattr(msg, "content", None) or ""
        if not text.strip():
            raise EmptyResponse(f"nội dung rỗng (finish_reason={getattr(choice, 'finish_reason', None)!r})")

        usage = getattr(r, "usage", None)
        cost = getattr(usage, "cost", None) if usage is not None else None
        return ChatResult(
            text=text,
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost=float(cost) if cost is not None else None,
            latency_s=latency,
            served_model=getattr(r, "model", None),
        )

    # ---- có thử lại
    def chat(self, model: str, messages: Sequence[Mapping], temperature: float, top_p: float,
             max_tokens: int, provider_order: Sequence[str] | None = None) -> ChatResult:
        max_attempts = self.max_attempts

        def stop(state: RetryCallState) -> bool:
            exc = state.outcome.exception() if state.outcome else None
            # BadRequest ở Qwen chỉ thoảng xảy ra, thử thêm một lần là đủ. Nếu vẫn lỗi thì để lượt chạy bù xử lý.
            limit = 2 if isinstance(exc, BadRequestError) else max_attempts
            return state.attempt_number >= limit

        retrying = Retrying(
            stop=stop,
            wait=wait_random_exponential(multiplier=1, max=30),
            retry=retry_if_exception(_is_retryable),
            sleep=self._sleep,
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    try:
                        result = self._once(model, messages, temperature, top_p, max_tokens, provider_order)
                    except BaseException as exc:
                        self.tracker.add_attempt_error(exc)
                        raise
            self.tracker.add_ok(result)  # chỉ đếm khi đã có kết quả hợp lệ
            return result
        except BaseException:
            self.tracker.add_final_failure()
            raise


def make_openrouter_client(cfg: Mapping, tracker: UsageTracker | None = None) -> ChatClient:
    env_name = cfg["openrouter"]["api_key_env"]
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"Thiếu biến môi trường {env_name}. Đặt khoá trong file .env ở gốc repo "
            f"(hoặc đổi tên biến ở configs/models.yaml). Không ghi khoá vào mã hay Notion."
        )
    gen = cfg["generation"]
    return ChatClient(
        base_url=cfg["openrouter"]["base_url"],
        api_key=key,
        timeout_s=gen["request_timeout_s"],
        max_attempts=gen["max_attempts"],
        request_usage_cost=cfg["openrouter"].get("request_usage_cost", True),
        tracker=tracker,
    )

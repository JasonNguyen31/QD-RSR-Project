"""
Prompt dùng chung. Sửa prompt ở đây, không sửa rải rác trong mã.

SYSTEM_PROMPT_BOXED lấy đúng câu của bài gốc RSR, đã xác nhận từ dữ liệu công khai Umean/RSR_data
(Notion mục P8, trong khối mã nên không có dấu chấm cuối). Đưa vào vai trò system, câu hỏi vào vai trò user.
Câu này thay cho prompt cũ yêu cầu "#### <answer>" (Notion mục R1).

Prompt giám khảo (JSON, 5 tiêu chí, kèm lời giải tham chiếu) thêm cùng lúc với a4_quality, sau khi đối chiếu
Table 28 của bài RSR.
"""
from __future__ import annotations

SYSTEM_PROMPT_BOXED = "Please reason step by step, and put your final answer within \\boxed{}"

SYSTEM_PROMPTS: dict[str, str] = {
    "boxed": SYSTEM_PROMPT_BOXED,
}


def get_system_prompt(prompt_id: str) -> str:
    try:
        return SYSTEM_PROMPTS[prompt_id]
    except KeyError:
        raise KeyError(f"Không có system prompt '{prompt_id}'. Hiện có: {sorted(SYSTEM_PROMPTS)}") from None


def build_generation_messages(question: str, prompt_id: str = "boxed") -> list[dict]:
    """Tin nhắn gửi cho mô hình dạy: system = câu \\boxed{}, user = nguyên văn đề bài."""
    return [
        {"role": "system", "content": get_system_prompt(prompt_id)},
        {"role": "user", "content": question},
    ]

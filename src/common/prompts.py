"""Prompt dùng chung. Sửa prompt ở đây, không sửa rải rác trong mã.

Bài RSR dùng HAI câu khác nhau ở hai giai đoạn khác nhau (Phụ lục A.2), và đề tài theo đúng như vậy:

  1. LÚC SINH CHUỖI (a2_generate) - prompt_id "rsr":
     câu GEN_INSTRUCTION_RSR được nối vào cuối đề bài, ở vai trò user, KHÔNG có system prompt.
     Nguyên văn A.2: mỗi bài toán được nối thêm chỉ dẫn "Return your final response within \\boxed{}."

  2. LÚC HUẤN LUYỆN VÀ ĐÁNH GIÁ (b3_train, c1_evaluate):
     câu SYSTEM_PROMPT_TRAIN làm system prompt cho mọi mẫu.
     Nguyên văn A.2: system prompt cố định "Please reason step by step, and put your final answer within \\boxed{}."

prompt_id "boxed" giữ lại cách cũ (câu huấn luyện đặt ở vai trò system lúc sinh), dùng cho lô pilot đã sinh
trước ngày đổi. Mỗi chuỗi sinh ra đều ghi prompt_id của nó, nên hai lô không lẫn vào nhau.

Bản PDF trích từ Notion hiển thị hai câu này thành "\\{\\}" do lệnh \\boxed bị mất khi trích văn bản; nội dung
ở đây khôi phục theo ngữ cảnh và theo kho dữ liệu công khai của bài gốc. Nếu đối chiếu PDF gốc thấy khác thì sửa tại đây.
"""
from __future__ import annotations

GEN_INSTRUCTION_RSR = "Return your final response within \\boxed{}."
SYSTEM_PROMPT_TRAIN = "Please reason step by step, and put your final answer within \\boxed{}"

# Cách ngăn giữa đề bài và chỉ dẫn. Bài gốc chỉ nói "nối vào", không nói rõ ký tự ngăn.
GEN_SEPARATOR = "\n\n"


def build_generation_messages(question: str, prompt_id: str = "rsr") -> list[dict]:
    """Tin nhắn gửi mô hình dạy.

    "rsr"   : [user] <đề bài>\\n\\nReturn your final response within \\boxed{}.
    "boxed" : [system] Please reason step by step...  +  [user] <đề bài>   (cách cũ, giữ cho lô pilot cũ)
    """
    if prompt_id == "rsr":
        return [{"role": "user", "content": question.rstrip() + GEN_SEPARATOR + GEN_INSTRUCTION_RSR}]
    if prompt_id == "boxed":
        return [{"role": "system", "content": SYSTEM_PROMPT_TRAIN},
                {"role": "user", "content": question}]
    raise KeyError(f"Không có prompt sinh chuỗi '{prompt_id}'. Hiện có: ['rsr', 'boxed']")


def describe_prompt(prompt_id: str) -> str:
    """Mô tả một dòng, ghi vào generation_runs.jsonl để tra lại sau."""
    msgs = build_generation_messages("<QUESTION>", prompt_id)
    return " | ".join(f"[{m['role']}] {m['content']}" for m in msgs)


# ---------------------------------------------------------------- Prompt giám khảo
# Nguyên văn Bảng 30 của bài RSR, giữ nguyên tiếng Anh và giữ nguyên hai đoạn quan trọng:
# bảng neo điểm năm mức, và đoạn chống lạm phát điểm. Hai chỗ đề tài thêm so với bản gốc:
#   - {reference_solution}: lời giải chuẩn của bộ dữ liệu, theo reference-guided grading của GenRM.
#     Bài gốc không có phần này vì dữ liệu của họ không kèm lời giải chuẩn.
#   - khối REFERENCE bị bỏ hẳn khi câu hỏi không có lời giải chuẩn (xem build_judge_prompt).
# Đầu ra là JSON: dimensional_evaluation (5 tiêu chí, mỗi tiêu chí có score và reason),
# overall_score, overall_reason. llm_score(t) lấy chính overall_score.

JUDGE_CRITERIA = ["factual_accuracy", "logical_rigor", "solution_completeness",
                  "reasoning_efficiency", "presentation_quality"]

JUDGE_PROMPT = """You are a meticulous and highly critical evaluator of AI reasoning. Your primary goal is to identify and quantify subtle flaws, logical gaps, inefficiencies, and hidden assumptions. Do not default to a high score. Your starting assumption should be critical, and you must rigorously justify every point awarded.

First, please carefully read the following problem statement:

{question}
{reference_block}
Now, please carefully read the following candidate's chain-of-thought reasoning:

{reasoning_to_evaluate}

When evaluating this reasoning, you must adhere to the following five key evaluation criteria and the scoring rubric below.

Scoring Guidelines and Calibration:
You must use the full 0.0 to 1.0 scale. Scores should not be clustered at the top. Use this rubric to anchor your scores:
1.0 (Exceptional/Flawless): Reserved for reasoning that is not only correct but also elegant, insightful, and comprehensive. It is perfectly structured and leaves no room for doubt. This score should be exceedingly rare.
0.8-0.9 (Excellent but Imperfect): The core reasoning is valid and well-supported, but there may be very minor, superficial issues (e.g., a trivial typo in a formula that doesn't affect the outcome, a slightly awkward phrasing). The conclusion is unaffected.
0.5-0.7 (Competent but Flawed): The reasoning is generally on the right track but contains noticeable and non-trivial flaws. Examples include: a minor factual error, a logical leap that requires the reader to fill in the blanks, an inefficient method where a much simpler one exists, or a partially incomplete answer.
0.2-0.4 (Poor): The reasoning contains fundamental flaws that largely invalidate the process or conclusion. Examples include: a significant factual error, a clear logical fallacy, misunderstanding of the core problem constraints.
0.0-0.1 (Unacceptable): The reasoning is completely incorrect, irrelevant, nonsensical, or makes no meaningful attempt to solve the problem.

Crucial Instruction for High Scores:
To combat score inflation, you must justify high scores with the same rigor as low scores. For any criterion where you assign a score of 0.9 or 1.0, your justification must explicitly state what makes the reasoning exceptional and why it lacks even subtle flaws.

Evaluation Criteria:
Factual Accuracy:
Scrutinize every claim, formula, and piece of domain knowledge. Is it precisely correct? Assess the application of problem constraints, paying close attention to edge cases and boundary conditions. Penalize any inaccuracy, no matter how small.
Logical Rigor:
Probe for hidden assumptions and unstated premises. Does each conclusion necessarily and unambiguously follow from the preceding steps? Identify any logical fallacies, contradictions, or jumps in reasoning. A chain is only as strong as its weakest link.
Solution Completeness:
Does the reasoning address all parts of the problem statement exhaustively? Does it consider all possible cases, sub-problems, and nuances? An answer that is correct for one case but ignores others is incomplete.
Reasoning Efficiency:
Is this the most direct and economical path to the solution? Penalize any unnecessary complexity, redundant steps, or exploration of irrelevant tangents, even if they eventually lead to the correct answer. The cognitive effort should be proportionate to the problem's complexity.
Presentation Quality:
How clearly is the reasoning communicated? Is the structure logical and easy to follow? Ambiguous language, poor organization, or a confusing sequence of steps should be penalized. An observer should be able to verify the reasoning process without difficulty.

For each of the five evaluation criteria, please give a score from 0.0 to 1.0 (in 0.1 increments) and a brief, clear justification for that score.

Your output must be a single, valid JSON object with no surrounding text and no markdown code fences. The format of the JSON object is as follows:
{{
    "dimensional_evaluation": {{
        "factual_accuracy": {{"score": <float between 0.0 and 1.0>, "reason": "<Your justification>"}},
        "logical_rigor": {{"score": <float between 0.0 and 1.0>, "reason": "<Your justification>"}},
        "solution_completeness": {{"score": <float between 0.0 and 1.0>, "reason": "<Your justification>"}},
        "reasoning_efficiency": {{"score": <float between 0.0 and 1.0>, "reason": "<Your justification>"}},
        "presentation_quality": {{"score": <float between 0.0 and 1.0>, "reason": "<Your justification>"}}
    }},
    "overall_score": <float between 0.0 and 1.0>,
    "overall_reason": "<A concise summary justifying the overall score by synthesizing the key findings from the dimensional evaluation.>"
}}"""

REFERENCE_BLOCK = """
For your reference, here is a correct solution to the problem. The candidate may legitimately reach the same answer by a different route, so do not penalize a different but valid approach:

{reference_solution}
"""


def build_judge_prompt(question: str, trajectory: str, reference_solution: str | None = None) -> str:
    """Prompt giám khảo. Bỏ hẳn khối tham chiếu khi không có lời giải chuẩn, thay vì để chỗ trống."""
    block = REFERENCE_BLOCK.format(reference_solution=reference_solution.strip()) if reference_solution else ""
    return JUDGE_PROMPT.format(question=question.strip(), reference_block=block,
                               reasoning_to_evaluate=trajectory.strip())


# ---------------------------------------------------------------- Prompt cho hai phép đo ở R2
# Dùng cho src/tools/compare_embedding.py, tái lập cách đo của Feng 2025 (RPD) ở quy mô nhỏ.

SUMMARIZE_STEPS_PROMPT = """Summarize the mathematical solution below as a numbered list of 3 to 5 high-level steps.
Each step states WHAT the solution does at that stage (the method or idea), not the arithmetic.
Write one short sentence per step. Output only the numbered list, nothing else.

SOLUTION:
{trajectory}"""

STRATEGY_LABEL_PROMPT = """You compare two solutions to the same mathematics problem and decide whether they follow
the SAME overall solution strategy or DIFFERENT strategies.

Same strategy means the two solutions take the same route to the answer: the same key idea, the same setup, the same
sequence of transformations. Cosmetic differences do not matter: wording, notation, how much arithmetic is shown,
the order of independent checks, or one solution verifying its answer while the other does not.

Different strategy means the route itself differs: for example one sets up an equation while the other tests cases,
one works forward from the givens while the other works backward from the answer, one is algebraic and the other
geometric or combinatorial.

PROBLEM:
{question}

SOLUTION A:
{a}

SOLUTION B:
{b}

Answer with a single JSON object and nothing else:
{{"verdict": "same" or "different", "reason": "<one short sentence>"}}"""


def build_summarize_prompt(trajectory: str) -> str:
    return SUMMARIZE_STEPS_PROMPT.format(trajectory=trajectory.strip())


def build_strategy_prompt(question: str, a: str, b: str) -> str:
    return STRATEGY_LABEL_PROMPT.format(question=question.strip(), a=a.strip(), b=b.strip())

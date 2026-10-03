"""Phần toán thuần của các tín hiệu nội tại, không cần torch. Đối chiếu bài gốc ngày 29/09/2026.

LARK (Yu và cộng sự, arXiv 2605.30651, mục 4.2 đến 4.3, Thuật toán 1; mã chính thức lark/score_lark.py
và lark/selection/select_chi2.py):

    ℓ_k      entropy chéo trung bình từng token của chuỗi k            (= mean_surprisal của b1)
    Brier_k  điểm Brier trung bình từng token, ‖π_t − δ(y_t)‖²          (= brier của b1)
    ρ̂_k     = Brier_k / ℓ_k
    ĝ_k     = ℓ_k / Σ_i ℓ_i · ( 2·ρ̂_k − Σ_i ρ̂_i·ℓ_i / Σ_i ℓ_i )       tổng lấy trên các ứng viên CÙNG câu hỏi

    Điểm để xếp hạng là ĝ_k, không phải Brier. ĝ_k phụ thuộc cả nhóm ứng viên của câu hỏi, nên b1 chỉ ghi
    Brier_k và ℓ_k, còn ĝ_k tính sau khi đã có đủ nhóm (hàm add_lark).

    Trọng số mềm khi chọn B chuỗi: q_i = (ĝ_i − ĝ_{B+1}) / Σ_{j≤B} (ĝ_j − ĝ_{B+1}) cho B chuỗi điểm cao nhất,
    còn lại bằng 0 (Bổ đề 3). Hàm lark_weights.

Local Naturalness (Just và cộng sự, arXiv 2510.03988, phương trình 2):

    Chia chuỗi thành các bước s_1..s_p (mỗi bước là một câu). Điểm của bước i là log xác suất trung bình
    từng token của s_i, khi mô hình chỉ thấy đề bài x và tối đa k bước liền trước (k = 4 như bài gốc và
    như LARK). Điểm của chuỗi là trung bình CỘNG THEO BƯỚC, không phải theo token.
    Hàm sentence_ends, token_steps, local_items dựng các bước; phần chạy mô hình nằm ở b1_fit.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.common.io_utils import split_tid

SIGNALS_VERSION = 2   # 1: LARK là Brier trung bình, LocalNat là khối 256 token. 2: đúng công thức bài gốc.


# ============================================================ LARK
def lark_ghat(ce: Sequence[float], brier: Sequence[float]) -> tuple[list[float], list[float]]:
    """ρ̂ và ĝ cho các ứng viên của MỘT câu hỏi. Chép đúng compute_g_hat của mã LARK chính thức."""
    if len(ce) != len(brier) or not ce:
        raise ValueError(f"ce và brier phải cùng độ dài và khác rỗng: {len(ce)}, {len(brier)}")
    rho = [b / l if l > 1e-30 else 0.0 for b, l in zip(brier, ce)]
    sum_l = sum(ce)
    if sum_l <= 1e-30:
        return rho, [float("nan")] * len(ce)
    sum_rho_l = sum(r * l for r, l in zip(rho, ce))
    g = [(l / sum_l) * (2.0 * r - sum_rho_l / sum_l) for r, l in zip(rho, ce)]
    return rho, g


def add_lark(rows: Iterable[MutableMapping]) -> list[MutableMapping]:
    """Ghi thêm lark_rho và lark (= ĝ) vào từng dòng của file fit, nhóm theo câu hỏi.

    Nhóm là toàn bộ ứng viên của câu hỏi có trong file (tức các chuỗi đúng mà b1 đã chấm). Nếu một phương
    án chọn từ nhóm khác thì phải gọi lại hàm này trên đúng nhóm đó.
    """
    rows = list(rows)
    by_q: dict[str, list] = defaultdict(list)
    for r in rows:
        if "brier" not in r or "mean_surprisal" not in r:
            raise ValueError("Dòng thiếu brier hoặc mean_surprisal: file fit tạo trước 29/09, phải chạy lại b1_fit.")
        by_q[r.get("qid") or split_tid(r["tid"])[0]].append(r)
    for group in by_q.values():
        rho, g = lark_ghat([float(r["mean_surprisal"]) for r in group], [float(r["brier"]) for r in group])
        for r, a, b in zip(group, rho, g):
            r["lark_rho"] = a
            r["lark"] = b
    return rows


def lark_weights(scores: Sequence[float], budget: int) -> list[float]:
    """Trọng số mềm của LARK cho B = budget chuỗi, theo Bổ đề 3 và select_chi2.py.

    Hai trường hợp biên, bài gốc không nói:
      - Câu hỏi chỉ có đúng B ứng viên (không có ĝ_{B+1}): trả trọng số đều 1/B, tức không chọn lọc.
        Mã chính thức đặt ngưỡng sát dưới điểm nhỏ nhất nên chuỗi cuối nhận trọng số gần 0; nhóm KHÔNG theo
        chỗ này vì nó làm mất một chuỗi, trái với ngân sách B. Ghi rõ trong paper.
      - Mọi chuỗi được chọn cùng điểm với ngưỡng: trọng số đều, như mã chính thức.
    Điểm NaN không bao giờ được chọn.
    """
    if budget < 1:
        raise ValueError("budget phải từ 1 trở lên")
    k = len(scores)
    clean = [s if s == s else float("-inf") for s in scores]
    order = sorted((i for i in range(k) if clean[i] != float("-inf")), key=lambda i: clean[i], reverse=True)
    if not order:
        return [0.0] * k
    b = min(budget, len(order))
    top = order[:b]
    w = [0.0] * k
    if b == len(order):
        for i in top:
            w[i] = 1.0 / b
        return w
    thr = clean[order[b]]
    denom = sum(clean[i] - thr for i in top)
    if denom <= 1e-12:
        for i in top:
            w[i] = 1.0 / b
        return w
    for i in top:
        w[i] = (clean[i] - thr) / denom
    return w


# ============================================================ Local Naturalness
# Ranh giới câu, hai loại:
#   - ngay sau . ! ? khi theo sau là dấu cách rồi tới chữ trên cùng dòng. Ranh giới đặt TRƯỚC dấu cách vì
#     token của Qwen mang dấu cách ở đầu (" The"), nên chữ đầu câu sau rơi đúng vào câu sau;
#   - sau một dãy dấu xuống dòng (dãy đó thuộc câu trước).
# Số thập phân như 3.5 không bị cắt vì sau dấu chấm không có dấu cách.
_PUNCT_END = re.compile(r"[.!?](?=[ \t]+\S)")
_NEWLINES = re.compile(r"\n+")
MIN_STEP_CHARS = 4    # mảnh ít hơn 4 ký tự khác trắng ("1.", "a)", "Ok.") gộp vào câu kế tiếp


def sentence_ends(text: str) -> list[int]:
    """Vị trí ký tự kết thúc (không bao gồm) của từng câu. Vị trí cuối cùng luôn là len(text)."""
    if not text:
        return []
    raw = {m.end() for m in _PUNCT_END.finditer(text)} | {m.end() for m in _NEWLINES.finditer(text)}
    raw = sorted(e for e in raw if 0 < e < len(text)) + [len(text)]
    ends, start = [], 0
    for e in raw:
        if len("".join(text[start:e].split())) >= MIN_STEP_CHARS:
            ends.append(e)
            start = e
    if not ends or ends[-1] != len(text):
        if ends:
            ends[-1] = len(text)      # phần đuôi quá ngắn gộp vào câu cuối
        else:
            ends = [len(text)]
    return ends


def token_steps(offsets: Sequence[tuple[int, int]], text: str,
                ends: Sequence[int] | None = None) -> list[tuple[int, int]]:
    """Chia token của chuỗi thành các bước: danh sách (token đầu, token cuối không bao gồm).

    Mặc định mỗi bước là một câu (sentence_ends). Truyền ends để dùng mốc bước có sẵn, ví dụ các bước do
    GLM-4.5-Air cắt (src/tools/segment_steps): ends là vị trí ký tự kết thúc của từng bước, tăng dần, mốc cuối
    bằng len(text).

    Token thuộc bước chứa ký tự đầu tiên của nó. Các bước liền nhau và phủ kín mọi token, nên tổng số token
    của các bước đúng bằng số token của chuỗi.
    """
    n = len(offsets)
    if n == 0:
        return []
    if ends is None:
        ends = sentence_ends(text)
    else:
        ends = list(ends)
        if not ends or ends[-1] != len(text) or any(b <= a for a, b in zip(ends, ends[1:])) or ends[0] <= 0:
            raise ValueError("ends phải tăng dần, dương, và mốc cuối bằng len(text)")
    steps, start, j = [], 0, 0
    for t, (a, _b) in enumerate(offsets):
        while j < len(ends) - 1 and a >= ends[j]:
            if t > start:
                steps.append((start, t))
                start = t
            j += 1
    steps.append((start, n))
    return steps


def local_items(steps: Sequence[tuple[int, int]], k: int) -> list[tuple[int, int, int]]:
    """Mỗi bước thành một mục (ngữ cảnh bắt đầu, bước bắt đầu, bước kết thúc), theo chỉ số token của chuỗi.

    Ngữ cảnh của bước i là k bước liền trước (ít hơn nếu i < k). Đề bài luôn được đặt phía trước, b1_fit
    ghép vào khi chạy mô hình.
    """
    if k < 0:
        raise ValueError("k phải không âm")
    return [(steps[max(0, i - k)][0], s, e) for i, (s, e) in enumerate(steps)]


def mean_over_steps(step_means: Sequence[float]) -> float:
    """Điểm Local Naturalness của chuỗi: trung bình theo bước, mỗi bước nặng như nhau bất kể dài ngắn."""
    if not step_means:
        raise ValueError("chuỗi không có bước nào")
    return sum(step_means) / len(step_means)


# ============================================================ dùng chung cho công cụ đọc file fit
def prepare_fit_rows(rows: Sequence[Mapping]) -> tuple[list[dict], list[str]]:
    """Chuẩn bị dòng của file fit để so sánh. Trả về (dòng, các tín hiệu đã bỏ).

    File theo định nghĩa mới: tính ĝ của LARK theo câu hỏi. File cũ (trước 29/09): bỏ lark và local_nat vì
    hai trường đó mang định nghĩa sai, để không ai vô tình đưa chúng vào ma trận tương quan hay paper.
    """
    rows = [dict(r) for r in rows]
    if not rows:
        return rows, []
    if int(rows[0].get("signals_version", 1)) >= SIGNALS_VERSION:
        return add_lark(rows), []
    stale = ["lark", "local_nat"]
    return [{k: v for k, v in r.items() if k not in stale} for r in rows], stale

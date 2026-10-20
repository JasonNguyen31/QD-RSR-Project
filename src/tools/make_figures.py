"""Vẽ mọi hình cho paper. Một file, mỗi hình một hàm.

    python -m src.tools.make_figures pipeline
    python -m src.tools.make_figures distribution
    python -m src.tools.make_figures distribution --stats data/stage_a/stats.json
    python -m src.tools.make_figures all

Chạy từ gốc repo. Hình xuất ra outputs/figures/ dạng PDF (vector, dùng cho Overleaf) và PNG (xem nhanh).
Không để lại file SVG trung gian.

    Hình 1  pipeline      Pipeline ba giai đoạn, đường nối chỉ rõ ô nào sang ô nào
    Hình 2  distribution  Phân bố số chuỗi đúng mỗi câu theo độ khó. Biện minh cho việc đổi dữ liệu
    Hình 3  selection     Ví dụ chọn lọc trên một câu hỏi. CHƯA LÀM: cần điểm Fit, Qual, Div thật

Hình KẾT QUẢ, đọc kết quả đánh giá (c1_evaluate) và select.<tên>.json (b2_select), gọi gộp bằng `results`:

    Paper  teacher_share     Acc@4 trên MATH-500 theo tỷ lệ chuỗi Qwen2.5-72B của tập huấn luyện (tiếng Anh)
    C-2    teacher_share_vi  Cùng hình, chữ tiếng Việt cho báo cáo
    C-1    forest            Chênh lệch so với Correct-Only kèm khoảng tin cậy 95% ghép cặp (tiếng Việt)

    python -m src.tools.make_figures results --seeds 42 43

Hình cho BÁO CÁO ĐỒ ÁN (Notion mục T4), đọc thẳng từ dữ liệu thật, gọi gộp bằng `report`:

    A-2  teachers      Tỷ lệ đúng theo mô hình dạy và mức độ khó
    A-3  truncation    Tỷ lệ chuỗi bị cắt, cho thấy nó dồn vào một mô hình dạy ở bài khó
    A-4  lengths       Phân bố độ dài chuỗi đúng, liên quan tới phương án đối chứng Token-Length
    A-6  judge         Phân bố điểm giám khảo, kèm tương quan giữa hai nửa của điểm chất lượng
    A-8  distance      Khoảng cách biểu diễn: cùng và khác mô hình dạy, và theo mức độ khó
    B-1  correlation   Ma trận tương quan giữa bốn tín hiệu nội tại và độ dài

Cần thư viện hệ thống cairo để xuất file:
    conda install -c conda-forge cairo      (hoặc: brew install cairo)
    pip install cairosvg
Phần dựng SVG không cần cairo, nên bài kiểm thử chạy được ở máy nào cũng được.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from collections import Counter, defaultdict

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl
from src.stage_a.a3_filter import group_of

FONT = "Helvetica, Arial, sans-serif"
BLUE, BLUE_BG, BLUE_TX = "#3b8ae0", "#e6f0fb", "#0f3d7a"
PUR, PUR_BG, PUR_TX = "#7b6fe0", "#eeecfc", "#2e2682"
GRN, GRN_BG, GRN_TX = "#1f9b72", "#e2f4ee", "#0b4a36"
AMB = "#c98a1b"
GREY, INK, DIM = "#8a8783", "#111111", "#555555"


# ================================================================ dụng cụ chung
def text_width(txt: str, size: float, bold: bool = False) -> float:
    """Độ rộng chữ theo đúng font cairo sẽ vẽ. Không có cairo thì ước lượng, đủ dùng cho bài kiểm thử."""
    try:
        import cairocffi as cairo

        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
        ctx = cairo.Context(surf)
        ctx.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(size)
        return ctx.text_extents(txt)[4]
    except (ImportError, OSError):
        return len(txt) * size * (0.58 if bold else 0.52)


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.parts: list[str] = [f'<rect width="{w}" height="{h}" fill="#ffffff"/>']
        self.defs: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def marker(self, mid: str, color: str) -> None:
        self.defs.append(
            f'<marker id="{mid}" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="12" markerHeight="12" '
            f'orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M1,1 L10,6 L1,11" fill="none" '
            f'stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></marker>')

    def text(self, x, y, s, size=34, color=INK, anchor="start", weight="400") -> None:
        self.add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{color}">{s}</text>')

    def line(self, x1, y1, x2, y2, color, width=3, dashed=False, marker=None) -> None:
        dash = ' stroke-dasharray="16 11"' if dashed else ""
        end = f' marker-end="url(#{marker})"' if marker else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash}{end}/>')

    def poly(self, points, color, marker, dashed=False, width=4) -> None:
        d = "M" + " L".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="16 11"' if dashed else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash} '
                 f'stroke-linejoin="round" marker-end="url(#{marker})"/>')

    def render(self) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}"><defs>{"".join(self.defs)}</defs>{"".join(self.parts)}</svg>')


def save(svg_text: str, name: str, outdir: Path) -> list[Path]:
    """Xuất PDF và PNG thẳng từ chuỗi SVG, không ghi file SVG ra đĩa."""
    try:
        import cairosvg
    except (ImportError, OSError) as exc:
        raise SystemExit(f"Không nạp được cairosvg ({exc}). Cài: conda install -c conda-forge cairo "
                         f"&& pip install cairosvg") from None
    outdir.mkdir(parents=True, exist_ok=True)
    pdf, png = outdir / f"{name}.pdf", outdir / f"{name}.png"
    data = svg_text.encode("utf-8")
    cairosvg.svg2pdf(bytestring=data, write_to=str(pdf))
    cairosvg.svg2png(bytestring=data, write_to=str(png))
    return [pdf, png]


# ================================================================ Hình 1: pipeline
def build_pipeline() -> str:
    s = Svg(2040, 1880)
    for mid, c in [("m-grey", GREY), ("m-blue", BLUE), ("m-pur", PUR), ("m-grn", GRN)]:
        s.marker(mid, c)

    def frame(x, y, w, h, color):
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="40" fill="none" stroke="{color}" '
              f'stroke-width="4" stroke-dasharray="22 14"/>')

    def header(x, y, bold, rest, anchor="start"):
        gap = 30
        wb, wr = text_width(bold, 40, True), text_width(rest, 40)
        x0 = x if anchor == "start" else x - (wb + gap + wr)
        s.text(x0, y, bold, 40, INK, weight="700")
        s.text(x0 + wb + gap, y, rest, 40, DIM)

    def box(cx, y, w, h, bg, stroke, tx, l1, l2):
        s.add(f'<rect x="{cx - w / 2}" y="{y}" width="{w}" height="{h}" rx="24" fill="{bg}" '
              f'stroke="{stroke}" stroke-width="2.4"/>')
        s.text(cx, y + 63, l1, 44, tx, "middle", "700")
        s.text(cx, y + 114, l2, 34, tx, "middle")

    def chain(xs, y, w):
        for a, b in zip(xs, xs[1:]):
            s.line(a + w / 2 + 4, y, b - w / 2 - 8, y, GREY, marker="m-grey")

    BH = 145
    # Stage A
    AY, AW, AX = 255, 300, [300, 661, 1022, 1383, 1743]
    frame(120, 120, 1804, 330, BLUE)
    header(163, 195, "Stage A:", "prepare data once")
    for cx, (a, b) in zip(AX, [("2,000", "questions"), ("3 teachers", "× 3 samples"), ("18,000", "trajectories"),
                               ("answer", "filtering"), ("Qual(t)", "embed(t)")]):
        box(cx, AY, AW, BH, BLUE_BG, BLUE, BLUE_TX, a, b)
    chain(AX, AY + BH / 2, AW)

    # Stage B
    BY, BW, BX = 900, 375, [368, 804, 1240, 1676]
    frame(120, 720, 1804, 560, PUR)
    header(1880, 795, "Stage B:", "per student × per method", anchor="end")
    # "exact search": chọn lọc bằng duyệt hết mọi tập con cỡ k (tối đa 84), không dùng tham lam, vì Div không
    # submodular. Số mẫu (cập nhật 29/09 sau khi chấm bù giám khảo, audit_stage_a 22/22): 1.899 câu còn lại
    # sau lọc đáp án, trừ 4 câu tụt dưới 3 ứng viên khi bỏ 75 chuỗi không có điểm giám khảo, còn
    # 1.895 câu × 3 = 5.685. Ô thứ ba của Stage A là 18.000 chuỗi SINH RA, chưa lọc; paper chỉ gọi
    # "candidates" cho 15.010 chuỗi đúng, nên ô này ghi "trajectories" để hai chỗ không lệch nhau.
    # Ô cuối ghi "LoRA" chứ không ghi "QLoRA" (chốt 02/10): mô hình 1,5 tỷ dùng LoRA 16-bit không lượng tử, chỉ
    # mô hình 7 tỷ dùng QLoRA 4-bit; phần Thiết lập thực nghiệm của paper nêu rõ từng mô hình.
    for cx, (a, b) in zip(BX, [("Fit(t, m)", "once per student"), ("exact search", "max F(S, m)"),
                               ("5,685 samples", "k = 3 per question"), ("LoRA", "fine-tuning")]):
        box(cx, BY, BW, BH, PUR_BG, PUR, PUR_TX, a, b)
    chain(BX, BY + BH / 2, BW)

    # A -> B: hai nhánh, nhánh vào Fit đi cao hơn để hai đường không cắt nhau
    ya_bot, y1, y2 = AY + BH, 545, 640
    s.poly([(AX[3], ya_bot), (AX[3], y1), (BX[0], y1), (BX[0], BY - 6)], BLUE, "m-blue")
    s.text((AX[3] + BX[0]) / 2, y1 - 16, "filtered candidates (Fit reads every chain)", 32, BLUE_TX, "middle")
    s.poly([(AX[4], ya_bot), (AX[4], y2), (BX[1], y2), (BX[1], BY - 6)], BLUE, "m-blue")
    s.text((AX[4] + BX[1]) / 2 + 60, y2 - 16, "cached Qual(t), embed(t), reused by every method", 32, BLUE_TX, "middle")

    # vòng lặp: vòng trong về greedy (đổi phương án), vòng ngoài về Fit (đổi mô hình học)
    yb_bot, q = BY + BH, BX[3]
    ya, yb = yb_bot + 70, yb_bot + 150
    s.poly([(q - 70, yb_bot), (q - 70, ya), (BX[1], ya), (BX[1], yb_bot + 6)], PUR, "m-pur", True, 3.4)
    s.text((q - 70 + BX[1]) / 2, ya - 14, "next method: reselect and retrain", 31, PUR_TX, "middle")
    s.poly([(q + 50, yb_bot), (q + 50, yb), (BX[0], yb), (BX[0], yb_bot + 6)], PUR, "m-pur", True, 3.4)
    s.text((q + 50 + BX[0]) / 2, yb - 14, "next student: recompute Fit, then repeat every method", 31, PUR_TX, "middle")

    # Stage C
    CY, CW, CX = 1590, 375, [820, 1256]
    frame(120, 1450, 1804, 340, GRN)
    header(163, 1525, "Stage C:", "evaluate and deploy")
    for cx, (a, b) in zip(CX, [("6 benchmarks", "Acc@4, Pass@4"), ("GGUF", "Ollama")]):
        box(cx, CY, CW, BH, GRN_BG, GRN, GRN_TX, a, b)
    chain(CX, CY + BH / 2, CW)

    # B -> C
    yg = 1365
    s.poly([(q + 150, yb_bot), (q + 150, yg), (CX[0], yg), (CX[0], CY - 6)], GRN, "m-grn")
    s.text((q + 150 + CX[0]) / 2, yg - 16, "fine-tuned LoRA adapter, one per student × method × seed", 32, GRN_TX, "middle")
    return s.render()


# ================================================================ Hình 2: phân bố số chuỗi đúng
GROUP_STYLE = {           # nhãn hiển thị và màu, theo thứ tự dễ tới khó
    "gsm8k":   ("GSM8K", GREY),
    "math-L3": ("MATH level 3", GRN),
    "math-L4": ("MATH level 4", BLUE),
    "math-L5": ("MATH level 5", PUR),
}


def distribution_series(stats: Mapping) -> list[tuple[str, str, int, list[float]]]:
    """Từ stats.json của a3 ra (nhãn, màu, số câu, tỷ lệ % theo số chuỗi đúng 0..N).

    Dùng TỶ LỆ chứ không dùng số câu, vì các nhóm có cỡ khác nhau (lô pilot: 100 câu GSM8K
    so với 28 và 32 câu MATH). Vẽ số tuyệt đối thì GSM8K lấn át và không so được hình dạng phân bố.
    """
    out = []
    for key, (lab, color) in GROUP_STYLE.items():
        g = stats["per_group"].get(key)
        if not g or not g.get("questions"):
            continue
        hist = [g["correct_count_hist"][str(i)] for i in range(len(g["correct_count_hist"]))]
        n = sum(hist)
        out.append((lab, color, n, [100 * v / n for v in hist]))
    if not out:
        raise SystemExit("stats.json không có nhóm nào trong gsm8k, math-L3, math-L4, math-L5.")
    return out


def build_distribution(stats: Mapping) -> str:
    series = distribution_series(stats)
    n_bins = len(series[0][3])
    min_correct = stats.get("min_correct", 3)

    W, H = 2040, 1180
    L, R, T, B = 190, 60, 150, 190            # lề trái, phải, trên, dưới của vùng vẽ
    pw, ph = W - L - R, H - T - B
    s = Svg(W, H)

    ymax = max(max(v) for *_, v in series)
    top = 100 if ymax > 60 else (60 if ymax > 40 else 40)
    ystep = 20 if top >= 60 else 10

    def X(i):  # tâm của cột nhóm thứ i
        return L + pw * (i + 0.5) / n_bins

    def Y(v):
        return T + ph * (1 - v / top)

    # vùng bị loại: số chuỗi đúng nhỏ hơn ngưỡng
    x_cut = L + pw * min_correct / n_bins
    s.add(f'<rect x="{L}" y="{T}" width="{x_cut - L}" height="{ph}" fill="#f3efe8"/>')
    s.text((L + x_cut) / 2, T + 44, f"dropped (fewer than {min_correct} correct)", 30, AMB, "middle", "700")

    # lưới và trục tung
    for v in range(0, top + 1, ystep):
        s.line(L, Y(v), L + pw, Y(v), "#e4e0da", 1.6)
        s.text(L - 18, Y(v) + 11, f"{v}%", 30, DIM, "end")
    s.line(L, T, L, T + ph, "#9a968f", 2.4)
    s.line(L, T + ph, L + pw, T + ph, "#9a968f", 2.4)
    s.line(x_cut, T, x_cut, T + ph, AMB, 3, dashed=True)

    # cột
    k = len(series)
    slot = pw / n_bins
    bw = slot * 0.8 / k
    for j, (lab, color, n, vals) in enumerate(series):
        for i, v in enumerate(vals):
            x = X(i) - slot * 0.4 + j * bw
            y = Y(v)
            s.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 3:.1f}" height="{T + ph - y:.1f}" rx="3" fill="{color}"/>')

    # nhãn trục hoành
    for i in range(n_bins):
        s.text(X(i), T + ph + 46, str(i), 32, INK, "middle")
    s.text(L + pw / 2, T + ph + 110, f"Correct trajectories per question (out of {n_bins - 1})", 36, INK, "middle")
    s.add(f'<text x="58" y="{T + ph / 2}" transform="rotate(-90 58 {T + ph / 2})" text-anchor="middle" '
          f'font-family="{FONT}" font-size="36" fill="{INK}">Share of questions</text>')

    # chú giải
    lx, ly = L + 10, 70
    for lab, color, n, _ in series:
        s.add(f'<rect x="{lx}" y="{ly - 26}" width="30" height="30" rx="4" fill="{color}"/>')
        txt = f"{lab} (n = {n})"
        s.text(lx + 44, ly, txt, 32, INK)
        lx += 44 + text_width(txt, 32) + 60
    return s.render()


# ================================================================ dụng cụ cho biểu đồ cột
def vn(v: float) -> str:
    """Số theo cách viết tiếng Việt: dấu phẩy thập phân."""
    return f"{v:g}".replace(".", ",")


def grouped_bars(title: str, groups: Sequence[str], series: Sequence[tuple[str, str, Sequence[float]]],
                 vmax: float, unit: str = "%", w: int = 1500, h: int = 900) -> str:
    """Biểu đồ cột nhóm. series là danh sách (tên, màu, giá trị theo từng nhóm)."""
    s = Svg(w, h)
    left, right, top, bottom = 150, 60, 110, 190
    plot_w, plot_h = w - left - right, h - top - bottom
    s.text(left, 58, title, size=34, color=DIM)
    s.line(left, top + plot_h, w - right, top + plot_h, GREY, 3)
    s.line(left, top, left, top + plot_h, GREY, 3)
    for frac in (0.0, 0.5, 1.0):                      # ba vạch chia là đủ, nhiều hơn thì rối
        y = top + plot_h - frac * plot_h
        s.text(left - 20, y + 11, vn(round(vmax * frac, 3)) + unit, size=28, color=GREY, anchor="end")
    gw = plot_w / len(groups)
    bw = min(96, gw / (len(series) + 1.4))
    for gi, g in enumerate(groups):
        cx = left + gw * (gi + 0.5)
        x0 = cx - bw * len(series) / 2
        for si, (_, color, vals) in enumerate(series):
            v = vals[gi]
            bh = plot_h * min(v / vmax, 1.0)
            x = x0 + si * bw
            s.add(f'<rect x="{x:.1f}" y="{top + plot_h - bh:.1f}" width="{bw - 8:.1f}" height="{bh:.1f}" '
                  f'fill="{color}" rx="4"/>')
            s.text(x + (bw - 8) / 2, top + plot_h - bh - 16, vn(v), size=26, color=INK, anchor="middle")
        s.text(cx, top + plot_h + 48, g, size=30, color=DIM, anchor="middle")
    lx = left
    for name, color, _ in series:
        s.add(f'<rect x="{lx}" y="{h - 78}" width="30" height="30" fill="{color}" rx="4"/>')
        s.text(lx + 44, h - 54, name, size=30, color=DIM)
        lx += 70 + text_width(name, 30)
    return s.render()


def percentiles(xs: Sequence[float], ps: Sequence[float]) -> list[float]:
    v = sorted(xs)
    return [v[min(int(p * len(v)), len(v) - 1)] for p in ps] if v else [0.0] * len(ps)


def load_labels(cfg, workdir: str) -> tuple[list, dict]:
    wd = resolve_path(cfg, workdir)
    files = cfg["stage_a_files"]
    labels = read_jsonl(wd / files["labels"])
    questions = {q["qid"]: q for q in read_jsonl(wd / files["questions"])}
    if not labels:
        raise SystemExit(f"Không thấy labels.jsonl trong {wd}. Chạy a3_filter trước.")
    return labels, questions


TEACHER_COLORS = [(GRN, "Llama-3.3-70B", "llama70b"), (BLUE, "DeepSeek-V3", "deepseek"),
                  (PUR, "Qwen2.5-VL-72B", "qwen72b")]
GROUPS = ["gsm8k", "math-L3", "math-L4", "math-L5"]
GROUP_LABELS = ["GSM8K", "MATH mức 3", "MATH mức 4", "MATH mức 5"]


# ================================================================ Hình A-2: tỷ lệ đúng theo mô hình dạy
def build_teachers(cfg, workdir: str) -> str:
    labels, questions = load_labels(cfg, workdir)
    tot: dict = defaultdict(int)
    ok: dict = defaultdict(int)
    for l in labels:
        key = (l["teacher"], group_of(questions[l["qid"]]))
        tot[key] += 1
        ok[key] += bool(l["correct"])
    series = [(name, color, [round(100 * ok[(t, g)] / max(tot[(t, g)], 1), 1) for g in GROUPS])
              for color, name, t in TEACHER_COLORS]
    return grouped_bars("Tỷ lệ chuỗi đúng theo mô hình dạy và mức độ khó", GROUP_LABELS, series, 100.0)


# ================================================================ Hình A-3: tỷ lệ chuỗi bị cắt
def build_truncation(cfg, workdir: str) -> str:
    labels, questions = load_labels(cfg, workdir)
    tot: dict = defaultdict(int)
    cut: dict = defaultdict(int)
    for l in labels:
        key = (l["teacher"], group_of(questions[l["qid"]]))
        tot[key] += 1
        cut[key] += l["reason"] == "truncated"
    series = [(name, color, [round(100 * cut[(t, g)] / max(tot[(t, g)], 1), 1) for g in GROUPS])
              for color, name, t in TEACHER_COLORS]
    vmax = max(max(v for _, _, vals in series for v in vals), 5.0)
    return grouped_bars("Tỷ lệ chuỗi bị cắt theo mô hình dạy và mức độ khó", GROUP_LABELS, series, vmax)


# ================================================================ Hình A-4: độ dài chuỗi đúng
def build_lengths(cfg, workdir: str) -> str:
    from src.stage_a.a2_generate import all_trajectory_files
    wd = resolve_path(cfg, workdir)
    files = cfg["stage_a_files"]
    correct = {l["tid"] for l in read_jsonl(wd / files["labels"]) if l["correct"]}
    by_t: dict = defaultdict(list)
    for path in all_trajectory_files(wd, files):
        for t in read_jsonl(path):
            if t["tid"] in correct and t.get("completion_tokens"):
                by_t[t["teacher"]].append(t["completion_tokens"])
    if not by_t:
        raise SystemExit("Không thấy file chuỗi. Hình này cần trajectories.*.jsonl trên máy có dữ liệu gốc.")
    ps = [0.5, 0.75, 0.9, 0.99]
    series = [(name, color, [round(v) for v in percentiles(by_t.get(t, []), ps)])
              for color, name, t in TEACHER_COLORS]
    vmax = max(v for _, _, vals in series for v in vals)
    return grouped_bars("Độ dài chuỗi đúng theo mô hình dạy, số token",
                        ["trung vị", "p75", "p90", "p99"], series, vmax, unit="")


# ================================================================ Hình A-6: phân bố điểm giám khảo
def build_judge(cfg, workdir: str) -> str:
    wd = resolve_path(cfg, workdir)
    rows = [r for r in read_jsonl(wd / "quality.jsonl") if r.get("llm_score") is not None]
    if not rows:
        raise SystemExit(f"Không thấy quality.jsonl có điểm giám khảo trong {wd}.")
    bins = [round(0.1 * i, 1) for i in range(11)]
    counts = Counter(round(round(r["llm_score"], 1), 1) for r in rows)
    vals = [counts.get(b, 0) for b in bins]
    from src.tools.compare_judges import spearman
    rule = [r["rule_score"] for r in rows]
    llm = [r["llm_score"] for r in rows]
    words = [r["n_words"] for r in rows]

    s = Svg(1500, 980)                                   # đủ chỗ cho ba dòng tương quan bên dưới
    left, top, plot_w, plot_h = 170, 120, 1270, 560
    s.text(left, 60, f"Phân bố điểm giám khảo trên {len(rows):,} chuỗi".replace(",", "."), size=34, color=DIM)
    s.line(left, top + plot_h, left + plot_w, top + plot_h, GREY, 3)
    s.line(left, top, left, top + plot_h, GREY, 3)
    vmax = max(vals)
    bw = plot_w / len(bins)
    for i, (b, v) in enumerate(zip(bins, vals)):
        bh = plot_h * v / vmax
        x = left + i * bw + 10
        color = BLUE if b >= 0.9 else GREY
        s.add(f'<rect x="{x:.1f}" y="{top + plot_h - bh:.1f}" width="{bw - 20:.1f}" height="{bh:.1f}" '
              f'fill="{color}" rx="4"/>')
        if v:
            s.text(x + (bw - 20) / 2, top + plot_h - bh - 14, f"{v:,}".replace(",", "."),
                   size=24, color=INK, anchor="middle")
        s.text(x + (bw - 20) / 2, top + plot_h + 44, f"{b:.1f}".replace(".", ","), size=28, color=DIM, anchor="middle")
    y = top + plot_h + 120
    s.text(left, y, "Tương quan hạng giữa hai nửa của điểm chất lượng", size=32, color=INK, weight="600")
    for i, (name, a, b) in enumerate([("điểm quy tắc và điểm giám khảo", rule, llm),
                                      ("điểm quy tắc và số từ", rule, words),
                                      ("điểm giám khảo và số từ", llm, words)]):
        r = spearman(a, b)
        s.text(left, y + 52 + i * 46, f"{name}: {r:+.2f}".replace(".", ","), size=30, color=DIM)
    return s.render()


# ================================================================ Hình A-8: khoảng cách biểu diễn
def build_distance(cfg, workdir: str) -> str:
    wd = resolve_path(cfg, workdir)
    probe = json.loads((wd / "embed_probe.json").read_text(encoding="utf-8"))
    # a5_embed ghi hai khoá within_teacher và across_teacher; hai tên cũ chỉ giữ để đọc file probe đời trước
    same = probe.get("within_teacher") or probe.get("same_teacher") or {}
    cross = probe.get("across_teacher") or probe.get("cross_teacher") or {}
    if "median" not in same or "median" not in cross:
        raise SystemExit("embed_probe.json thiếu within_teacher hoặc across_teacher. Chạy: a5_embed --probe --reuse")
    by_group = probe.get("by_group", {})
    if not by_group:
        raise SystemExit("embed_probe.json chưa có phần by_group. Chạy: a5_embed --probe --reuse")
    series = [("trung vị", BLUE, [round(same.get("median", 0), 3), round(cross.get("median", 0), 3)]
               + [round(by_group[g]["median"], 3) for g in GROUPS if g in by_group])]
    groups = ["Cùng mô hình", "Khác mô hình"] + [GROUP_LABELS[GROUPS.index(g)] for g in GROUPS
                                                          if g in by_group]
    vmax = max(series[0][2]) * 1.25
    return grouped_bars("Khoảng cách biểu diễn trung vị giữa hai chuỗi của cùng một câu hỏi",
                        groups, series, vmax, unit="")


# ================================================================ Hình B-1: ma trận tương quan
def build_correlation(cfg, workdir: str, fit_file: str) -> str:
    from src.stage_b.signals import prepare_fit_rows
    from src.tools.compare_fit import SIGNALS, signal_matrix
    wd = resolve_path(cfg, workdir)
    rows, stale = prepare_fit_rows(read_jsonl(wd / fit_file))
    if not rows:
        raise SystemExit(f"Không đọc được {wd / fit_file}. Chạy b1_fit trước.")
    if stale:
        raise SystemExit(f"{fit_file} tạo trước 29/09, LARK và LocalNat mang định nghĩa cũ. "
                         f"Chạy lại b1_fit rồi mới vẽ Hình B-1.")
    names = [n for n in SIGNALS if n in rows[0]]
    m = signal_matrix(rows, names)
    show = {"rsr": "RSR", "grape": "GRAPE", "local_nat": "LocalNat", "lark": "LARK", "n_tokens": "Độ dài"}

    cell, pad = 190, 230
    s = Svg(pad + cell * len(names) + 60, pad + cell * len(names) + 140)
    s.text(60, 70, f"Tương quan hạng trong từng câu hỏi, {len(rows):,} chuỗi".replace(",", "."),
           size=34, color=DIM)
    s.text(60, 118, "mọi tín hiệu đã đưa về hướng càng cao càng phù hợp, RSR đã đảo dấu", size=28, color=GREY)
    for j, y in enumerate(names):
        s.text(pad + cell * j + cell / 2, pad - 24, show[y], size=30, color=DIM, anchor="middle")
        s.text(pad - 24, pad + cell * j + cell / 2 + 10, show[y], size=30, color=DIM, anchor="end")
    for i, x in enumerate(names):
        for j, y in enumerate(names):
            v = 1.0 if x == y else (m.get((x, y)) or m.get((y, x)) or 0.0)
            # đậm theo độ lớn, xanh cho cùng chiều và cam cho ngược chiều
            t = min(abs(v), 1.0)
            base = (59, 138, 224) if v >= 0 else (201, 86, 27)
            mix = tuple(round(255 - (255 - c) * t) for c in base)
            fill = "#%02x%02x%02x" % mix
            s.add(f'<rect x="{pad + cell * j}" y="{pad + cell * i}" width="{cell - 10}" height="{cell - 10}" '
                  f'fill="{fill}" rx="6"/>')
            s.text(pad + cell * j + (cell - 10) / 2, pad + cell * i + cell / 2 + 10,
                   f"{v:+.2f}".replace(".", ","), size=32, anchor="middle",
                   color="#ffffff" if t > 0.55 else INK)
    return s.render()


# ================================================================ Hình kết quả: đọc kết quả đánh giá và tập chọn
# Hai hình dùng chung một bảng số: Acc@4 trên MATH-500 của từng tập huấn luyện (trung bình qua các seed),
# chênh lệch ghép cặp so với Correct-Only kèm khoảng tin cậy 95% (cùng cách tính với eval_table --versus),
# và tỷ lệ chuỗi Qwen2.5-72B trong tập (đọc từ select.<tên>.json do b2_select ghi). Chạy trên máy có kết quả đánh giá.
C_STUDENT, C_TEACHER, C_FREE = "#2a78d6", "#eb6834", "#52514e"
GRID_C, LEAD_C = "#e8e7e3", "#a9a8a2"
# tên tập chọn -> (nhãn tiếng Anh, nhãn tiếng Việt, nhóm). Nhóm quyết định màu. Tập không có trong bảng thì không vẽ.
RESULT_SETS = {
    "no_filter.k3": ("No-Filter", "No-Filter", "free"),
    "correct_only.k3": ("Correct-Only", "Correct-Only", "free"),
    "token_length.k3": ("Token-Length", "Token-Length", "free"),
    "grape.k3": ("GRAPE", "GRAPE", "student"),
    "local_naturalness.k3": ("Local Naturalness", "Local Naturalness", "student"),
    "rsr.k3": ("RSR", "RSR", "student"),
    "lark.k3": ("LARK", "LARK", "student"),
    "fit_quality.k3": ("QD-RSR (λ = 0)", "QD-RSR (λ = 0)", "student"),
    "qd_rsr.k3.lam0.4": ("λ = 0.4 (not selected)", "λ = 0,4 (không được chọn)", "student"),
    "teacher_qwen_first.k3": ("Prefer Qwen-72B", "Ưu tiên Qwen-72B", "teacher"),
    "teacher_qwen_last.k3": ("Avoid Qwen-72B", "Tránh Qwen-72B", "teacher"),
    "fq_select.k3": ("FQ-Select", "FQ-Select", "student"),
    "quality_only.k3": ("Quality-only", "Quality-only", "free"),
    "quality_within_teacher.k3": ("Quality within teacher", "Chất lượng trong từng mô hình dạy", "teacher"),
}
# Tập in đậm là phương án đề xuất: FQ-Select từ 20/10 (trước đó là fit_quality.k3 của hướng QD-RSR cũ).
OURS, BASE = "fq_select.k3", "correct_only.k3"
GROUP_COLOR = {"student": C_STUDENT, "teacher": C_TEACHER, "free": C_FREE}
RESULT_TEXT = {
    "en": {"x": "Share of Qwen2.5-72B trajectories in the training set (%)", "y": "MATH-500 Acc@4 (%)",
           "groups": ["selected by a student signal", "teacher-identity control", "no student signal"],
           "seeds": ["mean of 2+ seeds", "1 seed"], "base": "Correct-Only", "dec": "."},
    "vi": {"x": "Tỷ lệ chuỗi của Qwen2.5-72B trong tập huấn luyện (%)", "y": "Acc@4 trên MATH-500 (%)",
           "groups": ["chọn theo tín hiệu của mô hình học", "đối chứng theo mô hình dạy",
                      "không dùng tín hiệu của mô hình học"],
           "seeds": ["trung bình từ 2 seed", "1 seed"], "base": "mức Correct-Only", "dec": ","},
}


def result_rows(results: Path, seldir: Path, seeds: Sequence[int] | None = None, bench: str = "math500",
                base: str = BASE, teacher: str = "qwen72b", n_boot: int = 2000) -> list[dict]:
    """Một dòng cho mỗi tập trong RESULT_SETS có kết quả trên `bench`: acc (%), số seed, tỷ lệ chuỗi của `teacher`
    (%), và chênh lệch ghép cặp so với `base` (điểm %) trên các seed hai tập cùng có. Thiếu select.<tên>.json hay
    thiếu file từng lượt sinh thì dừng và báo, không vẽ thiếu."""
    from src.tools.eval_table import find_runs, paired_bootstrap, pooled_question_acc

    runs = find_runs(results)
    if base not in runs:
        raise SystemExit(f"Không thấy kết quả của {base} trong {results}.")
    base_seeds = {sd for sd, s in runs[base].items() if bench in s.get("benchmarks", {})}
    rows = []
    for tag, (en, vi, group) in RESULT_SETS.items():
        by_seed = {sd: s for sd, s in runs.get(tag, {}).items()
                   if (seeds is None or sd in seeds) and bench in s.get("benchmarks", {})}
        if not by_seed:
            continue
        sel = seldir / f"select.{tag}.json"
        if not sel.exists():
            raise SystemExit(f"Thiếu {sel}: cần tỷ lệ mô hình dạy của tập {tag}.")
        share = json.loads(sel.read_text(encoding="utf-8")).get("teacher_share", {})
        n_samples = int(next(iter(by_seed.values())).get("settings", {}).get("n_samples", 4))
        row = {"tag": tag, "en": en, "vi": vi, "group": group, "seeds": len(by_seed),
               "acc": 100 * sum(s["benchmarks"][bench]["acc@4"] for s in by_seed.values()) / len(by_seed),
               "share": 100 * float(share.get(teacher, 0.0)), "diff": None, "lo": None, "hi": None}
        if tag != base:
            shared = sorted(set(by_seed) & base_seeds)
            r = None
            if shared:
                r = paired_bootstrap(pooled_question_acc(runs[tag], shared, bench, n_samples),
                                     pooled_question_acc(runs[base], shared, bench, n_samples), n_boot)
            if r is None:
                raise SystemExit(f"Không so ghép cặp được {tag} với {base} trên {bench} (thiếu file {bench}.jsonl?).")
            row.update(diff=100 * r["diff"], lo=100 * r["lo"], hi=100 * r["hi"])
        rows.append(row)
    return rows


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Tương quan hạng Spearman, giá trị bằng nhau lấy hạng trung bình. Chỉ để mô tả, không kiểm định."""
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(list(x)), ranks(list(y))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def num(v: float, dec: str = ",", digits: int | None = None, sign: bool = False) -> str:
    """Số để in lên hình: dấu thập phân theo ngôn ngữ, dấu trừ thật."""
    if digits is None:
        txt = f"{v:g}"
    else:
        txt = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
    return txt.replace(".", dec).replace("-", "−")


def dot(s: Svg, x: float, y: float, color: str, filled: bool, r: float = 13) -> None:
    s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color if filled else "#ffffff"}" '
          f'stroke="{color}" stroke-width="5"/>')


def swatch(s: Svg, x: float, y: float, color: str) -> None:
    """Ô màu cho chú giải nhóm; dùng ô chứ không dùng chấm để không lẫn với chú giải chấm đặc, chấm rỗng."""
    s.add(f'<rect x="{x}" y="{y - 20}" width="36" height="20" fill="{color}"/>')


def build_teacher_share(rows: Sequence[Mapping], lang: str = "en") -> str:
    """Acc@4 trên MATH-500 theo tỷ lệ chuỗi Qwen2.5-72B của tập huấn luyện. Chỉ dùng chấm tròn: màu là cách chọn,
    chấm đặc là trung bình từ 2 seed, chấm rỗng là 1 seed. Cụm bên trái (tỷ lệ dưới 15%) xếp nhãn thành một cột
    có đường dẫn; các điểm khác thử lần lượt vài vị trí quanh điểm cho tới khi không đè lên gì."""
    t = RESULT_TEXT[lang]
    pts = [r for r in rows if r.get("acc") is not None]
    if not pts:
        raise SystemExit("Không có tập nào có kết quả để vẽ.")
    W, H = 1500, 940
    L, R, T, B = 140, 40, 30, 130
    pw, ph = W - L - R, H - T - B
    x0, x1 = -3.0, 90.0
    y0 = float(int(min(r["acc"] for r in pts)) - 1)
    y1 = float(int(max(r["acc"] for r in pts)) + 3)

    def X(v: float) -> float:
        return L + (v - x0) / (x1 - x0) * pw

    def Y(v: float) -> float:
        return T + ph - (v - y0) / (y1 - y0) * ph

    s = Svg(W, H)
    for v in range(0, 100, 20):
        s.line(X(v), T, X(v), T + ph, GRID_C, 2)
        s.text(X(v), T + ph + 42, str(v), 28, DIM, "middle")
    step = 2 if (y1 - y0) <= 24 else 5
    v = y0 + (-y0) % step
    while v <= y1:
        s.line(L, Y(v), L + pw, Y(v), GRID_C, 2)
        s.text(L - 16, Y(v) + 10, num(v, t["dec"]), 28, DIM, "end")
        v += step
    s.line(L, T + ph, L + pw, T + ph, GREY, 3)
    s.line(L, T, L, T + ph, GREY, 3)
    s.text(L + pw / 2, H - 30, t["x"], 30, INK, "middle")
    s.add(f'<text x="40" y="{T + ph / 2}" transform="rotate(-90 40 {T + ph / 2})" text-anchor="middle" '
          f'font-family="{FONT}" font-size="30" fill="{INK}">{t["y"]}</text>')
    base = next((r for r in pts if r["tag"] == BASE), None)
    if base:
        s.line(L, Y(base["acc"]), L + pw, Y(base["acc"]), LEAD_C, 2.5, dashed=True)
        s.text(L + pw - 8, Y(base["acc"]) - 12, t["base"], 24, DIM, "end")

    # chú giải ở góc dưới phải: hai dòng số seed, rồi ba dòng màu
    lw = max(text_width(x, 24) for x in t["groups"] + t["seeds"]) + 70
    lx, ly, dy = L + pw - lw, T + ph - 5 * 38 - 18, 38
    boxes = [(X(r["share"]) - 16, Y(r["acc"]) - 16, 32, 32) for r in pts] + [(lx - 10, ly - 10, lw + 10, 5 * dy + 20)]
    if base:                                             # nhãn không được đè lên đường gạch Correct-Only
        boxes += [(L, Y(base["acc"]) - 3, pw, 6), (L + pw - 8 - text_width(t["base"], 24), Y(base["acc"]) - 34,
                                                   text_width(t["base"], 24), 28)]
    for i, (filled, lab) in enumerate(zip([True, False], t["seeds"])):
        dot(s, lx + 18, ly + 18 + i * dy, C_FREE, filled, 10)
        s.text(lx + 50, ly + 27 + i * dy, lab, 24, DIM)
    for i, (g, lab) in enumerate(zip(["student", "teacher", "free"], t["groups"])):
        swatch(s, lx, ly + 2 * dy + 28 + i * dy, GROUP_COLOR[g])
        s.text(lx + 50, ly + 2 * dy + 27 + i * dy, lab, 24, DIM)

    def fits(bx: tuple) -> bool:
        x, y, w, h = bx
        return (x >= L + 4 and x + w <= L + pw and y >= T and y + h <= T + ph and
                not any(x < ox + ow and ox < x + w and y < oy + oh and oy < y + h for ox, oy, ow, oh in boxes))

    placed = []
    left = sorted([r for r in pts if r["share"] < 15], key=lambda r: -r["acc"])
    col_x, gap, ys = X(12), 40, []
    for r in left:                                       # cột nhãn theo thứ tự Acc@4, cách nhau ít nhất `gap`
        y = Y(r["acc"]) + 9
        ys.append(max(y, ys[-1] + gap) if ys else y)
    if ys and ys[-1] > T + ph - 8:                       # tràn xuống trục thì đẩy cả cột lên
        ys = [y - (ys[-1] - (T + ph - 8)) for y in ys]
    for r, y in zip(left, ys):
        w = text_width(r[lang], 26, r["tag"] == OURS)
        placed.append((r, col_x, y, "start"))
        boxes.append((col_x, y - 24, w, 32))
    for r in sorted([r for r in pts if r["share"] >= 15], key=lambda r: -r["acc"]):
        px, py = X(r["share"]), Y(r["acc"])
        w = text_width(r[lang], 26, r["tag"] == OURS)
        cands = [(px + 24, py + 9, "start"), (px - 24, py + 9, "end"), (px + 16, py - 24, "start"),
                 (px - 16, py - 24, "end"), (px + 16, py + 44, "start"), (px - 16, py + 44, "end"),
                 (px - 60, py - 60, "end"), (px + 60, py - 60, "start"), (px - 60, py + 80, "end"),
                 (px + 60, py + 80, "start")]
        for cx, cy, anc in cands:
            bx = (cx if anc == "start" else cx - w, cy - 24, w, 32)
            if fits(bx):
                break
        placed.append((r, cx, cy, anc))
        boxes.append(bx)
    for r, cx, cy, anc in placed:                        # đường dẫn vẽ trước để nằm dưới chấm và chữ
        px, py = X(r["share"]), Y(r["acc"])
        if abs(cx - px) > 26 or abs(cy - 9 - py) > 12:
            s.line(px, py, cx + (-6 if anc == "start" else 6), cy - 9, LEAD_C, 2)
    for r in pts:
        dot(s, X(r["share"]), Y(r["acc"]), GROUP_COLOR[r["group"]], r["seeds"] >= 2)
    for r, cx, cy, anc in placed:
        s.text(cx, cy, r[lang], 26, INK, anc, "700" if r["tag"] == OURS else "400")
    return s.render()


def build_forest(rows: Sequence[Mapping]) -> str:
    """Chênh lệch Acc@4 trên MATH-500 so với Correct-Only, mỗi dòng một tập: chấm là chênh lệch, đoạn ngang là
    khoảng tin cậy 95% ghép cặp theo câu hỏi. Chú giải nói rõ ký hiệu và cách đọc vạch 0 (bản tiếng Việt)."""
    t = RESULT_TEXT["vi"]
    rs = sorted([r for r in rows if r.get("diff") is not None], key=lambda r: -r["diff"])
    if not rs:
        raise SystemExit("Không có chênh lệch nào để vẽ.")
    W, row_h, L, R, T = 1500, 66, 430, 110, 24
    ph = row_h * len(rs)
    H = T + ph + 330
    lo = min(r["lo"] for r in rs) - 1.8
    hi = max(r["hi"] for r in rs) + 2.2

    def X(v: float) -> float:
        return L + (v - lo) / (hi - lo) * (W - L - R)

    s = Svg(W, H)
    step = 2.0 if hi - lo <= 22 else 4.0
    v = lo + (-lo) % step
    while v <= hi:
        s.line(X(v), T, X(v), T + ph, GRID_C, 2)
        s.text(X(v), T + ph + 42, num(round(v, 6), ","), 28, DIM, "middle")
        v += step
    s.line(X(0), T - 8, X(0), T + ph, GREY, 3)
    s.line(L, T + ph, W - R, T + ph, GREY, 3)
    s.text((L + W - R) / 2, T + ph + 92, "Chênh lệch Acc@4 trên MATH-500 so với Correct-Only (điểm %)", 30, INK,
           "middle")
    for i, r in enumerate(rs):
        y = T + row_h * (i + 0.5)
        c = GROUP_COLOR[r["group"]]
        s.text(L - 28, y + 10, r["vi"], 28, INK, "end", "700" if r["tag"] == OURS else "400")
        s.add(f'<line x1="{X(r["lo"]):.1f}" y1="{y:.1f}" x2="{X(r["hi"]):.1f}" y2="{y:.1f}" stroke="{c}" '
              f'stroke-width="7" stroke-linecap="round"/>')
        dot(s, X(r["diff"]), y, c, r["seeds"] >= 2, 12)
        if r["diff"] < 0:                                # số đặt phía ngoài đoạn ngang, không đè lên vạch 0
            s.text(X(r["lo"]) - 16, y + 10, num(r["diff"], ",", 1, sign=True), 27, INK, "end")
        else:
            s.text(X(r["hi"]) + 16, y + 10, num(r["diff"], ",", 1, sign=True), 27, INK)
    ly = T + ph + 160
    x = 60
    for g, lab in zip(["student", "teacher", "free"], t["groups"]):
        swatch(s, x, ly, GROUP_COLOR[g])
        s.text(x + 48, ly, lab, 25, DIM)
        x += 48 + text_width(lab, 25) + 50
    ly += 50
    dot(s, 78, ly - 9, C_FREE, True, 10)
    s.text(108, ly, t["seeds"][0], 25, DIM)
    x = 108 + text_width(t["seeds"][0], 25) + 60
    dot(s, x + 18, ly - 9, C_FREE, False, 10)
    s.text(x + 48, ly, t["seeds"][1], 25, DIM)
    x += 48 + text_width(t["seeds"][1], 25) + 60
    s.add(f'<line x1="{x}" y1="{ly - 9}" x2="{x + 60}" y2="{ly - 9}" stroke="{C_FREE}" stroke-width="7" '
          f'stroke-linecap="round"/>')
    dot(s, x + 30, ly - 9, C_FREE, True, 10)
    s.text(x + 80, ly, "chấm: chênh lệch; đoạn ngang: khoảng tin cậy 95%", 25, DIM)
    s.text(60, ly + 56, "Đoạn ngang không cắt vạch 0 nghĩa là chênh lệch đó lớn hơn mức nhiễu của tập câu hỏi.",
           25, DIM)
    return s.render()


_RESULT_CACHE: dict = {}


def load_results(cfg, a) -> list[dict]:
    """Đọc kết quả một lần cho mọi hình kết quả trong cùng lệnh, và in tương quan hạng để đối chiếu với báo cáo."""
    from src.common.config import path_of

    if "student" not in cfg:                             # các hình khác không cần cấu hình mô hình học
        cfg = load_config(student=a.student)
    tag = f"{cfg['student']['key']}_base"
    results = resolve_path(cfg, a.results) if a.results else path_of(cfg, "out_results") / "eval" / tag
    seldir = resolve_path(cfg, a.seldir) if a.seldir else path_of(cfg, "data_stage_b") / tag
    key = (str(results), str(seldir), tuple(a.seeds or ()))
    if key not in _RESULT_CACHE:
        rows = result_rows(results, seldir, a.seeds)
        _RESULT_CACHE[key] = rows
        for r in rows:
            ci = "" if r["diff"] is None else f"  so với Correct-Only {r['diff']:+.1f} [{r['lo']:+.1f}; {r['hi']:+.1f}]"
            print(f"  {r['tag']:<24} {r['seeds']} seed  Qwen-72B {r['share']:5.1f}%  Acc@4 {r['acc']:.1f}{ci}")
        if len(rows) > 2:
            rho = spearman_rho([r["share"] for r in rows], [r["acc"] for r in rows])
            print(f"Spearman giữa tỷ lệ chuỗi Qwen-72B và Acc@4 trên MATH-500: {rho:.2f} trên {len(rows)} tập")
    return _RESULT_CACHE[key]


# ================================================================ Hình 3: ví dụ chọn lọc
def build_selection(*_args, **_kw) -> str:
    raise SystemExit(
        "Hình 3 chưa làm được: cần điểm Fit từ b1_fit và tập đã chọn từ b2_select, tức phải xong Giai đoạn B.\n"
        "Nội dung dự kiến (Notion mục Q): một câu hỏi thật với 9 chuỗi ứng viên, mỗi ô ghi Fit/Qual/Div,\n"
        "bên dưới là ba hàng: RSR top-k chọn gì, Quality-only chọn gì, QD-RSR chọn gì.")


# ================================================================ dòng lệnh
FIGURES = {
    "pipeline": ("fig1_pipeline", lambda a, cfg: build_pipeline()),
    "distribution": ("fig2_distribution", lambda a, cfg: build_distribution(
        json.loads(resolve_path(cfg, a.stats).read_text(encoding="utf-8")))),
    "selection": ("fig3_selection", lambda a, cfg: build_selection()),
    # Hình cho báo cáo đồ án, đọc thẳng từ dữ liệu thật nên vẽ lại được khi dữ liệu đổi (Notion mục T4)
    "teachers": ("figA2_teachers", lambda a, cfg: build_teachers(cfg, a.workdir)),
    "truncation": ("figA3_truncation", lambda a, cfg: build_truncation(cfg, a.workdir)),
    "lengths": ("figA4_lengths", lambda a, cfg: build_lengths(cfg, a.workdir)),
    "judge": ("figA6_judge", lambda a, cfg: build_judge(cfg, a.workdir)),
    "distance": ("figA8_distance", lambda a, cfg: build_distance(cfg, a.workdir)),
    "correlation": ("figB1_correlation", lambda a, cfg: build_correlation(cfg, a.workdir, a.fit)),
    # Hình kết quả (thêm 12/10), đọc kết quả đánh giá và select.<tên>.json; gọi gộp bằng `results`
    "teacher_share": ("fig3_teacher_share", lambda a, cfg: build_teacher_share(load_results(cfg, a), "en")),
    "teacher_share_vi": ("figC2_teacher_share_vi", lambda a, cfg: build_teacher_share(load_results(cfg, a), "vi")),
    "forest": ("figC1_forest_vi", lambda a, cfg: build_forest(load_results(cfg, a))),
}
REPORT = ["teachers", "truncation", "lengths", "judge", "distance", "correlation"]
RESULTS = ["teacher_share", "teacher_share_vi", "forest"]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=list(FIGURES) + ["all", "report", "results"])
    ap.add_argument("--outdir", default="outputs/figures")
    ap.add_argument("--stats", default="data/stage_a/stats.json", help="stats.json cho Hình 2")
    ap.add_argument("--workdir", default="data/stage_a", help="thư mục dữ liệu cho các hình báo cáo")
    ap.add_argument("--fit", default="fit.qwen1_5b_base.jsonl", help="file điểm b1 cho hình tương quan")
    ap.add_argument("--student", default="qwen1_5b", help="hình kết quả: mô hình học (qwen1_5b, qwen7b)")
    ap.add_argument("--seeds", type=int, nargs="+", help="hình kết quả: chỉ lấy các seed này (mặc định mọi seed có)")
    ap.add_argument("--results", help="hình kết quả: thư mục kết quả, mặc định outputs/results/eval/<student>_base")
    ap.add_argument("--seldir", help="hình kết quả: thư mục select.*.json, mặc định data/stage_b/<student>_base")
    args = ap.parse_args(argv)

    cfg = load_config()
    outdir = resolve_path(cfg, args.outdir)
    if args.figure == "all":
        names = ["pipeline", "distribution"]
    elif args.figure == "report":
        names = REPORT
    elif args.figure == "results":
        names = RESULTS
    else:
        names = [args.figure]
    for key in names:
        fname, build = FIGURES[key]
        root = resolve_path(cfg, ".").resolve()
        for p in save(build(args, cfg), fname, outdir):
            shown = p.resolve().relative_to(root) if p.resolve().is_relative_to(root) else p
            print(f"đã ghi {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

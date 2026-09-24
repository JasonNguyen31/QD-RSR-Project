"""Vẽ mọi hình cho paper. Một file, mỗi hình một hàm.

    python -m src.tools.make_figures pipeline
    python -m src.tools.make_figures distribution
    python -m src.tools.make_figures distribution --stats data/stage_a/stats.json
    python -m src.tools.make_figures all

Chạy từ gốc repo. Hình xuất ra docs/figures/ dạng PDF (vector, dùng cho Overleaf) và PNG (xem nhanh).
Không để lại file SVG trung gian.

    Hình 1  pipeline      Pipeline ba giai đoạn, đường nối chỉ rõ ô nào sang ô nào
    Hình 2  distribution  Phân bố số chuỗi đúng mỗi câu theo độ khó. Biện minh cho việc đổi dữ liệu
    Hình 3  selection     Ví dụ chọn lọc trên một câu hỏi. CHƯA LÀM: cần điểm Fit, Qual, Div thật

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

from src.common.config import load_config, resolve_path

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
    for cx, (a, b) in zip(AX, [("2,000", "questions"), ("3 teachers", "× 3 samples"), ("18,000", "candidates"),
                               ("answer", "filtering"), ("Qual(t)", "embed(t)")]):
        box(cx, AY, AW, BH, BLUE_BG, BLUE, BLUE_TX, a, b)
    chain(AX, AY + BH / 2, AW)

    # Stage B
    BY, BW, BX = 900, 375, [368, 804, 1240, 1676]
    frame(120, 720, 1804, 560, PUR)
    header(1880, 795, "Stage B:", "per student × per method", anchor="end")
    # "exact search": chọn lọc bằng duyệt hết mọi tập con cỡ k (tối đa 84), không dùng tham lam, vì Div không
    # submodular. Số mẫu (cập nhật 23/09 sau khi thêm so tương đương sympy vào bộ chấm): 1.899 câu còn lại
    # sau lọc đáp án, trừ 9 câu tụt dưới 3 ứng viên khi bỏ các chuỗi không có điểm giám khảo, còn
    # 1.890 câu × 3 = 5.670.
    for cx, (a, b) in zip(BX, [("Fit(t, m)", "once per student"), ("exact search", "max F(S, m)"),
                               ("5,670 samples", "k = 3 per question"), ("QLoRA", "fine-tuning")]):
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
    for cx, (a, b) in zip(CX, [("6 benchmarks", "Acc@4, Acc@1"), ("GGUF", "Ollama")]):
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
}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=list(FIGURES) + ["all"])
    ap.add_argument("--outdir", default="docs/figures")
    ap.add_argument("--stats", default="data/pilot/run2_boxed/stats.json",
                    help="stats.json cho Hình 2. Khi có lô thật thì đổi sang data/stage_a/stats.json")
    args = ap.parse_args(argv)

    cfg = load_config()
    outdir = resolve_path(cfg, args.outdir)
    names = ["pipeline", "distribution"] if args.figure == "all" else [args.figure]
    for key in names:
        fname, build = FIGURES[key]
        root = resolve_path(cfg, ".").resolve()
        for p in save(build(args, cfg), fname, outdir):
            shown = p.resolve().relative_to(root) if p.resolve().is_relative_to(root) else p
            print(f"đã ghi {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

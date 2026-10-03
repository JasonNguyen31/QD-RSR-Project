"""c0_prepare_eval: tải sáu bộ đánh giá, đưa về một định dạng, kiểm số câu và kiểm rò rỉ. Cần mạng, chạy trên Mac.

    python -m src.stage_c.c0_prepare_eval --dry-run          # tải, kiểm, báo cáo, không ghi file
    python -m src.stage_c.c0_prepare_eval                    # ghi data/eval/<bộ>.jsonl và data/eval/manifest.json
    python -m src.stage_c.c0_prepare_eval --only bbh amc23   # chỉ làm lại vài bộ, các bộ khác giữ nguyên
    python -m src.stage_c.c0_prepare_eval --verify           # KHÔNG tải: so md5 các file đang có với manifest
                                                             # (chạy trên máy GPU sau khi chép data/eval sang)

Đọc  configs/data.yaml mục eval_data (nguồn, số câu kỳ vọng, commit và md5 đã ghim)
     data/raw/{gsm8k_pool,math_pool,validation,validation_matched}.jsonl, data/stage_a/questions.jsonl  (để kiểm rò rỉ)
Ghi  data/eval/<bộ>.jsonl      mỗi dòng: qid, benchmark, subset, question, gold, source_id
                               (thêm gold_raw ở dòng có đáp án gốc phải làm sạch, latex_fixed ở dòng có lệnh
                               LaTeX hỏng được khôi phục)
     data/eval/manifest.json   nguồn, commit, số câu, md5 của từng file, kết quả các phép kiểm

Vì sao tách khỏi c1_evaluate: máy GPU không cần mạng và không cần thư viện datasets; mọi lần đánh giá đọc đúng
một bản dữ liệu đã kiểm, có md5, nên hai lần chạy cách nhau một tháng vẫn chấm trên cùng bộ câu hỏi.

Sáu bộ (chốt nguồn 03/10/2026; bài RSR không công bố nguồn dữ liệu đánh giá của họ):
  gsm8k    openai/gsm8k, cấu hình main, split test: 1.319 câu. Kho huấn luyện lấy từ split train của cùng nguồn.
  math500  HuggingFaceH4/MATH-500: 500 câu. Kho huấn luyện đã loại đúng 500 câu này (a1_prepare).
  amc23    file amc23/test.jsonl trong kho QwenLM/Qwen2.5-Math: 40 câu của AMC 12A và 12B năm 2023, đã bỏ phần
           lựa chọn, đáp án là số nguyên. Bảng của bài RSR nhảy theo bội 0,625 điểm, tức đúng 40 câu x 4 lượt.
  aime24   Maxwell-Jia/AIME_2024: 30 câu, bản có sửa lỗi đề ở ba câu. Đối chiếu đáp án với bản trong kho
           Qwen2.5-Math (ghim theo commit); lệch đáp án thì dừng.
  aime25   opencompass/AIME2025, hai cấu hình AIME2025-I và AIME2025-II: 30 câu.
  bbh      kho chính thức suzgunmirac/BIG-Bench-Hard, thư mục bbh/: 27 file, 6.511 câu. Ghi đủ cả bộ; việc rút mẫu
           cho mỗi lần đánh giá dùng sample_per_subset để mọi phương án chấm trên cùng những câu đó.

Ba phép kiểm chạy MỖI LẦN, sai một phép là dừng và không ghi gì:
  1. số câu của từng bộ bằng số đã ghim; không câu nào thiếu đề hay thiếu đáp án; các bộ toán không có câu lặp
  2. các nguồn GitHub khớp md5 đã ghim; đáp án AIME 2024 khớp giữa hai nguồn độc lập
  3. không câu đánh giá nào trùng (sau norm của a1_prepare) với kho huấn luyện, kho câu hỏi gốc hay hai tập kiểm định

BBH chứa chuỗi canary yêu cầu không đưa dữ liệu vào kho văn bản huấn luyện. Thư mục data/ không nằm trong git,
nên các file ở data/eval không bị đẩy lên kho mã công khai.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from src.common.answers import extract_gold, normalize_answer
from src.common.config import load_config, path_of, resolve_path
from src.common.io_utils import ensure_dir, iter_jsonl, now_iso, read_json, write_json, write_jsonl
from src.stage_a.a1_prepare import norm

BENCHMARKS = ("gsm8k", "math500", "amc23", "aime24", "aime25", "bbh")
MATH_SETS = ("gsm8k", "math500", "amc23", "aime24", "aime25")        # không được có câu lặp, đáp án là biểu thức toán
BBH_TASKS = (
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "dyck_languages",
    "formal_fallacies", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation",
    "multistep_arithmetic_two", "navigate", "object_counting", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects", "web_of_lies",
    "word_sorting")
LEAK_FILES = ("gsm8k_pool.jsonl", "math_pool.jsonl", "validation.jsonl", "validation_matched.jsonl")
# Lệnh LaTeX bị hỏng do lỗi thoát ký tự ở nguồn: "\\tfrac" bị đọc thành ký tự tab rồi "frac", "\\frac" thành ký tự
# sang trang rồi "rac". Mỗi ký tự điều khiển đi với các phần đuôi lệnh mà nó hay nuốt mất chữ đầu. Chỉ sửa cho AIME:
# đề AIME không có tab thật, còn MATH-500 có tab thật trong lời văn (đo trên tập test của MATH: 22 chỗ) nên không đụng.
LATEX_TAILS = {
    "\t": ("t", ("frac", "imes", "ext", "extbf", "extit", "heta", "an", "riangle", "o", "au", "ilde", "binom", "herefore", "op")),
    "\x0c": ("f", ("rac", "lat", "orall")),
    "\x08": ("b", ("inom", "eta", "oxed", "ar", "egin", "ullet", "mod", "igl", "igr", "ot", "f")),
    "\r": ("r", ("ight", "ho", "angle", "floor", "ceil", "m")),
    "\x0b": ("v", ("ec", "arphi", "arepsilon", "dots")),
    "\x07": ("a", ("lpha", "ngle", "pprox", "st", "cute")),
    "\n": ("n", ("eq", "eg", "abla", "otin", "leq", "geq", "mid", "ewline", "parallel")),
}
CONTROL_CHARS = "\t\x0c\x08\r\x0b\x07"
REPAIR_SETS = ("aime24", "aime25")
CROSS_MIN_RATIO = 0.5            # dưới mức này coi như hai nguồn không cùng một câu
CROSS_NOTE_RATIO = 0.97          # dưới mức này thì in ra để người đọc xem chỗ khác nhau của đề


# ============================================================ phần thuần (không cần mạng)
def pick(row: Mapping, names: Sequence[str]):
    """Giá trị của khoá đầu tiên có trong dòng, không phân biệt hoa thường. Các bộ trên Hugging Face đặt tên cột
    khác nhau (Problem, problem, question), nên tra theo danh sách thay vì viết cứng một tên."""
    lower = {str(k).lower(): k for k in row}
    for n in names:
        if n.lower() in lower:
            return row[lower[n.lower()]]
    raise KeyError(f"Không thấy cột nào trong {list(names)}; dòng có các cột {sorted(map(str, row))}")


def clean_gold(value) -> str:
    """Đáp án về dạng chuỗi gọn: 27.0 thành '27', '025' giữ nguyên, khoảng trắng hai đầu bị bỏ."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value).strip()


def aime_part(text: str) -> str | None:
    """Đề I hay đề II của kỳ thi AIME, đọc từ mã câu hoặc đường dẫn ('2024-II-4', '2024_AIME_I_Problems')."""
    m = re.search(r"(?<![A-Za-z])(II|I)(?![A-Za-z])", str(text).replace("_", " "))
    return m.group(1) if m else None


def aime_gold(value) -> str:
    """Đáp án AIME về số trần: bỏ dấu độ và dấu đô la bao quanh. Bộ AIME 2025 ghi một đáp án là '336^\\circ'
    (câu hỏi về số đo cung tính theo độ); đáp án đúng của đề thi là số nguyên 336."""
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?|°", "", clean_gold(value))
    return s.strip().strip("$").strip()


def diff_excerpt(a: str, b: str, width: int = 170) -> tuple[str, str]:
    """Đoạn khác nhau dài nhất của mỗi bên, so theo từ. Cho thấy hai bản của một đề khác nhau ở chỗ nào."""
    wa, wb = a.split(), b.split()
    ops = [o for o in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes() if o[0] != "equal"]
    only_a = max((" ".join(wa[i1:i2]) for _t, i1, i2, _j1, _j2 in ops), key=len, default="")
    only_b = max((" ".join(wb[j1:j2]) for _t, _i1, _i2, j1, j2 in ops), key=len, default="")
    return only_a[:width], only_b[:width]


def repair_latex(text: str) -> tuple[str, list[str]]:
    """Khôi phục lệnh LaTeX bị hỏng vì ký tự điều khiển (xem LATEX_TAILS). Trả về (văn bản đã sửa, các lệnh đã
    khôi phục). Chỉ sửa khi ngay sau ký tự điều khiển là đúng phần đuôi của một lệnh và sau đó không còn chữ cái."""
    fixed = []
    for ctrl, (letter, tails) in LATEX_TAILS.items():
        pattern = re.compile(re.escape(ctrl) + "(" + "|".join(sorted(tails, key=len, reverse=True)) + ")(?![A-Za-z])")
        text, n = pattern.subn(lambda m: (fixed.append("\\" + letter + m.group(1)) or "\\" + letter + m.group(1)), text)
    return text, fixed


def control_chars(text: str) -> int:
    """Số ký tự điều khiển (tab, sang trang, lùi, về đầu dòng...) trong một đề. Xuống dòng không tính."""
    return sum(text.count(c) for c in CONTROL_CHARS)


def _row(bench: str, qid: str, subset, question, gold, source_id) -> dict:
    return {"qid": qid, "benchmark": bench, "subset": subset, "question": str(question).strip(),
            "gold": gold, "source_id": None if source_id is None else str(source_id)}


def build_gsm8k(rows: Sequence[Mapping]) -> list[dict]:
    """qid mang chữ test để không lẫn với gsm8k_xxxxx của kho huấn luyện (lấy từ split train)."""
    return [_row("gsm8k", f"gsm8k_test_{i:05d}", None, r["question"], extract_gold("gsm8k", r["answer"]), i)
            for i, r in enumerate(rows)]


def build_math500(rows: Sequence[Mapping]) -> list[dict]:
    return [_row("math500", f"math500_{i:05d}", r.get("subject"), r["problem"], clean_gold(r["answer"]),
                 r.get("unique_id", i)) for i, r in enumerate(rows)]


def build_amc23(rows: Sequence[Mapping]) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        m = re.search(r"2023_AMC_(12[AB])", str(r.get("url", "")))
        out.append(_row("amc23", f"amc23_{i:05d}", m.group(1) if m else None, pick(r, ["problem", "question"]),
                        clean_gold(r["answer"]), r.get("url", r.get("id", i))))
    return out


def build_aime(bench: str, rows: Sequence[Mapping], parts: Sequence[str | None] | None = None) -> list[dict]:
    """parts: đề I hay II của từng dòng nếu nguồn đã chia sẵn theo cấu hình; không có thì đọc từ mã câu.
    Dòng nào có đáp án gốc phải làm sạch (aime_gold) thì mang thêm khoá gold_raw ghi nguyên văn của nguồn;
    dòng nào có lệnh LaTeX được khôi phục (repair_latex) thì mang thêm khoá latex_fixed liệt kê các lệnh đó."""
    out = []
    for i, r in enumerate(rows):
        try:
            sid = pick(r, ["id", "unique_id", "problem_id", "url"])
        except KeyError:
            sid = i
        part = parts[i] if parts else aime_part(sid)
        raw = clean_gold(pick(r, ["answer", "expected_answer", "final_answer"]))
        question, fixed = repair_latex(str(pick(r, ["problem", "question"])))
        out.append(_row(bench, f"{bench}_{i:05d}", part, question, aime_gold(raw), sid))
        if out[-1]["gold"] != raw:
            out[-1]["gold_raw"] = raw
        if fixed:
            out[-1]["latex_fixed"] = fixed
    return out


def build_bbh(files: Mapping[str, Sequence[Mapping]]) -> list[dict]:
    """files: tên tác vụ -> danh sách {'input', 'target'} của file bbh/<tác vụ>.json. Đáp án giữ nguyên văn."""
    return [_row("bbh", f"bbh_{task}_{i:04d}", task, ex["input"], str(ex["target"]).strip(), f"{task}/{i}")
            for task in sorted(files) for i, ex in enumerate(files[task])]


def sample_per_subset(rows: Sequence[Mapping], n: int, seed: int) -> list:
    """n câu mỗi tập con, cố định theo sha256 của seed|qid, giữ thứ tự của file. Tập con có ít hơn n câu thì lấy
    hết. c1_evaluate dùng hàm này cho BBH để mọi phương án và mọi máy chấm trên cùng những câu đó."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["subset"], []).append(r)
    keep = set()
    for group in by.values():
        ranked = sorted(group, key=lambda r: hashlib.sha256(f"{seed}|{r['qid']}".encode()).hexdigest())
        keep.update(r["qid"] for r in ranked[:n])
    return [r for r in rows if r["qid"] in keep]


def check_rows(bench: str, rows: Sequence[Mapping], expected: int) -> list[str]:
    """Các lỗi của một bộ: sai số câu, thiếu đề, thiếu đáp án, trùng qid, và (bộ toán) câu lặp hay đáp án lạ."""
    errs = []
    if len(rows) != expected:
        errs.append(f"{bench}: có {len(rows)} câu, đã ghim {expected}")
    empty = [r["qid"] for r in rows if not r["question"] or r["gold"] in (None, "")]
    if empty:
        errs.append(f"{bench}: {len(empty)} câu thiếu đề hoặc thiếu đáp án, ví dụ {empty[:3]}")
    if len({r["qid"] for r in rows}) != len(rows):
        errs.append(f"{bench}: có qid trùng nhau")
    if bench in MATH_SETS:
        dup = len(rows) - len({norm(r["question"]) for r in rows})
        if dup:
            errs.append(f"{bench}: {dup} câu lặp lại trong cùng bộ")
        blank = [r["qid"] for r in rows if r["gold"] and not normalize_answer(r["gold"])]
        if blank:
            errs.append(f"{bench}: {len(blank)} đáp án bị bộ chấm chuẩn hoá thành rỗng, ví dụ {blank[:3]}")
    if bench in ("aime24", "aime25", "amc23"):
        bad = [r["qid"] for r in rows if not re.fullmatch(r"-?\d+", r["gold"] or "")]
        if bad:
            errs.append(f"{bench}: {len(bad)} đáp án không phải số nguyên, ví dụ {bad[:3]}")
    if bench in ("aime24", "aime25", "amc23", "gsm8k"):
        ctrl = [r["qid"] for r in rows if control_chars(r["question"])]
        if ctrl:
            errs.append(f"{bench}: {len(ctrl)} đề còn ký tự điều khiển (lệnh LaTeX hỏng chưa khôi phục được), ví dụ {ctrl[:3]}")
    if bench in ("aime24", "aime25"):
        out_of_range = [r["qid"] for r in rows if re.fullmatch(r"\d+", r["gold"] or "") and int(r["gold"]) > 999]
        if out_of_range:
            errs.append(f"{bench}: {len(out_of_range)} đáp án ngoài khoảng 0 đến 999, ví dụ {out_of_range[:3]}")
    return errs


def describe_rows(bench: str, rows: Sequence[Mapping]) -> str:
    subsets = Counter(r["subset"] for r in rows)
    if bench == "bbh":
        dup = len(rows) - len({(r["subset"], norm(r["question"])) for r in rows})
        return f"{len(subsets)} tác vụ, từ {min(subsets.values())} đến {max(subsets.values())} câu; {dup} câu lặp sẵn trong bộ gốc"
    if set(subsets) == {None}:
        return "không chia tập con"
    shown = ", ".join(f"{k}: {v}" for k, v in sorted(subsets.items(), key=lambda kv: str(kv[0]))[:8])
    return shown + (" ..." if len(subsets) > 8 else "")


def cross_check(primary: Sequence[Mapping], reference: Sequence[Mapping]) -> dict:
    """So hai nguồn độc lập của cùng một bộ. Ghép mỗi câu của nguồn chính với câu giống nhất của nguồn đối chiếu
    (so chữ và số, bỏ khoảng trắng và ký hiệu), rồi so đáp án dưới dạng số nguyên.

    Trả về errors (ghép không một-một, không tìm thấy câu tương ứng, lệch đáp án) và notes (đề khác nhau đáng kể:
    đây là chỗ một trong hai nguồn đã sửa đề, để người đọc xem chứ không phải lỗi)."""
    key = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
    ref_keys = [key(r["question"]) for r in reference]
    errors, notes, used = [], [], Counter()
    for p in primary:
        pk = key(p["question"])
        ratios = [difflib.SequenceMatcher(None, pk, rk, autojunk=False).ratio() for rk in ref_keys]
        j = max(range(len(ratios)), key=ratios.__getitem__) if ratios else None
        if j is None or ratios[j] < CROSS_MIN_RATIO:
            errors.append(f"{p['qid']}: không tìm thấy câu tương ứng ở nguồn đối chiếu")
            continue
        used[j] += 1
        r = reference[j]
        same = (int(p["gold"]) == int(r["gold"])) if re.fullmatch(r"-?\d+", p["gold"]) and \
            re.fullmatch(r"-?\d+", r["gold"]) else normalize_answer(p["gold"]) == normalize_answer(r["gold"])
        if not same:
            errors.append(f"{p['qid']}: đáp án {p['gold']!r}, nguồn đối chiếu ({r['qid']}) ghi {r['gold']!r}")
        if ratios[j] < CROSS_NOTE_RATIO:
            only_p, only_r = diff_excerpt(p["question"], r["question"])
            notes.append({"qid": p["qid"], "source_id": p["source_id"], "reference_qid": r["qid"],
                          "similarity": round(ratios[j], 3), "primary_chars": len(p["question"]),
                          "reference_chars": len(r["question"]), "primary_only": only_p, "reference_only": only_r})
    twice = [reference[j]["qid"] for j, c in used.items() if c > 1]
    if twice:
        errors.append(f"nhiều câu của nguồn chính cùng ghép vào một câu của nguồn đối chiếu: {twice[:3]}")
    if len(primary) != len(reference):
        errors.append(f"nguồn chính có {len(primary)} câu, nguồn đối chiếu có {len(reference)}")
    return {"errors": errors, "notes": notes, "matched": sum(1 for c in used.values() if c == 1)}


def leak_check(eval_rows: Mapping[str, Sequence[Mapping]], protected: Mapping[str, set]) -> dict[str, list[str]]:
    """Câu đánh giá trùng với kho huấn luyện hay tập kiểm định. protected: tên file -> tập đề đã qua norm.
    Trả về tên file -> danh sách qid đánh giá bị trùng; rỗng nghĩa là không rò rỉ."""
    hits: dict[str, list[str]] = {}
    for name, seen in protected.items():
        found = [r["qid"] for rows in eval_rows.values() for r in rows if norm(r["question"]) in seen]
        if found:
            hits[name] = found
    return hits


def read_protected(cfg: Mapping) -> tuple[dict[str, set], list[str]]:
    """Đề của kho huấn luyện và các tập kiểm định. Trả về (tên file -> tập đề đã norm, các file không thấy)."""
    raw = path_of(cfg, "data_raw")
    paths = [raw / n for n in LEAK_FILES] + [path_of(cfg, "data_stage_a") / cfg["stage_a_files"]["questions"]]
    protected, missing = {}, []
    for p in paths:
        if p.exists():
            protected[p.name] = {norm(r["question"]) for r in iter_jsonl(p)}
        else:
            missing.append(p.name)
    return protected, missing


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def combined_md5(parts: Mapping[str, str]) -> str:
    """Một md5 cho cả nhóm file: md5 của danh sách 'tên md5' xếp theo tên."""
    return hashlib.md5("\n".join(f"{k} {parts[k]}" for k in sorted(parts)).encode()).hexdigest()


def verify(eval_dir: Path) -> list[str]:
    """So md5 và số dòng của các file đang có với manifest. Trả về danh sách sai lệch; rỗng là khớp."""
    mpath = eval_dir / "manifest.json"
    if not mpath.exists():
        return [f"không thấy {mpath}; chép cả thư mục data/eval từ máy đã chạy c0_prepare_eval"]
    errs = []
    for bench, info in read_json(mpath)["benchmarks"].items():
        p = eval_dir / info["file"]
        if not p.exists():
            errs.append(f"{bench}: thiếu file {p.name}")
        elif file_md5(p) != info["md5"]:
            errs.append(f"{bench}: md5 của {p.name} khác manifest (file bị sửa hoặc chép hỏng)")
    return errs


# ============================================================ phần cần mạng
def fetch(url: str, md5: str | None = None, tries: int = 3) -> bytes:
    """Tải một file qua HTTPS. Có md5 thì nội dung phải khớp, lệch là dừng."""
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "qd-rsr"}), timeout=60) as f:
                data = f.read()
            break
        except Exception as e:                       # mạng chập chờn: thử lại vài lần rồi mới bỏ
            last = e
            time.sleep(2 * (attempt + 1))
    else:
        raise SystemExit(f"Không tải được {url}: {last}")
    if md5 and hashlib.md5(data).hexdigest() != md5:
        raise SystemExit(f"{url}: md5 {hashlib.md5(data).hexdigest()} khác md5 đã ghim {md5}. Nguồn đã đổi nội dung.")
    return data


def github_raw(src: Mapping, path: str) -> str:
    return f"https://raw.githubusercontent.com/{src['repo']}/{src['commit']}/{path}"


def fetch_jsonl(src: Mapping) -> list[dict]:
    text = fetch(github_raw(src, src["path"]), src.get("md5")).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_hf(repo: str, config: str | None = None, split: str | None = None) -> list[dict]:
    """Các dòng của một bộ trên Hugging Face. Không chỉ split thì nhận bộ chỉ có một split, hoặc split test."""
    from datasets import load_dataset  # nhập muộn: phần thuần và --verify chạy được khi không có thư viện

    args = [repo] + ([config] if config else [])
    if split:
        return list(load_dataset(*args, split=split))
    ds = load_dataset(*args)
    names = list(ds)
    if len(names) == 1 or "test" in names:
        return list(ds["test" if "test" in names else names[0]])
    raise SystemExit(f"{repo} có các split {names}; ghi rõ split trong configs/data.yaml")


def hf_revision(repo: str) -> str | None:
    """Commit hiện tại của bộ dữ liệu trên Hugging Face, ghi vào manifest để biết đã tải bản nào."""
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


def download(bench: str, src: Mapping) -> tuple[list[dict], dict]:
    """Tải một bộ và dựng các dòng chuẩn. Trả về (các dòng, thông tin nguồn để ghi vào manifest)."""
    info = {k: src[k] for k in ("kind", "repo", "commit", "path", "dir", "config", "configs", "split") if k in src}
    if src["kind"] == "hf":
        info["revision"] = hf_revision(src["repo"])
    if bench == "gsm8k":
        return build_gsm8k(load_hf(src["repo"], src.get("config"), src.get("split"))), info
    if bench == "math500":
        return build_math500(load_hf(src["repo"], src.get("config"), src.get("split"))), info
    if bench == "amc23":
        return build_amc23(fetch_jsonl(src)), info
    if bench in ("aime24", "aime25"):
        if src.get("configs"):
            rows, parts = [], []
            for c in src["configs"]:
                got = load_hf(src["repo"], c, src.get("split"))
                rows += got
                parts += [aime_part(c)] * len(got)
            return build_aime(bench, rows, parts), info
        return build_aime(bench, load_hf(src["repo"], src.get("config"), src.get("split"))), info
    if bench == "bbh":
        files, md5s = {}, {}
        for task in BBH_TASKS:
            data = fetch(github_raw(src, f"{src['dir']}/{task}.json"))
            md5s[task] = hashlib.md5(data).hexdigest()
            files[task] = json.loads(data)["examples"]
        got = combined_md5(md5s)
        if src.get("md5") and got != src["md5"]:
            raise SystemExit(f"BBH: md5 gộp {got} khác md5 đã ghim {src['md5']}. Nguồn đã đổi nội dung.")
        info["md5_raw"] = got
        return build_bbh(files), info
    raise SystemExit(f"Không biết bộ đánh giá '{bench}'. Có: {list(BENCHMARKS)}")


# ============================================================ dòng lệnh
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=BENCHMARKS, help="chỉ làm lại các bộ này")
    ap.add_argument("--dry-run", action="store_true", help="tải, kiểm, báo cáo, không ghi file")
    ap.add_argument("--verify", action="store_true", help="không tải; so md5 các file đang có với manifest")
    ap.add_argument("--skip-leak-check", action="store_true",
                    help="bỏ phép kiểm rò rỉ (chỉ khi máy này không có kho câu hỏi; manifest sẽ ghi là chưa kiểm)")
    ap.add_argument("--outdir", help="mặc định paths.data_eval")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.override)
    eval_dir = resolve_path(cfg, args.outdir) if args.outdir else path_of(cfg, "data_eval")
    if args.verify:
        errs = verify(eval_dir)
        for e in errs:
            print(f"[c0] SAI: {e}")
        if not errs:
            m = read_json(eval_dir / "manifest.json")
            print(f"[c0] {eval_dir} khớp manifest: "
                  + ", ".join(f"{b} {i['n']}" for b, i in m["benchmarks"].items())
                  + f" | kiểm rò rỉ: {m['leak_check']['status']}")
        return 1 if errs else 0

    sources = cfg["eval_data"]["sources"]
    todo = list(args.only) if args.only else list(BENCHMARKS)
    data, infos, errors = {}, {}, []
    for bench in todo:
        print(f"[c0] tải {bench} từ {sources[bench]['repo']} ...", flush=True)
        data[bench], infos[bench] = download(bench, sources[bench])
        errors += check_rows(bench, data[bench], int(sources[bench]["n"]))
        print(f"[c0]   {len(data[bench])} câu | {describe_rows(bench, data[bench])}")
        for r in data[bench]:
            if "gold_raw" in r:
                print(f"[c0]   làm sạch đáp án {r['qid']}: {r['gold_raw']!r} thành {r['gold']!r}")
            if "latex_fixed" in r:
                print(f"[c0]   khôi phục lệnh LaTeX bị hỏng ở {r['qid']} ({r['source_id']}): {', '.join(r['latex_fixed'])}")
        n_ctrl = sum(1 for r in data[bench] if control_chars(r["question"]))
        if n_ctrl:
            print(f"[c0]   {n_ctrl} đề có ký tự tab hoặc ký tự điều khiển khác trong lời văn (giữ nguyên như nguồn)")

    cross = None
    if "aime24" in data and sources["aime24"].get("cross_check"):
        ref_src = sources["aime24"]["cross_check"]
        print(f"[c0] đối chiếu aime24 với {ref_src['repo']} ...", flush=True)
        cross = cross_check(data["aime24"], build_aime("aime24_ref", fetch_jsonl(ref_src)))
        errors += [f"aime24 đối chiếu: {e}" for e in cross["errors"]]
        print(f"[c0]   ghép được {cross['matched']}/{len(data['aime24'])} câu, {len(cross['errors'])} lỗi, "
              f"{len(cross['notes'])} câu có đề khác nhau đáng kể giữa hai nguồn")
        for n in cross["notes"]:
            print(f"[c0]   {n['qid']} ({n['source_id']}, giống {n['similarity']:.2f}), đoạn khác nhau dài nhất:\n"
                  f"        nguồn chính ({n['primary_chars']} ký tự): {n['primary_only']!r}\n"
                  f"        đối chiếu   ({n['reference_chars']} ký tự): {n['reference_only']!r}")

    if args.skip_leak_check:
        leak = {"status": "chưa kiểm (--skip-leak-check)", "files": [], "hits": {}}
        print("[c0] BỎ QUA phép kiểm rò rỉ theo yêu cầu.")
    else:
        protected, missing = read_protected(cfg)
        if missing:
            errors.append(f"thiếu file để kiểm rò rỉ: {missing}. Chạy trên máy có đủ data/raw và data/stage_a, "
                          f"hoặc thêm --skip-leak-check")
        hits = leak_check(data, protected)
        errors += [f"RÒ RỈ: {len(q)} câu đánh giá có mặt trong {name}, ví dụ {q[:3]}" for name, q in hits.items()]
        leak = {"status": "không trùng câu nào" if not hits and not missing else "có vấn đề",
                "files": {name: len(s) for name, s in protected.items()}, "hits": hits}
        print(f"[c0] kiểm rò rỉ với {sum(len(s) for s in protected.values())} đề trong {len(protected)} file: "
              f"{sum(map(len, hits.values()))} câu trùng")

    if errors:
        for e in errors:
            print(f"[c0] SAI: {e}")
        print("[c0] Dừng, không ghi file nào.")
        return 1
    total = sum(len(v) for v in data.values())
    if args.dry_run:
        print(f"[c0] Chạy thử: {total} câu của {len(data)} bộ đều qua các phép kiểm. Không ghi gì.")
        return 0

    ensure_dir(eval_dir)
    mpath = eval_dir / "manifest.json"
    manifest = read_json(mpath) if mpath.exists() else {"benchmarks": {}}
    for bench, rows in data.items():
        write_jsonl(eval_dir / f"{bench}.jsonl", rows)
        manifest["benchmarks"][bench] = {"file": f"{bench}.jsonl", "n": len(rows), "md5": file_md5(eval_dir / f"{bench}.jsonl"),
                                         "subsets": len({r["subset"] for r in rows}), "source": infos[bench],
                                         "gold_fixes": [{"qid": r["qid"], "raw": r["gold_raw"], "gold": r["gold"]}
                                                        for r in rows if "gold_raw" in r],
                                         "latex_fixes": [{"qid": r["qid"], "commands": r["latex_fixed"]}
                                                         for r in rows if "latex_fixed" in r],
                                         "prepared": now_iso()}
    manifest["leak_check"] = leak
    if cross is not None:
        manifest["aime24_cross_check"] = {"matched": cross["matched"], "notes": cross["notes"]}
    write_json(mpath, manifest)
    print(f"[c0] Đã ghi {total} câu của {len(data)} bộ vào {eval_dir}. Chép cả thư mục sang máy GPU rồi chạy "
          f"python -m src.stage_c.c0_prepare_eval --verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())

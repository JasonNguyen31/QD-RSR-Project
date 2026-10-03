"""segment_steps: cắt chuỗi suy luận thành các bước bằng GLM-4.5-Air, cho Local Naturalness đúng bài gốc.

Bài gốc (Just, Ko, Jia, bản mới nhất của arXiv 2510.03988, đổi tên phương pháp thành LALP) cắt mỗi lời giải thành
các NHÓM CÂU là các bước lập luận, bằng GLM-4.5-Air với lời nhắc ở phụ lục B.1, rồi chấm từng bước dưới ngữ cảnh
là đề bài và vài bước liền trước. Cột local_nat hiện có trong file fit cắt bước theo dấu câu, tức chỉ là bản
xấp xỉ. Công cụ này làm phần cắt bước; việc chấm lại LocalNat trên các bước mới là việc của b1_fit, làm sau.

    python -m src.tools.segment_steps --pilot 200            # lô thử: 200 chuỗi rút cố định, khoảng 0,2 đô
    python -m src.tools.segment_steps --report               # đọc lại kết quả đã có, không gọi mạng
    python -m src.tools.segment_steps --all --max-cost 15    # mọi ứng viên, sau khi đã đọc kết quả lô thử

Đọc  data/stage_a/candidates.jsonl, questions.jsonl, quality.jsonl
Ghi  data/stage_a/steps.glm.jsonl     một dòng mỗi chuỗi: mốc kết thúc của từng bước (vị trí ký tự), trạng thái,
                                      số token và chi phí. Chạy lại thì bỏ qua chuỗi đã có.

Lời nhắc KHÔNG nằm trong mã: nó đọc từ configs/prompts/lalp_segment.txt, nơi cần dán nguyên văn lời nhắc của phụ
lục B.1. File đó chưa có lời nhắc thì công cụ dừng. Mô hình và cài đặt (tắt chế độ suy nghĩ) ở configs/models.yaml,
mục segmenter.

Vì sao không dùng thẳng văn bản mô hình trả về: mô hình chép lại lời giải vào JSON, mà LaTeX trong JSON rất dễ hỏng
(\\frac thành ký tự điều khiển cộng "rac", \\text thành dấu tab cộng "ext"), và mô hình có thể sửa khoảng trắng hoặc
bỏ sót câu. Nếu chấm trên văn bản trả về thì mô hình học sẽ chấm một chuỗi khác chuỗi dùng để huấn luyện. Vì thế
công cụ chỉ lấy từ mô hình VỊ TRÍ các ranh giới bước, rồi áp các ranh giới đó lên văn bản GỐC:
  1. Đọc JSON theo hai cách (chuẩn, và coi mọi dấu gạch chéo ngược là ký tự thường) để chịu được LaTeX hỏng.
  2. Với mỗi nhóm, tìm chỗ bắt đầu của nó trong văn bản gốc bằng cách so phần chữ và số ở đầu nhóm.
  3. Bước i kéo dài từ chỗ bắt đầu của nhóm i tới chỗ bắt đầu của nhóm i+1. Các bước vì vậy luôn liền nhau và
     phủ kín văn bản gốc; mô hình có bỏ sót câu thì câu đó vẫn nằm trong một bước.
Nhóm nào không tìm được chỗ bắt đầu thì gộp vào bước trước và được đếm lại. Mỗi dòng kết quả ghi ba số đo mức
"giữ nguyên câu chữ": tỷ lệ câu trả về có mặt nguyên văn trong chuỗi gốc, tỷ lệ độ dài, số nhóm không định vị được.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import JsonlWriter, now_iso, read_jsonl
from src.stage_b.signals import sentence_ends

OUT_NAME = "steps.glm.jsonl"
PLACEHOLDER = "CHƯA CÓ LỜI NHẮC"
KEY_LENGTHS = (40, 24, 12)       # số ký tự chữ và số ở đầu nhóm dùng để định vị, thử từ dài tới ngắn
MIN_KEY = 6
FUZZY_MIN = 8                    # khớp gần đúng phải dài ít nhất ngần này ký tự
DONE = ("ok", "partial")         # các trạng thái không cần gọi lại mô hình


# ============================================================ lời nhắc
def load_prompt(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Không thấy file lời nhắc {path}.")
    text = path.read_text(encoding="utf-8")
    if PLACEHOLDER in text or "{problem}" not in text or "{solution}" not in text:
        raise SystemExit(f"{path} chưa có lời nhắc. Dán vào đó nguyên văn lời nhắc ở phụ lục B.1 của bài gốc "
                         f"(arXiv 2510.03988), giữ hai chỗ điền {{problem}} và {{solution}}.")
    return text


def build_prompt(template: str, problem: str, solution: str) -> str:
    """Thay hai chỗ điền bằng replace, không dùng format: lời nhắc có khung JSON với dấu ngoặc nhọn."""
    return template.replace("{problem}", problem).replace("{solution}", solution)


# ============================================================ đọc JSON mô hình trả về
def _json_body(raw: str) -> str | None:
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start:end + 1] if 0 <= start < end else None


def _groups_of(obj) -> list[list[str]] | None:
    """Rút danh sách nhóm câu từ JSON đã đọc. Chấp nhận nhóm là danh sách câu hoặc một chuỗi."""
    if isinstance(obj, Mapping) and "sentence_groups" in obj:
        obj = obj["sentence_groups"]
    if isinstance(obj, Mapping):
        obj = list(obj.values())
    if not isinstance(obj, list) or not obj:
        return None
    groups = []
    for g in obj:
        if isinstance(g, str):
            g = [g]
        if not isinstance(g, list):
            return None
        sents = [s for s in g if isinstance(s, str) and s.strip()]
        if sents:
            groups.append(sents)
    return groups or None


def _repair(body: str, literal: bool) -> str:
    """Sửa các dấu gạch chéo ngược trong thân JSON, xét TỪNG CẶP (gạch chéo, ký tự sau) từ trái sang phải để một
    cặp đã thoát đúng (hai gạch chéo liền nhau) không bị đọc nhầm thành hai lần thoát."""
    def fix(m: re.Match) -> str:
        ch = m.group(1)
        if ch in '"\\':
            return m.group(0)
        if not literal:
            if ch in "/bfnrt":
                return m.group(0)
            if ch == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", body[m.end():m.end() + 4] or ""):
                return m.group(0)
        return "\\\\" + ch
    return re.sub(r"\\(.)", fix, body, flags=re.S)


def parse_groups(raw: str) -> dict | None:
    """Đọc phản hồi theo hai cách. Trả về {'standard': nhóm hoặc None, 'literal': nhóm hoặc None}, hay None nếu
    cả hai cách đều hỏng.

    standard  đọc JSON đúng chuẩn; các dấu gạch chéo ngược không hợp lệ (\\(, \\s, ...) được nhân đôi trước.
    literal   coi MỌI dấu gạch chéo ngược không đứng trước dấu nháy là ký tự thường. Cách này giữ được \\frac,
              \\text, \\boxed, \\nu (cách chuẩn biến chúng thành ký tự điều khiển), đổi lại một dấu xuống dòng
              viết là \\n sẽ thành chữ "n".
    """
    body = _json_body(raw)
    if body is None:
        return None
    out = {}
    for name in ("standard", "literal"):
        try:
            out[name] = _groups_of(json.loads(_repair(body, literal=name == "literal")))
        except (json.JSONDecodeError, RecursionError):
            out[name] = None
    return out if any(out.values()) else None


# ============================================================ áp ranh giới lên văn bản gốc
def alnum_index(text: str) -> tuple[str, list[int]]:
    """Chuỗi chỉ gồm chữ và số, kèm vị trí của từng ký tự đó trong văn bản gốc. Không đổi hoa thường: với vài
    ký tự Unicode việc đổi làm thay số ký tự và lệch bảng vị trí."""
    chars, index = [], []
    for i, ch in enumerate(text):
        if ch.isalnum():
            chars.append(ch)
            index.append(i)
    return "".join(chars), index


def _alnum(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def locate(variants: Sequence[str], original: str, cursor: int) -> int | None:
    """Vị trí (trong chuỗi chữ-số của văn bản gốc) nơi một nhóm bắt đầu, tìm từ cursor trở đi."""
    for length in KEY_LENGTHS:
        for v in variants:
            key = v[:length]
            if len(key) >= min(length, MIN_KEY):
                pos = original.find(key, cursor)
                if pos >= 0:
                    return pos
    for v in variants:                                   # khớp gần đúng, khi đầu nhóm bị hỏng vài ký tự
        if len(v) < FUZZY_MIN:
            continue
        window = original[cursor:cursor + 4 * len(v) + 400]
        m = SequenceMatcher(None, window, v[:300], autojunk=False).find_longest_match(0, len(window), 0, min(len(v), 300))
        if m.size >= FUZZY_MIN and m.b <= 60:
            return cursor + max(0, m.a - m.b)
    return None


def _symbols(text: str) -> str:
    """Bỏ khoảng trắng, dấu gạch chéo ngược và ký tự điều khiển: phần còn lại so được dù LaTeX bị hỏng lối thoát."""
    return "".join(ch for ch in text if not ch.isspace() and ch != "\\" and ord(ch) >= 32)


def cut_position(text: str, c: int, first_sentence: str) -> int:
    """Vị trí ký tự bắt đầu một bước, biết c là vị trí ký tự chữ-số đầu tiên của nó trong văn bản gốc.

    Nhóm thường mở đầu bằng ký hiệu không phải chữ số ("\\[ x = 1", "$y$", "**Step"). Phần ký hiệu đó lấy từ câu
    đầu mà mô hình trả về, rồi dò ngược trong văn bản gốc; khớp thì ranh giới lùi về trước nó, để "\\[" không bị
    bỏ lại ở cuối bước trước. Không khớp thì chỉ lùi qua các ký hiệu dính liền, tới khoảng trắng gần nhất.
    """
    lead = ""
    for ch in first_sentence:
        if ch.isalnum():
            break
        lead += ch
    want, k, matched = _symbols(lead), c, 0
    while want and k > 0 and matched < len(want):
        ch = text[k - 1]
        if ch.isspace() or ch == "\\":
            k -= 1
        elif ch == want[len(want) - 1 - matched]:
            matched += 1
            k -= 1
        else:
            break
    if want and matched == len(want):
        while k > 0 and text[k - 1] == "\\":
            k -= 1
        return k
    while c > 0 and not text[c - 1].isspace():
        c -= 1
    return c


def _verbatim(groups: Sequence[Sequence[str]], original: str) -> int:
    return sum(1 for g in groups for s in g if _alnum(s) and _alnum(s) in original)


def align(text: str, parsed: Mapping) -> dict:
    """Áp các nhóm mô hình trả về lên văn bản gốc. Trả về mốc kết thúc từng bước và các số đo độ trung thành."""
    original, index = alnum_index(text)
    readings = [g for g in (parsed.get("standard"), parsed.get("literal")) if g]
    # Bản đọc chính là bản có nhiều câu khớp nguyên văn với chuỗi gốc hơn (hoà thì lấy bản đọc chuẩn).
    groups = max(readings, key=lambda g: _verbatim(g, original))
    other = next((g for g in readings if g is not groups and len(g) == len(groups)), None)

    cursor, starts, unlocated = 0, [], 0                  # starts: (vị trí chữ-số, câu đầu của nhóm)
    for gi, group in enumerate(groups):
        variants = [_alnum("".join(group))] + ([_alnum("".join(other[gi]))] if other else [])
        variants = [v for v in dict.fromkeys(variants) if v]
        pos = locate(variants, original, cursor) if variants else None
        if pos is None:
            unlocated += 1
            continue
        if not starts or pos > starts[-1][0]:
            starts.append((pos, group[0]))
        cursor = pos + max(1, len(variants[0]) // 2)

    # Nhóm định vị được đầu tiên không tạo ranh giới: bước đầu luôn bắt đầu ở ký tự 0.
    cuts = []
    for pos, first in starts[1:]:
        c = cut_position(text, index[pos], first)
        if c > (cuts[-1] if cuts else 0):
            cuts.append(c)
    ends = cuts + [len(text)]

    n_sent = sum(len(g) for g in groups)
    return {"ends": ends, "n_groups": len(groups), "n_steps": len(ends), "unlocated": unlocated,
            "n_sentences": n_sent,
            "verbatim_share": _verbatim(groups, original) / n_sent if n_sent else 0.0,
            "length_ratio": len(_alnum("".join(s for g in groups for s in g))) / len(original) if original else 0.0}


def segment_one(client, seg: Mapping, template: str, chain: Mapping, question: str, keep_raw: bool) -> dict:
    """Gọi mô hình cho một chuỗi và trả về một dòng kết quả. Không ném lỗi: lỗi nào cũng thành một trạng thái."""
    row = {"tid": chain["tid"], "qid": chain["qid"], "model": seg["model_id"], "ts": now_iso(),
           "text_md5": hashlib.md5(chain["text"].encode("utf-8")).hexdigest(), "n_chars": len(chain["text"])}
    try:
        res = client.chat(seg["model_id"], [{"role": "user", "content": build_prompt(template, question, chain["text"])}],
                          temperature=float(seg["temperature"]), top_p=1.0, max_tokens=int(seg["max_tokens"]),
                          provider_order=seg.get("provider_order") or None,
                          extra_body={"reasoning": {"enabled": bool(seg["reasoning_enabled"])}})
    except Exception as exc:                                     # lỗi mạng sau khi đã thử lại: để lần chạy sau làm
        return {**row, "status": "api_error", "error": f"{type(exc).__name__}: {exc}"[:300]}
    row.update(prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens, cost=res.cost,
               finish_reason=res.finish_reason, served_model=res.served_model)
    if keep_raw:
        row["raw"] = res.text
    parsed = parse_groups(res.text)
    if parsed is None:
        return {**row, "status": "truncated" if res.finish_reason == "length" else "json_error"}
    a = align(chain["text"], parsed)
    status = "ok" if a["unlocated"] == 0 and res.finish_reason != "length" else "partial"
    return {**row, **a, "status": status}


# ============================================================ chọn chuỗi và báo cáo
def usable_chains(cands: Sequence[Mapping], quality: Mapping[str, Mapping]) -> list:
    """Các ứng viên có Qual, tức đúng những chuỗi mà các phương án chọn được dùng tới."""
    return [c for c in cands if quality.get(c["tid"], {}).get("qual") is not None]


def pilot_sample(chains: Sequence[Mapping], n: int, seed: int) -> list:
    """n chuỗi rút cố định theo sha256(seed|tid): chạy lại hay đổi máy vẫn ra đúng các chuỗi đó."""
    order = sorted(chains, key=lambda c: hashlib.sha256(f"{seed}|{c['tid']}".encode("utf-8")).hexdigest())
    return order[:n]


def _pct(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def report(rows: Sequence[Mapping], chains: Mapping[str, Mapping], total_chars: int) -> None:
    if not rows:
        print("[bước] chưa có kết quả nào.")
        return
    status = Counter(r["status"] for r in rows)
    done = [r for r in rows if r["status"] in DONE]
    print(f"[bước] {len(rows)} chuỗi: " + ", ".join(f"{k} {v}" for k, v in sorted(status.items())))
    billed = [r for r in rows if r.get("cost") is not None]
    if billed:
        cost = sum(r["cost"] for r in billed)
        chars = sum(r["n_chars"] for r in billed)
        print(f"[bước] chi phí {cost:.4f} đô cho {len(billed)} lượt; trung bình {statistics.mean(r['prompt_tokens'] for r in billed):.0f} "
              f"token vào, {statistics.mean(r['completion_tokens'] for r in billed):.0f} token ra mỗi chuỗi")
        if chars and total_chars:
            print(f"[bước] ước cho toàn bộ {len(chains)} chuỗi dùng được (theo số ký tự): {cost / chars * total_chars:.2f} đô")
    if not done:
        return
    steps = [r["n_steps"] for r in done]
    sent = [len(sentence_ends(chains[r["tid"]]["text"])) for r in done if r["tid"] in chains]
    print(f"[bước] số bước mỗi chuỗi: trung vị {statistics.median(steps):.0f} (phân vị 10 và 90: {_pct(steps, 0.1):.0f}, "
          f"{_pct(steps, 0.9):.0f}); cắt theo dấu câu cho trung vị {statistics.median(sent):.0f} câu")
    print(f"[bước] chuỗi chỉ có 1 bước: {sum(s == 1 for s in steps)}; chuỗi có nhóm không định vị được: "
          f"{sum(r['unlocated'] > 0 for r in done)} (tổng {sum(r['unlocated'] for r in done)} nhóm)")
    verb = [r["verbatim_share"] for r in done]
    ratio = [r["length_ratio"] for r in done]
    print(f"[bước] giữ nguyên câu chữ: trung bình {statistics.mean(verb):.1%} số câu trả về có mặt nguyên văn trong chuỗi "
          f"gốc; {sum(v >= 0.95 for v in verb)}/{len(done)} chuỗi đạt từ 95% trở lên")
    print(f"[bước] tỷ lệ độ dài trả về trên gốc: trung vị {statistics.median(ratio):.3f}; ngoài khoảng 0,95 đến 1,05: "
          f"{sum(not 0.95 <= x <= 1.05 for x in ratio)} chuỗi (mô hình bỏ sót hoặc thêm nội dung)")
    print("[bước] Các bước luôn phủ kín chuỗi gốc dù mô hình bỏ sót câu; ba số trên cho biết ranh giới đáng tin tới đâu.")


# ============================================================ dòng lệnh
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--pilot", type=int, metavar="N", help="chạy lô thử N chuỗi rút cố định")
    ap.add_argument("--all", action="store_true", help="chạy mọi ứng viên dùng được; bắt buộc có --max-cost")
    ap.add_argument("--report", action="store_true", help="chỉ đọc kết quả đã có")
    ap.add_argument("--max-cost", type=float, help="dừng khi chi phí cộng dồn (đô) chạm mức này; lô thử mặc định 1 đô")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-raw", action="store_true", help="ghi cả phản hồi thô (lô thử luôn ghi)")
    ap.add_argument("--out", default=OUT_NAME)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)
    if sum(bool(x) for x in (args.pilot, args.all, args.report)) != 1:
        ap.error("chọn đúng một trong --pilot N, --all, --report")

    cfg = load_config(overrides=args.override)
    wd = resolve_path(cfg, args.workdir)
    files = cfg["stage_a_files"]
    cands = read_jsonl(wd / files["candidates"])
    quality = {r["tid"]: r for r in read_jsonl(wd / "quality.jsonl")}
    chains = usable_chains(cands, quality)
    if not chains:
        raise SystemExit(f"Không có ứng viên dùng được trong {wd}.")
    by_tid = {c["tid"]: c for c in chains}
    total_chars = sum(len(c["text"]) for c in chains)
    out = wd / args.out
    if args.report:
        report(list({r["tid"]: r for r in read_jsonl(out)}.values()), by_tid, total_chars)
        return 0

    seg = cfg["segmenter"]
    template = load_prompt(resolve_path(cfg, seg["prompt_file"]))
    questions = {q["qid"]: q["question"] for q in read_jsonl(wd / files["questions"])}
    if args.all and args.max_cost is None:
        raise SystemExit("Chạy toàn bộ tốn tiền thật: bắt buộc có --max-cost <số đô>. Xem ước tính ở --report sau lô thử.")
    budget = args.max_cost if args.max_cost is not None else 1.0
    todo = pilot_sample(chains, args.pilot, int(cfg["selection"]["random_seed"])) if args.pilot else chains
    last = {r["tid"]: r for r in read_jsonl(out)}
    stale = [t for t, r in last.items() if t in by_tid
             and r.get("text_md5") != hashlib.md5(by_tid[t]["text"].encode("utf-8")).hexdigest()]
    if stale:
        raise SystemExit(f"{out.name} có {len(stale)} chuỗi mà văn bản đã khác candidates.jsonl. File cũ không còn khớp.")
    todo = [c for c in todo if last.get(c["tid"], {}).get("status") not in DONE]
    print(f"[bước] {seg['model_id']}, suy nghĩ {'bật' if seg['reasoning_enabled'] else 'tắt'}, nhiệt độ {seg['temperature']}; "
          f"cần cắt {len(todo)} chuỗi, trần chi phí {budget:.2f} đô")
    if not todo:
        report(list(last.values()), by_tid, total_chars)
        return 0

    from src.common.api import make_openrouter_client
    client = make_openrouter_client(cfg)
    lock, spent, stopped = threading.Lock(), [0.0], [False]

    def work(chain):
        if stopped[0]:
            return None
        row = segment_one(client, seg, template, chain, questions[chain["qid"]], args.keep_raw or bool(args.pilot))
        with lock:
            spent[0] += row.get("cost") or 0.0
            if spent[0] >= budget:
                stopped[0] = True
        return row

    with JsonlWriter(out) as writer, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, row in enumerate(pool.map(work, todo), 1):
            if row is None:
                continue
            writer.append(row)
            if n % 25 == 0:
                print(f"[bước] {n}/{len(todo)} | đã tiêu {spent[0]:.4f} đô", flush=True)
    if stopped[0]:
        print(f"[bước] DỪNG vì chạm trần chi phí {budget:.2f} đô; chạy lại cùng lệnh để làm tiếp.")
    final = {r["tid"]: r for r in read_jsonl(out)}          # dòng sau cùng của mỗi chuỗi là kết quả hiện hành
    report(list(final.values()), by_tid, total_chars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

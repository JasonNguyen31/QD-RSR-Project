"""Ba phép kiểm đầu vào trước khi viết b2_select (mục "Việc cần đo và kiểm tra lại" trên Notion, 1 đến 3).

    python -m src.tools.check_inputs ceiling  --workdir data/stage_a            # 1. điểm giám khảo dồn trần
    python -m src.tools.check_inputs salvaged --workdir data/stage_a --show 20  # 2. chuỗi lấy điểm qua cứu hộ
    python -m src.tools.check_inputs steps    --workdir data/stage_a --fit fit.qwen1_5b_base.jsonl --show 5
                                                                                # 3. cắt câu của LocalNat

Không gọi mạng, không cần GPU. Chạy được trên cả hai máy; phép 3 cần candidates.jsonl để in chuỗi thật.

1. ceiling. Qual = 0,5 × rule_norm + 0,5 × llm_norm, mỗi nửa min-max TRONG câu hỏi. Ở câu mà mọi chuỗi được
   giám khảo chấm cùng một điểm, llm_norm = 0,5 cho tất cả, nên Qual chỉ còn xếp hạng theo điểm quy tắc.
   Phép này đo điều đó xảy ra ở bao nhiêu câu, và quan trọng hơn: ở bao nhiêu câu tập top-k theo Qual
   KHÁC tập top-k theo riêng điểm quy tắc. Đó là phần đóng góp thật của giám khảo vào việc chọn.
   Chỉ tính các câu có hơn k ứng viên, vì câu có đúng k ứng viên thì phương án nào cũng chọn hết.

2. salvaged. Phản hồi gốc của giám khảo không được lưu, nên không đọc lại được lý giải. Thay vào đó: so phân
   bố điểm cứu hộ với điểm thường, và in từng chuỗi kèm đáp án chuẩn, đuôi chuỗi, điểm quy tắc, điểm giám khảo
   của các chuỗi cùng câu, để người đọc tự đánh giá điểm cứu hộ có hợp lý không.

3. steps. Cắt câu (signals.sentence_ends) mới được thử trên văn bản giả. Phép này đo trên chuỗi thật: số câu
   mỗi chuỗi, độ dài mỗi câu, tỷ lệ chuỗi chỉ có một câu (LocalNat khi đó bằng GRAPE), câu quá dài (cắt thiếu)
   và câu quá ngắn (cắt thừa), rồi in vài chuỗi thật đã cắt để xem bằng mắt.
"""
from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter, defaultdict
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl
from src.stage_b.signals import sentence_ends

LONG_STEP_CHARS = 400      # câu dài hơn mức này nghi là cắt thiếu (nhiều câu dính nhau)
SHORT_STEP_CHARS = 15      # câu ngắn hơn mức này nghi là cắt thừa


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} = {a / b:.1%}" if b else f"{a}/0"


def _quantiles(xs: Sequence[float]) -> str:
    xs = sorted(xs)
    if not xs:
        return "không có"
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return f"p5 {q(0.05)}, trung vị {q(0.5)}, p95 {q(0.95)}, lớn nhất {xs[-1]}"


def top_k(rows: Sequence[Mapping], key: str, k: int) -> frozenset:
    """Tập tid của k chuỗi điểm cao nhất; hoà thì theo tid để kết quả tái lập được."""
    ranked = sorted(rows, key=lambda r: (-float(r[key]), r["tid"]))
    return frozenset(r["tid"] for r in ranked[:k])


# ============================================================ 1. điểm giám khảo dồn trần
def ceiling_stats(quality_rows: Sequence[Mapping], k: int) -> dict:
    by_q: dict[str, list] = defaultdict(list)
    for r in quality_rows:
        if r.get("qual") is not None:            # chuỗi thiếu điểm giám khảo bị loại khỏi mọi phương án
            by_q[r["qid"]].append(r)
    usable = [rs for rs in by_q.values() if len(rs) >= k]
    multi = [rs for rs in usable if len(rs) > k]
    all_scores = [r["llm_score"] for rs in usable for r in rs]

    const = [rs for rs in multi if len({r["llm_score"] for r in rs}) == 1]
    const_top = [rs for rs in const if rs[0]["llm_score"] >= 1.0 - 1e-9]
    distinct = Counter(len({r["llm_score"] for r in rs}) for rs in multi)
    same_as_rule = sum(top_k(rs, "qual", k) == top_k(rs, "rule_norm", k) for rs in multi)
    same_as_llm = sum(top_k(rs, "qual", k) == top_k(rs, "llm_norm", k) for rs in multi)
    # Giám khảo chỉ phân biệt được ở đỉnh nếu chuỗi thứ k và thứ k+1 theo giám khảo có điểm khác nhau.
    tie_at_cut = 0
    for rs in multi:
        s = sorted((r["llm_score"] for r in rs), reverse=True)
        tie_at_cut += abs(s[k - 1] - s[k]) < 1e-9
    return {
        "chains": len(all_scores), "questions": len(usable), "multi": len(multi),
        "top_score_share": sum(s >= 1.0 - 1e-9 for s in all_scores),
        "const": len(const), "const_top": len(const_top), "distinct": dict(sorted(distinct.items())),
        "same_as_rule": same_as_rule, "same_as_llm": same_as_llm, "tie_at_cut": tie_at_cut,
    }


def cmd_ceiling(cfg, wd, args) -> int:
    rows = read_jsonl(wd / "quality.jsonl")
    if not rows:
        raise SystemExit(f"Không thấy {wd / 'quality.jsonl'}.")
    k = cfg["selection"]["k"]
    s = ceiling_stats(rows, k)
    m = s["multi"]
    print(f"[1] Điểm giám khảo dồn trần, k = {k}")
    print(f"    {s['chains']} chuỗi có Qual, {s['questions']} câu dùng được, {m} câu có hơn {k} ứng viên "
          f"(chỉ nhóm này mới thực sự phải chọn)\n")
    print(f"    chuỗi được giám khảo chấm 1,0                  : {_pct(s['top_score_share'], s['chains'])}")
    print(f"    câu mà mọi chuỗi cùng một điểm giám khảo       : {_pct(s['const'], m)}")
    print(f"      trong đó cùng là 1,0                          : {_pct(s['const_top'], m)}")
    print(f"    câu mà chuỗi thứ {k} và {k + 1} theo giám khảo hoà    : {_pct(s['tie_at_cut'], m)}")
    print(f"    số mức điểm giám khảo khác nhau trong một câu  : "
          + ", ".join(f"{d} mức: {n}" for d, n in s["distinct"].items()))
    print(f"\n    tập top-{k} theo Qual TRÙNG tập top-{k} theo riêng điểm quy tắc : {_pct(s['same_as_rule'], m)}")
    print(f"    tập top-{k} theo Qual TRÙNG tập top-{k} theo riêng giám khảo     : {_pct(s['same_as_llm'], m)}")
    print(f"\n    Cách đọc: tỷ lệ trùng với điểm quy tắc là phần câu mà giám khảo KHÔNG đổi được lựa chọn của Qual.")
    print(f"    Nếu tỷ lệ này rất cao, thành phần Qual của QD-RSR thực chất là điểm quy tắc, và paper phải nói")
    print(f"    rõ như vậy (hoặc nhóm cân nhắc lại alpha hay cách chấm giám khảo) TRƯỚC khi chạy huấn luyện.")
    return 0


# ============================================================ 2. chuỗi lấy điểm qua cứu hộ
def cmd_salvaged(cfg, wd, args) -> int:
    judge = read_jsonl(wd / "judge.jsonl")
    if not judge:
        raise SystemExit(f"Không thấy {wd / 'judge.jsonl'}.")
    salv = [r for r in judge if r.get("salvaged")]
    normal = [r["overall_score"] for r in judge if not r.get("salvaged")]
    print(f"[2] Chuỗi lấy điểm qua lớp cứu hộ JSON: {_pct(len(salv), len(judge))}")
    for how, n in Counter(r["salvaged"] for r in salv).most_common():
        sc = [r["overall_score"] for r in salv if r["salvaged"] == how]
        print(f"    {how:<24} {n:>5} chuỗi, điểm trung bình {statistics.mean(sc):.3f}")
    if salv and normal:
        print(f"    so với chuỗi chấm bình thường: trung bình {statistics.mean(normal):.3f}, "
              f"tỷ lệ 1,0 là {sum(s >= 1 - 1e-9 for s in normal) / len(normal):.1%} "
              f"(cứu hộ: {sum(r['overall_score'] >= 1 - 1e-9 for r in salv) / len(salv):.1%})")
    if not salv or not args.show:
        return 0

    cands = {c["tid"]: c for c in read_jsonl(wd / "candidates.jsonl")}
    questions = {q["qid"]: q for q in read_jsonl(wd / "questions.jsonl")}
    qual = {r["tid"]: r for r in read_jsonl(wd / "quality.jsonl")}
    by_q = defaultdict(list)
    for r in judge:
        by_q[r["qid"]].append(r)
    # rút đều theo từng cách cứu hộ, cố định seed để hai người xem cùng một mẫu
    rng = random.Random(cfg["project"]["data_seed"])
    groups = defaultdict(list)
    for r in salv:
        groups[r["salvaged"]].append(r)
    picked = []
    for rs in groups.values():
        picked += rng.sample(rs, min(len(rs), max(1, args.show * len(rs) // len(salv))))
    for r in picked[: args.show]:
        c, q, ql = cands.get(r["tid"], {}), questions.get(r["qid"], {}), qual.get(r["tid"], {})
        text = c.get("text", "")
        others = sorted(((o["overall_score"], "*" if o.get("salvaged") else "") for o in by_q[r["qid"]]
                         if o["tid"] != r["tid"]), reverse=True)
        print("\n" + "-" * 100)
        print(f"{r['tid']}   cứu hộ: {r['salvaged']}   điểm giám khảo {r['overall_score']:.2f}   "
              f"điểm quy tắc {ql.get('rule_score', float('nan')):+.2f}   {len(text.split())} từ")
        print(f"các chuỗi khác cùng câu (dấu * là cũng cứu hộ): "
              + ", ".join(f"{s:.2f}{m}" for s, m in others))
        print(f"đáp án chuẩn: {q.get('gold')}")
        print(f"đề: {q.get('question', '')[:300]}")
        print(f"...đuôi chuỗi: {text[-400:]}")
    print("\n    Cần xem: điểm cứu hộ có lệch hẳn so với các chuỗi khác cùng câu không, nhất là khi chuỗi dài")
    print("    và đúng mà điểm thấp. Kiểu mean_of_N_dimensions tự tính trung bình đều, khác cách giám khảo tổng hợp.")
    return 0


# ============================================================ 3. cắt câu của LocalNat
def step_stats(texts: Sequence[str]) -> dict:
    n_steps, lens = [], []
    for t in texts:
        ends = sentence_ends(t)
        n_steps.append(len(ends))
        start = 0
        for e in ends:
            lens.append(len(t[start:e].strip()))
            start = e
    return {"chains": len(texts), "n_steps": n_steps, "lens": lens,
            "one_step": sum(n == 1 for n in n_steps),
            "long": sum(x > LONG_STEP_CHARS for x in lens), "short": sum(x < SHORT_STEP_CHARS for x in lens)}


def show_split(text: str, limit: int = 1200) -> str:
    parts, start = [], 0
    for e in sentence_ends(text):
        parts.append(text[start:e].strip().replace("\n", " ⏎ "))
        start = e
    out = " ‖ ".join(parts)
    return out if len(out) <= limit else out[:limit] + " …"


def cmd_steps(cfg, wd, args) -> int:
    cands = read_jsonl(wd / "candidates.jsonl")
    if not cands:
        raise SystemExit(f"Không thấy {wd / 'candidates.jsonl'}.")
    s = step_stats([c["text"] for c in cands])
    print(f"[3] Cắt câu của LocalNat trên {s['chains']} chuỗi thật (tính theo ký tự, không cần tokenizer)")
    print(f"    số câu mỗi chuỗi : {_quantiles(s['n_steps'])}")
    print(f"    ký tự mỗi câu    : {_quantiles(s['lens'])}")
    print(f"    chuỗi chỉ có 1 câu (LocalNat khi đó bằng GRAPE)  : {_pct(s['one_step'], s['chains'])}")
    print(f"    câu dài hơn {LONG_STEP_CHARS} ký tự (nghi cắt thiếu)          : {_pct(s['long'], len(s['lens']))}")
    print(f"    câu ngắn hơn {SHORT_STEP_CHARS} ký tự (nghi cắt thừa)          : {_pct(s['short'], len(s['lens']))}")

    if args.fit:
        fit = {r["tid"]: r for r in read_jsonl(wd / args.fit)}
        pairs = [(len(sentence_ends(c["text"])), fit[c["tid"]]["n_steps"]) for c in cands
                 if c["tid"] in fit and "n_steps" in fit[c["tid"]]]
        if pairs:
            same = sum(a == b for a, b in pairs)
            fewer = sum(b < a for a, b in pairs)
            print(f"\n    đối chiếu {args.fit}: n_steps khớp số câu theo ký tự ở {_pct(same, len(pairs))}; "
                  f"ít hơn ở {fewer} chuỗi (b1 cắt chuỗi ở {cfg['training']['max_seq_len']} token, "
                  f"hoặc hai câu rơi vào cùng một token)")

    if args.show:
        rng = random.Random(cfg["project"]["data_seed"])
        # một chuỗi mỗi mô hình dạy trước, rồi rút thêm cho đủ
        by_teacher = defaultdict(list)
        for c in cands:
            by_teacher[c["tid"].split("|")[-2]].append(c)
        picked = [rng.choice(v) for _, v in sorted(by_teacher.items())]
        picked += rng.sample(cands, max(0, args.show - len(picked)))
        for c in picked[: args.show]:
            print("\n" + "-" * 100)
            print(f"{c['tid']}   {len(sentence_ends(c['text']))} câu   (‖ là ranh giới câu, ⏎ là xuống dòng)")
            print(show_split(c["text"]))
    return 0


# ============================================================ dòng lệnh
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["ceiling", "salvaged", "steps"])
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--show", type=int, default=0, help="số chuỗi in ra để xem bằng mắt")
    ap.add_argument("--fit", help="file fit để đối chiếu n_steps (phép steps)")
    args = ap.parse_args(argv)
    cfg = load_config()
    wd = resolve_path(cfg, args.workdir)
    return {"ceiling": cmd_ceiling, "salvaged": cmd_salvaged, "steps": cmd_steps}[args.check](cfg, wd, args)


if __name__ == "__main__":
    raise SystemExit(main())

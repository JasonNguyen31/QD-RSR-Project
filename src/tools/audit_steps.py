"""audit_steps: kiểm các bước cuối cùng mà Local Naturalness sẽ dùng, trước khi tốn giờ GPU để chấm.

    python -m src.tools.audit_steps                  # mọi chuỗi đã có dòng trong steps.glm.jsonl
    python -m src.tools.audit_steps --pilot 200      # chỉ lô thử, để đối chiếu với số đã kiểm tay ngày 03/10
    python -m src.tools.audit_steps --show 8         # in thêm 8 ranh giới bị lùi, trước và sau
    python -m src.tools.audit_steps --list-broken    # một dòng cho mỗi chuỗi cắt hỏng: cứu được hay cắt dự phòng

Đọc  data/stage_a/candidates.jsonl, quality.jsonl, steps.glm.jsonl
Ghi  không gì cả. Không gọi mạng, không cần GPU, chạy được khi segment_steps còn đang ghi file.

Công cụ này không tự xử lý gì: nó gọi src/stage_b/steps.final_steps, đúng hàm mà b1_localnat gọi trên GPU, rồi
đếm. In ra năm phần:
  1. ranh giới nằm ở đâu, bao nhiêu ranh giới bị lùi về đầu dòng và sau dấu đầu dòng nào;
  2. số bước mỗi chuỗi trước và sau khi xử lý;
  3. chuỗi cắt hỏng: cứu được bao nhiêu từ phản hồi thô, bao nhiêu phải cắt dự phòng, và bao nhiêu chuỗi trong
     số đó thuộc câu chỉ có đúng k ứng viên (lý do không loại chúng khỏi kho);
  4. ba quy tắc cắt dự phòng giống cách cắt của GLM tới đâu, đo trên các chuỗi GLM cắt được;
  5. với k = 1, 2, 4 bước ngữ cảnh, mô hình còn thấy bao nhiêu phần đứng trước. Bản mới nhất của bài gốc cho
     biết cửa sổ 5% đến 25% xếp hạng đúng còn 50% đến 75% thì hội tụ về cách chấm toàn cục (GRAPE).
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter
from typing import Mapping, Sequence

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_jsonl
from src.stage_b import steps as st
from src.tools.segment_steps import DONE, OUT_NAME, pilot_sample, usable_chains

LONG_CHARS = 1500
KS = (1, 2, 4)


def _pct(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def _spread(values: Sequence[int]) -> str:
    return f"trung vị {statistics.median(values):.0f} (phân vị 10 và 90: {_pct(values, 0.1):.0f}, {_pct(values, 0.9):.0f})"


def _lengths(ends: Sequence[int]) -> list[int]:
    return [b - a for a, b in zip([0] + list(ends[:-1]), ends)]


def analyse(chains: Sequence[Mapping], rows: Mapping[str, Mapping], k_select: int,
            per_q: Mapping[str, int] | None = None) -> dict:
    """Chạy final_steps trên mọi chuỗi đã có dòng và gom số liệu. Tách khỏi phần in để kiểm thử được.
    per_q: số chuỗi dùng được của từng câu hỏi trên TOÀN kho (mặc định đếm từ chains)."""
    per_q = per_q if per_q is not None else Counter(c["qid"] for c in chains)
    out = {"chains": len(chains), "with_row": 0, "status": Counter(), "source": Counter(), "kinds": Counter(),
           "markers": Counter(), "moved_chains": 0, "moves": 0, "dropped": 0, "tiny": 0, "spaces": 0, "before": [], "after": [],
           "one_step": 0, "long_few": 0, "broken": [], "examples": [], "no_raw": 0, "broken_in_exact_k": 0,
           "rules": {r: {"steps": [], "hit": 0, "glm": 0, "own": 0, "same_n": 0} for r in st.FALLBACK_RULES},
           "ctx": {k: {"share": [], "full": 0, "steps": 0} for k in KS}, "done": 0}
    for c in chains:
        row = rows.get(c["tid"])
        if row is None:
            continue
        text = c["text"]
        out["with_row"] += 1
        out["status"][row.get("status")] += 1
        f = st.final_steps(text, row)
        out["source"][f["source"]] += 1
        n = len(f["ends"])
        out["after"].append(n)
        out["one_step"] += n == 1
        out["long_few"] += len(text) >= LONG_CHARS and n <= 2
        lens = _lengths(f["ends"])
        for k in KS:
            shares = st.context_share(lens, k)
            out["ctx"][k]["share"].extend(shares)
            out["ctx"][k]["full"] += 1 + sum(s >= 1.0 - 1e-12 for s in shares)      # bước đầu luôn thấy đủ (chỉ có đề)
            out["ctx"][k]["steps"] += n
        if f["source"] != "glm":
            out["no_raw"] += not row.get("raw")
            out["broken_in_exact_k"] += f["source"].startswith("fallback") and per_q[c["qid"]] == k_select
            out["broken"].append({"tid": c["tid"], "status": row.get("status"), "finish": row.get("finish_reason"),
                                  "out_tokens": row.get("completion_tokens"), "chars": len(text), "has_raw": bool(row.get("raw")),
                                  "source": f["source"], "n_steps": n, "salvage": f["salvage"]})
            continue
        out["done"] += 1
        cuts = row["ends"][:-1]
        out["before"].append(len(row["ends"]))
        for cut in cuts:
            out["kinds"][st.cut_kind(text, cut)] += 1
        out["moves"] += len(f["moves"])
        out["moved_chains"] += bool(f["moves"])
        out["dropped"] += sum(m["dropped"] for m in f["moves"])
        out["tiny"] += f["tiny_merged"]
        out["spaces"] += f["spaces_moved"]
        for m in f["moves"]:
            out["markers"][m["marker"]] += 1
            if len(out["examples"]) < 200:
                out["examples"].append((c["tid"], text, m))
        glm = set(f["ends"][:-1])
        for rule, acc in out["rules"].items():
            own = st.rule_ends(text, rule)
            acc["steps"].append(len(own))
            acc["hit"] += len(glm & set(own[:-1]))
            acc["glm"] += len(glm)
            acc["own"] += len(own) - 1
            acc["same_n"] += len(own) == n
    return out


def _share(a: int, b: int) -> str:
    return f"{a / b:.1%}" if b else "—"


def print_report(a: Mapping, k_select: int, show: int, list_broken: bool) -> None:
    tag = "[kiểm bước]"
    print(f"{tag} {a['chains']} chuỗi dùng được; đã có dòng trong {OUT_NAME} cho {a['with_row']}; chưa cắt "
          f"{a['chains'] - a['with_row']}")
    if not a["with_row"]:
        return
    print(f"{tag} trạng thái: " + ", ".join(f"{k} {v}" for k, v in sorted(a["status"].items(), key=lambda x: str(x[0]))))

    total = sum(a["kinds"].values())
    print(f"\n{tag} 1. Ranh giới của {a['done']} chuỗi GLM cắt được ({', '.join(DONE)}): {total}")
    print(f"    ở đầu dòng                {a['kinds']['line']:>8}  ({_share(a['kinds']['line'], total)})")
    print(f"    ngay sau dấu đầu dòng     {a['kinds']['marker']:>8}  ({_share(a['kinds']['marker'], total)})  thuộc "
          f"{a['moved_chains']} chuỗi; LÙI về đầu dòng")
    print(f"    giữa dòng                 {a['kinds']['mid']:>8}  ({_share(a['kinds']['mid'], total)})  giữ nguyên")
    if a["markers"]:
        print("    dấu đầu dòng gặp nhiều nhất: " + ", ".join(f"{m!r} {n}" for m, n in a["markers"].most_common(12)))
    print(f"    ranh giới bỏ hẳn sau khi lùi (bước trước chỉ gồm dấu đầu dòng, hoặc dấu ở đầu văn bản): {a['dropped']}")
    print(f"    ranh giới dời về trước dấu cách để chữ đầu của bước thuộc đúng bước đó (không đổi nội dung bước): {a['spaces']}")
    print(f"    bước quá ngắn (dưới {st.MIN_STEP_CHARS} ký tự khác trắng) được gộp vào bước kế tiếp: {a['tiny']}")

    print(f"\n{tag} 2. Số bước mỗi chuỗi")
    if a["before"]:
        print(f"    GLM trả về (chuỗi cắt được):  {_spread(a['before'])}")
    print(f"    sau khi xử lý (mọi chuỗi):    {_spread(a['after'])}")
    print(f"    chuỗi chỉ có 1 bước: {a['one_step']}; chuỗi dài từ {LONG_CHARS} ký tự mà có không quá 2 bước: {a['long_few']}")

    broken = a["broken"]
    print(f"\n{tag} 3. Chuỗi cắt hỏng: {len(broken)} ({_share(len(broken), a['with_row'])} số chuỗi đã có dòng)")
    for src, n in sorted((s, n) for s, n in a["source"].items() if s != "glm"):
        label = "cứu được từ phản hồi thô" if src == "glm_salvaged" else f"cắt dự phòng, quy tắc {src.split('_', 1)[1]}"
        print(f"    {label:<34}{n:>6}")
    if broken:
        tried = [b for b in broken if b["salvage"] is not None]
        print(f"    không có phản hồi thô (không chạy với --keep-raw, hoặc lỗi mạng): {a['no_raw']}; thử cứu {len(tried)}, "
              f"nhận {sum(b['salvage']['accepted'] for b in tried)}")
        print(f"    chuỗi cắt dự phòng thuộc câu chỉ có đúng {k_select} ứng viên: {a['broken_in_exact_k']} "
              f"(loại các chuỗi này khỏi kho thì ngần ấy câu rơi khỏi tập chung)")
    if list_broken:
        for b in broken:
            s = b["salvage"]
            detail = (f"cứu: {s['n_groups']} nhóm, {s['unlocated']} không định vị, tỷ lệ độ dài {s['length_ratio']:.2f}, "
                      f"{'NHẬN' if s['accepted'] else 'KHÔNG nhận'}") if s else "không có gì để cứu"
            print(f"      {b['tid']} | {b['status']} | finish {b['finish']} | {b['out_tokens']} token ra | {b['chars']} ký tự | "
                  f"{detail} | {b['source']}, {b['n_steps']} bước")

    print(f"\n{tag} 4. Quy tắc cắt dự phòng so với GLM, đo trên {a['done']} chuỗi GLM cắt được")
    print(f"    {'quy tắc':<12}{'số bước trung vị':>18}{'tìm lại ranh giới GLM':>24}{'ranh giới trùng GLM':>22}{'cùng số bước':>15}")
    for rule, r in a["rules"].items():
        if r["steps"]:
            print(f"    {rule:<12}{statistics.median(r['steps']):>18.0f}{_share(r['hit'], r['glm']):>24}"
                  f"{_share(r['hit'], r['own']):>22}{_share(r['same_n'], a['done']):>15}")
    print("    Quy tắc tốt là quy tắc có số bước gần GLM và cả hai tỷ lệ cùng cao. Thứ tự đang dùng: "
          + " rồi ".join(st.FALLBACK_RULES) + " (quy tắc đầu tiên cho từ hai bước trở lên).")

    print(f"\n{tag} 5. Ngữ cảnh mô hình thấy khi chấm một bước, nếu chỉ nhìn k bước liền trước")
    for k, c in a["ctx"].items():
        mean = statistics.mean(c["share"]) if c["share"] else 1.0
        print(f"    k = {k}: thấy trung bình {mean:.0%} phần đứng trước (tính theo ký tự, từ bước thứ hai); "
              f"{_share(c['full'], c['steps'])} số bước thấy TOÀN BỘ phần trước, tức được chấm y như GRAPE")

    n = a["with_row"]
    fb = sum(v for s, v in a["source"].items() if s.startswith("fallback"))
    print(f"\n{tag} Nguồn bước sau khi xử lý: glm {a['source']['glm']}, cứu từ phản hồi thô {a['source']['glm_salvaged']}, "
          f"cắt dự phòng {fb} ({_share(fb, n)}). Phiên bản quy tắc: {st.STEPS_VERSION}.")

    for tid, text, m in a["examples"][:show]:
        lo, hi = max(0, m["to"] - 60), min(len(text), m["from"] + 60)
        print(f"\n    {tid}: dấu {m['marker']!r}" + (" (ranh giới bị bỏ)" if m["dropped"] else ""))
        print("      trước: " + repr(text[lo:m["from"]] + " ‖ " + text[m["from"]:hi]))
        print("      sau:   " + repr(text[lo:m["to"]] + " ‖ " + text[m["to"]:hi]))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--steps", default=OUT_NAME, help="file kết quả của segment_steps")
    ap.add_argument("--pilot", type=int, metavar="N", help="chỉ xét N chuỗi của lô thử (cùng cách rút với segment_steps)")
    ap.add_argument("--show", type=int, default=0, metavar="N", help="in N ranh giới bị lùi, trước và sau")
    ap.add_argument("--list-broken", action="store_true", help="in một dòng cho mỗi chuỗi cắt hỏng")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.override)
    wd = resolve_path(cfg, args.workdir)
    quality = {r["tid"]: r for r in read_jsonl(wd / "quality.jsonl")}
    chains = usable_chains(read_jsonl(wd / cfg["stage_a_files"]["candidates"]), quality)
    if not chains:
        raise SystemExit(f"Không có ứng viên dùng được trong {wd}.")
    per_q = Counter(c["qid"] for c in chains)
    if args.pilot:
        chains = pilot_sample(chains, args.pilot, int(cfg["selection"]["random_seed"]))
    rows = st.current_rows(read_jsonl(wd / args.steps))
    k = int(cfg["selection"]["k"])
    print_report(analyse(chains, rows, k, per_q), k, args.show, args.list_broken)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

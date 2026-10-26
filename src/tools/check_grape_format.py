"""check_grape_format: kiểm định xem đổi khuôn chấm GRAPE có làm đổi tập top-3 được chọn hay không.

    python -m src.tools.check_grape_format --workdir data/stage_a --student qwen1_5b                 # GPU: chấm rồi báo cáo
    python -m src.tools.check_grape_format --workdir data/stage_a --student qwen1_5b --report        # chỉ báo cáo, không cần GPU
    python -m src.tools.check_grape_format --n-questions 3 --out grape_format.smoke.jsonl           # chạy thử nhanh

Vấn đề. GRAPE gốc chấm log xác suất của câu trả lời trong khuôn "Question: ... Answer: ...". Đề tài chấm trong
khuôn hội thoại của Qwen kèm system prompt huấn luyện (b1_fit.build_inputs), để điểm đo đúng văn bản mà mô hình học
sẽ được huấn luyện. Nhưng mô hình Base chưa từng học các dấu hội thoại, nên phần khuôn có thể làm nhiễu điểm. Công cụ
này chấm lại cùng các chuỗi trong khuôn Question/Answer, rồi đếm bao nhiêu câu giữ đúng tập top-3.

Quy tắc chốt trước (trang Trạng thái hiện tại, 26/10/2026, mục 3 việc 2):
    từ 80% số câu cùng tập top-3: giữ cách chấm hiện tại, ghi kết quả vào bài báo;
    dưới 80%: chấm lại toàn kho, chọn lại, huấn luyện và đánh giá lại GRAPE và FQ-Select (1,5B, ba seed).

Mẫu. Mặc định 200 câu rút từ các câu có HƠN k chuỗi đúng (project.data_seed). Câu có đúng k chuỗi giữ cả k chuỗi dù
chấm cách nào, nên đưa vào sẽ làm tỷ lệ đẹp giả; compare_fit.topk_agreement cũng đếm theo cách này. Báo cáo in thêm
ước lượng cho cả kho (tính cả câu đúng k chuỗi) để đối chiếu, nhưng quyết định theo tỷ lệ trên các câu có hơn k chuỗi.

Cách làm cho phép đo sạch:
  - "hiện tại" là cột grape của fit.<student>_base.jsonl, tức đúng thứ b2_select đã dùng để chọn tập huấn luyện;
  - mỗi chuỗi được chấm HAI lần trong cùng một lượt chạy: khuôn Question/Answer (B) và khuôn hội thoại (A'), cùng
    b1_fit.token_stats và b1_fit.aggregate, nên khác biệt A' với B chỉ đến từ khuôn;
  - A' so với "hiện tại" là sàn nhiễu: hai lần chấm cùng khuôn mà vẫn lệch tập thì phần lệch đó không do khuôn
    (khác phiên bản thư viện, khác độ dài tối đa lúc b1 chạy, ...). Sàn nhiễu phải gần 100%;
  - tập top-3 chọn bằng b2_select.pick_topk trên chuỗi xếp theo thứ tự cố định (order_key), tức đúng cách phá hoà
    của b2; nếu có data/stage_b/<student>_base/train.grape.k3.jsonl thì đối chiếu thêm với tập đã huấn luyện.

Khuôn. --template mặc định "Question: {question}\\nAnswer: {answer}". Dấu cách đứng ngay trước {answer} được chuyển
sang đầu câu trả lời (token của Qwen mang dấu cách ở đầu), như cách Qwen vẫn tách " The". Không thêm BOS (Qwen không
dùng) và không chấm token kết thúc, giống b1_fit. Mô hình luôn là bản Base (cfg.student.base_model_id).

Ghi  <workdir>/grape_format.<student>_base.jsonl        một dòng mỗi chuỗi, ghi dần nên chạy bù được
     <workdir>/grape_format_check.<student>_base.json   số liệu của báo cáo
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from src.common.config import load_config, path_of, resolve_path
from src.common.io_utils import JsonlWriter, load_done_keys, now_iso, read_jsonl, write_json
from src.stage_b.b2_select import check_fit_rows, order_key, pick_topk, student_tag, usable_pool
from src.tools.compare_judges import spearman

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it

PLAIN_TEMPLATE = "Question: {question}\nAnswer: {answer}"
THRESHOLD = 0.80          # quy tắc chốt trước: từ 80% thì giữ
CHECK_VERSION = 1


# ============================================================ khuôn
def parse_template(template: str) -> str:
    """Phần đứng trước {answer}. Khuôn phải có đúng một {question}, đứng trước đúng một {answer}, và {answer} ở cuối
    (công cụ không chấm chữ đứng sau câu trả lời)."""
    if template.count("{question}") != 1 or template.count("{answer}") != 1:
        raise SystemExit(f"Khuôn phải có đúng một {{question}} và đúng một {{answer}}: {template!r}")
    head, tail = template.split("{answer}")
    if tail or "{question}" not in head:
        raise SystemExit(f"{{answer}} phải đứng cuối khuôn và sau {{question}}: {template!r}")
    return head


def plain_split(template: str, question: str) -> tuple[str, str]:
    """(văn bản phần đề, phần đệm đầu câu trả lời). Dấu cách cuối phần đề chuyển sang đầu câu trả lời để ranh giới
    token rơi đúng chỗ: 'Answer:' rồi ' The', không phải 'Answer: ' rồi 'The'."""
    head = parse_template(template).replace("{question}", question)
    body = head.rstrip(" ")
    return body, head[len(body):]


# ============================================================ kho và mẫu
def build_pool(cands: Sequence[Mapping], fit: Mapping[str, Mapping], quality: Mapping[str, Mapping] | None,
               min_correct: int, seed: int) -> tuple[dict, dict]:
    """Kho chuỗi đúng dùng được của từng câu, xếp theo thứ tự cố định của b2_select.

    Có quality.jsonl thì dùng đúng usable_pool của b2 (loại chuỗi thiếu Qual). Không có thì chỉ lọc theo số chuỗi đúng,
    và ghi rõ trong info: kho có thể hơn b2 vài chuỗi, nên bước đối chiếu với file train là chỗ bắt lệch.
    """
    if quality is not None:
        kept, raw = usable_pool(cands, quality, min_correct)
        info = {k: v for k, v in raw.items() if k not in ("excluded_tids", "dropped_qids")}
        info["pool_source"] = "usable_pool của b2 (có quality.jsonl)"
    else:
        grouped: dict[str, list] = defaultdict(list)
        for c in cands:
            grouped[c["qid"]].append(c)
        kept = {q: v for q, v in grouped.items() if len(v) >= min_correct}
        info = {"candidates": len(cands), "questions": len(kept), "usable_candidates": sum(len(v) for v in kept.values()),
                "pool_source": "candidates.jsonl, KHÔNG có quality.jsonl nên không loại được chuỗi thiếu Qual"}
    missing = [c["tid"] for v in kept.values() for c in v if c["tid"] not in fit]
    if missing:
        raise SystemExit(f"{len(missing)} chuỗi dùng được không có dòng trong file fit (ví dụ {missing[:3]}). "
                         f"File fit chưa đủ hoặc không khớp candidates.jsonl.")
    by_q = {}
    for qid, v in kept.items():
        chains = sorted(v, key=lambda c: order_key(seed, c["tid"]))
        by_q[qid] = [{"tid": c["tid"], "teacher": c["teacher"], "text": c["text"],
                      "current": float(fit[c["tid"]]["grape"]), "n_tokens": int(fit[c["tid"]]["n_tokens"])}
                     for c in chains]
    return by_q, info


def sample_questions(by_q: Mapping[str, Sequence], k: int, n: int, seed: int) -> list[str]:
    """n câu rút ngẫu nhiên (cùng seed thì cùng mẫu, trên mọi máy) từ các câu có HƠN k chuỗi."""
    qids = sorted(q for q, v in by_q.items() if len(v) > k)
    if n >= len(qids):
        return qids
    return sorted(random.Random(seed).sample(qids, n))


# ============================================================ so tập top-k
def top_set(scores: Sequence[float], k: int) -> frozenset[int]:
    """Chỉ số k chuỗi điểm cao nhất, hoà thì theo thứ tự cố định sẵn có: đúng pick_topk của b2 cho GRAPE (cực đại)."""
    return frozenset(pick_topk(np.asarray(scores, dtype=float), k, "max")["idx"])


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Khoảng tin cậy Wilson 95% cho một tỷ lệ."""
    if n == 0:
        return None
    p = successes / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d, (centre + half) / d


def compare_scores(rows: Mapping[str, Mapping], key_a: str, key_b: str, k: int) -> dict:
    """So hai cách chấm trên các câu. rows[qid] có tids, teacher, n_tokens và hai danh sách điểm key_a, key_b.

    same: số câu cùng tập top-k. mean_shared: số chuỗi chung trung bình trong k chuỗi. spearman: tương quan hạng trong
    từng câu rồi lấy trung bình. Tỷ lệ chuỗi mỗi mô hình dạy và độ dài trung bình của tập chọn bên A và bên B đi kèm,
    để thấy đổi khuôn kéo tập chọn về phía nào.
    """
    same = shared = 0
    spear, per_q = [], []
    sel = {"a": Counter(), "b": Counter()}
    toks = {"a": [], "b": []}
    for qid in sorted(rows):
        r = rows[qid]
        sa, sb = top_set(r[key_a], k), top_set(r[key_b], k)
        same += sa == sb
        shared += len(sa & sb)
        s = spearman(r[key_a], r[key_b])
        if s is not None:
            spear.append(s)
        per_q.append({"qid": qid, "same": sa == sb, "shared": len(sa & sb), "candidates": len(r["tids"])})
        for side, chosen in (("a", sa), ("b", sb)):
            for i in chosen:
                sel[side][r["teacher"][i]] += 1
                toks[side].append(r["n_tokens"][i])
    n = len(rows)
    out = {"questions": n, "same": same, "rate": same / n if n else None, "ci": wilson(same, n),
           "mean_shared": shared / n if n else None, "spearman": statistics.mean(spear) if spear else None,
           "per_question": per_q}
    for side in ("a", "b"):
        total = sum(sel[side].values())
        out[f"teacher_share_{side}"] = {t: c / total for t, c in sorted(sel[side].items())} if total else {}
        out[f"mean_tokens_{side}"] = statistics.mean(toks[side]) if toks[side] else None
    return out


def current_tids(chains: Sequence[Mapping], k: int) -> set[str]:
    return {chains[i]["tid"] for i in top_set([c["current"] for c in chains], k)}


def train_consistency(train_rows: Sequence[Mapping], by_q: Mapping[str, Sequence[Mapping]],
                      qids: Sequence[str], k: int) -> dict:
    """Tập top-k tính lại từ cột grape của file fit có trùng tập mà GRAPE đã được huấn luyện không. Lệch nghĩa là kho
    hoặc file fit dùng ở đây khác lúc chọn, và mọi so sánh bên dưới phải đọc kèm cảnh báo đó."""
    chosen: dict[str, set] = defaultdict(set)
    for r in train_rows:
        chosen[r["qid"]].add(r["tid"])
    n = same = 0
    for q in qids:
        if q in chosen:
            n += 1
            same += current_tids(by_q[q], k) == chosen[q]
    return {"questions": n, "same": same}


def decide(rate: float | None, threshold: float = THRESHOLD) -> str:
    if rate is None:
        raise SystemExit("Không có câu nào để quyết định.")
    return "keep" if rate >= threshold - 1e-12 else "rerun"


# ============================================================ file điểm
def settings_of(model: str, template: str, max_len: int, sample_n: int) -> dict:
    return {"model": model, "template": template, "max_len": max_len, "sample_n": sample_n,
            "check_version": CHECK_VERSION}


def check_resume(first_row: Mapping | None, want: Mapping, name: str) -> None:
    """Chặn chạy bù lên file điểm của một cài đặt khác: chấm lẫn hai khuôn hay hai cỡ mẫu sẽ cho số vô nghĩa."""
    if first_row is None:
        return
    diff = [f"{k}: file có {first_row.get(k)!r}, lần này {v!r}" for k, v in want.items() if first_row.get(k) != v]
    if diff:
        raise SystemExit(f"{name} được chấm với cài đặt khác ({'; '.join(diff)}). Chạy bù sẽ làm số vô nghĩa. "
                         f"Giữ đúng cài đặt cũ, hoặc ghi ra file khác bằng --out.")


def score_all(todo: Sequence[tuple[str, Mapping]], questions: Mapping[str, Mapping],
              scorer: Callable[[str, str], dict | None], writer, meta: Mapping, progress=tqdm) -> int:
    """Chấm từng chuỗi và ghi dần. scorer trả None khi không chấm được (hết bộ nhớ, chuỗi rỗng); trả về số chuỗi như vậy."""
    failed = 0
    for qid, c in progress(todo, desc="chấm hai khuôn"):
        v = scorer(questions[qid]["question"], c["text"])
        if v is None:
            failed += 1
            continue
        writer.append({"tid": c["tid"], "qid": qid, **meta, **v, "ts": now_iso()})
    return failed


def rows_for_report(by_q: Mapping[str, Sequence[Mapping]], qids: Sequence[str],
                    scores: Mapping[str, Mapping]) -> tuple[dict, list[str]]:
    """Các câu có ĐỦ chuỗi được chấm cả hai khuôn. Câu thiếu dù một chuỗi bị bỏ khỏi báo cáo (top-k không so được),
    và được trả về để báo cáo nêu ra."""
    rows, skipped = {}, []
    for q in qids:
        chains = by_q[q]
        got = [scores.get(c["tid"]) for c in chains]
        if any(g is None or "grape_plain" not in g or "grape_chat" not in g for g in got):
            skipped.append(q)
            continue
        rows[q] = {"tids": [c["tid"] for c in chains], "teacher": [c["teacher"] for c in chains],
                   "n_tokens": [c["n_tokens"] for c in chains], "current": [c["current"] for c in chains],
                   "chat": [g["grape_chat"] for g in got], "plain": [g["grape_plain"] for g in got]}
    return rows, skipped


def build_report(by_q: Mapping[str, Sequence[Mapping]], qids: Sequence[str], scores: Mapping[str, Mapping],
                 k: int, info: Mapping, train_rows: Sequence[Mapping] | None = None) -> dict:
    rows, skipped = rows_for_report(by_q, qids, scores)
    if not rows:
        raise SystemExit("Chưa có câu nào được chấm đủ cả hai khuôn. Chạy chấm trước, bỏ cờ --report.")
    primary = compare_scores(rows, "current", "plain", k)
    n_big = sum(len(v) > k for v in by_q.values())
    n_all = len(by_q)
    est = ((n_all - n_big) + primary["rate"] * n_big) / n_all if n_all else None
    rep = {"k": k, "threshold": THRESHOLD, "pool": dict(info), "questions_in_pool": n_all,
           "questions_with_more_than_k": n_big, "planned": len(qids), "scored": len(rows), "skipped": skipped,
           "current_vs_plain": primary,
           "chat_vs_plain": compare_scores(rows, "chat", "plain", k),
           "current_vs_chat": compare_scores(rows, "current", "chat", k),
           "pool_estimate_same_set": est, "decision": decide(primary["rate"])}
    if train_rows:
        rep["train_consistency"] = train_consistency(train_rows, by_q, list(rows), k)
    return rep


# ============================================================ in báo cáo
def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _line(name: str, c: Mapping, k: int) -> str:
    lo, hi = c["ci"] if c["ci"] else (None, None)
    sp = "—" if c["spearman"] is None else f"{c['spearman']:.3f}"
    return (f"  {name:<44}{c['same']:>4}/{c['questions']:<4}= {_pct(c['rate']):>6}  [{_pct(lo)}, {_pct(hi)}]"
            f"   chung {c['mean_shared']:.2f}/{k}   Spearman {sp}")


def print_report(rep: Mapping, student: str, template: str) -> None:
    k = rep["k"]
    p = rep["current_vs_plain"]
    print(f"\n[kiểm định dạng GRAPE] {student}_base, top-{k}, khuôn {template!r}")
    print(f"  kho: {rep['questions_in_pool']} câu, trong đó {rep['questions_with_more_than_k']} câu có hơn {k} chuỗi đúng"
          f" (nguồn kho: {rep['pool']['pool_source']})")
    print(f"  mẫu: {rep['planned']} câu có hơn {k} chuỗi, chấm đủ {rep['scored']}"
          + (f", BỎ {len(rep['skipped'])} câu thiếu điểm (ví dụ {rep['skipped'][:3]})" if rep["skipped"] else ""))
    print(f"\n  Cùng tập top-{k} / số câu, khoảng tin cậy Wilson 95%, số chuỗi chung trong {k}, tương quan hạng trong câu")
    print("  A = hiện tại (cột grape của file fit, khuôn hội thoại)   A' = chấm lại khuôn hội thoại trong lượt này")
    print("  B = khuôn Question/Answer\n")
    print(_line("A so B     (KẾT QUẢ CHÍNH)", p, k))
    print(_line("A' so B    (chỉ khác khuôn)", rep["chat_vs_plain"], k))
    print(_line("A so A'    (sàn nhiễu, phải gần 100%)", rep["current_vs_chat"], k))
    tc = rep.get("train_consistency")
    if tc:
        print(f"\n  Đối chiếu với tập đã huấn luyện: top-{k} tính lại từ file fit trùng train.grape.k{k}.jsonl ở "
              f"{tc['same']}/{tc['questions']} câu" + ("" if tc["same"] == tc["questions"] else
              "  <-- LỆCH: kho hoặc file fit ở đây khác lúc chọn, đọc kết quả với cảnh báo này"))
    else:
        print(f"\n  Không có train.grape.k{k}.jsonl để đối chiếu với tập đã huấn luyện.")
    floor = rep["current_vs_chat"]["rate"]
    if floor is not None and floor < 0.98:
        print(f"  CẢNH BÁO: sàn nhiễu chỉ {_pct(floor)}. Chấm lại cùng khuôn đã lệch tập ở {100 * (1 - floor):.1f}% số câu, "
              f"nên một phần chênh lệch A so B không do khuôn. Kiểm tra phiên bản torch, transformers, GPU, max_seq_len lúc b1 chạy.")
    print("\n  Tập chọn của hai bên (theo mô hình dạy, và độ dài tính bằng token của file fit):")
    teachers = sorted(set(p["teacher_share_a"]) | set(p["teacher_share_b"]))
    print(f"    {'':<10}" + "".join(f"{t:>20}" for t in teachers) + f"{'token TB':>12}")
    for side, name in (("a", "A"), ("b", "B")):
        sh = p[f"teacher_share_{side}"]
        print(f"    {name:<10}" + "".join(f"{_pct(sh.get(t, 0)):>20}" for t in teachers)
              + f"{p[f'mean_tokens_{side}']:>12.0f}")
    print(f"\n  Ước lượng cho cả kho (tính cả câu đúng {k} chuỗi, vốn luôn cùng tập): {_pct(rep['pool_estimate_same_set'])}."
          f" Chỉ để đối chiếu, quyết định theo A so B ở trên.")
    word = "GIỮ cách chấm hiện tại" if rep["decision"] == "keep" else "CHẤM LẠI"
    print(f"\n  Quy tắc chốt trước: từ {100 * rep['threshold']:.0f}% thì giữ, dưới thì chấm lại.")
    print(f"  Kết quả: {_pct(p['rate'])} cùng tập top-{k}  =>  {word}")
    if rep["scored"] < 100:
        print(f"  CHỈ LÀ CHẠY THỬ: mới {rep['scored']} câu, chưa đủ để quyết định. Quyết định theo lượt chạy 200 câu.")
    if rep["decision"] == "rerun":
        print("  Việc kế tiếp: chấm lại toàn kho bằng khuôn Question/Answer, chọn lại, huấn luyện và đánh giá lại GRAPE và "
              "FQ-Select 1,5B ba seed (khoảng 2 ngày GPU).")
    lo, hi = p["ci"] if p["ci"] else (None, None)
    if lo is not None and lo < rep["threshold"] <= hi:
        print(f"  Lưu ý: khoảng tin cậy [{_pct(lo)}, {_pct(hi)}] chứa {100 * rep['threshold']:.0f}%. Quy tắc vẫn theo điểm "
              f"ước lượng như đã chốt; ghi cả khoảng này vào bài báo.")


# ============================================================ phần chạy trên GPU
def make_gpu_scorer(model, tok, template: str, max_len: int, rank_clip: int, chunk: int) -> Callable:
    """Hàm chấm một chuỗi ở cả hai khuôn bằng ĐÚNG b1_fit.token_stats và b1_fit.aggregate (cùng cách tính
    grape = -tổng bất ngờ / số token). Trả None nếu chuỗi quá ngắn sau khi cắt hoặc hết bộ nhớ."""
    import torch
    from src.stage_b.b1_fit import aggregate, build_inputs, token_stats

    def grape(ids, n_prompt):
        s = token_stats(model, ids, n_prompt, chunk)
        a = aggregate(s["ranks"], s["surprisals"], s["sum_sq"], s["target_probs"], rank_clip)
        return a["grape"], a["n_tokens"]

    def score(question: str, text: str) -> dict | None:
        try:
            ids_c, n_c, *_ = build_inputs(tok, question, text, max_len)
            ids_p, n_p = plain_inputs(tok, question, text, max_len, template)
            if ids_c is None or ids_p is None:
                return None
            g_c, t_c = grape(ids_c, n_c)
            g_p, t_p = grape(ids_p, n_p)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        return {"grape_plain": g_p, "n_tokens_plain": t_p, "grape_chat": g_c, "n_tokens_chat": t_c}

    return score


def plain_token_ids(tok, question: str, trajectory: str, max_len: int, template: str) -> tuple[list, list] | None:
    """(token phần đề, token phần chuỗi đã cắt) theo khuôn Question/Answer. Phần đề và phần chuỗi được mã hoá riêng
    rồi nối, đúng như b1_fit.build_inputs làm với khuôn hội thoại. Cắt cùng độ dài tối đa (đề + chuỗi); None nếu còn
    dưới 2 token của chuỗi để chấm."""
    prompt, lead = plain_split(template, question)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(lead + trajectory, add_special_tokens=False)["input_ids"]
    r_ids = r_ids[:max(0, min(len(r_ids), max_len - len(p_ids)))]
    return None if len(r_ids) < 2 else (p_ids, r_ids)


def plain_inputs(tok, question: str, trajectory: str, max_len: int, template: str, device: str = "cuda"):
    """(ids trên thiết bị, số token phần đề), hoặc (None, 0). Cùng dạng trả về với b1_fit.build_inputs."""
    import torch

    got = plain_token_ids(tok, question, trajectory, max_len, template)
    if got is None:
        return None, 0
    p_ids, r_ids = got
    return torch.tensor([p_ids + r_ids], device=device), len(p_ids)


# ============================================================ dòng lệnh
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--student", default="qwen1_5b", help="tên file trong configs/student/")
    ap.add_argument("--n-questions", type=int, default=200, help="số câu rút từ các câu có hơn k chuỗi đúng")
    ap.add_argument("--template", default=PLAIN_TEMPLATE, help="khuôn chấm, có {question} rồi {answer} ở cuối")
    ap.add_argument("--fit", help="file fit, mặc định fit.<student>_base.jsonl")
    ap.add_argument("--out", help="tên file điểm trong workdir, mặc định grape_format.<student>_base.jsonl")
    ap.add_argument("--report", action="store_true", help="chỉ đọc file điểm và báo cáo, không cần GPU")
    ap.add_argument("--load-4bit", action="store_true", help="nạp mô hình ở 4-bit (chỉ khi file fit của nó cũng chấm ở 4-bit)")
    ap.add_argument("--chunk", type=int, default=512, help="số vị trí xếp hạng mỗi lần, như b1_fit")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(student=args.student, overrides=args.override)
    parse_template(args.template)
    k = int(cfg["selection"]["k"])
    model_id = cfg["student"]["base_model_id"]
    max_len = int(cfg["training"]["max_seq_len"])
    tag = student_tag(cfg)
    wd: Path = resolve_path(cfg, args.workdir)
    files = cfg["stage_a_files"]

    fit_name = args.fit or f"fit.{tag}.jsonl"
    fit_rows = read_jsonl(wd / fit_name)
    cands = read_jsonl(wd / files["candidates"])
    if not fit_rows or not cands:
        raise SystemExit(f"Thiếu dữ liệu trong {wd}: {fit_name} {len(fit_rows)} dòng, candidates {len(cands)} dòng.")
    check_fit_rows(fit_rows, fit_name, model_id)
    q_path = wd / "quality.jsonl"
    quality = {r["tid"]: r for r in read_jsonl(q_path)} if q_path.exists() else None
    by_q, info = build_pool(cands, {r["tid"]: r for r in fit_rows}, quality,
                            int(cfg["data"]["min_correct_per_question"]), int(cfg["selection"]["random_seed"]))
    qids = sample_questions(by_q, k, args.n_questions, int(cfg["project"]["data_seed"]))
    if not qids:
        raise SystemExit(f"Không có câu nào có hơn {k} chuỗi đúng trong {wd}.")

    score_path = wd / (args.out or f"grape_format.{tag}.jsonl")
    want = settings_of(model_id, args.template, max_len, len(qids))
    first = next(iter(read_jsonl(score_path)), None)
    check_resume(first, want, score_path.name)

    if not args.report:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("Không thấy GPU. Lệnh này chạy trên máy Windows có card NVIDIA; trên Mac dùng --report.")
        from src.stage_b.b1_fit import load_student
        questions = {q["qid"]: q for q in read_jsonl(wd / files["questions"])}
        done = load_done_keys(score_path, lambda r: r["tid"])
        todo = [(q, c) for q in qids for c in by_q[q] if c["tid"] not in done]
        n_chains = sum(len(by_q[q]) for q in qids)
        print(f"[format] {model_id}, {'4-bit' if args.load_4bit else '16-bit'}, khuôn {args.template!r}, độ dài tối đa {max_len}")
        print(f"[format] {len(qids)} câu, {n_chains} chuỗi, đã có {n_chains - len(todo)}, cần chấm {len(todo)} (mỗi chuỗi hai khuôn)")
        model, tok = load_student(model_id, args.load_4bit)
        scorer = make_gpu_scorer(model, tok, args.template, max_len, int(cfg["selection"]["rsr_rank_cap"]), args.chunk)
        with JsonlWriter(score_path) as w:
            failed = score_all(todo, questions, scorer, w, {"model": model_id, "template": args.template,
                                                            "max_len": max_len, "sample_n": len(qids),
                                                            "check_version": CHECK_VERSION})
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"[format] xong, đỉnh bộ nhớ {peak:.2f} GB" + (f", {failed} chuỗi không chấm được" if failed else ""))

    scores = {r["tid"]: r for r in read_jsonl(score_path)}
    train_path = path_of(cfg, "data_stage_b") / tag / f"train.grape.k{k}.jsonl"
    train_rows = read_jsonl(train_path) if train_path.exists() else None
    rep = build_report(by_q, qids, scores, k, info, train_rows)
    rep.update({"model": model_id, "template": args.template, "score_file": score_path.name, "fit_file": fit_name})
    print_report(rep, args.student, args.template)
    out = wd / f"grape_format_check.{tag}.json"
    write_json(out, rep)
    print(f"\n[format] số liệu ghi ở {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

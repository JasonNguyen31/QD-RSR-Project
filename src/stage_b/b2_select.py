"""b2_select: chọn k chuỗi mỗi câu hỏi cho từng phương án và ghi file huấn luyện.

    python -m src.stage_b.b2_select --student qwen1_5b --method rsr
    python -m src.stage_b.b2_select --student qwen1_5b --ablation fit_quality
    python -m src.stage_b.b2_select --student qwen1_5b --all                      # mọi phương án và biến thể
    python -m src.stage_b.b2_select --student qwen1_5b --method qd_rsr --lambda-grid
    python -m src.stage_b.b2_select --student qwen1_5b --method rsr --override selection.k=1
    python -m src.stage_b.b2_select --student qwen7b --all                        # ba phương án của 7B
    python -m src.stage_b.b2_select --student qwen1_5b --report                   # bảng I.9 và độ trùng
    python -m src.stage_b.b2_select --student qwen1_5b --report --versus qwen7b   # 7B chọn khác 1,5B bao nhiêu

Đọc  <workdir>/candidates.jsonl, questions.jsonl, quality.jsonl, embeddings.npz, fit.<student>_base.jsonl
     Kho "all" đọc thêm trajectories*.jsonl (chỉ có trên máy giữ dữ liệu gốc). Kho "all" đi với hàm mục tiêu
     (biến thể no_prefilter) còn cần ba file của kho mở rộng do src/tools/backfill_wrong tạo:
     quality.all.jsonl, embeddings.wrong.npz, fit.<student>_base.wrong.jsonl
Ghi  data/stage_b/<student>_base/train.<tên>.jsonl    một dòng mỗi chuỗi được chọn, kèm đề bài, văn bản, trọng số
     data/stage_b/<student>_base/select.<tên>.json    tham số đã dùng, số liệu của tập chọn, md5 của file train
     data/stage_b/<student>_base/pool.json            số liệu của kho ứng viên (dòng đầu của bảng I.9)

Tên file do selection_tag quyết định: <phương án>.k<k>, thêm .lam<λ> khi tập chọn phụ thuộc λ. b3_train phải
gọi đúng hàm đó để không đọc nhầm file của một λ cũ.

Ba quy tắc chọn (khoá select.rule trong configs/method và configs/ablation):
    random     k chuỗi đầu theo thứ tự cố định (xem dưới). Correct-Only, No-Filter.
    topk       k chuỗi đứng đầu theo một tín hiệu. CHỈ RSR LÀ CỰC TIỂU, chiều lấy từ signal_direction.
    objective  duyệt hết mọi tập con cỡ k, lấy tập tối đa hoá F(S) = Σ Fit^a·Qual^b + λ·Div(S) (objective.py).

Các quy ước đã chốt mà mã này thực thi:
  - Chuỗi không có điểm giám khảo (qual = null trong quality.jsonl) bị loại khỏi kho của MỌI phương án, kể cả
    No-Filter. Câu hỏi còn dưới data.min_correct_per_question chuỗi đúng thì bị loại. Mọi phương án vì thế
    dùng chung một tập câu hỏi và ra đúng k chuỗi mỗi câu; mã dừng nếu không đúng như vậy.
  - Mọi chuỗi đúng dùng được phải có dòng trong file fit và có biểu diễn, kể cả khi phương án không cần tín
    hiệu. Nhờ vậy kho của Correct-Only không thể lệch giữa hai mô hình học mà không ai biết.
  - Fit = 1 − minmax(RSR) và ĝ của LARK tính trên kho đã loại các chuỗi nói trên (signals.add_lark dặn rõ:
    đổi nhóm ứng viên thì phải tính lại ĝ).
  - Qual lấy thẳng cột qual (mỗi nửa đã min-max trong câu hỏi rồi trộn), giống lambda_scan.
  - Kho mở rộng có Qual RIÊNG (quality.all.jsonl): thêm chuỗi sai làm đổi min-max trong từng câu, nên Qual
    của cả chuỗi đúng cũng khác quality.jsonl. Fit và ĝ cũng tính lại trên nhóm mở rộng. Chuỗi sai mà giám
    khảo không chấm được thì bị loại khỏi kho này, cùng quy tắc với chuỗi đúng thiếu Qual. No-Filter không
    dùng tín hiệu nào nên không đọc các file này và không đổi khi chấm bù.
  - Phương án có khoá select.blocked trong file cấu hình chưa được phép dùng để huấn luyện; b2 từ chối trừ
    khi thêm --allow-blocked, và khi đó tên file mang đuôi .tam để không lẫn với bản chính thức.
  - LARK ghi trọng số mềm (signals.lark_weights) nhân k, để trung bình bằng 1 như trọng số của các phương án
    khác. b3_train chuẩn hoá loss theo tổng trọng số trong lô nên phép nhân này không đổi kết quả.
  - Thứ tự cố định: các chuỗi của một câu được xếp theo sha256("<random_seed>|<tid>"). Thứ tự này vừa là cách
    rút ngẫu nhiên (Correct-Only, No-Filter) vừa là cách phá hoà của mọi phương án. Xếp theo tid thì mọi lần
    hoà đều nghiêng về deepseek (đứng đầu bảng chữ cái); xếp theo mã băm thì không nghiêng về ai, không phụ
    thuộc phiên bản Python, máy, mô hình học hay seed huấn luyện. Dùng hashlib chứ KHÔNG dùng hash() có sẵn
    của Python: hash() của chuỗi đổi theo PYTHONHASHSEED ở mỗi lần chạy. Hệ quả có chủ đích: No-Filter và
    Correct-Only chỉ khác nhau ở những câu mà một chuỗi sai chen được vào k vị trí đầu.

Tham lam chỉ là phân tích phụ (in tỷ lệ trùng nghiệm tối ưu), không ảnh hưởng tập được chọn.
Chỉ đọc file, không cần GPU, không gọi mạng.
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.common.config import load_config, path_of, resolve_path, signal_direction
from src.common.io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from src.stage_b.objective import (base_of_subsets, choose, distance_matrix, div_of_subsets, div_scale,
                                   fit_from_rsr, subsets)
from src.stage_b.signals import SIGNALS_VERSION, lark_ghat, lark_weights

SIGNAL_COLUMN = {"token_length": "n_tokens"}      # tên tín hiệu trong cấu hình -> cột của file fit
FIT_COLUMNS = ("rsr", "grape", "local_nat", "n_tokens")
PINNED_IN_ALL = ("selection.a", "selection.b", "selection.lambda_div")
TIE_EPS = 1e-12


# ============================================================ tên file và thứ tự cố định
def order_key(seed: int, tid: str) -> str:
    return hashlib.sha256(f"{seed}|{tid}".encode("utf-8")).hexdigest()


def uses_lambda(cfg: Mapping) -> bool:
    """Tập chọn có phụ thuộc λ không. k = 1 thì Div = 0; a = b = 0 thì mọi λ > 0 cho cùng một tập."""
    sel = cfg["selection"]
    return (cfg["select"]["rule"] == "objective" and int(sel["k"]) > 1 and float(sel["lambda_div"]) > 0
            and (float(sel["a"]) != 0 or float(sel["b"]) != 0))


def selection_tag(cfg: Mapping) -> str:
    """Tên của một tập chọn: 'rsr.k3', 'qd_rsr.k3.lam0.4', 'qd_rsr.k1'. b3_train dùng lại hàm này.
    Phương án đang bị khoá (select.blocked) mang đuôi .tam."""
    sel = cfg["selection"]
    tag = f"{cfg['select']['name']}.k{int(sel['k'])}"
    tag += f".lam{float(sel['lambda_div']):g}" if uses_lambda(cfg) else ""
    return tag + (".tam" if cfg["select"].get("blocked") else "")


def student_tag(cfg: Mapping) -> str:
    return f"{cfg['student']['key']}_base"


def extended_files(cfg: Mapping) -> dict:
    """Tên các file của kho mở rộng (chuỗi đúng cộng chuỗi sai). backfill_wrong ghi, b2_select đọc."""
    return {"wrong": "candidates.wrong.jsonl", "judge": "judge.wrong.jsonl", "quality": "quality.all.jsonl",
            "embeddings": "embeddings.wrong.npz", "fit": f"fit.{student_tag(cfg)}.wrong.jsonl"}


def check_fit_rows(rows: Sequence[Mapping], name: str, want_model: str) -> None:
    if int(rows[0].get("signals_version", 1)) < SIGNALS_VERSION:
        raise SystemExit(f"{name} tạo bằng định nghĩa tín hiệu cũ (trước 29/09). Dùng file fit mới.")
    models = {r.get("model") for r in rows}
    if models != {want_model}:
        raise SystemExit(f"{name} chấm bằng {sorted(map(str, models))}, cấu hình đòi {want_model}. "
                         f"Sai file fit hoặc sai --student.")


# ============================================================ dựng kho ứng viên
def usable_pool(cands: Sequence[Mapping], quality: Mapping[str, Mapping], min_correct: int) -> tuple[dict, dict]:
    """Chuỗi đúng dùng được của từng câu hỏi được giữ, và thống kê việc loại.

    Mọi ứng viên phải có dòng trong quality.jsonl; thiếu là lỗi dữ liệu, không phải chuỗi thiếu Qual.
    """
    absent = [c["tid"] for c in cands if c["tid"] not in quality]
    if absent:
        raise SystemExit(f"{len(absent)} ứng viên không có dòng trong quality.jsonl (ví dụ {absent[:3]}). "
                         f"quality.jsonl không khớp candidates.jsonl: chạy lại a4_score_quality --rule-only.")
    excluded = {c["tid"] for c in cands if quality[c["tid"]].get("qual") is None}
    by_q: dict[str, list] = defaultdict(list)
    for c in cands:
        if c["tid"] not in excluded:
            by_q[c["qid"]].append(c)
    all_q = {c["qid"] for c in cands}
    kept = {q: v for q, v in by_q.items() if len(v) >= min_correct}
    return kept, {"candidates": len(cands), "excluded_no_qual": len(excluded), "excluded_tids": excluded,
                  "questions_before": len(all_q), "questions": len(kept),
                  "dropped_qids": sorted(all_q - set(kept)),
                  "usable_candidates": sum(len(v) for v in kept.values())}


def all_chains_pool(trajectories: Sequence[Mapping], kept_qids: set, excluded: set) -> tuple[dict, int]:
    """Kho không lọc đáp án: mọi chuỗi (đúng, sai, bị cắt) của các câu được giữ, trừ chuỗi rỗng và chuỗi đã
    bị loại vì thiếu Qual. Trả về (kho, số chuỗi rỗng bị bỏ)."""
    by_q: dict[str, list] = defaultdict(list)
    empty = 0
    for t in trajectories:
        if t["qid"] not in kept_qids or t["tid"] in excluded:
            continue
        if not (t.get("text") or "").strip():
            empty += 1
            continue
        by_q[t["qid"]].append(t)
    return dict(by_q), empty


def assemble(pool: Mapping[str, Sequence[Mapping]], correct_tids: set, quality: Mapping[str, Mapping],
             fit: Mapping[str, Mapping], emb_row: Mapping[str, int], vecs: np.ndarray, seed: int,
             need_signals: bool) -> dict:
    """Ghép mọi nguồn theo tid cho từng câu hỏi, các chuỗi xếp theo thứ tự cố định.

    need_signals: kho này có buộc phải đủ tín hiệu không. Buộc mà thiếu ở bất kỳ chuỗi nào thì dừng, không âm
    thầm bỏ chuỗi (bỏ chuỗi sai chưa chấm sẽ biến biến thể không lọc thành chính QD-RSR).
    """
    miss = Counter()
    for chains in pool.values():
        for c in chains:
            t = c["tid"]
            miss["file fit"] += t not in fit
            miss["Qual"] += quality.get(t, {}).get("qual") is None
            miss["biểu diễn"] += t not in emb_row
    complete = not sum(miss.values())
    if need_signals and not complete:
        detail = ", ".join(f"{k} thiếu {v} chuỗi" for k, v in miss.items() if v)
        raise SystemExit(f"Kho này cần đủ tín hiệu cho mọi chuỗi, nhưng {detail}. Nếu là kho 'all' thì chuỗi sai "
                         f"chưa được chấm: phải chạy a4, a5 và b1 cho chúng trước.")
    out = {}
    for qid in sorted(pool):
        chains = sorted(pool[qid], key=lambda c: order_key(seed, c["tid"]))
        tids = [c["tid"] for c in chains]
        frows = [fit.get(t) for t in tids]
        q = {"qid": qid, "tids": tids, "teacher": [c["teacher"] for c in chains],
             "text": [c["text"] for c in chains], "correct": [t in correct_tids for t in tids],
             "llm": [quality.get(t, {}).get("llm_score") for t in tids],
             # độ dài: token của mô hình học (file fit); chuỗi chưa chấm thì lấy số token mà API báo
             "n_tokens": [(f["n_tokens"] if f else c.get("completion_tokens")) for f, c in zip(frows, chains)],
             "api_length": [f is None for f in frows], "signals": {}}
        if complete:
            for col in FIT_COLUMNS:
                if all(col in f for f in frows):
                    q["signals"][col] = np.array([float(f[col]) for f in frows])
            _rho, ghat = lark_ghat([float(f["mean_surprisal"]) for f in frows], [float(f["brier"]) for f in frows])
            q["signals"]["lark"] = np.array(ghat)
            q["fit"] = fit_from_rsr(q["signals"]["rsr"])
            q["qual"] = np.array([float(quality[t]["qual"]) for t in tids])
            q["vecs"] = vecs[[emb_row[t] for t in tids]]
        out[qid] = q
    return out


# ============================================================ ba quy tắc chọn
def pick_topk(scores: np.ndarray, k: int, direction: str) -> dict:
    """k chuỗi đứng đầu. Hoà thì giữ thứ tự cố định sẵn có (sắp xếp ổn định)."""
    if np.isnan(scores).any():
        raise SystemExit("Có điểm NaN trong tín hiệu dùng để xếp hạng; kiểm tra file fit.")
    s = scores if direction == "max" else -scores
    order = sorted(range(len(s)), key=lambda i: -s[i])
    return {"idx": sorted(order[:k]), "tie": bool(len(s) > k and s[order[k - 1]] == s[order[k]])}


def greedy_set(per: np.ndarray, dist: np.ndarray, k: int, lam: float) -> list[int]:
    """Tham lam theo độ lợi biên của F, bắt đầu từ chuỗi có Fit^a·Qual^b lớn nhất. Chỉ để phân tích phụ."""
    chosen = [int(np.argmax(per))]
    while len(chosen) < k:
        best, best_v = -1, -np.inf
        for j in range(len(per)):
            if j in chosen:
                continue
            s = chosen + [j]
            v = float(per[s].sum() + lam * div_of_subsets(dist, np.array([s]))[0])
            if v > best_v + TIE_EPS:
                best, best_v = j, v
        chosen.append(best)
    return sorted(chosen)


def pick_objective(q: Mapping, k: int, a: float, b: float, lam: float, how: str) -> dict:
    sets = subsets(len(q["tids"]), k)
    base = base_of_subsets(q["fit"], q["qual"], sets, a, b)
    dist = distance_matrix(q["vecs"])
    div = div_scale(div_of_subsets(dist, sets), how)
    val = base + lam * div
    best = choose(base, div, lam)
    out = {"idx": [int(i) for i in sets[best]], "tie": int((val >= val[best] - TIE_EPS).sum()) > 1}
    if how == "raw" and lam > 0 and k > 1 and len(q["tids"]) > k:
        per = np.power(q["fit"], a) * np.power(q["qual"], b)
        out["greedy_same"] = greedy_set(per, dist, k, lam) == out["idx"]
    return out


def select_all(questions: Mapping[str, Mapping], cfg: Mapping) -> dict:
    """Chạy quy tắc của cfg trên mọi câu hỏi. Trả về qid -> {idx, weights, tie, ...}."""
    spec, sel = cfg["select"], cfg["selection"]
    k, rule = int(sel["k"]), spec["rule"]
    picks = {}
    for qid, q in questions.items():
        if len(q["tids"]) < k:
            raise SystemExit(f"Câu {qid} chỉ có {len(q['tids'])} chuỗi trong kho, không đủ k = {k}.")
        if rule == "random":
            p = {"idx": list(range(k)), "tie": False}
        elif rule == "topk":
            name = spec["signal"]
            col = SIGNAL_COLUMN.get(name, name)
            if col not in q["signals"]:
                raise SystemExit(f"File fit không có tín hiệu '{col}' (mô hình 7 tỷ bỏ LocalNat theo quyết định "
                                 f"30/09). Phương án {spec['name']} không chạy được với file này.")
            p = pick_topk(q["signals"][col], k, signal_direction(cfg, name))
        elif rule == "objective":
            p = pick_objective(q, k, float(sel["a"]), float(sel["b"]), float(sel["lambda_div"]), sel["div_scale"])
        else:
            raise SystemExit(f"select.rule không hợp lệ: {rule!r} (random, topk, objective)")
        if spec.get("weights", "uniform") == "lark_soft":
            w = lark_weights([float(x) for x in q["signals"]["lark"]], k)
            if not {i for i, x in enumerate(w) if x > 0} <= set(p["idx"]):
                raise SystemExit(f"Câu {qid}: trọng số LARK rơi ngoài tập top-{k} theo ĝ.")
            p["weights"] = [w[i] * k for i in p["idx"]]
        else:
            p["weights"] = [1.0] * k
        picks[qid] = p
    return picks


# ============================================================ số liệu của một tập chọn
def _shares(teachers: Sequence[str]) -> dict:
    n = len(teachers)
    return {t: c / n for t, c in sorted(Counter(teachers).items())}


def pool_stats(questions: Mapping[str, Mapping]) -> dict:
    teachers = [t for q in questions.values() for t in q["teacher"]]
    toks = [n for q in questions.values() for n in q["n_tokens"] if n is not None]
    return {"questions": len(questions), "chains": len(teachers), "teacher_share": _shares(teachers),
            "mean_tokens": statistics.mean(toks) if toks else None,
            "judge_zero": sum(1 for q in questions.values() for s in q["llm"] if s is not None and s <= 0.0),
            "n_candidates": {qid: len(q["tids"]) for qid, q in questions.items()}}


def summarize(questions: Mapping[str, Mapping], picks: Mapping[str, Mapping], k: int) -> dict:
    teachers, toks, weights, divs, fits, quals, greedy = [], [], [], [], [], [], []
    judge_zero = wrong = api_len = 0
    for qid, p in picks.items():
        q = questions[qid]
        for i in p["idx"]:
            teachers.append(q["teacher"][i])
            if q["n_tokens"][i] is not None:
                toks.append(q["n_tokens"][i])
            judge_zero += q["llm"][i] is not None and q["llm"][i] <= 0.0
            wrong += not q["correct"][i]
            api_len += q["api_length"][i]
        weights.extend(p["weights"])
        if "vecs" in q and k > 1:
            divs.append(float(div_of_subsets(distance_matrix(q["vecs"]), np.array([p["idx"]]))[0]))
        if "fit" in q:
            fits.extend(float(x) for x in q["fit"][p["idx"]])
            quals.extend(float(x) for x in q["qual"][p["idx"]])
        if "greedy_same" in p:
            greedy.append(p["greedy_same"])
    w = np.array(weights)
    return {"questions": len(picks), "samples": len(teachers), "k": k,
            "more_than_k": sum(len(questions[q]["tids"]) > k for q in picks),
            "teacher_share": _shares(teachers),
            "mean_tokens": statistics.mean(toks) if toks else None, "lengths_from_api": int(api_len),
            "judge_zero_kept": int(judge_zero), "wrong_kept": int(wrong),
            "ties": sum(bool(p["tie"]) for p in picks.values()),
            "mean_div": statistics.mean(divs) if divs else None,
            "mean_fit": statistics.mean(fits) if fits else None,
            "mean_qual": statistics.mean(quals) if quals else None,
            "greedy_same": (sum(greedy) / len(greedy)) if greedy else None, "greedy_questions": len(greedy),
            "weight_below_0.1": int((w < 0.1).sum()),
            "effective_samples": float(w.sum() ** 2 / (w * w).sum())}


def train_rows(questions: Mapping[str, Mapping], picks: Mapping[str, Mapping], qtext: Mapping[str, Mapping]) -> list:
    rows = []
    for qid in sorted(picks):
        q, p = questions[qid], picks[qid]
        for i, w in sorted(zip(p["idx"], p["weights"]), key=lambda x: q["tids"][x[0]]):
            rows.append({"qid": qid, "tid": q["tids"][i], "teacher": q["teacher"][i], "correct": q["correct"][i],
                         "weight": round(float(w), 6), "n_tokens": q["n_tokens"][i],
                         "question": qtext[qid]["question"], "text": q["text"][i]})
    return rows


def check_shape(picks: Mapping[str, Mapping], kept_qids: set, k: int, name: str) -> None:
    """Lời hứa của bảng phương án: mọi phương án cùng tập câu hỏi, đúng k chuỗi khác nhau mỗi câu."""
    if set(picks) != kept_qids:
        raise SystemExit(f"{name}: tập câu hỏi lệch khỏi tập chung ({len(picks)} so với {len(kept_qids)}).")
    bad = [q for q, p in picks.items() if len(set(p["idx"])) != k]
    if bad:
        raise SystemExit(f"{name}: {len(bad)} câu không có đúng {k} chuỗi khác nhau (ví dụ {bad[:3]}).")


# ============================================================ đọc dữ liệu
class Inputs:
    """Đọc một lần mọi file của giai đoạn A và file fit của một mô hình học, dùng lại cho mọi phương án."""

    def __init__(self, cfg: Mapping, workdir: Path, fit_name: str):
        files = cfg["stage_a_files"]
        self.cfg, self.workdir, self.fit_name = cfg, workdir, fit_name
        self.cands = read_jsonl(workdir / files["candidates"])
        self.qtext = {q["qid"]: q for q in read_jsonl(workdir / files["questions"])}
        self.quality = {r["tid"]: r for r in read_jsonl(workdir / "quality.jsonl")}
        fit_rows = read_jsonl(workdir / fit_name)
        if not self.cands or not self.quality or not fit_rows:
            raise SystemExit(f"Thiếu dữ liệu trong {workdir}: candidates {len(self.cands)}, quality "
                             f"{len(self.quality)}, {fit_name} {len(fit_rows)}.")
        check_fit_rows(fit_rows, fit_name, cfg["student"]["base_model_id"])
        self.fit = {r["tid"]: r for r in fit_rows}
        npz = np.load(workdir / "embeddings.npz")
        self.emb_row = {str(t): i for i, t in enumerate(npz["tids"])}
        self.vecs = npz["vectors"]
        self.seed = int(cfg["selection"]["random_seed"])
        self.correct_tids = {c["tid"] for c in self.cands}
        self.pool, self.info = usable_pool(self.cands, self.quality, int(cfg["data"]["min_correct_per_question"]))
        self._questions: dict[tuple, dict] = {}

    def questions(self, pool: str, rule: str) -> dict:
        need = pool == "correct" or rule != "random"
        if (pool, need) in self._questions:
            return self._questions[(pool, need)]
        if pool == "correct":
            chains = self.pool
        elif pool == "all":
            from src.stage_a.a2_generate import all_trajectory_files
            paths = all_trajectory_files(self.workdir, self.cfg["stage_a_files"])
            if not paths:
                raise SystemExit("Kho 'all' cần trajectories*.jsonl, máy này không có. Chạy trên máy giữ dữ liệu "
                                 "gốc rồi chép file train sang (so md5 trong select.<tên>.json).")
            chains, empty = all_chains_pool([t for p in paths for t in read_jsonl(p)], set(self.pool),
                                            self.info["excluded_tids"])
            if empty:
                print(f"[b2] kho 'all': bỏ {empty} chuỗi rỗng")
        else:
            raise SystemExit(f"select.pool không hợp lệ: {pool!r} (correct, all)")
        quality, fit, emb_row, vecs = self.quality, self.fit, self.emb_row, self.vecs
        if pool == "all" and need:
            quality, fit, emb_row, vecs = self.extended()
            unjudged = {c["tid"] for v in chains.values() for c in v
                        if c["tid"] in quality and quality[c["tid"]].get("qual") is None}
            if unjudged:
                print(f"[b2] kho mở rộng: loại {len(unjudged)} chuỗi sai giám khảo không chấm được")
                chains = {q: [c for c in v if c["tid"] not in unjudged] for q, v in chains.items()}
        out = assemble(chains, self.correct_tids, quality, fit, emb_row, vecs, self.seed, need)
        self._questions[(pool, need)] = out
        return out

    def extended(self) -> tuple[dict, dict, dict, np.ndarray]:
        """Tín hiệu của kho mở rộng: Qual riêng của kho, fit và biểu diễn gộp phần chuỗi đúng với phần chuỗi sai."""
        names = extended_files(self.cfg)
        absent = [names[k] for k in ("quality", "embeddings", "fit") if not (self.workdir / names[k]).exists()]
        if absent:
            raise SystemExit(f"Kho 'all' với hàm mục tiêu cần tín hiệu của chuỗi sai chưa được chấm: thiếu "
                             f"{', '.join(absent)}. Chạy src.tools.backfill_wrong (prepare, judge, embed, fit, "
                             f"combine) trước.")
        quality = {r["tid"]: r for r in read_jsonl(self.workdir / names["quality"])}
        stale = [t for t, r in self.quality.items()
                 if t in quality and r.get("llm_score") != quality[t].get("llm_score")]
        if stale:
            raise SystemExit(f"{names['quality']} lệch điểm giám khảo với quality.jsonl ở {len(stale)} chuỗi (ví dụ "
                             f"{stale[:3]}): file cũ, chạy lại backfill_wrong --step combine.")
        wrong_fit = read_jsonl(self.workdir / names["fit"])
        if not wrong_fit:
            raise SystemExit(f"{names['fit']} rỗng.")
        check_fit_rows(wrong_fit, names["fit"], self.cfg["student"]["base_model_id"])
        overlap_fit = [r["tid"] for r in wrong_fit if r["tid"] in self.fit]
        if overlap_fit:
            raise SystemExit(f"{names['fit']} chứa {len(overlap_fit)} chuỗi đã có trong {self.fit_name}; hai file "
                             f"phải rời nhau.")
        npz = np.load(self.workdir / names["embeddings"])
        emb_row = dict(self.emb_row)
        emb_row.update({str(t): len(self.vecs) + i for i, t in enumerate(npz["tids"])})
        return (quality, {**self.fit, **{r["tid"]: r for r in wrong_fit}}, emb_row,
                np.vstack([self.vecs, npz["vectors"]]))


# ============================================================ chạy một phương án
def run_one(inp: Inputs, cfg: Mapping, outdir: Path) -> dict:
    spec, sel = cfg["select"], cfg["selection"]
    k, tag = int(sel["k"]), selection_tag(cfg)
    questions = inp.questions(spec["pool"], spec["rule"])
    picks = select_all(questions, cfg)
    check_shape(picks, set(inp.pool), k, tag)
    path = outdir / f"train.{tag}.jsonl"
    write_jsonl(path, train_rows(questions, picks, inp.qtext))
    summary = {"tag": tag, "name": spec["name"], "pool": spec["pool"], "rule": spec["rule"],
               "signal": spec.get("signal"), "weights": spec.get("weights", "uniform"),
               "a": float(sel["a"]), "b": float(sel["b"]), "lambda_div": float(sel["lambda_div"]),
               "div_scale": sel["div_scale"], "random_seed": inp.seed, "fit_file": inp.fit_name,
               "student": cfg["student"]["base_model_id"], "blocked": spec.get("blocked"),
               **summarize(questions, picks, k),
               "train_file": path.name, "md5": hashlib.md5(path.read_bytes()).hexdigest()}
    write_json(outdir / f"select.{tag}.json", summary)
    return summary


def print_summary(s: Mapping) -> None:
    share = "  ".join(f"{t} {v:.1%}" for t, v in s["teacher_share"].items())
    line = (f"[b2] {s['tag']:<26} {s['questions']} câu, {s['samples']} mẫu | {s['mean_tokens']:.0f} token | "
            f"{share} | phá hoà {s['ties']} câu")
    if s["wrong_kept"]:
        line += f" | chuỗi sai {s['wrong_kept']}"
    if s["lengths_from_api"]:
        line += f" | {s['lengths_from_api']} chuỗi đo độ dài bằng token của API"
    if s["greedy_same"] is not None:
        line += f" | tham lam trùng {s['greedy_same']:.1%}"
    if s.get("blocked"):
        line += " | BẢN TẠM, không dùng để huấn luyện"
    if s["weights"] != "uniform":
        line += f" | trọng số < 0,1: {s['weight_below_0.1']} chuỗi, cỡ mẫu hiệu dụng {s['effective_samples']:.0f}"
    print(line)


def method_list(cfg: Mapping, root: Path) -> list[tuple[str, str | None]]:
    """(method, ablation) cho --all. Mô hình học có khoá methods thì chỉ chạy đúng các phương án đó."""
    if "methods" in cfg:
        return [(m, None) for m in cfg["methods"]]
    methods = sorted(p.stem for p in (root / "configs" / "method").glob("*.yaml"))
    ablations = sorted(p.stem for p in (root / "configs" / "ablation").glob("*.yaml"))
    return [(m, None) for m in methods] + [("qd_rsr", a) for a in ablations]


# ============================================================ báo cáo
def _tid_sets(path: Path) -> dict:
    by_q: dict[str, set] = defaultdict(set)
    for r in read_jsonl(path):
        by_q[r["qid"]].add(r["tid"])
    return by_q


def overlap(a: Mapping[str, set], b: Mapping[str, set], only: set | None = None) -> dict:
    """Hai tập chọn giống nhau tới đâu: tỷ lệ mẫu chung, và tỷ lệ câu chọn đúng cùng tập."""
    qs = sorted(q for q in set(a) & set(b) if only is None or q in only)
    shared = sum(len(a[q] & b[q]) for q in qs)
    total = sum(len(a[q]) for q in qs)
    return {"questions": len(qs), "sample_share": shared / total if total else None,
            "same_set": sum(a[q] == b[q] for q in qs) / len(qs) if qs else None}


def report(outdir: Path, versus: Path | None = None) -> int:
    summaries = sorted((read_json(p) for p in outdir.glob("select.*.json")), key=lambda s: s["tag"])
    if not summaries:
        raise SystemExit(f"Chưa có tập chọn nào trong {outdir}. Chạy --all trước.")
    pool = read_json(outdir / "pool.json")
    teachers = sorted(pool["teacher_share"])
    print(f"Kho: {pool['questions']} câu, {pool['chains']} chuỗi dùng được, trong đó {pool['judge_zero']} chuỗi "
          f"đáp án đúng nhưng giám khảo chấm 0,0.\n")
    head = ["Cách chọn"] + teachers + ["Token TB", "Div TB", "Giám khảo 0,0", "Chuỗi sai", "Câu phá hoà"]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    print("| Toàn kho ứng viên | " + " | ".join(f"{pool['teacher_share'].get(t, 0):.1%}" for t in teachers)
          + f" | {pool['mean_tokens']:.0f} | | {pool['judge_zero']} | 0 | |")
    for s in summaries:
        div = f"{s['mean_div']:.3f}" if s["mean_div"] is not None else ""
        print(f"| {s['tag']} | " + " | ".join(f"{s['teacher_share'].get(t, 0):.1%}" for t in teachers)
              + f" | {s['mean_tokens']:.0f} | {div} | {s['judge_zero_kept']} | {s['wrong_kept']} | {s['ties']} |")

    sets = {s["tag"]: _tid_sets(outdir / s["train_file"]) for s in summaries}
    tags = list(sets)
    print("\nTỷ lệ mẫu của hàng cũng có trong cột (%), đánh số theo danh sách:")
    for i, t in enumerate(tags, 1):
        print(f"  {i:>2}. {t}")
    print("      " + "".join(f"{i:>5}" for i in range(1, len(tags) + 1)))
    for i, x in enumerate(tags, 1):
        print(f"  {i:>2}. " + "".join(f"{100 * overlap(sets[x], sets[y])['sample_share']:>5.0f}" for y in tags))

    zero = next((s for s in summaries if s["rule"] == "objective" and s["pool"] == "correct" and s["k"] > 1
                 and s["a"] == 1 and s["b"] == 1 and s["lambda_div"] == 0), None)
    lams = [s for s in summaries if zero and s["name"] == "qd_rsr" and s["k"] == zero["k"] and s["lambda_div"] > 0]
    if lams:
        big = {q for q, n in pool["n_candidates"].items() if n > zero["k"]}
        print(f"\nTỷ lệ câu đổi tập so với λ = 0 ({zero['tag']}), trên {len(big)} câu có hơn k ứng viên. Phải khớp"
              f" lambda_scan (lệch nhỏ do cách phá hoà):")
        for s in lams:
            same = overlap(sets[s["tag"]], sets[zero["tag"]], big)["same_set"]
            print(f"  λ = {s['lambda_div']:g}: {1 - same:.1%}" + (f", tham lam trùng nghiệm tối ưu {s['greedy_same']:.1%}"
                                                           if s["greedy_same"] is not None else ""))

    if versus is not None:
        print(f"\nSo với {versus.name}: cùng phương án, hai mô hình học")
        print(f"  {'phương án':<26}{'mẫu chung':>11}{'câu cùng tập':>14}{'trong các câu có hơn k ứng viên':>34}")
        for s in summaries:
            other = versus / s["train_file"]
            if not other.exists():
                continue
            theirs = _tid_sets(other)
            big = {q for q, n in pool["n_candidates"].items() if n > s["k"]}
            o, ob = overlap(sets[s["tag"]], theirs), overlap(sets[s["tag"]], theirs, big)
            print(f"  {s['tag']:<26}{o['sample_share']:>11.1%}{o['same_set']:>14.1%}"
                  f"{ob['same_set']:>24.1%} ({ob['questions']} câu)")
    return 0


# ============================================================ dòng lệnh
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/stage_a")
    ap.add_argument("--student", default="qwen1_5b", help="tên file trong configs/student/")
    ap.add_argument("--method", help="tên file trong configs/method/")
    ap.add_argument("--ablation", help="tên file trong configs/ablation/ (đè lên phương án qd_rsr)")
    ap.add_argument("--all", action="store_true", help="mọi phương án và biến thể của mô hình học này")
    ap.add_argument("--lambda-grid", action="store_true",
                    help="chỉ dùng với --method qd_rsr: chạy từng λ trong selection.lambda_grid")
    ap.add_argument("--fit", help="file fit, mặc định fit.<student>_base.jsonl")
    ap.add_argument("--outdir", help="mặc định data/stage_b/<student>_base")
    ap.add_argument("--report", action="store_true", help="in bảng I.9 và độ trùng từ các tập đã chọn")
    ap.add_argument("--versus", help="với --report: so cùng phương án với mô hình học khác, ví dụ qwen7b")
    ap.add_argument("--expect-questions", type=int, help="dừng với mã lỗi nếu số câu hỏi khác giá trị này")
    ap.add_argument("--allow-blocked", action="store_true",
                    help="vẫn chọn cho phương án đang bị khoá (select.blocked), chỉ để phân tích; file mang đuôi .tam")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    base = load_config(student=args.student, overrides=args.override)
    outdir = resolve_path(base, args.outdir) if args.outdir else path_of(base, "data_stage_b") / student_tag(base)
    if args.report:
        versus = None
        if args.versus:
            versus = path_of(base, "data_stage_b") / student_tag(load_config(student=args.versus))
        return report(outdir, versus)

    if args.all:
        # Các biến thể tự ghim a, b, λ; đè từ dòng lệnh sẽ xoá mất chỗ ghim (override được gộp sau cùng).
        pinned = [o for o in args.override if o.split("=")[0].strip() in PINNED_IN_ALL]
        if pinned or args.lambda_grid:
            ap.error(f"--all không nhận {pinned or '--lambda-grid'}: đổi λ thì sửa selection.lambda_div trong "
                     f"configs/base.yaml, để các biến thể vẫn giữ giá trị ghim của chúng")
        todo = method_list(base, Path(base["root"]))
    elif args.ablation:
        todo = [(args.method or "qd_rsr", args.ablation)]
    elif args.method:
        todo = [(args.method, None)]
    else:
        ap.error("cần --method, --ablation, --all hoặc --report")
    if args.lambda_grid and (args.ablation or args.method != "qd_rsr"):
        ap.error("--lambda-grid chỉ dùng với --method qd_rsr")

    inp = Inputs(base, resolve_path(base, args.workdir), args.fit or f"fit.{student_tag(base)}.jsonl")
    i = inp.info
    print(f"[b2] {inp.fit_name} ({base['student']['base_model_id']}); {i['candidates']} ứng viên, loại "
          f"{i['excluded_no_qual']} chuỗi thiếu Qual, {i['questions_before']} → {i['questions']} câu "
          f"({i['usable_candidates']} chuỗi dùng được)"
          + (f"; câu bị loại: {i['dropped_qids']}" if i["dropped_qids"] else ""))
    if args.expect_questions is not None and i["questions"] != args.expect_questions:
        raise SystemExit(f"[b2] DỪNG: {i['questions']} câu, mong đợi {args.expect_questions}.")
    ensure_dir(outdir)
    write_json(outdir / "pool.json", pool_stats(inp.questions("correct", "topk")))

    failed = []
    for method, ablation in todo:
        cfg = load_config(method=method, ablation=ablation, student=args.student, overrides=args.override)
        if "select" not in cfg:
            failed.append((ablation or method, "file cấu hình chưa khai báo khoá select"))
            continue
        for lam in (cfg["selection"]["lambda_grid"] if args.lambda_grid else [None]):
            if lam is not None:
                cfg = load_config(method=method, student=args.student,
                                  overrides=list(args.override) + [f"selection.lambda_div={lam}"])
            try:
                why = cfg["select"].get("blocked")
                if why and not args.allow_blocked:
                    raise SystemExit(f"đang bị khoá, không dùng để huấn luyện: {why} Muốn có tập chọn để phân tích "
                                     f"thì thêm --allow-blocked (file mang đuôi .tam).")
                print_summary(run_one(inp, cfg, outdir))
            except SystemExit as e:
                if not args.all:
                    raise
                failed.append((selection_tag(cfg), str(e)))
    for name, why in failed:
        print(f"[b2] CHƯA CHẠY ĐƯỢC {name}: {why}")
    print(f"[b2] file ghi ở {outdir}. Nhắc lại: chỉ RSR là cực tiểu.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

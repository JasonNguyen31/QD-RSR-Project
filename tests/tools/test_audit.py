"""Kiểm thử công cụ kiểm tra toàn vẹn: dựng một thư mục dữ liệu nhỏ đúng chuẩn, rồi phá từng thứ một."""
import json

import numpy as np
import pytest

from src.common.io_utils import write_json, write_jsonl
from src.tools import audit_stage_a as audit


def build(tmp_path, *, break_what=None):
    """Dựng một bộ dữ liệu 2 câu × 3 chuỗi hợp lệ. break_what chọn chỗ cố ý làm hỏng."""
    wd = tmp_path / "stage_a"
    wd.mkdir()
    qs = [{"qid": f"q{i}", "source": "math", "level": 4, "question": "?", "gold": "1"} for i in (1, 2)]
    write_jsonl(wd / "questions.jsonl", qs)

    # Số mô hình dạy và số mẫu phải khớp cấu hình thật, vì công cụ kiểm tra dựa vào đó để biết đủ chuỗi chưa.
    from src.common.config import load_config
    cfg = load_config()
    teachers = [t["key"] for t in cfg["teachers"]]
    n_samples = cfg["generation"]["samples_per_teacher"]

    trajs, labels, cands = [], [], []
    for q in qs:
        for teacher in teachers:
            for s in range(n_samples):
                tid = f"{q['qid']}|{teacher}|{s}"
                trajs.append({"tid": tid, "qid": q["qid"], "teacher": teacher, "sample_idx": s,
                              "text": f"x{s}", "prompt_id": "rsr", "finish_reason": "stop",
                              "completion_tokens": 10})
                labels.append({"tid": tid, "qid": q["qid"], "teacher": teacher, "correct": True,
                               "pred": "1", "reason": "match"})
                cands.append({"tid": tid, "qid": q["qid"], "text": f"x{s}"})
    if break_what == "truncated_but_correct":
        labels[0]["reason"] = "truncated"
    for teacher in teachers:
        write_jsonl(wd / f"trajectories.{teacher}.jsonl", [t for t in trajs if t["teacher"] == teacher])
    write_jsonl(wd / "labels.jsonl", labels)
    if break_what == "thin_question":
        cands = [c for c in cands if c["qid"] == "q1"] + [c for c in cands if c["qid"] == "q2"][:1]
    write_jsonl(wd / "candidates.jsonl", cands)
    # Đúng cấu trúc a3 ghi ra: một đối tượng có khoá qids, không phải danh sách trần
    write_json(wd / "kept_qids.json", {"min_correct": 3, "qids": [q["qid"] for q in qs]})

    write_jsonl(wd / "quality.jsonl", [{"tid": c["tid"], "qid": c["qid"], "rule_score": 0.0,
                                        "llm_score": 0.9, "qual": 0.45, "n_words": 5} for c in cands])
    tids = [c["tid"] for c in cands]
    if break_what == "stale_embeddings":
        tids = tids[:-1]
    rng = np.random.default_rng(0)                      # vector ngẫu nhiên đã chuẩn hoá L2
    v = rng.normal(size=(len(tids), 4)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.savez_compressed(wd / "embeddings.npz", tids=np.array(tids), vectors=v)
    return wd


def run(wd, capsys):
    code = audit.main(["--workdir", str(wd), "--validation", str(wd / "khong-co.jsonl")])
    return code, capsys.readouterr().out


def test_clean_dataset_passes(tmp_path, capsys):
    code, out = run(build(tmp_path), capsys)
    assert code == 0 and "HỎNG" not in out


def test_detects_stale_embeddings(tmp_path, capsys):
    """Lỗi nguy hiểm nhất: candidates đã đổi nhưng embeddings.npz còn là bản cũ."""
    code, out = run(build(tmp_path, break_what="stale_embeddings"), capsys)
    assert code >= 1 and "ĐÃ LỆCH" in out


def test_detects_truncated_marked_correct(tmp_path, capsys):
    code, out = run(build(tmp_path, break_what="truncated_but_correct"), capsys)
    assert code >= 1 and "chuỗi bị cắt luôn tính là sai" in out


def test_detects_question_with_too_few_candidates(tmp_path, capsys):
    code, out = run(build(tmp_path, break_what="thin_question"), capsys)
    assert code >= 1


def test_reports_usable_pool_size(tmp_path, capsys):
    _, out = run(build(tmp_path), capsys)
    assert "Tập dùng cho Giai đoạn B: 2 câu hỏi" in out


def test_reads_kept_qids_written_as_object_or_plain_list(tmp_path, capsys):
    """a3 ghi {"min_correct": .., "qids": [..]}; công cụ phải đọc được cả dạng danh sách trần."""
    wd = build(tmp_path)
    write_json(wd / "kept_qids.json", ["q1", "q2"])
    code, out = run(wd, capsys)
    assert code == 0 and "ứng viên đúng bằng tập câu được giữ" in out


def test_machine_without_trajectories_reports_skipped_not_failed(tmp_path, capsys):
    """Máy Giai đoạn B chỉ có candidates, quality và embeddings. Thiếu chuỗi và nhãn KHÔNG phải lỗi."""
    wd = build(tmp_path)
    for f in list(wd.glob("trajectories.*.jsonl")) + [wd / "labels.jsonl"]:
        f.unlink()
    code, out = run(wd, capsys)
    assert code == 0, "thiếu file chuỗi không được tính là hỏng"
    assert "BỎ QUA" in out and "bỏ qua" in out
    assert "Tập dùng cho Giai đoạn B: 2 câu hỏi" in out

"""Kiểm thử công thức tín hiệu đối chiếu bài gốc (29/09/2026). Không cần torch.

Mỗi phép thử công thức đều so với một cách tính độc lập viết lại từ bài báo hoặc mã chính thức, không so
với chính hàm đang thử, để tránh lỗi "dữ liệu giả sai giống mã" từng gặp ở hình A-8.
"""
import math

import pytest

from src.stage_b.signals import (SIGNALS_VERSION, add_lark, lark_ghat, lark_weights, local_items,
                                 mean_over_steps, prepare_fit_rows, sentence_ends, token_steps)


# ============================================================ LARK: ĝ
def official_compute_g_hat(Ls, Bs):
    """Chép nguyên compute_g_hat trong lark/score_lark.py (repo Tianrun-Yu/LARK, commit 38cfd4f)."""
    rhos = [b / l if l > 1e-30 else 0.0 for b, l in zip(Bs, Ls)]
    sum_L = sum(Ls)
    sum_rho_L = sum(r * l for r, l in zip(rhos, Ls))
    return rhos, [(L / sum_L) * (2.0 * rho - sum_rho_L / sum_L) for L, rho in zip(Ls, rhos)]


def test_ghat_matches_official_code():
    ce = [0.42, 0.61, 0.35, 0.90, 0.55]
    br = [0.30, 0.41, 0.22, 0.52, 0.47]
    rho, g = lark_ghat(ce, br)
    rho_o, g_o = official_compute_g_hat(ce, br)
    assert rho == pytest.approx(rho_o) and g == pytest.approx(g_o)


def test_ghat_matches_equation_7_by_hand():
    """Hai ứng viên: ℓ = (1, 2), Brier = (0.5, 0.6) nên ρ̂ = (0.5, 0.3), Σρ̂ℓ = 1.1, Σℓ = 3."""
    _rho, g = lark_ghat([1.0, 2.0], [0.5, 0.6])
    assert g[0] == pytest.approx((1 / 3) * (2 * 0.5 - 1.1 / 3))
    assert g[1] == pytest.approx((2 / 3) * (2 * 0.3 - 1.1 / 3))


def test_ghat_ranking_differs_from_brier_ranking():
    """Lý do mã cũ sai: xếp theo Brier thì chuỗi B đứng đầu, xếp theo ĝ thì chuỗi A đứng đầu.

    ĝ_k tỷ lệ với 2·Brier_k − c·ℓ_k (c chung cho cả câu hỏi), nên một chuỗi Brier hơi cao hơn nhưng entropy
    chéo cao hơn nhiều sẽ bị ĝ đẩy xuống.
    """
    ce = [0.40, 1.60, 0.80]
    br = [0.35, 0.40, 0.30]
    _rho, g = lark_ghat(ce, br)
    assert max(range(3), key=lambda i: br[i]) == 1
    assert max(range(3), key=lambda i: g[i]) == 0


def test_ghat_is_proportional_to_two_brier_minus_c_loss():
    ce = [0.3, 0.7, 1.1, 0.5]
    br = [0.25, 0.44, 0.61, 0.20]
    _rho, g = lark_ghat(ce, br)
    s = sum(ce)
    c = sum(br) / s                      # Σρ̂ℓ = ΣBrier
    for gi, l, b in zip(g, ce, br):
        assert gi == pytest.approx((2 * b - c * l) / s)


def test_ghat_rejects_bad_input():
    with pytest.raises(ValueError):
        lark_ghat([], [])
    with pytest.raises(ValueError):
        lark_ghat([1.0], [0.5, 0.5])


def test_add_lark_groups_by_question_not_by_file():
    """ĝ của một chuỗi chỉ phụ thuộc các ứng viên cùng câu hỏi."""
    rows = [
        {"tid": "q1|deepseek|0", "qid": "q1", "mean_surprisal": 1.0, "brier": 0.5},
        {"tid": "q1|llama70b|0", "qid": "q1", "mean_surprisal": 2.0, "brier": 0.6},
        {"tid": "q2|deepseek|0", "qid": "q2", "mean_surprisal": 5.0, "brier": 0.9},
    ]
    out = add_lark(rows)
    _r, g = lark_ghat([1.0, 2.0], [0.5, 0.6])
    assert out[0]["lark"] == pytest.approx(g[0]) and out[1]["lark"] == pytest.approx(g[1])
    # câu q2 chỉ có một ứng viên: ĝ = 1 · (2ρ̂ − ρ̂) = ρ̂
    assert out[2]["lark"] == pytest.approx(0.9 / 5.0)


def test_add_lark_refuses_old_files():
    with pytest.raises(ValueError, match="chạy lại b1_fit"):
        add_lark([{"tid": "q1|a|0", "rsr": 3.0, "lark": 0.4, "mean_surprisal": 1.0}])


# ============================================================ LARK: trọng số mềm
def lemma3(g, B):
    """Bổ đề 3 viết lại trực tiếp: q_i = (ĝ_i − ĝ_{B+1}) / Σ_{j≤B}(ĝ_j − ĝ_{B+1})."""
    order = sorted(range(len(g)), key=lambda i: g[i], reverse=True)
    thr = g[order[B]]
    den = sum(g[i] - thr for i in order[:B])
    w = [0.0] * len(g)
    for i in order[:B]:
        w[i] = (g[i] - thr) / den
    return w


def test_weights_follow_lemma_3():
    g = [0.12, 0.40, 0.05, 0.33, 0.21, 0.18]
    assert lark_weights(g, 3) == pytest.approx(lemma3(g, 3))


def test_weights_sum_to_one_and_select_exactly_b():
    g = [0.3, 0.1, 0.7, 0.5, 0.2, 0.6, 0.4, 0.0, 0.8]
    w = lark_weights(g, 3)
    assert sum(w) == pytest.approx(1.0)
    assert sum(1 for x in w if x > 0) == 3
    assert {i for i, x in enumerate(w) if x > 0} == {8, 2, 5}


def test_weights_are_not_uniform():
    """Trọng số mềm: chuỗi điểm cao nhất nặng hơn, khác hẳn 1/3 của các baseline."""
    w = lark_weights([0.9, 0.5, 0.4, 0.1], 3)
    assert w[0] > w[1] > w[2] > 0 and w[3] == 0


def test_budget_one_puts_all_weight_on_the_best():
    assert lark_weights([0.2, 0.9, 0.5], 1) == pytest.approx([0.0, 1.0, 0.0])


def test_exactly_b_candidates_gives_uniform_weights():
    """Quyết định của nhóm, khác mã chính thức (vốn cho chuỗi cuối trọng số gần 0)."""
    assert lark_weights([0.5, 0.2, 0.9], 3) == pytest.approx([1 / 3] * 3)


def test_all_selected_tied_with_threshold_gives_uniform():
    assert lark_weights([0.5, 0.5, 0.5, 0.5], 3) == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.0])


def test_nan_is_never_selected():
    w = lark_weights([float("nan"), 0.2, 0.1, 0.3], 2)
    assert w[0] == 0.0 and sum(w) == pytest.approx(1.0)


# ============================================================ Local Naturalness: cắt câu
def pieces(text):
    ends = sentence_ends(text)
    return [text[a:b] for a, b in zip([0] + ends[:-1], ends)]


def test_sentences_split_after_period_before_the_space():
    """Dấu cách thuộc câu sau, vì token của Qwen mang dấu cách ở đầu."""
    assert pieces("We need 3 times 4. Since 12 is the product, done.") == \
        ["We need 3 times 4.", " Since 12 is the product, done."]


def test_decimals_are_not_sentence_ends():
    assert len(pieces("The value is 3.5 and the other is 0.25 exactly.")) == 1


def test_newlines_end_a_sentence_and_belong_to_it():
    assert pieces("First line here\n\nSecond line here") == ["First line here\n\n", "Second line here"]


def test_tiny_fragments_are_merged_into_the_next_sentence():
    """"1." đứng riêng sẽ thành một bước nặng ngang cả một câu dài: gộp vào câu sau."""
    assert pieces("1. Compute the sum first. Then divide it.") == \
        ["1. Compute the sum first.", " Then divide it."]


def test_pieces_cover_the_whole_text():
    t = "Let x = 2. Then y = x + 1.\nSo y = 3. We check: 3 - 1 = 2. Therefore \\boxed{3}."
    assert "".join(pieces(t)) == t and len(pieces(t)) == 5


def test_token_steps_follow_token_start():
    text = "Ann has 5. Tom has 8."
    # token giả kiểu Qwen: "Ann", " has", " 5", ".", " Tom", " has", " 8", "."
    toks = ["Ann", " has", " 5", ".", " Tom", " has", " 8", "."]
    offsets, pos = [], 0
    for t in toks:
        offsets.append((pos, pos + len(t)))
        pos += len(t)
    assert pos == len(text)
    assert token_steps(offsets, text) == [(0, 4), (4, 8)]


def test_token_steps_are_contiguous_and_cover_all_tokens():
    text = "Let x = 2. Then y = 3.\n\nSo the sum is 5. Done here."
    offsets = [(i, i + 1) for i in range(len(text))]          # mỗi ký tự một token
    steps = token_steps(offsets, text)
    assert steps[0][0] == 0 and steps[-1][1] == len(offsets)
    assert all(a[1] == b[0] for a, b in zip(steps, steps[1:]))


def test_local_items_use_at_most_k_previous_steps():
    steps = [(0, 5), (5, 9), (9, 20), (20, 22), (22, 30), (30, 41)]
    items = local_items(steps, k=4)
    assert items[0] == (0, 0, 5)          # câu đầu: chỉ có đề bài làm ngữ cảnh
    assert items[3] == (0, 20, 22)        # 3 câu trước
    assert items[4] == (0, 22, 30)        # đúng 4 câu trước
    assert items[5] == (5, 30, 41)        # câu 0 đã ra khỏi cửa sổ
    assert local_items(steps, k=0)[5] == (30, 30, 41)


def test_local_score_averages_over_steps_not_tokens():
    """Câu ngắn và câu dài nặng như nhau, theo phương trình 2 của Just 2025."""
    step_means = [-0.2, -1.0]            # câu 1 có 2 token, câu 2 có 18 token
    assert mean_over_steps(step_means) == pytest.approx(-0.6)
    token_mean = (2 * -0.2 + 18 * -1.0) / 20
    assert mean_over_steps(step_means) != pytest.approx(token_mean)


# ============================================================ đọc file fit
def test_prepare_new_file_adds_lark():
    rows = [{"tid": f"q1|t|{i}", "qid": "q1", "rsr": 3.0 + i, "grape": -1.0, "local_nat": -0.9,
             "brier": 0.3 + 0.1 * i, "mean_surprisal": 1.0 + i, "signals_version": SIGNALS_VERSION}
            for i in range(3)]
    out, stale = prepare_fit_rows(rows)
    assert stale == [] and all("lark" in r for r in out)


def test_prepare_old_file_drops_the_two_redefined_signals():
    rows = [{"tid": "q1|t|0", "rsr": 3.0, "grape": -1.0, "local_nat": -0.9, "lark": 0.4}]
    out, stale = prepare_fit_rows(rows)
    assert set(stale) == {"lark", "local_nat"}
    assert "lark" not in out[0] and "local_nat" not in out[0] and out[0]["rsr"] == 3.0

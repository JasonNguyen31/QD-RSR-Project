"""Sửa 20/10: khi hết bộ nhớ, c1_evaluate phải thả bộ nhớ của lần thử hỏng TRƯỚC khi sinh lại với nửa lô."""
import gc
import weakref

import pytest

torch = pytest.importorskip("torch")

from src.stage_c import c1_evaluate as c1


class Big:
    """Đóng vai bộ nhớ đệm KV của một lần sinh: chỉ được giữ bởi khung gọi của generate."""


class FakeModel:
    device = "cpu"

    def __init__(self, fits):
        self.fits, self.alive_at_retry, self.sizes, self.last = fits, [], [], None

    def generate(self, input_ids, **kw):
        self.alive_at_retry.append(self.last is not None and self.last() is not None)
        cache = Big()                              # tham chiếu duy nhất nằm trong khung gọi này
        self.last = weakref.ref(cache)
        n = input_ids.shape[0]
        self.sizes.append(n)
        if n > self.fits:
            raise torch.OutOfMemoryError("giả lập hết bộ nhớ")
        new = torch.full((n, 2), 7, dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)


class FakeTok:
    pad_token_id = 0


EV = {"temperature": 0.6, "top_p": 0.95, "top_k": -1}


def test_oom_splits_in_half_and_releases_the_failed_attempt_first(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    m = FakeModel(fits=2)
    rows, splits = c1._generate_batch(m, FakeTok(), [[1, 2, 3]] * 8, 2, [9], EV, 0)
    assert rows == [[7, 7]] * 8
    assert m.sizes == [8, 4, 2, 2, 4, 2, 2] and splits == 3
    # mỗi lần sinh lại sau một lần hết bộ nhớ: bộ nhớ của lần hỏng trước đó đã được thả
    assert not any(m.alive_at_retry)


def test_a_single_row_that_does_not_fit_still_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with pytest.raises(torch.OutOfMemoryError):
        c1._generate_batch(FakeModel(fits=0), FakeTok(), [[1, 2]] * 2, 2, [9], EV, 0)


def test_seven_b_uses_smaller_generation_batches():
    from src.common.config import load_config
    assert load_config(student="qwen7b").eval.gen_batch == [32, 16, 8]
    assert load_config(student="qwen1_5b").eval.gen_batch == [64, 32, 16]     # 1,5B giữ nguyên

"""
Đọc ghi jsonl, ghi nối tiếp an toàn khi chạy đa luồng, và lấy lại tiến độ.

Định danh chuỗi: tid = "<qid>|<teacher>|<sample_idx>". Mọi bảng điểm ở Giai đoạn A và B nối với nhau
qua tid, nên không được tự ý đổi định dạng này.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def traj_id(qid: str, teacher: str, sample_idx: int) -> str:
    return f"{qid}|{teacher}|{int(sample_idx)}"


def split_tid(tid: str) -> tuple[str, str, int]:
    qid, teacher, idx = tid.rsplit("|", 2)  # qid được phép chứa "|"
    return qid, teacher, int(idx)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    """Đọc từng dòng. File không tồn tại thì coi như rỗng. Dòng cuối hỏng (tiến trình bị ngắt giữa chừng)
    thì bỏ qua kèm cảnh báo; dòng hỏng ở giữa file là lỗi thật nên báo lỗi."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    last_nonblank = max((i for i, ln in enumerate(lines) if ln.strip()), default=-1)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if i == last_nonblank:
                print(f"[io] cảnh báo: bỏ dòng cuối hỏng ở {path} (dòng {i + 1}), "
                      f"có thể tiến trình đã bị dừng giữa chừng", file=sys.stderr)
                return
            raise ValueError(f"{path}: dòng {i + 1} không phải JSON hợp lệ") from None


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    return list(iter_jsonl(path))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    """Ghi cả file một lần, nguyên tử (không để lại file dở dang nếu bị ngắt). Trả về số dòng."""
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    _atomic_write_text(Path(path), "".join(ln + "\n" for ln in lines))
    return len(lines)


def read_json(path: str | os.PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    _atomic_write_text(Path(path), json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


class JsonlWriter:
    """Ghi nối tiếp từng dòng, an toàn khi nhiều luồng cùng ghi, flush và fsync sau mỗi dòng.

    Mỗi dòng sinh ra tốn tiền API nên thà chậm vài mili giây còn hơn mất tiến độ khi máy sập.
    """

    def __init__(self, path: str | os.PathLike, fsync: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._lock = threading.Lock()
        needs_newline = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as rf:
                rf.seek(-1, os.SEEK_END)
                needs_newline = rf.read(1) != b"\n"
        self._f = self.path.open("a", encoding="utf-8")
        if needs_newline:  # dòng cuối của lần chạy trước bị cắt, tách ra để không dính vào dòng mới
            self._f.write("\n")
            self._f.flush()

    def append(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            self._f.write(line)
            self._f.flush()
            if self._fsync:
                os.fsync(self._f.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_done_keys(path: str | os.PathLike, key_fn: Callable[[dict], str]) -> set[str]:
    """Tập khoá đã có trong file jsonl, dùng để bỏ qua phần đã làm khi chạy lại."""
    return {key_fn(r) for r in iter_jsonl(path)}

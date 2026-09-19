"""
Đọc và gộp cấu hình YAML.

Thứ tự gộp (file sau đè file trước): base, data, models, student, method, ablation, overrides.

    cfg = load_config(method="qd_rsr", student="qwen15b")
    cfg.selection.k                       # truy cập bằng thuộc tính
    path_of(cfg, "data_stage_a")          # đường dẫn tuyệt đối theo gốc repo

Đặt QDRSR_ROOT nếu chạy mã từ nơi khác gốc repo.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


class AttrDict(dict):
    """dict cho phép truy cập bằng thuộc tính, gõ sai khoá sẽ báo lỗi thay vì trả None."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Cấu hình không có khoá '{name}'. Các khoá hiện có: {sorted(self)}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Gộp đệ quy: dict thì gộp từng khoá, còn lại (kể cả list) thì thay hẳn."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def repo_root() -> Path:
    env = os.environ.get("QDRSR_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]  # src/common/config.py -> gốc repo


def load_env_file(root: Path) -> None:
    """Nạp .env ở gốc repo vào os.environ (không đè biến đã có). Không cần thư viện ngoài."""
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file cấu hình: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_override(text: str) -> dict:
    """'selection.k=1' -> {'selection': {'k': 1}}. Giá trị đi qua yaml nên 1, 0.5, true, [1,2] đều đúng kiểu."""
    if "=" not in text:
        raise ValueError(f"Override phải có dạng khoá.con=giá_trị, nhận được: {text!r}")
    dotted, _, raw = text.partition("=")
    value = yaml.safe_load(raw)
    node: dict = {}
    cur = node
    parts = dotted.strip().split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return node


def load_config(
    method: str | None = None,
    ablation: str | None = None,
    student: str | None = None,
    overrides: list[str] | None = None,
    root: Path | None = None,
) -> AttrDict:
    root = Path(root) if root else repo_root()
    load_env_file(root)
    cfg_dir = root / "configs"

    merged: dict = {}
    for name in ("base", "data", "models"):
        merged = deep_merge(merged, _read_yaml(cfg_dir / f"{name}.yaml"))
    if student:
        merged = deep_merge(merged, _read_yaml(cfg_dir / "student" / f"{student}.yaml"))
    if method:
        merged = deep_merge(merged, _read_yaml(cfg_dir / "method" / f"{method}.yaml"))
    if ablation:
        merged = deep_merge(merged, _read_yaml(cfg_dir / "ablation" / f"{ablation}.yaml"))
    for text in overrides or []:
        merged = deep_merge(merged, parse_override(text))

    merged["root"] = str(root)
    return _wrap(merged)


def path_of(cfg: Mapping, key: str) -> Path:
    """Đường dẫn tuyệt đối của một khoá trong cfg.paths."""
    return Path(cfg["root"]) / cfg["paths"][key]


def resolve_path(cfg: Mapping, p: str | os.PathLike) -> Path:
    """Đường dẫn người dùng gõ trên dòng lệnh: tuyệt đối thì giữ, tương đối thì tính từ gốc repo."""
    p = Path(p)
    return p if p.is_absolute() else Path(cfg["root"]) / p


def signal_direction(cfg: Mapping, name: str) -> str:
    """'min' hoặc 'max'. Chỉ RSR là min."""
    try:
        d = cfg["signal_direction"][name]
    except KeyError:
        raise KeyError(f"Chưa khai báo chiều tối ưu cho tín hiệu '{name}' trong base.yaml") from None
    if d not in ("min", "max"):
        raise ValueError(f"Chiều tối ưu phải là min hoặc max, nhận được {d!r} cho '{name}'")
    return d

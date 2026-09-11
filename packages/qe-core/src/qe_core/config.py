"""Dotted-key YAML configuration with overlays and relative paths."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "repo_root"]

_MISSING = object()


def repo_root() -> Path:
    """Workspace root — the directory holding the top-level pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "packages").is_dir():
            return parent
    return here.parents[4]


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(text: str) -> Any:
    """Best-effort scalar parse for CLI overrides, via YAML's own rules."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


class Config:
    """Nested config addressed by dotted key."""

    def __init__(self, data: dict | None = None) -> None:
        self._data: dict = copy.deepcopy(data or {})

    # -- loading ----------------------------------------------------------

    @classmethod
    def load(cls, *paths: str | Path, overrides: list[str] | None = None) -> Config:
        """Load and deep-merge YAML files left to right, then apply `key=value` overrides."""
        data: dict = {}
        for p in paths:
            with open(p) as fh:
                loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise TypeError(f"{p}: top level must be a mapping, got {type(loaded).__name__}")
            data = _deep_merge(data, loaded)

        cfg = cls(data)
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"override must be key=value, got {item!r}")
            key, _, raw = item.partition("=")
            cfg.set(key.strip(), _coerce(raw.strip()))
        return cfg

    # -- access -----------------------------------------------------------

    def get(self, key: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"config key not found: {key!r}")
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def path(self, key: str) -> Path:
        """A path from config, resolved against the repo root unless already absolute."""
        raw = Path(str(self.get(key))).expanduser()
        return raw if raw.is_absolute() else repo_root() / raw

    def __contains__(self, key: str) -> bool:
        return self.get(key, None) is not None

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"<Config {sorted(self._data)}>"

#!/usr/bin/env python3
"""Repo-relative path helpers for committed evidence JSON."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

_PATH_KEY_HINTS = frozenset({
    "kernel", "camodel_log", "archived_kernel", "landed_kernel", "archive_dir",
    "kernel_path", "evidence_file", "root",
})


def repo_relative(path: Path | str, *, root: Path = REPO_ROOT) -> str:
    """Return a repo-relative POSIX path, or a portable placeholder when outside."""
    p = Path(path)
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    root_resolved = root.resolve()
    try:
        return resolved.relative_to(root_resolved).as_posix()
    except ValueError:
        eval_root = os.environ.get("PYASC_EVAL_ROOT", "")
        if eval_root:
            try:
                if resolved == Path(eval_root).resolve():
                    return "$PYASC_EVAL_ROOT"
            except OSError:
                pass
        return p.name


def portable_pyasc_revision(rev: dict[str, Any], *, root: Path = REPO_ROOT) -> dict[str, Any]:
    """Rewrite pyasc_revision.root for committed evidence."""
    out = dict(rev)
    raw_root = out.get("root")
    if not raw_root:
        return out
    try:
        Path(str(raw_root)).resolve().relative_to(root.resolve())
        out["root"] = repo_relative(str(raw_root), root=root)
    except ValueError:
        out["root"] = "$PYASC_EVAL_ROOT"
    return out


def _maybe_relativize_string(value: str, *, root: Path) -> str:
    if not value or value.startswith("$"):
        return value
    if value.startswith("/workspace/"):
        return value[len("/workspace/") :]
    p = Path(value)
    if not p.is_absolute():
        return value
    try:
        p.resolve().relative_to(root.resolve())
        return repo_relative(value, root=root)
    except (ValueError, OSError):
        return value


def relativize_evidence(obj: Any, *, root: Path = REPO_ROOT, parent_key: str = "") -> Any:
    """Recursively rewrite absolute paths under ``root`` in evidence structures."""
    if isinstance(obj, dict):
        return {
            k: relativize_evidence(v, root=root, parent_key=k)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [relativize_evidence(v, root=root, parent_key=parent_key) for v in obj]
    if isinstance(obj, str):
        if parent_key in _PATH_KEY_HINTS or (
            parent_key.endswith("_log")
            and obj.startswith("/")
        ):
            return _maybe_relativize_string(obj, root=root)
        if obj.startswith(str(root.resolve()) + "/"):
            return repo_relative(obj, root=root)
        home_prefix = obj.startswith("/home/") or obj.startswith("/Users/")
        if home_prefix and (
            parent_key in _PATH_KEY_HINTS
            or "/evidence/" in obj
            or "/golden/" in obj
            or "/teams/" in obj
        ):
            return _maybe_relativize_string(obj, root=root)
    return obj

#!/usr/bin/env python3
"""Unload every model currently resident in the host Ollama daemon.

CI memory hygiene for the Mac runner. The generative gates (nightly-gate,
local-stability-gate) and the perf-gate all serialize on the single
self-hosted arm64 runner, and they drive the Mac's *native* (Metal) Ollama
over host.docker.internal. Ollama keeps a model warm for `keep_alive` (5 min
by default), so when one leg's model (e.g. qwen3-coder:30b ~18 GB) is still
resident as the next leg loads a different one (e.g. gpt-oss:120b ~68 GB),
both sit in RAM at once (~86 GB) on top of the 46 GB Parallels VM + Docker
Desktop VM + macOS -- which overruns the 128 GB Mac and thrashes swap.

Calling this before each leg forces every resident model to unload (POST
/api/generate with keep_alive=0), so the peak Ollama footprint is bounded to
the single model the upcoming leg actually loads.

This is best-effort: an unreachable daemon or a failed unload is a warning,
never a hard error (the gates that call it are report-only / skip-guarded).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def _ollama_root() -> str:
    # OLLAMA_BASE_URL is the OpenAI-compatible URL (".../v1"); the management
    # endpoints (/api/ps, /api/generate) live at the daemon root.
    base = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
    return base[: -len("/v1")] if base.endswith("/v1") else base.rstrip("/")


def _resident_models(root: str) -> list[str]:
    try:
        with urllib.request.urlopen(root + "/api/ps", timeout=10) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"::warning::could not query Ollama /api/ps at {root}: {exc}")
        return []
    names = []
    for entry in data.get("models", []):
        name = entry.get("name") or entry.get("model")
        if name:
            names.append(name)
    return names


def _unload(root: str, model: str) -> None:
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(
        root + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        print(f"unloaded resident Ollama model: {model}")
    except (urllib.error.URLError, OSError) as exc:
        print(f"::warning::failed to unload Ollama model {model}: {exc}")


def main() -> int:
    root = _ollama_root()
    models = _resident_models(root)
    if not models:
        print(f"no resident Ollama models to unload at {root}")
        return 0
    for model in models:
        _unload(root, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())

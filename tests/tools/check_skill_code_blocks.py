#!/usr/bin/env python3
"""Check that Python code blocks in the skill docs are syntactically valid.

The skills are read by an agent and copied into kernels, so a code block that
does not parse is a defect that propagates. This is a cheap static guard: it
does not need pyasc, Docker, or a runtime, so it belongs in the PR gate.

Some blocks are *deliberately* not standalone Python — a few lines of a larger
function, or a counter-example showing syntax the compiler rejects. Those must
be marked with an HTML comment on the line before the fence, otherwise an
unmarked broken block is indistinguishable from a marked one::

    <!-- skill-code: fragment -->
    ```python
    out = x + y        # not a complete module
    ```

    <!-- skill-code: counter-example -->
    ```python
    valid = size - off
    if valid > tile:   # rejected by the compiler; shown on purpose
        valid = tile
    ```

Both markers skip the parse check. They are distinct so the intent is readable;
``counter-example`` additionally documents that the block is *meant* to be
wrong.

Usage::

    python3 check_skill_code_blocks.py [--root skills] [--json]

Exit codes: 0 = all good, 1 = at least one unmarked block failed to parse.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>python|py)[ \t]*$",
    re.MULTILINE,
)
MARKER_RE = re.compile(r"<!--\s*skill-code:\s*(fragment|counter-example)\s*-->")


def iter_blocks(text: str):
    """Yield (start_line, code, marker) for each fenced python block."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        indent = m.group("indent")
        closing = f"{indent}```"
        body: list[str] = []
        j = i + 1
        while j < len(lines) and lines[j].rstrip() != closing.rstrip():
            body.append(lines[j][len(indent):] if lines[j].startswith(indent) else lines[j])
            j += 1

        # The marker is the nearest preceding non-blank line.
        marker = None
        k = i - 1
        while k >= 0 and not lines[k].strip():
            k -= 1
        if k >= 0:
            found = MARKER_RE.search(lines[k])
            if found:
                marker = found.group(1)

        yield i + 1, "\n".join(body), marker
        i = j + 1


def _display_path(path: Path) -> str:
    """Repo-relative where possible, so output is stable across checkouts."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def check_file(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [{"file": _display_path(path), "line": 0,
                 "status": "unreadable", "error": str(exc)}]

    results: list[dict] = []
    for line_no, code, marker in iter_blocks(text):
        entry = {
            "file": _display_path(path),
            "line": line_no,
            "marker": marker,
        }
        if marker is not None:
            entry["status"] = "skipped"
            results.append(entry)
            continue
        if not code.strip():
            entry["status"] = "empty"
            results.append(entry)
            continue
        try:
            ast.parse(code)
            entry["status"] = "ok"
        except SyntaxError as exc:
            entry["status"] = "syntax-error"
            entry["error"] = f"{exc.msg} (block line {exc.lineno})"
        results.append(entry)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT / "skills"),
        help="Directory to scan for markdown (default: skills/)",
    )
    parser.add_argument("--json", action="store_true", help="JSON report on stdout")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    all_results: list[dict] = []
    for md in sorted(root.rglob("*.md")):
        all_results.extend(check_file(md))

    bad = [r for r in all_results if r["status"] in ("syntax-error", "unreadable")]
    ok = [r for r in all_results if r["status"] == "ok"]
    skipped = [r for r in all_results if r["status"] == "skipped"]

    if args.json:
        print(json.dumps({
            "checked": len(all_results),
            "parsed": len(ok),
            "skipped": len(skipped),
            "failed": len(bad),
            "failures": bad,
        }, indent=2))
    else:
        print(
            f"  {len(all_results)} python blocks: {len(ok)} parsed, "
            f"{len(skipped)} marked non-standalone, {len(bad)} broken"
        )
        for r in bad:
            print(f"  [FAIL] {r['file']}:{r['line']}: {r.get('error', r['status'])}")
            print("         mark it <!-- skill-code: fragment --> if intentional")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

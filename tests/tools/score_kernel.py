#!/usr/bin/env python3
"""Automated code-review scoring for pyasc kernels.

Implements 10 checklist categories derived from
skills/pyasc-codegen-workflow/references/code-review-checklist.md.

Each category is scored 0 or 1; the final score is out of 10.
The workflow acceptance threshold is >= 8.5.

Usage:
  python score_kernel.py <kernel.py> [--json]
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple


class Check(NamedTuple):
    name: str
    passed: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _is_asc_jit(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "asc" and node.attr == "jit"
    if isinstance(node, ast.Call):
        return _is_asc_jit(node.func)
    return False


def _jit_funcs(tree: ast.Module) -> list[ast.FunctionDef]:
    return [
        n for n in ast.iter_child_nodes(tree)
        if isinstance(n, ast.FunctionDef) and any(_is_asc_jit(d) for d in n.decorator_list)
    ]


def _all_calls(node: ast.AST) -> list[str]:
    """Return a flat list of callee names found under *node*."""
    names = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            c = n.func
            if isinstance(c, ast.Name):
                names.append(c.id)
            elif isinstance(c, ast.Attribute):
                names.append(c.attr)
    return names


BANNED_BUILTINS = {"print", "input", "open", "eval", "exec", "compile"}
BANNED_STMTS = (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith,
                ast.Yield, ast.YieldFrom, ast.Try,
                ast.Raise, ast.Global, ast.Nonlocal,
                ast.Import, ast.ImportFrom, ast.Lambda, ast.With)
try:
    BANNED_STMTS = (*BANNED_STMTS, ast.TryStar)
except AttributeError:
    pass


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(path: str) -> list[Check]:
    source = Path(path).read_text()
    tree = ast.parse(source)
    checks: list[Check] = []

    jit = _jit_funcs(tree)

    # 1. @asc.jit decorator present
    checks.append(Check("jit_decorator", len(jit) > 0,
                         f"{len(jit)} JIT function(s)" if jit else "missing"))

    # 2. Kernel does not return a value
    ret_ok = True
    for f in jit:
        for n in ast.walk(f):
            if isinstance(n, ast.Return) and n.value is not None:
                ret_ok = False
    checks.append(Check("no_return_value", ret_ok,
                         "OK" if ret_ok else "kernel returns a value"))

    # 3. No banned constructs
    ban_issues = []
    for f in jit:
        for n in ast.walk(f):
            if isinstance(n, BANNED_STMTS):
                ban_issues.append(type(n).__name__)
            if isinstance(n, ast.Call):
                c = n.func
                if isinstance(c, ast.Name) and c.id in BANNED_BUILTINS:
                    ban_issues.append(c.id)
            if isinstance(n, (ast.Break, ast.Continue)):
                ban_issues.append(type(n).__name__)
    checks.append(Check("no_banned_constructs", len(ban_issues) == 0,
                         "OK" if not ban_issues else f"found: {', '.join(ban_issues[:5])}"))

    # 4. sync flags (set_flag + wait_flag)
    calls_in_jit = []
    for f in jit:
        calls_in_jit.extend(_all_calls(f))
    has_sync = "set_flag" in calls_in_jit and "wait_flag" in calls_in_jit
    checks.append(Check("sync_flags", has_sync,
                         "set_flag+wait_flag" if has_sync else "missing sync primitives"))

    # 5. data_copy present
    has_dc = "data_copy" in calls_in_jit
    checks.append(Check("data_copy", has_dc,
                         "present" if has_dc else "missing data_copy"))

    # 6. Correct sync events (MTE2_V, V_MTE3, MTE3_MTE2 referenced anywhere)
    events_found = set()
    for event in ("MTE2_V", "V_MTE3", "MTE3_MTE2"):
        if event in source:
            events_found.add(event)
    checks.append(Check("sync_events", len(events_found) >= 2,
                         f"events: {', '.join(sorted(events_found)) or 'none'}"))

    # 7. Verification call (torch.allclose or numpy.allclose in file)
    has_verify = "allclose" in source
    checks.append(Check("verification", has_verify,
                         "allclose present" if has_verify else "no allclose call"))

    # 8. Launch pattern: kernel[core_num, stream](...)
    has_launch = bool(re.search(r'\w+\[.*,\s*rt\.\w+\(.*\)\]\(', source) or
                      re.search(r'\w+\[\w+,\s*\w+\]\(', source) or
                      re.search(r'\w+\[.*,\s*\w+\]\(', source))
    checks.append(Check("launch_pattern", has_launch,
                         "kernel launch found" if has_launch else "no launch pattern"))

    # 9. Tensor types used (GlobalTensor / LocalTensor / GlobalAddress)
    tensor_types = {"GlobalTensor", "LocalTensor", "GlobalAddress"}
    found_types = {t for t in tensor_types if t in source}
    checks.append(Check("tensor_types", len(found_types) >= 2,
                         f"found: {', '.join(sorted(found_types)) or 'none'}"))

    # 10. File is syntactically valid Python (already parsed above, so pass)
    checks.append(Check("valid_python", True, "parsed without errors"))

    return checks


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel.py> [--json]", file=sys.stderr)
        sys.exit(2)

    path = sys.argv[1]
    use_json = "--json" in sys.argv

    try:
        checks = score(path)
    except SyntaxError as exc:
        if use_json:
            print(json.dumps({"file": path, "score": 0.0, "error": str(exc)}))
        else:
            print(f"FAIL: SyntaxError: {exc}")
        sys.exit(1)
    except FileNotFoundError:
        if use_json:
            print(json.dumps({"file": path, "score": 0.0, "error": "file not found"}))
        else:
            print(f"FAIL: file not found: {path}")
        sys.exit(1)

    total = sum(1 for c in checks if c.passed)
    final_score = total  # out of 10

    if use_json:
        data = {
            "file": path,
            "score": float(final_score),
            "threshold": 8.5,
            "accepted": final_score >= 8.5,
            "checks": {c.name: {"passed": c.passed, "detail": c.detail} for c in checks},
        }
        print(json.dumps(data, indent=2))
    else:
        for c in checks:
            tag = "PASS" if c.passed else "FAIL"
            print(f"  [{tag}] {c.name}: {c.detail}")
        print(f"\n  Score: {final_score}/10 (threshold: 8.5)")
        if final_score >= 8.5:
            print("  ACCEPTED")
        else:
            print("  REJECTED")

    sys.exit(0 if final_score >= 8.5 else 1)


if __name__ == "__main__":
    main()

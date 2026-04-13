#!/usr/bin/env python3
"""Automated code-review scoring for pyasc asc2 kernels.

Implements 10 checklist categories for asc2 kernels.
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

def _is_jit(node: ast.expr) -> bool:
    """Match @asc.jit, @asc2.jit, or their call forms."""
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            if node.value.id in ("asc", "asc2") and node.attr == "jit":
                return True
    if isinstance(node, ast.Call):
        return _is_jit(node.func)
    return False


def _jit_funcs(tree: ast.Module) -> list[ast.FunctionDef]:
    return [
        n for n in ast.iter_child_nodes(tree)
        if isinstance(n, ast.FunctionDef) and any(_is_jit(d) for d in n.decorator_list)
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

    # 1. @asc2.jit or @asc.jit decorator present
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

    # 4. asc2.load / asc2.store present in JIT functions
    calls_in_jit = []
    for f in jit:
        calls_in_jit.extend(_all_calls(f))
    has_load_store = "load" in calls_in_jit and "store" in calls_in_jit
    checks.append(Check("asc2_load_store", has_load_store,
                         "load+store" if has_load_store else "missing asc2.load/asc2.store"))

    # 5. asc2.tensor present in JIT functions
    has_tensor = "tensor" in calls_in_jit
    checks.append(Check("asc2_tensor", has_tensor,
                         "present" if has_tensor else "missing asc2.tensor"))

    # 6. asc2.range or range used for loops in JIT
    has_range = "range" in calls_in_jit
    checks.append(Check("loop_range", has_range,
                         "range/asc2.range present" if has_range else "no range call in kernel"))

    # 7. Verification call (allclose or assert_allclose in file)
    has_verify = "allclose" in source
    checks.append(Check("verification", has_verify,
                         "allclose present" if has_verify else "no allclose call"))

    # 8. Launch pattern: kernel[core_num](...) — asc2 style (no stream)
    has_launch = bool(re.search(r'\w+\[\s*\w+\s*\]\(', source) or
                      re.search(r'\w+\[\s*\d+\s*\]\(', source))
    checks.append(Check("launch_pattern", has_launch,
                         "kernel launch found" if has_launch else "no launch pattern"))

    # 9. GlobalAddress used in kernel parameter types
    has_ga = "GlobalAddress" in source
    checks.append(Check("global_address", has_ga,
                         "GlobalAddress found" if has_ga else "missing GlobalAddress type"))

    # 10. File is syntactically valid Python (already parsed above)
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

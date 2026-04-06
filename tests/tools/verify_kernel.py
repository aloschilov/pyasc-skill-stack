#!/usr/bin/env python3
"""Static AST verifier for pyasc kernels.

Parses a kernel .py file and checks it against the pyasc syntax constraints:
  - @asc.jit decorated kernel function exists
  - Kernel does not return a value
  - No banned constructs inside JIT-decorated functions
  - Only allowed loop forms (for ... in range(...))
  - set_flag / wait_flag sync calls present
  - data_copy calls present
  - Verification with torch.allclose or numpy.allclose present

Exit 0 = all checks pass, exit 1 = one or more failures.
"""

import ast
import json
import sys
from pathlib import Path
from typing import NamedTuple


class CheckResult(NamedTuple):
    name: str
    passed: bool
    detail: str = ""


BANNED_BUILTINS_IN_JIT = {"print", "input", "open", "eval", "exec", "compile"}

BANNED_NODE_TYPES_IN_JIT = {
    ast.AsyncFunctionDef: "async def",
    ast.AsyncFor: "async for",
    ast.AsyncWith: "async with",
    ast.Yield: "yield",
    ast.YieldFrom: "yield from",
    ast.Try: "try/except",
    ast.Raise: "raise",
    ast.Global: "global",
    ast.Nonlocal: "nonlocal",
    ast.Import: "import",
    ast.ImportFrom: "from ... import",
}

try:
    BANNED_NODE_TYPES_IN_JIT[ast.TryStar] = "try/except*"
except AttributeError:
    pass


def _is_asc_jit_decorator(node: ast.expr) -> bool:
    """Return True if *node* represents ``@asc.jit`` or ``@asc.jit(...)``."""
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "asc"
            and node.attr == "jit"
        )
    if isinstance(node, ast.Call):
        return _is_asc_jit_decorator(node.func)
    return False


def _find_jit_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """Return top-level functions decorated with @asc.jit."""
    results = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if _is_asc_jit_decorator(dec):
                    results.append(node)
                    break
    return results


def _check_no_return_value(func: ast.FunctionDef) -> list[str]:
    """Kernel functions must not return a value (bare ``return`` is OK)."""
    issues = []
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            issues.append(
                f"line {node.lineno}: return with value in kernel '{func.name}'"
            )
    return issues


def _check_banned_constructs(func: ast.FunctionDef) -> list[str]:
    issues = []
    for node in ast.walk(func):
        node_type = type(node)
        if node_type in BANNED_NODE_TYPES_IN_JIT:
            label = BANNED_NODE_TYPES_IN_JIT[node_type]
            lineno = getattr(node, "lineno", "?")
            issues.append(f"line {lineno}: banned construct '{label}' in '{func.name}'")

        if isinstance(node, ast.FunctionDef) and node is not func:
            issues.append(
                f"line {node.lineno}: nested function '{node.name}' in '{func.name}'"
            )

        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Name) and callee.id in BANNED_BUILTINS_IN_JIT:
                issues.append(
                    f"line {node.lineno}: banned builtin '{callee.id}()' in '{func.name}'"
                )

        if isinstance(node, ast.Lambda):
            issues.append(
                f"line {node.lineno}: lambda in '{func.name}'"
            )

        if isinstance(node, ast.With):
            issues.append(
                f"line {node.lineno}: with statement in '{func.name}'"
            )

        if isinstance(node, (ast.Break, ast.Continue)):
            kw = "break" if isinstance(node, ast.Break) else "continue"
            issues.append(
                f"line {node.lineno}: '{kw}' in '{func.name}'"
            )

    return issues


def _check_loop_forms(func: ast.FunctionDef) -> list[str]:
    """Only ``for ... in range(...)`` and ``for ... in asc.static_range(...)`` allowed."""
    issues = []
    for node in ast.walk(func):
        if isinstance(node, ast.While):
            issues.append(
                f"line {node.lineno}: while loop in '{func.name}' (only for/range allowed)"
            )
        if isinstance(node, ast.For):
            it = node.iter
            ok = False
            if isinstance(it, ast.Call):
                callee = it.func
                if isinstance(callee, ast.Name) and callee.id == "range":
                    ok = True
                elif isinstance(callee, ast.Attribute) and callee.attr in (
                    "range",
                    "static_range",
                ):
                    ok = True
            if not ok:
                issues.append(
                    f"line {node.lineno}: non-range for loop in '{func.name}'"
                )
    return issues


def _source_has_call(source: str, name: str) -> bool:
    """Check if the entire source file contains a call whose name ends with *name*."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Name) and callee.id == name:
                return True
            if isinstance(callee, ast.Attribute) and callee.attr == name:
                return True
    return False


def verify(path: str) -> list[CheckResult]:
    source = Path(path).read_text()
    tree = ast.parse(source)
    results: list[CheckResult] = []

    jit_funcs = _find_jit_functions(tree)
    results.append(CheckResult(
        "jit_decorator",
        len(jit_funcs) > 0,
        f"Found {len(jit_funcs)} @asc.jit function(s)" if jit_funcs else "No @asc.jit function found",
    ))

    if not jit_funcs:
        results.extend([
            CheckResult("no_return_value", False, "Skipped (no JIT function)"),
            CheckResult("no_banned_constructs", False, "Skipped (no JIT function)"),
            CheckResult("loop_forms", False, "Skipped (no JIT function)"),
        ])
    else:
        ret_issues: list[str] = []
        ban_issues: list[str] = []
        loop_issues: list[str] = []
        for func in jit_funcs:
            ret_issues.extend(_check_no_return_value(func))
            ban_issues.extend(_check_banned_constructs(func))
            loop_issues.extend(_check_loop_forms(func))

        results.append(CheckResult(
            "no_return_value",
            len(ret_issues) == 0,
            "; ".join(ret_issues) if ret_issues else "OK",
        ))
        results.append(CheckResult(
            "no_banned_constructs",
            len(ban_issues) == 0,
            "; ".join(ban_issues) if ban_issues else "OK",
        ))
        results.append(CheckResult(
            "loop_forms",
            len(loop_issues) == 0,
            "; ".join(loop_issues) if loop_issues else "OK",
        ))

    has_sync = _source_has_call(source, "set_flag") and _source_has_call(source, "wait_flag")
    results.append(CheckResult("sync_calls", has_sync,
                               "set_flag + wait_flag present" if has_sync else "Missing set_flag/wait_flag"))

    has_dc = _source_has_call(source, "data_copy")
    results.append(CheckResult("data_copy", has_dc,
                               "data_copy present" if has_dc else "Missing data_copy"))

    has_verify = (
        _source_has_call(source, "allclose")
        or "allclose" in source
    )
    results.append(CheckResult("verification", has_verify,
                               "allclose verification present" if has_verify else "Missing torch/numpy allclose"))

    return results


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel.py> [--json]", file=sys.stderr)
        sys.exit(2)

    path = sys.argv[1]
    use_json = "--json" in sys.argv

    try:
        results = verify(path)
    except SyntaxError as exc:
        print(f"FAIL: SyntaxError parsing {path}: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"FAIL: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    if use_json:
        data = {
            "file": path,
            "passed": all(r.passed for r in results),
            "checks": {r.name: {"passed": r.passed, "detail": r.detail} for r in results},
        }
        print(json.dumps(data, indent=2))
    else:
        all_ok = True
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}: {r.detail}")
            if not r.passed:
                all_ok = False
        if all_ok:
            print("\nAll static checks passed.")
        else:
            print("\nSome checks FAILED.")

    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from contextlib import ExitStack
import importlib
import importlib.util
import os
import sys
import types
from unittest.mock import patch

from asc.lib.profiling import Profiler, task_time_median
from asc.runtime import config
import pytest

# --- Local addition, not upstream ------------------------------------------
# The vendored test files are kept byte-identical to upstream so that
# scripts/sync-target-suite.sh can compare them by hash. They all do a bare
# `import asctile`, which is how a legacy source build exposes the module. The
# published pyasc-dev wheel exposes it as asc.experimental.asctile instead, so
# without this alias the entire suite fails to import against a wheel install.
#
# Registering the alias here rather than editing 18 test files keeps the
# snapshot diffable against upstream. conftest.py is imported before any test
# module, so the alias is in place by the time they run.
#
# The alias is installed unconditionally when asc.experimental.asctile exists,
# and deliberately NOT guarded on `"asctile" not in sys.modules`. A bare
# `import asctile` can succeed and still be useless: a half-built or stale
# editable checkout leaves an `asctile/` directory with no __init__.py, which
# Python happily imports as an empty namespace package. The tests then fail
# with `AttributeError: module 'asctile' has no attribute 'jit'`, which reads
# like a broken test suite rather than a broken environment. Preferring the
# installed runtime makes "the runtime you installed is the runtime you test"
# true by construction.
try:
    from asc.experimental import asctile as _asctile
except ImportError:  # pragma: no cover - legacy source build exposes it top-level
    _asctile = None
else:
    sys.modules["asctile"] = _asctile

if _asctile is None or not hasattr(_asctile, "jit"):
    # Neither path produced a usable module. Say so here, once, instead of
    # letting all 24 test modules die at import with
    # `AttributeError: module 'asctile' has no attribute 'jit'`, which reads
    # like a broken suite rather than a broken environment.
    try:
        import asctile as _fallback
    except ImportError:
        _found = "nothing importable as `asctile`"
    else:
        _found = f"`asctile` from {_fallback.__file__!r}"
        if hasattr(_fallback, "jit"):
            _asctile = _fallback  # legacy source build; nothing more to do
    if _asctile is None or not hasattr(_asctile, "jit"):
        try:
            import asc as _asc
            _asc_where = _asc.__file__
        except ImportError:
            _asc_where = "not installed"
        raise RuntimeError(
            "No usable asctile runtime on this interpreter.\n"
            f"  python           {sys.executable}\n"
            f"  asc              {_asc_where}\n"
            f"  asctile          {_found}\n"
            "A directory named `asctile/` with no __init__.py imports as an "
            "empty namespace package, so `import asctile` can succeed and "
            "still be useless -- that is usually a stale editable checkout on "
            "sys.path shadowing the real runtime.\n"
            "Install the published wheel into a clean venv:\n"
            "  python3 -m venv .venv && .venv/bin/pip install --pre pyasc-dev\n"
            "See reference/README.md."
        )
# ---------------------------------------------------------------------------


class StubProfiler:

    def profile(self, *args, **kwargs):
        return self

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def pytest_addoption(parser: pytest.Parser):
    # Local deviation: each option is registered defensively.
    #
    # The published pyasc-dev wheel declares no entry points, so it ships no
    # pytest plugin and these options exist only because this file adds them.
    # A previous revision of this conftest assumed the opposite -- that the
    # runtime image provided --backend and --platform -- and skipped them. Against
    # a wheel install that turns `pytest --backend Model` into "unrecognized
    # arguments", and every fixture silently falls back to its default.
    #
    # Registering defensively is correct either way: it works now, and it keeps
    # working if a future runtime does start shipping a plugin.
    def add(*args, **kwargs):
        try:
            parser.addoption(*args, **kwargs)
        except ValueError:
            pass  # already registered by a runtime-provided plugin

    add("--backend", type=config.Backend, default=config.Backend.Model, help="Runtime backend")
    add("--platform", type=config.Platform, default=config.Platform.Ascend950PR_9599,
        help="Runtime platform")
    add("--device", type=int, default=None, help="Device ID")
    add("--profile", action="store_true", help="Enable NPU profiling (if available)")
    add("--profile-path", type=str, default=None, help="Directory to save profiling results")
    add("--runs", type=int, default=1, help="Number of kernel launches")
    add("--compile-only", action="store_true", help="Stop after the kernel compilation (do not launch)")
    add("--torch-seed", type=int, default=0, help="Define the seed value for torch.manual_seed(...)")


def pytest_configure(config):
    config.profiling_results = []
    config.addinivalue_line(
        "markers",
        "asctile_xfail(reason, compile_ok, when): expected failure; the apply_xfail fixture attaches xfail at run time",
    )


def pytest_make_parametrize_id(val):
    if callable(val):
        return val.__name__
    return str(val)


def pytest_terminal_summary(terminalreporter, config):
    if not config.profiling_results:
        return
    terminalreporter.write_sep("=", "Profiling results")
    for entry in config.profiling_results:
        terminalreporter.write_line(f"{entry['test']}: {entry['duration']} μs")


@pytest.fixture(autouse=True)
def apply_xfail(request: pytest.FixtureRequest, compile_only):
    mark = request.node.get_closest_marker("asctile_xfail")
    if mark is None:
        return
    reason = mark.kwargs.get("reason", "")
    compile_ok = mark.kwargs.get("compile_ok", False)
    when = mark.kwargs.get("when")
    if compile_ok and compile_only:
        return
    if when is not None:
        callspec = getattr(request.node, "callspec", None)
        ctx = types.SimpleNamespace(
            params=callspec.params if callspec else {},
            config=request.config,
            node=request.node,
        )
        if not when(ctx):
            return
    request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))


@pytest.fixture
def backend(request: pytest.FixtureRequest):
    return request.config.getoption("--backend")


@pytest.fixture
def platform(request: pytest.FixtureRequest):
    return request.config.getoption("--platform")


@pytest.fixture
def device_id(request: pytest.FixtureRequest):
    return request.config.getoption("--device", default=None)


@pytest.fixture(autouse=True)
def set_platform(request: pytest.FixtureRequest, backend: config.Backend, platform: config.Platform, device_id: int):
    if not request.node.get_closest_marker("no_set_platform"):
        config.set_platform(backend, platform, device_id, check=False)
    yield


@pytest.fixture
def require_c310(platform: config.Platform):

    def require_c310_impl():
        if config.platform_to_arch(platform) != config.CompilationArch.C310:
            pytest.skip(f"{platform.value} platform is not supported")

    return require_c310_impl


@pytest.fixture
def profiler(request, tmp_path_factory, backend):
    if backend != config.Backend.NPU or not request.config.getoption("--profile"):
        yield StubProfiler()
        return
    profiling_path = request.config.getoption("--profile-path")
    if profiling_path is not None:
        safe_name = request.node.nodeid.replace(":", "_").replace("/", "_").replace(" ", "")
        result_path = os.path.join(profiling_path, safe_name)
        os.makedirs(result_path, exist_ok=True)
    else:
        result_path = str(tmp_path_factory.mktemp("profiler"))
    profiler = Profiler(result_path)
    yield profiler
    request.config.profiling_results.append({
        "test": request.node.nodeid,
        "duration": task_time_median(profiler.last_result.tasks, skip=1),
    })


@pytest.fixture
def runs(request: pytest.FixtureRequest):
    return request.config.getoption("--runs")


@pytest.fixture(scope="session", autouse=True)
def compile_only(request: pytest.FixtureRequest):
    if not request.config.getoption("--compile-only"):
        yield False
        return
    os.environ["PYASC_DRY_RUN"] = "1"
    with ExitStack() as stack:
        try:
            importlib.import_module("numpy")
            stack.enter_context(patch("numpy.testing.assert_allclose"))
        except ModuleNotFoundError:
            pass
        try:
            importlib.import_module("torch")
            stack.enter_context(patch("torch.testing.assert_close"))
            for fn in ("torch.allclose", "torch.equal"):
                stack.enter_context(patch(fn)).return_value = True
        except ModuleNotFoundError:
            pass
        yield True


@pytest.fixture
def torch_seed(request: pytest.FixtureRequest):
    return request.config.getoption("--torch-seed")


@pytest.fixture(autouse=True)
def configure_torch(backend: config.Backend, device_id: int, torch_seed: int):
    try:
        import torch
        if backend == config.Backend.NPU and importlib.util.find_spec("torch_npu") is not None:
            device = "npu" if device_id is None else f"npu:{device_id}"
        else:
            device = "cpu"
        with torch.device(device):
            torch.manual_seed(torch_seed)
            yield
    except ModuleNotFoundError:
        yield

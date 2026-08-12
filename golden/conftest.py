# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
from unittest.mock import patch

from asc.lib.profiling import Profiler, task_time_median
from asc.runtime import config
import pytest


class StubProfiler:

    def profile(self, *args, **kwargs):
        return self

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


# NOTE: --backend + --platform are provided by the image's pyasc pytest plugin.
# We register ONLY the additional options the image's plugin does NOT provide.
def pytest_addoption(parser: pytest.Parser):
    for opt in ("--device", "--profile", "--profile-path", "--runs",
                "--compile-only", "--torch-seed"):
        try:
            if opt == "--device":
                parser.addoption(opt, type=int, default=None, help="Device ID")
            elif opt == "--profile":
                parser.addoption(opt, action="store_true", default=False, help="Enable NPU profiling")
            elif opt == "--profile-path":
                parser.addoption(opt, default=None, help="Profiling output dir")
            elif opt == "--runs":
                parser.addoption(opt, type=int, default=1, help="Number of kernel launches")
            elif opt == "--compile-only":
                parser.addoption(opt, action="store_true", default=False, help="Compile only, no launch")
            elif opt == "--torch-seed":
                parser.addoption(opt, type=int, default=0, help="torch.manual_seed value")
        except ValueError:
            pass  # already registered by the image's plugin


def pytest_configure(config):
    config.profiling_results = []


@pytest.fixture
def backend(request: pytest.FixtureRequest):
    return request.config.getoption("--backend", default=config.Backend.Model)


@pytest.fixture
def platform(request: pytest.FixtureRequest):
    return request.config.getoption("--platform", default=config.Platform.Ascend950PR_9599)


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
    if backend != config.Backend.NPU or not request.config.getoption("--profile", default=False):
        yield StubProfiler()
        return
    profiling_path = request.config.getoption("--profile-path", default=None)
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
    return request.config.getoption("--runs", default=1)


@pytest.fixture(scope="session", autouse=True)
def compile_only(request: pytest.FixtureRequest):
    if not request.config.getoption("--compile-only", default=False):
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
    return request.config.getoption("--torch-seed", default=0)


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

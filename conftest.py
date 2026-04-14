"""Shared pytest fixtures for pyasc kernel tests.

Usage:
    pytest golden/kernels/abs_f16.py --backend Model --platform Ascend910B1
"""

import asc.runtime.config as config
import pytest


def pytest_addoption(parser):
    parser.addoption("--backend", default="Model", help="Backend: Model or NPU")
    parser.addoption("--platform", default=None, help="Platform SoC, e.g. Ascend910B1")


@pytest.fixture
def backend(request):
    name = request.config.getoption("--backend")
    if name not in config.Backend.__members__:
        pytest.skip(f"Unsupported backend: {name}")
    return config.Backend(name)


@pytest.fixture
def platform(request):
    name = request.config.getoption("--platform")
    if name is None:
        return None
    platform_values = [p.value for p in config.Platform]
    if name not in platform_values:
        pytest.skip(f"Unsupported platform: {name}")
    return config.Platform(name)

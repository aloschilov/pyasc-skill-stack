"""One-time pyasc NPU platform initialisation shared by all operator modules.

The cann-bench harness imports ``cann_bench`` inside a torch_npu process and
calls the operator wrappers with NPU-resident tensors. pyasc must be switched
to its NPU backend once, before the first kernel launch. The platform can be
overridden with ``CANN_BENCH_PYASC_PLATFORM`` (SoC version string).
"""

import os
import threading

_lock = threading.Lock()
_initialised = False


def ensure_npu_platform() -> None:
    """Switch pyasc to the NPU backend exactly once (thread-safe)."""
    global _initialised
    if _initialised:
        return
    with _lock:
        if _initialised:
            return
        import asc.runtime.config as config

        override = os.environ.get("CANN_BENCH_PYASC_PLATFORM")
        if override:
            try:
                platform = config.Platform(override)
            except ValueError as exc:
                raise RuntimeError(
                    f"unsupported pyasc platform override {override!r}; "
                    f"available: {[p.value for p in config.Platform]}"
                ) from exc
            config.set_platform(config.Backend.NPU, platform)
        else:
            # CANNBench's 950PR runner pool contains multiple SoC revisions
            # (for example 957c and 9599). Let pyasc query the active device so
            # the submitted wheel remains portable across those runners.
            config.set_platform(config.Backend.NPU)
        _initialised = True

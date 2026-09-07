"""Host-only platform setup; no compiler changes or numerical helpers."""
import threading
import asctile

_lock = threading.Lock()
_initialised = False


def ensure_npu_platform():
    global _initialised
    if not _initialised:
        with _lock:
            if not _initialised:
                asctile.set_platform(asctile.Backend.NPU)
                _initialised = True

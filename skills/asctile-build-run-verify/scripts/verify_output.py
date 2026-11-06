#!/usr/bin/env python3
"""Verify pyasc kernel output correctness."""

import argparse
import importlib.util
import sys
import os


def load_kernel_module(kernel_path):
    """Dynamically load a kernel module from file path."""
    spec = importlib.util.spec_from_file_location("kernel_module", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Verify pyasc kernel output")
    parser.add_argument("kernel_path", help="Path to kernel.py")
    parser.add_argument("--backend", default="Model", help="Backend: Model or NPU")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    args = parser.parse_args()

    if not os.path.exists(args.kernel_path):
        print(f"[ERROR] File not found: {args.kernel_path}")
        sys.exit(1)

    print(f"[INFO] Verifying kernel: {args.kernel_path}")
    print(f"[INFO] Backend: {args.backend}, atol: {args.atol}")

    try:
        import asc.runtime.config as config

        backend = config.Backend(args.backend)
        config.set_platform(backend, None)

        module = load_kernel_module(args.kernel_path)

        if hasattr(module, "run_kernel"):
            module.run_kernel(backend, None)
            print("[PASS] Kernel verification passed")
        else:
            print("[WARN] No run_kernel function found. Running module directly.")
            # Module execution at import already runs if __name__ == "__main__" won't trigger
            print("[INFO] Run the kernel directly: python " + args.kernel_path)

    except AssertionError as e:
        print(f"[FAIL] Verification failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()

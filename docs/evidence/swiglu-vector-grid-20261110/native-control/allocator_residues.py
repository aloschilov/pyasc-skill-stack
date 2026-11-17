"""Measure restored input/output address residues while a separate allocation is live."""
import argparse
import importlib.util
import json
from pathlib import Path


def run(worker, package, task, case, prefix_bytes, output):
    import torch
    import torch_npu  # Registers the NPU backend on the qualified wheel.

    if prefix_bytes <= 0 or prefix_bytes % 512:
        raise ValueError("the allocator probe requires a positive multiple of 512 bytes")
    spec = importlib.util.spec_from_file_location("restored_worker", worker)
    restored = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restored)
    output.mkdir()
    # Keep the storage live across the whole ordinary evaluation. It is never
    # read, initialized, or supplied to the kernel as an operand.
    reservation = torch.empty(prefix_bytes, dtype=torch.uint8, device="npu:0")
    restored.run(package, task, case, output, record_alignment=True)
    observations = json.loads((output / "alignment.json").read_text())
    period = observations["period_bytes"]
    record = {"bytes": reservation.numel(), "residue": reservation.data_ptr() % period,
              "period_bytes": period}
    (output / "reservation.json").write_text(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("worker", "package", "task"):
        parser.add_argument(key, type=Path)
    parser.add_argument("case", type=int)
    parser.add_argument("prefix_bytes", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.worker.resolve(), args.package.resolve(), args.task.resolve(),
        args.case, args.prefix_bytes, args.output.resolve())

#!/usr/bin/env python3
"""Generate golden input/expected-output pairs for pyasc kernel verification.

Produces numpy arrays for given shapes and dtypes, suitable for comparison
against agent-generated kernels.

Usage:
    python gen_golden.py --op abs --dtype float16 --shapes 1,128 4,2048 32,4096 [--output-dir golden_data/]
    python gen_golden.py --op add --dtype float32 --shapes 1024
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


OPS = {
    "abs": lambda x: np.abs(x),
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
}

DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "int32": np.int32,
}


def parse_shape(s: str) -> tuple:
    return tuple(int(x) for x in s.split(","))


def generate(op: str, dtype_str: str, shapes: list[str], output_dir: str, seed: int = 42):
    np.random.seed(seed)
    dtype = DTYPES[dtype_str]
    op_fn = OPS[op]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = {"op": op, "dtype": dtype_str, "shapes": [], "files": []}
    n_inputs = 2 if op in ("add", "sub", "mul") else 1

    for shape_str in shapes:
        shape = parse_shape(shape_str)
        tag = f"{op}_{dtype_str}_{'x'.join(str(s) for s in shape)}"

        inputs = []
        for i in range(n_inputs):
            x = np.random.randn(*shape).astype(dtype)
            fname = f"{tag}_input{i}.npy"
            np.save(out_path / fname, x)
            inputs.append(x)
            manifest["files"].append(fname)

        expected = op_fn(*inputs)
        fname = f"{tag}_expected.npy"
        np.save(out_path / fname, expected)
        manifest["files"].append(fname)
        manifest["shapes"].append({"shape": list(shape), "tag": tag})

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate golden test data")
    parser.add_argument("--op", required=True, choices=list(OPS.keys()))
    parser.add_argument("--dtype", default="float16", choices=list(DTYPES.keys()))
    parser.add_argument("--shapes", nargs="+", required=True, help="Shapes as comma-separated dims")
    parser.add_argument("--output-dir", default="golden_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest = generate(args.op, args.dtype, args.shapes, args.output_dir, args.seed)

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"Generated {len(manifest['files'])} files in {args.output_dir}/")
        for entry in manifest["shapes"]:
            print(f"  Shape {entry['shape']}: {entry['tag']}")


if __name__ == "__main__":
    main()

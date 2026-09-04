#!/usr/bin/env python3
"""Generate a 42-column TTK manifest for all CANNBench GeLU cases."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CASES_PATH = REPO / "integrations/cannbench/tasks/gelu/cases.yaml"
REFERENCE_PATH = Path(
    "/home/aloschilov/Documents/Codex/2026-09-04/new-chat/outputs/"
    "ttk-csv-recovered/concatd_selected_representative_adjusted.csv"
)
OUTPUT_PATH = HERE / "gelu_cannbench_ttk.csv"


def py(value: object) -> str:
    """Render structured CSV cells as Python literals."""
    return repr(value)


def numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def bucket(size: int) -> str:
    if size < 100_000:
        return "small"
    if size < 5_000_000:
        return "medium"
    return "big"


def literal_endpoint(value: object) -> object:
    """Keep special floating-point endpoints literal-evaluable and TTK-readable."""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def build_row(case: dict, index: int) -> dict[str, str]:
    shape = tuple(case["input_shape"][0])
    dtype = case["dtype"][0]
    mode = case["attrs"]["approximate"]
    params = {"approximate": mode}
    dynamic_shape = tuple(-1 for _ in shape)
    dynamic_range = tuple((1, None) for _ in shape)
    data_range = tuple(literal_endpoint(v) for v in case["value_range"])

    return {
        "network_name": "UNKNOWN",
        "testcase_name": f"gelu_v2_{index:05d}",
        "op_name": "gelu_v2",
        "stc_input_dtypes": py((dtype,)),
        "stc_ori_inputs": py((shape,)),
        "stc_ori_outputs": py((shape,)),
        "stc_input_ori_formats": py(("ND",)),
        "output_ori_formats": py(("ND",)),
        "other_compilation_params": py(params),
        "other_runtime_params": py(params),
        "stc_inputs": py((shape,)),
        "output_dtypes": py((dtype,)),
        "stc_outputs": py((shape,)),
        "stc_input_formats": py(("ND",)),
        "output_formats": py(("ND",)),
        "dyn_inputs": py((dynamic_shape,)),
        "dyn_input_dtypes": py((dtype,)),
        "dyn_outputs": py((dynamic_shape,)),
        "dyn_ori_inputs": py((dynamic_shape,)),
        "dyn_ori_outputs": py((dynamic_shape,)),
        "dyn_input_formats": py(("ND",)),
        "dyn_input_ori_formats": py(("ND",)),
        "dyn_input_ranges": py((dynamic_range,)),
        "dyn_output_ranges": py((dynamic_range,)),
        "dyn_input_as_list_distribution": py(()),
        "stc_input_as_list_distribution": py(()),
        "input_as_variable": py(()),
        "stc_op_name": "gelu_v2",
        "const_input_indexes": py(()),
        "precision_tolerances": "",
        "input_data_ranges": py((data_range,)),
        "strict_precision_mode": "1",
        "absolute_precision": "",
        "shape_check": "1",
        "output_inplace_indexes": py(()),
        "random_buff": "",
        "is_enabled": "1",
        "bucket": bucket(numel(shape)),
        "arity": "1",
        "position": "",
        "tiling_key": "7",
        "kernel": "gelu",
    }


def main() -> None:
    with REFERENCE_PATH.open(newline="", encoding="utf-8") as source:
        header = next(csv.reader(source))
    if len(header) != 42:
        raise ValueError(f"reference has {len(header)} columns, expected 42")

    cases = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    rows = [build_row(case, index) for index, case in enumerate(cases, 1)]
    for row in rows:
        if set(row) != set(header):
            raise ValueError(
                f"column mismatch: missing={set(header) - set(row)}, "
                f"extra={set(row) - set(header)}"
            )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT_PATH} ({len(rows)} rows, {len(header)} columns)")


if __name__ == "__main__":
    main()

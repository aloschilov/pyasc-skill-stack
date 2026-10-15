#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Accuracy Verification Tool

Responsibilities:
1. Provide tensor comparison verification function
2. Adopt ecological operator open source accuracy standard (MERE/MARE)
3. Passing conditions: MERE < threshold, MARE < 10 * threshold

Error index (standard formula):
- MERE (mean relative error) = avg(|actual - golden| / (|golden| + 1e-7))
- MARE (maximum relative error) = max(|actual - golden| / (|golden| + 1e-7))

Special scene processing:
- Small value range processing: when |golden| < small_value_threshold, use the ErrorCount ratio standard
- Cancellation processing: When output ≈ 0 and golden is near the accuracy boundary, the CPU same-accuracy comparison standard is used

Theoretical basis:
- IEEE 754 floating-point standard: the number of digits of precision determines the range of significant digits
- Kahan's catastrophic cancellation theory: subtraction of close to large numbers leads to loss of accuracy"""

import math
from typing import Union, Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field

import torch


@dataclass
class SingleOutputResult:
    """Comparison results of a single output (for independent judgment of multi-output operators)"""
    index: int                      # Output index
    name: str = ""                  # Output name (optional)
    dtype: str = ""                 # data type
    dtype_category: str = ""        # 'float' or 'int'
    passed: bool = True
    threshold: float = 0.0
    # Floating point type indicator
    mere: float = 0.0               # average relative error
    mare: float = 0.0               # maximum relative error
    max_diff: float = 0.0
    mean_diff: float = 0.0
    # Integer type indicator
    mismatch_count: int = 0         # Number of unmatched elements
    total_count: int = 0            # total number of elements
    max_abs_diff: int = 0           # maximum absolute difference
    # Small value range/cancellation indicator
    small_value_error_count: int = 0
    small_value_cpu_error_count: int = 0
    small_value_total_count: int = 0
    cancel_error_count: int = 0
    cancel_cpu_error_count: int = 0
    cancel_total_count: int = 0
    # error message
    error_msg: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'name': self.name,
            'dtype': self.dtype,
            'dtype_category': self.dtype_category,
            'passed': self.passed,
            'threshold': self.threshold,
            'mere': self.mere,
            'mare': self.mare,
            'mismatch_count': self.mismatch_count,
            'total_count': self.total_count,
            'max_abs_diff': self.max_abs_diff,
            'error_msg': self.error_msg,
        }

    def format_summary(self) -> str:
        """Format single output decision summary (for logging)"""
        dtype_str = f"{self.dtype}[{self.name or self.index}]"

        if self.dtype_category == 'int':
            if self.passed:
                return f"{dtype_str}: ✅ (exact match)"
            else:
                ratio = self.mismatch_count / max(self.total_count, 1)
                return f"{dtype_str}: ❌ mismatch={self.mismatch_count}/{self.total_count} ({ratio:.2%}), max_diff={self.max_abs_diff}"
        else:  # float
            if self.passed:
                return f"{dtype_str}: ✅ MERE={self.mere:.6f}, MARE={self.mare:.6f}"
            else:
                mare_threshold = 10 * self.threshold
                return f"{dtype_str}: ❌ MERE={self.mere:.6f}, MARE={self.mare:.6f} (threshold={self.threshold:.6f}, mare_threshold={mare_threshold:.6f})"


# Accuracy threshold table (using ecological operator open source accuracy standard)
PRECISION_THRESHOLDS: Dict[str, float] = {
    'float16': 2**-10,      # ≈ 0.000976
    'bfloat16': 2**-7,      # ≈ 0.007812
    'float32': 2**-13,      # ≈ 0.000122
    'float64': 2**-13,      # Use float32 threshold
    'hifloat32': 2**-11,    # ≈ 0.000488
    'float8_e4m3': 2**-3,   # ≈ 0.125
    'float8_e5m2': 2**-2,   # ≈ 0.25
    'int8': 0,              # exactly equal
    'int16': 0,
    'int32': 0,
    'int64': 0,
    'uint8': 0,
    'uint16': 0,
    'uint32': 0,
    'uint64': 0,
}

# Small value range threshold table (from docs/kernel_bench_design_v1.0.md)
# When |golden| < small_value_threshold, the small value range criterion is used
SMALL_VALUE_THRESHOLDS: Dict[str, float] = {
    'float16': 2**-11,      # ≈ 4.88e-4
    'bfloat16': 2**-8,      # ≈ 3.91e-3
    'float32': 2**-14,      # ≈ 6.10e-5
    'float64': 2**-14,      # Use float32 threshold
    'hifloat32': 2**-12,    # ≈ 2.44e-4
    'float8_e4m3': 2**-4,   # ≈ 0.0625
    'float8_e5m2': 2**-3,   # ≈ 0.125
}

# Small range error threshold table (from docs/kernel_bench_design_v1.0.md)
# Counted in ErrorCount when |golden| < small_value_threshold and |actual - golden| > small_value_error
SMALL_VALUE_ERROR_THRESHOLDS: Dict[str, float] = {
    'float16': 2**-16,      # ≈ 1.53e-5
    'bfloat16': 2**-16,     # ≈ 1.53e-5
    'float32': 2**-30,      # ≈ 9.31e-10
    'float64': 2**-30,      # Use float32 threshold
    'hifloat32': 2**-28,    # ≈ 3.73e-9
    'float8_e4m3': 2**-6,   # ≈ 1.56e-2
    'float8_e5m2': 2**-5,   # ≈ 3.12e-2
}

# ============================================================================
# Destructive precision boundary threshold table (based on IEEE 754 digits of precision theory)
# ============================================================================
#
# Theoretical basis:
# 1. IEEE 754 standard: The number of mantissa digits of different dtypes determines the range of valid digits
#    - FP32: 23-digit mantissa, relative accuracy ~2^-23 ≈ 10^-7, approximately 7 significant digits
#    - FP16: 10 digits of mantissa, relative accuracy ~2^-10 ≈ 10^-3, approximately 3 significant digits
#    - BF16: 7-digit mantissa, relative accuracy ~2^-7 ≈ 10^-2, approximately 2 significant digits
#
# 2. Kahan's catastrophic cancellation theory:
#    When two close large numbers are subtracted, the number of significant digits in the result is drastically lost.
#    For example: subtracting two ~10^4 numbers in FP32 results in ~10^-3, but the accuracy is only enough to represent 7 digits.
#    The result loses precision relative to the original operand and may be output as 0.
#
# 3. Cancellation judgment conditions:
#    - output ≈ 0: precision is lost due to cancellation, the result is close to zero
#    - golden near the precision boundary: small non-zero values, but smaller than the range that the dtype can reliably represent
#    - Not within the small value range (excluding minimum values)
#
# Threshold selection principle:
#    cancel_boundary should cover the range where cancellation may occur due to loss of digits of precision.
#    For FP32, when the operand size is ~10^4, the result ~10^-3 may be destructively lost.
#    Set cancel_boundary = 2^-8 ≈ 0.004 to cover common cancellation scenarios.
#
CANCEL_BOUNDARY_THRESHOLDS: Dict[str, float] = {
    # FP32: Accuracy ~7 digits, setting 2^-8 ≈ 0.004
    # When golden < 0.004 and output ≈ 0, it may be caused by FP32 cancellation
    'float32': 2**-8,       # ≈ 3.91e-3 ≈ 0.004
    'float64': 2**-8,       # Use float32 threshold

    # FP16: Accuracy ~3 digits, setting 2^-5 ≈ 0.031
    # When golden < 0.031 and output ≈ 0, it may be caused by FP16 cancellation
    'float16': 2**-5,       # ≈ 3.12e-2 ≈ 0.031

    # BF16: Accuracy ~2 digits, setting 2^-3 ≈ 0.125
    # When golden < 0.125 and output ≈ 0, it may be caused by BF16 cancellation
    'bfloat16': 2**-3,      # ≈ 1.25e-1 ≈ 0.125

    'hifloat32': 2**-8,     # ≈ 3.91e-3
    'float8_e4m3': 2**-1,   # ≈ 0.5
    'float8_e5m2': 2**-0,   # ≈ 1.0
}

# Cancellation output zero value judgment threshold
# When |output| < cancel_zero_threshold, it is determined that output ≈ 0 (precision is lost due to cancellation)
CANCEL_ZERO_THRESHOLDS: Dict[str, float] = {
    # Consistent with cancel_boundary, ensure that when output is close to zero, it is determined to be cancelled.
    'float32': 2**-8,       # ≈ 0.004
    'float64': 2**-8,
    'float16': 2**-5,       # ≈ 0.031
    'bfloat16': 2**-3,      # ≈ 0.125
    'hifloat32': 2**-8,
    'float8_e4m3': 2**-1,
    'float8_e5m2': 2**-0,
}


@dataclass
class CompareResult:
    """Comparison results (supports multi-output operators)"""
    passed: bool
    dtype: str
    threshold: float  # Accuracy threshold (aggregated value)
    mere: float = 0.0  # Average relative error (aggregated value)
    mare: float = 0.0  # Maximum relative error (aggregated value)
    max_diff: float = 0.0
    mean_diff: float = 0.0
    mismatch_count: int = 0
    total_count: int = 0
    mismatch_ratio: float = 0.0
    small_value_error_count: int = 0  # Small range NPU error count
    small_value_cpu_error_count: int = 0  # Small range CPU error count
    small_value_total_count: int = 0  # total count of small ranges
    cancel_error_count: int = 0  # Cancellation position NPU error count
    cancel_cpu_error_count: int = 0  # Cancellation position CPU error count
    cancel_total_count: int = 0  # Total count of cancellation positions
    error_msg: Optional[str] = None
    output_results: List[SingleOutputResult] = field(default_factory=list)  # Independent results for each output

    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'dtype': self.dtype,
            'threshold': self.threshold,
            'mere': self.mere,
            'mare': self.mare,
            'max_diff': self.max_diff,
            'mean_diff': self.mean_diff,
            'mismatch_count': self.mismatch_count,
            'total_count': self.total_count,
            'mismatch_ratio': self.mismatch_ratio,
            'small_value_error_count': self.small_value_error_count,
            'small_value_cpu_error_count': self.small_value_cpu_error_count,
            'small_value_total_count': self.small_value_total_count,
            'cancel_error_count': self.cancel_error_count,
            'cancel_cpu_error_count': self.cancel_cpu_error_count,
            'cancel_total_count': self.cancel_total_count,
            'error_msg': self.error_msg,
            'output_results': [r.to_dict() for r in self.output_results],
        }

    def format_all_outputs(self) -> str:
        """Format all output judgment results (for logging)"""
        lines = []
        for r in self.output_results:
            lines.append(f"  - {r.format_summary()}")
        return "\n".join(lines)


def get_threshold(dtype_str: str) -> float:
    """Get accuracy threshold"""
    dtype_lower = dtype_str.lower()
    if dtype_lower not in PRECISION_THRESHOLDS:
        # Use float32 threshold by default
        return PRECISION_THRESHOLDS['float32']
    return PRECISION_THRESHOLDS[dtype_lower]


def get_small_value_threshold(dtype_str: str) -> float:
    """Get the small value range threshold"""
    dtype_lower = dtype_str.lower()
    if dtype_lower not in SMALL_VALUE_THRESHOLDS:
        return SMALL_VALUE_THRESHOLDS['float32']
    return SMALL_VALUE_THRESHOLDS[dtype_lower]


def get_small_value_error(dtype_str: str) -> float:
    """Get the small range error threshold"""
    dtype_lower = dtype_str.lower()
    if dtype_lower not in SMALL_VALUE_ERROR_THRESHOLDS:
        return SMALL_VALUE_ERROR_THRESHOLDS['float32']
    return SMALL_VALUE_ERROR_THRESHOLDS[dtype_lower]


def get_cancel_boundary(dtype_str: str) -> float:
    """    Gets the destructive precision boundary threshold (based on IEEE 754 digits of precision theory)

    When |golden| < cancel_boundary and |output| ≈ 0, it is determined as a potential cancellation position.

    Theoretical basis:
    - IEEE 754 The number of mantissa digits determines the range of valid digits
    - Kahan's catastrophic cancellation theory: subtraction of close to large numbers leads to loss of accuracy
    """
    dtype_lower = dtype_str.lower()
    if dtype_lower not in CANCEL_BOUNDARY_THRESHOLDS:
        return CANCEL_BOUNDARY_THRESHOLDS['float32']
    return CANCEL_BOUNDARY_THRESHOLDS[dtype_lower]


def get_cancel_zero_threshold(dtype_str: str) -> float:
    """    Get the cancellation output zero value judgment threshold

    When |output| < cancel_zero_threshold, it is determined that output ≈ 0 (precision is lost due to cancellation).
    """
    dtype_lower = dtype_str.lower()
    if dtype_lower not in CANCEL_ZERO_THRESHOLDS:
        return CANCEL_ZERO_THRESHOLDS['float32']
    return CANCEL_ZERO_THRESHOLDS[dtype_lower]


def compare_tensors(
    output: Union[torch.Tensor, Tuple, List],
    golden: Union[torch.Tensor, Tuple, List],
    dtype: str = 'float32',
    threshold: Optional[float] = None,
    cpu_output: Optional[Union[torch.Tensor, Tuple, List]] = None,
    ignore_output_indices: Optional[List[int]] = None,
    custom_thresholds: Optional[Dict[str, float]] = None,
) -> CompareResult:
    """    Compare the output tensor with the Golden reference result (using MERE/MARE standard + small value range processing)

    Args:
        output: operator output (single tensor or multiple tensors)
        golden: Golden reference output (single tensor or multiple tensors), usually FP64 precision
        dtype: data type string
        threshold: precision threshold (optional, automatically selected based on dtype by default)
        cpu_output: CPU output at the same precision (optional, used for small range comparison)
                    If not provided, golden truncation to target precision is used as CPU output
        ignore_output_indices: List of output indices that need to be ignored for comparison
        custom_thresholds: Custom precision threshold table (priority higher than default threshold)

    Returns:
        CompareResult: Compare results

    Passing conditions:
        Normal value range: MERE < threshold and MARE < 10 * threshold
        Small value range: ErrorCount_npu / max(ErrorCount_cpu, 1) <= 2
    """
    # Get the threshold (custom threshold is preferred)
    if custom_thresholds is None:
        custom_thresholds = {}

    def _get_output_threshold(dtype_str: str) -> float:
        """Get the threshold of a single output (customized first, default second)"""
        dtype_lower = dtype_str.lower()
        if dtype_lower in custom_thresholds:
            return custom_thresholds[dtype_lower]
        return get_threshold(dtype_str)

    if threshold is None:
        threshold = _get_output_threshold(dtype)

    try:
        # Handle multiple output situations
        outputs = _normalize_outputs(output)
        goldens = _normalize_outputs(golden)
        cpu_outputs = _normalize_outputs(cpu_output) if cpu_output is not None else None

        if len(outputs) != len(goldens):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg=f"The output quantity does not match: output={len(outputs)}, golden={len(goldens)}"
            )

        if cpu_outputs is not None and len(cpu_outputs) != len(goldens):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg=f"CPU output quantity mismatch: cpu_output={len(cpu_outputs)}, golden={len(goldens)}"
            )

        # Compare one by one
        all_passed = True
        mere_sum = 0.0
        mare_max = 0.0
        max_diff = 0.0
        mean_diff = 0.0
        mismatch_count = 0
        total_count = 0
        small_value_error_count = 0
        small_value_cpu_error_count = 0
        small_value_total_count = 0
        cancel_error_count = 0
        cancel_cpu_error_count = 0
        cancel_total_count = 0

        # Record independent judgment results for each output
        single_output_results: List[SingleOutputResult] = []

        for i, (out_tensor, gold_tensor) in enumerate(zip(outputs, goldens)):
            # Skip output that does not require comparison
            if ignore_output_indices and i in ignore_output_indices:
                # Create a SingleOutputResult that skips markers
                single_output_results.append(SingleOutputResult(
                    index=i,
                    name="",  # The name is populated by the caller
                    dtype=str(out_tensor.dtype).replace('torch.', ''),
                    dtype_category='int' if out_tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8) else 'float',
                    passed=True,  # Skipped output is considered passed
                    error_msg="(skip comparison)"
                ))
                continue

            # Get thresholds based on the actual dtype of each output (custom thresholds take precedence)
            out_dtype_str = str(out_tensor.dtype).replace('torch.', '')
            out_threshold = _get_output_threshold(out_dtype_str)
            out_dtype_category = 'int' if out_tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8) else 'float'

            cpu_tensor = cpu_outputs[i] if cpu_outputs is not None else None
            result = _compare_single_tensor(out_tensor, gold_tensor, out_threshold, out_dtype_str, cpu_tensor)

            # Convert CompareResult to SingleOutputResult
            single_result = SingleOutputResult(
                index=i,
                name="",  # The name is populated by the caller
                dtype=out_dtype_str,
                dtype_category=out_dtype_category,
                passed=result.passed,
                threshold=out_threshold,
                mere=result.mere,
                mare=result.mare,
                max_diff=result.max_diff,
                mean_diff=result.mean_diff,
                mismatch_count=result.mismatch_count,
                total_count=result.total_count,
                max_abs_diff=int(result.max_diff) if out_dtype_category == 'int' else 0,
                small_value_error_count=result.small_value_error_count,
                small_value_cpu_error_count=result.small_value_cpu_error_count,
                small_value_total_count=result.small_value_total_count,
                cancel_error_count=result.cancel_error_count,
                cancel_cpu_error_count=result.cancel_cpu_error_count,
                cancel_total_count=result.cancel_total_count,
                error_msg=result.error_msg or "",
            )
            single_output_results.append(single_result)

            is_passed = result.passed
            all_passed = all_passed and is_passed
            mere_sum += result.mere * result.total_count
            mare_max = max(mare_max, result.mare)
            max_diff = max(max_diff, result.max_diff)
            mean_diff += result.mean_diff * result.total_count
            mismatch_count += result.mismatch_count
            total_count += result.total_count
            small_value_error_count += result.small_value_error_count
            small_value_cpu_error_count += result.small_value_cpu_error_count
            small_value_total_count += result.small_value_total_count
            cancel_error_count += result.cancel_error_count
            cancel_cpu_error_count += result.cancel_cpu_error_count
            cancel_total_count += result.cancel_total_count

        if total_count > 0:
            mere = mere_sum / total_count
            mean_diff = mean_diff / total_count
        else:
            mere = 0.0

        # Final pass conditions
        passed = all_passed

        # Determine the returned dtype and threshold
        # If there is a failure output, return the threshold information of the first failure output.
        # Otherwise, return the threshold information of the first output
        result_dtype = dtype
        result_threshold = threshold
        for sr in single_output_results:
            if not sr.passed and not sr.error_msg.startswith("(skip"):
                result_dtype = sr.dtype
                result_threshold = sr.threshold
                break

        return CompareResult(
            passed=passed,
            dtype=result_dtype,
            threshold=result_threshold,
            mere=mere,
            mare=mare_max,
            max_diff=max_diff,
            mean_diff=mean_diff,
            mismatch_count=mismatch_count,
            total_count=total_count,
            mismatch_ratio=mismatch_count / total_count if total_count > 0 else 0.0,
            small_value_error_count=small_value_error_count,
            small_value_cpu_error_count=small_value_cpu_error_count,
            small_value_total_count=small_value_total_count,
            cancel_error_count=cancel_error_count,
            cancel_cpu_error_count=cancel_cpu_error_count,
            cancel_total_count=cancel_total_count,
            output_results=single_output_results,  # New: Independent results for each output
        )

    except Exception as e:
        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            error_msg=str(e)
        )


def _normalize_outputs(output: Any) -> List[torch.Tensor]:
    """Normalize output to list of tensors"""
    if isinstance(output, torch.Tensor):
        return [output]
    elif isinstance(output, (tuple, list)):
        result = []
        for item in output:
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, (tuple, list)):
                for sub_item in item:
                    if isinstance(sub_item, torch.Tensor):
                        result.append(sub_item)
        return result
    else:
        return []


def _compare_single_tensor(
    output: torch.Tensor,
    golden: torch.Tensor,
    threshold: float,
    dtype: str,
    cpu_output: Optional[torch.Tensor] = None,
) -> CompareResult:
    """Compare individual tensors (calculate MERE/MARE + small range processing)

    Ecological operator accuracy standard:
    1. Mean relative error (MERE) = avg(|actual - golden| / (|golden| + 1e-7))
    2. Maximum relative error (MARE) = max(|actual - golden| / (|golden| + 1e-7))
    3. Passing criteria: MERE < threshold and MARE < 10 * threshold

    Small range handling (from docs/kernel_bench_design_v1.0.md):
    When |golden| < small_value_threshold, the small value range passing criterion is adopted:
    - ErrorCount = Count the number of positions that satisfy (|golden| < threshold and |actual - golden| > error)
    - Passing standard: ErrorCount_npu / max(ErrorCount_cpu, 1) <= 2

    Args:
        output: NPU/operator output tensor
        golden: Golden reference output (FP64 precision)
        threshold: precision threshold
        dtype: data type string
        cpu_output: CPU output at the same precision (optional)
                   If not provided, uses golden truncation to target precision as CPU output
    """
    # Golden runs on CPU while the AI op runs on NPU.
    # Normalize both sides to CPU so subtract/equal don't trip on mixed devices.
    if output.is_cuda or output.device.type == "npu":
        output = output.cpu()
    if golden.is_cuda or golden.device.type == "npu":
        golden = golden.cpu()

    # Check shape
    if output.shape != golden.shape:
        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            error_msg=f"Shape mismatch: output={output.shape}, golden={golden.shape}"
        )

    # For integer types, use absolute difference tolerance comparison
    if output.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                        torch.uint8):
        if torch.equal(output, golden):
            return CompareResult(
                passed=True,
                dtype=dtype,
                threshold=threshold,
                mere=0.0,
                mare=0.0,
                max_diff=0.0,
                mean_diff=0.0,
                mismatch_count=0,
                total_count=output.numel(),
                mismatch_ratio=0.0,
                cancel_error_count=0,
                cancel_cpu_error_count=0,
                cancel_total_count=0,
            )

        # When not exactly equal, check whether the difference is within the tolerance range
        diff = torch.abs(output.long() - golden.long())
        mismatch_mask = diff > max(threshold, 0)
        mismatch_count = int(mismatch_mask.sum())

        diff_float = diff.float()
        if mismatch_count == 0:
            return CompareResult(
                passed=True,
                dtype=dtype,
                threshold=threshold,
                mere=0.0,
                mare=0.0,
                max_diff=float(diff.max()) if diff.numel() > 0 else 0.0,
                mean_diff=float(diff_float.mean()) if diff.numel() > 0 else 0.0,
                mismatch_count=0,
                total_count=output.numel(),
                mismatch_ratio=0.0,
                cancel_error_count=0,
                cancel_cpu_error_count=0,
                cancel_total_count=0,
            )

        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            mere=0.0,
            mare=0.0,
            max_diff=float(diff.max()) if diff.numel() > 0 else 0.0,
            mean_diff=float(diff_float.mean()) if diff.numel() > 0 else 0.0,
            mismatch_count=mismatch_count,
            total_count=output.numel(),
            mismatch_ratio=mismatch_count / output.numel() if output.numel() > 0 else 0.0,
            cancel_error_count=0,
            cancel_cpu_error_count=0,
            cancel_total_count=0,
            error_msg=f"Integer type difference exceeds tolerance ({threshold}): {mismatch_count}/{output.numel()} elements do not match",
        )

    # Floating point type comparison
    # Golden actively truncates to output.dtype to simulate the precision limit of operator output.
    target_dtype = output.dtype
    golden_truncated = golden.to(target_dtype).double()
    output_fp64 = output.double()

    # Handling NaN
    if torch.any(torch.isnan(output_fp64)) or torch.any(torch.isnan(golden_truncated)):
        nan_out = torch.isnan(output_fp64)
        nan_gold = torch.isnan(golden_truncated)
        if not torch.all(nan_out == nan_gold):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg="NaN position mismatch"
            )

    # Handle Inf
    # Saturation boundary processing: When one side of inf has a finite value and the other side has a finite value, replace inf with the maximum finite value of dtype and continue the comparison.
    # This handles fp16 saturation boundary scenarios (NPU fp32→fp16 truncated to inf, golden fp64→fp16 not out of bounds).
    # MRE/MARE determines pass/fail after replacement - only the underlying value itself is close enough to pass.
    replaced_inf = False
    inf_match_mask = torch.zeros_like(output_fp64, dtype=torch.bool)  # initialize inf match mask
    if torch.any(torch.isinf(output_fp64)) or torch.any(torch.isinf(golden_truncated)):
        inf_out = torch.isinf(output_fp64)
        inf_gold = torch.isinf(golden_truncated)
        inf_mismatch = inf_out != inf_gold

        if torch.any(inf_mismatch):
            # Get the maximum finite value of the target dtype
            if target_dtype == torch.float16:
                max_finite = float(torch.finfo(torch.float16).max)  # 65504.0
            elif target_dtype == torch.bfloat16:
                max_finite = float(torch.finfo(torch.bfloat16).max)  # ~3.389e38
            elif target_dtype == torch.float32:
                max_finite = float(torch.finfo(torch.float32).max)
            else:
                max_finite = float(torch.finfo(target_dtype).max)

            mismatch_count = int(inf_mismatch.sum())
            print(f"[inf_sat] {mismatch_count} element(s) saturated to inf on one side only, "
                  f"replacing inf with {max_finite} and continuing comparison")

            # Replace inf with maximum finite value (sign preserved)
            if torch.any(inf_out & ~inf_gold):
                mask = inf_out & ~inf_gold
                output_fp64[mask] = torch.sign(output_fp64[mask]) * max_finite
                replaced_inf = True
            if torch.any(inf_gold & ~inf_out):
                mask = inf_gold & ~inf_out
                golden_truncated[mask] = torch.sign(golden_truncated[mask]) * max_finite
                replaced_inf = True

        # Positions where both sides are inf and have the same sign are directly regarded as matches and subsequent comparisons are excluded.
        both_inf = inf_out & inf_gold
        if torch.any(both_inf):
            if not torch.all(torch.sign(output_fp64[both_inf]) == torch.sign(golden_truncated[both_inf])):
                return CompareResult(
                    passed=False,
                    dtype=dtype,
                    threshold=threshold,
                    error_msg="Inf symbol mismatch"
                )
            inf_match_mask[both_inf] = True  # Update inf match mask

    # Calculate relative error (standard formula)
    # Formula: |actual - golden| / (|golden| + 1e-7)
    diff = torch.abs(output_fp64 - golden_truncated)
    golden_abs = torch.abs(golden_truncated)
    denominator = golden_abs + 1e-7  # Prevent division by 0

    relative_error = diff / denominator

    # Exclude NaN, Inf, and matching Inf positions
    valid_mask = ~(torch.isnan(relative_error) | torch.isinf(relative_error) | inf_match_mask)
    valid_relative_error = relative_error[valid_mask]

    if len(valid_relative_error) == 0:
        # All positions are NaN/Inf and match, considered passed
        return CompareResult(
            passed=True,
            dtype=dtype,
            threshold=threshold,
            mere=0.0,
            mare=0.0,
            max_diff=0.0,
            mean_diff=0.0,
            mismatch_count=0,
            total_count=output.numel(),
            mismatch_ratio=0.0,
            cancel_error_count=0,
            cancel_cpu_error_count=0,
            cancel_total_count=0,
        )

    mere = float(valid_relative_error.mean())
    mare = float(valid_relative_error.max())

    # Calculate absolute difference statistics
    valid_diff = diff[valid_mask]
    max_diff = float(valid_diff.max()) if len(valid_diff) > 0 else 0.0
    mean_diff = float(valid_diff.mean()) if len(valid_diff) > 0 else 0.0
    total_count = output.numel()

    # ============================================================
    # The first stage: overall relative error determination (priority determination)
    # ============================================================
    # Principle: Relative error is the main criterion. If the overall relative error meets the standard, it will be passed directly.
    # Small value range/cancellation judgment is only to deal with the special situation where "the relative error may be unreasonable"
    mare_threshold = 10 * threshold

    # Calculate the overall relative error (including all valid positions)
    overall_mere = float(valid_relative_error.mean())
    overall_mare = float(valid_relative_error.max())

    # If the overall relative error passes, return directly without requiring small value range/cancellation judgment.
    if overall_mere < threshold and overall_mare < mare_threshold:
        return CompareResult(
            passed=True,
            dtype=dtype,
            threshold=threshold,
            mere=overall_mere,
            mare=overall_mare,
            max_diff=max_diff,
            mean_diff=mean_diff,
            mismatch_count=0,
            total_count=total_count,
            mismatch_ratio=0.0,
            small_value_error_count=0,
            small_value_cpu_error_count=0,
            small_value_total_count=0,
            cancel_error_count=0,
            cancel_cpu_error_count=0,
            cancel_total_count=0,
        )

    # ============================================================
    # The second stage: analyze the reasons for the failure and use the bottom-up judgment
    # ============================================================
    # The overall relative error does not pass. It is necessary to analyze which positions caused it and determine whether it is a special situation.

    # Find the point where the relative error exceeds the standard
    mismatch_mask = relative_error > mare_threshold
    mismatch_mask[~valid_mask] = False
    mismatch_count = int(mismatch_mask.sum())

    # === Small range processing ===
    # Get the small value range threshold
    small_value_threshold = get_small_value_threshold(dtype)
    small_value_error = get_small_value_error(dtype)

    # Filter small value range position: |golden| < small_value_threshold
    small_value_mask = golden_abs < small_value_threshold
    small_value_mask[~valid_mask] = False  # Exclude NaN/Inf
    small_value_total_count = int(small_value_mask.sum())

    # Small range NPU error count: |golden| < threshold and |output - golden| > error
    small_value_npu_error_mask = small_value_mask & (diff > small_value_error)
    small_value_error_count = int(small_value_npu_error_mask.sum())

    # Small range CPU error count
    # Important: The comparison baselines of CPU and NPU must be consistent, both use golden_truncated (FP64 → target_dtype → FP64)
    # In this way, we can fairly compare the error difference between the two and the theoretical true value under the same accuracy limit.
    if cpu_output is not None:
        cpu_output_fp64 = cpu_output.double()
        # CPU differences are also compared to truncated golden, keeping the benchmark consistent
        cpu_diff = torch.abs(cpu_output_fp64 - golden_truncated)
    else:
        # golden is truncated to the target precision and then raised to FP64. This is the "ideal" output of the CPU at the same precision.
        cpu_output_fp64 = golden.to(target_dtype).double()
        # Compare with truncated golden (when cpu_output_fp64 == golden_truncated, diff = 0)
        cpu_diff = torch.abs(cpu_output_fp64 - golden_truncated)

    # CPU small range error count: |golden| < threshold and |cpu_output - golden_truncated| > error
    cpu_small_value_mask = small_value_mask  # Directly use the NPU's small value range mask to ensure consistency.
    cpu_small_value_error_mask = cpu_small_value_mask & (cpu_diff > small_value_error)
    small_value_cpu_error_count = int(cpu_small_value_error_mask.sum())

    # Small value range judgment: ErrorCount_npu / max(ErrorCount_cpu, 1) <= 2
    if small_value_total_count > 0:
        max_cpu_error = max(small_value_cpu_error_count, 1)
        small_value_ratio = small_value_error_count / max_cpu_error
        small_value_passed = small_value_ratio <= 2
    else:
        small_value_passed = True

    # === Cancellation position processing ===
    # Get the cancellation threshold
    cancel_boundary = get_cancel_boundary(dtype)
    cancel_zero_threshold = get_cancel_zero_threshold(dtype)

    # Detect cancellation positions
    output_abs = torch.abs(output_fp64)
    output_near_zero = output_abs < cancel_zero_threshold
    golden_in_cancel_range = (golden_abs < cancel_boundary) & (golden_abs >= small_value_threshold)
    cancel_mask = output_near_zero & golden_in_cancel_range & valid_mask
    cancel_total_count = int(cancel_mask.sum())

    # Cancellation position "wrong" judgment: relative error exceeds mare_threshold
    cancel_npu_error_mask = cancel_mask & (relative_error > mare_threshold)
    cancel_error_count = int(cancel_npu_error_mask.sum())

    # CPU relative error exceedance count
    cpu_relative_error = cpu_diff / (golden_abs + 1e-7)
    cancel_cpu_error_mask = cancel_mask & (cpu_relative_error > mare_threshold)
    cancel_cpu_error_count = int(cancel_cpu_error_mask.sum())

    # Cancellation position determination: ErrorCount_npu / max(ErrorCount_cpu, 1) <= 2
    if cancel_total_count > 0:
        max_cpu_cancel_error = max(cancel_cpu_error_count, 1)
        cancel_ratio = cancel_error_count / max_cpu_cancel_error
        cancel_passed = cancel_ratio <= 2
    else:
        cancel_passed = True

    # === Analysis of failure reasons ===
    # Check whether the points with excessive relative errors are all within the small value range/cancellation range
    mismatch_in_small_value = mismatch_mask & small_value_mask
    mismatch_in_cancel = mismatch_mask & cancel_mask
    mismatch_in_normal = mismatch_mask & ~small_value_mask & ~cancel_mask

    normal_mismatch_count = int(mismatch_in_normal.sum())

    # Final decision logic:
    # 1. If there is a point in the normal value range where the relative error exceeds the standard → fail directly (not a special case)
    # 2. If only the small value range/cancellation position exceeds the standard → use the bottom line judgment
    if normal_mismatch_count > 0:
        # If the error in the normal value range exceeds the standard, it will fail directly.
        passed = False
        # Calculate MERE/MARE after excluding small value range/cancellation for display
        normal_mask = ~small_value_mask & ~cancel_mask & valid_mask
        normal_relative_error = relative_error[normal_mask]
        if len(normal_relative_error) > 0:
            display_mere = float(normal_relative_error.mean())
            display_mare = float(normal_relative_error.max())
        else:
            display_mere = 0.0
            display_mare = 0.0
    else:
        # Only special positions (small value range/cancellation) where the error exceeds the standard, use the bottom line judgment
        passed = small_value_passed and cancel_passed
        display_mere = overall_mere
        display_mare = overall_mare

    return CompareResult(
        passed=passed,
        dtype=dtype,
        threshold=threshold,
        mere=display_mere,
        mare=display_mare,
        max_diff=max_diff,
        mean_diff=mean_diff,
        mismatch_count=mismatch_count,
        total_count=total_count,
        mismatch_ratio=mismatch_count / total_count if total_count > 0 else 0.0,
        small_value_error_count=small_value_error_count,
        small_value_cpu_error_count=small_value_cpu_error_count,
        small_value_total_count=small_value_total_count,
        cancel_error_count=cancel_error_count,
        cancel_cpu_error_count=cancel_cpu_error_count,
        cancel_total_count=cancel_total_count,
    )


def compare_with_custom_threshold(
    output: torch.Tensor,
    golden: torch.Tensor,
    threshold_dict: Dict[str, float] = None,
) -> CompareResult:
    """    Compare using a custom threshold table

    Args:
        output: operator output tensor
        golden: Golden reference output tensor
        threshold_dict: Custom threshold table, the format is {dtype: threshold}

    Returns:
        CompareResult: Compare results
    """
    if threshold_dict is None:
        threshold_dict = PRECISION_THRESHOLDS

    # Infer dtype from tensor
    dtype_str = str(output.dtype).replace('torch.', '')
    threshold = get_threshold(dtype_str)
    if dtype_str.lower() in threshold_dict:
        threshold = threshold_dict[dtype_str.lower()]

    return compare_tensors(output, golden, dtype_str, threshold)
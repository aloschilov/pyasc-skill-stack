# TTK — the CANN Test Tool Kit manifest

## Why you care

When a result says a kernel reaches "0.43 of CANN" or "parity with CANN", TTK
is what CANN means there. It is the CANN-side test harness, and its manifest is
the case definition both sides are measured on. Without it you have two
numbers measured on two different problems.

Do not invent a baseline. If no TTK manifest backs a comparison, report the
raw tick counts and mark the ratio `blocked`.

## The 42 columns

One row per case. Grouped by what they pin down:

**Identity** — `network_name`, `testcase_name`, `op_name`, `stc_op_name`

**Static shapes and types** — `stc_input_dtypes`, `stc_ori_inputs`,
`stc_ori_outputs`, `stc_inputs`, `stc_outputs`, `output_dtypes`

**Formats** — `stc_input_ori_formats`, `stc_input_formats`,
`output_ori_formats`, `output_formats`. Original vs. actual differ whenever
CANN re-lays-out a tensor (`NCHW` -> `NC1HWC0`), and getting this wrong is a
common source of an unexplained ratio.

**Dynamic shapes** — `dyn_inputs`, `dyn_input_dtypes`, `dyn_outputs`,
`dyn_ori_inputs`, `dyn_ori_outputs`, `dyn_input_formats`,
`dyn_input_ori_formats`, `dyn_input_ranges`, `dyn_output_ranges`. A dynamic
dimension is `-1` with its range carried alongside as `(1, None)`.

**Parameters** — `other_compilation_params`, `other_runtime_params`,
`const_input_indexes`, `input_as_variable`,
`stc_input_as_list_distribution`, `dyn_input_as_list_distribution`

**Numerics** — `precision_tolerances`, `strict_precision_mode`,
`absolute_precision`, `input_data_ranges`, `random_buff`. `input_data_ranges`
must match the CANNBench case's `value_range`; a GeLU measured on `[-1, 1]` and
one measured on `[-10, 10]` exercise different branches.

**Execution** — `shape_check`, `output_inplace_indexes`, `is_enabled`,
`bucket`, `arity`, `position`, `tiling_key`, `kernel`

`bucket` classifies by element count: `small` under 100K, `medium` under 5M,
`big` above. `tiling_key` selects the CANN tiling branch — two rows differing
only in `tiling_key` compare different CANN code paths.

Structured cells are Python literals (`repr`), not JSON: tuples stay tuples and
`nan` / `inf` / `-inf` are written as bare literal-evaluable tokens.

## Mapping a CANNBench case onto a row

`integrations/cannbench/tasks/<op>/cases.yaml` carries `input_shape`, `dtype`,
`attrs`, `value_range`. The translation:

| CANNBench | TTK |
|---|---|
| `input_shape` | `stc_ori_inputs`, `stc_inputs`; `(-1, …)` into the `dyn_*` columns with `dyn_input_ranges = ((1, None), …)` |
| `dtype` | `stc_input_dtypes`, `output_dtypes` |
| `attrs` | `other_compilation_params` |
| `value_range` | `input_data_ranges` |
| `prod(input_shape)` | `bucket` |

## What TTK bought, concretely

A near-parity TTK export on upstream MR #391 moved reduce-max from a geomean of
**0.426 of CANN** to **~1.04 (parity)**. The export did not make the kernel
faster — it revealed which three tiling levers CANN was pulling that the
generated kernel was not. Those levers are written up in
[`../../pyasc-api-patterns/references/reduction-tiling.md`](../../pyasc-api-patterns/references/reduction-tiling.md).

That is the normal use: a TTK manifest is a diagnostic instrument first and a
scoreboard second.

## Reproducing a manifest — currently blocked

The generator that produced the 20-case GeLU manifest
(`integrations/cannbench/comparisons/gelu-handwritten-deepdive-20260903/generate_ttk_csv.py`,
now on `archive/campaign-2026`) read its reference rows from an absolute path
outside the repository:

```
~/Documents/<date>/.../ttk-csv-recovered/
    concatd_selected_representative_adjusted.csv
```

So it cannot be re-run from a clean clone. Either vendor that input or document
the TTK export that produces it; until then, treat TTK-backed ratios as
`blocked` rather than reconstructing the manifest by hand.

Recover the historical artefacts with:

```bash
git checkout archive/campaign-2026 -- \
  integrations/cannbench/comparisons/gelu-handwritten-deepdive-20260903/
```

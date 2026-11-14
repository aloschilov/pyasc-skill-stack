"""Run one public unary case through the installed CANNBench restored-input evaluator."""
import argparse
import hashlib
import importlib
import inspect
import json
import statistics
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(package, task, case, output):
    import torch
    from kernel_eval.benches.cann_loader import CannCaseLoader
    from kernel_eval.config import Config
    from kernel_eval.data.data_generator import DataGenerator
    from kernel_eval.data.tensor_cache import case_seed
    from kernel_eval.eval.op_runner import OpRunner
    from kernel_eval.eval.perf_eval import PerfEvaluator
    from kernel_eval.utils.device_manager import DeviceConfig, DeviceManager
    from kernel_eval.utils.tensor_utils import non_contiguous_reason

    operator = task.name
    sys.path[:0] = [str(package.parent), str(task)]
    candidate = getattr(importlib.import_module(f"{package.name}.{operator}"), operator)
    golden = getattr(importlib.import_module("golden"), operator)
    tasks_root = task.parents[1]
    config = Config(tasks_root=str(tasks_root), reports_dir=str(output))
    rel_path = task.relative_to(tasks_root).as_posix()
    specs = CannCaseLoader(tasks_root=str(tasks_root)).scan_by_rel_path(rel_path)
    spec, = (entry for entry in specs if entry.case_num == case)
    seed = case_seed(spec, config.eval_seed)
    generator = DataGenerator(input_dist=config.input_dist)
    input_tensor, = generator.generate_input_tensors_from_case(
        spec.input_shapes, spec.dtypes, spec.value_ranges, seed=seed
    )
    attrs = spec.attrs
    expected = golden(input_tensor, **attrs)
    checked = []

    def validate(actual, args, kwargs, index, phase):
        if str(actual.device) != manager.get_device() or actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError("output device, dtype or shape changed")
        layout_error = non_contiguous_reason(actual)
        if layout_error is not None:
            raise ValueError(layout_error)
        actual_cpu = actual.cpu().flatten()
        for part, ref in zip(actual_cpu.split(1048576), expected.flatten().split(1048576)):
            torch.testing.assert_close(part, ref, equal_nan=True)
        checked.append(phase)

    if not config.perf_restore_inputs:
        raise ValueError("CANNBench restored-input protocol required")
    manager = DeviceManager(DeviceConfig(type="npu", device_id=0))
    perf = PerfEvaluator(config=config, device_manager=manager)
    params = dict(inspect.signature(candidate).bind(input_tensor, **attrs).arguments)
    try:
        result = OpRunner(manager, perf).run_ai_op(candidate, params, spec.get_case_id_str(),
                                                   [input_tensor], validator=validate)
        if not result.success:
            raise RuntimeError(result.error)
        stats = result.perf_result
        meta = stats.metadata
        phases = ["preflight"] + ["warmup"] * perf.warmup + ["measured"] * perf.repeat
        if (checked != phases or meta["perf_protocol"] != "restored_io_v2"
                or meta["io_retired_calls"] != len(phases) or meta["validated_calls"] != perf.repeat
                or len(meta["step_kernel_us"]) != perf.repeat
                or round(statistics.median(meta["step_kernel_us"]), 2) != result.elapsed_us):
            raise ValueError(f"incomplete restored run: {meta}")
        with (output / "result.json").open("x") as f:
            json.dump(dict(operator=operator, case=case, elapsed_us=result.elapsed_us,
                           case_id=spec.get_case_id_str(), eval_seed=config.eval_seed, fixture_seed=seed,
                           input_dist=config.input_dist, cases_yaml_sha256=digest(task / "cases.yaml"),
                           metadata=meta, validated=checked, source_sha256=digest(package / f"{operator}.py"),
                           evaluator_sha256=digest(Path(inspect.getfile(PerfEvaluator))),
                           cases_sha256=digest(task / "cases.csv"), golden_sha256=digest(task / "golden.py")), f)
    finally:
        perf.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("task", type=Path)
    parser.add_argument("case", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run(args.package.resolve(), args.task.resolve(), args.case, args.output.resolve())

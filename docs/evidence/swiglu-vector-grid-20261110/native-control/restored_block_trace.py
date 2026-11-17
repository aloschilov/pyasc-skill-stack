"""Run restored unary inputs under an external msprof --task-block=on session."""
import argparse
import hashlib
import importlib
import json
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
    from kernel_eval.eval.perf_eval import PerfEvaluator
    from kernel_eval.eval.restored_inputs import RestoredCall, migrate_call_inputs
    from kernel_eval.utils.device_manager import DeviceConfig, DeviceManager
    from kernel_eval.utils.tensor_utils import non_contiguous_reason

    name = task.name
    sys.path[:0] = [str(package.parent), str(task)]
    candidate = getattr(importlib.import_module(f"{package.name}.{name}"), name)
    golden = getattr(importlib.import_module("golden"), name)
    tasks = task.parents[1]
    config = Config(tasks_root=str(tasks), reports_dir=str(output.parent))
    loader = CannCaseLoader(tasks_root=str(tasks))
    spec, = (row for row in loader.scan_by_rel_path(task.relative_to(tasks).as_posix())
             if row.case_num == case)
    seed = case_seed(spec, config.eval_seed)
    host_input, = DataGenerator(input_dist=config.input_dist).generate_input_tensors_from_case(
        spec.input_shapes, spec.dtypes, spec.value_ranges, seed=seed)
    expected = golden(host_input, **spec.attrs)
    manager = DeviceManager(DeviceConfig(type="npu", device_id=0))
    perf = PerfEvaluator(config=config, device_manager=manager)
    checked = []

    def validate(actual, args, kwargs, index, phase):
        if str(actual.device) != manager.get_device() or non_contiguous_reason(actual):
            raise ValueError("unexpected output device or layout")
        torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)
        checked.append((index, phase))

    # The external collector captures preparation and validation too. Filter
    # exported blocks by the candidate's task ID before analyzing durations.
    args, kwargs, _ = migrate_call_inputs((host_input,), spec.attrs, manager.to_device)
    call = RestoredCall(candidate, args, kwargs, validator=validate, synchronize=manager.synchronize)
    phases = ["preflight"] + ["warmup"] * perf.warmup + ["measured"] * perf.repeat
    try:
        manager.synchronize()
        if perf.freq_boost:
            perf._prepare_warmup_tensors()
        for index, phase in enumerate(phases):
            call.prepare(index, phase)
            manager.synchronize()
            if perf.freq_boost and phase == "measured":
                perf._clear_cache()
            call()
            manager.synchronize()
            call.complete(call.io.output, index, phase)
            call.retire(preserve_output=index == len(phases) - 1)
    finally:
        call.release()
        perf.shutdown()
    if checked != list(enumerate(phases)):
        raise ValueError("incomplete block-trace validation")
    receipt = dict(case_id=spec.get_case_id_str(), fixture_seed=seed, checked=checked,
                   source_sha256=digest(package / f"{name}.py"),
                   golden_sha256=digest(task / "golden.py"),
                   cases_sha256=digest(task / "cases.csv"), freq_boost=perf.freq_boost)
    with output.open("x") as stream:
        json.dump(receipt, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("task", type=Path)
    parser.add_argument("case", type=int)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    run(args.package.resolve(), args.task.resolve(), args.case, args.receipt.resolve())

"""Time an unedited generated module on host or NPU Torch tensors with the same profiler."""
import argparse
import csv
import hashlib
import importlib
import json
import statistics
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    import torch_npu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("task", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("case", choices=(11, 13), type=int)
    parser.add_argument("placement", choices=("host", "device"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--conditioned", action="store_true", help="use CANNBench frequency warmup and L2 clean")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    sys.path.insert(0, str(args.candidate.resolve().parent))
    sys.path.insert(0, str(args.task.resolve()))
    wrapper = importlib.import_module(args.candidate.name + ".swi_glu")
    from golden import swi_glu as golden

    with (args.task / "cases.csv").open(newline="") as csvfile:
        row, = (r for r in csv.DictReader(csvfile) if int(r["case_id"]) == args.case)
    shape, = json.loads(row["input_shape"])
    attrs = json.loads(row["attrs"])
    cpu = torch.from_file(str(args.raw), size=args.raw.stat().st_size // 4, dtype=torch.float32).reshape(shape)
    expected = golden(cpu, **attrs)
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    x = cpu.to(device) if args.placement == "device" else cpu

    def invoke():
        y = wrapper.swi_glu(x, **attrs)
        if y.device != x.device:
            raise ValueError(f"input on {x.device}, output on {y.device}")
        return y

    def check(y):
        torch.testing.assert_close(y.cpu(), expected, equal_nan=True)

    torch.npu.synchronize()
    check(invoke())
    if args.conditioned:
        from cann_bench_utils import cann_bench_cache_clean, cann_bench_warmup
        a, b, scratch = [torch.rand(shape, dtype=torch.float16).to(device)
                         for shape in ((10240, 10240), (10240, 10240), (96, 1024, 1024))]
        cann_bench_warmup(a, b)
        torch.npu.synchronize()
        cann_bench_cache_clean(scratch)
        torch.npu.synchronize()
    profiler = torch_npu.profiler
    trace = args.output / "trace"
    with profiler.profile(activities=[profiler.ProfilerActivity.NPU],
                          schedule=profiler.schedule(wait=0, warmup=3, active=5, repeat=1),
                          on_trace_ready=profiler.tensorboard_trace_handler(str(trace)),
                          experimental_config=profiler._ExperimentalConfig(
                              export_type=[profiler.ExportType.Text],
                              profiler_level=profiler.ProfilerLevel.Level1),
                          record_shapes=False, profile_memory=False, with_stack=False) as session:
        for step in range(8):
            if args.conditioned and step >= 3:
                cann_bench_cache_clean(scratch)
                torch.npu.synchronize()
            y = invoke()
            torch.npu.synchronize()
            session.step()
    check(y)
    file, = trace.rglob("kernel_details.csv")
    with file.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    # Include every vector-core task. A changed kernel or extra device arithmetic
    # is an unexpected invocation path for these two one-wave specializations.
    vector = [r for r in records if r["Accelerator Core"].upper() == "AI_VECTOR_CORE"]
    sequence = ["clean" if "cannbenchcacheclean" in r["Name"].lower() else r["Name"] for r in vector]
    target = "row_single_wave_fp32"
    if sequence != (["clean", target] if args.conditioned else [target]) * 5:
        raise ValueError(f"unexpected measured vector tasks: {sequence}")
    selected = [r for r in vector if r["Name"] == target]
    expected_grid = {11: 35, 13: 36}[args.case]
    if len(selected) != 5 or any(r["Name"] != "row_single_wave_fp32" or
                                int(r["Block Num"]) != expected_grid for r in selected):
        raise ValueError([(r["Name"], r["Accelerator Core"], r["Block Num"]) for r in records])
    times = [float(r["Duration(us)"]) for r in selected]
    if any(t <= 0 for t in times):
        raise ValueError(times)
    result = dict(case=args.case, device=str(x.device), output_device=str(y.device), correct=True,
                  times_us=times, median_us=statistics.median(times), grid=expected_grid, conditioned=args.conditioned,
                  input_sha256=digest(args.raw), golden_sha256=digest(args.task / "golden.py"),
                  source_sha256=digest(Path(wrapper.__file__)), csv_sha256=digest(file),
                  torch=torch.__version__, torch_npu=torch_npu.__version__,
                  finite_output=int(expected.isfinite().sum()), shape=shape)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

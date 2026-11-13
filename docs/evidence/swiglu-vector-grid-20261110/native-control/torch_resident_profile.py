"""Compare an emitted two-pointer SwiGLU ELF and native SwiGLU on resident torch_npu inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct
import sys

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("task", type=Path)
    parser.add_argument("host", type=Path, help="directory containing device_host.py")
    parser.add_argument("case", type=int, choices=(11, 13))
    parser.add_argument("operator", choices=("emitted", "native"))
    parser.add_argument("mode", choices=("plain", "conditioned"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    sys.path[:0] = [str(args.task.resolve()), str(args.host.resolve())]
    from golden import swi_glu
    from device_host import launch, stream_ptr

    shape, dim, blocks = {11: ((3, 7, 13, 1018), -1, 35),
                          13: ((11, 13, 16, 67), 2, 36)}[args.case]
    data = torch.from_file(str(args.raw), dtype=torch.float32, size=args.raw.stat().st_size // 4)
    cpu = data.reshape(shape)
    oracle = swi_glu(cpu, dim)
    x = cpu.npu(0)
    code = args.elf.read_bytes()
    symbol = "row_single_wave_fp32"
    stream = stream_ptr(x)

    def invoke():
        if args.operator == "native":
            return torch_npu.npu_swiglu(x, dim=dim)
        y = torch.empty(oracle.shape, dtype=x.dtype, device=x.device)
        launch(code, symbol, blocks, struct.pack("<QQ", x.data_ptr(), y.data_ptr()), stream)
        return y

    def check(result):
        actual = result.cpu()
        torch.testing.assert_close(actual, oracle, equal_nan=True)

    check(invoke())
    if args.mode == "conditioned":
        from cann_bench_utils import cann_bench_cache_clean, cann_bench_warmup
        a, b, scratch = [torch.rand(s, dtype=torch.float16).to(x.device)
                         for s in [(10240, 10240), (10240, 10240), (96, 1024, 1024)]]
        cann_bench_warmup(a, b)
        torch.npu.synchronize(x.device)
        cann_bench_cache_clean(scratch)
        torch.npu.synchronize(x.device)

    prof = torch_npu.profiler
    config = prof._ExperimentalConfig(export_type=[prof.ExportType.Text],
                                      profiler_level=prof.ProfilerLevel.Level1,
                                      aic_metrics=prof.AiCMetrics.AiCoreNone)
    trace = args.output / "trace"
    with prof.profile(activities=[prof.ProfilerActivity.NPU],
                      schedule=prof.schedule(wait=0, warmup=3, active=5, repeat=1),
                      on_trace_ready=prof.tensorboard_trace_handler(str(trace)),
                      experimental_config=config, record_shapes=False,
                      profile_memory=False, with_stack=False) as session:
        for step in range(8):
            if args.mode == "conditioned" and step >= 3:
                cann_bench_cache_clean(scratch)
                torch.npu.synchronize(x.device)
            result = invoke()
            torch.npu.synchronize(x.device)
            session.step()
    check(result)
    files = list(trace.rglob("kernel_details.csv"))
    if len(files) != 1:
        raise ValueError(f"expected one kernel_details.csv, got {files}")
    with files[0].open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    target = symbol if args.operator == "emitted" else "SwiGlu"
    selected = [r for r in rows if target.lower() in r["Name"].lower()]
    clean = [r for r in rows if "cannbenchcacheclean" in r["Name"].lower()]
    if len(selected) != 5 or len(clean) != (5 if args.mode == "conditioned" else 0):
        raise ValueError(f"unexpected trace: {[(r['Name'], r['Duration(us)']) for r in rows]}")
    times = [float(r["Duration(us)"]) for r in selected]
    if any(t <= 0 for t in times):
        raise ValueError(times)
    summary = dict(case=args.case, operator=args.operator, mode=args.mode, golden=True,
                   input_nan=int(cpu.isnan().sum()), finite_output=int(oracle.isfinite().sum()),
                   input_sha256=hashlib.sha256(args.raw.read_bytes()).hexdigest(),
                   elf_sha256=hashlib.sha256(code).hexdigest(), device=x.get_device(),
                   torch=torch.__version__, torch_npu=torch_npu.__version__,
                   timings_us=times, median_us=statistics.median(times),
                   names=[r["Name"] for r in selected], cache_clean=len(clean))
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{args.case} {args.operator} {args.mode} {times}", flush=True)


if __name__ == "__main__":
    main()

"""Numerical and CaModel tick comparison on fixed deterministic inputs."""
import argparse
import importlib
import importlib.util
import json
import hashlib
import math
import time
from pathlib import Path
import sys

import torch
import asc.runtime.config as config
from asc.lib import runtime as rt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', required=True)
    p.add_argument('--size', type=int, default=13824)
    p.add_argument('--dtype', default='float32')
    p.add_argument('--mode', default='none')
    p.add_argument('--tile', type=int)
    p.add_argument('--cores', type=int)
    p.add_argument('--low', type=float, default=-8.)
    p.add_argument('--high', type=float, default=8.)
    p.add_argument('--output', type=Path)
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--special', action='store_true')
    p.add_argument('--case-group', action='store_true', help='Sample every official case for this dtype/mode, not full shapes')
    a = p.parse_args()
    torch.set_num_threads(2)
    config.set_platform(config.Backend.Model, config.Platform.Ascend950PR_9599)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mod = importlib.import_module(a.package + '.gelu')
    source_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
    mod.ensure_npu_platform = lambda: None
    if a.tile:
        for k in ('_TILE', '_EXACT_F16_TILE', '_ERF_TILE', '_TANH_TILE'):
            if hasattr(mod, k): setattr(mod, k, a.tile)
    if a.cores: mod._MAX_CORES = a.cores
    x = torch.linspace(a.low, a.high, a.size).to(getattr(torch, a.dtype))
    if a.special:
        values = torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,1.,-1.,1.e-30,-1.e-30])
        x = values.repeat((a.size+8)//9)[:a.size].to(x.dtype)
    segments=[]
    if a.case_group:
        import yaml
        cases=yaml.safe_load((Path(__file__).resolve().parents[2]/'tasks/gelu/cases.yaml').read_text())['cases']
        cases=[c for c in cases if c['dtype'][0]==a.dtype and c['attrs']['approximate']==a.mode]
        torch.manual_seed(20260907)
        chunks=[]
        offset=0
        for c in cases:
            lo,hi=map(float,c['value_range'])
            n=2053
            if math.isnan(lo):
                v=torch.randn(n)
                v[::2]=float('nan')
            elif math.isinf(lo) or math.isinf(hi):
                v=torch.randn(n)
                v[:103]=-float('inf')
                v[-103:]=float('inf')
            else:
                v=torch.linspace(lo,hi,n)
            chunks.append(v)
            segments.append((c['case_id'],offset,offset+n))
            offset+=n
        # Additional nonfinite/zero values exercise the mathematical contract,
        # regardless of whether a particular official range contains them.
        chunks.append(torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,1.,-1.]))
        x=torch.cat(chunks).to(x.dtype)
    before = rt.current_tick()
    start = time.monotonic()
    for _ in range(a.repeat):
        y = mod.gelu(x, approximate=a.mode)
    after = rt.current_tick()
    golden = (x.double() * .5 * torch.erfc(-x.double() / 2.**.5) if a.mode=='none' else
              x.double() / (1 + torch.exp(-2 * (2/torch.pi)**.5 * (x.double() + .044715*x.double()**3))))
    golden = golden.to(x.dtype)
    finite = torch.isfinite(golden) & torch.isfinite(y)
    error = (y.double()-golden.double()).abs()[finite]
    relative = error / golden.double().abs()[finite].clamp_min(1.e-30)
    spec = importlib.util.spec_from_file_location('cannbench_precision', '/home/aloschilov/workspace/cann-bench/src/kernel_eval/utils/precision.py')
    precision = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = precision
    spec.loader.exec_module(precision)
    comparison = precision.compare_tensors(y, torch.nn.functional.gelu(x.double(), approximate=a.mode), dtype=a.dtype,
                                          cpu_output=torch.nn.functional.gelu(x, approximate=a.mode))
    result = dict(package=a.package, size=x.numel(), dtype=a.dtype, mode=a.mode, tile=a.tile,
                  cores=a.cores, low=a.low, high=a.high,
                  ticks=None if before is None or after is None else after-before,
                  seconds=time.monotonic()-start, max_abs=error.max().item() if error.numel() else None,
                  max_rel=relative.max().item() if error.numel() else None,
                  mean_rel=relative.mean().item() if error.numel() else None,
                  nan_match=torch.equal(torch.isnan(y),torch.isnan(golden)),
                  special=a.special, repeat=a.repeat,
                  source_sha256=source_sha256,
                  checker_passed=comparison.passed, checker_error=comparison.error_msg)
    if a.case_group:
        result['coverage']='Reduced-size samples of all official value ranges; not full CANNBench shapes'
        result['cases']=[]
        for case_id,lo,hi in segments:
            v=x[lo:hi]
            c=precision.compare_tensors(y[lo:hi],torch.nn.functional.gelu(v.double(),approximate=a.mode),dtype=a.dtype,
                                        cpu_output=torch.nn.functional.gelu(v,approximate=a.mode))
            result['cases'].append(dict(case_id=case_id,passed=c.passed,mere=c.mere,mare=c.mare,error=c.error_msg))
    if a.special:
        result['sample_input'] = x[:9].float().tolist()
        result['sample_output'] = y[:9].float().tolist()
        result['sample_reference'] = torch.nn.functional.gelu(x[:9].double(),approximate=a.mode).tolist()
    print(json.dumps(result), flush=True)
    if a.output: a.output.write_text(json.dumps(result, indent=2)+'\n')

if __name__ == '__main__': main()

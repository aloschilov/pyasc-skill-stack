"""Bounded original-target diagnostics. No NPU submission or compiler changes."""
import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
PIN='adadd7d66ed0ee16d33d79487bf584899a26ef1e'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['numerical','perf','compile','tail'])
    p.add_argument('--variant',choices=['corrected','original','stable','promoted','exact','exact_stable','exact_fast','submission'],default='corrected')
    p.add_argument('--mode',choices=['none','tanh'],default='none',help='Final combined submission mode')
    p.add_argument('--dtype',default='float32',choices=['float32','float16','bfloat16'])
    p.add_argument('--tile',type=int,default=15872)
    p.add_argument('--tune-geometry',action='store_true',help='Honor explicit tile/unroll/VF for matched local tuning')
    p.add_argument('--unroll',type=int,default=2)
    p.add_argument('--cores',type=int,default=1)
    p.add_argument('--size',type=int,default=31744)
    p.add_argument('--reuse',type=int,default=1)
    p.add_argument('--vf',type=int,default=0)
    p.add_argument('--static',choices=['default','false','true'],default='default')
    p.add_argument('--compact-stress',action='store_true',help='Bounded independent random-tail and switch-boundary checks')
    p.add_argument('--repeat',type=int,default=1,help='Numerical launches including cached repeats; not timing')
    p.add_argument('--perf-range',type=float,nargs=2,default=[-2.0,2.0])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output=a.output.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
    os.environ['PYASC_CACHE_DIR']=str(a.output.parent/'cache')
    os.environ['PYASC_DUMP_PATH']=str(a.output.parent/'diagnostic')
    r={'pin':PIN,'args':vars(a)|{'output':str(a.output)},'completed':False,'observations':[]}
    r['harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def save():a.output.write_text(json.dumps(r,indent=2,default=str)+'\n')
    try:
        import asctile
        import torch
        import yaml
        from target_kernel import gelu
        from asc.runtime.jit import CompilePrereqs, MockTensor
        from asc.common.compat import get_annotations
        from asc.lib import runtime
        torch.set_num_threads(2)
        asctile.set_platform(asctile.Backend.Model,asctile.Platform.Ascend950PR_9599)
        r['runtime_file']=asctile.__file__
        expected_runtime=ROOT.parent/'gelu-asctile-jit-20260907/runtime/source/python/asctile/__init__.py'
        assert Path(asctile.__file__).resolve()==expected_runtime.resolve()
        source=ROOT/'target_kernel.py'
        upstream=expected_runtime.parents[1]/'test/asctile/target/test_gelu.py'
        def kernel_ast(path):
            return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='gelu'))
        assert kernel_ast(source)==kernel_ast(upstream),'Authored target was changed'
        r['target_function_ast_unchanged']=True
        r['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        r['upstream_file_sha256']=hashlib.sha256(upstream.read_bytes()).hexdigest()
        spec=importlib.util.spec_from_file_location('official_precision','/home/aloschilov/workspace/cann-bench/src/kernel_eval/utils/precision.py')
        precision=importlib.util.module_from_spec(spec);sys.modules[spec.name]=precision;spec.loader.exec_module(precision)
        r['checker_sha256']=hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
        dtype=getattr(torch,a.dtype)
        if a.variant=='submission' and not a.tune_geometry:
            from candidate.gelu import geometry
            a.tile,a.unroll,a.vf=geometry(dtype==torch.float32,a.mode)
        options=dict(reuse_alloc=a.reuse,static_alloc={'default':None,'false':False,'true':True}[a.static],
                     vf_fusion=bool(a.vf),insert_sync=True,opt_level=3,debug=False)
        r['requested_options']=options
        factor=1/.044715 if a.variant=='original' else .044715
        scale=-1.595769121*.044715 if a.variant=='original' else -math.sqrt(8/math.pi)
        r['coefficients']={'TANH_APPROX_FACTOR':factor,'NEG_SQRT_EIGHT_OVER_PI':scale}
        kernel=gelu
        mode='none' if a.variant in ('exact','exact_stable','exact_fast') else 'tanh'
        if a.variant=='submission':
            mode=a.mode
            from candidate.gelu import gelu_kernel
            kernel=gelu_kernel
            r['target_function_ast_unchanged']=False
            r['source_sha256']=hashlib.sha256((ROOT/'candidate/gelu.py').read_bytes()).hexdigest()
            r['selected_geometry']={'tile':a.tile,'unroll':a.unroll,'vf_fusion':bool(a.vf)}
        r['approximate']=mode
        if a.variant=='exact':
            from exact_kernel import gelu as exact_gelu
            kernel=exact_gelu
            r['target_function_ast_unchanged']=False
            r['source_sha256']=hashlib.sha256((ROOT/'exact_kernel.py').read_bytes()).hexdigest()
            r['coefficients']={'inverse_sqrt_two':0.7071067811865476}
            r['change_scope']='New exact target-style template; FP32 intermediates, scale x first, explicit tails'
        if a.variant=='exact_stable':
            from exact_stable_kernel import gelu as stable_exact_gelu
            kernel=stable_exact_gelu
            r['target_function_ast_unchanged']=False
            r['source_sha256']=hashlib.sha256((ROOT/'exact_stable_kernel.py').read_bytes()).hexdigest()
            r['coefficients']={'inverse_sqrt_two':0.7071067811865476,'tail_cutoff':-2.0,'continued_fraction_depth':12}
            r['change_scope']='Public Erf central route plus cancellation-free normal tail continued fraction'
        if a.variant=='exact_fast':
            from exact_fast_kernel import gelu as fast_exact_gelu
            kernel=fast_exact_gelu
            r['target_function_ast_unchanged']=False
            r['source_sha256']=hashlib.sha256((ROOT/'exact_fast_kernel.py').read_bytes()).hexdigest()
            r['coefficients']={'inverse_sqrt_two':0.7071067811865476,'tail_cutoff':-3.0,'continued_fraction_depth':6}
            r['change_scope']='Public Erf central route plus six-level normal tail for x < -3'
        if a.variant=='promoted':
            from promoted_kernel import gelu as promoted_gelu
            kernel=promoted_gelu
            r['target_function_ast_unchanged']=False
            r['source_sha256']=hashlib.sha256((ROOT/'promoted_kernel.py').read_bytes()).hexdigest()
            r['change_scope']='Separate diagnostic extension: FP32 intermediate cast and cast-back; not coefficient-only target'
        if a.variant=='stable':
            sys.path.insert(0,str(ROOT.parent/'gelu-asctile-jit-20260907'))
            from candidate.gelu import gelu_kernel
            kernel=gelu_kernel
            r['source_sha256']=hashlib.sha256((ROOT.parent/'gelu-asctile-jit-20260907/candidate/gelu.py').read_bytes()).hexdigest()
        segments=[];chunks=[]
        def add(name,values):
            start=sum(v.numel() for v in chunks)
            chunks.append(values.to(dtype));segments.append((name,start,start+values.numel()))
        gen=torch.Generator().manual_seed(20260907)
        if a.phase=='numerical':
            add('normal_seed_20260907',torch.randn(512 if a.compact_stress else 4096,generator=gen))
            add('dense_negative_tails',torch.linspace(-10,10,2049 if a.compact_stress else 8193))
            if a.compact_stress:
                add('random_negative_tail',-2.0-6.0*torch.rand(2048,generator=gen))
                switch=-3.0 if a.variant in ('exact_fast','submission') else -2.0
                add('switch_boundary',torch.linspace(switch-.001,switch+.001,257))
                edge=torch.tensor(switch,dtype=dtype)
                add('adjacent_representable_switch',torch.stack([edge,torch.nextafter(edge,torch.tensor(-float('inf'),dtype=dtype)),torch.nextafter(edge,torch.tensor(float('inf'),dtype=dtype))]))
            cases=yaml.safe_load((REPO/'integrations/cannbench/tasks/gelu/cases.yaml').read_text())['cases']
            for c in cases:
                if c['dtype'][0]!=a.dtype or c['attrs']['approximate']!=mode:continue
                lo,hi=c['value_range']
                count=65 if a.compact_stress else 513
                if math.isnan(lo):v=torch.full((count,),float('nan'))
                elif not math.isfinite(lo):v=torch.tensor([-float('inf'),float('inf'),0.])
                else:v=torch.linspace(lo,hi,count)
                add('official_range_'+str(c['case_id']),v)
            add('special_values',torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,1e-30,-1e-30]))
            lim=torch.finfo(dtype).max
            add('finite_limits',torch.tensor([lim,lim*.75,-lim,-lim*.75,.5,-.5,2.,-2.]))
            values=torch.cat(chunks)
            n=math.ceil(values.numel()/(a.tile*a.cores))*a.tile*a.cores
            x=torch.zeros(n,dtype=dtype);x[:values.numel()]=values
            y=torch.full_like(x,float('nan'))
        elif a.phase=='tail':
            n=a.size
            allocated=math.ceil(math.ceil(n/a.cores)/a.tile)*a.tile*a.cores
            backing_x=torch.zeros(allocated+256,dtype=dtype)
            backing_x[:n]=torch.linspace(-2,2,n).to(dtype)
            backing_y=torch.full_like(backing_x,12345.)
            x=backing_x[:n];y=backing_y[:n]
            segments=[('tail_valid_region',0,n)]
            r['guard_allocation_elements']=allocated+256
        elif a.phase=='compile':
            n=a.size;x=MockTensor(getattr(asctile,a.dtype));y=MockTensor(getattr(asctile,a.dtype))
        else:
            n=a.size;x=torch.linspace(*a.perf_range,n).to(dtype);y=torch.full_like(x,float('nan'))
            if a.phase=='perf' and a.variant!='submission':assert n%(a.tile*a.cores)==0,'Original target requires aligned input'
            segments=[('finite_perf_input',0,n)]
        args=(x,y,n,a.tile,factor,scale,a.unroll) if a.variant!='stable' else (x,y,n,a.tile,True,a.unroll)
        if a.variant in ('exact','exact_stable','exact_fast'):args=(x,y,n,a.tile,a.unroll)
        if a.variant=='submission':args=(x,y,n,a.tile,a.unroll,mode=='tanh',mode=='none' and a.dtype=='float32')
        r['elements']=n;r['segments']=[{'name':name,'start':start,'end':end} for name,start,end in segments]
        def observe(frame,event,arg):
            if event=='call' and frame.f_code.co_name=='_run_compiler':
                r['observations'].append({'effective_options':dataclasses.asdict(frame.f_locals['options'])})
        def compile_preflight():
            bound=inspect.signature(kernel.fn).bind(*args).arguments
            runtime_args,constants=kernel.split_args(bound,get_annotations(kernel.fn))
            configured=dict(kernel.default_options)|options|{'verify_sync':True,'always_compile':True}
            codegen=kernel.extract_kwargs(kernel.codegen.options_cls,configured)
            compile_options=kernel.extract_kwargs(kernel.compiler.options_cls,configured)
            assert not configured,configured
            prereqs=CompilePrereqs({k:kernel.get_arg_type(v) for k,v in runtime_args.items()},constants,codegen,compile_options)
            return kernel._compile_kernel(prereqs)
        sys.setprofile(observe)
        compiled=compile_preflight()
        sys.setprofile(None)
        r['memory_consumed']=compiled.meta.memory_consumed
        r['binary_sha256']=hashlib.sha256(compiled.binary).hexdigest()
        ub=compiled.meta.memory_consumed.get('UB')
        if ub is None:
            r['ub_fit']='unknown';r['compile_passed']=True
            if a.phase!='compile':raise RuntimeError('No UB metadata: numerical/perf launch not allowed without accounting')
        else:
            r['ub_fit']=ub<=253952
        r['compile_passed']=True;save()
        if a.phase=='compile':
            r['completed']=True;save();print(json.dumps({'output':str(a.output),'UB':ub,'fit':r['ub_fit']}),flush=True);return
        if not r['ub_fit']:raise RuntimeError('Pruned UB overflow before Model execution')
        r['phase']='model_launch';save();start=time.monotonic()
        kernel[a.cores](*args,**options,verify_sync=True)
        r['diagnostic_wall_seconds']=time.monotonic()-start
        def compare(actual,values):
            return precision.compare_tensors(actual,torch.nn.functional.gelu(values.double(),approximate=mode),
                    dtype=a.dtype,cpu_output=torch.nn.functional.gelu(values,approximate=mode))
        r['checks']=[]
        for name,start,end in segments:
            c=compare(y[start:end],x[start:end]);r['checks'].append({'segment':name,'passed':c.passed,'details':c.to_dict()})
            if not c.passed:
                import numpy as np
                np.savez_compressed(a.output.parent/f'failure-{name}.npz',x=x[start:end].float().numpy(),actual=y[start:end].float().numpy())
        r['numerical_passed']=all(c['passed'] for c in r['checks'])
        if a.repeat>1:
            original=y.clone()
            r['cached_repeats']=[]
            for _ in range(a.repeat-1):
                y.fill_(float('nan'))
                kernel[a.cores](*args,**options)
                matched=bool(torch.all((y==original)|(torch.isnan(y)&torch.isnan(original))))
                passed=all(compare(y[start:end],x[start:end]).passed for _,start,end in segments)
                r['cached_repeats'].append({'same_output_including_nan':matched,'accuracy_passed':passed})
                r['numerical_passed'] &= matched and passed
        if a.phase=='tail':
            r['writes_outside_logical_size']=int((backing_y[n:]!=12345.).sum())
            r['outer_guard_untouched']=bool((backing_y[allocated:]==12345.).all())
        if a.phase=='perf':
            r['phase']='timing_warmup';save()
            kernel[a.cores](*args,**options)
            ticks=[];hashes=[];correct=[]
            for _ in range(3):
                y.fill_(float('nan'))
                before=runtime.current_tick();kernel[a.cores](*args,**options)
                ticks.append(runtime.current_tick()-before)
                hashes.append(hashlib.sha256(y.float().numpy().tobytes()).hexdigest())
                correct.append(compare(y,x).passed)
            r.update(warm_ticks=ticks,median_ticks=statistics.median(ticks),repeat_accuracy=correct,repeated_outputs_equal=len(set(hashes))==1,
                     performance_scope='CaModel ticks on aligned finite input; no NPU latency or CANNBench speedup')
        r['completed']=True;r['phase']='complete'
    except Exception:
        r['error']=traceback.format_exc()
    finally:
        sys.setprofile(None);save()
    print(json.dumps({k:r.get(k) for k in ['completed','numerical_passed','median_ticks','memory_consumed','writes_outside_logical_size','error']}),flush=True)


if __name__=='__main__':main()

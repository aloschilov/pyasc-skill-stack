#!/usr/bin/env python3
"""Explicit simulator-only Add comparison; no generation, hardware or global catalog."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_REVISION = 'a63d5261c7c8077a0081e867f43a0e884fec066b'
REFERENCE_PATH = 'examples/01_simd_cpp_api/00_introduction/01_add/add_tpipe_tque'
REFERENCE_SHA256 = '3fd854a223214eec19b03eebf4be3397c81358ebd526d07b17202a5fd522b654'
TASK = {'operator': 'Add', 'shape': [8, 2048], 'dtype': 'float32', 'blocks': 8}
VARIANTS = ('reference', 'candidate', 'fusion')
METRIC = 'simulator_launch_tick_delta'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str)+'\n')


def load(path):
    return json.loads(Path(path).read_text())


def command(argv, **kwargs):
    return subprocess.check_output(argv, text=True, timeout=60, **kwargs).strip()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def extract_reference(source):
    if hashlib.sha256(source.encode()).hexdigest() != REFERENCE_SHA256:
        raise ValueError('Official reference bytes changed')
    # Keep the complete kernel and copyright notice. Never execute its NPU main.
    return source.split('int32_t main(', 1)[0].replace(
        '#include "acl/acl.h"\n', '').replace('#include "data_utils.h"\n', '')


def summarize(intent, reports):
    """Fail closed on incomplete/unpaired numerics, protocols or attribution."""
    if set(reports) != set(VARIANTS):
        raise ValueError('Missing or duplicate variant')
    identities = []
    for variant in VARIANTS:
        r = reports[variant]
        if (r.get('status') != 'passed' or r.get('variant') != variant
                or r.get('task') != TASK or r.get('metric') != METRIC
                or r.get('input_hashes') != intent['input_hashes']):
            raise ValueError('Unqualified or misattributed variant: '+variant)
        identities.append(r['runtime_hashes'])
        hashes = [r['binary_sha256'], *r['runtime_hashes'].values(), *r['input_hashes'].values()]
        if ('@loaded_runtime_wrapper' not in r['runtime_hashes'] or not r['commands']
                or any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in hashes)):
            raise ValueError('Missing runtime/compiler evidence')
        for phase, count in [('warmup', intent['warmup']), ('samples', intent['samples'])]:
            if len(r[phase]) != count:
                raise ValueError('Missing measurements')
            for index, sample in enumerate(r[phase]):
                if (sample['index'] != index or sample['launches'] != 1 or sample['numerical_checks'] != 3
                        or type(sample['tick_before']) is not int or type(sample['tick_after']) is not int
                        or sample['tick_before'] < 0
                        or sample['ticks'] != sample['tick_after']-sample['tick_before']
                        or not positive(sample['ticks'])):
                    raise ValueError('Invalid per-launch sample')
        if r['control'] != {'status': 'rejected', 'launch_attempts': 1, 'launches': 0}:
            raise ValueError('Missing attributable launch-suppression rejection')
    if any(identity != identities[0] for identity in identities):
        raise ValueError('Runtime bytes differ between variants')
    a, b = reports['candidate'], reports['fusion']
    for key in ('requested_options', 'effective_options'):
        left, right = dict(a[key]), dict(b[key])
        if left.pop('vf_fusion') is not False or right.pop('vf_fusion') is not True or left != right:
            raise ValueError('Fusion comparison changes more than vf_fusion')
    medians = {n: statistics.median(s['ticks'] for s in r['samples']) for n, r in reports.items()}
    return {'status': 'passed', 'metric': METRIC, 'median_ticks': medians,
            'reference_over_candidate': medians['reference']/medians['candidate'],
            'candidate_over_fusion': medians['candidate']/medians['fusion'],
            'paired_sample_ratios': {
                'reference_over_candidate': [x['ticks']/y['ticks'] for x,y in zip(reports['reference']['samples'],a['samples'])],
                'candidate_over_fusion': [x['ticks']/y['ticks'] for x,y in zip(a['samples'],b['samples'])]},
            'scope': 'One fixed Add sample; descriptive simulator ratios, not NPU speedup or statistical significance'}


def run(args):
    destination = args.output.resolve()
    if any((p/'.git').exists() for p in (destination, *destination.parents)) or any(c in str(destination) for c in ':,\n'):
        raise ValueError('Use a new output path outside every checkout, without Docker mount delimiters')
    reference = args.reference.resolve()
    if command(['git', '-C', str(reference), 'rev-parse', 'HEAD']) != REFERENCE_REVISION:
        raise ValueError('Reference revision differs')
    source = reference/REFERENCE_PATH/'add_tpipe_tque.asc'
    fragment = extract_reference(source.read_text())
    pin = load(ROOT/'reference/images/ENV.json')
    image = pin['dev_image']['ref']
    inspected = json.loads(command(['docker', 'image', 'inspect', image]))[0]
    if image not in inspected['RepoDigests']:
        raise ValueError('Pinned simulator image is not available locally; pull it explicitly')
    destination.mkdir(parents=True, exist_ok=False)
    inputs = destination/'inputs'; inputs.mkdir()
    for original, name in [(args.candidate, 'candidate.py'), (source, 'reference.asc'),
                           (reference/'LICENSE', 'LICENSE'), (reference/REFERENCE_PATH/'README_en.md', 'reference-README.md'),
                           (Path(__file__), 'simulator_comparison.py'), (ROOT/'reference/images/ENV.json', 'ENV.json')]:
        shutil.copy2(original, inputs/name)
    (inputs/'reference.cpp').write_text(fragment)
    intent = {'schema_version': 1, 'task': TASK, 'metric': METRIC, 'warmup': args.warmup, 'samples': args.samples,
              'stack_commit': command(['git','-C',str(ROOT),'rev-parse','HEAD']),
              'stack_dirty': bool(command(['git','-C',str(ROOT),'status','--porcelain'])),
              'reference_repository': 'https://gitcode.com/cann/asc-devkit.git', 'reference_revision': REFERENCE_REVISION,
              'reference_path': REFERENCE_PATH, 'image': image, 'image_id': inspected['Id'],
              'architecture': inspected['Architecture'], 'pyasc_source': pin['source_sha'],
              'platform': 'Model/Ascend950PR_9599', 'hardware': 'not-run',
              'input_hashes': {p.name: sha(p) for p in sorted(inputs.iterdir())}}
    save(inputs/'intent.json', intent)
    name = 'pyasc-timing-'+uuid.uuid4().hex
    cmd = ['docker','run','--rm','--name',name,'--network','none','--user',f'{os.getuid()}:{os.getgid()}',
           '--ulimit','nofile=65536:65536','-v',str(inputs)+':/input:ro','-v',str(destination)+':/result',
           '-w','/result','-e','HOME=/tmp','-e','PYTHONPYCACHEPREFIX=/tmp/pycache',image,'bash','-c',
           'status=0; for variant in reference candidate fusion; do '
           'python3 -I /input/simulator_comparison.py probe --variant "$variant" || status=1; done; exit "$status"']
    save(destination/'command.json', cmd)
    started = time.monotonic()
    try:
        with (destination/'output.log').open('w') as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=360).returncode
        for filename, checksum in intent['input_hashes'].items():
            if sha(inputs/filename) != checksum: raise ValueError('Input changed: '+filename)
        if code: raise ValueError('Probe failed; inspect retained variant reports/output.log')
        result = summarize(intent, {n: load(destination/n/'report.json') for n in VARIANTS})
    except Exception:
        result = {'status': 'failed', 'reason': traceback.format_exc(), 'metric': METRIC}
    finally:
        subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30,check=False)
    result.update(seconds=time.monotonic()-started, intent=intent)
    save(destination/'result.json', result)
    save(destination/'SHA256SUMS.json', {str(p.relative_to(destination)):sha(p) for p in sorted(destination.rglob('*')) if p.is_file()})
    print(json.dumps(result,default=str))
    return 0 if result['status']=='passed' else 1


def probe(variant):
    # Each variant gets a fresh Python/Model process and private compile caches.
    directory = Path('/result')/variant; directory.mkdir(); os.chdir(directory)
    os.environ['PYASC_CACHE_DIR'] = str(directory/'cache')
    os.environ['PYASC_DUMP_PATH'] = str(directory/'dump')
    os.environ['PYASC_DRY_RUN'] = '0'
    intent = load('/input/intent.json')
    result = {'status':'failed','variant':variant,'task':TASK,'metric':METRIC,
              'input_hashes':intent['input_hashes']}
    try:
        import numpy as np
        import asc
        from asc.runtime import config
        from asc.runtime.compiler import Compiler, CompileOptions
        from asc.runtime.launcher import Launcher, LaunchOptions
        from asc.runtime.kernel_meta import CompiledKernel, KernelMeta, LaunchedKernel
        from asc._C import ir
        from asc.lib import runtime as rt
        from asc.lib.runtime import state
        config.set_platform('Model','Ascend950PR_9599',check=True)
        assert rt.is_model()
        rng = np.random.default_rng(0)
        x = rng.normal(size=TASK['shape']).astype(np.float32)
        y = rng.normal(size=TASK['shape']).astype(np.float32)
        saved_x, saved_y = x.copy(), y.copy(); expected = x+y
        out = np.full_like(x,np.nan)
        commands=[]; captured=[]; options=[]
        started=time.monotonic()
        if variant == 'reference':
            requested={'kernel_type':config.KernelType.AIV_ONLY,'auto_sync':False,'opt_level':3}
            compiler=Compiler(CompileOptions(**requested)); original=compiler._run_cmd
            def observe(cmd,kind):
                commands.append({'kind':kind,'argv':cmd}); return original(cmd,kind)
            compiler._run_cmd=observe
            compiler._gen_dst_kernel(str(directory),Path('/input/reference.cpp'),directory/'binary.o')
            symbols=command(['readelf','-Ws',str(directory/'binary.o')])
            Path('symbols.txt').write_text(symbols+'\n')
            entries={line.split()[-1] for line in symbols.splitlines() if 'GLOBAL' in line and 'UND' not in line and 'add_custom' in line}
            if len(entries)!=1: raise ValueError('Ambiguous reference entry symbol: '+repr(entries))
            entry=entries.pop()
            kernel=CompiledKernel(Path('binary.o').read_bytes(),KernelMeta(kernel_args=(ir.KernelArgument.Explicit,)*4))
            arguments=(x,y,out,np.uint32(x.size)); effective=dataclasses.asdict(compiler.options)
        else:
            spec=importlib.util.spec_from_file_location('selected_candidate','/input/candidate.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            fn=module.candidate
            original_launcher=fn.launcher
            class Capture(original_launcher):
                def run(self,kernel,function_name,user_args,*args,**kwargs):
                    captured.append((kernel,function_name,user_args))
            original_compiler=fn.compiler
            class Observe(original_compiler):
                def _run_cmd(self,cmd,kind):
                    commands.append({'kind':kind,'argv':cmd});return super()._run_cmd(cmd,kind)
                def run(self,*args,**kwargs):
                    kernel=super().run(*args,**kwargs);options.append(dataclasses.asdict(self.options));return kernel
            fn.launcher=Capture;fn.compiler=Observe
            requested={'vf_fusion':variant=='fusion','always_compile':True,'opt_level':3}
            fn[8](x,y,out,**requested)
            if len(captured)!=1 or len(options)!=1:raise ValueError('Expected one compilation and captured launch')
            kernel,entry,arguments=captured[0]; effective=options[0]
            Path('binary.o').write_bytes(kernel.binary)
            if kernel.meta.enable_debug: raise ValueError('Debug candidate cannot be used for timing')
        result.update(compile_seconds=time.monotonic()-started,binary_sha256=sha('binary.o'),entry=entry,
                      requested_options=json.loads(json.dumps(requested,default=str)),
                      effective_options=json.loads(json.dumps(effective,default=str)),commands=commands)
        # Same pinned wrapper/Model/compiler payload across all three independent processes.
        cann=Path(os.environ['ASCEND_HOME_PATH'])
        runtime_files=[p for p in Path(asc.__file__).parent.rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.so')]
        runtime_files += [p for p in cann.glob('*/simulator/dav_3510/lib/*.so*') if p.is_file()]
        runtime_files += [Path(c['argv'][0]) for c in commands]
        if not any('libmodel_top' in p.name for p in runtime_files):raise ValueError('Model library provenance missing')
        result['runtime_hashes']={str(p):sha(p) for p in sorted(runtime_files)}
        wrapper=Path(state.lib.lib._name)
        result['runtime_wrapper_path']=str(wrapper)
        result['runtime_hashes']['@loaded_runtime_wrapper']=sha(wrapper)
        handle=rt.register_device_binary_kernel(kernel.binary,rt.magic_elf_value(kernel.meta.core_type))
        function=rt.register_function(handle,entry,mode=0)
        launched=LaunchedKernel.from_compiled(kernel,function)
        launcher=Launcher(LaunchOptions(core_num=8))
        original_launch=rt.launch_kernel
        observations=[]; suppress=False
        def measured(*args,**kwargs):
            if suppress:
                observations.append({'launch_attempts':1,'launches':0});return
            before=rt.current_tick()
            original_launch(*args,**kwargs);rt.synchronize()
            after=rt.current_tick()
            observations.append({'tick_before':before,'tick_after':after,'ticks':after-before,'launches':1})
        rt.launch_kernel=measured
        try:
            for phase,count in [('warmup',intent['warmup']),('samples',intent['samples'])]:
                result[phase]=[]
                for index in range(count):
                    out.fill(np.nan);observations.clear()
                    launcher.run(launched,entry,arguments)
                    np.testing.assert_array_equal(out,expected)
                    np.testing.assert_array_equal(x,saved_x);np.testing.assert_array_equal(y,saved_y)
                    if len(observations)!=1:raise ValueError('Expected exactly one measured launch')
                    result[phase].append(dict(observations[0],index=index,numerical_checks=3))
            np.save('actual.npy',out);np.save('expected.npy',expected)
            suppress=True;out.fill(np.nan);observations.clear()
            launcher.run(launched,entry,arguments)
            if observations != [{'launch_attempts':1,'launches':0}]:raise ValueError('Control did not reach launch')
            try:np.testing.assert_array_equal(out,expected)
            except AssertionError:result['control']={'status':'rejected',**observations[0]}
            else:raise ValueError('Suppressed-launch control incorrectly passed')
        finally:
            rt.launch_kernel=original_launch;rt.unregister_device_binary_kernel(handle)
        result['status']='passed'
    except Exception:
        result.update(status='failed',reason=traceback.format_exc())
    save('report.json',result)
    print(json.dumps({'variant':variant,'status':result['status']}))
    return 0 if result['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('run');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--warmup',type=int,choices=range(1,6),default=1)
    p.add_argument('--samples',type=int,choices=range(3,26),default=5)
    p=sub.add_parser('probe');p.add_argument('--variant',choices=VARIANTS,required=True)
    args=parser.parse_args()
    return run(args) if args.action=='run' else probe(args.variant)


if __name__=='__main__':sys.exit(main())

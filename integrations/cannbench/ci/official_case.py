"""Check one original CANNBench case in the pinned Model image, offline."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[3]
BENCH_REV = '597a915ec49735ee7d53057554bde426ef8c1d96'
CASES = {'exp/1': 'level1/exp', 'softmax/1': 'level2/softmax', 'transpose/2': 'level3/transpose'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def verify(key, candidate, benchmark, output):
    if key not in CASES:
        raise ValueError('Supported cases: ' + ', '.join(CASES))
    if output.exists():
        raise ValueError('Choose a new results directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    pin = json.loads((ROOT / 'reference/images/ENV.json').read_text())
    report = dict(case=key, backend='Model', benchmark_revision=BENCH_REV,
                  image=pin['dev_image']['ref'], runtime_revision=pin['source_sha'],
                  platform='Ascend950PR_9599', status='failed', stage='inputs', phases={})
    try:
        frozen = output / 'candidate.py'
        shutil.copyfile(candidate, frozen)
        report['candidate_sha256'] = sha(frozen)
        # Export committed evaluator and task code only. Ignore working-tree overrides.
        archive = output / 'benchmark.tar'
        subprocess.run(['git', '-C', str(benchmark), 'archive', '--output', str(archive),
                        BENCH_REV, 'src/kernel_eval', 'tasks/' + CASES[key]], check=True)
        report['benchmark_archive_sha256'] = sha(archive)
        # Snapshot the checker. A local checkout can also contain credentials and user files.
        checker = output / 'checker'
        for rel in ('integrations/cannbench/ci/official_case.py', 'integrations/cannbench/workers/contracts.py',
                    'tests/tools/execute_kernel.py', 'tests/tools/numerical_checks.py', 'tests/tools/pytest_verify_kernel.py'):
            target = checker / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, target)
        report['checker_sha256'] = {str(p.relative_to(checker)): sha(p) for p in checker.rglob('*.py')}
        for mode in ('real', 'suppressed'):
            report['stage'] = mode
            phase_dir = output / mode
            phase_dir.mkdir()
            phase_start = time.monotonic()
            # The original archive, candidate and checker stay read-only in both phases.
            with (phase_dir / 'simulator.log').open('wb') as log:
                proc = subprocess.run([
                    'docker', 'run', '--rm', '--network', 'none', '--ulimit', 'nofile=65536:65536',
                    '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'HOME=/tmp',
                    '-e', 'PYASC_CACHE_DIR=/tmp/cache', '-e', 'PYASC_DUMP_PATH=/tmp/dumps', '-w', '/tmp',
                    '-v', f'{checker}:/stack:ro', '-v', f'{archive}:/bench.tar:ro',
                    '-v', f'{frozen}:/candidate.py:ro', '-v', f'{phase_dir}:/results',
                    report['image'], 'timeout', '-k', '5', '300', 'sh', '-ec',
                    'tar -xf /bench.tar && exec python3.11 /stack/integrations/cannbench/ci/official_case.py "$@"',
                    'probe', key, mode], stdout=log, stderr=subprocess.STDOUT)
            result = phase_dir / 'probe.json'
            if not result.exists():
                raise ValueError(f'{mode} stopped with exit {proc.returncode}; see {mode}/simulator.log')
            phase = json.loads(result.read_text())
            phase['elapsed_seconds'] = round(time.monotonic() - phase_start, 3)
            report['phases'][mode] = phase
            report['stage'] = mode + '/' + phase['stage']
            if proc.returncode or phase['status'] != 'passed':
                raise ValueError(phase.get('error', f'{mode} failed'))
            if phase['candidate_sha256'] != report['candidate_sha256']:
                raise ValueError('Candidate changed between checks')
        if sha(frozen) != report['candidate_sha256']:
            raise ValueError('Candidate hash changed')
        report.update(status='passed', stage='numerically_verified')
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report['error'] = str(exc)
    report['elapsed_seconds'] = round(time.monotonic() - start, 3)
    write(output / 'result.json', report)
    print(f"{key} | CaModel | {report['status']} | {report['stage']} | {output / 'result.json'}")
    if 'error' in report:
        print(report['error'], file=sys.stderr)
    return int(report['status'] != 'passed')


def probe(key, mode, result):
    import torch
    import asc
    import asc.runtime.config as config
    from asc.experimental import asctile
    sys.path[:0] = ['/tmp/src', str(ROOT / 'tests/tools'), str(ROOT / 'integrations/cannbench/workers')]
    from kernel_eval.data.case_loader import CaseLoader
    from kernel_eval.data.data_generator import DataGenerator
    from kernel_eval.data.golden_loader import GoldenLoader
    from kernel_eval.data.operator_loader import OperatorLoader
    from kernel_eval.eval.accuracy_eval import AccuracyEvaluator
    from kernel_eval.utils.param_builder import ParamBuilder
    from contracts import static_check
    from execute_kernel import _install_launch_spy

    op, number = key.split('/')
    task = CASES[key]
    source = Path('/candidate.py')
    result['candidate_sha256'] = sha(source)
    result['stage'] = 'static'
    problems = static_check(source, op, Path('/tmp/tasks') / task)
    if problems:
        raise ValueError('; '.join(problems))
    case, = [c for c in CaseLoader('/tmp/tasks').scan_by_rel_path(task) if c.case_id == int(number)]
    loader = GoldenLoader('/tmp/tasks')
    info = OperatorLoader('/tmp/tasks').get_operator(task)
    if loader.get_input_function(task) or len(info.outputs) != 1 or not info.outputs[0].compare:
        raise ValueError('Unsupported official input or output contract')
    golden = loader.get_golden_function(task)
    builder, oracle = ParamBuilder(), AccuracyEvaluator()
    config.set_platform(config.Backend.Model, config.Platform.Ascend950PR_9599)
    _, _, counter = _install_launch_spy(asc, mode == 'suppressed')
    package = types.ModuleType('checked_candidate')
    package.__path__ = []
    helper = types.ModuleType('checked_candidate._pyasc_runtime')
    helper.ensure_npu_platform = lambda: None  # The checker owns the Model platform.
    sys.modules.update({'checked_candidate': package, 'checked_candidate._pyasc_runtime': helper, 'asctile': asctile})
    spec = importlib.util.spec_from_file_location('checked_candidate.' + op, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Poison candidate allocations for the negative control; keep the oracle's Torch unchanged.
    class PoisonAllocations:
        def __getattr__(self, name):
            original = getattr(torch, name)
            if name in ('empty', 'empty_like'):
                return lambda *args, **kwargs: original(*args, **kwargs).fill_(float('nan'))
            return original

    if mode == 'suppressed' and getattr(module, 'torch', None) is torch:
        module.torch = PoisonAllocations()
    result.update(stage='numerics', checks=[], input_shapes=case.input_shapes, dtypes=case.dtypes,
                  attributes=case.attrs, task_hashes={f: sha(Path('/tmp/tasks') / task / f)
                  for f in ('cases.yaml', 'proto.yaml', 'golden.py')})
    for seed in (19, 23):
        xs = DataGenerator(seed=seed).generate_input_tensors_from_case(case.input_shapes, case.dtypes, case.value_ranges)
        originals = [x.clone() for x in xs]
        params = builder.build_call_params(golden, case, xs)
        native = golden(**params)
        fp64 = oracle.compute_golden_fp64(golden, xs, builder, case)
        before = counter['calls']
        result['stage'] = 'compile-and-launch'
        with torch.no_grad():
            output = getattr(module, op)(**params)
        result['stage'] = 'numerics'
        calls = counter['calls'] - before
        if not calls or output.shape != native.shape or output.dtype != native.dtype:
            raise ValueError('Missing launch or incorrect output shape/dtype')
        if not all(torch.equal(x, saved) for x, saved in zip(xs, originals)):
            raise ValueError('Candidate modified an input')
        verdict = oracle.evaluate(output, fp64, case.dtypes[0], custom_thresholds=info.precision_thresholds,
                                 cpu_output=native)
        result['checks'].append(dict(seed=seed, launcher_calls=calls, numerical_pass=verdict.passed,
                                     threshold=verdict.threshold, mere=str(verdict.mere), mare=str(verdict.mare)))
        if verdict.passed != (mode == 'real'):
            raise ValueError('Official numerical comparison failed' if mode == 'real' else 'Suppressed launch passed comparison')
    result.update(status='passed', stage='control-rejected' if mode == 'suppressed' else 'numerically_verified')


if __name__ == '__main__':
    report = {'status': 'failed', 'stage': 'setup'}
    try:
        probe(sys.argv[1], sys.argv[2], report)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write(Path('/results/probe.json'), report)

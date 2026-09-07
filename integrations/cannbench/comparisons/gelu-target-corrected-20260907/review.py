"""Read-only independent review; evidence delivered verbatim, no claimed tool reads."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import argparse

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
spec = importlib.util.spec_from_file_location('helper', ROOT.parent/'gelu-asctile-jit-20260907/run_workers.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
model = 'dashscope/qwen3.7-max'
parser = argparse.ArgumentParser()
parser.add_argument('--exact', action='store_true')
parser.add_argument('--remedy', action='store_true')
parser.add_argument('--fast', action='store_true')
args = parser.parse_args()
out = ROOT/('workers-fast' if args.fast else 'workers-remedy' if args.remedy else 'workers-exact' if args.exact else 'workers')/model.replace('/', '_')
out.mkdir(parents=True, exist_ok=True)
paths = [REPO/'skills/pyasc-cannbench-kernel/SKILL.md', ROOT/'target_kernel.py',
         ROOT/'promoted_kernel.py', ROOT/'check.py', ROOT/'evidence/summary.json', ROOT/'README.md']
prompt = ('Critically review this bounded corrected-target GeLU experiment. All files follow verbatim. '
          'No tools, execution, or edits. Challenge formula accuracy claims, oracle and sampling coverage, '
          'FP16 overflow inference, BF16 limitation, UB accounting, and performance comparability. '
          '18 compile configurations are NOT 20 official cases. The tanh target does NOT implement exact GeLU. '
          'CaModel ticks are NOT NPU latency or CANNBench speedup. Corrected coefficients and explicit FP32 '
          'casts are DIFFERENT variants. State concrete mandatory corrections versus optional further work. '
          'You are an independent reviewer, not the decision maker. Maximum 600 words.\n')
if args.fast:
    paths = [REPO/'skills/pyasc-cannbench-kernel/SKILL.md', ROOT/'exact_fast_kernel.py',
             ROOT/'check.py', ROOT/'evidence/exact-fast-qualification.json', ROOT/'EXACT_REMEDY.md']
    prompt = ('Critically review a bounded optimization of the previously validated high-level exact '
              'GeLU remedy: public Erf remains for x>=-3, and a six-level normal-tail Mills-ratio '
              'continued fraction handles x<-3. The old12-level variant switched at -2. The new '
              'CPU double tail relative error on100001 points of[3,14] is 2.430293858e-5. '
              'All evidence inline; no tools or claimed execution. Check mathematical scope, '
              'finite-depth switch discontinuity vs tolerance, compiled dense/random/boundary '
              'accuracy, repeated multicore tail behavior, and timing comparability. The new '
              'variant is intended only for FP32 exact; FP16/BF16 retain their previously passing '
              'simple public-Erf FP32-intermediate route. Do not claim full-case hardware or all '
              'JIT-matrix validation. Identify concrete corrections; maximum400 words.\n')
elif args.remedy:
    paths = [REPO/'skills/pyasc-cannbench-kernel/SKILL.md', ROOT/'exact_stable_kernel.py',
             ROOT/'check.py', ROOT/'evidence/exact-stable-qualification.json',
             ROOT/'evidence/exact-stable-f32-initial/result.json', ROOT/'evidence/tail-math.json',
             ROOT/'EXACT_REMEDY.md',
             Path('/home/aloschilov/workspace/ops-nn/activation/gelu_v2/op_kernel/arch35/gelu_v2_dag.h')]
    prompt = ('Independently review the new public-AscTile exact GeLU remedy. All evidence inline; '
              'no tools or claimed execution. It retains Erf for x>=-2 and evaluates the Laplace '
              'normal Mills ratio continued fraction to depth12 for x<-2, using only high-level '
              'tile math, with FP32 intermediates. This is not a copied ops-nn polynomial. '
              'Check the recurrence, FP32 arithmetic, NaN/Inf, finite limits, switch boundary, '
              'tile/where alignment and repeat evidence, oracle correctness, and limitations of '
              'sampled validation. ops-nn FP32 uses Vec::Erf whose installed definition selects '
              'SUBSECTION_POLYNOMIAL_APPROXIMATION; asctile.erf emits default PADE and lacks '
              'a public algorithm option. Do not assume the specialized ops-nn path was executed. '
              'Perf is still pending; no speedup claim or full20 hardware claim is made. '
              'Distinguish concrete blockers from optional further tests and source API limitations '
              'from proven compiler defects. Maximum600 words; you are not the decision maker.\n')
elif args.exact:
    paths = [REPO/'skills/pyasc-cannbench-kernel/SKILL.md', ROOT/'exact_kernel.py',
             ROOT/'check.py', ROOT/'evidence/exact-summary.json', ROOT/'EXACT_CHECK.md']
    prompt = ('Independently challenge this local exact GeLU qualification and decision not to spend '
              'a diagnostic submission credit on a known failing candidate. All files follow verbatim. '
              'No tools, execution, or edits. Check that the oracle uses approximate=none, that dtype '
              'promotion and early half scaling are tested, and that claims distinguish sampled tests '
              'from official full-shape runs, numerical API limitations from proven compiler defects, '
              'and compile success from accuracy. Do not assume exact means infinite precision. '
              'Identify concrete unsupported claims and required corrections. You are not the decision '
              'maker. Maximum 500 words.\n')
prompt += '\n'.join(f'FILE {p}\n```\n{p.read_text()}\n```\n' for p in paths)
(out/'prompt.txt').write_text(prompt)
env = helper.environment(out, paths, True)
config = json.loads(env['OPENCODE_CONFIG_CONTENT'])
config['permission'] = {'*': 'deny'}
env['OPENCODE_CONFIG_CONTENT'] = json.dumps(config)
with (out/'events.jsonl').open('w') as log, (out/'stderr.log').open('w') as err:
    proc = subprocess.Popen(['opencode', 'run', '--pure', '--format', 'json', '--model', model,
                             '--dir', str(out), prompt], cwd=out, env=env,
                            stdout=log, stderr=err, start_new_session=True)
    try:
        code = proc.wait(timeout=240)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        code = 124
trace = helper.trace(out/'events.jsonl', paths)
(out/'review.txt').write_text(trace.pop('response_text'))
result = dict(model=model, exit_code=code, input_delivery='full_inline_text; no tool execution',
              author='Codex main agent; model identifier unavailable to this harness',
              prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
              inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, **trace)
(out/'provenance.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'model': model, 'exit_code': code}))

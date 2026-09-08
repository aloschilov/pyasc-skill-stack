"""Bounded read-only OpenCode review, with complete inline source provenance."""
import hashlib
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
spec = importlib.util.spec_from_file_location('helper', ROOT.parent/'gelu-asctile-jit-20260907/run_workers.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
parser=argparse.ArgumentParser()
parser.add_argument('--revision',action='store_true')
args=parser.parse_args()
out = ROOT/('workers-submission-v2' if args.revision else 'workers-submission')
out.mkdir(exist_ok=True)
assert not (out/'provenance.json').exists(), 'Preserve completed review'
paths = [ROOT/'candidate/gelu.py', ROOT/'tune.py', ROOT/'check.py', REPO/'integrations/cannbench/tasks/gelu/cases.yaml']
if args.revision:
    paths += [ROOT/'REVIEW_ADJUDICATION.md',ROOT/'history/gelu_zero_length_dma.py']
prompt = ('Independently and critically review this new combined high-level AscTile GeLU module and '
          'the proposed upstream Erf API issue. All inputs follow verbatim, no tools or edits allowed. '
          'Do not claim you executed anything. Check tail partitioning, six dtype/mode routes, finite '
          'limits/NaN/Inf, FP32 continued fraction, corrected tanh coefficients, and issue evidence '
          'versus speculation. Final combined module validation is pending. Distinguish concrete '
          'blockers from optional improvements. The configured ops-nn Erf has NOT been measured. '
          'Propose corrections in <=500 words; coordinator decides independently.\n')
if args.revision:
    prompt += ('Revision focus: fixed local_tiles loop bound to avoid zero-length DMA on idle end tiles. '
               'The old guarded-output Model run passed values but logged illegal DMA burst lengths, '
               'so it was rejected. New source must skip those iterations. Check variable unroll tail '
               'and contiguous block partition. Full final package tests are underway. The previous '
               'review made incorrect claims about -Inf and unexecuted SciPy tests: see adjudication. '
               'Do not repeat those. Oracle requires GeLU(-Inf)=NaN. <=300 words.\n')
prompt = ('Review this matched geometry/reuse-allocation tuning experiment critically. Mathematics is frozen from a 20/20 correct high-level GeLU submission. Explain why UB fill alone is not a speed metric, check matched logical work, guard tails, declared versus active AIV blocks, reuse=2 synchronization risks, and timing validation gaps. Do not execute tools or claim measurements. Propose concrete blockers only. <=450 words.\n')
prompt += '\n'.join(f'FILE {p}\n{p.read_text()}\n' for p in paths)
(out/'prompt.txt').write_text(prompt)
env = helper.environment(out, paths, True)
config = json.loads(env['OPENCODE_CONFIG_CONTENT'])
config['permission'] = {'*':'deny'}
env['OPENCODE_CONFIG_CONTENT'] = json.dumps(config)
model = 'dashscope/qwen3.7-max'
with (out/'events.jsonl').open('w') as log, (out/'stderr.log').open('w') as err:
    proc = subprocess.Popen(['opencode','run','--pure','--format','json','--model',model,
                             '--dir',str(out),prompt],cwd=out,env=env,stdout=log,stderr=err,start_new_session=True)
    try:
        code = proc.wait(timeout=240)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        code = 124
trace = helper.trace(out/'events.jsonl',paths)
(out/'review.txt').write_text(trace.pop('response_text'))
(out/'provenance.json').write_text(json.dumps(dict(model=model,exit_code=code,
    input_delivery='full inline text; tools denied',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},**trace),indent=2)+'\n')
print(json.dumps({'model':model,'exit_code':code}))

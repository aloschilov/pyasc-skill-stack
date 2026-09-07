"""Bounded OpenCode reviews; workers may only edit their own output folder."""
import concurrent.futures
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
TASKS = {
    'math': ('dashscope/glm-5.2', 'Design fast numerically stable exact GeLU in pure pyasc/asctile (no embedded C++). Inspect CANN gelu_v2 approximation and CANNBench checker, propose implementation that preserves negative tails and BF16. Write candidate.py and findings.md in your output directory. Do not run NPU submissions or Docker. Explicitly distinguish tanh from exact mode.'),
    'pipeline': ('dashscope/qwen3.7-max', 'Inspect generated pyasc v2 compiler and GeLU iteration03. Diagnose concrete bottlenecks, correctness risks in overlapping tail stores, core grid, buffering, VF fusion. Design a pure pyasc candidate using smaller VF tiles or improved pipeline in candidate.py and findings.md in your output directory. Never submit remotely; no Docker. Read source, do not assume missing fused Gelu alone proves cause.'),
    'native': ('alibaba/qwen3.8-max', 'Inspect CANN 950 GeLU/V2 implementation and pyasc inline/inline_vf API. Assess an explicitly labelled pyasc-hosted AscendC alternative using fused primitive, including exact/tanh contract and dtype support. Write findings.md and optional candidate.py ONLY in your output folder. No remote submissions, Docker or dependency installs. Compare against pure pyasc and disclose embedded C++ if used.'),
}

def run(name, model, task):
    out = ROOT / 'workers' / name
    out.mkdir(parents=True, exist_ok=True)
    prompt = f'''You are an isolated performance research worker. Output directory: {out}.
Only write files there. Do not commit, push, edit other files, read credentials, or submit jobs.
Task: {task}
Sources: {REPO}/integrations/cannbench/comparisons/gelu-handwritten-deepdive-20260903 (candidates, results iteration03, local_validation),
/home/aloschilov/workspace/pyasc-fork (pinned v2 source), /home/aloschilov/workspace/ops-nn/activation/gelu_v2,
/usr/local/Ascend (installed CANN headers), {REPO}/integrations/cannbench/tasks/gelu.
Baseline iteration03 passes 20/20, speedup .4748 geomean; native fp16 exact .4095, FP32 erfc .3207, BF16 erfc .1863, lowlevel tanh .66-.76. 72 cores and tile13824. VF+reuse tanh previously timed out at 67M fp16. Aim toward >=1x without sacrificing correctness. Use source evidence, bounded search, no huge recursive logs. Finish within 10 minutes.'''
    with (out / 'events.jsonl').open('w') as stdout, (out / 'stderr.txt').open('w') as stderr:
        try:
            result = subprocess.run(['opencode', 'run', '--pure', '--format', 'json', '--model', model, '--dir', str(out), prompt], stdout=stdout, stderr=stderr, timeout=700)
            return {'worker': name, 'model': model, 'exit_code': result.returncode}
        except subprocess.TimeoutExpired:
            return {'worker': name, 'model': model, 'status': 'timeout'}

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, name, *spec) for name, spec in TASKS.items()]
        for f in concurrent.futures.as_completed(futures):
            print(json.dumps(f.result()), flush=True)

"""Publish only whitelisted results; never copy raw private service responses."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
old={int(r['case_id'].rsplit('_',1)[-1]):r for r in csv.DictReader((ROOT.parent/'gelu-target-corrected-20260907/case-results.csv').open())}
dispatch=json.loads((ROOT/'evidence/dispatch.json').read_text())
package=json.loads((ROOT/'evidence/package-x86.json').read_text()) if (ROOT/'evidence/package-x86.json').exists() else None
job_path=ROOT/'evidence/remote/job.json'
job=json.loads(job_path.read_text())['job'] if job_path.exists() else None
cases={int(c['case_id'].rsplit('_',1)[-1]):c for c in job['results']['operators'][0]['cases']} if job else {}
job_url=f"https://cannbench.com/workspace/jobs/{job['id']}" if job else None
rows=[]
source_lines=(ROOT/'candidate/gelu.py').read_text().splitlines()
anchors={mode:next(i for i,s in enumerate(source_lines,1) if needle in s)
         for mode,needle in [('tanh','if approximate:'),('none','out = (x * 0.5)')]}
for d in dispatch['cases']:
    i=d['case_id'];prior=old[i];c=cases.get(i,{})
    ub=None
    if package:
        specs=[s for s in package['specializations'] if s['constexprs']['tile_length']==f"ConstExpr[int]({json.loads(d['tile_shape'])[0]})"
               and s['constexprs']['approximate']==f"ConstExpr[bool]({d['mode']=='tanh'})"]
        assert specs and len({s['memory_consumed']['UB'] for s in specs})==1
        ub=specs[0]['memory_consumed']['UB']
    rows.append(d|dict(ub_bytes=ub,ub_budget_bytes=253952,ub_percent=100*ub/253952 if ub else None,
        previous_us=float(prior['elapsed_us']),elapsed_us=c.get('elapsed_us'),reference_us=c.get('baseline_perf_us'),
        speedup=c.get('speedup'),accuracy=c.get('accuracy',{}).get('passed'),
        implementation=f"candidate/gelu.py#L{anchors[d['mode']]}",job_url=job_url))
with (ROOT/'case-results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fmt(v):return 'pending' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
local=[]
for p in sorted((ROOT/'evidence').glob('perf-n*/result.json')):
    r=json.loads(p.read_text())
    if r.get('completed'):
        local.append(dict(name=p.parent.name,config=r['args'],valid=r.get('numerical_passed') and all(r.get('repeat_accuracy',[])) and not r.get('unexpected_errors'),
            ub_bytes=r.get('memory_consumed',{}).get('UB'),ticks=r.get('median_ticks'),samples=r.get('warm_ticks'),
            evidence=str(p.relative_to(ROOT))))
(ROOT/'local-summary.json').write_text(json.dumps(local,indent=2)+'\n')
lines=['# GeLU: matched tiling and reuse-allocation follow-up','',
    'Pinned pyasc v2: `adadd7d66ed0ee16d33d79487bf584899a26ef1e`. Mathematics is unchanged from the [20/20 corrected target run](../gelu-target-corrected-20260907/README.md); only high-level launch/tiling/JIT configuration is tuned. No compiler or low-level-kernel changes.','',
    f"Hardware: [{job['id']}]({job_url}), {job['status']}, {job['passed_cases']}/{job['total_cases']} correct." if job else 'Hardware: upload in progress; no evaluation job confirmed yet. Local qualification passed: 14/14 perf configurations, 12/12 numerical/tail checks, 20/20 official dispatch/compile cases. Do not interpret local ticks as new CANNBench results.', '',
    '## Per-case navigation and actual launch configuration','',
    '`tile_shape=[N]` is a one-dimensional tile of the flattened input, not a row of the original shape. `AIV launched/useful` distinguishes the launch count from the number of partitions containing data. Useful count is analytically reconstructed, not hardware occupancy telemetry. This vector kernel launches AIV blocks, not AIC matrix blocks. All entries explicitly state reuse_alloc. Static allocation is requested as None and resolves to enabled; insert_sync=True, opt_level=3, debug=False. UB is compiler-reported per-specialization storage; the available budget is 248 KiB (253952 B), not a target to fill artificially.','',
    '| Case | Shape | dtype | Mode | tile_shape | Unroll | AIV launched/useful | reuse_alloc | VF fusion | UB KiB / budget % | Previous µs | New µs | Reference µs | Speedup | Accuracy | Kernel |',
    '|---|---|---|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---|---|']
for r in rows:
    case=f"[{r['case_id']}]({job_url})" if job_url else str(r['case_id'])
    ub=f"{r['ub_bytes']/1024:g} / {r['ub_percent']:.1f}%" if r['ub_bytes'] else 'pending'
    lines.append(f"| {case} | {r['shape']} | {r['dtype']} | {r['mode']} | {r['tile_shape']} | {r['unroll']} | {r['launched_AIV_blocks']}/{r['useful_AIV_blocks']} | {r['reuse_alloc']} | {r['vf_fusion']} | {ub} | {fmt(r['previous_us'])} | {fmt(r['elapsed_us'])} | {fmt(r['reference_us'])} | {fmt(r['speedup'])} | {fmt(r['accuracy'])} | [source]({r['implementation']}) |")
lines+=['','Previous timings: [job_6589259af036](https://cannbench.com/workspace/jobs/job_6589259af036). Cross-run comparisons are observational, not a repeated controlled A/B. [CSV](case-results.csv), [host partition audit](evidence/dispatch.json), [package verification](evidence/package-x86.json).','',
    '## Matched local screen','',
    'One AIV block per timing probe. FP32 exact uses 8192 logical elements; other routes use 32768. Compare only identical dtype/mode/size. Three warmed Model samples, with output checks, are ticks, not hardware microseconds. Tail/padding overhead is included. The initial stopped 65536-element probes are excluded.','',
    '| dtype | Mode | Elements | Tile | Unroll | reuse_alloc | VF | UB KiB | Median ticks | Valid | Evidence |',
    '|---|---|---:|---:|---:|---:|---|---:|---:|---|---|']
for r in local:
    a=r['config']
    lines.append(f"| {a['dtype']} | {a['mode']} | {a['size']} | {a['tile']} | {a['unroll']} | {a['reuse']} | {a['vf']} | {r['ub_bytes']/1024:g} | {r['ticks']} | {r['valid']} | [result]({r['evidence']}) |")
lines+=['','## Interpretation and limits','',
    'Launch/partition finding: case8 ([1537,769], FP32 tanh, tile_shape=[15872]) launches 72 AIV blocks but only 38 receive data under the current contiguous assignment. Each partition reserves two tiles; maximum UB usage does not imply balanced work. A size-aware tile choice or balanced/cyclic tile assignment is a high-level follow-up, not necessarily a compiler-pass change. Fewer idle blocks alone also does not prove lower latency: the longest active partition and memory traffic matter. This submission deliberately isolates one geometry constant; it does not claim complete launch/partition optimization.','',
    'Selected configuration: FP32 exact changes tile_shape from [1024] to [5120], keeping reuse_alloc=1, unroll=1 and VF fusion enabled. On the matched 8192-element one-block screen this improves median ticks from 15619 to 14162 (1.1029×); static UB increases from 48 to 240 KiB. Mode2 at [5120] gives 14215 ticks, so it is not selected. Unroll2 at [5120] needs 320 KiB and was pruned before simulation. All other routes keep their previous geometry and reuse_alloc=1. [Selection evidence](evidence/selection-reasons.json).','',
    'For FP16 exact, [8192]/96 KiB takes 19255 ticks, [16384]/192 KiB takes 19210, and [20480]/240 KiB takes 22454 on 32768 logical elements. The aligned larger tile differs by only 0.23%; the maximal-UB tile pays padding cost. FP16 tanh [15872]/248 KiB takes 10436 ticks versus 7546 for [8192]/128 KiB. These small-input observations do not prove the ordering at every full hardware shape. A pragmatic 3% selection margin retains the proven configuration for near-ties; it is not a statistical confidence interval. BF16 inherits the same low-precision geometry and is independently numerically qualified, not independently perf-tuned in this screen.','',
    'UB is not a utilization target by itself. At identical FP32 exact tile1024, disabling reuse increases storage from 48 KiB to 136.125 KiB without increasing useful work. At tile5120, mode1 needs 240 KiB, mode2 needs 220 KiB. Conversely FP16 exact at tile8192 needs 96 KiB with mode1 but 128 KiB with mode2; FP16 tanh uses 128 versus 96 KiB. More occupied UB can represent redundant intermediates, not better tiling. The earlier cross-case table mixed mathematical routes, dtypes and sizes and therefore did not establish causation.','',
    '[Compile matrix](evidence/matrix-summary.json): 68 unique configurations, 50 within UB; overflow cases were not simulated. Two additional controls checked aligned FP16 exact tile16384 and FP32 exact tile5120/unroll2 (overflow). Fourteen matched perf configurations passed. Static allocation remains enabled and required synchronization remains on. [Review adjudication](REVIEW_ADJUDICATION.md) explains which OpenCode claims were accepted or rejected. Final local qualification samples all six dtype/mode routes with special values, numerical boundaries, guarded multi-core tails and cached repeated execution; it is not full-shape device execution. Three harness-import failures are retained in [history](evidence/history-import-shadowing/README.md) and excluded.','',
    'The authored compute-kernel AST is identical to the previous submission; only one host geometry constant changed. [Package manifest](evidence/package.json) pins source/runtime/wheel/archive hashes; QEMU verified the installed evaluator bytes and all 20 specializations/dispatch cases. Six compiled specializations belong to one authored GeLU kernel. The exact [integration gate](evidence/tools/local_compile_gate.py) and [source contract](evidence/tools/source_contract.py) are archived, and the verifier imports those snapshots rather than unpublished workspace changes. Scripts still require the pinned runtime/toolchain and official precision checker described by the previous report; this is not a standalone installer.','',
    'Transport note: the first HTTP upload stalled before reading the complete archive (6.68 MB of 25.90 MB). That incomplete connection was terminated, absence of a job and unchanged credits were reconciled, and the identical archive/tag was retried once using the public server address with HTTPS validation retained. No complete/ambiguous upload is retried automatically. This is one intended evaluation, not multiple benchmark submissions.','']
if job:
    valid=[r for r in rows if r['accuracy'] and r['speedup'] and r['speedup']>0]
    summary=dict(job_id=job['id'],job_url=job_url,status=job['status'],passed=job['passed_cases'],total=job['total_cases'],
        api_summary=job['results']['summary'],environment=job['results']['setup_info']['environment'],
        independent_geometric_mean=statistics.geometric_mean(r['speedup'] for r in valid) if valid else None,
        faster_than_previous=sum(r['elapsed_us']<r['previous_us'] for r in valid),
        source_sha256=dispatch['source_sha256'])
    (ROOT/'hardware-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines+=['## Hardware conclusion','',f"Independent geometric mean reference/new: {fmt(summary['independent_geometric_mean'])}×. Faster than previous in {summary['faster_than_previous']}/20 cases. API aggregates are retained separately in [hardware summary](hardware-summary.json); do not confuse its field name with the independently calculated geometric mean.",'']
(ROOT/'README.md').write_text('\n'.join(lines)+'\n')
print('Report updated; hardware present:',bool(job))

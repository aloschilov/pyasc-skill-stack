"""Publishable result whitelist and shape/dtype/mode implementation navigation."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import yaml

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence/remote'
job=json.loads((OUT/'job.json').read_text())['job']
definitions={c['case_id']:c for c in yaml.safe_load((ROOT.parents[3]/'integrations/cannbench/tasks/gelu/cases.yaml').read_text())['cases']}
old=json.loads((ROOT.parent/'gelu-upstream-diagnostics-20260907/evidence/remote/job.json').read_text())['job']
old_cases={c['case_id']:c for c in old['results']['operators'][0]['cases']}
historic=json.loads((ROOT.parent/'gelu-perf-20260907/remote_runs/job.json').read_text())
historic=historic.get('job',historic)
historic_cases={c['case_id']:c for c in historic.get('results',{}).get('operators',[{'cases':[]}])[0]['cases']}
rows=[]
source_lines=(ROOT/'candidate/gelu.py').read_text().splitlines()
source_lines_by_mode={'tanh':next(i for i,s in enumerate(source_lines,1) if 'if approximate:' in s),
                     'none':next(i for i,s in enumerate(source_lines,1) if 'out = (x * 0.5)' in s)}
for c in job['results']['operators'][0]['cases']:
    definition=definitions[int(c['case_id'].rsplit('_',1)[-1])]
    mode=definition['attrs']['approximate'];dtype=definition['dtype'][0]
    prior=old_cases.get(c['case_id'],{})
    legacy=historic_cases.get(c['case_id'],{})
    route='fp32-exact-tail' if mode=='none' and dtype=='float32' else 'corrected-tanh' if mode=='tanh' else 'promoted-exact-erf'
    rows.append(dict(case_id=c['case_id'],shape=json.dumps(definition['input_shape'][0]),dtype=dtype,approximate=mode,
        value_range=str(definition['value_range']),status=c['status'],accuracy_passed=c.get('accuracy',{}).get('passed'),
        elapsed_us=c.get('elapsed_us'),reference_us=c.get('baseline_perf_us'),speedup=c.get('speedup'),
        previous_high_level_us=prior.get('elapsed_us'),previous_high_level_passed=prior.get('accuracy',{}).get('passed'),
        historical_low_level_us=legacy.get('elapsed_us'),route=route,
        tile=1024 if route=='fp32-exact-tail' else 15872 if dtype=='float32' else 8192,
        unroll=1 if route=='fp32-exact-tail' else 2,vf_fusion=route=='fp32-exact-tail',
        implementation=f'candidate/gelu.py#L{source_lines_by_mode[mode]}',
        job_url=f"https://cannbench.com/workspace/jobs/{job['id']}"))
positive=[r['speedup'] for r in rows if r['status']=='success' and r['speedup'] and r['speedup']>0]
matched=[r for r in rows if r['accuracy_passed'] and r['previous_high_level_passed'] and r['elapsed_us'] and r['previous_high_level_us']]
summary=dict(job_id=job['id'],job_url=f"https://cannbench.com/workspace/jobs/{job['id']}",status=job['status'],
    passed=job['passed_cases'],total=job['total_cases'],score=job['result_score'],
    environment=job['results']['setup_info']['environment'],api_summary=job['results']['summary'],
    independent_geometric_mean=statistics.geometric_mean(positive) if positive else None,
    independent_arithmetic_mean=statistics.mean(positive) if positive else None,
    previous_high_level_job=old['id'],previous_high_level_passed=old['passed_cases'],
    matched_passing_cases=len(matched),faster_than_previous_high_level=sum(r['elapsed_us']<r['previous_high_level_us'] for r in matched),
    matched_latency_geomean_improvement=statistics.geometric_mean([r['previous_high_level_us']/r['elapsed_us'] for r in matched]) if matched else None,
    historical_low_level_job=historic['id'],
    hardware_success_for_publication=job['passed_cases']==20 and any(r['elapsed_us']<r['previous_high_level_us'] for r in matched),
    raw_job_sha256=hashlib.sha256((OUT/'job.json').read_bytes()).hexdigest(),
    metric_note='Independent means use positive passing-case speedup ratios. Compare API field names to those actual means; do not assume its geometric_mean_speedup name is accurate. Cross-run timing is observational, not a controlled A/B rerun.')
summary['by_mode']={mode:{'cases':len(group),'passed':sum(bool(r['accuracy_passed']) for r in group),
    'independent_geometric_mean':statistics.geometric_mean(r['speedup'] for r in group if r['speedup'] and r['status']=='success')}
    for mode in ['none','tanh'] if (group:=[r for r in rows if r['approximate']==mode])}
legacy_matched=[r for r in rows if r['elapsed_us'] and r['historical_low_level_us']]
summary['historical_low_level_comparison']={'matched_cases':len(legacy_matched),
    'faster_cases':sum(r['elapsed_us']<r['historical_low_level_us'] for r in legacy_matched),
    'geomean_old_over_new_latency':statistics.geometric_mean(r['historical_low_level_us']/r['elapsed_us'] for r in legacy_matched)}
summary['at_least_reference_speed_cases']=[r['case_id'] for r in rows if r['accuracy_passed'] and r['speedup']>=1]
(ROOT/'hardware-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (ROOT/'case-results.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def fmt(v): return '—' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
lines=['# GeLU hardware results','',f"[CANNBench job {job['id']}]({summary['job_url']}) — {job['passed_cases']}/{job['total_cases']} correct; status `{job['status']}`.",'',
    'All times below are hardware microseconds. Speedup = official reference / candidate; ≥1 means at least as fast as the reference. Links may require CANNBench sign-in.', '',
    '| Case | Shape | dtype | Mode | Candidate µs | Reference µs | Speedup | Accuracy | Implementation |',
    '|---|---|---|---|---:|---:|---:|---|---|']
for r in rows:
    lines.append(f"| [{r['case_id']}]({r['job_url']}) | {r['shape']} | {r['dtype']} | {r['approximate']} | {fmt(r['elapsed_us'])} | {fmt(r['reference_us'])} | {fmt(r['speedup'])}× | {r['accuracy_passed']} | [{r['route']}]({r['implementation']}) |")
lines+=['','See [CSV](case-results.csv) for previous high-level and historical low-level latencies and geometry, and [sanitized summary](hardware-summary.json) for environment and exact aggregates.',
    '',f"Previous high-level job: [{old['id']}](https://cannbench.com/workspace/jobs/{old['id']}); {len(matched)} common passing cases, {summary['faster_than_previous_high_level']} faster, geometric mean of old/new latency ratios {fmt(summary['matched_latency_geomean_improvement'])}×.",
    '',f"Historical low-level comparison: [{historic['id']}](https://cannbench.com/workspace/jobs/{historic['id']}). This is not the same authored API/style or a controlled matched rerun.",
    '',f"Independent geometric mean over {len(positive)} positive passing-case reference/candidate ratios: {fmt(summary['independent_geometric_mean'])}×; arithmetic mean: {fmt(summary['independent_arithmetic_mean'])}×. API aggregates are preserved verbatim in the sanitized summary."]
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))

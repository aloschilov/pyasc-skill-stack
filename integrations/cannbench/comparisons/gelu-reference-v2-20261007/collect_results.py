"""GET-only collection and independent, all-case result accounting."""
import csv
import json
import math
import time
from pathlib import Path

from submit import ReadOnlyClient, ROOT, TERMINAL, TAG, save


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    artifact = ROOT/'repair' if args.repair else ROOT
    tag = TAG+'-buildfix' if args.repair else TAG
    state = json.loads((artifact/'remote/state.json').read_text())
    client = ReadOnlyClient().client
    client.timeout, client.retries = 45, 1
    payload = client.get_job(state['job_id'])
    job = payload.get('job', payload)
    assert job['job_tag'] == tag
    save(artifact/'remote/job.json', payload)
    state.update(state=job['status'], observed=time.time())
    save(artifact/'remote/state.json', state)
    if not job.get('job_completed') and job['status'] not in TERMINAL:
        print(json.dumps({k: job.get(k) for k in ('id', 'status', 'stage_summary')}))
        return
    save(artifact/'remote/logs.json', client.get_job_logs(job['id']))
    save(artifact/'remote/credits-after.json', client.get_credits())
    results = job.get('results') or {}
    cases = [case for operator in results.get('operators', []) for case in operator.get('cases', [])]
    by_id = {}
    for case in cases:
        case_id = int(str(case['case_id']).rsplit('_', 1)[-1])
        assert case_id not in by_id
        by_id[case_id] = case
    official = yaml.safe_load((ROOT/'inputs/cases.yaml').read_text())['cases']
    rows = []
    ratios = []
    for case in official:
        case_id = case['case_id']
        result = by_id.get(case_id, {})
        elapsed, baseline = result.get('elapsed_us'), result.get('baseline_perf_us')
        ratio = baseline / elapsed if isinstance(baseline, (float, int)) and isinstance(elapsed, (float, int)) and baseline > 0 and elapsed > 0 else None
        accuracy = (result.get('accuracy') or {}).get('passed')
        passed = result.get('status') == 'success' and accuracy is True
        if passed and ratio is not None:
            ratios.append(ratio)
        rows.append(dict(case_id=case_id, shape='x'.join(map(str, case['input_shape'][0])),
            dtype=case['dtype'][0], mode=case['attrs']['approximate'], tile_shape='[15872]',
            unroll=1, reuse_alloc=2, vf_fusion=True, compiled_ub_bytes=190464,
            status=result.get('status', 'no_case_result'), accuracy=accuracy,
            elapsed_us=elapsed, baseline_perf_us=baseline, derived_speedup=ratio,
            reported_speedup=result.get('speedup'), error=result.get('error_msg'),
            kernel='kernel.py', job_url='https://cannbench.com/workspace/jobs/'+job['id']))
    all20 = len(ratios) == len(official) == 20
    summary = dict(job_id=job['id'], status=job['status'], total_expected=20,
        received_cases=len(cases), valid_successful_ratios=len(ratios),
        gm_all20=math.exp(sum(map(math.log, ratios))/20) if all20 else None,
        faster_than_reference=sum(row['derived_speedup'] is not None and row['derived_speedup'] > 1 and row['accuracy'] is True and row['status'] == 'success' for row in rows),
        server_summary=results.get('summary'), environment=(results.get('setup_info') or {}).get('environment'),
        runner_id=job.get('runner_id'), runner_name=job.get('runner_name'),
        error_code=job.get('error_code'), error_message=job.get('error_message'),
        scope='One diagnostic; no controlled A/B or reproducibility claim')
    save(artifact/'hardware-summary.json', summary)
    with (artifact/'hardware-cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

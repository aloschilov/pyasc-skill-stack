"""Compare measured cases, keeping official aggregates and baseline anchors distinct."""
import csv
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parent

def main():
    old=json.loads((ROOT.parent/'gelu-handwritten-deepdive-20260903/remote_runs/iteration-03-lowlevel-tanh-safe-tile13824/results.json').read_text())
    new=json.loads((ROOT/'remote_runs/results.json').read_text())
    cases=yaml.safe_load((ROOT.parents[1]/'tasks/gelu/cases.yaml').read_text())['cases']
    configs={c['case_id']:c for c in cases}
    reference={c['case_id']:c for c in old['operators'][0]['cases']}
    rows=[]
    for c in new['operators'][0]['cases']:
        b=reference[c['case_id']]
        cfg=configs[int(c['case_id'].rsplit('_',1)[1])]
        rows.append(dict(case_id=c['case_id'],shape=str(cfg['input_shape'][0]),dtype=cfg['dtype'][0],
             mode=cfg['attrs']['approximate'],old_us=b.get('elapsed_us'),new_us=c.get('elapsed_us'),
             old_speedup=b.get('speedup'),new_speedup=c.get('speedup'),
             old_baseline_us=b.get('baseline_perf_us'),new_baseline_us=c.get('baseline_perf_us'),
             relative_to_iteration03=b['elapsed_us']/c['elapsed_us'] if c.get('elapsed_us') and b.get('elapsed_us') else None,
             status=c['status'],accuracy=c.get('accuracy',{}).get('passed')))
    with (ROOT/'hardware-comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(reference_job='job_a375a6e244ca',candidate_job='job_7b4caccdc21f',
                old_summary=old['summary'],new_summary=new['summary'],
                cases_at_least_1x=sum((r['new_speedup'] or 0)>=1 for r in rows),
                cases_faster_than_iteration03=sum((r['relative_to_iteration03'] or 0)>1 for r in rows),
                baseline_anchors_changed=any(r['old_baseline_us']!=r['new_baseline_us'] for r in rows))
    (ROOT/'hardware-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

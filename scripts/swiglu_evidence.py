"""Project the retained SwiGLU device study alongside the independent service runs."""
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import geometric_mean, median

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'docs/evidence/swiglu-vector-grid-20261110'
RELEASE = 'https://github.com/aloschilov/pyasc-skill-stack/releases'
CASES = set(range(1, 21))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def times(rows, service=False):
    result = {}
    for row in rows:
        number = int(row['id'].removeprefix('level1/swi_glu_')) if service else row['case']
        value = row['elapsed_us'] if service else row['median_us']
        require(row['correct'] is True and number not in result and type(value) in (int,float)
                and math.isfinite(value) and value > 0, 'Invalid official timing')
        if service:
            require(row['kernels'] and row['id'] == f'level1/swi_glu_{number}', 'Missing device kernel')
        else:
            samples = row['samples_us']
            require(len(samples) == 3 and all(math.isfinite(v) and v > 0 for v in samples)
                    and math.isclose(median(samples),value), 'Invalid device samples')
        result[number] = value
    require(result.keys() == CASES, 'Incomplete case set')
    return result


def project(fetch, pin):
    tag, asset = pin['release'], pin['asset']
    require(all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', v) for v in (tag, asset)), 'Invalid release asset')
    source = fetch(f'{RELEASE}/download/{tag}/{asset}')
    require(hashlib.sha256(source).hexdigest() == pin['sha256'], 'Service study digest differs')
    studies = json.loads(source)['jobs']
    require(set(studies) == {'wide','parent','reference'}, 'Missing service arm')
    for job in studies.values():
        require(job['benchmark_version'] == '2.0.0' and job['target_hardware'] == '950pr'
                and job['anti_cheat_failed'] == 0 and math.isfinite(job['result_score']), 'Incompatible service cohort')
    service = {k:times(v['cases'], True) for k,v in studies.items()}
    shape, archive = (json.loads((REPORT/path).read_bytes()) for path in
                      ('shape/measurements.json','native/archive-measurements.json'))
    require(hashlib.sha256((REPORT/'shape/swi_glu.py').read_bytes()).hexdigest() == shape['source_sha256'], 'Shape candidate changed')
    require(shape['python'] == archive['python'] and shape['pyasc_extension_sha256'] == archive['extension_sha256']
            and shape['device'] == archive['target'] and shape['CANN'] == archive['cann'], 'Replay runtime differs')
    new, old = times(shape['cases']), times(archive['cases'])
    by_case = {r['case']:r for r in archive['cases']}
    require(all(all(r[k] == by_case[r['case']][k] for k in ('shape','dtype','dim')) for r in shape['cases']),
            'Replay workload differs')
    names = {r['case']:{task[0] for task in r['tasks']} for r in shape['cases']}
    generated = {i for i,k in names.items() if len(k)==1 and k <= {'swiglu_rows','swiglu_flat','swiglu_slab'}}
    require(generated == CASES-{12} and names[12]=={'_swiglu_pair_groups'}, 'Unexpected local compute attribution')
    with (REPORT/'comparison.csv').open() as stream:
        paired = [row for row in csv.DictReader(stream) if row['case']!='GM']
    require(len(paired)==20 and {int(r['case']) for r in paired}==CASES
            and all(math.isclose(float(r['shape_us']),new[int(r['case'])]) for r in paired), 'Replay comparison changed')
    baselines = {int(r['case']):float(r['cannbench_us']) for r in paired}
    require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in baselines.values()), 'Invalid service baseline')
    return {'status':'available','local_generated_ratio':geometric_mean(baselines[i]/new[i] for i in generated),
            'cases':[{'case':i, **{k:v[i] for k,v in service.items()},
                'shape_us':new[i], 'replay_us':old[i]} for i in sorted(CASES)],
            'scores':{k:v['result_score'] for k,v in studies.items()},
            'release':f'{RELEASE}/tag/{tag}', 'archive_over_shape':geometric_mean(old[i]/new[i] for i in CASES),
            'parent_over_wide':geometric_mean(service['parent'][i]/service['wide'][i] for i in (4,6,8,16,17,19)),
            'shape_evidence':'https://github.com/aloschilov/pyasc-skill-stack/blob/3a1499363b10fb3c4ef68d82d3c2833f8c63677e/docs/evidence/swiglu-vector-grid-20261110/README.md',
            'extension_sha256':shape['pyasc_extension_sha256']}

def service_projection(fetch, pin):
    tag = pin['release']
    match = re.fullmatch(r'qualification-swiglu-performance-([1-9][0-9]+)', tag)
    require(match and re.fullmatch(r'[0-9a-f]{64}',pin['sha256']), 'Invalid performance publication')
    raw = fetch(f'{RELEASE}/download/{tag}/performance.json')
    require(hashlib.sha256(raw).hexdigest()==pin['sha256'], 'Performance receipt hash differs')
    result = json.loads(raw)
    evaluation = match.group(1)
    origin = str(result['origin_run'])
    require(result['schema']==1 and re.fullmatch(r'[1-9][0-9]+',origin) and str(result['evaluation_run'])==evaluation
            and result['hardware']=='950pr' and result['benchmark_version']=='2.0.0'
            and re.fullmatch(r'job_[0-9a-f]+',result['job_id']) and re.fullmatch(r'sub_[0-9a-f]+',result['submission_id'])
            and all(re.fullmatch(r'[0-9a-f]{64}',result[k]) for k in ['candidate_sha256','package_sha256']),
            'Unexpected generation or submission identity')
    cases = result['cases']
    require(len(cases)==20 and {c['id'] for c in cases}=={f'level1/swi_glu_{n}' for n in CASES},
            'Missing or duplicate service case')
    for c in cases:
        require(c['kernel'] in {'rows','flat','slab','_swiglu_pair_groups'}
                and (c['id']=='level1/swi_glu_12')==(c['kernel']=='_swiglu_pair_groups')
                and all(type(c[k]) in (int,float) and math.isfinite(c[k]) and c[k]>0
                        for k in ['elapsed_us','baseline_us']), 'Invalid measured kernel or timing')
        require(all(type(c[k]) in (int,float) and math.isfinite(c[k]) and c[k]>=0
                    for k in ('perf_score','t_hw_us') if k in c), 'Invalid service case metric')
    ratio = geometric_mean(c['baseline_us']/c['elapsed_us'] for c in cases if c['kernel']!='_swiglu_pair_groups')
    require(math.isclose(ratio,result['generated_ratio'],rel_tol=1e-10)
            and math.isfinite(result['service_score']), 'Service aggregate differs from per-case times')
    require(re.fullmatch(r'[a-f0-9]{40}', result['revision'])
            and re.fullmatch(r'[A-Za-z0-9./_-]{1,100}', result['model']), 'Invalid generation attribution')
    finished = datetime.fromisoformat(result['finished_at'].replace('Z','+00:00'))
    finished = (finished if finished.tzinfo else finished.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    # Publish only fields with a checked contract, even if the service adds metadata.
    skill_revision=result.get('skills_revision',result['revision'])
    require(re.fullmatch(r'[0-9a-f]{40}',skill_revision),'Invalid skill revision')
    if 'policy_sha256' in result:
        require(re.fullmatch(r'[a-f0-9]{64}',result['policy_sha256']), 'Invalid model policy identity')
    public_cases = [{k:c[k] for k in ('id','kernel','baseline_us','elapsed_us','perf_score','t_hw_us') if k in c}
                    for c in sorted(cases,key=lambda c:int(c['id'].split('_')[-1]))]
    return {'status':'available','ratio':ratio,'score':result['service_score'],'job':result['job_id'],
            'submission':result['submission_id'],'generation':RELEASE.removesuffix('/releases')+'/actions/runs/'+origin,
            'all_ratio':geometric_mean(c['baseline_us']/c['elapsed_us'] for c in cases),
            'finished_at':finished.isoformat(),'model':result['model'],'revision':result['revision'],
            'skills_revision':skill_revision, 'policy_sha256':result.get('policy_sha256'),
            'candidate_sha256':result['candidate_sha256'], 'release':f'{RELEASE}/tag/{tag}', 'cases':public_cases}


def service_history(fetch, pinned, discover):
    """Retain comparable service receipts; a broken publication cannot erase earlier measurements."""
    pins = {pinned['release']:pinned} if pinned else {}
    complete = True
    errors = (OSError, ValueError, LookupError, TypeError, ArithmeticError, AttributeError)
    if discover:
        page = 1
        try:
            while True:
                releases = json.loads(fetch(f'https://api.github.com/repos/aloschilov/pyasc-skill-stack/releases?per_page=100&page={page}'))
                require(isinstance(releases,list), 'Invalid releases response')
                for release in releases:
                    tag = release['tag_name']
                    if release['draft'] or release['prerelease'] or not re.fullmatch(r'qualification-swiglu-performance-[1-9][0-9]+',tag):
                        continue
                    try:
                        asset, = [a for a in release['assets'] if a['name']=='performance.json']
                        require(0 < asset['size'] < 128*1024 and re.fullmatch(r'sha256:[a-f0-9]{64}',asset['digest']), 'Invalid performance asset')
                        pin = {'release':tag, 'sha256':asset['digest'][7:]}
                        if tag in pins:
                            require(pins[tag]['sha256']==pin['sha256'], 'Reviewed performance pin differs')
                        pins[tag] = pin
                    except errors:
                        complete = False
                if len(releases)<100:
                    break
                page += 1
        except errors:
            complete = False
    runs = {}
    for pin in pins.values():
        try:
            record = service_projection(fetch,pin)
            old = runs.setdefault(record['job'],record)
            require(old==record,'Conflicting service job identity')
        except errors:
            complete = False
    return {'complete':complete,'runs':sorted(runs.values(),key=lambda r:r['finished_at'],reverse=True)}

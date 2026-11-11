"""Project the retained SwiGLU device study alongside the independent service runs."""
import hashlib
import json
import math
import re
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
    return {'status':'available','cases':[{'case':i, **{k:v[i] for k,v in service.items()},
                'shape_us':new[i], 'replay_us':old[i]} for i in sorted(CASES)],
            'scores':{k:v['result_score'] for k,v in studies.items()},
            'release':f'{RELEASE}/tag/{tag}', 'archive_over_shape':geometric_mean(old[i]/new[i] for i in CASES),
            'parent_over_wide':geometric_mean(service['parent'][i]/service['wide'][i] for i in (4,6,8,16,17,19)),
            'shape_evidence':'https://github.com/aloschilov/pyasc-skill-stack/blob/3a1499363b10fb3c4ef68d82d3c2833f8c63677e/docs/evidence/swiglu-vector-grid-20261110/README.md',
            'extension_sha256':shape['pyasc_extension_sha256']}

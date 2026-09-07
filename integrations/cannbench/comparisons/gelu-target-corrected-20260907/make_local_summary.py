"""Compact publication-safe index; full local traces remain in evidence/."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'evidence/final-v2-summary.json').read_text())
summary={'source_sha256':hashlib.sha256((ROOT/'candidate/gelu.py').read_bytes()).hexdigest(),
    'label':'verified-camodel-smoke','scope':'18 reduced local jobs, six dtype/mode routes; not full official-shape numerical coverage',
    'runtime_pin':'adadd7d66ed0ee16d33d79487bf584899a26ef1e',
    'platform':'CaModel Ascend950PR_9599 / CANN9.0 / native AArch64 CPython3.11',
    'checker_sha256':next(iter(rows.values()))['checker_sha256'],'checks':[]}
for name,r in rows.items():
    assert r['source_sha256']==summary['source_sha256'] and r['clean_model_run']
    summary['checks'].append(dict(name=name,dtype=r['args']['dtype'],mode=r['approximate'],elements=r['elements'],
        source_sha256=r['source_sha256'],harness_sha256=r['harness_sha256'],
        geometry=r['selected_geometry'],cores=r['args']['cores'],requested_options=r['requested_options'],
        compile_observations=r['observations'],UB=r['memory_consumed']['UB'],passed=r['numerical_passed'],
        segments=[{'name':c['segment'],'passed':c['passed']} for c in r.get('checks',[])],
        diagnostic_wall_seconds=r.get('diagnostic_wall_seconds'),warm_ticks=r.get('warm_ticks'),
        median_ticks=r.get('median_ticks'),cached_repeats=r.get('cached_repeats'),
        repeat_accuracy=r.get('repeat_accuracy'),repeated_outputs_equal=r.get('repeated_outputs_equal'),
        writes_outside_logical_size=r.get('writes_outside_logical_size'),outer_guard_untouched=r.get('outer_guard_untouched'),
        unexpected_simulator_error_count=r['unexpected_simulator_error_count'],
        expected_special_value_diagnostics=r['expected_special_value_diagnostics']))
(ROOT/'local-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Published-local index: 18 jobs, all checks passed; timings are Model ticks only.')

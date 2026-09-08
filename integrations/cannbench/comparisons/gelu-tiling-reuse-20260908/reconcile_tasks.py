"""Read-only reconciliation of the current official GeLU task and local fixtures."""
import hashlib
import json
import math
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
sys.path.insert(0,str(REPO/'integrations/cannbench/workers'))
from evalqueue import EvalQueue
task=EvalQueue()._client.get_benchmark_task('official-tasks','Gelu')
folder=REPO/'integrations/cannbench/tasks/gelu'
local=yaml.safe_load((folder/'cases.yaml').read_text())['cases']
remote=task['cases_yaml']
assert task['case_count']==len(local)==len(remote)==20
by_id={c['case_id']:c for c in remote}
encoding=[]
for c in local:
    r=by_id[c['case_id']]
    for k in ['operator','case_id','input_shape','dtype','attrs','note']:
        assert c[k]==r[k],(c['case_id'],k)
    if any(not math.isfinite(v) for v in c['value_range']):
        assert r['value_range']==[None,None]
        encoding.append(c['case_id'])
    else:assert c['value_range']==r['value_range']
assert task['golden_source']==(folder/'golden.py').read_text()
result={'status':'passed','official_case_count':20,'shape_dtype_mode_notes_all_match':True,
    'golden_source_identical':True,'golden_sha256':hashlib.sha256(task['golden_source'].encode()).hexdigest(),
    'api_null_nonfinite_range_cases':encoding,
    'note':'API serializes Inf/NaN bounds as null; local YAML preserves them. Task API name is case-sensitive Gelu.'}
(ROOT/'evidence/task-reconciliation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

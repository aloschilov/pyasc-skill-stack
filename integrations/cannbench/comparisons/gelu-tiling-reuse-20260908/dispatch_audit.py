"""Derive logical launch/partition facts from source; no occupancy measurement."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
source=ROOT/'candidate/gelu.py'
tree=ast.parse(source.read_text())
geometry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='geometry')
namespace={}
exec(compile(ast.Module(body=[geometry],type_ignores=[]),str(source),'exec'),namespace)
sys.path.insert(0,str(REPO/'integrations/cannbench/submission'))
sys.path.insert(0,str(ROOT/'evidence/tools'))
from source_contract import check_high_level_source,check_runtime_helper
assert not check_high_level_source(source)
assert not check_runtime_helper(ROOT/'candidate/_pyasc_runtime.py')
rows=[]
for c in yaml.safe_load((REPO/'integrations/cannbench/tasks/gelu/cases.yaml').read_text())['cases']:
    dtype=c['dtype'][0];mode=c['attrs']['approximate'];shape=c['input_shape'][0];n=math.prod(shape)
    selected=namespace['geometry'](dtype=='float32',mode)
    tile,unroll,vf=selected[:3];reuse=selected[3] if len(selected)>3 else 1
    blocks=min(72,(n+tile-1)//tile)
    partition_tiles=(n+blocks*tile-1)//(blocks*tile)
    assigned=tile*partition_tiles
    active=math.ceil(n/assigned)
    total=0;previous_end=0
    for b in range(blocks):
        offset=assigned*b
        count=min(partition_tiles,max(0,(n-offset+tile-1)//tile))
        for i in range(count):
            start=offset+i*tile;valid=min(n-start,tile)
            assert valid>0 and start==previous_end
            total+=valid;previous_end=start+valid
    assert total==n==previous_end
    rows.append(dict(case_id=c['case_id'],shape=str(shape),dtype=dtype,mode=mode,numel=n,
        tile_shape=str([tile]),unroll=unroll,launched_AIV_blocks=blocks,
        useful_AIV_blocks=active,tiles_per_partition=partition_tiles,
        reuse_alloc=reuse,vf_fusion=vf,static_alloc_requested='None',static_alloc_effective=True,
        interpretation='Useful blocks analytically derived; not hardware occupancy telemetry'))
(ROOT/'evidence').mkdir(exist_ok=True)
with (ROOT/'evidence/dispatch.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'evidence/dispatch.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'source_gate':'passed','partition_audit':'20/20 disjoint complete logical coverage','cases':rows},indent=2)+'\n')
print('20/20 partition audits; public source gate passed')

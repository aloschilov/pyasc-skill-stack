"""Check the committed publication manifest, local links, body extracts and case mapping."""
from pathlib import Path
import ast,csv,hashlib,json,re
root=Path(__file__).resolve().parent.parent
manifest=json.loads((root/'SHA256.json').read_text())
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='SHA256.json' and '__pycache__' not in p.parts}
assert actual==set(manifest),(actual-set(manifest),set(manifest)-actual)
for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
for p in root.rglob('*.md'):
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' not in target:assert (p.parent/target.split('#')[0]).is_file(),target
with (root/'report.csv').open() as f:
 r=csv.DictReader(f);assert r.fieldnames[-2:]==['asctile_code','ascendc_code'];rows=list(r)
assert len(rows)==20 and {int(r['case_id']) for r in rows}==set(range(1,21))
assert len({r['ascendc_code'] for r in rows})==6
assert all((root/r[k]).is_file() for r in rows for k in ['asctile_code','ascendc_code'])
source={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((root/'asctile/gelu.py').read_text()).body if isinstance(n,ast.FunctionDef)}
for p in (root/'asctile').glob('*.py'):
 if p.name=='gelu.py':continue
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)]
 assert len(nodes)==1 and ast.dump(nodes[0],include_attributes=False)==source[nodes[0].name],str(p)
print(f'PASS: {len(manifest)} file hashes, Markdown links, 20 rows, 6 C++ exports, exact AscTile body ASTs')

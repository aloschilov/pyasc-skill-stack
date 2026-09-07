"""ABI, exact payload, effective JIT override and official route verification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
import asctile
import asc._C.libpyasc as lib
assert sys.version_info[:2]==(3,12) and platform.machine()=='x86_64'
wheel = ROOT/'bundle/wheel/cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl'
package = Path(importlib.util.find_spec('cann_bench').origin).parent
with zipfile.ZipFile(wheel) as z:
    for p,member in [(Path(lib.__file__),'asc/_C/'+Path(lib.__file__).name),
                     (Path(asctile.__file__),'asctile/__init__.py')]+[(package/n,'cann_bench/'+n) for n in ['gelu.py','_pyasc_runtime.py','__init__.py']]:
        assert p.read_bytes()==z.read(member),member
sys.path.insert(0,str(REPO/'integrations/cannbench/workers'))
import local_compile_gate as gate
report = gate.evaluate(package/'gelu.py','gelu',REPO/'integrations/cannbench/tasks/gelu/cases.yaml')
report.update(pyasc_commit='adadd7d66ed0ee16d33d79487bf584899a26ef1e',installed_files_match_wheel=True,
    wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    library_sha256=hashlib.sha256(Path(lib.__file__).read_bytes()).hexdigest(),
    gate_sha256=hashlib.sha256(Path(gate.__file__).read_bytes()).hexdigest(),
    scope='20 official host dispatch/codegen/pass/translation checks with invocation JIT overrides; no NPU execution')
(ROOT/'evidence/package-x86.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps({k:report[k] for k in ['status','dispatch_passed','compile_passed']}))
assert report['status']=='passed'

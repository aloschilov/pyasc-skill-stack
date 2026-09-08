"""Build an isolated GeLU-only evaluator wheel from verified runtime bytes."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
OLD = ROOT.parent/'gelu-asctile-jit-20260907'
PIN = 'adadd7d66ed0ee16d33d79487bf584899a26ef1e'
sys.path.insert(0,str(REPO/'integrations/cannbench/submission'))
sys.path.insert(0,str(ROOT/'evidence/tools'))
from source_contract import check_high_level_source, check_runtime_helper


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive-only',action='store_true')
    args=parser.parse_args()
    if args.archive_only:
        manifest=json.loads((ROOT/'evidence/package.json').read_text())
        assert sha(ROOT/'candidate/gelu.py')==manifest['candidate_sha256']
        assert sha(ROOT/'bundle/wheel/cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl')==manifest['wheel_sha256']
        build_archive(manifest)
        return
    source = ROOT/'candidate/gelu.py'
    helper = ROOT/'candidate/_pyasc_runtime.py'
    assert not check_high_level_source(source)
    assert not check_runtime_helper(helper)
    runtime = OLD/'runtime/wheels/pyasc-1.1.1-cp312-cp312-linux_x86_64.whl'
    assert sha(runtime)=='feb82abc15a09607dfb5b403af530867242809d91f26b400ba43258e303b2af8'
    pybind = REPO/'integrations/cannbench/submission/vendor/runtime-wheels/pybind11-2.13.6-py3-none-any.whl'
    assert sha(pybind)=='237c41e29157b962835d356b370ededd57594a26d5894a795960f0047cb5caf5'
    bundle = ROOT/'bundle'
    (bundle/'wheel').mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gelu-qualified-wheel-') as tmp:
        base = Path(tmp)
        for wheel,prefixes in [(runtime,('asc/','asctile/')),(pybind,('pybind11/',))]:
            with zipfile.ZipFile(wheel) as z:
                for n in z.namelist():
                    if n.startswith(prefixes): z.extract(n,base)
        package = base/'cann_bench'
        package.mkdir()
        for name in ['gelu.py','_pyasc_runtime.py','__init__.py']:
            shutil.copy2(ROOT/'candidate'/name,package/name)
        shutil.copytree(package,bundle/'cann_bench',dirs_exist_ok=True)
        info = base/'cann_bench-1.1.0.dist-info'
        info.mkdir()
        (info/'METADATA').write_text('Metadata-Version: 2.1\nName: cann-bench\nVersion: 1.1.0\n'
            'Summary: Corrected target-style GeLU with stable FP32 exact tail\nRequires-Python: >=3.12\n'
            f'X-PyAsc-Source-Commit: {PIN}\n\nHigh-level AscTile, sampled local qualification; hardware evaluation pending.\n')
        (info/'WHEEL').write_text('Wheel-Version: 1.0\nGenerator: pyasc-gelu-corrected\nRoot-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n')
        (info/'licenses').mkdir()
        shutil.copy2(OLD/'runtime/source/LICENSE',info/'licenses/PYASC-LICENSE')
        with zipfile.ZipFile(pybind) as z:
            (info/'licenses/PYBIND11-LICENSE').write_bytes(z.read('pybind11-2.13.6.dist-info/LICENSE'))
        subprocess.run([sys.executable,'-m','wheel','pack','--dest-dir',str(bundle/'wheel'),str(base)],check=True)
    wheel = bundle/'wheel/cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl'
    (bundle/'wheel.sha256.txt').write_text(f'{sha(wheel)}  wheel/{wheel.name}\n')
    manifest = dict(pin=PIN,candidate_sha256=sha(source),runtime_sha256=sha(runtime),wheel_sha256=sha(wheel),
        diagnostic_only=True,source_gate='passed',
        routes=json.loads((ROOT/'selection.json').read_text()),
        common_requested_flags={'reuse_alloc':1,'static_alloc':None,'insert_sync':True,'opt_level':3,'debug':False},
        expected_static_alloc=True,
        source_files={p.name:sha(p) for p in (ROOT/'candidate').glob('*.py')},
        task_files={p.name:sha(p) for p in (REPO/'integrations/cannbench/tasks/gelu').glob('*') if p.suffix in {'.py','.yaml'}})
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    build_archive(manifest)


def build_archive(manifest):
    bundle=ROOT/'bundle'
    assert (bundle/'setup.py').is_file() and (bundle/'build.sh').is_file(), 'Required documented package layout'
    with zipfile.ZipFile(ROOT/'diagnostic.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file() and not any(part in {'dist','__pycache__'} for part in p.relative_to(bundle).parts):
                assert p.suffix in {'.py','.sh','.txt','.json','.whl'},p
                z.write(p,p.relative_to(bundle))
    manifest['archive_sha256']=sha(ROOT/'diagnostic.zip')
    (ROOT/'evidence/package.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__': main()

"""Freeze an evidence-gated GeLU-only archive using the proven pinned-v2 runtime."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'gelu-handwritten-deepdive-20260903'
COMMIT='0a631f70968c3cb7c33ce45330a85768dd5a6f06'
TAG='pyasc-v2-gelu-register-fma-db8192-i04-20260907'

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    candidate=ROOT/'candidate/gelu.py'
    sha=digest(candidate)
    evidence=[]
    covered=set()
    for mode in ('exact','tanh'):
        for dtype in ('f16','bf16','f32'):
            p=ROOT/f'final-gate-{mode}-{dtype}.json'
            r=json.loads(p.read_text())
            assert r['source_sha256']==sha and r['checker_passed'], p
            assert r['repeat']>=2 and all(c['passed'] for c in r['cases']),p
            covered.update(c['case_id'] for c in r['cases'])
            evidence.append(dict(path=p.name,sha256=digest(p)))
    assert covered==set(range(1,21)),covered
    p=ROOT/'final-compile-x86.json'
    compile_report=json.loads(p.read_text())
    assert compile_report['pyasc_commit']==COMMIT
    assert compile_report['compile_passed']==20 and compile_report['dispatch_passed']==20
    assert all(s['status']=='passed' and not s['has_ffts_arg'] and s['memory_consumed']['UB']<=253952
               for s in compile_report['specializations'])
    evidence.append(dict(path=p.name,sha256=digest(p)))
    baseline=PREVIOUS/'submissions/iteration-03-lowlevel-tanh-safe-tile13824.zip'
    assert digest(baseline)=='dc385abe5d804a6088fb09e605935b668383c8be67eafa1fc829dba573cc9869'
    manifest=dict(schema_version=1,tag=TAG,operator='gelu',pyasc_commit=COMMIT,
                  source_class='handwritten-pyasc-inline-AscendC-register-math',
                  candidate_sha256=sha,runtime_from_archive_sha256=digest(baseline),
                  launch=dict(tile=8192,max_vector_blocks=72,unroll=2,reuse_alloc=0,
                              tail='disjoint-real-shape-no-overlapping-writes'),
                  math=dict(exact='stable erfc Numerical Recipes polynomial with FMA',
                            tanh='single exponential cancellation-free',casts='register FP32 compute'),
                  local_evidence=evidence,
                  limitations=['CaModel 9599/CANN9.0 is not remote 9589/CANN9.1',
                               'Numerical checks use reduced sizes, not full benchmark shapes'])
    archive=ROOT/(TAG+'.zip')
    if archive.exists(): raise FileExistsError('Immutable archive already exists: '+str(archive))
    with zipfile.ZipFile(baseline) as src, zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if info.filename=='cann_bench/gelu.py': data=candidate.read_bytes()
            elif info.filename=='PROVENANCE.json': data=(json.dumps(manifest,indent=2)+'\n').encode()
            else: data=src.read(info.filename)
            dst.writestr(info,data)
    manifest['archive']=dict(path=archive.name,sha256=digest(archive),size_bytes=archive.stat().st_size)
    (ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

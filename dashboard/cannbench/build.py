#!/usr/bin/env python3
"""Build the reviewed dashboard only against its matching skill/source snapshot."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent

def build(source_root, output):
    manifest=json.loads((HERE/'release-manifest.json').read_text())
    for base, entries in [(source_root,manifest['source_files']),(HERE,manifest['payload_files'])]:
        for name,expected in entries.items():
            path=base/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Release/source mismatch: '+name)
    actual=set(subprocess.check_output(['git','-C',str(source_root),'ls-files','--cached','--others','--exclude-standard','skills'],text=True).splitlines())
    expected={x for x in manifest['source_files'] if x.startswith('skills/')}
    if actual!=expected:raise ValueError('Skill file inventory changed; review and rebind the release')
    data=json.loads((HERE/'campaign-data.json').read_text())
    rows=data['rows']
    if (len(rows)!=53 or len({r['name'] for r in rows})!=53 or sum(r['total'] for r in rows)!=1060
        or sum(r['passed'] for r in rows)!=data['passed']
        or sum(r['passed']>0 for r in rows)!=data['with_pass']
        or sum(r['passed']==r['total'] for r in rows)!=data['full']
        or any(r['passed']!=len(set(r['case_ids'])) or any(c<1 or c>r['total'] for c in r['case_ids']) for r in rows)):
        raise ValueError('Invalid official case accounting')
    hardware=json.loads((HERE/'cannbench/hardware-summary.json').read_text())
    catalog=json.loads((HERE/'cannbench/manifest.json').read_text())
    if (catalog['required_operators'],catalog['required_cases'],hardware['correct_cases'],hardware['measured_cases'])!=(53,1060,20,20):
        raise ValueError('Historical summary no longer matches the reviewed view')
    output.mkdir(parents=True,exist_ok=True)
    html=(HERE/'template.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c'))
    (output/'index.html').write_text(html)
    shutil.copyfile(HERE/'capabilities-hardware.html',output/'capabilities-hardware.html')
    (output/'cannbench').mkdir(exist_ok=True)
    for p in (HERE/'cannbench').glob('*.json'):shutil.copyfile(p,output/'cannbench'/p.name)
    (output/'.nojekyll').write_text('')
    release={'git_commit':subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip(),
             'skill_source_binding_sha256':hashlib.sha256(json.dumps(manifest['source_files'],sort_keys=True).encode()).hexdigest(),
             'evidence_as_of':data['checked_at'],'interpretation':manifest['interpretation']}
    (output/'release.json').write_text(json.dumps(release,indent=2)+'\n')
    print('Validated 53 operators / 1060 cases; '+str(data['passed'])+' development PASS. Source binding matches. Built '+str(output))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,default=HERE.parents[1]);p.add_argument('--output-dir',type=Path,default=Path('_site'));a=p.parse_args()
    build(a.source_root.resolve(),a.output_dir.resolve())

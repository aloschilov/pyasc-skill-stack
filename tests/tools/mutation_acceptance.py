"""Prove guards detect the two historical regressions in isolated copies."""
import argparse
from pathlib import Path
import shutil
import sys
from run_artifacts import ROOT, RunArtifacts, sha
from run_and_verify import pinned_sim_image


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results-dir',required=True)
    args=ap.parse_args();run=RunArtifacts(args.results_dir,'mutation-acceptance')
    image=pinned_sim_image();run.data['image']=image
    template=run.path/'bad-template.py'
    source=(ROOT/'skills/pyasc-codegen-workflow/templates/kernel-template.py').read_text()
    old='        valid = min(size - tile_offset, tile_size)'
    assert source.count(old)==1
    template.write_text(source.replace(old,'        valid = size - tile_offset\n        if valid > tile_size:\n            valid = tile_size'))
    mutant=run.path/'no-platform'
    shutil.copytree(ROOT/'tests/tools',mutant/'tests/tools',ignore=shutil.ignore_patterns('__pycache__'))
    exe=mutant/'tests/tools/execute_kernel.py';s=exe.read_text()
    old='''        config.set_platform(
            config.Backend(args.backend),
            config.Platform(args.platform),
            args.device_id,
            check=False,
        )'''
    assert s.count(old)==1;exe.write_text(s.replace(old,'        pass  # deliberate regression: platform unset'))
    kernel=run.path/'run-1.py';shutil.copy2(ROOT/'docs/evidence/generation-gate-20261028/run-1/kernel.py',kernel)
    run.data['mutations']={'bad-template.py':sha(template),'execute_kernel.py':sha(exe)}
    failed=[]
    for label,driver,path,reason in [
        ('template-regression', ROOT/'tests/tools/run_and_verify.py',template,'UnsupportedSyntaxError'),
        ('platform-regression',mutant/'tests/tools/run_and_verify.py',kernel,'107001')]:
        code,report=run.command(label,[sys.executable,str(driver),str(path),'--mode','execute',
                    '--runtime-backend','docker','--docker-image',image,'--platform','Ascend950PR_9599',
                    '--timeout','120','--json'],timeout=390)
        ok=code==1 and report.get('status')=='FAIL' and reason in report.get('detail','')
        print(label, 'PASS (regression detected)' if ok else 'FAIL', flush=True)
        if not ok:failed.append(label)
    run.data['failed_cases']=failed;run.finish('failed' if failed else 'passed')
    return int(bool(failed))

if __name__=='__main__':sys.exit(main())

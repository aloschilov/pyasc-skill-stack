"""One fresh generation attempt with external evidence; no retries or commits."""
import argparse
import os
import sys
from run_artifacts import ROOT, RunArtifacts
from run_and_verify import pinned_sim_image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', required=True)
    args = ap.parse_args()
    run = RunArtifacts(args.results_dir, 'fresh-generation')
    run.data.update(image=pinned_sim_image(), profile=os.environ.get('PYASC_MODEL_PROFILE', 'cloud-glm-5.3'))
    run.save()
    env = dict(os.environ, PYASC_GENERATION_OUTPUT=str(run.path/'generated'))
    import shutil
    if shutil.which('opencode'):
        run.command('opencode-version', ['opencode', '--version'], timeout=30, env=env)
    code, _ = run.command('generation', ['bash', str(ROOT/'tests/integration/test-kernel-generation.sh'),
                                        '--strict'], timeout=1700, env=env)
    run.finish('passed' if code == 0 else 'blocked' if code in (2,124) else 'failed')
    print(f'Generation exit={code}; results: {run.path}')
    return code


if __name__ == '__main__':
    sys.exit(main())

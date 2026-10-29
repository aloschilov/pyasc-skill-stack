"""Strict, immutable generated-artifact qualification: static, score, JIT, execute."""
import argparse
from pathlib import Path
import shutil
import sys
from run_artifacts import ROOT, RunArtifacts, sha
from run_and_verify import pinned_sim_image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('kernel', type=Path)
    ap.add_argument('--results-dir', required=True)
    args = ap.parse_args()
    run = RunArtifacts(args.results_dir, 'fresh-generated-kernel-qualification')
    image = pinned_sim_image()
    if '@sha256:' not in image: raise ValueError('Digest-pinned image required')
    run.data['image'] = image
    original = sha(args.kernel)
    kernel = run.path/'kernel.py'; shutil.copy2(args.kernel, kernel)
    run.data['generated_sha256'] = original; run.save()
    status = 'failed'; result = 1
    try:
        checks = [('static', [sys.executable, str(ROOT/'tests/tools/verify_kernel.py'), str(kernel)]),
                  ('score', [sys.executable, str(ROOT/'tests/tools/score_kernel.py'), str(kernel), '--json'])]
        for name, argv in checks:
            code, report = run.command(name, argv)
            if code or (name == 'score' and not report.get('accepted')): return 1
        for mode, level in [('jit', 'compiled'), ('execute', 'numerically_verified')]:
            code, report = run.command(mode, [sys.executable, str(ROOT/'tests/tools/run_and_verify.py'),
                str(kernel), '--mode', mode, '--runtime-backend', 'docker', '--docker-image', image,
                '--platform', 'Ascend950PR_9599', '--timeout', '120', '--json'], timeout=390)
            if code == 2:
                status = 'blocked'; return 2
            if code or report.get('status') != 'PASS' or report.get('verification_level') != level:
                return 1
        if sha(kernel) != original or sha(args.kernel) != original:
            raise RuntimeError('Kernel changed during qualification')
        status = 'passed'; result = 0
        return result
    finally:
        run.finish(status)


if __name__ == '__main__':
    sys.exit(main())

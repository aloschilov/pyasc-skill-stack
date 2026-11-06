"""Fixed source-runtime probe in a clean offline venv; never calls an NPU."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

MODE = sys.argv[1]
if MODE not in ('compile', 'model', 'control'):
    raise ValueError('Unknown probe')
os.environ['PYASC_CACHE_DIR'] = str(Path.cwd()/('cache-'+MODE))
os.environ['PYASC_DUMP_PATH'] = str(Path.cwd()/('dump-'+MODE))
os.environ['PYASC_DRY_RUN'] = '1' if MODE == 'compile' else '0'

import asc
from asc.experimental import asctile
from asc.runtime import config
from asc.runtime.launcher import Launcher
from asc.lib import runtime as rt
import numpy as np
from cann_bench import runtime_probe


def main():
    started = time.monotonic()
    assert sys.prefix != sys.base_prefix
    assert sys.version_info[:2] == (3, 12)
    site = Path(asc.__file__).resolve().parent.parent
    assert site.is_relative_to(Path(sys.prefix))
    assert Path(asctile.__file__).resolve().is_relative_to(site/'asc/experimental/asctile')
    manifest = json.loads(Path('/input/bundle-manifest.json').read_text())
    package = json.loads(Path('package.json').read_text())
    wheel = Path(package['wheel'])
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == package['wheel_sha256']
    with zipfile.ZipFile(wheel) as archive:
        for name, checksum in manifest['payload_sha256'].items():
            data = (site/name).read_bytes()
            assert data == archive.read(name), name
            assert hashlib.sha256(data).hexdigest() == checksum, name
    config.set_platform('Model', 'Ascend950PR_9599', check=True)
    counts = {'launcher': 0, 'device': 0}
    original_run, original_device = Launcher.run, rt.launch_kernel
    def observed(self, *args, **kwargs):
        counts['launcher'] += 1
        if MODE != 'control':
            return original_run(self, *args, **kwargs)
    def device(*args, **kwargs):
        counts['device'] += 1
        return original_device(*args, **kwargs)
    Launcher.run, rt.launch_kernel = observed, device
    x = np.random.default_rng(0).normal(size=128).astype(np.float32)
    y = np.arange(128, dtype=np.float32)
    x_saved, y_saved = x.copy(), y.copy()
    out = np.full(128, np.nan, dtype=np.float32)
    try:
        runtime_probe(x, y, out)
    finally:
        Launcher.run, rt.launch_kernel = original_run, original_device
    binary = Path('dump-'+MODE)/'binary.o'
    assert binary.stat().st_size > 0
    assert counts['launcher'] == 1
    if MODE == 'compile':
        assert counts['device'] == 0 and np.isnan(out).all()
    elif MODE == 'control':
        assert counts['device'] == 0
        try:
            np.testing.assert_array_equal(out, x_saved+y_saved)
        except AssertionError:
            pass
        else:
            raise AssertionError('Suppressed-launch control did not reject')
    else:
        assert counts['device'] == 1
        np.testing.assert_array_equal(out, x_saved+y_saved)
        np.testing.assert_array_equal(x, x_saved)
        np.testing.assert_array_equal(y, y_saved)
        np.save('actual.npy', out); np.save('expected.npy', x_saved+y_saved)
    return {'status': 'rejected-control' if MODE == 'control' else 'passed',
            'mode': MODE, 'launches': counts, 'numerical_checks': 3 if MODE == 'model' else 0,
            'payload_files_verified': len(manifest['payload_sha256']),
            'runtime_sha256': manifest['runtime_sha256'], 'source_sha': manifest['source_sha'],
            'submission_wheel_sha256': package['wheel_sha256'],
            'python': sys.version, 'prefix': sys.prefix, 'asc': asc.__file__, 'asctile': asctile.__file__,
            'cann_home': os.environ.get('ASCEND_HOME_PATH'), 'platform': 'Model/Ascend950PR_9599',
            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'seconds': time.monotonic()-started, 'case': 'add-f32[128], one block'}


if __name__ == '__main__':
    try:
        result = main()
    except Exception:
        Path(MODE+'.json').write_text(json.dumps({'status': 'failed', 'traceback': traceback.format_exc()}, indent=2)+'\n')
        raise
    Path(MODE+'.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

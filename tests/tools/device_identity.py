"""Probe the actual NPU runtime; never infer a SoC from a simulator default."""
import glob
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import subprocess


def toolkit_info():
    roots = [os.environ.get('ASCEND_HOME_PATH', ''), '/usr/local/Ascend/ascend-toolkit/latest',
             '/usr/local/Ascend/cann', '/usr/local/Ascend/cann/latest']
    roots += glob.glob('/usr/local/Ascend/cann-*') + glob.glob('/home/*/Ascend/cann*')
    for root in filter(None, roots):
        for pattern in ('ascend_toolkit_install.info', '*/ascend_toolkit_install.info',
                        '*/*/ascend_toolkit_install.info'):
            for filename in sorted(glob.glob(str(Path(root)/pattern))):
                raw = Path(filename).read_text()
                match = re.search(r'^version\s*=\s*(.+)$', raw, re.M | re.I)
                if match:
                    return {'path': filename, 'version': match[1].strip(), 'raw': raw}
    raise RuntimeError('ascend_toolkit_install.info with a version was not found')


def probe():
    identity = {'hostname': platform.node(), 'architecture': platform.machine(),
                'python': platform.python_version(), 'backend': 'NPU'}
    try:
        smi = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20)
        identity['npu_smi'] = {'exit_code': smi.returncode, 'stdout': smi.stdout, 'stderr': smi.stderr}
        if smi.returncode:
            raise RuntimeError('npu-smi info failed')
        # Preserve the full table (including card/chip IDs); do not assume
        # physical card IDs equal the runtime's logical device indices.
        models = sorted(set(re.findall(r'\b(?:Ascend)?(?:950|910|310)[A-Za-z0-9_]*\b', smi.stdout)))
        identity['models'] = models
        if not models:
            raise RuntimeError('npu-smi did not identify a supported Ascend model')
        identity['cann'] = toolkit_info()
        import asc
        from asc.lib import runtime as rt
        rt.use_npu()
        identity['soc_version'] = rt.current_platform()
        identity['device_count'] = rt.device_count()
        if not re.fullmatch(r'Ascend[A-Za-z0-9_]+', identity['soc_version']) or identity['device_count'] < 1:
            raise RuntimeError('No usable NPU/SoC reported by the native runtime')
        identity['pyasc_module'] = str(Path(asc.__file__).resolve())
        for package in ('pyasc-dev', 'pyasc'):
            try:
                identity['pyasc_version'] = importlib.metadata.version(package)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        if 'pyasc_version' not in identity:
            raise RuntimeError('Installed pyasc distribution version unavailable')
        identity['status'] = 'ready'
    except Exception as exc:
        identity.update(status='unavailable', reason=f'{type(exc).__name__}: {exc}')
    return identity

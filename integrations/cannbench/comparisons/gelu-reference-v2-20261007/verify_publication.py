"""Verify the published snapshot without a device, account, network or imports of pyasc."""
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    '.': 'a7819eede1ac0a1867364db81c97fe9283193219fa871f7d5a112c8ac8b62cd0',
    'repair': 'ef5269b46a56437463f5dcef53fd408dcf173c3a502e0306726a797a6f840ceb',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_archive(folder, expected):
    metadata = json.loads((folder / 'package.json').read_text())
    archive_bytes = (folder / 'diagnostic.zip').read_bytes()
    require(sha(archive_bytes) == expected == metadata['archive_sha256'], 'Archive hash')
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate ZIP members')
        members = {name: archive.read(name) for name in archive.namelist()}
    # The first manifest was created before source.sha256.txt itself existed.
    require(set(members) == set(metadata['members']) | {'source.sha256.txt'}, 'ZIP inventory')
    for name, digest in metadata['members'].items():
        require(sha(members[name]) == digest, f'Member hash: {name}')
    for line in members['source.sha256.txt'].decode().splitlines():
        digest, name = line.split('  ', 1)
        require(sha(members[name]) == digest, f'In-build manifest: {name}')
    require(members['cann_bench/kernel.py'] == (ROOT / 'kernel.py').read_bytes(), 'Kernel identity')
    require(sha(members['cann_bench/kernel.py']) == metadata['kernel_sha256'], 'Kernel hash')
    for path in (ROOT / 'bundle').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            name = path.relative_to(ROOT / 'bundle').as_posix()
            if name != 'source.sha256.txt':
                require(path.read_bytes() == members[name], f'Readable source identity: {name}')
    wheels = [name for name in members if name.endswith('.whl')]
    require(len(wheels) == 1, 'One self-contained wheel')
    require(sha(members[wheels[0]]) == metadata['wheel_sha256'], 'Wheel hash')
    with zipfile.ZipFile(io.BytesIO(members[wheels[0]])) as wheel:
        names = wheel.namelist()
        require(len(names) == len(set(names)), 'Duplicate wheel members')
        records = [name for name in names if name.endswith('.dist-info/RECORD')]
        require(len(records) == 1, 'Wheel RECORD')
        rows = list(csv.reader(io.StringIO(wheel.read(records[0]).decode())))
        require(len(rows) == len(names) and {row[0] for row in rows} == set(names), 'RECORD inventory')
        for name, digest, size in rows:
            if name == records[0]:
                require(not digest and not size, 'RECORD self entry')
                continue
            payload = wheel.read(name)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip('=')
            require(digest == 'sha256=' + encoded and int(size) == len(payload), f'RECORD: {name}')
        require(wheel.read('cann_bench/kernel.py') == members['cann_bench/kernel.py'], 'Wheel kernel')
        for suffix in ('LICENSE-pyasc', 'LICENSE-pybind11', 'LICENSE-LLVM', 'LICENSE-MLIR', 'REFERENCE-NOTICE'):
            require(any(name.endswith('/' + suffix) for name in names), f'License: {suffix}')
    return members


def main():
    original, repaired = [verify_archive(ROOT / folder, digest) for folder, digest in EXPECTED.items()]
    require(set(original) == set(repaired), 'Repair inventory')
    require([name for name in original if original[name] != repaired[name]] == ['source.sha256.txt'], 'Manifest-only repair')
    require(b'  build.sh\n' in original['source.sha256.txt'], 'Original entrypoint hash')
    require(b'  build.sh\n' not in repaired['source.sha256.txt'], 'Repaired entrypoint exclusion')
    summary = json.loads((ROOT / 'repair/hardware-summary.json').read_text())
    with (ROOT / 'repair/hardware-cases.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / 'inputs/dispatch-trace/cases.csv').open(newline='') as stream:
        trace = {row['case_id']: row for row in csv.DictReader(stream)}
    require(len(rows) == 20 and {int(row['case_id']) for row in rows} == set(range(1, 21)), 'All 20 cases')
    require(set(trace) == {row['case_id'] for row in rows}, 'Trace cases')
    ratios = []
    for row in rows:
        source = trace[row['case_id']]
        require(row['shape'] == 'x'.join(map(str, json.loads(source['shape']))), 'Case shape')
        require(row['dtype'] == source['dtype'] and row['mode'] == source['approximate'], 'Case dtype/mode')
        require(row['status'] == 'success' and row['accuracy'] == 'True', 'Hardware accuracy')
        ratio = float(row['baseline_perf_us']) / float(row['elapsed_us'])
        require(math.isclose(ratio, float(row['derived_speedup']), rel_tol=1e-12), 'Case speedup')
        require(row['job_url'].endswith('/' + summary['job_id']), 'Job identity')
        ratios.append(ratio)
    gm = math.exp(sum(map(math.log, ratios)) / 20)
    require(math.isclose(gm, summary['gm_all20'], rel_tol=1e-12), 'Independent GM')
    require(math.isclose(sum(ratios) / 20, summary['server_summary']['geometric_mean_speedup'], rel_tol=1e-12), 'Server arithmetic-equivalent field')
    for folder in (ROOT, ROOT / 'repair'):
        result = json.loads((folder / 'x86-qualification/x86-results.json').read_text())
        require(result['passed'] and len(result['dispatches']) == 80 and len(result['specializations']) == 6, 'Offline qualification record')
    for report in (ROOT / 'README.md', ROOT / 'inputs/dispatch-trace/README.md'):
        for target in re.findall(r'\]\(([^)]+)\)', report.read_text()):
            if '://' not in target and not target.startswith('#'):
                require((report.parent / target.split('#')[0]).exists(), f'Report link: {target}')
    print(f'PASS: both exact archives, wheel RECORDs, source identity, manifest-only repair, links; 20/20, GM={gm:.9f}x')


if __name__ == '__main__':
    main()

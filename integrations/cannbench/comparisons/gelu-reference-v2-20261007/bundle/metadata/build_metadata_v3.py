"""Build our metadata C ABI against the caller's installed SDK, not vendored CANN.

Intended for the evaluator's ordinary build stage, before wheel installation and
timing. Library-relative metadata discovery draft; not wired into an archive. No device query/initialization,
resource setter, kernel compilation or custom pyasc pass pipeline is involved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

SOURCE = Path(__file__).with_name('platform_capacity.cpp')
REQUIRED = ('include/platform/platform_info.h',
            'asc/include/utils/tiling/platform/platform_ascendc.h',
            'lib64/libtiling_api.a', 'lib64/libplatform.so',
            'lib64/libunified_dlog.so', 'lib64/libascend_protobuf.so.3.13.0.0',
            'lib64/libmmpa.so', 'lib64/libc_sec.so')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metadata_layout(sdk):
    sdk=Path(sdk).resolve(strict=True)
    platform_link=sdk/'lib64/libplatform.so'
    platform_real=platform_link.resolve(strict=True)
    config_dir=(platform_real.parent.parent/'data/platform_config').resolve(strict=True)
    configs=sorted(p for p in config_dir.glob('*.ini') if p.is_file())
    if not configs:
        raise ValueError('installed_sdk_platform_metadata_missing: '+str(config_dir))
    return dict(sdk=str(sdk),platform_link=str(platform_link),platform_real=str(platform_real),
                config_directory=str(config_dir),config_files=[str(p) for p in configs],
                policy='RealSoFilePath(libplatform)/../data/platform_config; public no-SoC InitializePlatformInfo')


def installed_inputs(sdk):
    sdk = Path(sdk).resolve(strict=True)
    inputs = [sdk/item for item in REQUIRED]
    if any(not item.is_file() for item in inputs):
        raise ValueError('installed_sdk_files_missing')
    layout=metadata_layout(sdk)
    configs=[Path(p) for p in layout['config_files']]
    inputs.append(Path(layout['platform_real']))
    # Public SDK headers can include other SDK headers; record their complete
    # trees rather than claiming identity from two entrypoint headers alone.
    headers = sorted(set((sdk/'include').rglob('*.h')) |
                     set((sdk/'asc/include/utils/tiling').rglob('*.h')))
    return sdk, sorted(set(inputs+configs+headers))


def compile_command(sdk, compiler, output):
    return [str(compiler), '-std=c++17', '-O2', '-shared', '-fPIC', '-D_GLIBCXX_USE_CXX11_ABI=0',
            '-I'+str(sdk/'include'), '-I'+str(sdk/'asc/include/utils/tiling'),
            str(SOURCE), str(sdk/'lib64/libtiling_api.a'), '-L'+str(sdk/'lib64'),
            '-Wl,-rpath,'+str(sdk/'lib64'), '-Wl,-rpath-link,'+str(sdk/'lib64'),
            '-Wl,-z,defs', '-lplatform', '-lunified_dlog', '-ldl', '-pthread',
            '-o', str(output/'platform_capacity.so')]


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def build(sdk, output, *, compiler=None):
    sdk, inputs = installed_inputs(sdk)
    layout=metadata_layout(sdk)
    compiler = compiler or shutil.which('c++')
    if compiler is None:
        raise ValueError('host_cpp_compiler_unavailable')
    compiler = Path(compiler).absolute()
    compiler.resolve(strict=True)  # Check existence; preserve c++/clang++ argv[0] driver mode.
    output = Path(output).absolute()
    # Exclusive directory: no existing binary or manifest is overwritten/reused.
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    inputs += [SOURCE, Path(__file__), compiler]
    hashes = {str(path): digest(path) for path in inputs}
    command = compile_command(sdk, compiler, output)
    save(output/'START.json', dict(started=time.time(), sdk=str(sdk), inputs=hashes, command=command,
         abi='C entrypoint; SDK C++ libstdc++ ABI0', source_only_transport=True,
         scope='Host metadata helper, not kernel/pass compilation', metadata_layout=layout))
    with (output/'compiler-version.log').open('x') as log:
        try:
            version_code = subprocess.run([str(compiler), '--version'], stdout=log,
                                          stderr=subprocess.STDOUT, timeout=15).returncode
        except subprocess.TimeoutExpired:
            version_code = 124
    if version_code:
        save(output/'RESULTS.json', dict(passed=False, reason='compiler_version_failed', exit_code=version_code))
        return False
    with (output/'compile.log').open('x') as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180).returncode
        except subprocess.TimeoutExpired:
            code = 124
    binary = output/'platform_capacity.so'
    unchanged = (metadata_layout(sdk)==layout and
                 all(digest(Path(path)) == value for path, value in hashes.items()))
    binary_present = binary.is_file() and binary.stat().st_size > 0
    passed = code == 0 and binary_present and unchanged
    save(output/'RESULTS.json', dict(passed=passed, compile_exit=code, inputs_unchanged=unchanged,
         binary_sha256=digest(binary) if binary_present else None,
         loadability_qualified=False, active_device_qualified=False, submission_qualified=False))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    passed = build(args.sdk, args.output)
    print(json.dumps(dict(passed=passed, output=str(args.output))))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())

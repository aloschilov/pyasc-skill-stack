#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
vendor_dir="${script_dir}/submission/vendor/runtime-wheels"
output_dir="${script_dir}/.qemu-runtime-dist"
image="python:3.12.13-slim-bookworm"

if ! docker run --rm --platform linux/amd64 "${image}" true >/dev/null 2>&1; then
    echo "[qemu] registering the amd64 binfmt handler"
    docker run --privileged --rm tonistiigi/binfmt --install amd64
fi

rm -rf "${output_dir}"
mkdir -p "${output_dir}"
docker buildx build \
    --platform linux/amd64 \
    --file "${script_dir}/Dockerfile.cp312-x86" \
    --target wheel \
    --output "type=local,dest=${output_dir}" \
    "${script_dir}"

wheel="$(find "${output_dir}" -maxdepth 1 -type f -name 'pyasc-*.whl' -print -quit)"
test -n "${wheel}"
test "$(unzip -p "${wheel}" 'pyasc-*.dist-info/WHEEL' | sed -n 's/^Tag: //p' | head -1)" = \
    "cp312-cp312-linux_x86_64"

mkdir -p "${vendor_dir}"
find "${vendor_dir}" -maxdepth 1 -type f -name 'pyasc-*.whl' -delete
cp "${wheel}" "${vendor_dir}/"
python3 "${script_dir}/recompress-runtime-wheel.py" \
    "${vendor_dir}/$(basename "${wheel}")"
(
    cd "${vendor_dir}"
    sha256sum pyasc-*.whl > pyasc.sha256.txt
)

echo "[qemu] runtime wheel refreshed from pyasc v2 ac1222a48c8914d3f81297c7570d1a84f0f26778"
cat "${vendor_dir}/pyasc.sha256.txt"

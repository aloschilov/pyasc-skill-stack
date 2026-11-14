"""Compile one recorded AscTile emission with inline or outlined register loops."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from shutil import copyfile
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DIGEST = {
    11: "03511e5913f17f4cff457df755c3f00ea769e844212a642e750c0dafe1c695ba",
    13: "0d86a5a49f430d83c24c64eb4c9e972fb015f4668b37903baf2b218616320e8a",
}
SCOPE = re.compile(
    r"    __VEC_SCOPE__\n    \{\n      using namespace AscendC::Reg;\n"
    r"      auto\* x = (reinterpret_cast<__ubuf__ float\*>\(v\d+\.GetPhyAddr\(\)\));\n"
    r"      auto\* y = (reinterpret_cast<__ubuf__ float\*>\(v\d+\.GetPhyAddr\(\)\));\n"
    r"(.*?)^    \};", re.S | re.M)


def outline(source, count):
    helpers = {}
    def replace(match):
        x, y, body = match.groups()
        name = helpers.setdefault(body, f"register_loop_{len(helpers)}")
        return f"    asc_vf_call<{name}>({x}, {y});"

    body, found = SCOPE.subn(replace, source)
    if found != count or "__VEC_SCOPE__" in body:
        raise ValueError("unexpected register scopes in the recorded compiler output")
    functions = "".join(
        f"__simd_vf__ inline void {name}(__ubuf__ float* x, __ubuf__ float* y) {{\n"
        "  using namespace AscendC::Reg;\n" + code + "}\n"
        for code, name in helpers.items()
    )
    return body.replace('extern "C"', functions + 'extern "C"', 1)


def driver(case, kernel):
    text = (HERE / "direct-simt/short_rows.asc").read_text()
    start = text.index("static void check(")
    includes = text[text.index("#include <cstdlib>"):text.index("#ifndef DIRECT_VECTOR\ntemplate")]
    text = text[start:]
    start = text.index("#ifdef DIRECT_VECTOR")
    end = text.index("        check(aclrtSynchronizeStream", start)
    blocks = 35 if case == 11 else 36
    text = text[:start] + f'''    if (which != "{case}") return 2;
    for (int i = 0; i < 17; ++i) {{
        staged_rows<<<{blocks},nullptr,stream>>>(x, y);
''' + text[end:]
    # The ASC host/device frontend requires an explicit vector kernel marker.
    # The original wheel supplies its vector target through its compile command.
    kernel = kernel.replace("__global__ __aicore__ void staged_rows", "__global__ __vector__ void staged_rows")
    return '#include "acl/acl.h"\n' + includes + kernel + "\n" + text


def prepare(dumps):
    cmake = ["cmake_minimum_required(VERSION 3.16)", "find_package(ASC REQUIRED)",
             "project(emitted_vf LANGUAGES ASC CXX)"]
    for c, digest in DIGEST.items():
        file = dumps / f"c{c}w1r0/dump/ascendc.cpp"
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError(f"compiler source hash mismatch for {c}")
        code = file.read_text()
        for mode, kernel in [("scope", code), ("call", outline(code, 3 if c == 11 else 4))]:
            target = f"{mode}{c}"
            Path(target + ".asc").write_text(driver(c, kernel))
            cmake += [f"add_executable({target} {target}.asc)",
                      f"target_compile_options({target} PRIVATE --npu-arch=dav-3510)"]
    Path("CMakeLists.txt").write_text("\n".join(cmake) + "\n")


def measure(task):
    schedule = []
    for round_id in (1, 2):
        order = [(round_id, c, mode) for c in DIGEST for mode in ("scope", "call")]
        random.Random(730 + round_id).shuffle(order)
        schedule.extend(order)
    Path("plan.json").write_text(json.dumps(schedule))
    for r, c, mode in schedule:
        out = Path(f"r{r}c{c}{mode}")
        out.mkdir()
        copyfile(Path("build") / f"{mode}{c}", out / "bench")
        (out / "bench").chmod(0o755)
        with (out / "metrics.json").open("w") as metrics, (out / "log").open("w") as log:
            run = subprocess.run([sys.executable, str(HERE / "replay.py"), str(task.resolve()),
                                  str(c), "--compiled-entry", "staged_rows"], cwd=out,
                                 stdout=metrics, stderr=log)
        with Path("status").open("a") as status:
            print(r, c, mode, run.returncode, file=status)
    Path("done").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "measure"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else measure)(args.path)

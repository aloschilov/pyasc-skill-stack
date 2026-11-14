"""Change only the host launch stream on two independently generated short-stream paths."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil


def render(source, operator):
    identifiers = {
        "exp": ("112da3565aed706120834d408b6f0c9f3f74a57320ac6273d60531a53bc5df08",
                "_exp_kernel", "x"),
        "swi_glu": ("cfab02487ec89844382d51a2b44cb093fd3c30613f9b54bae2ee830060c3e06b",
                    "two_wave_swiglu", "input"),
    }
    checksum, kernel, tensor = identifiers[operator]
    if hashlib.sha256(source.encode()).hexdigest() != checksum:
        raise ValueError("unexpected generated source")
    host, = (n for n in ast.parse(source).body if getattr(n, "name", None) == operator)
    launches = sorted((node for node in ast.walk(host) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Subscript)
                       and getattr(node.func.value, "id", None) == kernel), key=lambda node: node.lineno)
    # In Exp, the first _exp_kernel launch is the narrow identity path. Later
    # uses of that kernel implement the general fallback and keep their grid.
    launch = launches[0]
    blocks = ast.unparse(launch.func.slice)
    lines = source.splitlines(keepends=True)
    idx = launch.lineno - 1
    indent = lines[idx][:launch.col_offset]
    signature = f"{kernel}[{blocks}]"
    if not lines[idx].startswith(indent + signature + "("):
        raise ValueError("unexpected launch expression")
    lines[idx] = indent + f"grid = {blocks}\n" + indent + f'if {tensor}.device.type == "npu":\n' + (
        indent + "    import ctypes\n"
        + indent + f"    stream = torch.npu.current_stream({tensor}.device).npu_stream\n"
        + indent + "    grid = (grid, ctypes.c_void_p(stream))\n"
        + lines[idx].replace(signature, f"{kernel}[grid]", 1)
    )
    result = "".join(lines)
    ast.parse(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("operator", choices=("exp", "swi_glu"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    name = args.operator + ".py"
    source = render((args.package / name).read_text(), args.operator)
    args.output.mkdir()
    (args.output / name).write_text(source)
    shutil.copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")

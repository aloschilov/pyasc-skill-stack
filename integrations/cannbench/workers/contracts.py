"""Task contract helpers; no tensor runtime or compiler is imported here."""
import ast
import json
from pathlib import Path
import yaml


def public_signature(task_dir: Path, name: str) -> ast.arguments:
    tree = ast.parse((task_dir / "golden.py").read_text())
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"Golden must define exactly one {name}")
    return matches[0].args


def parameter_names(task_dir: Path, name: str) -> tuple[str, ...]:
    args = public_signature(task_dir, name)
    return tuple(a.arg for a in args.posonlyargs + args.args + args.kwonlyargs)


def case_arguments(proto: dict, case: dict, make_tensor) -> dict:
    """Build official input/attr kwargs, including tensor lists and None slots."""
    inputs = proto["operator"]["inputs"]
    shapes = case["input_shape"]
    dtypes = case["dtype"]
    if len(shapes) > len(inputs):
        raise ValueError("More case inputs than task inputs")

    def convert(shape, dtype):
        if shape is None:
            return None
        if not isinstance(shape, list):
            raise ValueError("Case shape must be a list or None")
        if all(type(x) is int for x in shape):
            if not isinstance(dtype, str):
                raise ValueError("Tensor dtype must be a string")
            return make_tensor(shape, dtype)
        if isinstance(dtype, list) and len(dtype) != len(shape):
            raise ValueError("Tensor-list shape/dtype lengths disagree")
        return [convert(s, dtype[i] if isinstance(dtype, list) else dtype) for i, s in enumerate(shape)]

    values = {}
    for i, shape in enumerate(shapes):
        dtype = dtypes[i] if i < len(dtypes) else dtypes[0] if len(dtypes) == 1 else None
        values[inputs[i]["name"]] = convert(shape, dtype)
    attrs = case.get("attrs") or {}
    for name, value in attrs.items():
        if name in values and values[name] is not None and value != values[name]:
            raise ValueError("Case attr conflicts with tensor input: " + name)
        values[name] = value
    for entry in inputs[len(shapes):]:
        if entry["name"] not in values and not entry.get("optional"):
            raise ValueError("Missing required case input: " + entry["name"])
    return values


def load_contract(task_dir: Path):
    return yaml.safe_load((task_dir / "proto.yaml").read_text())


ALLOWED_TORCH_CALLS = {"empty", "empty_like", "zeros", "zeros_like", "tensor"}
BANNED_TORCH_SUBMODULES = {"nn", "ops", "functional", "linalg", "special",
                           "fft", "cuda", "npu"}
def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def static_check(path: Path, callable_name: str, task_dir: Path) -> list[str]:
    problems = []
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    callable_node = next((
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == callable_name
    ), None)
    if callable_node is None:
        problems.append(
            f"missing top-level public callable def {callable_name}(...)")
    else:
        actual = tuple(arg.arg for arg in callable_node.args.posonlyargs + callable_node.args.args + callable_node.args.kwonlyargs)
        expected = parameter_names(task_dir, callable_name)
        if actual != expected:
            problems.append(
                f"public signature parameters {actual!r} != {expected!r}"
            )
        if callable_node.args.vararg or callable_node.args.kwarg:
            problems.append("public callable must not use *args or **kwargs")
        expected_args = public_signature(task_dir, callable_name)
        def defaults(arguments):
            positional = arguments.posonlyargs + arguments.args
            values = dict(zip((arg.arg for arg in positional[-len(arguments.defaults):]), arguments.defaults)) if arguments.defaults else {}
            values.update((arg.arg, value) for arg, value in zip(arguments.kwonlyargs, arguments.kw_defaults) if value is not None)
            return {key: ast.dump(value) for key, value in values.items()}
        expected_defaults = defaults(expected_args)
        actual_defaults = defaults(callable_node.args)
        if actual_defaults != expected_defaults:
            problems.append("Public callable defaults differ from official golden")

    has_jit = any(
        isinstance(n, ast.FunctionDef) and any(
            (_dotted(d) or _dotted(getattr(d, "func", None) or d)) in ("asc2.jit", "asctile.jit", "asc.jit")
            for d in n.decorator_list)
        for n in ast.walk(tree))
    if not has_jit:
        problems.append("no public pyasc JIT kernel found")

    if "ensure_npu_platform" not in source:
        problems.append("missing ensure_npu_platform import/call "
                        "(from ._pyasc_runtime import ensure_npu_platform)")
    if "import torch_npu" in source:
        problems.append("do not import torch_npu (not needed, confuses the "
                        "harness environment)")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = _dotted(node)
            if name and name.startswith("torch."):
                sub = name.split(".")[1]
                if sub in BANNED_TORCH_SUBMODULES:
                    problems.append(f"banned torch usage: {name}")
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "data_ptr":
                problems.append(
                    "banned Tensor.data_ptr(): pass the Tensor itself to the "
                    "pyasc v2 JIT launch so dtype specialization is preserved"
                )
            if name and name.startswith("torch.") and name.count(".") == 1:
                fn = name.split(".")[1]
                if fn not in ALLOWED_TORCH_CALLS:
                    problems.append(
                        f"banned torch call: {name}() — torch is allowed only "
                        f"for {sorted(ALLOWED_TORCH_CALLS)}; all math must be "
                        f"asctile kernels")
    for function in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        decorators = [_dotted(getattr(dec, "func", dec)) for dec in function.decorator_list]
        if any(name in ("asctile.jit", "asc.jit", "asc2.jit") for name in decorators):
            continue
        # Tensor allocation/view assembly is host metadata; copying values into
        # output tensors belongs in pyasc kernels and must not bypass codegen.
        allocated = set()
        for node in ast.walk(function):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
                call = _dotted(node.value.func)
                if call in {"torch.empty", "torch.empty_like", "torch.zeros", "torch.zeros_like"}:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    allocated.update(target.id for target in targets if isinstance(target, ast.Name))
        for node in ast.walk(function):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "copy_"):
                problems.append("banned host Tensor.copy_(): write outputs inside pyasc JIT kernels")
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                            and target.value.id in allocated):
                        problems.append("banned host Tensor indexed assignment: " + target.value.id)
    return sorted(set(problems))

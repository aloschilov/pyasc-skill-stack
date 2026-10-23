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

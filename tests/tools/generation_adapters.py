"""Trusted probes for three explicit upstream task anchors (not the full suite)."""
import ast
import json
import math
import re
from pathlib import Path

# Deliberately small contracts. A different upstream row must be reviewed before
# it can acquire a numerical claim, even when its name still matches.
TASKS = {
    ('test_vadd.py', 'vadd_test_2'): {
        'inputs': [[199], [199]], 'output': [199], 'dtype': 'float32',
        'operation': 'add', 'blocks': 1, 'rtol': 1e-5, 'atol': 1e-6},
    ('test_reduce_sum.py', 'reduce_sum_test_1'): {
        'inputs': [[61]], 'output': [1], 'dtype': 'int32',
        'operation': 'sum', 'blocks': 1, 'rtol': 0, 'atol': 0},
    ('test_broadcast.py', 'broadcast_test_2'): {
        'inputs': [[1, 5, 8]], 'output': [4, 5, 8], 'dtype': 'float32',
        'operation': 'broadcast', 'blocks': 4, 'rtol': 0, 'atol': 0},
}


def task_contract(root, cell):
    if len(cell['cases']) != 1 or (cell['module'], cell['cases'][0]) not in TASKS:
        raise ValueError('Unsupported numerical adapter: ' + cell['id'])
    contract = dict(TASKS[cell['module'], cell['cases'][0]])
    source = (Path(root)/'reference/asctile/target'/cell['module']).read_text()
    block = re.search(r'# PYASC_TESTS_BEGIN\n(.+?)# PYASC_TESTS_END', source, re.S)
    rows = ast.parse('[' + block[1] + ']', mode='eval').body.elts
    row = next(r for r in rows if r.elts[0].value == cell['cases'][0]).elts
    def dtypes(node):
        if not isinstance(node, ast.Tuple): raise ValueError('Invalid dtype declaration')
        for d in node.elts:
            if not isinstance(d, ast.Attribute) or not isinstance(d.value, ast.Name) or d.value.id != 'torch':
                raise ValueError('Invalid dtype declaration')
        return [d.attr for d in node.elts]
    if (ast.literal_eval(row[1]) != contract['blocks']
            or list(ast.literal_eval(row[2])) != contract['inputs']
            or list(ast.literal_eval(row[4])) != [contract['output']]
            or dtypes(row[3]) != [contract['dtype']] * len(contract['inputs'])
            or dtypes(row[5]) != [contract['dtype']]):
        raise ValueError('Adapter differs from upstream shape/dtype/block contract')
    if contract['operation'] == 'sum' and ast.literal_eval(row[7]) != (2, [0], True):
        raise ValueError('Reduction axes/keepdim contract changed')
    if contract['operation'] == 'broadcast' and ast.literal_eval(row[7]) != ([4, 5, 8],):
        raise ValueError('Broadcast contract changed')
    return dict(contract, seeds=[0, 1, 7], entrypoint='launch',
                arguments=['x', 'y', 'out'] if contract['operation']=='add' else ['x', 'out'])


def static_check(source, contract):
    """Constrain host code to a launch; no Torch fallback or arbitrary imports.

    This is an accidental-cheating guard, not a hostile-code sandbox. Candidate
    execution still belongs in a networkless container without credentials.
    """
    tree = ast.parse(source)
    kernels, launch = {}, None
    for node in tree.body:
        if isinstance(node, ast.Import) and [(a.name, a.asname) for a in node.names] == [('asc', None)]:
            continue
        if (isinstance(node, ast.ImportFrom) and node.module == 'asc.experimental' and node.level == 0
                and [(a.name, a.asname) for a in node.names] == [('asctile', None)]):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            ast.literal_eval(node.value)
            continue
        if not isinstance(node, ast.FunctionDef): raise ValueError('Only imports, constants and functions allowed')
        if node.name == 'launch':
            if launch is not None: raise ValueError('Duplicate launch')
            launch = node
        else:
            decorators = node.decorator_list
            if (len(decorators) != 1 or not isinstance(decorators[0], ast.Call)
                    or ast.unparse(decorators[0].func) != 'asctile.jit'):
                raise ValueError('Device functions must use asctile.jit')
            if node.name in kernels: raise ValueError('Duplicate kernel')
            kernels[node.name] = node
    if not kernels or launch is None: raise ValueError('Missing JIT kernel or launch')
    if (launch.decorator_list or launch.args.defaults or launch.args.kw_defaults or launch.args.kwonlyargs
            or launch.args.posonlyargs or launch.args.vararg or launch.args.kwarg
            or [a.arg for a in launch.args.args] != contract['arguments']):
        raise ValueError('launch signature differs from task')
    body = [n for n in launch.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                                         and isinstance(n.value.value, str))]
    if len(body) != 1 or not isinstance(body[0], ast.Expr) or not isinstance(body[0].value, ast.Call):
        raise ValueError('Host launch must contain exactly one kernel invocation')
    call = body[0].value
    if (not isinstance(call.func, ast.Subscript) or not isinstance(call.func.value, ast.Name)
            or call.func.value.id not in kernels or call.keywords
            or ast.literal_eval(call.func.slice) != contract['blocks']):
        raise ValueError('Host launch must select the task block count')
    n = len(contract['arguments'])
    if [ast.unparse(a) for a in call.args[:n]] != contract['arguments']:
        raise ValueError('Input/output tensors must be passed unchanged')
    for a in call.args[n:]: ast.literal_eval(a)
    # Disallow runtime reconfiguration, introspection, host writes and imports
    # inside functions as well as import-time calls hidden in defaults.
    allowed_names = {'min', 'max', 'range', 'len', 'int', 'float', 'bool', 'abs'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
            raise ValueError('Nested import')
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or ast.unparse(node).startswith('asc.runtime')):
            raise ValueError('Runtime/introspection access')
        if isinstance(node, ast.FunctionDef) and (node.args.defaults or node.args.kw_defaults):
            raise ValueError('Function defaults are not supported')
        if isinstance(node, ast.Call) and node is not call:
            name = ast.unparse(node.func)
            if not (name.startswith('asctile.') or name in allowed_names or (isinstance(node.func, ast.Attribute) and node.func.attr in ('reshape', 'broadcast_to'))):
                raise ValueError('Unsupported device call: ' + name)
    return {'status': 'passed', 'kernel_names': sorted(kernels), 'entrypoint': 'launch'}


def wrapper(contract, *, check=True):
    """Comparisons stay in this top-level file for harness attribution."""
    args = ', '.join(contract['arguments'])
    setup = '    y = ((torch.arange(199) * 3 + seed) % 19 - 9).to(dtype) / 4\n' if contract['operation']=='add' else ''
    expected = {'add': 'x + y', 'sum': 'x.sum(dim=0, keepdim=True, dtype=dtype)',
                'broadcast': 'x.expand(4, 5, 8).clone()'}[contract['operation']]
    checks = (f'    torch.testing.assert_close(out, expected, rtol={contract["rtol"]}, atol={contract["atol"]})\n'
              '    torch.testing.assert_close(x, original_x, rtol=0, atol=0)\n') if check else ''
    if check and contract['operation']=='add': checks += '    torch.testing.assert_close(y, original_y, rtol=0, atol=0)\n'
    poison = 'float("nan")' if contract['dtype']=='float32' else '-2147483648'
    return ('import torch\nfrom candidate import launch\n'
            f'dtype = torch.{contract["dtype"]}\n'
            'for seed in (0, 1, 7):\n'
            f'    x = ((torch.arange({math.prod(contract["inputs"][0])}) * 7 + seed) % 23 - 11).to(dtype).reshape({contract["inputs"][0]!r})\n'
            + setup + '    original_x = x.clone()\n'
            + ('    original_y = y.clone()\n' if contract['operation']=='add' else '')
            + f'    expected = {expected}\n'
            + f'    out = torch.full({contract["output"]!r}, {poison}, dtype=dtype)\n'
            + f'    launch({args})\n' + checks)


def write_probes(directory, contract):
    directory = Path(directory)
    (directory/'adapter.json').write_text(json.dumps(contract, indent=2)+'\n')
    for name, check in [('probe.py', True), ('compile_probe.py', False)]:
        (directory/name).write_text(wrapper(contract, check=check))

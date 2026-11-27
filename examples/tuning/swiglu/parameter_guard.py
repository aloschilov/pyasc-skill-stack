"""Prove a narrowly bounded scalar-parameter derivative without importing it."""
import ast
import copy
import hashlib

BASELINE_SHA256 = 'd25fac7aef1183600c8a96c42bdc740fba274ac9ec4cf153da28f5be09ac6199'

def dump(x):
    return ast.dump(x, include_attributes=False)

def scalar(node, depth=0):
    if depth > 12:
        raise ValueError('Scalar expression depth')
    if isinstance(node, ast.Constant):
        if type(node.value) is not int or not 1 <= node.value <= 32768:
            raise ValueError('Only bounded positive integer constants')
    elif isinstance(node, ast.Name):
        if node.id not in {'R', 'H', 'aiv', 'T', 'W', '_UB_USABLE_LANES'}:
            raise ValueError('Unapproved scalar name')
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
        scalar(node.left, depth + 1); scalar(node.right, depth + 1)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.keywords or node.func.id not in {'min', 'max', '_pow2_ceil'}:
            raise ValueError('Unapproved scalar call')
        if len(node.args) != (1 if node.func.id == '_pow2_ceil' else 2):
            raise ValueError('Scalar call arity')
        for arg in node.args:
            scalar(arg, depth + 1)
    else:
        raise ValueError('Only bounded scalar arithmetic; no indexing, attributes or dispatch tables')

def validate(baseline, candidate):
    if hashlib.sha256(baseline).hexdigest() != BASELINE_SHA256:
        raise ValueError('Baseline pin')
    if len(candidate) > 32768:
        raise ValueError('Source bound')
    before, after = ast.parse(baseline), ast.parse(candidate)
    def geometry(tree):
        rows = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_row_geometry']
        if len(rows) != 1:
            raise ValueError('One existing geometry function required')
        return rows[0]
    original, changed = geometry(before), geometry(after)
    replaced = copy.deepcopy(after)
    replaced.body[after.body.index(changed)] = copy.deepcopy(original)
    if dump(before) != dump(replaced):
        raise ValueError('Only _row_geometry scalar parameters may change')
    signature = copy.deepcopy(changed); signature.body = copy.deepcopy(original.body)
    if dump(signature) != dump(original):
        raise ValueError('Geometry signature/decorators changed')
    body = changed.body
    if not 9 <= len(body) <= 13 or sum(1 for _ in ast.walk(changed)) > 250:
        raise ValueError('Bounded geometry body')
    if [dump(x) for x in body[:2]] != [dump(x) for x in original.body[:2]]:
        raise ValueError('Width calculation changed')
    initial = body[2]
    if not (isinstance(initial, ast.Assign) and len(initial.targets) == 1
            and isinstance(initial.targets[0], ast.Name) and initial.targets[0].id == 'T'
            and isinstance(initial.value, ast.Constant) and type(initial.value.value) is int
            and initial.value.value in (1, 2, 4, 8, 16, 32, 64, 128)):
        raise ValueError('Initial tile height must be one bounded power of two')
    if dump(body[3]) != dump(original.body[3]):
        raise ValueError('Existing UB shrink loop must stay unchanged')
    if [dump(x) for x in body[-4:]] != [dump(x) for x in original.body[-4:]]:
        raise ValueError('Positive height, tile count, launch count and return must stay unchanged')
    caps = body[4:-4]
    if not 1 <= len(caps) <= 5:
        raise ValueError('One to five bounded caps')
    for stmt in caps:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == 'T'
                and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == 'min' and len(stmt.value.args) == 2
                and isinstance(stmt.value.args[0], ast.Name) and stmt.value.args[0].id == 'T'):
            raise ValueError('Scalar height cap only: T = min(T, expression)')
        scalar(stmt.value)
    return {'baseline_sha256': BASELINE_SHA256,
            'candidate_sha256': hashlib.sha256(candidate).hexdigest(),
            'restriction': 'Only initial T and bounded scalar height caps; kernel, resource query, width, UB loop, launch formula unchanged',
            'source_changed': baseline != candidate, 'ast_changed': dump(before) != dump(after),
            'host_tensor_compute': False, 'no_host_tensor_computation': True}

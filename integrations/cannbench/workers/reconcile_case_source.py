"""Restore only nonfinite values demonstrably lost by JSON capture.

Pure metadata code: no evaluator, torch or CANN imports. Callers must retain
both source files and bind the returned evidence before executing an overlay.
"""
from copy import deepcopy
import hashlib
import math
import yaml

# These fields describe measurements or prose, not generated inputs.
NON_INPUT_FIELDS = frozenset({'note', 'baseline_perf_us', 't_hw_us'})


def reconcile_case(captured, source, *, captured_bytes, source_bytes,
                   expected_captured_sha256, expected_source_sha256,
                   source_commit):
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('full upstream commit required')
    for data, expected in ((captured_bytes, expected_captured_sha256),
                           (source_bytes, expected_source_sha256)):
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('source binding hash mismatch')
    def signature(value):
        if isinstance(value, float) and not math.isfinite(value):
            return ('nonfinite', 'nan' if math.isnan(value) else str(value))
        if isinstance(value, dict): return {k: signature(v) for k, v in value.items()}
        if isinstance(value, list): return [signature(v) for v in value]
        return (type(value).__name__, value)
    for supplied, raw in ((captured, captured_bytes), (source, source_bytes)):
        document = yaml.safe_load(raw)
        matching = [c for c in document['cases'] if c.get('case_id') == supplied.get('case_id')]
        if len(matching) != 1 or signature(matching[0]) != signature(supplied):
            raise ValueError('case is not bound to hashed source bytes')
    changes = []
    def compare(a, b, path):
        if isinstance(b, float) and not math.isfinite(b):
            if a is not None:
                raise ValueError(f'nonfinite source lacks captured null at {path}')
            label = 'nan' if math.isnan(b) else ('+inf' if b > 0 else '-inf')
            changes.append({'path': path, 'source_value': label})
            return b
        if type(a) is not type(b):
            raise ValueError(f'type mismatch at {path}')
        if isinstance(b, dict):
            ignored = NON_INPUT_FIELDS if not path else frozenset()
            ka = set(a) - ignored; kb = set(b) - ignored
            if ka != kb:
                raise ValueError(f'field mismatch at {path}')
            result = deepcopy(a)
            for k in sorted(kb): result[k] = compare(a[k], b[k], path + [k])
            return result
        if isinstance(b, list):
            if len(a) != len(b): raise ValueError(f'length mismatch at {path}')
            return [compare(x, y, path + [i]) for i, (x, y) in enumerate(zip(a, b))]
        if a != b: raise ValueError(f'value mismatch at {path}')
        return deepcopy(a)
    restored = compare(captured, source, [])
    return restored, {'source_commit': source_commit,
                      'captured_sha256': expected_captured_sha256,
                      'source_sha256': expected_source_sha256,
                      'case_id': captured.get('case_id'), 'changes': changes}


def reconcile_api_float_types(captured_bytes, api_bytes, case_id, *,
                              expected_captured_sha256, expected_api_sha256):
    """Restore finite JSON numbers misread as strings by YAML 1.1.

    Every other field must match the raw official API case exactly. This does
    not coerce user strings, replace nulls, or import an older YAML contract.
    """
    import json
    for data, expected in ((captured_bytes, expected_captured_sha256),
                           (api_bytes, expected_api_sha256)):
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('API restoration source binding mismatch')
    def select(cases):
        matches = [c for c in cases if type(c.get('case_id')) is int and c['case_id'] == case_id]
        if len(matches) != 1:
            raise ValueError('API restoration case missing or duplicated')
        return matches[0]
    captured = select(yaml.safe_load(captured_bytes)['cases'])
    api = select(json.loads(api_bytes)['cases_yaml'])
    changes = []
    def compare(a, b, path):
        if type(a) is str and type(b) is float and math.isfinite(b):
            encoded = json.dumps(b, allow_nan=False)
            if a == encoded and type(yaml.safe_load(encoded)) is str:
                changes.append({'path': path, 'source_value': b, 'captured_string': a})
                return b
        if type(a) is not type(b):
            raise ValueError(f'API type mismatch at {path}')
        if isinstance(a, dict):
            if set(a) != set(b): raise ValueError(f'API field mismatch at {path}')
            return {k: compare(a[k], b[k], path + [k]) for k in a}
        if isinstance(a, list):
            if len(a) != len(b): raise ValueError(f'API length mismatch at {path}')
            return [compare(x, y, path + [i]) for i, (x, y) in enumerate(zip(a, b))]
        if a != b: raise ValueError(f'API value mismatch at {path}')
        return deepcopy(a)
    restored = compare(captured, api, [])
    return restored, {'source_kind': 'official-api-json-float-types',
                      'captured_sha256': expected_captured_sha256,
                      'source_sha256': expected_api_sha256,
                      'case_id': case_id, 'changes': changes}


def reconcile_api_then_nonfinite(captured_bytes, api_bytes, source_bytes, case_id, *,
                                 expected_captured_sha256, expected_api_sha256,
                                 expected_source_sha256, source_commit):
    """Compose two independently bound serialization repairs without source drift."""
    fixed, api_evidence = reconcile_api_float_types(
        captured_bytes, api_bytes, case_id,
        expected_captured_sha256=expected_captured_sha256,
        expected_api_sha256=expected_api_sha256)
    normalized = yaml.safe_dump({'cases': [fixed]}, sort_keys=True).encode()
    matches = [c for c in yaml.safe_load(source_bytes)['cases'] if c.get('case_id') == case_id]
    if len(matches) != 1:
        raise ValueError('upstream case missing or duplicated')
    restored, source_evidence = reconcile_case(
        fixed, matches[0], captured_bytes=normalized, source_bytes=source_bytes,
        expected_captured_sha256=hashlib.sha256(normalized).hexdigest(),
        expected_source_sha256=expected_source_sha256, source_commit=source_commit)
    if not api_evidence['changes'] or not source_evidence['changes']:
        raise ValueError('composed repair requires both demonstrated losses')
    return restored, {**source_evidence,
                      'source_kind': 'api-float-types-then-upstream-nonfinite',
                      'normalized_captured_sha256': source_evidence['captured_sha256'],
                      'captured_sha256': expected_captured_sha256,
                      'api_sha256': expected_api_sha256,
                      'api_restoration': api_evidence}


def load_verified_overlay(root, manifest_sha256, task_dir, case_id):
    """Recompute a staged restoration from its hashed sources before use.

    Returns None when this operator/case has no restoration. The manifest hash
    must come from the supervisor's reviewed launch configuration.
    """
    import json
    from pathlib import Path
    root = Path(root).resolve()
    task_dir = Path(task_dir).resolve()
    manifest_bytes = (root / 'MANIFEST.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ValueError('overlay manifest hash mismatch')
    manifest = json.loads(manifest_bytes)
    records = [r for r in manifest['records']
               if r['operator'] == task_dir.name and r['case_id'] == case_id]
    if not records:
        return None
    if len(records) != 1:
        raise ValueError('duplicate overlay case')
    record = records[0]
    def contained(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('overlay path escapes root')
        return path
    captured = (task_dir / 'cases.yaml').read_bytes()
    archived = contained(task_dir.name + '/captured.cases.yaml').read_bytes()
    if captured != archived:
        raise ValueError('overlay does not bind current catalog case file')
    if record.get('source_kind') == 'api-float-types-then-upstream-nonfinite':
        api = contained(task_dir.name + '/task.raw.json').read_bytes()
        if api != (task_dir / 'task.raw.json').read_bytes():
            raise ValueError('overlay does not bind current raw API task')
        source = contained(task_dir.name + '/upstream.cases.yaml').read_bytes()
        restored, evidence = reconcile_api_then_nonfinite(
            captured, api, source, case_id,
            expected_captured_sha256=record['captured_sha256'],
            expected_api_sha256=record['api_sha256'],
            expected_source_sha256=record['source_sha256'], source_commit=record['source_commit'])
        if evidence['api_restoration'] != record['api_restoration']:
            raise ValueError('API restoration evidence mismatch')
    elif record.get('source_kind') == 'official-api-json-float-types':
        source = contained(task_dir.name + '/task.raw.json').read_bytes()
        if source != (task_dir / 'task.raw.json').read_bytes():
            raise ValueError('overlay does not bind current raw API task')
        restored, evidence = reconcile_api_float_types(
            captured, source, case_id,
            expected_captured_sha256=record['captured_sha256'],
            expected_api_sha256=record['source_sha256'])
    else:
        if record.get('source_kind') not in (None, 'upstream-nonfinite'):
            raise ValueError('unknown overlay source kind')
        source = contained(task_dir.name + '/upstream.cases.yaml').read_bytes()
        def select(raw):
            matches = [c for c in yaml.safe_load(raw)['cases'] if c['case_id'] == case_id]
            if len(matches) != 1:
                raise ValueError('source case missing or duplicated')
            return matches[0]
        restored, evidence = reconcile_case(
            select(captured), select(source), captured_bytes=captured,
            source_bytes=source, expected_captured_sha256=record['captured_sha256'],
            expected_source_sha256=record['source_sha256'], source_commit=record['source_commit'])
    if evidence['changes'] != record['changes'] or not evidence['changes']:
        raise ValueError('overlay restoration evidence mismatch')
    overlay_bytes = contained(record['overlay']).read_bytes()
    if hashlib.sha256(overlay_bytes).hexdigest() != record['overlay_sha256']:
        raise ValueError('overlay case hash mismatch')
    # Compare complete canonical YAML, including types and NaN/Inf. Python's
    # equality treats True == 1 and NaN != NaN, so it is unsuitable here.
    expected = yaml.safe_dump({'cases': [restored]}, sort_keys=True)
    if yaml.safe_dump(yaml.safe_load(overlay_bytes), sort_keys=True) != expected:
        raise ValueError('overlay case differs from recomputed restoration')
    return restored, {**evidence, 'overlay_manifest_sha256': manifest_sha256,
                      'overlay_case_sha256': record['overlay_sha256']}

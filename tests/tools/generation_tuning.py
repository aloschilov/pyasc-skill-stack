"""One explicit tuner proposal, fresh review and external measurement handoff.

No measurement command, provider fallback, upload, automatic retry or NPU call.
Device receipts are supplied by a trusted caller, never by the generated client.
Their authenticity remains the caller's responsibility; hashes prove continuity.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import re
import signal
import subprocess
from urllib.parse import urlsplit

from generation_adapters import task_contract
from generation_compare import require, utc
from generation_executor import external, save
from generation_plan import ROOT, canonical, digest, exact_keys, write_plan
from generation_review import binding, build_review, read, run_review, seal, validate_plan


@contextmanager
def locked(root):
    """Serialize transitions in this prepared handoff; model lease is still shared."""
    with (root/'tuning.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read_measurement(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024*1024, 'Invalid measurement file')
    return read(path)


def measurement(value, bound, runtime, contract):
    """Validate a caller's device receipt, not an independently attested NPU run."""
    exact_keys(value, ('schema_version', 'kind', 'candidate_sha256', 'task_sha256',
        'pyasc_revision', 'backend', 'hardware', 'method', 'measured_at', 'source_url',
        'source_sha256', 'cases'), 'external measurement')
    require(type(value['schema_version']) is int and value['schema_version'] == 1
        and value['kind'] == 'caller-device-measurement' and value['backend'] == 'NPU', 'Device receipt required; simulator time is not performance')
    require(value['candidate_sha256'] == bound['candidate_sha256'] and value['task_sha256'] == bound['task_sha256']
        and value['pyasc_revision'] == runtime['source_revision'], 'Measurement candidate/task/runtime differs')
    require(all(isinstance(value[k], str) and value[k].strip() for k in ('hardware', 'method')), 'Missing hardware/measurement method')
    require(utc(value['measured_at']) <= datetime.now(timezone.utc), 'Future measurement')
    source = urlsplit(value['source_url'])
    require(source.scheme == 'https' and source.netloc and not source.username and not source.password and not source.query
        and re.fullmatch(r'[0-9a-f]{64}', value['source_sha256']), 'Stable evidence URL and checksum required')
    rows = value['cases']; require(isinstance(rows, list) and len(rows) == len(contract['seeds']), 'Incomplete measured case inventory')
    expected = {f'seed-{seed}' for seed in contract['seeds']}; seen = set()
    for row in rows:
        exact_keys(row, ('id', 'elapsed_us', 'samples', 'numerically_correct', 'device_kernels'), 'measured case')
        require(row['id'] in expected and row['id'] not in seen, 'Unexpected/duplicate measured case'); seen.add(row['id'])
        require(row['numerically_correct'] is True and type(row['samples']) is int and row['samples'] >= 2, 'Missing repeated correct measurement')
        require(type(row['elapsed_us']) in (int, float) and math.isfinite(row['elapsed_us']) and row['elapsed_us'] > 0, 'Invalid device timing')
        kernels = row['device_kernels']
        require(isinstance(kernels, list) and kernels and all(isinstance(k, str) and k.strip() for k in kernels), 'Missing device kernels')
    require(seen == expected, 'Missing measured cases')
    return value


def handoff_inputs(value):
    return {'incumbent.py': value['candidate'].encode(), 'incumbent-design.md': value['design'].encode(),
        'tuning-context.json': canonical({'measurement': value['measurement'], 'budget': value['budget']})}


def validate_handoff(value, plan, source=ROOT):
    exact_keys(value, ('schema_version', 'candidate', 'design', 'binding', 'measurement', 'budget',
        'incumbent_index_sha256', 'incumbent_plan_sha256'), 'tuning handoff')
    require(type(value['schema_version']) is int and value['schema_version'] == 1, 'Unknown tuning handoff')
    require(all(isinstance(value[n], str) and 0 < len(value[n].encode()) <= 1024*1024 for n in ('candidate', 'design')), 'Missing/large incumbent bytes')
    contract = task_contract(source, plan['common']['cells'][0])
    require(binding(value['candidate'].encode(), value['design'].encode(), contract) == value['binding'], 'Incumbent bytes/task changed')
    measurement(value['measurement'], value['binding'], plan['common']['runtime'], contract)
    require(value['budget'] == {'proposals': 1, 'external_measurements': 1, 'minimum_observed_speedup': 1.01}, 'Unexpected tuner budget/criterion')
    for k in ('incumbent_index_sha256', 'incumbent_plan_sha256'):
        require(isinstance(value[k], str) and re.fullmatch(r'[0-9a-f]{64}', value[k]), 'Missing incumbent provenance')
    return value


def prepare(output, incumbent, device_receipt):
    """Freeze the verified incumbent and one proposal plan; no model call."""
    from generation_review_evidence import Evidence, report_review
    output = external(output); incumbent = external(incumbent)
    require(not output.exists(), 'Never overwrite a tuning handoff')
    require(not output.is_relative_to(incumbent), 'Keep the tuning handoff outside its immutable incumbent')
    report = report_review(incumbent); require(report['validated_acceptance'], 'Incumbent lacks review/numerical acceptance')
    e = Evidence(incumbent); old_plan = e.json('plan.json'); row = e.json('state.json')['cycles'][-1]
    plan = build_review()
    require(plan['common']['cells'] == old_plan['common']['cells'] and plan['common']['runtime'] == old_plan['common']['runtime'], 'Incumbent task/runtime no longer matches')
    handoff = {'schema_version': 1, 'candidate': e.bytes(row['candidate']).decode(), 'design': e.bytes(row['design']).decode(),
        'binding': row['binding'], 'measurement': read_measurement(Path(device_receipt)),
        'budget': {'proposals': 1, 'external_measurements': 1, 'minimum_observed_speedup': 1.01},
        'incumbent_index_sha256': digest((incumbent/'files.json').read_bytes()), 'incumbent_plan_sha256': old_plan['plan_sha256']}
    validate_handoff(handoff, plan)
    plan['tuning'] = handoff
    plan['checkout_revision'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    output.mkdir(parents=True)
    (output/'tuning.lock').touch()
    # Only indexed public bytes. In particular, never copy private provider/native state.
    for name in e.index:
        path = output/'incumbent'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(e.bytes(name))
    (output/'incumbent/files.json').write_bytes((incumbent/'files.json').read_bytes())
    plan = write_plan(plan, output/'plan')
    save(output/'state.json', {'status': 'prepared', 'plan_sha256': plan['plan_sha256'],
        'selected_candidate_sha256': handoff['binding']['candidate_sha256'], 'proposal_started': False,
        'measurement_consumed': False, 'hardware_invoked': False})
    seal(output)
    return plan


def verify(root):
    from generation_review_evidence import Evidence, report_review
    e = Evidence(root); plan = e.json('plan/plan.json'); handoff = plan['tuning']
    unsigned = dict(plan); signature = unsigned.pop('plan_sha256')
    require(digest(canonical(unsigned)) == signature, 'Tuning plan changed')
    validate_handoff(handoff, plan, source=Path(root)/'incumbent/source')
    require(digest(e.bytes('incumbent/files.json')) == handoff['incumbent_index_sha256'], 'Retained incumbent changed')
    old_plan = e.json('incumbent/plan.json')
    require(old_plan['plan_sha256'] == handoff['incumbent_plan_sha256'], 'Incumbent plan changed')
    require(all(old_plan['common'][k] == plan['common'][k] for k in ('cells', 'runtime')), 'Tuning task/runtime differs from incumbent')
    require(e.json('incumbent/state.json')['cycles'][-1]['binding'] == handoff['binding'], 'Incumbent design/task binding changed')
    accepted = report_review(Path(root)/'incumbent')
    require(accepted['validated_acceptance'] and accepted['candidate_sha256'] == handoff['binding']['candidate_sha256'], 'Incumbent acceptance changed')
    state = e.json('state.json'); require(state['plan_sha256'] == signature and state['hardware_invoked'] is False
        and type(state['measurement_consumed']) is bool and type(state['proposal_started']) is bool, 'Tuning state/plan mismatch')
    return plan, state


def propose(root, approve, state_dir):
    root = external(root)
    with locked(root): return _propose(root, approve, state_dir)


def _propose(root, approve, state_dir):
    root = external(root); plan, state = verify(root)
    require(approve == plan['plan_sha256'], 'Exact tuning plan approval required')
    require(state['status'] == 'prepared' and not state['proposal_started'] and not (root/'proposal').exists(), 'Never replay a tuning proposal')
    validate_plan(root/'plan/plan.json')
    state.update(status='proposing', proposal_started=True); save(root/'state.json', state); seal(root)
    try:
        result = run_review(plan, root/'proposal', state_dir, initial_inputs=handoff_inputs(plan['tuning']))
        state.update(status='awaiting-measurement' if result['status'] == 'accepted' else 'blocked' if result['status'] == 'blocked' else 'kept-incumbent',
            proposal_status=result['status'], proposed_candidate_sha256=result['accepted_candidate_sha256'])
    except BaseException:
        state['status'] = 'blocked'; raise
    finally:
        # Even an accepted simulator proposal cannot change the selected incumbent.
        save(root/'state.json', state); seal(root)
    return state


def select(root, device_receipt=None):
    """File-only selection; a missing or invalid receipt never promotes a proposal."""
    root = external(root)
    with locked(root): return _select(root, device_receipt)


def _select(root, device_receipt):
    from generation_review_evidence import report_review
    root = external(root); plan, state = verify(root)
    require(state['status'] == 'awaiting-measurement' and not state['measurement_consumed'], 'No unmeasured accepted proposal')
    handoff = plan['tuning']; before = handoff['measurement']; contract = task_contract(root/'incumbent/source', plan['common']['cells'][0])
    result = {'status': 'kept-incumbent', 'selected_candidate_sha256': handoff['binding']['candidate_sha256'],
        'measurement_trust': 'caller-supplied; authenticity not independently verified by this selector',
        'hardware_invoked': False, 'general_performance_claim': False, 'observed_speedup': None}
    proposal = report_review(root/'proposal')
    require(proposal['validated_acceptance'] and proposal['candidate_sha256'] == state['proposed_candidate_sha256']
        and proposal['plan_sha256'] == plan['plan_sha256'], 'Proposal review/verification changed')
    if device_receipt is None:
        result['reason'] = 'No measurement supplied; incumbent retained'
    else:
        try:
            require(not Path(device_receipt).resolve().is_relative_to(root), 'Measurement must be supplied outside model/handoff directories')
            after = read_measurement(Path(device_receipt))
            bound = dict(handoff['binding'], candidate_sha256=proposal['candidate_sha256'])
            measurement(after, bound, plan['common']['runtime'], contract)
            # Store only validated receipt fields, never arbitrary failed input.
            save(root/'proposed-measurement.json', after)
            ratio = compare_measurements(before, after)
            result['observed_speedup'] = ratio
            if ratio > handoff['budget']['minimum_observed_speedup']:
                result.update(status='promoted', selected_candidate_sha256=proposal['candidate_sha256'], reason='Observed device timing ratio exceeds the predeclared threshold')
            else: result['reason'] = 'No sufficient measured improvement'
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            result['reason'] = 'Invalid or incomparable external measurement; incumbent retained'
    state.update(status=result['status'], selected_candidate_sha256=result['selected_candidate_sha256'], measurement_consumed=True)
    save(root/'decision.json', result); save(root/'state.json', state); seal(root)
    return result


def compare_measurements(before, after):
    require(all(after[k] == before[k] for k in ('hardware', 'method', 'pyasc_revision', 'task_sha256')), 'Incomparable measurement scope')
    require(after['source_sha256'] != before['source_sha256'], 'Measurement reuses incumbent evidence')
    old = {c['id']: c for c in before['cases']}; new = {c['id']: c for c in after['cases']}
    require(set(old) == set(new) and all(old[k]['samples'] == new[k]['samples'] for k in old), 'Measurement sampling/cases differ')
    ratio = math.exp(math.fsum(math.log(old[k]['elapsed_us'])-math.log(new[k]['elapsed_us']) for k in old)/len(old))
    require(math.isfinite(ratio), 'Nonfinite observed speedup')
    return ratio


def report(root):
    """Recheck selection from immutable role/measurement evidence, without calls."""
    from generation_review_evidence import report_review
    root = external(root); plan, state = verify(root); handoff = plan['tuning']
    require(state['status'] in ('prepared', 'proposing', 'awaiting-measurement', 'kept-incumbent', 'promoted', 'blocked'), 'Unknown tuning state')
    selected = handoff['binding']['candidate_sha256']
    result = {'status': state['status'], 'selected_candidate_sha256': selected, 'hardware_invoked': False,
        'measurement_trust': 'caller-supplied; authenticity not independently verified by this selector',
        'general_performance_claim': False, 'observed_speedup': None}
    if state['status'] == 'promoted' or state.get('proposal_status') == 'accepted':
        proposal = report_review(root/'proposal')
        require(proposal['validated_acceptance'] and proposal['candidate_sha256'] == state['proposed_candidate_sha256']
            and proposal['plan_sha256'] == plan['plan_sha256'], 'Proposal acceptance changed')
    if state['measurement_consumed']:
        decision = read(root/'decision.json')
        require(decision['status'] == state['status'] and decision['hardware_invoked'] is False
            and decision['general_performance_claim'] is False and decision['measurement_trust'] == result['measurement_trust'], 'Overstated tuning decision')
        if decision['observed_speedup'] is not None:
            after = read(root/'proposed-measurement.json')
            bound = dict(handoff['binding'], candidate_sha256=state['proposed_candidate_sha256'])
            measurement(after, bound, plan['common']['runtime'], task_contract(root/'incumbent/source', plan['common']['cells'][0]))
            ratio = compare_measurements(handoff['measurement'], after)
            require(decision['observed_speedup'] == ratio, 'Observed tuning ratio changed')
            promoted = ratio > handoff['budget']['minimum_observed_speedup']
            require((state['status'] == 'promoted') == promoted, 'Selection differs from approved criterion')
            if promoted: selected = state['proposed_candidate_sha256']
            result['observed_speedup'] = ratio
        else: require(state['status'] == 'kept-incumbent', 'Unmeasured promotion')
        require(decision['selected_candidate_sha256'] == selected, 'Decision selects another candidate')
    else: require(state['status'] != 'promoted', 'Promotion has no measurement receipt')
    require(state['selected_candidate_sha256'] == selected, 'Selected incumbent changed')
    result['selected_candidate_sha256'] = selected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'propose', 'select', 'report'))
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--incumbent', type=Path)
    parser.add_argument('--measurement', type=Path); parser.add_argument('--approve-plan', default='')
    parser.add_argument('--state-dir', type=Path, default=Path.home()/'.local/state/pyasc-generation')
    args = parser.parse_args()
    if args.action == 'plan':
        require(args.incumbent is not None and args.measurement is not None and not args.approve_plan, 'Incumbent and its measurement required')
        result = prepare(args.root, args.incumbent, args.measurement); print(result['plan_sha256']); return 0
    require(args.incumbent is None, 'Use the frozen incumbent')
    if args.action == 'propose':
        require(args.measurement is None, 'Device measurement belongs after proposal qualification')
        result = propose(args.root, args.approve_plan, args.state_dir)
    elif args.action == 'select':
        require(not args.approve_plan, 'Selection does not execute a model')
        result = select(args.root, args.measurement)
    else:
        require(not args.approve_plan and args.measurement is None, 'Report only rechecks retained data')
        result = report(args.root)
    print(canonical(result).decode()); return 2 if result['status'] == 'blocked' else 0


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    raise SystemExit(main())

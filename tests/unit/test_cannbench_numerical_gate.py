import hashlib
import io
import zipfile
import warnings
import copy
import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integrations/cannbench/ci'))
gate = importlib.import_module('numerical_gate')


class NumericalGateTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT / 'integrations/cannbench/catalogs/official-tasks-1.1.2-20261001/manifest.json').read_text())
        self.bundle = {'catalog_sha256': self.catalog['catalog_sha256'], 'operators': {}}
        runtime = {'extension_sha256': {'lib.so': 'native'}, 'wheel_metadata_sha256': 'wheel'}
        self.policy = {'catalog_sha256': self.catalog['catalog_sha256'], 'pyasc_commit': gate.PIN,
                       'runner_sha256': ['runner'], 'runtime': runtime,
                       'kernel_eval_file_sha256': {'eval': 'hash', 'kernel_eval.data.data_generator': 'generator'}, 'cases': {}}
        self.reports = []
        for spec in self.catalog['operators']:
            op = spec['function_name']
            self.bundle['operators'][op] = {'candidate_sha256': op, 'models': {phase: 'gateway/glm-5.3' for phase in ('design', 'implement', 'review')}}
            for case_id in spec['case_ids']:
                case = int(case_id.rsplit('_', 1)[1])
                contract = {'thresholds': {'mere': .01}, 'ignore_output_indices': [], 'has_get_input': False, 'seed': 0}
                contract['input_generation'] = {
                    'case_id': case_id, 'eval_seed': 0,
                    'case_seed': int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:8], 'big') % (2**31),
                    'catalog_manifest_sha256': 'manifest',
                }
                contract['reference_functions'] = {'oracle': op+'_oracle', 'native': op}
                contract['index_gather_check'] = {'required': False, 'passed': True}
                if op in {'cummin', 'arg_max', 'top_k'}:
                    contract['index_gather_check'].update(required=True,
                        source_sha256='5ba91699fd586345a0bc715a469259feddb33cee9afca3674d8a009d69efdc12',
                        declarations=[{'index_gather': {'input': 'x'}}])
                self.policy['cases'][f'{op}/{case}'] = dict(contract, source_restoration=None)
                self.reports.append(dict(contract, operator=op, case_id=case, status='passed', phase='done',
                    evidence='camodel-case', accuracy={'passed': True}, candidate_sha256=op, runner_sha256='runner',
                    provenance={'runtime': dict(runtime, expect_source=gate.PIN, has_native_extension=True),
                        'kernel_eval_file_sha256': {'eval': 'hash', 'kernel_eval.data.data_generator': 'generator'},
                        'task_files_sha256': {n: spec['task_files'][f'tasks/{op}/{n}'] for n in ('proto.yaml','cases.yaml','golden.py')}}))

                if op == 'moe_gating_top_k_softmax' and case in {6, 7, 8}:
                    self.reports[-1]['provenance']['bool_generator_policy'] = {
                        'scope': 'official-generator-unmodified',
                        'source_sha256': 'generator'}

    def validate(self):
        return gate.validate_reports(self.catalog, self.bundle, self.reports, self.policy)

    def test_full_official_matrix_is_required(self):
        self.assertEqual(self.validate()['cases'], 1060)
        self.reports.pop()
        with self.assertRaises(ValueError): self.validate()

    def test_duplicate_cannot_replace_missing_case(self):
        self.reports[-1] = copy.deepcopy(self.reports[0])
        with self.assertRaises(ValueError): self.validate()

    def test_reports_cannot_self_approve_source_or_execution_drift(self):
        original = copy.deepcopy(self.reports[0])
        mutations = [('status', 'timeout'), ('candidate_sha256', 'changed'), ('runner_sha256', 'changed'),
                     ('evidence', 'compiler'), ('seed', True), ('thresholds', {'mere': 1})]
        for field, value in mutations:
            with self.subTest(field=field):
                self.reports[0] = dict(copy.deepcopy(original), **{field: value})
                with self.assertRaises(ValueError): self.validate()
        for field in ('runtime', 'task_files_sha256', 'kernel_eval_file_sha256'):
            with self.subTest(field=field):
                self.reports[0] = copy.deepcopy(original)
                self.reports[0]['provenance'][field] = {}
                with self.assertRaises(ValueError): self.validate()
        self.reports[0] = copy.deepcopy(original)
        self.reports[0]['provenance']['source_restoration'] = {'unreviewed': True}
        with self.assertRaises(ValueError): self.validate()

    def test_other_implementation_model_is_rejected(self):
        self.bundle['operators']['exp']['models']['implement'] = 'qwen'
        with self.assertRaises(ValueError): self.validate()

    def test_other_review_or_design_model_is_rejected(self):
        for phase in ('design', 'review'):
            with self.subTest(phase=phase):
                models = self.bundle['operators']['exp']['models']
                models[phase] = 'gateway/qwen3.8-max'
                with self.assertRaises(ValueError): self.validate()
                models[phase] = 'gateway/glm-5.3'

    def test_policy_must_cover_every_input_contract(self):
        self.policy['cases'].pop('exp/1')
        with self.assertRaises(ValueError): self.validate()

    def test_skipped_index_check_cannot_be_approved_by_policy(self):
        for op in ('cummin', 'arg_max', 'top_k'):
            report = next(r for r in self.reports if r['operator'] == op)
            report['index_gather_check'] = {'required': False, 'passed': True}
            self.policy['cases'][f"{op}/{report['case_id']}"]['index_gather_check'] = report['index_gather_check']
            with self.assertRaisesRegex(ValueError, 'not executed'):
                self.validate()

    def test_legacy_reports_without_index_evidence_are_rejected(self):
        self.reports[0].pop('index_gather_check')
        with self.assertRaisesRegex(ValueError, 'Missing or failed'):
            self.validate()

    def test_legacy_zero_bool_generator_is_rejected(self):
        report = next(r for r in self.reports if r['operator']=='moe_gating_top_k_softmax' and r['case_id']==6)
        report['provenance'].pop('bool_generator_policy')
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.validate()

    def test_incorrect_seed_cannot_be_approved_by_policy(self):
        report = self.reports[0]
        report['input_generation']['case_seed'] += 1
        self.policy['cases']['exp/1']['input_generation'] = report['input_generation']
        with self.assertRaisesRegex(ValueError, 'seeded'):
            self.validate()

    def test_missing_reference_selection_rejected(self):
        self.reports[0].pop('reference_functions')
        self.policy['cases']['exp/1'].pop('reference_functions')
        with self.assertRaisesRegex(ValueError, 'reference'):
            self.validate()


class ArchiveCandidateBindingTests(unittest.TestCase):
    def archive(self, *, changed=False, duplicate=False, missing=False):
        catalog = {'operators': [{'function_name': 'exp'}, {'function_name': 'sigmoid'}]}
        bundle = {'operators': {op: {'candidate_sha256': hashlib.sha256(op.encode()).hexdigest()}
                                for op in ('exp', 'sigmoid')}}
        wheel_buf = io.BytesIO()
        with zipfile.ZipFile(wheel_buf, 'w') as wheel:
            wheel.writestr('cann_bench/exp.py', b'changed' if changed else b'exp')
            if not missing:
                wheel.writestr('cann_bench/sigmoid.py', b'sigmoid')
            if duplicate:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    wheel.writestr('cann_bench/exp.py', b'exp')
        bundle['wheel_sha256'] = hashlib.sha256(wheel_buf.getvalue()).hexdigest()
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, 'w') as archive:
            archive.writestr('dist/candidates.whl', wheel_buf.getvalue())
        return zipfile.ZipFile(outer), bundle, catalog

    def test_exact_sources_match(self):
        source, bundle, catalog = self.archive()
        with source:
            gate.validate_archive_candidates(source, bundle, catalog)

    def test_repacked_wheel_cannot_reuse_other_candidate_reports(self):
        source, bundle, catalog = self.archive(changed=True)
        with source, self.assertRaisesRegex(ValueError, 'differs from wheel'):
            gate.validate_archive_candidates(source, bundle, catalog)

    def test_missing_candidate_rejected(self):
        source, bundle, catalog = self.archive(missing=True)
        with source, self.assertRaisesRegex(ValueError, 'differs from wheel'):
            gate.validate_archive_candidates(source, bundle, catalog)

    def test_duplicate_wheel_member_rejected(self):
        source, bundle, catalog = self.archive(duplicate=True)
        with source, self.assertRaisesRegex(ValueError, 'Duplicate wheel'):
            gate.validate_archive_candidates(source, bundle, catalog)

    def test_wrong_wheel_digest_rejected(self):
        source, bundle, catalog = self.archive()
        bundle['wheel_sha256'] = 'wrong'
        with source, self.assertRaisesRegex(ValueError, 'wheel hash'):
            gate.validate_archive_candidates(source, bundle, catalog)

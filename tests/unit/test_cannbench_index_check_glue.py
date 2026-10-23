"""Raw declarations and fail-closed binding without importing torch on Mac."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'integrations/cannbench/workers'))
import run_camodel_cases as runner


class IndexCheckGlueTests(unittest.TestCase):
    def write_proto(self, directory, outputs):
        Path(directory, 'proto.yaml').write_text(json.dumps({'operator': {'outputs': outputs}}))

    def test_no_declaration_does_not_import_checker(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_proto(d, [{'name': 'y'}])
            with patch.object(runner.importlib.util, 'module_from_spec') as load:
                self.assertEqual(runner.validate_declared_indices(d, {}, {}, None),
                                 {'required': False, 'passed': True})
                load.assert_not_called()

    def test_raw_declaration_and_original_inputs_reach_official_checker(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_proto(d, [{'name': 'indices', 'compare': False,
                                 'index_gather': {'input': 'x', 'value_reduce': 'amax'}}])
            params = {'x': object()}
            attrs = {'dim': 1}
            candidate = object()
            seen = []
            def validate(op, passed_params, passed_attrs, output):
                seen.append(op.outputs[0].index_gather)
                self.assertIs(params, passed_params)
                self.assertIs(attrs, passed_attrs)
                self.assertIs(candidate, output)
                return False, 'wrong indices'
            module = SimpleNamespace(validate_index_gather_outputs=validate)
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None))
            with patch.object(runner.importlib.util, 'spec_from_file_location', return_value=spec), \
                 patch.object(runner.importlib.util, 'module_from_spec', return_value=module):
                result = runner.validate_declared_indices(d, params, attrs, candidate)
            self.assertFalse(result['passed'])
            self.assertTrue(result['required'])
            self.assertEqual(result['source_sha256'], runner.INDEX_CHECK_SHA256)
            self.assertEqual(seen, [{'input': 'x', 'value_reduce': 'amax'}])

    def test_hash_mismatch_fails_before_import(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_proto(d, [{'name': 'idx', 'index_gather': {'input': 'x'}}])
            with patch.object(runner, '_sha256_file', return_value='wrong'):
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    runner.validate_declared_indices(d, {}, {}, None)

    def test_empty_declaration_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_proto(d, [{'name': 'idx', 'index_gather': {}}])
            with self.assertRaisesRegex(ValueError, 'Invalid official'):
                runner.validate_declared_indices(d, {}, {}, None)

    def test_resume_rejects_changed_candidate_runner_and_contract(self):
        import copy
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ('proto.yaml', 'cases.yaml', 'golden.py', 'exp.py'):
                (root / name).write_text(name)
            summary = {'runner_sha256': 'current', 'kernel_eval_file_sha256': {'eval': 'sha'},
                       'expect_pyasc_source': 'pin', 'seed': None}
            prior = {'operator': 'exp', 'case_id': 1, 'phase': 'done',
                     'accuracy': {'passed': True}, 'runner_sha256': 'current', 'seed': None,
                     'candidate_sha256': runner._sha256_file(root/'exp.py'),
                     'index_gather_check': {'required': False, 'passed': True},
                     'provenance': {'runtime': {'expect_source': 'pin'},
                         'kernel_eval_file_sha256': {'eval': 'sha'},
                         'task_files_sha256': {n: runner._sha256_file(root/n)
                             for n in ('proto.yaml', 'cases.yaml', 'golden.py')}}}
            def validate(record):
                runner.validate_resumed_pass(record, op='exp', case_id=1, summary=summary,
                                             task_dir=root, candidate=root/'exp.py')
            validate(prior)
            for key, value in [('runner_sha256', 'old'), ('seed', 0), ('case_id', True),
                               ('index_gather_check', {}), ('candidate_sha256', 'old')]:
                with self.subTest(key=key):
                    changed = copy.deepcopy(prior); changed[key] = value
                    with self.assertRaisesRegex(ValueError, 'Cannot reuse'):
                        validate(changed)
            (root/'exp.py').write_text('new kernel')
            with self.assertRaisesRegex(ValueError, 'Cannot reuse'):
                validate(prior)
            self.assertTrue((root/'exp.py').exists())

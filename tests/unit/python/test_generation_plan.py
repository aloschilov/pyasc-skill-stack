"""Planning contracts: no provider calls, runtime imports or qualified results."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
import generation_plan as planner


class GenerationPlanning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'checkout'; self.root.mkdir()
        for name in ('skills', 'reference/asctile', 'reference/images', 'integrations/opencode/profiles', 'tests/generation'):
            shutil.copytree(ROOT/name, self.root/name, ignore=shutil.ignore_patterns('__pycache__'))
        (self.root/'tests/tools').mkdir()
        for path in (ROOT/'tests/tools').glob('*.py'):
            shutil.copyfile(path, self.root/'tests/tools'/path.name)
        self.matrix = json.loads((self.root/planner.DEFAULT_MATRIX).read_text())

    def plan(self, matrix=None):
        if matrix is not None:
            (self.root/planner.DEFAULT_MATRIX).write_text(json.dumps(matrix))
        return planner.build_plan(self.root)

    def test_plan_is_deterministic_and_never_executes(self):
        with patch.object(planner.subprocess, 'run', side_effect=AssertionError('No command during planning')):
            result = self.plan()
            self.assertEqual(result, self.plan())
        self.assertEqual(len(result['cells']), 3)
        self.assertEqual(result['maximum_work'], {'model_attempts': 3, 'output_tokens': 98304, 'execution_seconds': 1800})
        self.assertFalse(result['execution_enabled']); self.assertFalse(result['schedule_enabled'])
        self.assertEqual(result['status'], 'not-run')
        self.assertIsNone(result['model']['observed'])
        self.assertTrue(all(c['status']=='not-run' for c in result['cells']))
        self.assertIn('suppressed-launch-control', result['required_gates'])

    def test_invalid_or_over_budget_matrices_fail_before_output(self):
        modifications = [
            lambda x: x.update(cells=[]),
            lambda x: x['cells'].append(x['cells'][0]),
            lambda x: x['cells'][0].update(cases=[]),
            lambda x: x['cells'][0].update(cases=['not_a_case']),
            lambda x: x['cells'][0].update(cases=['vadd_test_2', 'vadd_test_2']),
            lambda x: x['cells'][0].update(module='../../outside.py'),
            lambda x: x['cells'][0].update(command='echo arbitrary-command'),
            lambda x: x.update(profile='../private'),
            lambda x: x.update(skills=[]),
            lambda x: x['skills'].append(x['skills'][0]),
            lambda x: x.update(schema_version=True),
            lambda x: x.update(response_model=''),
            lambda x: x.update(response_model=None),
            lambda x: x['limits'].update(attempts_per_cell=True),
            lambda x: x['limits'].update(rounds_per_attempt=0),
            lambda x: x['limits'].update(attempts_per_cell=2),
            lambda x: x['limits'].update(max_output_tokens=98303),
            lambda x: x['limits'].update(max_execution_seconds=1799),
        ]
        for modify in modifications:
            data=copy.deepcopy(self.matrix);modify(data)
            with self.subTest(data=data), self.assertRaises(ValueError):self.plan(data)

    def test_aliasing_same_task_as_different_cells_is_rejected(self):
        self.matrix['cells'][1] = dict(self.matrix['cells'][0], id='alias')
        with self.assertRaisesRegex(ValueError, 'repeated selected'): self.plan(self.matrix)

    def test_task_drift_and_unknown_runtime_fail(self):
        task=self.root/'reference/asctile/target/test_vadd.py';source=task.read_bytes()
        task.write_bytes(source+b'\n# drift\n')
        with self.assertRaisesRegex(ValueError, 'source drift'):self.plan()
        task.write_bytes(source)
        path=self.root/'reference/images/ENV.json';pin=json.loads(path.read_text())
        for change, message in [(lambda p:p.update(source_sha='a'*40), 'revisions differ'),
                                (lambda p:p['dev_image'].update(ref='image:latest'), 'Digest-pinned')]:
            value=copy.deepcopy(pin);change(value);path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, message):self.plan()

    def test_skill_change_changes_plan_identity(self):
        before=planner.digest(planner.canonical(self.plan()))
        path=self.root/'skills/pyasc-api-patterns/references/api-jit-options.md'
        path.write_text(path.read_text()+'\nChanged instruction.\n')
        self.assertNotEqual(before,planner.digest(planner.canonical(self.plan())))

    def test_executor_dependency_change_changes_plan_identity(self):
        before = planner.digest(planner.canonical(self.plan()))
        path = self.root/'tests/tools/generation_provider.py'
        path.write_text(path.read_text()+'\n# changed budget gate\n')
        self.assertNotEqual(before, planner.digest(planner.canonical(self.plan())))

    def test_profile_budget_and_external_symlink_fail(self):
        path=self.root/'integrations/opencode/profiles/cloud-glm-5.3.json'
        data=json.loads(path.read_text());data['provider']['gateway']['models']['glm-5.3']['limit']['output']=1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'model output limit'):self.plan()
        path.unlink();path.symlink_to(ROOT/'integrations/opencode/profiles/cloud-glm-5.3.json')
        with self.assertRaisesRegex(ValueError, 'external input'):self.plan()

    def test_case_parser_does_not_evaluate_expressions_and_rejects_duplicates(self):
        source=b'# PYASC_TESTS_BEGIN\n("x", side_effect()),\n# PYASC_TESTS_END\n'
        self.assertEqual(list(planner.declared_cases(source)), ['x'])
        for data in (b'no cases', source.replace(b'("x", side_effect()),', b'("x", 1), ("x", 2),')):
            with self.assertRaises(ValueError):planner.declared_cases(data)

    def test_output_is_external_new_and_hash_bound(self):
        plan=self.plan();out=Path(self.tmp.name)/'result'
        payload=planner.write_plan(plan,out)
        stored=json.loads((out/'plan.json').read_text())
        self.assertEqual(stored,payload)
        self.assertEqual(stored.pop('plan_sha256'), hashlib.sha256(planner.canonical(stored)).hexdigest())
        with self.assertRaises(FileExistsError):planner.write_plan(plan,out)
        with self.assertRaisesRegex(ValueError, 'outside checkout'):planner.write_plan(plan,ROOT/'forbidden-plan')
        link=Path(self.tmp.name)/'checkout-link';link.symlink_to(ROOT,target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'outside checkout'):planner.write_plan(plan,link/'forbidden-plan')


if __name__ == '__main__':unittest.main()

"""Schedule integrity tests without CANN or runtime imports."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'integrations/cannbench/workers'))
from run_camodel_cases import load_case_schedule
import run_camodel_cases as runner
from unittest.mock import patch

class ScheduleTests(unittest.TestCase):
    def test_order_defer_and_reject_invalid_coverage_or_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            cases = {}
            for op in ('a', 'b'):
                (root / op).mkdir()
                (root / op / 'cases.yaml').write_text('official cases')
                cases[op] = [SimpleNamespace(case_id=i) for i in (1, 2)]
            for op, i, stage in [('a', 1, 'operator_first_case'), ('b', 1, 'operator_first_case'),
                                 ('a', 2, 'remaining_by_input_size'), ('b', 2, 'requires_source_reconciliation')]:
                rows.append(dict(operator=op, case_id=i, stage=stage,
                                 cases_sha256=hashlib.sha256(b'official cases').hexdigest()))
            plan = root / 'plan.json'
            def run(data):
                plan.write_text(json.dumps(data))
                return load_case_schedule(plan, root, 'manifest', ['a', 'b'], cases)
            data = dict(catalog_manifest_sha256='manifest', cases=rows)
            schedule = run(data)
            self.assertEqual([(op,c.case_id,d) for op,c,d in schedule],
                             [('a',1,None),('b',1,None),('a',2,None),('b',2,'requires_source_reconciliation')])
            for changed in (rows[:-1], rows + [rows[0]], [dict(rows[0], case_id=99)] + rows[1:],
                            [dict(rows[0], cases_sha256='wrong')] + rows[1:]):
                with self.assertRaises(ValueError): run(dict(data, cases=changed))
            with self.assertRaises(ValueError): run(dict(data, catalog_manifest_sha256='wrong'))
            bound = dict(data, requires_source_overlay_sha256='overlay-hash')
            with self.assertRaisesRegex(ValueError, 'source overlay'): run(bound)
            plan.write_text(json.dumps(bound))
            with self.assertRaisesRegex(ValueError, 'source overlay'):
                load_case_schedule(plan,root,'manifest',['a','b'],cases,source_overlay_sha256='wrong')
            self.assertEqual(len(load_case_schedule(plan,root,'manifest',['a','b'],cases,
                                                    source_overlay_sha256='overlay-hash')),4)
            (root / 'b/cases.yaml').write_text('modified')
            with self.assertRaises(ValueError): run(data)

    def test_deferred_scheduling_budget_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = {}
            for op in ('a',):
                (root / op).mkdir()
                (root / op / 'cases.yaml').write_text('official cases')
                cases[op] = [SimpleNamespace(case_id=i) for i in (1, 2)]
            digest = hashlib.sha256(b'official cases').hexdigest()
            rows = [dict(operator='a', case_id=1, stage='operator_first_case', cases_sha256=digest),
                    dict(operator='a', case_id=2, stage='deferred_scheduling_budget', cases_sha256=digest)]
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(catalog_manifest_sha256='manifest', cases=rows)))
            schedule = load_case_schedule(plan, root, 'manifest', ['a'], cases)
            self.assertEqual([(op, c.case_id, d) for op, c, d in schedule],
                             [('a', 1, None), ('a', 2, 'deferred_scheduling_budget')])
            # a scheduling deferral is never silently accepted as a
            # source-reconciliation deferral or vice versa
            rows[1]['stage'] = 'requires_source_reconciliation'
            plan.write_text(json.dumps(dict(catalog_manifest_sha256='manifest', cases=rows)))
            schedule = load_case_schedule(plan, root, 'manifest', ['a'], cases)
            self.assertEqual(schedule[1][2], 'requires_source_reconciliation')
            rows[1]['stage'] = 'slow_case'
            plan.write_text(json.dumps(dict(catalog_manifest_sha256='manifest', cases=rows)))
            with self.assertRaises(ValueError):
                load_case_schedule(plan, root, 'manifest', ['a'], cases)
            # any deferred_<reason> stage validates; non-deferred unknowns reject
            rows[1]['stage'] = 'deferred_prior_pass'
            plan.write_text(json.dumps(dict(catalog_manifest_sha256='manifest', cases=rows)))
            schedule = load_case_schedule(plan, root, 'manifest', ['a'], cases)
            self.assertEqual(schedule[1][2], 'deferred_prior_pass')

    def test_run_parent_separates_scheduling_deferrals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = {"deferred_cases": [], "deferrals": {}}
            for cid, reason in ((2, 'deferred_scheduling_budget'),
                                (3, 'requires_source_reconciliation')):
                case = SimpleNamespace(case_id=cid)
                if reason == 'requires_source_reconciliation':
                    summary['deferred_cases'].append({'operator': 'a', 'case_id': case.case_id})
                else:
                    summary['deferrals'].setdefault(reason, []).append({'operator': 'a', 'case_id': case.case_id})
            self.assertEqual(summary['deferred_cases'], [{'operator': 'a', 'case_id': 3}])
            self.assertEqual(summary['deferrals'],
                             {'deferred_scheduling_budget': [{'operator': 'a', 'case_id': 2}]})


class BoundaryTests(unittest.TestCase):
    def test_stop_after_report_before_next_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tasks/a').mkdir(parents=True)
            (root / 'manifest.json').write_text(json.dumps({'operators':[{'function_name':'a'}]}))
            cases = [SimpleNamespace(case_id=i, rel_path='a') for i in (1,2)]
            kem = SimpleNamespace(file_sha256={}, modules={'kernel_eval.data.case_loader':
                SimpleNamespace(CaseLoader=lambda _: SimpleNamespace(scan_all_cases=lambda:cases))})
            stop = root / 'STOP'
            args = SimpleNamespace(catalog=root,candidate_root=root,kernel_eval_src=root,
                output_dir=root/'out',ops=None,timeout_seconds=1,max_rss_gb=1,platform='test',
                seed=None,force=False,expect_pyasc_version=None,expect_pyasc_source=None,stop_file=stop)
            def child(command, *_):
                result = Path(command[command.index('--result-file')+1])
                result.parent.mkdir(parents=True,exist_ok=True)
                result.write_text(json.dumps({'status':'passed'}))
                stop.touch()
                return 0,'','',False,None
            with patch.object(runner,'KernelEvalModules',return_value=kem), \
                 patch.object(runner,'create_case_loader',side_effect=lambda kem, root: kem.modules['kernel_eval.data.case_loader'].CaseLoader(str(root))), \
                 patch.object(runner,'run_bounded_case',side_effect=child) as launch:
                self.assertEqual(runner.run_parent(args),1)
            self.assertEqual(launch.call_count,1)
            summary=json.loads((root/'out/summary.json').read_text())
            self.assertEqual(summary['status'],'interrupted')
            self.assertEqual(summary['totals']['planned'],2)
            self.assertEqual(summary['totals']['passed'],1)
            self.assertFalse((root/'out/a/case_0002.json').exists())

if __name__ == '__main__': unittest.main()

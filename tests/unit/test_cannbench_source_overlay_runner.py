"""Restored contracts reach the official parser without reusing prior reports."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'integrations/cannbench/workers'))
import run_camodel_cases as runner

class OverlayRunnerTests(unittest.TestCase):
    def test_verified_case_uses_official_parser_and_preserves_provenance(self):
        loader = Mock()
        loader._parse_case.return_value = SimpleNamespace(case_id=12)
        raw = {'case_id':12,'value_range':[float('-inf'),float('inf')]}
        evidence = {'overlay_manifest_sha256':'a'*64}
        with patch('reconcile_case_source.load_verified_overlay', return_value=(raw,evidence)) as verify:
            case, provenance = runner.load_execution_case(loader,Path('/catalog/exp'),12,Path('/overlay'),'a'*64)
        verify.assert_called_once_with(Path('/overlay'),'a'*64,Path('/catalog/exp'),12)
        loader._parse_case.assert_called_once_with(raw,'exp','/catalog/exp/cases.yaml','exp')
        loader.scan_by_rel_path.assert_not_called()
        self.assertEqual(case.case_id,12);self.assertEqual(provenance,evidence)

    def test_verification_failure_never_falls_back_to_captured_case(self):
        loader = Mock()
        with patch('reconcile_case_source.load_verified_overlay', side_effect=ValueError('drift')):
            with self.assertRaisesRegex(ValueError,'drift'):
                runner.load_execution_case(loader,Path('/catalog/exp'),12,Path('/overlay'),'a'*64)
        loader._parse_case.assert_not_called();loader.scan_by_rel_path.assert_not_called()

    def test_no_restoration_keeps_original_case(self):
        original=SimpleNamespace(case_id=1)
        loader=Mock();loader.scan_by_rel_path.return_value=[original]
        with patch('reconcile_case_source.load_verified_overlay',return_value=None):
            case,evidence=runner.load_execution_case(loader,Path('/catalog/exp'),1,Path('/overlay'),'a'*64)
        self.assertIs(case,original);self.assertIsNone(evidence)
        loader._parse_case.assert_not_called()

    def test_prior_reports_rejected_even_with_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'out/exp').mkdir(parents=True)
            (root/'out/exp/case_0012.json').write_text('{"status":"passed"}')
            args=SimpleNamespace(catalog=root,candidate_root=root,output_dir=root/'out',
                                 source_overlay=root,source_overlay_sha256='a'*64,force=True)
            with patch.object(runner,'KernelEvalModules') as kem:
                with self.assertRaisesRegex(ValueError,'fresh result'):
                    runner.run_parent(args)
            kem.assert_not_called()

    def test_child_receives_pinned_overlay(self):
        args=SimpleNamespace(candidate_root='/candidate',kernel_eval_src='/eval',max_rss_gb=24,
            platform='test',seed=None,expect_pyasc_version=None,expect_pyasc_source=None,
            source_overlay='/overlay',source_overlay_sha256='a'*64)
        command=runner._child_command(args,Path('/task'),12,Path('/result'))
        self.assertEqual(command[command.index('--source-overlay')+1],'/overlay')
        self.assertEqual(command[command.index('--source-overlay-sha256')+1],'a'*64)

    def test_missing_candidate_report_precedes_all_runtime_and_input_work(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=SimpleNamespace(task_dir=root/'tasks/exp',candidate_root=root/'candidates',
                result_file=root/'out/case_0001.json',case_id=1,platform='test',seed=None,
                max_rss_gb=24,kernel_eval_src=root/'eval')
            with patch.object(runner,'KernelEvalModules') as kem, patch.object(runner,'_set_address_space_limit') as limits:
                self.assertEqual(runner.run_case_child(args),1)
            kem.assert_not_called();limits.assert_not_called()
            report=json.loads(args.result_file.read_text())
            self.assertEqual(report['failure_class'],'missing_candidate')
            self.assertEqual(report['phase'],'candidate_presence')
            self.assertEqual(report['status'],'error')

if __name__ == '__main__': unittest.main()

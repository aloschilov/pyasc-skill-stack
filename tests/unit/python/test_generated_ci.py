"""Fail-closed attribution and continuity tests; no model or hardware calls."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('single_operator', ROOT/'integrations/cannbench/ci/single_operator.py')
ci = importlib.util.module_from_spec(spec); spec.loader.exec_module(ci)


class GeneratedCI(unittest.TestCase):
    def test_native_skill_trace_requires_each_actual_loaded_directory(self):
        root = Path('/recorded/skills')
        events = [{'sessionID': 'session-1', 'type': 'tool_use', 'part': {'tool': 'skill', 'state': {
            'status': 'completed', 'input': {'name': s}, 'metadata': {'dir': str(root/s)}}}} for s in ci.SKILLS]
        encode = lambda e: '\n'.join(map(json.dumps,e))
        self.assertEqual(ci.trace(encode(events),root)['session_id'],'session-1')
        with self.assertRaises(ValueError): ci.trace(encode(events[:-1]), root)
        events[-1]['part']['state']['metadata']['dir']='/different/skill'
        with self.assertRaises(ValueError): ci.trace(encode(events),root)
        with self.assertRaises(ValueError): ci.trace('I loaded all the skills',root)

    def test_evidence_paths_and_bytes_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'candidate.py').write_text('original')
            h=ci.hashes(root); ci.check_hashes(root,h)
            (root/'candidate.py').write_text('replacement')
            with self.assertRaises(ValueError): ci.check_hashes(root,h)
            with self.assertRaises(ValueError): ci.check_hashes(root,{'../escape':'x'})

    def test_quota_unknown_and_over_working_budget_refuse_submission(self):
        for c in ({}, {'remaining':10}, {'remaining':0,'used':0}, {'remaining':99,'used':10}, {'remaining':True,'used':0}):
            with self.subTest(c=c), self.assertRaises(ValueError): ci.require_quota({'credits':c})
        ci.require_quota({'credits':{'remaining':9,'used':1}})
        ci.require_quota({'credits':{'remaining':9,'daily_limit':10}})

    def test_submit_cannot_skip_central_checkpoint(self):
        queue=Mock()
        with patch.object(ci,'generated_binding',return_value=({}, {}, {})), patch.object(ci.smoke,'client',return_value=queue), patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'Central prepared checkpoint'):
                ci.submit(Path('/unused'))
        self.assertEqual(queue.mock_calls, [])

    def test_remote_host_cannot_receive_download_credentials(self):
        queue=Mock()
        for url in ('https://attacker.invalid/api/log', '//attacker.invalid/api/log', '/api/../secret'):
            with self.assertRaises(ValueError): ci.download(queue,url)
        self.assertEqual(queue.mock_calls, [])

    def test_archive_resume_rejects_tampered_original_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'evidence'; root.mkdir(); archive=Path(tmp)/'q.tar.gz'
            ci.save(root/'acceptance.json',{'passed':True})
            (root/'candidate.py').write_text('original')
            ci.save(root/'EVIDENCE.json',{'files':ci.hashes(root)})
            with tarfile.open(archive,'w:gz') as z: z.add(root,arcname='evidence')
            ci.verify_archive(archive,root)
            (root/'candidate.py').write_text('tampered')
            with self.assertRaises(ValueError): ci.verify_archive(archive,root)

    def test_nested_evaluator_wheel_cannot_substitute_another_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'package/candidate').mkdir(parents=True)
            helper=root/'source/integrations/cannbench/smoke-kernel/_pyasc_runtime.py'; helper.parent.mkdir(parents=True)
            helper.write_bytes(b'helper')
            ci.save(root/'model-config.json',{})
            ci.save(root/'catalog.json',{});ci.save(root/'task.json',{});ci.save(root/'package/task.json',{})
            ci.save(root/'generation.json',{'files':{},'model':ci.MODEL,'provider_model_revision':'unknown',
                'native_trace':{'loaded_skills':list(ci.SKILLS)},'candidate_sha256':'candidate-digest',
                'model_configuration_sha256':ci.smoke.sha(root/'model-config.json')})
            for n,b in [('swi_glu.py',b'original'),('_pyasc_runtime.py',b'helper')]: (root/'package/candidate'/n).write_bytes(b)
            wheel=io.BytesIO()
            with zipfile.ZipFile(wheel,'w') as z:
                z.writestr('cann_bench/swi_glu.py',b'substituted');z.writestr('cann_bench/_pyasc_runtime.py',b'helper')
            with zipfile.ZipFile(root/'package/submission.zip','w') as z:
                z.writestr('vendor/test.whl',wheel.getvalue())
                z.writestr('cann_bench/swi_glu.py',b'original');z.writestr('cann_bench/_pyasc_runtime.py',b'helper')
            prepared={'files':{'candidate/swi_glu.py':'candidate-digest','candidate/_pyasc_runtime.py':ci.smoke.sha(helper)},
                'runtime_bundle_sha256':ci.PIN['runtime_bundle_sha256'],'catalog':{},'wheel_sha256':hashlib.sha256(wheel.getvalue()).hexdigest()}
            with patch.object(ci,'inputs',return_value={}),patch.object(ci.smoke,'verify',return_value=prepared):
                with self.assertRaisesRegex(ValueError,'Source/wheel candidate mismatch'): ci.generated_binding(root)

if __name__=='__main__': unittest.main()

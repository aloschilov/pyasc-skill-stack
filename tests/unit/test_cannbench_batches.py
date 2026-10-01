from contextlib import nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integrations/cannbench/ci'))
module = importlib.import_module('batch_campaign')
submission = importlib.import_module('submission')
campaign = importlib.import_module('run')


class Client:
    def __init__(self):
        self.balance = 1
        self.posts = []
        self.lost = False
        self.visible = True
        self.status = 'succeeded'
    def get_credits(self):
        return {'credits': {'remaining': self.balance}}
    def list_jobs(self, **kwargs):
        return {'jobs': [{'id': 'job_%d' % i, 'job_tag': tag, 'status': self.status}
                         for i, (_, tag) in enumerate(self.posts)] if self.visible else []}
    def get_job(self, job_id):
        return {'job': {'id': job_id, 'status': self.status}}
    def _submit_streaming(self, archive, names, tag):
        with zipfile.ZipFile(archive) as source:
            self.last_metadata = json.loads(source.read('BATCH.json'))
        self.posts.append((names, tag))
        self.balance -= len(names)
        if self.lost:
            raise TimeoutError('lost response')
        return {'job_id': 'job_%d' % (len(self.posts) - 1)}
    @property
    def _client(self):
        return self


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = {'catalog_sha256': 'catalog', 'benchmark_slug': 'official-tasks', 'benchmark_version': '1.1.2',
                        'operators': [{'function_name': name, 'operator': name.title(), 'case_count': 1, 'case_ids': [name + '_1']} for name in ['exp', 'gelu', 'mish']]}
        self.bundle = self.root / 'bundle'; self.bundle.mkdir()
        dist = self.bundle / 'dist'; dist.mkdir()
        (dist / 'test.whl').write_bytes(b'qualified wheel')
        digest = hashlib.sha256(b'qualified wheel').hexdigest()
        self.manifest = {'catalog_sha256': 'catalog', 'benchmark_version': '1.1.2', 'wheel_sha256': digest,
                         'operators': {name: {} for name in ['exp', 'gelu', 'mish']}}
        (self.bundle / 'BUNDLE.json').write_text(json.dumps(self.manifest))
        (self.bundle / 'qualification.json').write_text(json.dumps({'catalog_sha256': 'catalog', 'wheel_sha256': digest, 'status': 'passed', 'operators': [{'operator': n, 'status': 'passed', 'compile_passed': 1, 'dispatch_passed': 1, 'case_results': [{'case_id': n + '_1', 'dispatch': 'passed', 'compile': 'passed'}]} for n in ['exp', 'gelu', 'mish']]}))
        self.client = Client()
        for p in [patch.object(submission, 'STATE_ROOT', self.root / 'state'),
                  patch.object(submission, 'shared_locks', side_effect=lambda _: nullcontext()),
                  patch.object(submission, 'site_queue', return_value=self.client)]:
            p.start(); self.addCleanup(p.stop)
        self.path = module.initialize(self.bundle, self.catalog)
    def advance(self, allow=True):
        return module.advance(self.path, self.catalog, ROOT, allow_submit=allow)
    def test_quota_refill_sends_only_new_function_names_and_same_wheel(self):
        _, state = self.advance()
        self.assertEqual(self.client.posts[0][0], ['exp'])
        self.assertEqual(state['remaining_operators'], ['gelu', 'mish'])
        self.advance()
        self.assertEqual(len(self.client.posts), 1)
        self.client.balance = 2
        payloads, state = self.advance()
        self.assertEqual(self.client.posts[1][0], ['gelu', 'mish'])
        self.assertEqual(len(payloads), 2)
        self.assertEqual(state['remaining_operators'], [])
        self.assertEqual(self.client.last_metadata['wheel_sha256'], self.manifest['wheel_sha256'])
        self.assertNotEqual(state['batches'][0]['archive_sha256'], state['batches'][1]['archive_sha256'])
    def test_get_only_never_spends_credits(self):
        self.advance(False)
        self.assertEqual(self.client.posts, [])
    def test_ambiguous_intent_blocks_future_post_even_after_refill(self):
        self.client.lost = True; self.client.visible = False
        with self.assertRaisesRegex(RuntimeError, 'POST will not be repeated'):
            self.advance()
        self.client.balance = 10
        with self.assertRaisesRegex(RuntimeError, 'POST will not be repeated'):
            self.advance()
        self.assertEqual(len(self.client.posts), 1)
    def test_lost_response_reconciles_get_and_does_not_repeat_operator(self):
        self.client.lost = True
        self.advance()
        self.client.lost = False; self.client.balance = 1
        self.advance()
        self.assertEqual([x[0] for x in self.client.posts], [['exp'], ['gelu']])
    def test_active_batch_prevents_next_batch(self):
        self.client.status = 'running'
        self.advance()
        self.client.balance = 10
        self.advance()
        self.assertEqual(len(self.client.posts), 1)
    def test_changed_base_or_batch_rejected_before_post(self):
        self.advance()
        state = json.loads(self.path.read_text())
        Path(state['batches'][0]['archive_path']).write_bytes(b'tampered')
        self.client.balance = 10
        with self.assertRaisesRegex(ValueError, 'Batch archive changed'):
            self.advance()
        self.assertEqual(len(self.client.posts), 1)
    def test_partition_tampering_rejected_even_with_unchanged_archive_hash(self):
        self.advance()
        state = json.loads(self.path.read_text())
        state['batches'][0]['selected_operators'] = ['mish']
        state['batches'][0]['binding']['selected_operators'] = ['mish']
        submission.save(self.path, state)
        self.client.balance = 10
        with self.assertRaisesRegex(ValueError, 'Batch bytes differ'):
            self.advance()
        self.assertEqual(len(self.client.posts), 1)

    def test_multiple_job_binding_requires_exact_same_wheel_and_current_sources(self):
        manifest = dict(self.manifest, generation_sources={'source.py': hashlib.sha256(b'source').hexdigest()})
        (self.root / 'source.py').write_bytes(b'source')
        (self.bundle / 'BUNDLE.json').write_text(json.dumps(manifest))
        # Create a new frozen campaign from the revised complete fixture.
        (self.root / 'campaign-bundle.zip').unlink()
        self.path = module.initialize(self.bundle, self.catalog)
        self.advance()
        self.client.balance = 2
        _, state = self.advance()
        ids = [b['job_id'] for b in state['batches']]
        self.assertTrue(campaign.verify_binding(self.catalog, state, ids, self.root))
        self.assertFalse(campaign.verify_binding(self.catalog, state, ids[:1], self.root))
        (self.root / 'source.py').write_bytes(b'changed')
        self.assertFalse(campaign.verify_binding(self.catalog, state, ids, self.root))

    def test_full_scope_and_qualified_wheel_required(self):
        changed = dict(self.catalog, operators=self.catalog['operators'][:1])
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            module.advance(self.path, changed, ROOT, allow_submit=True)
        self.assertEqual(self.client.posts, [])

if __name__ == '__main__':
    unittest.main()

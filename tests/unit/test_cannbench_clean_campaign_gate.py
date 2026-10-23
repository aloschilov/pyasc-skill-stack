import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integrations/cannbench/ci'))
gate = importlib.import_module('numerical_gate')


class CleanCampaignGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy_path = self.root / 'integrations/cannbench/ci/clean-campaign-policy.json'
        self.policy_path.parent.mkdir(parents=True)
        self.manifest = dict(mode='clean-frozen', campaign_id='new-campaign',
                             model='gateway/glm-5.3', operators=53, cases=1060,
                             generation_origin='clean-state', reused_candidates=[],
                             qualification_mutations=[])
        for key in ('skill_files_sha256', 'model_profile_sha256', 'opencode_protocol_sha256',
                    'context_contract_sha256', 'runtime_identity_sha256',
                    'evaluator_identity_sha256', 'catalog_sha256',
                    'native_generation_evidence_sha256'):
            self.manifest[key] = 'test-evidence-binding'

    def archive(self, *, approve=False, bundle=b'bundle', embedded_approval=False):
        artifacts = {'CLEAN_CAMPAIGN.json': json.dumps(self.manifest).encode(),
                     'BUNDLE.json': bundle, 'numerical-reports.json': b'reports'}
        policy = dict(schema_version=1, mode='clean-frozen', campaign_id='new-campaign',
                      artifact_sha256={k: hashlib.sha256(v).hexdigest() for k, v in artifacts.items()})
        if approve:
            self.policy_path.write_text(json.dumps(policy))
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            for key, value in artifacts.items():
                archive.writestr(key, value)
            if embedded_approval:
                archive.writestr('clean-campaign-policy.json', json.dumps(policy))
        return zipfile.ZipFile(data)

    def test_archive_cannot_approve_itself(self):
        with self.archive(embedded_approval=True) as source:
            with self.assertRaisesRegex(ValueError, 'independently reviewed'):
                gate.require_clean_campaign(source, self.root)

    def test_external_audit_binds_exact_artifacts(self):
        with self.archive(approve=True) as source:
            gate.require_clean_campaign(source, self.root)
        with self.archive(bundle=b'other-candidates') as source:
            with self.assertRaisesRegex(ValueError, 'bind artifact'):
                gate.require_clean_campaign(source, self.root)

    def test_review_cannot_relax_clean_invariants(self):
        for field, value in [('reused_candidates', ['old.py']),
                             ('qualification_mutations', ['kernel patch']),
                             ('generation_origin', 'development'), ('cases', 1059),
                             ('model', 'other')]:
            original = self.manifest[field]
            with self.subTest(field=field):
                self.manifest[field] = value
                with self.archive(approve=True) as source:
                    with self.assertRaisesRegex(ValueError, 'invariants'):
                        gate.require_clean_campaign(source, self.root)
                self.manifest[field] = original

    def test_frozen_evidence_required(self):
        self.manifest.pop('context_contract_sha256')
        with self.archive(approve=True) as source:
            with self.assertRaisesRegex(ValueError, 'context_contract'):
                gate.require_clean_campaign(source, self.root)


if __name__ == '__main__':
    unittest.main()

"""Evidence collection works in runtime images that do not ship Git."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from run_artifacts import RunArtifacts


class RevisionWithoutGit(unittest.TestCase):
    def test_explicit_revision_does_not_invoke_git(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'PYASC_SOURCE_REVISION': 'a' * 40}), \
                patch('run_artifacts.subprocess.run') as command:
            run = RunArtifacts(Path(tmp) / 'run', 'test')
            command.assert_not_called()
            run.finish('passed')
            self.assertEqual(json.loads((run.path / 'manifest.json').read_text())['revision'], 'a' * 40)
            self.assertTrue(run.data['inputs_unchanged'])

    def test_missing_git_retains_hashes_and_unknown_revision(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'PYASC_SOURCE_REVISION': ''}), \
                patch('run_artifacts.subprocess.run', side_effect=FileNotFoundError):
            run = RunArtifacts(Path(tmp) / 'run', 'test')
            run.finish('passed')
            self.assertIsNone(run.data['revision'])
            self.assertTrue(run.data['input_sha256'])


if __name__ == '__main__':
    unittest.main()

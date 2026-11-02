"""Matrix policy tests without contacting hardware or recording qualification."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import hardware_acceptance as hardware
from verification_contract import matches_outcome


class HardwareAcceptance(unittest.TestCase):
    def test_shared_policy_rejects_wrong_failure_and_auto_fallback(self):
        expected = dict(mode='auto', expected=1, level=None, calls=1,
                        reason='deliberate post-launch')
        report = {'status': 'FAIL', 'verification_level': None, 'launcher_calls': 1,
                  'detail': 'deliberate post-launch', 'mode': 'auto (execute)'}
        self.assertTrue(matches_outcome(report, 1, **expected))
        for field, value in [('mode', 'auto (jit)'), ('detail', 'compiler error'),
                             ('launcher_calls', 0), ('status', 'SKIP')]:
            with self.subTest(field=field):
                self.assertFalse(matches_outcome(dict(report, **{field: value}), 1, **expected))
        self.assertFalse(matches_outcome(report, 2, **expected))

    def test_blocked_stops_matrix_and_does_not_erase_prior_failure(self):
        for prior_failure in (False, True):
            with self.subTest(prior_failure=prior_failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                kernel = root / 'kernel.py'; kernel.write_text('pass')
                run = Mock(path=root / 'results', data={})
                blocked = (2, {'status': 'SKIP', 'detail': 'device unavailable'})
                run.command.side_effect = ([(1, {'status': 'FAIL', 'detail': 'unexpected'})]
                                           if prior_failure else []) + [blocked]
                cases = [(label, kernel, 'execute', 0, 'numerically_verified', 1, None)
                         for label in ('first', 'second', 'must-not-run')]
                with patch.object(sys, 'argv', ['hardware_acceptance.py', '--results-dir', str(run.path)]), \
                     patch.object(hardware, 'RunArtifacts', return_value=run), \
                     patch.object(hardware, 'core_cases', return_value=cases):
                    self.assertEqual(hardware.main(), 1 if prior_failure else 2)
                self.assertEqual(run.command.call_count, 2 if prior_failure else 1)
                self.assertEqual(run.data['failed_cases'], ['first'] if prior_failure else [])
                run.finish.assert_called_once_with('failed' if prior_failure else 'blocked')

    def test_changed_device_cannot_qualify_a_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kernel = root / 'kernel.py'; kernel.write_text('pass')
            run = Mock(path=root / 'results', data={})
            identity = {'status': 'ready', 'hostname': 'test-only', 'soc_version': 'soc-a',
                        'cann': {'version': 'test'}, 'pyasc_version': 'test-only'}
            report = {'status': 'PASS', 'verification_level': 'numerically_verified',
                      'launcher_calls': 1, 'device_identity': identity}
            run.command.side_effect = [(0, report), (0, dict(report, device_identity=
                                               dict(identity, soc_version='soc-b')))]
            cases = [(label, kernel, 'execute', 0, 'numerically_verified', 1, None)
                     for label in ('first', 'changed')]
            with patch.object(sys, 'argv', ['hardware_acceptance.py', '--results-dir', str(run.path)]), \
                 patch.object(hardware, 'RunArtifacts', return_value=run), \
                 patch.object(hardware, 'core_cases', return_value=cases):
                self.assertEqual(hardware.main(), 1)
            self.assertEqual(run.data['failed_cases'], ['changed'])
            run.finish.assert_called_once_with('failed')

    def test_optional_launcher_count_and_required_count(self):
        for expected_calls, observed_calls, expected_exit in (
                (None, 3, 0), (1, 1, 0), (1, 0, 1)):
            with self.subTest(expected=expected_calls, observed=observed_calls), \
                 tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                kernel = root / 'kernel.py'
                kernel.write_text('pass')
                identity = {'status': 'ready', 'hostname': 'test-only',
                            'soc_version': 'test-only', 'cann': {}, 'pyasc_version': 'test-only'}
                run = Mock(path=root / 'results', data={})
                run.command.return_value = (0, {
                    'status': 'PASS', 'verification_level': 'executed',
                    'launcher_calls': observed_calls, 'device_identity': identity,
                })
                case = ('count-policy', kernel, 'execute', 0, 'executed', expected_calls, None)
                with patch.object(sys, 'argv', ['hardware_acceptance.py', '--results-dir', str(run.path)]), \
                     patch.object(hardware, 'RunArtifacts', return_value=run), \
                     patch.object(hardware, 'core_cases', return_value=[case]):
                    self.assertEqual(hardware.main(), expected_exit)
                run.finish.assert_called_once_with('failed' if expected_exit else 'passed')


if __name__ == '__main__':
    unittest.main()

import fnmatch
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import opencode_profile


def effective_action(rules, path):
    # OpenCode permission contract: the last matching pattern wins.
    return next((action for pattern, action in reversed(list(rules.items()))
                 if fnmatch.fnmatchcase(path, pattern)), 'ask')


class ProfilePermissions(unittest.TestCase):
    def grant(self, rules):
        with patch.object(opencode_profile, 'REPO_ROOT', Path('/repo')):
            config = opencode_profile.grant_repo_access(
                {'permission': {'external_directory': rules}}, ('/fixtures',))
        return config['permission']['external_directory']

    def test_catch_all_does_not_override_authorized_roots(self):
        rules = self.grant({'*': 'deny'})
        for path in ('/repo', '/repo/skills/example/SKILL.md', '/fixtures/kernel.py'):
            self.assertEqual(effective_action(rules, path), 'allow', path)
        self.assertEqual(effective_action(rules, '/unrelated/key'), 'deny')

    def test_specific_denials_still_override_generated_grants(self):
        rules = self.grant({'/repo/private/*': 'deny', '*': 'deny'})
        self.assertEqual(effective_action(rules, '/repo/skills/example/SKILL.md'), 'allow')
        self.assertEqual(effective_action(rules, '/repo/private/key'), 'deny')

    def test_explicit_root_denial_is_preserved(self):
        rules = self.grant({'*': 'deny', '/repo/*': 'deny'})
        self.assertEqual(effective_action(rules, '/repo/skills/example/SKILL.md'), 'deny')
        self.assertEqual(effective_action(rules, '/fixtures/kernel.py'), 'allow')

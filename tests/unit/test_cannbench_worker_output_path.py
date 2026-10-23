import sys
import unittest
from pathlib import Path

WORKERS = Path(__file__).resolve().parents[2] / "integrations" / "cannbench" / "workers"
sys.path.insert(0, str(WORKERS))
import driver


class CampaignOutputPathTests(unittest.TestCase):
    def test_default_uses_ignored_scratch_and_timestamp(self):
        self.assertEqual(
            driver.campaign_output_root(timestamp="fixed"),
            driver.WORKERS_DIR / ".scratch" / "cannbench" / "runs" / "fixed",
        )

    def test_explicit_output_path_takes_precedence(self):
        requested = Path("custom-output")
        self.assertEqual(driver.campaign_output_root(requested, "ignored"), requested.resolve())


if __name__ == "__main__":
    unittest.main()

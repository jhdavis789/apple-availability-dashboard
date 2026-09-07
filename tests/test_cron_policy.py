from pathlib import Path
import unittest


CRON = (Path(__file__).resolve().parents[1] / "cron_update.sh").read_text()


class CronPolicyTests(unittest.TestCase):
    def test_ebay_refresh_is_age_based_not_clock_slot_based(self):
        self.assertIn('stat -f%m "$EBAY_DB"', CRON)
        self.assertIn("EBAY_MAX_AGE=7200", CRON)
        self.assertNotIn("HOUR=$(date '+%H')", CRON)
        self.assertNotIn("MINUTE=$(date '+%M')", CRON)


if __name__ == "__main__":
    unittest.main()

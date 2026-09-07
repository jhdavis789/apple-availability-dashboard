import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "availability_matrix_csv_rest.py"
SPEC = importlib.util.spec_from_file_location("collector", MODULE_PATH)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class PickupStateTests(unittest.TestCase):
    def test_active_products_exclude_retired_mac_skus(self):
        retired = {"MU9D3LL/A", "MU9E3LL/A", "MCYT4LL/A", "MCX44LL/A",
                   "MU963LL/A", "MU973LL/A"}
        self.assertTrue(retired.isdisjoint(collector.TRACKED_PRODUCTS.values()))

    def test_current_desktop_skus_are_tracked(self):
        expected = {"MHQK4LL/A", "MHQN4LL/A", "MHL64LL/A", "MHL74LL/A"}
        self.assertTrue(expected.issubset(collector.TRACKED_PRODUCTS.values()))

    def test_macbook_air_labels_are_m5(self):
        names = set(collector.TRACKED_PRODUCTS)
        self.assertIn('MacBook Air 13" M5 ($1,299)', names)
        self.assertIn('MacBook Air 15" M5 ($1,499)', names)
        self.assertFalse(any("MacBook Air" in name and " M4 " in name for name in names))

    def test_all_ineligible_is_not_published_as_zero(self):
        with self.assertRaisesRegex(RuntimeError, "false 0% availability"):
            collector.assert_no_all_ineligible({
                "retired": {"R001": "ineligible", "R002": "ineligible"},
            })

    def test_real_unavailable_is_allowed(self):
        collector.assert_no_all_ineligible({
            "preorder": {"R001": "unavailable", "R002": "ineligible"},
        })


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_data.py"
SPEC = importlib.util.spec_from_file_location("build_data", MODULE_PATH)
build_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_data)


def product(name, value):
    return {"model": name, "values": {"NYC": value}}


class DiscontinuedTailTests(unittest.TestCase):
    def test_latest_snapshot_defines_active_products_immediately(self):
        snapshots = [
            {"timestamp": "2026-09-07T08:00:00", "products": [
                product("Retired Mac", 50), product("Current Mac", 50),
            ]},
            {"timestamp": "2026-09-07T08:30:00", "products": [
                product("Retired Mac", 0), product("Current Mac", 50),
            ]},
            {"timestamp": "2026-09-07T09:00:00", "products": [
                product("Current Mac", 0),
            ]},
        ]

        build_data._trim_discontinued_tails(snapshots)

        self.assertEqual(
            [[p["model"] for p in snap["products"]] for snap in snapshots],
            [["Retired Mac", "Current Mac"], ["Current Mac"], ["Current Mac"]],
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

import yaml


class DatasetConfigTests(unittest.TestCase):
    def test_example_uses_consecutive_zero_based_class_ids(self):
        config_path = Path(__file__).parents[1] / "data.yaml.example"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        names = config["names"]

        self.assertEqual(list(names), [0, 1])
        self.assertEqual(list(names.values()), ["cup", "espresso"])


if __name__ == "__main__":
    unittest.main()

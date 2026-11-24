import pathlib
import unittest

import yaml

FLOW_DIR = pathlib.Path(__file__).resolve().parents[1] / "configs" / "flow"


class FlowConfigSyncTests(unittest.TestCase):
    def test_short_chain_config_differs_from_maf_only_in_thin(self):
        prod = yaml.safe_load((FLOW_DIR / "maf.yaml").read_text())
        short = yaml.safe_load((FLOW_DIR / "maf_chain1000.yaml").read_text())
        self.assertEqual((prod["mcmc"]["thin"], short["mcmc"]["thin"]), (10, 1))
        short["mcmc"]["thin"] = prod["mcmc"]["thin"]
        self.assertEqual(short, prod, "maf_chain1000.yaml has drifted from maf.yaml; copy the change across")


if __name__ == "__main__":
    unittest.main()

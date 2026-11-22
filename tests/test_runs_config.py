import glob
import os
import tempfile
import unittest

import yaml

from msfm.utils.input_output import read_yaml
from msi.utils.runs import dataset_runs_dir, load_runs_config

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class RunsConfigTests(unittest.TestCase):
    def _write(self, conf):
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(conf, handle)
        handle.close()
        self.addCleanup(os.remove, handle.name)
        return handle.name

    def test_existing_configs_load_unchanged(self):
        paths = sorted(glob.glob(os.path.join(REPO_DIR, "configs/runs/desy3/*/*/*.yaml")))
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(path=os.path.relpath(path, REPO_DIR)):
                self.assertEqual(load_runs_config(path), read_yaml(path))

    def test_relative_pred_dir_resolves_against_the_dataset(self):
        runs_dir = dataset_runs_dir("desy6/v1/default")
        self.assertEqual(runs_dir, "/ritom/scratch/cscs/athomsen/deep_lss/runs/desy6/v1/default")
        path = self._write(
            {
                "dataset": "desy6/v1/default",
                "runs": {
                    "maps_gcnn": {"lensing": {"pred_dir": "maps_gcnn/lensing/v1"}},
                    "cls": {"lensing": {"pred_dir": "/elsewhere/cls/lensing/v1"}},
                    "maps_transformer": None,
                },
            }
        )
        runs = load_runs_config(path)["runs"]
        self.assertEqual(runs["maps_gcnn"]["lensing"]["pred_dir"], os.path.join(runs_dir, "maps_gcnn/lensing/v1"))
        self.assertEqual(runs["cls"]["lensing"]["pred_dir"], "/elsewhere/cls/lensing/v1")
        self.assertIsNone(runs["maps_transformer"])

    def test_relative_pred_dir_needs_a_dataset(self):
        path = self._write({"runs": {"maps_gcnn": {"lensing": {"pred_dir": "maps_gcnn/lensing/v1"}}}})
        with self.assertRaisesRegex(ValueError, "needs a dataset key"):
            load_runs_config(path)

    def test_unknown_dataset_fails(self):
        path = self._write({"dataset": "desy6/v9/default", "runs": {}})
        with self.assertRaisesRegex(FileNotFoundError, "has no msfm config"):
            load_runs_config(path)


if __name__ == "__main__":
    unittest.main()

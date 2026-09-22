import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from msi.apps.score_coverage_round import holm_adjust, score, tarp_ranks


class CoverageRoundTests(unittest.TestCase):
    def test_tarp_is_invariant_to_units_and_batch_size(self):
        rng = np.random.default_rng(8)
        samples, truth = rng.uniform(size=(257, 12, 2)), rng.uniform(size=(12, 2))
        rank = tarp_ranks(samples, truth, [[0, 1], [0, 1]], 17)
        scale, shift = np.array([.2, 70]), np.array([.1, 64])
        converted = tarp_ranks(samples * scale + shift, truth * scale + shift,
                               np.column_stack([shift, shift + scale]), 17)
        np.testing.assert_array_equal(rank, converted)
        ref = np.random.default_rng(17).uniform(size=truth.shape)
        direct = (np.sum((samples - ref) ** 2, axis=-1) < np.sum((truth - ref) ** 2, axis=-1)).mean(axis=0)
        np.testing.assert_array_equal(rank, direct)

    def test_holm_adjustment(self):
        np.testing.assert_allclose(holm_adjust([.03, .001, .02]), [.04, .003, .04])

    def test_scorer_pairs_mocks_and_handles_extended_params(self):
        self._check_scorer(10, None)

    def test_scorer_handles_single_probes_and_checkpoint_names(self):
        for dimension, checkpoint in [(6, 126000), (7, 126500)]:
            with self.subTest(dimension=dimension):
                self._check_scorer(dimension, checkpoint)

    def _check_scorer(self, dimension, checkpoint):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params = ["Om", "s8", "w0", "Aia", "n_Aia", "bta", "bg1", "bg2", "bg3", "bg4"]
            params = params[:dimension]
            step = checkpoint if checkpoint is not None else 229900
            ids = np.column_stack([np.arange(12), np.full(12, 78), np.zeros(12)]).astype(int)
            np.save(root / "mock_ids.npy", ids)
            manifest = {"base_params": params, "prior_intervals": {p: [0, 1] for p in params},
                        "arms": {"all_long": {"extend_params": []},
                                 "extended_long": {"extend_params": ["ns", "Ob", "H0"]}}}
            if checkpoint is not None:
                manifest["checkpoint"] = checkpoint
            (root / "manifest.json").write_text(json.dumps(manifest))
            for arm, spec in manifest["arms"].items():
                labels = params + spec["extend_params"]
                path = root / f"{arm}_ensemble_flow_{step}"
                path.mkdir()
                rng = np.random.default_rng(17)
                with h5py.File(path / "mcmc_samples.h5", "w") as f:
                    f.attrs["params"] = labels
                    f["real_idx"] = ids
                    f["theta_sample"] = rng.uniform(size=(64, 12, len(labels)))
                    f["theta_true"] = rng.uniform(size=(12, len(labels)))
                    f["log_prob_sample"] = rng.normal(size=(64, 12))
                    f["log_prob_true"] = rng.normal(size=12)
                (root / f"{arm}.complete").touch()
            score(root, root / "scores.json")
            report = json.loads((root / "scores.json").read_text())
            self.assertEqual(report["arms"]["extended_long"]["hpd_dimension"], dimension + 3)
            self.assertNotIn("hpd_cov68", report["paired_comparisons"]["extended_long minus all_long"])
            with h5py.File(root / f"all_long_ensemble_flow_{step}/mcmc_samples.h5", "a") as f:
                f["real_idx"][0] = [999, 78, 0]
            with self.assertRaises(ValueError):
                score(root, root / "scores.json")


if __name__ == "__main__":
    unittest.main()

"""Small CPU integration checks; no real-data training or cluster submission."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("enflows"),
                     "Requires the existing PyTorch environment")
class CoverageIntegrationTests(unittest.TestCase):
    def test_paired_likelihood_and_posterior_selection(self):
        import torch
        from torch.utils.data import TensorDataset, Subset
        from msi.utils.coverage import sample_coverage_posteriors, run_likelihood_coverage

        with tempfile.TemporaryDirectory() as tmp:
            ids = np.array([[i, s, 0] for i in range(4) for s in range(4)])
            x = np.arange(32, dtype=np.float32).reshape(16, 2)
            requested = ids[[15, 3]]
            mock_file = Path(tmp) / "mock_ids.npy"
            np.save(mock_file, requested)

            class FakeFlow:
                params = ["Om", "s8"]
                model_dir = tmp

                def _prepare_data(self, x, theta, group_ids, **kwargs):
                    data = TensorDataset(torch.tensor(x), torch.tensor(theta))
                    self.vali_dset = Subset(data, np.flatnonzero(group_ids == 3))

                def sample_posterior_batched(self, x_obs, **kwargs):
                    self.posterior_x = x_obs.numpy()
                    return np.tile(self.posterior_x[:, None, :], (1, 16, 1)), None

                def log_likelihood(self, x_obs, theta, **kwargs):
                    return np.zeros(len(theta))

                def sample_likelihood(self, theta, **kwargs):
                    self.likelihood_theta = theta
                    return np.tile(theta[:, None, :], (1, 10, 1))

            conf = {"diagnostics": {"n_obs": 2, "mock_ids_file": str(mock_file), "subsample_seed": 17},
                    "mcmc": {"n_steps": 4, "n_walkers": 4, "n_burnin_steps": 1}}
            flow = FakeFlow()
            out = sample_coverage_posteriors(flow, x, x, ids[:, 1], conf,
                                             i_sobol=ids[:, 0], i_noise=ids[:, 2])
            with patch("msi.utils.coverage.run_likelihood_coverage_tests") as diagnostic:
                run_likelihood_coverage(flow, x, x, ids[:, 1], conf, i_sobol=ids[:, 0], i_noise=ids[:, 2])
                np.testing.assert_array_equal(diagnostic.call_args.args[0], x[[15, 3]])
            np.testing.assert_array_equal(out["real_idx"], requested)
            np.testing.assert_array_equal(flow.posterior_x, flow.likelihood_theta)
            with h5py.File(Path(tmp) / "mcmc_samples.h5") as f:
                self.assertEqual(list(f.attrs["params"]), ["Om", "s8"])
            # A training realization must never be accepted as a requested coverage mock.
            np.save(mock_file, ids[[0, 3]])
            with self.assertRaisesRegex(ValueError, "not held out"):
                sample_coverage_posteriors(flow, x, x, ids[:, 1], conf,
                                           i_sobol=ids[:, 0], i_noise=ids[:, 2])

    def test_weighted_sequential_and_fused_cpu_training(self):
        import msfm
        import yaml
        from msi.utils.flow import build_flow_architecture
        from msi.flow_conductor.likelihood_flow import LikelihoodFlow, LikelihoodFlowEnsemble

        conf_path = Path(msfm.__file__).resolve().parents[1] / "configs/v18/default.yaml"
        with open(conf_path) as f:
            msfm_conf = yaml.safe_load(f)
        rng = np.random.default_rng(8)
        x, theta = rng.normal(size=(64, 2)), rng.normal(size=(64, 2))
        weights = np.where(np.arange(64) % 3, 1.0, .021)
        groups = np.repeat(np.arange(4), 16)
        config = {"context_embedding": {"dim": 4, "hidden_dim": 8, "n_blocks": 1},
                  "transform": {"type": "maf", "n_layers": 1, "hidden_dim": 8,
                                "maf": {"num_blocks": 1, "dropout_probability": .1}}}
        with tempfile.TemporaryDirectory() as tmp:
            embedding, transform = build_flow_architecture(2, 2, config)
            single = LikelihoodFlow(["Om", "s8"], conf=msfm_conf, out_dir=tmp, feature_dim=2,
                                    embedding_net=embedding, transform=transform, device="cpu", load_existing=False)
            history = single.fit(x, theta, n_epochs=2, batch_size=16, vali_split=.25,
                                 scheduler_type="cosine", group_ids=groups, row_weights=weights, save_model=False)
            self.assertTrue(np.isfinite(history["vali_loss"]).all())
            ensemble = LikelihoodFlowEnsemble(
                ["Om", "s8"], conf=msfm_conf, n_flows=2, out_dir=tmp, feature_dim=2, device="cpu", load_existing=False,
                embedding_net_fn=lambda: build_flow_architecture(2, 2, config)[0],
                transform_fn=lambda: build_flow_architecture(2, 2, config)[1],
            )
            # Call the fused backend directly so an accidental fallback cannot hide a failure.
            for flow in ensemble.flows:
                flow._embedding_net.set_stats(theta)
            ensemble._fit_fused(
                x, theta, n_epochs=2, batch_size=16, vali_split=.25, learning_rate=.001,
                weight_decay=0, clip_by_global_norm=1, scheduler_type="cosine", scheduler_kwargs=None,
                save_model=False, seed=7, group_ids=groups, row_weights=weights,
                run_c2st=False, c2st_hidden_dim=8, c2st_n_epochs=1,
            )
            self.assertTrue(np.isfinite(ensemble.validation_losses).all())


if __name__ == "__main__":
    unittest.main()

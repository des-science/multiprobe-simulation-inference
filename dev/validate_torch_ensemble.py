# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Standalone fidelity checks for the GPU-batched coverage-test machinery. Run in the torch environment
(no trained flow or GPU required; CPU is fine and fast):

    source ~/dlss/torch_env/bin/activate
    python dev/validate_torch_ensemble.py --conf <path/to/a/config.yaml>

It verifies two things:

1. prior: prior.in_grid_prior_torch reproduces the numpy prior.in_grid_prior on random parameter draws,
   including points near the Om-s8 hull and the w0 threshold.
2. sampler: msi.utils.torch_ensemble.run_ensemble_torch reproduces emcee's default StretchMove on an
   analytic correlated-Gaussian target (so any disagreement is in the sampler, not the flow). We compare
   recovered means/covariances to the truth and to emcee, within MCMC sampling noise.
"""

import argparse
import numpy as np
import torch

import emcee

from msfm.utils import prior, parameters
from msi.utils import torch_ensemble


def check_prior(conf):
    params = parameters.get_parameters(None, conf)
    intervals = parameters.get_prior_intervals(params, conf)

    rng = np.random.default_rng(0)
    # draw a bit outside the box on purpose so we exercise inside/outside/boundary
    lo, hi = intervals[:, 0], intervals[:, 1]
    span = hi - lo
    cosmos = rng.uniform(lo - 0.1 * span, hi + 0.1 * span, size=(200_000, len(params)))

    ref = prior.in_grid_prior(cosmos, conf=conf, params=params)

    prior_data = prior.get_torch_prior_data(params, conf=conf, device="cpu", floatx=torch.float64)
    got = prior.in_grid_prior_torch(torch.as_tensor(cosmos, dtype=torch.float64), prior_data).numpy()

    n_disagree = int(np.sum(ref != got))
    frac_in = float(np.mean(ref))
    print(f"[prior] {len(params)} params, {frac_in*100:.1f}% inside; disagreements: {n_disagree} / {len(cosmos)}")
    assert n_disagree == 0, "torch prior disagrees with numpy in_grid_prior"
    print("[prior] OK")


def check_sampler(n_walkers=512, n_steps=2000, n_burnin_steps=2000):
    n_dim = 4
    # two distinct targets to exercise the batched (n_obs) axis
    means = np.array([np.zeros(n_dim), np.linspace(-1.0, 1.0, n_dim)])
    A = np.tril(np.random.default_rng(1).normal(size=(n_dim, n_dim)))
    cov = A @ A.T + np.eye(n_dim)
    inv_cov = np.linalg.inv(cov)

    def neglogp(delta):  # delta: (..., n_dim)
        return 0.5 * np.einsum("...i,ij,...j->...", delta, inv_cov, delta)

    # --- emcee reference, one chain per target ---
    emcee_samples = []
    for mu in means:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, lambda t: -neglogp(t - mu), vectorize=True)
        state = sampler.run_mcmc(mu + 1e-3 * np.random.default_rng(2).normal(size=(n_walkers, n_dim)), n_burnin_steps)
        sampler.reset()
        sampler.run_mcmc(state, n_steps)
        emcee_samples.append(sampler.get_chain(flat=True))

    # --- torch batched sampler, both targets at once ---
    means_t = torch.as_tensor(means, dtype=torch.float64)
    inv_cov_t = torch.as_tensor(inv_cov, dtype=torch.float64)

    def log_prob_fn(theta):  # theta: (n_obs, k, n_dim)
        delta = theta - means_t.unsqueeze(1)
        return -0.5 * torch.einsum("oki,ij,okj->ok", delta, inv_cov_t, delta)

    gen = torch.Generator(device="cpu").manual_seed(3)
    theta_0 = means_t.unsqueeze(1) + 1e-3 * torch.randn((2, n_walkers, n_dim), dtype=torch.float64, generator=gen)
    chain, _ = torch_ensemble.run_ensemble_torch(
        log_prob_fn, theta_0, n_steps=n_steps, n_burnin_steps=n_burnin_steps, generator=gen, progress=False
    )

    for i, mu in enumerate(means):
        m_emcee, m_torch = emcee_samples[i].mean(0), chain[i].mean(0)
        c_emcee, c_torch = np.cov(emcee_samples[i].T), np.cov(chain[i].T)
        print(
            f"[sampler] target {i}: |mean-truth| emcee={np.abs(m_emcee-mu).max():.3f} torch={np.abs(m_torch-mu).max():.3f}"
        )
        print(
            f"[sampler] target {i}: max|cov_torch-cov_true|={np.abs(c_torch-cov).max():.3f} "
            f"max|cov_torch-cov_emcee|={np.abs(c_torch-c_emcee).max():.3f}"
        )
        # both samplers target the same Gaussian; agreement to a few % of the scale is expected
        assert np.abs(m_torch - mu).max() < 0.1, "torch posterior mean off from truth"
        assert np.abs(c_torch - cov).max() < 0.3, "torch posterior covariance off from truth"
        assert np.abs(c_torch - c_emcee).max() < 0.3, "torch and emcee covariances disagree"
    print("[sampler] OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conf", default=None, help="config for the prior check (None = repo default)")
    p.add_argument("--skip_prior", action="store_true")
    args = p.parse_args()

    if not args.skip_prior:
        check_prior(args.conf)
    check_sampler()
    print("All validation checks passed.")

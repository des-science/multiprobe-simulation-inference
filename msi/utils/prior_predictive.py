# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2026
Author: Arne Thomsen

Prior-level visualization of the network summary statistics, wired so it runs automatically in
run_inference.py (no need to execute y3-deep-lss/dev/notebooks/results/summary_space.ipynb or
deep_lss_paper/paper_2/pre-unblinding/0_prior_predictive_checks.ipynb separately).

The prior predictive distribution of the summaries is the marginal distribution the likelihood flow
has to learn the density of; visualizing it as a corner plot of grid_preds colored by S8 gives an
idea of how complex the VMIM latent summary space is. The plot is saved with a 0_ prefix under
flow.model_dir/unblinding_plots, analogous to the likelihood- (1_) and posterior-level (2_) coverage
plots in coverage.py.

summary_typicality is the quantitative counterpart: whether an observed summary is a typical draw from
the grid summaries, before any posterior is evaluated. It computes and does not plot.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve_triangular
from scipy.spatial import cKDTree
from trianglechain import TriangleChain

from msfm.utils import logger
from msi.utils import plotting

LOGGER = logger.get_logger(__file__)


def _save(fig, plot_dir, name):
    plot_file = os.path.join(plot_dir, name)
    fig.savefig(plot_file, bbox_inches="tight", dpi=plotting.PLOT_DPI)
    plt.close(fig)
    LOGGER.info(f"Saved {plot_file}")


def run_prior_predictive(flow, grid_preds, grid_cosmos, params, flow_conf, n_rand=10000):
    """Plot the prior predictive distribution of the network summaries (summary space).

    A TriangleChain corner plot of a random subsample of grid_preds, colored by S8, saved to
    flow.model_dir/unblinding_plots/0_prior_predictive_summary_space.png. Visualizes the marginal
    distribution the likelihood flow must learn.
    """
    plot_dir = os.path.join(flow.model_dir, "unblinding_plots")
    os.makedirs(plot_dir, exist_ok=True)

    grid_preds = np.asarray(grid_preds)
    grid_cosmos = np.asarray(grid_cosmos)

    n_rand = flow_conf.get("diagnostics", {}).get("n_prior_predictive", n_rand)
    n_rand = min(n_rand, grid_preds.shape[0])
    rng = np.random.default_rng(0)
    i_rand = rng.choice(grid_preds.shape[0], size=n_rand, replace=False)

    # color the summary-space scatter by S8 to show how cosmology maps into the latent space; fall
    # back to a plain scatter if Om/s8 are not among the inferred parameters.
    if "Om" in params and "s8" in params:
        S8 = plotting.sigma8_to_S8(grid_cosmos[i_rand, params.index("s8")], grid_cosmos[i_rand, params.index("Om")])
        tri = TriangleChain(
            size=2,
            cmap="viridis",
            colorbar=True,
            colorbar_label=r"$S_8 = \sigma_8 \sqrt{\Omega_m / 0.3}$",
        )
        tri.scatter_prob(
            grid_preds[i_rand],
            prob=S8,
            scatter_kwargs={"s": 10, "marker": "o"},
            normalize_prob2D=False,
        )
    else:
        tri = TriangleChain(size=2)
        tri.scatter(grid_preds[i_rand], scatter_kwargs={"s": 10, "marker": "o"})

    tri.fig.suptitle(f"prior predictive summary space | x_dim={grid_preds.shape[-1]}", fontsize=20)
    _save(tri.fig, plot_dir, "0_prior_predictive_summary_space.png")


# neighbour rank of the distance statistic
K_NEIGHBORS = 10


def summary_typicality(s_obs, preds, cosmo_ids, k=K_NEIGHBORS, chunk_size=10000):
    """Prior predictive check in summary space: is s_obs a typical draw from the simulated summaries?

    The statistic is the distance from a summary to its k-th nearest simulated summary, with all summaries
    whitened by the covariance of the simulated ones. Its null distribution is the same distance for every
    simulated summary in turn, so the p-value is exact if s_obs is exchangeable with them.

    Each simulated summary is scored leave-one-cosmology-out, i.e. against the others with every
    realization of its own cosmology removed, since the observation's cosmology is not among them either.
    Keeping the siblings shrinks the null and fails a typical observation.

    Args:
        s_obs: (n_summaries,) summary of the observation.
        preds: (..., n_summaries) summaries of simulations the compression network was not trained on, drawn
            from the analysis prior, e.g. grid/preds/test of preds_*.h5 restricted to the wide Sobol sequence.
            A sampling density that differs from the prior enters the statistic.
        cosmo_ids: preds.shape[:-1] cosmology id per summary, e.g. i_sobol.
        k (int): neighbour rank of the distance.
        chunk_size (int): null summaries queried at once, bounding the memory of the neighbour search.

    Returns:
        dict: t_data (float), t_null (n_preds,), p_value (float), k, n_ref (number of simulated summaries).
    """
    n_summaries = np.shape(s_obs)[-1]
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, n_summaries)
    cosmo_ids = np.asarray(cosmo_ids).reshape(-1)
    s_obs = np.asarray(s_obs, dtype=np.float64).reshape(1, n_summaries)

    mean = preds.mean(axis=0)
    chol = np.linalg.cholesky(np.cov(preds, rowvar=False))

    def whiten(x):
        return solve_triangular(chol, (x - mean).T, lower=True).T

    white = whiten(preds)
    tree = cKDTree(white)
    t_data = tree.query(whiten(s_obs), k=[k])[0][0, 0]

    # enough neighbours that k remain after dropping the largest possible set of siblings
    n_query = k + np.unique(cosmo_ids, return_counts=True)[1].max()
    t_null = np.empty(preds.shape[0])
    for start in range(0, preds.shape[0], chunk_size):
        stop = min(start + chunk_size, preds.shape[0])
        dist, idx = tree.query(white[start:stop], k=n_query)
        dist[cosmo_ids[idx] == cosmo_ids[start:stop, None]] = np.inf
        t_null[start:stop] = np.sort(dist, axis=1)[:, k - 1]

    # permutation p-value, counting the observation itself, so it is never 0
    p_value = (1 + np.sum(t_null >= t_data)) / (1 + t_null.size)

    return {"t_data": t_data, "t_null": t_null, "p_value": p_value, "k": k, "n_ref": preds.shape[0]}

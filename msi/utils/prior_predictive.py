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
"""

import os

import numpy as np
import matplotlib.pyplot as plt

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

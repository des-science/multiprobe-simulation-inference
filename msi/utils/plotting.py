"""
Created June 2023
Author: Arne Thomsen

Utils to plot the 1D and 2D marginals of samples from a posterior distribution.
"""

import os
import numpy as np

from trianglechain import TriangleChain
from seaborn import color_palette

from msfm.utils import parameters, logger

LOGGER = logger.get_logger(__file__)

method_label_dict = {
    "gp_abc": "GP ABC",
    "flow_likelihood": "flow (likelihood)",
    "flow_posterior": "flow (posterior)",
    "gaussian_mixture": "Gaussian (mixture)",
    "gaussian_likelihood": "Gaussian (likelihood)",
}

param_label_dict = {
    "Om": r"$\Omega_m$",
    "s8": r"$\sigma_8$",
    "Aia": r"$A_{IA}$",
    "n_Aia": r"$\eta_{A_{IA}}$",
    "bg": r"$b_g$",
    "n_bg": r"$\eta_{b_g}$",
}


def plot_chain(chain, params, out_dir=None, label="temp", scale_to_prior=True):
    """Plot a given MCMC chain as a triangle plot.

    Args:
        chain (np.ndarray): Array of MCMC samples of shape (n_samples, n_summaries)
        params (list): List of strings of the constrained cosmological parameters.
        out_dir (str, optional): Output directory to store the plot at. Defaults to None, then the plot is not saved.
        label (str, optional): Marks which inference method has been used. Defaults to "temp".
        scale_to_prior (bool, optional): Scale the cosmological parameter ranges to their respective priors. Defaults
            to True.
    """

    if scale_to_prior:
        ranges = dict(zip(params, parameters.get_prior_intervals(params)))
    else:
        ranges = None

    # initialize plot
    tri = TriangleChain(
        labels=[param_label_dict[param] for param in params],
        scatter_kwargs={"s": 500, "marker": "*", "zorder": 299},
        grid=True,
        fill=True,
        show_values=False,
        bestfit_method="median",
        ranges=ranges,
    )

    # plot contours
    tri.contour_cl(chain, names=params, label=label)

    # plot fiducial
    tri.scatter(
        dict(zip(params, parameters.get_fiducials(params))),
        label="fiducial",
        plot_histograms_1D=False,
        color="k",
        show_legend=True,
        scatter_vline_1D=True,
    )

    # save figure
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        tri.fig.savefig(os.path.join(out_dir, f"contours_{label}.png"), bbox_inches="tight", dpi=300)
    else:
        LOGGER.warning(f"Not saving the plot")


def make_contour_plot(
    model_dir,
    n_steps,
    params,
    out_dir=None,
    out_file="contours.png",
    methods=["gp_abc", "flow_likelihood", "flow_posterior", "gaussian_mixture", "gaussian_likelihood"],
    scale_to_prior=True,
):
    assert all(
        method in method_label_dict.keys() for method in methods
    ), f"Only {method_label_dict.keys()} are allowed"

    # cosmetics
    color_iter = iter(color_palette("tab10"))

    if scale_to_prior:
        ranges = dict(zip(params, parameters.get_prior_intervals(params)))
    else:
        ranges = None

    # initialize plot
    tri = TriangleChain(
        labels=[param_label_dict[param] for param in params],
        scatter_kwargs={"s": 500, "marker": "*", "zorder": 299},
        grid=True,
        fill=True,
        show_values=False,
        bestfit_method="median",
        ranges=ranges,
    )

    # plot contours
    for method in methods:
        chain = np.load(os.path.join(model_dir, f"chain_{n_steps}_{method}.npy"))
        tri.contour_cl(chain, names=params, label=method_label_dict[method], color=next(color_iter))

    # plot fiducial
    tri.scatter(
        dict(zip(params, parameters.get_fiducials(params))),
        label="fiducial",
        plot_histograms_1D=False,
        color="k",
        show_legend=True,
        scatter_vline_1D=True,
    )

    # save figure
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        tri.fig.savefig(os.path.join(out_dir, out_file), bbox_inches="tight", dpi=300)
    else:
        LOGGER.warning(f"Not saving the plot")

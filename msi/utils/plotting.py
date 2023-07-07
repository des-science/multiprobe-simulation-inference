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


def plot_chains(chains, params, out_dir=None, labels="temp", scale_to_prior=True, plot_fiducial=True):
    """Plot a given MCMC chain as a triangle plot.

    Args:
        chains (np.ndarray): Array of MCMC samples of shape (n_samples, n_summaries) or list of such arrays.
        params (list): List of strings of the constrained cosmological parameters or list of such lists.
        out_dir (str, optional): Output directory to store the plot at. Defaults to None, then the plot is not saved.
        label (str, optional): String label or list of labels to use in the plot. Defaults to "temp".
        scale_to_prior (bool, optional): Scale the cosmological parameter ranges to their respective priors. Defaults
            to True.
    """

    is_params_list_of_lists = any(isinstance(el, list) for el in params)

    if is_params_list_of_lists:
        all_params = set.union(*map(set, params))
    else:
        all_params = params

    if scale_to_prior:
        ranges = dict(zip(all_params, parameters.get_prior_intervals(all_params)))
    else:
        ranges = None

    # initialize plot
    tri = TriangleChain(
        params=all_params,
        labels=[param_label_dict[param] for param in all_params],
        scatter_kwargs={"s": 500, "marker": "*", "zorder": 299},
        grid=True,
        fill=True,
        show_values=False,
        bestfit_method="median",
        ranges=ranges,
    )

    # multiple chains
    if isinstance(chains, list):
        if isinstance(labels, list):
            # different parameters like https://cosmo-docs.phys.ethz.ch/trianglechain/multichains/multichains.html#plot-2-chains-with-different-parameters
            if is_params_list_of_lists:
                for chain, param, label in zip(chains, params, labels):
                    tri.contour_cl(chain, names=param, label=label)
            # shared parameters
            else:
                for chain, label in zip(chains, labels):
                    tri.contour_cl(chain, names=params, label=label)
        else:
            raise NotImplementedError

    # single chain
    elif isinstance(labels, str) and isinstance(params[0], str):
        tri.contour_cl(chains, names=params, label=chains)

    else:
        raise NotImplementedError

    # fiducial
    if plot_fiducial:
        tri.scatter(
            dict(zip(all_params, parameters.get_fiducials(all_params))),
            label="fiducial",
            plot_histograms_1D=False,
            color="k",
            show_legend=True,
            scatter_vline_1D=True,
        )

    # save figure
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        tri.fig.savefig(os.path.join(out_dir, f"contours_{labels}.png"), bbox_inches="tight", dpi=300)
    else:
        LOGGER.warning(f"Not saving the plot")


def plot_method_comparison(
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
        out_file = os.path.join(out_dir, out_file)
    else:
        out_file = os.path.join(model_dir, out_file)

    tri.fig.savefig(out_file, bbox_inches="tight", dpi=300)

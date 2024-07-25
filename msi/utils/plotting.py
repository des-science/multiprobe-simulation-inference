"""
Created June 2023
Author: Arne Thomsen

Utils to plot the 1D and 2D marginals of samples from a posterior distribution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from trianglechain import TriangleChain
from trianglechain.utils_plots import get_lines_and_labels
from seaborn import color_palette

from msi.utils.chains import load_des_y3_key_project_chain
from msfm.utils import parameters, logger, files

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
    "H0": r"$H_0$",
    "Ob": r"$\Omega_b$",
    "ns": r"$n_s$",
    "w0": r"$w_0$",
    "Aia": r"$A_{IA}$",
    "n_Aia": r"$\eta_{A_{IA}}$",
    "bg": r"$b_g$",
    "n_bg": r"$\eta_{b_g}$",
    "bg2": r"$b_{g,2}$",
    "n_bg2": r"$\eta_{b_{g,2}}$",
}


def plot_chains(
    chains,
    params,
    conf=None,
    # file
    out_dir=None,
    file_label=None,
    file_type="png",
    # cosmetics
    title=None,
    plot_labels="chain",
    scale_to_prior=True,
    group_params=False,
    tri_kwargs={},
    # cosmo
    plot_fiducial=True,
    fiducial_point=None,
    with_des_chain=False,
    des_tri="upper",
):
    """Plot a given MCMC chain as a triangle plot.

    Args:
        chains (np.ndarray): Array of MCMC samples of shape (n_samples, n_summaries) or list of such arrays.
        params (list): List of strings of the constrained cosmological parameters or list of such lists.
        out_dir (str, optional): Output directory to store the plot at. Defaults to None, then the plot is not saved.
        labels (str, optional): String label or list of labels to use in the plot. Defaults to "temp".
        title (str, optional): String to use as the super title within the figure. Defaults to None, then it is
            discarded.
        scale_to_prior (bool, optional): Scale the cosmological parameter ranges to their respective priors. Defaults
            to True.
        group_params (bool, optional): Whether to group the parameters by cosmological, intrinsic alignment and galaxy
            bias or not in the plot. Defaults to False.
        plot_fiducial (bool, optional): Whether to include the fiducial synthetic observation or not. Defaults to True.
        fiducial_point (dict, optional): Dictionary of parameter values for the given fiducial observation. This is
            meant for test purposes when a grid cosmology is used as the observation. Defaults to None, then the config
            values are used.
        with_des_chain (bool, optional): Whether to include a DES Y3 key project chain to compare to or not. Defaults
            to False.
        des_tri (str, optional): Determines whether the DES chain is included in the upper or lower triangle of the
            plot. Defaults to "upper".
    """
    conf = files.load_config(conf)

    is_params_list_of_lists = any(isinstance(el, list) for el in params)

    # different parameters
    if is_params_list_of_lists:
        # only keep unique params
        all_params = set.union(*map(set, params))

        # check that all of the params are supported
        config_params = (
            conf["analysis"]["params"]["cosmo"]
            + conf["analysis"]["params"]["ia"]
            + conf["analysis"]["params"]["bg"]["linear"]
        )

        if conf["analysis"]["modelling"]["quadratic_biasing"]:
            config_params += conf["analysis"]["params"]["bg"]["quadratic"]

        assert all([param in config_params for param in all_params])

        # reorder the params for the plot
        all_params = [param for param in config_params if param in all_params]

    # shared parameters
    else:
        all_params = params

    n_cosmo_params = sum([param in conf["analysis"]["params"]["cosmo"] for param in all_params])
    n_bg_params = sum(
        [
            param in conf["analysis"]["params"]["bg"]["linear"] + conf["analysis"]["params"]["bg"]["quadratic"]
            for param in all_params
        ]
    )
    n_ia_params = sum([param in conf["analysis"]["params"]["ia"] for param in all_params])
    includes_clustering_params = n_bg_params > 0
    includes_lensing_params = n_ia_params > 0

    if scale_to_prior:
        ranges = dict(zip(all_params, parameters.get_prior_intervals(all_params, conf=conf)))
    else:
        ranges = None

    # density estimation kwargs, used to smooth the plots, have to be defined globally
    de_kwargs = {
        "smoothing_parameter1D": 0.5,
        "smoothing_parameter2D": 0.5,
    }
    if group_params:
        n_per_group = [n_cosmo_params, n_ia_params, n_bg_params]
        n_per_group = list(filter(lambda x: x != 0, n_per_group))
        grouping_kwargs = {"empty_ratio": 0.1, "n_per_group": n_per_group}
    else:
        grouping_kwargs = {}

    # initialize plot
    tri = TriangleChain(
        params=all_params,
        ranges=ranges,
        show_values=False,
        # cosmetics
        labels=[param_label_dict[param] for param in all_params],
        n_ticks=3,
        grid=True,
        fill=True,
        de_kwargs=de_kwargs,
        grouping_kwargs=grouping_kwargs,
        scatter_kwargs={"s": 500, "marker": "*", "zorder": 299},
        **tri_kwargs,
    )

    # multiple chains
    if isinstance(chains, list):
        if isinstance(plot_labels, list):
            # different parameters like https://cosmo-docs.phys.ethz.ch/trianglechain/multichains/multichains.html#plot-2-chains-with-different-parameters
            if is_params_list_of_lists:
                for chain, param, label in zip(chains, params, plot_labels):
                    tri.contour_cl(chain, names=param, label=label)
            # shared parameters
            else:
                for chain, label in zip(chains, plot_labels):
                    tri.contour_cl(chain, names=params, label=label)
        else:
            raise NotImplementedError

    # single chain
    elif isinstance(plot_labels, str) and isinstance(params[0], str):
        tri.contour_cl(chains, names=params, label=plot_labels)

    else:
        raise NotImplementedError

    # DES key project chains
    if with_des_chain:
        # clustering params (2x2pt)
        if includes_clustering_params and not includes_lensing_params:
            # TODO treat the (per bin) galaxy bias somehow?
            des_params = ["Om", "s8"]
            des_probes = "2x2pt"
            des_ia = "tatt"
        # lensing params (1x2pt)
        elif includes_lensing_params and not includes_clustering_params:
            des_params = ["Om", "s8", "Aia", "n_Aia"]
            des_probes = "1x2pt"
            des_ia = "tatt"
        # combined probes (3x2pt)
        else:
            # TODO treat the (per bin) galaxy bias somehow?
            # des_params = ["Om", "s8", "Aia", "n_Aia"]
            des_params = ["Om", "s8", "H0", "Ob", "ns", "w0", "Aia", "n_Aia"]
            des_probes = "3x2pt"
            des_ia = "nla"

        # always compare to wCDM, not LambdaCDM
        des_cosmo = "wCDM"
        des_chain, des_weight = load_des_y3_key_project_chain(des_params, des_probes, des_cosmo, des_ia)

        tri.contour_cl(
            des_chain,
            prob=des_weight,
            names=des_params,
            label=f"des_kp_{des_probes}",
            fill=False,
            color="k",
            line_kwargs={"linewidth": 0.01000, "linestyle": "-"},
            tri=des_tri,
        )

    # fiducial
    if plot_fiducial:
        if fiducial_point is None:
            fiducial_point = dict(zip(all_params, parameters.get_fiducials(all_params)))

        tri.scatter(
            fiducial_point,
            label="synthetic truth",
            plot_histograms_1D=False,
            color="k",
            scatter_vline_1D=True,
            show_legend=True,
        )

    # title
    if title is not None:
        tri.fig.suptitle(title, fontsize=24, y=0.9)

    # save figure
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        if file_label is not None:
            out_file = os.path.join(out_dir, f"contours_{file_label}.{file_type}")
        else:
            out_file = os.path.join(out_dir, f"contours.{file_type}")

        tri.fig.savefig(os.path.join(out_file), bbox_inches="tight", dpi=300)
        LOGGER.info(f"Saved the plot to {out_file}")
    else:
        LOGGER.warning(f"Not saving the plot")

    return tri


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


def plot_human_summary(
    fidu_summs, grid_summs, out_dir=None, label=None, n_random_indices=20, bin_size=None, bin_names=None
):
    fig, ax = plt.subplots(figsize=(20, 10), nrows=2, sharex=True, sharey=True)

    # fiducial
    random_indices = np.random.choice(np.arange(fidu_summs.shape[0]), n_random_indices)

    for i in random_indices:
        ax[0].plot(fidu_summs[i])

    ax[0].set(xscale="linear", yscale="log", title="fiducial", xlabel="data vec dim", ylabel=r"$C_\ell$")
    ax[0].grid(True)

    # grid
    random_indices = np.random.choice(np.arange(grid_summs.shape[0]), n_random_indices)

    for i in random_indices:
        ax[1].plot(grid_summs[i])

    ax[1].set(title="grid", xlabel="data vec index", ylabel=r"$C_\ell$")
    ax[1].grid(True)

    # cosmetics
    if bin_size is not None and bin_names is not None:
        x = 0
        ticks = []
        for i, x in enumerate(np.arange(0, len(bin_names) * bin_size, bin_size)):
            ax[0].axvline(x, color="k", linestyle="--")
            ax[1].axvline(x, color="k", linestyle="--")

            ax[0].text(x + 3, ax[0].get_ylim()[1] - 0.5 * ax[0].get_ylim()[1], bin_names[i])
            ax[1].text(x + 3, ax[0].get_ylim()[1] - 0.5 * ax[0].get_ylim()[1], bin_names[i])

            ticks.append(x)

        ax[0].set_xticks(ticks)
        ax[1].set_xticks(ticks)

    # saving
    if out_dir is not None:
        if label is not None:
            out_file = os.path.join(out_dir, f"plot_{label}.png")
        else:
            out_file = os.path.join(out_dir, f"plot.png")

        fig.savefig(out_file, bbox_inches="tight", dpi=100)
        LOGGER.info(f"Saved the summary plot to {out_file}")

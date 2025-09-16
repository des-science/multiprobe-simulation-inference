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

from msfm.utils import parameters, logger, files, cross_statistics, prior
from msi.utils.chains import load_des_y3_key_project_chain

LOGGER = logger.get_logger(__file__)

method_label_dict = {
    "gp_abc": "GP ABC",
    "flow_likelihood": "flow (likelihood)",
    "flow_posterior": "flow (posterior)",
    "gaussian_mixture": "Gaussian (mixture)",
    "gaussian_likelihood": "Gaussian (likelihood)",
}

param_label_dict = {
    # cosmo
    "Om": r"$\Omega_m$",
    "s8": r"$\sigma_8$",
    "S8": r"$S_8$",
    "H0": r"$H_0$",
    "Ob": r"$\Omega_b$",
    "ns": r"$n_s$",
    "w0": r"$w_0$",
    # IA
    "Aia": r"$A_{IA}$",
    "n_Aia": r"$\eta_{A_{IA}}$",
    "bta": r"$b_{TA}$",
    # biasing
    "bg": r"$b_g$",
    "n_bg": r"$\eta_{b_g}$",
    "qbg": r"$b_{g,2}$",
    "n_qbg": r"$\eta_{b_{g,2}}$",
    "bg1": r"$b_g^1$",
    "bg2": r"$b_g^2$",
    "bg3": r"$b_g^3$",
    "bg4": r"$b_g^4$",
    "qbg1": r"$b_{g,q}^1$",
    "qbg2": r"$b_{g,q}^2$",
    "qbg3": r"$b_{g,q}^3$",
    "qbg4": r"$b_{g,q}^4$",
    "rg": r"$r_g$",
}


def plot_chains(
    chains,
    params,
    conf=None,
    # cosmetics
    tri_kwargs={},
    title=None,
    colors=None,
    fills=True,
    zorders=None,
    plot_labels="chain",
    group_params=False,
    density=False,
    show_legend=True,
    linestyles="-",
    # parameters
    params_plot=None,
    use_S8=False,
    ranges=None,
    scale_to_prior=True,
    include_prior=False,
    prior_params="cosmo",
    np_seed=8,
    # cosmo
    obs_cosmo="fiducial",
    obs_as_star=False,
    obs_label=None,
    with_des_chain=False,
    des_tri="upper",
    # file
    out_dir=None,
    file_label=None,
    file_type="png",
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

    use_S8 = use_S8 or ("S8" in params_plot if params_plot is not None else False)
    is_params_list_of_lists = any(isinstance(element, list) for element in params)
    multi_chain = isinstance(chains, list)
    assert not (
        is_params_list_of_lists and not multi_chain
    ), "If params is a list of lists, chains must be a list of arrays."

    # different parameters
    if is_params_list_of_lists:
        # only keep unique params
        all_params = set.union(*map(set, params))

        # check that all of the params are supported
        config_params = conf["analysis"]["params"]["cosmo"].copy()

        config_params += conf["analysis"]["params"]["ia"]["nla"]
        if conf["analysis"]["modelling"]["lensing"]["extended_nla"]:
            config_params += conf["analysis"]["params"]["ia"]["tatt"]

        config_params += conf["analysis"]["params"]["bg"]["linear"]
        if conf["analysis"]["modelling"]["clustering"]["quadratic_biasing"]:
            config_params += conf["analysis"]["params"]["bg"]["quadratic"]

        assert all([param in config_params for param in all_params])

        # reorder the params for the plot
        all_params = [param for param in config_params if param in all_params]

    # shared parameters
    else:
        all_params = params

    def sigma8_to_S8(sigma8, Om):
        return sigma8 * np.sqrt(Om / 0.3)

    if use_S8:
        all_params = [param if param != "s8" else "S8" for param in all_params]

        if is_params_list_of_lists:
            for param, chain in zip(params, chains):
                Om_index = param.index("Om")
                s8_index = param.index("s8")

                param[s8_index] = "S8"
                chain[:, s8_index] = sigma8_to_S8(chain[:, s8_index], chain[:, Om_index])
        else:
            Om_index = params.index("Om")
            s8_index = params.index("s8")
            params[s8_index] = "S8"

            if multi_chain:
                for chain in chains:
                    chain[:, s8_index] = sigma8_to_S8(chain[:, s8_index], chain[:, Om_index])
            else:
                chains[:, s8_index] = sigma8_to_S8(chains[:, s8_index], chains[:, Om_index])

    if params_plot is not None:
        all_params = [param for param in all_params if param in params_plot]

        if is_params_list_of_lists:
            for param, chain in zip(params, chains):
                param_indices = np.array([i for i, p in enumerate(param) if p in params_plot])
                chain = chain[:, param_indices]
                param = [param[i] for i in param_indices]
        else:
            param_indices = np.array([i for i, param in enumerate(params) if param in params_plot])
            params = [params[i] for i in param_indices]

            if multi_chain:
                chains = [chain[:, param_indices] for chain in chains]
            else:
                chains = chains[:, param_indices]

    n_cosmo_params = sum([param in conf["analysis"]["params"]["cosmo"] + ["S8"] for param in all_params])
    n_ia_params = sum(
        [
            param in conf["analysis"]["params"]["ia"]["nla"] + conf["analysis"]["params"]["ia"]["tatt"]
            for param in all_params
        ]
    )
    n_bg_params = sum(
        [
            param in conf["analysis"]["params"]["bg"]["linear"] + conf["analysis"]["params"]["bg"]["quadratic"]
            for param in all_params
        ]
    )

    includes_clustering_params = n_bg_params > 0
    includes_lensing_params = n_ia_params > 0

    if ranges is None:
        if scale_to_prior:
            ranges = dict(zip(all_params, parameters.get_prior_intervals(all_params, conf=conf)))
        else:
            ranges = {}
    elif isinstance(ranges, list):
        ranges = dict(zip(all_params, ranges))
    elif isinstance(ranges, dict):
        pass
    else:
        raise ValueError("ranges must be a list or a dict")

    if group_params:
        n_per_group = [n_cosmo_params, n_ia_params, n_bg_params]
        n_per_group = list(filter(lambda x: x != 0, n_per_group))
        grouping_kwargs = {"empty_ratio": 0.1, "n_per_group": n_per_group}
    else:
        grouping_kwargs = {}

    # TriangleChain
    tri_kwargs.setdefault("size", 2)

    tri_kwargs.setdefault(
        "de_kwargs", {"levels": [0.68, 0.95], "smoothing_parameter1D": 0.2, "smoothing_parameter2D": 0.2}
    )
    tri_kwargs.setdefault("line_kwargs", {"linestyles": linestyles, "linewidths": 1})  # 2d
    tri_kwargs.setdefault("hist_kwargs", {"linestyle": "-", "lw": 1})  # 1d
    tri_kwargs.setdefault("axlines_kwargs", {"linestyle": "--", "lw": 1})  # 1d

    tri_kwargs.setdefault("tick_fontsize", 8)
    tri_kwargs.setdefault("label_fontsize", 12)
    tri_kwargs.setdefault("legend_fontsize", 12)

    tri_kwargs.setdefault("n_ticks", 3)
    tri_kwargs.setdefault("tick_length", 2)
    tri_kwargs.setdefault("show_values", False)

    tri_kwargs.setdefault("fill", True)
    tri_kwargs.setdefault("grid", True)

    tri_kwargs.setdefault("scatter_kwargs", {"s": 500, "marker": "*", "zorder": 299})

    tri = TriangleChain(
        params=all_params,
        ranges=ranges,
        # cosmetics
        show_legend=show_legend,
        labels=[param_label_dict[param] for param in all_params],
        grouping_kwargs=grouping_kwargs,
        **tri_kwargs,
    )

    # multiple chains
    if multi_chain:
        colors = colors if isinstance(colors, list) else [colors] * len(chains)
        plot_labels = plot_labels if isinstance(plot_labels, list) else [plot_labels] * len(chains)
        params = params if is_params_list_of_lists else [params] * len(chains)
        linestyles = linestyles if isinstance(linestyles, list) else [linestyles] * len(chains)
        fills = fills if isinstance(fills, list) else [fills] * len(chains)
        zorders = zorders if isinstance(zorders, list) else [zorders] * len(chains)

        for chain, param, color, label, linestyle, fill, zorder in zip(
            chains, params, colors, plot_labels, linestyles, fills, zorders
        ):
            tri.contour_cl(
                chain,
                names=param,
                label=label,
                color=color,
                line_kwargs={**tri_kwargs["line_kwargs"], "linestyles": linestyle, "zorder": zorder},
                hist_kwargs={**tri_kwargs["hist_kwargs"], "linestyle": linestyle, "zorder": zorder},
                fill=fill,
            )

            if density:
                tri.density_image(chain, names=param, label=label)

    # single chain
    else:
        tri.contour_cl(chains, names=params, label=plot_labels, color=colors, fill=fills)

        if density:
            tri.density_image(chains, names=params, label=plot_labels)

    # DES key project chains
    if with_des_chain:
        # clustering params (2x2pt)
        if includes_clustering_params and not includes_lensing_params:
            des_params = ["Om", "s8", "bg1", "bg2", "bg3", "bg4"]
            des_probes = "2x2pt"
            des_ia = "tatt"
        # lensing params (1x2pt)
        elif includes_lensing_params and not includes_clustering_params:
            des_params = ["Om", "s8", "Aia", "n_Aia"]
            des_probes = "1x2pt"
            des_ia = "tatt"
        # combined probes (3x2pt)
        else:
            des_params = ["Om", "s8", "H0", "Ob", "ns", "w0", "Aia", "n_Aia", "bg1", "bg2", "bg3", "bg4"]
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
            tri=des_tri,
        )

    # observation
    if obs_cosmo is not None:
        if obs_cosmo == "fiducial":
            obs_cosmo = dict(zip(all_params, parameters.get_fiducials(all_params, conf=conf)))

        elif use_S8:
            obs_cosmo["S8"] = sigma8_to_S8(obs_cosmo["s8"], obs_cosmo["Om"])

        if obs_as_star:
            tri.scatter(
                obs_cosmo,
                params=all_params,
                label=obs_label,
                plot_histograms_1D=False,
                color="k",
                scatter_vline_1D=True,
            )
        else:
            tri.axlines(obs_cosmo, params=all_params, label=obs_label, color="k")

    if include_prior:
        if prior_params == "cosmo":
            prior_params = [param for param in all_params if param in conf["analysis"]["params"]["cosmo"] + ["S8"]]

        np.random.seed(np_seed)
        prior_rands = prior.generate_randoms(
            params=prior_params,
            n_draws=10_000_000,
            conf=conf,
            output_S8=use_S8,
        )
        tri.contour_cl(
            prior_rands,
            names=prior_params,
            color="k",
            alpha=0.1,
            plot_histograms_1D=False,
            de_kwargs={
                **tri_kwargs["de_kwargs"],
                "levels": [0.99],
                "n_points": 100,
                "n_levels_check": 2000,
                "inverted": True,
            },
            line_kwargs={**tri_kwargs["line_kwargs"], "zorder": -10},
            fill=True,
        )

    # only keep the last legend
    try:
        for legend in tri.fig.legends[:-1]:
            legend.remove()
    except AttributeError:
        pass

    # title
    if title is not None:
        tri.fig.suptitle(title, fontsize=16, y=0.9)

    # save figure
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        if file_label is not None:
            out_file = os.path.join(out_dir, f"contours_{file_label}.{file_type}")
        else:
            out_file = os.path.join(out_dir, f"contours.{file_type}")

        tri.fig.savefig(os.path.join(out_file), bbox_inches="tight", dpi=100)
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
        scatter_vline_1D=True,
    )

    # save figure
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, out_file)
    else:
        out_file = os.path.join(model_dir, out_file)

    tri.fig.savefig(out_file, bbox_inches="tight", dpi=300)


def plot_single_power_spectrum(
    cls,
    bin_size=None,
    bin_names=None,
    yscale="linear",
    with_lensing=None,
    with_clustering=None,
    with_cross_z=None,
    with_cross_probe=None,
    ylim=None,
    out_file=None,
):
    cls = np.squeeze(cls)

    if bin_names is None:
        _, bin_names = cross_statistics.get_cross_bin_indices(
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_cross_z=with_cross_z,
            with_cross_probe=with_cross_probe,
        )

    fig, ax = plt.subplots(figsize=(20, 5))

    ax.plot(cls)
    ax.set(
        title="mock observation", xscale="linear", yscale=yscale, xlabel="data vec dim", ylabel=r"$C_\ell$", ylim=ylim
    )
    ax.grid(True)

    # cosmetics
    if bin_size is not None:
        x = 0
        ticks = []
        for i, x in enumerate(np.arange(0, len(bin_names) * bin_size, bin_size)):
            ax.axvline(x, color="k", linestyle="--")
            ax.text(x + 3, 0.1, bin_names[i], transform=ax.get_xaxis_transform())
            ticks.append(x)

        ax.set_xticks(ticks)

    if out_file is not None:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        fig.savefig(out_file, bbox_inches="tight", dpi=100)

    return fig, ax


def plot_human_summary(
    fidu_summs,
    grid_summs,
    out_dir=None,
    label=None,
    n_random_indices=20,
    bin_size=None,
    bin_names=None,
    yscale="log",
    with_lensing=None,
    with_clustering=None,
    with_cross_z=None,
    with_cross_probe=None,
):
    if bin_names is None:
        _, bin_names = cross_statistics.get_cross_bin_indices(
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_cross_z=with_cross_z,
            with_cross_probe=with_cross_probe,
        )

    fig, ax = plt.subplots(figsize=(20, 10), nrows=2, sharex=True, sharey=True)

    # fiducial
    random_indices = np.random.choice(np.arange(fidu_summs.shape[0]), n_random_indices)

    for i in random_indices:
        ax[0].plot(fidu_summs[i])

    ax[0].set(xscale="linear", yscale=yscale, title="fiducial", xlabel="data vec dim", ylabel=r"$C_\ell$")
    ax[0].grid(True)

    # grid
    random_indices = np.random.choice(np.arange(grid_summs.shape[0]), n_random_indices)

    for i in random_indices:
        ax[1].plot(grid_summs[i])

    ax[1].set(title="grid", xlabel="data vec index", ylabel=r"$C_\ell$")
    ax[1].grid(True)

    # cosmetics
    if bin_size is not None:
        x = 0
        ticks = []
        for i, x in enumerate(np.arange(0, len(bin_names) * bin_size, bin_size)):
            ax[0].axvline(x, color="k", linestyle="--")
            ax[1].axvline(x, color="k", linestyle="--")

            ax[0].text(x + 3, 0.1, bin_names[i], transform=ax[0].get_xaxis_transform())
            ax[1].text(x + 3, 0.1, bin_names[i], transform=ax[1].get_xaxis_transform())

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

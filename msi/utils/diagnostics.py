# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen

Utils to check the quality of the density estimation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tarp import get_tarp_coverage

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def _assert_and_return_grid_pred_shapes(grid_preds_true, grid_preds_sample):
    """Utility function to assert the shapes of the grid predictions and return the (meaningful) dimensions.

    Args:
        grid_preds_true (np.ndarray): Of shape (n_cosmos, n_examples, n_summaries)
        grid_preds_sample (np.ndarray): Of shape (n_cosmos, n_samples, n_summaries)

    Returns:
        ints: The shapes.
    """
    assert grid_preds_true.shape[0] == grid_preds_sample.shape[0], "n_cosmos must be the same for both arrays"
    assert grid_preds_true.shape[-1] == grid_preds_sample.shape[-1], "n_summaries must be the same for both arrays"
    assert (
        grid_preds_sample.ndim == 3
    ), "grid_preds_sample must have 3 dims containing (n_cosmos, n_samples, n_summaries)"

    n_cosmos = grid_preds_true.shape[0]
    n_summaries = grid_preds_true.shape[-1]
    n_samples = grid_preds_sample.shape[1]

    if grid_preds_true.ndim == 3:
        n_examples = grid_preds_true.shape[1]
    elif grid_preds_true.ndim == 2:
        n_examples = 1

    return n_cosmos, n_summaries, n_examples, n_samples


def plot_histogram_check(grid_preds_true, grid_preds_sample, n_random_indices=10, out_dir=None):
    """Plot histograms to compare samples from the true distribution and from the (normalizing flow) model.

    Args:
        grid_preds_true (ndarray): The true predicted summaries (directly from the CosmoGrid evaluations) of shape
            (n_cosmos, n_examples, n_summaries).
        grid_preds_sample (ndarray): Samples from the model of shape (n_cosmos, n_samples, n_summaries).
        n_random_indices (int, optional): The number of random indices to select for plotting. Defaults to 10.
        out_dir (str, optional): The output directory to save the plot to. Defaults to None, then it isn't saved.
    """

    n_cosmos, n_summaries, _, _ = _assert_and_return_grid_pred_shapes(grid_preds_true, grid_preds_sample)

    fig, ax = plt.subplots(
        figsize=(5 * n_summaries, 3 * n_random_indices),
        nrows=n_random_indices,
        ncols=n_summaries,
        sharex="col",
        sharey="col",
    )

    cosmo_indices = np.random.randint(0, n_cosmos, n_random_indices)
    for i, i_cosmo in enumerate(cosmo_indices):
        for j in range(n_summaries):
            current_true = grid_preds_true[i_cosmo, :, j]
            current_sample = grid_preds_sample[i_cosmo, :, j]

            # different bins each
            current_all = np.concatenate([current_sample, current_true]).ravel()
            current_min = np.quantile(current_all, 0.01)
            current_max = np.quantile(current_all, 0.99)
            current_bins = np.linspace(current_min, current_max, 30)

            ax[i, j].hist(current_sample, bins=current_bins, density=True, label="model samples", alpha=0.5)
            ax[i, j].hist(current_true, bins=current_bins, density=True, label="true distribution", alpha=0.5)

            # cosmetics
            if i == 0:
                title = f"summary {j}, example index {i_cosmo}"
            else:
                title = f"example index {i_cosmo}"
            ax[i, j].set(title=title)
            ax[i, j].grid(True)

            if i == 0:
                ax[i, j].legend(loc="upper left")

    if out_dir is not None:
        fig.savefig(os.path.join(out_dir, "diagnostic_histogram.png"), bbox_inches="tight", dpi=100)


def plot_deeplss_check(grid_preds_true, grid_preds_sample, plot_per_summary_dim_hist=True, out_dir=None):
    """Plot the diagnostics presented in Appendix C and Fig. 9 of https://arxiv.org/pdf/2203.09616. These are
    relative statisticsfor mean and std, and per summary dimension histograms.

    Args:
        grid_preds_true (ndarray): The true predicted summaries (directly from the CosmoGrid evaluations) of shape
            (n_cosmos, n_examples, n_summaries).
        grid_preds_sample (ndarray): Samples from the model of shape (n_cosmos, n_samples, n_summaries).
        plot_per_summary_dim_hist (bool, optional): Whether to plot the per summary dimension histograms. Defaults to
            True.
        out_dir (str, optional): The output directory to save the plot to. Defaults to None, then it isn't saved.
    """

    _, n_summaries, n_examples, _ = _assert_and_return_grid_pred_shapes(grid_preds_true, grid_preds_sample)

    # "true" network predictions
    if n_examples > 1:
        mean_true = np.mean(grid_preds_true, axis=1)
        std_true = np.std(grid_preds_true, axis=1)
    else:
        raise NotImplementedError
        mean_true = grid_preds_true
        # the std should have to be computed within a single cosmology
        std_true = np.std(grid_preds_true, axis=0)
        # https://github.com/tomaszkacprzak/deep_lss/blob/main/deep_lss/notebooks/utils_plots.py#L1268C51-L1268C60
        # https://github.com/tomaszkacprzak/deep_lss/blob/main/deep_lss/notebooks/figures_deeplss_paper1.ipynb
        # https://github.com/tomaszkacprzak/deep_lss/blob/main/deep_lss/notebooks/likemodel_check.ipynb

    # samples from the model
    mean_sample = np.mean(grid_preds_sample, axis=1)
    std_sample = np.std(grid_preds_sample, axis=1)

    assert mean_sample.shape == std_sample.shape == mean_true.shape == std_true.shape

    # test statistics, these should be tightly centered around zero
    Delta_mu = (mean_sample - mean_true) / std_true
    Delta_sigma = (std_sample - std_true) / std_true

    Delta_mu = Delta_mu.ravel()
    Delta_sigma = Delta_sigma.ravel()

    # plot
    fig, ax = plt.subplots(ncols=2, figsize=(15, 10), sharey=True)

    ax[0].hist(Delta_mu, bins=50, density=True)
    ax[0].axvline(0, color="k", linestyle="--")
    ax[0].set(title="relative mean statistic", xlabel=r"$\Delta_\mu$", ylabel="probability density")
    ax[0].grid(True)

    ax[1].hist(Delta_sigma, bins=50, density=True)
    ax[1].axvline(0, color="k", linestyle="--")
    ax[1].set(title="relative spread statistic", xlabel=r"$\Delta_\sigma$")
    ax[1].grid(True)

    if out_dir is not None:
        fig.savefig(os.path.join(out_dir, "diagnostic_deeplss_relative_stat.png"), bbox_inches="tight", dpi=100)

    if plot_per_summary_dim_hist and n_summaries < 20 and n_examples != 1:
        fig, ax = plt.subplots(figsize=(15, 5), ncols=n_summaries, nrows=2)

        # per summary dimension mean
        # shared binning per summary dimension
        bin_mins = np.quantile(mean_sample, 0.01, axis=0)
        bin_maxs = np.quantile(mean_sample, 0.99, axis=0)
        bins = np.linspace(bin_mins, bin_maxs, 20)

        for i in range(n_summaries):
            ax[0, i].hist(mean_sample[:, i], bins=bins[:, i], label="model samples", density=True, alpha=0.5)
            ax[0, i].hist(mean_true[:, i], bins=bins[:, i], label="true distribution", density=True, alpha=0.5)

            ax[0, i].set(title=f"summary {i}", xlabel="mean")
            ax[0, i].grid(True)

        ax[0, 0].legend()
        ax[0, 0].set(ylabel="probability density")

        # per summary dimension std
        # shared binning per summary dimension
        bin_mins = np.quantile(std_sample, 0.01, axis=0)
        bin_maxs = np.quantile(std_sample, 0.99, axis=0)
        bins = np.linspace(bin_mins, bin_maxs, 20)

        for i in range(n_summaries):
            ax[1, i].hist(std_sample[:, i], bins=bins[:, i], label="model samples", density=True, alpha=0.5)
            ax[1, i].hist(std_true[:, i], bins=bins[:, i], label="true distribution", density=True, alpha=0.5)

            ax[1, i].set(xlabel="std")
            ax[1, i].grid(True)

        ax[1, 0].set(ylabel="probability density")

        fig.tight_layout()

        if out_dir is not None:
            fig.savefig(os.path.join(out_dir, "diagnostic_deeplss_per_summary.png"), bbox_inches="tight", dpi=100)


def plot_eecp_check(grid_preds_true, grid_preds_sample, grid_cosmos, model, n_confidence_levels=100, out_dir=None):
    """Plot the Expected Empirical Coverage Probability (EECP) developed in https://arxiv.org/pdf/2211.12346 for a
    given model. Note that this test was also performed in Appendix A.3 of https://arxiv.org/pdf/2211.12346.

    Args:
        grid_preds_true (ndarray): The true predicted summaries (directly from the CosmoGrid evaluations) of shape
            (n_cosmos, n_examples, n_summaries).
        grid_preds_sample (ndarray): Samples from the model of shape (n_cosmos, n_samples, n_summaries).
        grid_cosmos (ndarray): Array of shape (n_cosmos, n_params) of cosmological parameters.
        model (object): Model object with a `log_likelihood` method. This is for example a
            msi.flow_conductor.likelihood_flow.FlowLikelihood
        n_confidence_levels (int, optional): Number of confidence levels to plot. Defaults to 100.
        out_dir (str, optional): The output directory to save the plot to. Defaults to None, then it isn't saved.
    """

    n_cosmos, _, n_examples, n_samples = _assert_and_return_grid_pred_shapes(grid_preds_true, grid_preds_sample)
    assert grid_cosmos.shape[0] == n_cosmos, "n_cosmos must be the same for grid_cosmos and grid_preds"
    assert grid_cosmos.ndim == 2, "grid_cosmos must have 2 dims containing (n_cosmos, n_params)"

    # shape (n_cosmos, n_samples)
    log_probs_sample = model.log_likelihood(
        grid_preds_sample,
        np.repeat(grid_cosmos[:, np.newaxis, :], grid_preds_sample.shape[1], axis=1),
        return_numpy=True,
    )

    if n_examples > 1:
        # shape (n_cosmos, n_examples)
        log_probs_true = model.log_likelihood(
            grid_preds_true,
            np.repeat(grid_cosmos[:, np.newaxis, :], grid_preds_true.shape[1], axis=1),
            return_numpy=True,
        )
        # empirical expected coverage probability
        eecp = np.zeros((n_cosmos, n_examples, n_confidence_levels))

        # cosmos
        for i in LOGGER.progressbar(range(n_cosmos), at_level="info", desc="EECP: looping through cosmos"):
            sample_log_prob = log_probs_sample[i]
            sample_log_prob = np.sort(sample_log_prob)[::-1]

            # shape (n_conficence_levels,)
            log_prob_at_cls = sample_log_prob[:: n_samples // n_confidence_levels]

            # examples
            for j in range(n_examples):
                true_log_prob = log_probs_true[i, j]

                # per cosmology
                eecp[i, j] = true_log_prob >= log_prob_at_cls

        # mean over all cosmologies and examples
        eecp = np.mean(eecp, axis=(0, 1))

    else:
        # shape (n_cosmos)
        log_probs_true = model.log_likelihood(
            grid_preds_true,
            grid_cosmos,
            return_numpy=True,
        )
        # empirical expected coverage probability
        eecp = np.zeros((n_cosmos, n_confidence_levels))

        # cosmos
        for i in LOGGER.progressbar(range(n_cosmos), at_level="info", desc="EECP: looping through cosmos"):
            sample_log_prob = log_probs_sample[i]
            sample_log_prob = np.sort(sample_log_prob)[::-1]

            # shape (n_conficence_levels,)
            log_prob_at_cls = sample_log_prob[:: n_samples // n_confidence_levels]
            true_log_prob = log_probs_true[i]

            # per cosmology
            eecp[i] = true_log_prob >= log_prob_at_cls

        # mean over all cosmologies
        eecp = np.mean(eecp, axis=0)

    # plot
    true_coverage = np.linspace(0, 1, n_confidence_levels)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(true_coverage, eecp, label="model")
    ax.plot([0, 1], [0, 1], color="k", linestyle="--", label="ideal")

    ax.set(
        title="Expected Empirical Coverage Probability (EECP)",
        xlabel="credibility level",
        ylabel="empirical coverage",
        aspect="equal",
    )
    ax.text(0.2, 0.5, "underconfident", fontsize="large", alpha=0.5, rotation=45)
    ax.text(0.5, 0.2, "overconfident", fontsize="large", alpha=0.5, rotation=45)
    ax.legend(loc="upper left")
    ax.grid(True)

    if out_dir is not None:
        fig.savefig(os.path.join(out_dir, "diagnostic_eecp.png"), bbox_inches="tight", dpi=100)


def plot_tarp_check(
    grid_preds_true,
    grid_preds_sample,
    # random reference points
    grid_cosmos=None,
    randoms_dist=None,
    randoms_scale=1.0,
    randoms_dependence=False,
    np_seed=17,
    # plotting
    n_bootstrap=100,
    n_sigma=2,
    out_dir=None,
):
    """Plot the Tests of Accuracy with Random Points (TARP) diagnostic introduced in https://arxiv.org/pdf/2302.03026
    from https://github.com/Ciela-Institute/tarp.

    This function plots the TARP diagnostic, which measures the accuracy of a model's predictions
    using random reference points. It calculates the expected coverage and uncertainty at different
    credibility levels.

    Args:
        grid_preds_true (ndarray): The true predicted summaries (directly from the CosmoGrid evaluations) of shape
            (n_cosmos, n_examples, n_summaries).
        grid_preds_sample (ndarray): Samples from the model of shape (n_cosmos, n_samples, n_summaries).
        grid_cosmos (ndarray): Array of shape (n_cosmos, n_params) of cosmological parameters underlying the
            simulations. These might be used to generate data dependent randoms. Defaults to None.
        randoms_dist (str, optional): The distribution used to generate random reference points.
            Can be "normal", "uniform" or "constant". Defaults to None, then the TARP default is used.
        randoms_scale (float, optional): The scale parameter for the reference distribution. Defaults to 1.0.
        randoms_dependence (bool, optional): Whether the reference points should be dependent on the true
            predictions or not, see section 4.3 of the paper. Defaults to False.
        np_seed (int, optional): The seed for the NumPy random number generator. Defaults to 17.
        n_sigma (int, optional): The number of standard deviations to include in the uncertainty band in the plot.
            Defaults to 2.
        out_dir (str, optional): The output directory to save the plot to. Defaults to None, then it isn't saved.

    Raises:
        ValueError: If an invalid value is provided for reference_dist.

    Returns:
        None
    """

    # these come from the likelihood p(x|theta), not the posterior as in the TARP paper
    _, _, n_examples, _ = _assert_and_return_grid_pred_shapes(grid_preds_true, grid_preds_sample)

    # for its samples, tarp expects (n_samples, n_cosmos, n_summaries)
    grid_preds_sample = np.transpose(grid_preds_sample, (1, 0, 2))

    # manually rescale to [0, 1] like
    # https://github.com/Ciela-Institute/tarp/blob/b40f118d25dbc29cf00f3342be633c536d0464ab/src/tarp/drp.py#L89
    low = np.min(grid_preds_true, axis=(0, 1), keepdims=True)
    high = np.max(grid_preds_true, axis=(0, 1), keepdims=True)

    grid_preds_true = (grid_preds_true - low) / (high - low + 1e-10)
    grid_preds_sample = (grid_preds_sample - low) / (high - low + 1e-10)

    # define how the random reference points are generated
    rng = np.random.default_rng(seed=np_seed)

    if randoms_dist is None:
        assert not randoms_dependence, "randoms_dependence can only be used if randoms_dist is not None"
        LOGGER.info(f"TARP random reference points: Using the default")
        get_randoms = lambda shape: "random"

    elif randoms_dist == "constant":
        assert not randoms_dependence, "randoms_dependence can only be used if randoms_dist is not None"
        LOGGER.info(f"TARP random reference points: Using a constant value")
        get_randoms = lambda shape: np.ones(shape) * 0.5

    elif randoms_dist == "normal":
        LOGGER.info(f"TARP random reference points: Using a normal distribution, dependence = {randoms_dependence}")

        def get_randoms(shape):
            randoms = rng.normal(loc=0.5, scale=randoms_scale, size=shape)
            if randoms_dependence:
                assert grid_cosmos is not None, "grid_cosmos must be provided for data dependent randoms"
                randoms += np.mean(grid_cosmos)
            return np.clip(randoms, 0.0, 1.0)

    elif randoms_dist == "uniform":
        LOGGER.info(f"TARP random reference points: Using a uniform distribution, dependence = {randoms_dependence}")

        def get_randoms(shape):
            randoms = rng.uniform(low=(1 - randoms_scale) / 2, high=(1 + randoms_scale) / 2, size=shape)
            if randoms_dependence:
                assert grid_cosmos is not None, "grid_cosmos must be provided for data dependent randoms"
                randoms += np.mean(grid_cosmos)
            return np.clip(randoms, 0.0, 1.0)

    else:
        raise ValueError

    if n_examples > 1:
        LOGGER.info(f"TARP uncertainty from {n_examples} examples per cosmology")

        # there's multiple truth samples for each cosmology in this case
        ecps, alphas = [], []
        for i in LOGGER.progressbar(range(n_examples), at_level="info", desc="TARP: looping through examples"):
            # shape (n_sims, n_dim), these are summaries x from the true distribution p(x|theta) in this case
            truth = grid_preds_true[:, i, :]
            randoms = get_randoms(truth.shape)

            ecp, alpha = get_tarp_coverage(
                # shape (n_samples, n_sims, n_dim)
                samples=grid_preds_sample,
                # shape (n_sims, n_dim)
                theta=truth,
                references=randoms,
                metric="euclidean",
                # this could be used in addition to the sample loop
                bootstrap=False,
                norm=False,
            )
            ecps.append(ecp)
            alphas.append(alpha)

        # the alphas are all the same
        alpha = alphas[0]

        ecp_mean = np.mean(ecps, axis=0)
        ecp_std = np.std(ecps, axis=0)

    else:
        LOGGER.info(f"TARP uncertainty from {n_bootstrap} bootstrap samples")

        ecp, alpha = get_tarp_coverage(
            # shape (n_samples, n_sims, n_dim)
            samples=grid_preds_sample,
            # shape (n_sims, n_dim)
            theta=grid_preds_true,
            references=get_randoms(grid_preds_true.shape),
            metric="euclidean",
            bootstrap=True,
            num_bootstrap=n_bootstrap,
            norm=False,
        )
        ecp_mean = np.mean(ecp, axis=0)
        ecp_std = np.std(ecp, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(alpha, ecp_mean, label="model")
    ax.fill_between(
        alpha,
        ecp_mean - n_sigma * ecp_std,
        ecp_mean + n_sigma * ecp_std,
        label=str(n_sigma) + r"$\sigma$ uncertainty",
        alpha=0.3,
    )
    ax.plot([0, 1], [0, 1], color="k", ls="--", label="ideal")
    ax.set(
        title="Tests of Accuracy with Random Points (TARP)",
        xlabel="credibility level",
        ylabel="expected coverage",
        aspect="equal",
    )
    ax.legend(loc="upper left")
    ax.grid(True)

    if out_dir is not None:
        fig.savefig(os.path.join(out_dir, "diagnostic_tarp.png"), bbox_inches="tight", dpi=100)


def plot_sbc_checks():
    raise NotImplementedError

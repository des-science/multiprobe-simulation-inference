# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen

Utils to check the quality of the density estimation.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_sample_comparison(grid_preds_true, grid_preds_sample, n_random_indices=10):
    assert (
        grid_preds_true.ndim == grid_preds_sample.ndim == 3
    ), "grid_preds_gt and grid_preds_samples must have 3 dims containing (n_cosmos, n_samples, n_summaries)"
    assert grid_preds_true.shape[0] == grid_preds_sample.shape[0], "n_cosmos must be the same for both arrays"
    assert grid_preds_true.shape[2] == grid_preds_sample.shape[2], "n_summaries must be the same for both arrays"

    n_cosmos = grid_preds_true.shape[0]
    n_summaries = grid_preds_true.shape[-1]

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
            ax[i, j].set(title=f"example {i_cosmo}")
            ax[i, j].grid(True)

            if i == 0 and j == 0:
                ax[i, j].legend(loc="upper left")

    # TODO save fig?


def plot_deeplss_checks(grid_preds_true, grid_preds_sample):
    # like Appendix C and Fig. 9 in https://arxiv.org/pdf/2203.09616.pdf
    assert (
        grid_preds_true.ndim == grid_preds_sample.ndim == 3
    ), "grid_preds_gt and grid_preds_samples must have 3 dims containing (n_cosmos, n_samples, n_summaries)"
    assert grid_preds_true.shape[0] == grid_preds_sample.shape[0], "n_cosmos must be the same for both arrays"
    assert grid_preds_true.shape[2] == grid_preds_sample.shape[2], "n_summaries must be the same for both arrays"

    n_summaries = grid_preds_true.shape[-1]

    # "true" network predictions
    mean_true = np.mean(grid_preds_true, axis=1)
    std_true = np.std(grid_preds_true, axis=1)

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
    fig, ax = plt.subplots(ncols=2, figsize=(15, 5), sharey=True)

    ax[0].hist(Delta_mu, bins=50, density=True)
    ax[0].set(title="relative mean statistic", xlabel=r"$\Delta_\mu$", ylabel="probability density")
    ax[0].grid(True)

    ax[1].hist(Delta_sigma, bins=50, density=True)
    ax[1].set(title="relative spread statistic", xlabel=r"$\Delta_\sigma$")
    ax[1].grid(True)

    # TODO save fig?

    # per summary dimension mean
    fig, ax = plt.subplots(figsize=(15, 5), ncols=n_summaries)

    # shared binning per summary dimension
    bin_mins = np.quantile(mean_sample, 0.01, axis=0)
    bin_maxs = np.quantile(mean_sample, 0.99, axis=0)
    bins = np.linspace(bin_mins, bin_maxs, 20)

    for i in range(n_summaries):
        ax[i].hist(mean_true[:, i], bins=bins[:, i], label="true distribution", density=True, alpha=0.5)
        ax[i].hist(mean_sample[:, i], bins=bins[:, i], label="model samples", density=True, alpha=0.5)

        ax[i].set(title=f"summary {i}", xlabel="std")
        ax[i].grid(True)

    ax[0].legend()
    ax[0].set(ylabel="probability density")

    # TODO save fig?

    # per summary dimension std
    fig, ax = plt.subplots(figsize=(15, 5), ncols=n_summaries)

    # shared binning per summary dimension
    bin_mins = np.quantile(std_sample, 0.01, axis=0)
    bin_maxs = np.quantile(std_sample, 0.99, axis=0)
    bins = np.linspace(bin_mins, bin_maxs, 20)

    for i in range(n_summaries):
        ax[i].hist(std_true[:, i], bins=bins[:, i], label="true distribution", density=True, alpha=0.5)
        ax[i].hist(std_sample[:, i], bins=bins[:, i], label="model samples", density=True, alpha=0.5)

        ax[i].set(title=f"summary {i}", xlabel="std")
        ax[i].grid(True)

    ax[0].legend()
    ax[0].set(ylabel="probability density")

    # TODO save fig?


def compute_empirical_expected_coverate_probability():
    pass

    # n_confidence_levels = 100
    # n_examples = grid_preds.shape[1]

    # # shape (n_cosmos, n_examples, n_summary)
    # true_preds = grid_preds

    # # shape (n_cosmos, n_examples)
    # true_probs = np.exp(model.log_likelihood(np.repeat(grid_cosmos[:,np.newaxis,:], true_preds.shape[1], axis=1), true_preds))

    # # shape (n_cosmos, n_samples_per_cosmo)
    # sample_log_probs = model.log_likelihood(np.repeat(grid_cosmos[np.newaxis,:,:], pred_samples.shape[0], axis=0), pred_samples).T

    # # empirical expected coverage probability
    # eecp = np.zeros((n_cosmos, n_examples, n_confidence_levels))

    # # cosmos
    # for i in tqdm(range(n_cosmos)):
    #     true_cosmo = grid_cosmos[i]

    #     sample_log_prob = sample_log_probs[i]
    #     sample_prob = np.sort(np.exp(sample_log_prob))[::-1]

    #     # shape (n_conficence_levels,)
    #     prob_at_cls = sample_prob[::n_samples_per_cosmo//n_confidence_levels]

    #     # examples
    #     for j in range(n_examples):
    #         true_pred = true_preds[i,j]
    #         true_prob = true_probs[i,j]

    #         # per cosmology
    #         eecp[i,j] = true_prob >= prob_at_cls

    # # mean over all cosmologies and examples
    # eecp = np.mean(eecp, axis=(0,1))

    # # plot
    # true_coverage = np.linspace(0, 1, n_confidence_levels)

    # fig, ax = plt.subplots()

    # ax.plot(true_coverage, eecp, label="Gaussian Mixture Model")
    # ax.plot([0, 1], [0, 1], color="k", linestyle="--")

    # ax.set(xlabel="True coverage", ylabel="Empirical coverage")
    # ax.text(0.2, 0.5, "Conservative", fontsize="large", alpha=0.5, rotation=45)
    # ax.text(0.5, 0.2, "Overconfident", fontsize="large", alpha=0.5, rotation=45)
    # ax.set_aspect("equal")
    # ax.legend(loc="upper left", frameon=False)
    # ax.grid(True)

# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Posterior-level coverage testing, wired so it can run automatically at the end of run_inference.py's
--sample_posterior path (no need to execute deep_lss_paper/paper_2/pre-unblinding/2_posterior_coverage.ipynb
separately).

Two stages:
  1. sample_coverage_posteriors: GPU-batched sampling of the held-out mock observations -> mcmc_samples.h5
     (a minimal, single-GPU version of run_mcmc_for_coverage_tests.py: no sharding, no per-index files).
  2. run_coverage_tests / run_lc2st: the coverage diagnostics from the notebook (HPD, TARP, TARP marginals,
     SBC, l-C2ST), each saved as a plot under flow.model_dir/unblinding_plots.

run_coverage() orchestrates both. Every individual test is wrapped in its own try/except so a missing
optional dependency (sbi) or a single failure never aborts the others or the surrounding inference job.
The cheap, mcmc_samples-only tests (HPD, TARP, TARP marginals) depend only on msi.utils.diagnostics; SBC and
l-C2ST additionally need `sbi`.
"""

import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

from msfm.utils import logger, cosmogrid
from msfm.utils import files as msfm_files
from msi.utils import diagnostics

LOGGER = logger.get_logger(__file__)

# Coverage diagnostics enabled per run via flow_conf["diagnostics"]["tests"]. All default on: the whole
# coverage stage is already gated by run_inference.py's --sample_posterior, so a missing config block
# runs every test. hpd/tarp apply at both the likelihood- and posterior-level stages; sbc/lc2st are
# posterior-only.
_DEFAULT_TESTS = {"hpd": True, "tarp": True, "sbc": True, "lc2st": True}


def _test_flags(flow_conf):
    """Return the enabled-coverage-test flags, merging flow_conf overrides onto _DEFAULT_TESTS."""
    tests = dict(_DEFAULT_TESTS)
    tests.update(flow_conf.get("diagnostics", {}).get("tests", {}))
    return tests


def wide_prior_sobol_indices(msfm_conf):
    """Return the sobol_index values of the wide-grid CosmoGridV1 cosmologies (id_param < n_cosmos/2).

    CosmoGridV1 draws cosmologies from two Sobol sequences -- a wide and a narrow one -- so the grid
    sampling density is denser in the Om-s8 centre than the (wide) analysis prior. The held-out mock
    observations used for posterior coverage must follow the analysis prior, so we keep only the wide
    half. Membership is read from data/CosmoGridV1_metainfo.h5's parameters/grid table, where id_param
    in [0, n_cosmos/2) is the wide grid; each cosmology's sobol_index matches the per-row i_sobol stored
    in the preds files, so callers can mask held-out rows by `np.isin(i_sobol, wide_set)`.
    """
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(msfm_files.__file__), "../.."))
    meta_info_file = os.path.join(repo_dir, msfm_conf["files"]["meta_info"])
    params_info = cosmogrid.get_cosmo_params_info(meta_info_file, "grid")
    n_cosmos = msfm_conf["analysis"]["grid"]["n_cosmos"]
    return params_info["sobol_index"][params_info["id_param"] < n_cosmos // 2]


def _held_out_split(flow, grid_preds, grid_cosmos, i_signal, flow_conf):
    """Reproduce the flow's deterministic, signal-grouped held-out validation split and return
    (x_vali, theta_vali) as torch tensors.

    Shared by the likelihood- and posterior-level coverage stages so both judge calibration on the
    identical held-out mocks (seen by neither the compression network nor the flow -- see
    sample_coverage_posteriors for the 80%/20%/10% split chain). Works for a single LikelihoodFlow and a
    LikelihoodFlowEnsemble (whose _prepare_data exposes vali_dset on the ensemble).
    """
    vali_split = flow_conf.get("training", {}).get("vali_split", 0.1)
    flow._prepare_data(x=grid_preds, theta=grid_cosmos, batch_size=10000, vali_split=vali_split, group_ids=i_signal)
    x_vali = flow.vali_dset.dataset.tensors[0][flow.vali_dset.indices]
    theta_vali = flow.vali_dset.dataset.tensors[1][flow.vali_dset.indices]
    return x_vali, theta_vali


def sample_coverage_posteriors(
    flow, grid_preds, grid_cosmos, i_signal, flow_conf, i_sobol=None, msfm_conf=None, i_noise=None
):
    """Sample the posterior for the held-out mock observations and write flow.model_dir/mcmc_samples.h5.

    Reproduces the flow's signal-grouped validation split -- exactly as
    run_mcmc_for_coverage_tests._set_up_flow does -- so the mock observations were seen by neither network:
    configs/data sets signal_indices=0.8, so the compression network trains only on the first 80% of signal
    realizations and preds_*.h5 holds only the held-out 20% (grid/.../test); the flow then validates on a
    signal-grouped 10% of those, which is what we resample here.

    When flow_conf["diagnostics"]["prior_selection"] == "wide", the held-out mocks are further restricted to
    the wide-grid cosmologies (via i_sobol + msfm_conf) so the coverage truths follow the wide analysis
    prior instead of the wide+narrow CosmoGrid Sobol density -- a prerequisite for valid TARP/SBC/HPD.

    When both i_sobol and i_noise are given (alongside i_signal), the per-mock realization indices
    (i_sobol, i_signal, i_noise) are tracked through the SAME selection as x_true and saved as a
    `real_idx` dataset. This lets the cross-probe PPC calibration pair each obs-probe mock with the
    rep-probe summary of the SAME sky realization (msi.utils.ppc.PosteriorPredictiveChecks); it is
    skipped (with a warning) when those indices are unavailable.

    Returns:
        dict: {x_true, theta_true, log_prob_true, theta_sample, log_prob_sample} as numpy arrays, with the
        same layout/shapes written to mcmc_samples.h5 (i.e. what the paper's coverage notebook expects).
    """
    import torch

    mcmc_conf = flow_conf.get("mcmc", {})
    n_walkers = mcmc_conf.get("n_walkers", 1024)
    n_steps = mcmc_conf.get("n_steps", 1000)
    n_burnin_steps = mcmc_conf.get("n_burnin_steps", 1000)
    use_validation_weights = mcmc_conf.get("use_validation_weights", True)
    method = mcmc_conf.get("method", "ensemble")  # same switch as the plotted MCMC chains
    n_sims = flow_conf.get("diagnostics", {}).get("n_obs", 1000)
    n_samples_out = min(10000, n_steps * n_walkers)

    # reproduce the exact held-out validation split (deterministic, grouped by signal realization)
    x_vali, theta_vali = _held_out_split(flow, grid_preds, grid_cosmos, i_signal, flow_conf)

    # Track per-row realization indices through the SAME selection as x_vali (-> wide mask -> stride),
    # so the saved mocks carry their (i_sobol, i_signal, i_noise) identity for cross-probe pairing.
    save_real_idx = i_sobol is not None and i_noise is not None
    if save_real_idx:
        vali_indices = np.asarray(flow.vali_dset.indices)
        real_idx_vali = np.stack([np.asarray(a).reshape(-1) for a in (i_sobol, i_signal, i_noise)], axis=1)[
            vali_indices
        ]
    else:
        real_idx_vali = None
        LOGGER.warning(
            "i_sobol/i_noise not provided; not saving real_idx to mcmc_samples.h5 -- cross-probe PPC "
            "calibration will be unavailable for this run."
        )

    # optionally keep only wide-grid cosmologies so the coverage truths follow the (wide) analysis prior
    # rather than the wide+narrow Sobol density. flow.vali_dset.indices (set by _held_out_split) maps the
    # held-out rows back into the row-aligned i_sobol, so the mask lines up with x_vali/theta_vali.
    prior_selection = flow_conf.get("diagnostics", {}).get("prior_selection", "all")
    if prior_selection == "wide":
        if i_sobol is None or msfm_conf is None:
            raise ValueError("diagnostics.prior_selection='wide' requires i_sobol and msfm_conf.")
        wide_set = wide_prior_sobol_indices(msfm_conf)
        i_sobol_vali = np.asarray(i_sobol)[flow.vali_dset.indices]
        keep = np.isin(i_sobol_vali, wide_set)
        LOGGER.info(f"prior_selection='wide': {keep.sum()} / {keep.size} held-out mocks on the wide grid")
        keep_t = torch.as_tensor(keep, device=x_vali.device)
        x_vali, theta_vali = x_vali[keep_t], theta_vali[keep_t]
        if save_real_idx:
            real_idx_vali = real_idx_vali[keep]
    elif prior_selection != "all":
        raise ValueError(f"Unknown diagnostics.prior_selection={prior_selection!r}; expected 'all' or 'wide'.")

    # Sort the held-out rows by realization identity before subsampling. The stride below selects by row
    # POSITION, so without this the mock set depends on how the upstream pipeline happened to pack its
    # (i_signal, i_noise) example axis. The maps and Cls pipelines pack it transposed, and the stride
    # shares a factor with the per-cosmology block size, so the two aliased onto DIFFERENT realizations:
    # their 1000-mock sets covered the same 1000 cosmologies but overlapped in only 200 rows, leaving the
    # two-point baseline unpairable against the map runs. Sorting on (i_sobol, i_signal, i_noise) makes
    # the selection a function of realization identity alone, so every pipeline yields the same mocks
    # regardless of layout. This is a no-op for the maps pipeline, whose rows are already in this order.
    if save_real_idx:
        canonical = np.lexsort((real_idx_vali[:, 2], real_idx_vali[:, 1], real_idx_vali[:, 0]))
        if not np.array_equal(canonical, np.arange(canonical.size)):
            LOGGER.info(
                "held-out rows were not in (i_sobol, i_signal, i_noise) order; reordering so the mock "
                "selection matches other pipelines' runs"
            )
            order_t = torch.as_tensor(canonical, device=x_vali.device)
            x_vali, theta_vali = x_vali[order_t], theta_vali[order_t]
            real_idx_vali = real_idx_vali[canonical]
    else:
        LOGGER.warning(
            "without i_sobol/i_noise the held-out rows cannot be put in a canonical order, so the mock "
            "selection below depends on the upstream example-axis layout and these mocks may not be "
            "pairable with runs from another pipeline."
        )

    n_cosmos = x_vali.shape[0]
    if n_cosmos < n_sims:
        LOGGER.warning(f"only {n_cosmos} held-out mocks available; reducing n_obs from {n_sims} to {n_cosmos}")
        n_sims = n_cosmos
    x_true = x_vali[:: n_cosmos // n_sims][:n_sims]
    theta_true = theta_vali[:: n_cosmos // n_sims][:n_sims]
    if save_real_idx:
        real_idx_true = real_idx_vali[:: n_cosmos // n_sims][:n_sims]
    LOGGER.info(f"Coverage sampling {n_sims} held-out mock observations in a single batched pass")

    # one batched run over all mock observations (batch size = n_sims, fits a single GPU)
    chain, _ = flow.sample_posterior_batched(
        x_true,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burnin_steps=n_burnin_steps,
        use_validation_weights=use_validation_weights,
        method=method,
    )  # (n_sims, n_steps * n_walkers, n_params)

    # raw flow log-likelihood of the true cosmology for each observation (batched). use_validation_weights
    # keeps the saved log-probs consistent with the (weighted) ensemble posterior; no-op for a single flow.
    log_prob_true = flow.log_likelihood(
        x_true, theta_true, return_numpy=True, use_validation_weights=use_validation_weights
    )

    x_true_np = np.asarray(x_true.cpu())
    n_params = chain.shape[-1]
    theta_sample = np.empty((n_samples_out, n_sims, n_params), dtype=np.float32)
    log_prob_sample = np.empty((n_samples_out, n_sims), dtype=np.float32)
    for i in range(n_sims):
        # too many samples make the test slow and are not needed
        sel = np.random.choice(chain.shape[1], n_samples_out, replace=False)
        samples = chain[i][sel]
        theta_sample[:, i] = samples
        x_rep = np.repeat(x_true_np[i][None, :], n_samples_out, axis=0)
        log_prob_sample[:, i] = flow.log_likelihood(
            x_rep, samples, return_numpy=True, use_validation_weights=use_validation_weights
        )
    del chain  # free the (potentially tens of GB) full-chain host buffer

    samples = {
        "x_true": x_true_np,
        "theta_true": np.asarray(theta_true.cpu()),
        "log_prob_true": np.asarray(log_prob_true),
        "theta_sample": theta_sample,
        "log_prob_sample": log_prob_sample,
    }
    if save_real_idx:
        samples["real_idx"] = real_idx_true

    out_file = os.path.join(flow.model_dir, "mcmc_samples.h5")
    with h5py.File(out_file, "w") as f:
        for key, value in samples.items():
            f.create_dataset(key, data=value)
    LOGGER.info(f"Saved coverage samples to {out_file}")

    return samples


def _save(fig, plot_dir, name):
    plot_file = os.path.join(plot_dir, name)
    fig.savefig(plot_file, bbox_inches="tight", dpi=100)
    plt.close(fig)
    LOGGER.info(f"Saved {plot_file}")


def _coverage_axes(title):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="k", linestyle="--")
    ax.set(title=title, aspect="equal", xlabel="credibility level", ylabel="expected coverage")
    return fig, ax


def run_coverage_tests(samples, params, plot_dir, tests=None):
    """Run the mcmc_samples-only coverage diagnostics (HPD, TARP, TARP marginals, SBC) and save plots.

    ``tests`` is the enabled-test flag dict (see _test_flags); each diagnostic runs only if its flag is
    set. Defaults to _DEFAULT_TESTS (all on) when omitted.
    """
    tests = tests or _DEFAULT_TESTS
    theta_true = samples["theta_true"]
    theta_sample = samples["theta_sample"]
    log_prob_true = samples["log_prob_true"]
    log_prob_sample = samples["log_prob_sample"]

    # highest posterior density (HPD)
    if tests["hpd"]:
        try:
            hpd_alpha, hpd_ecp = diagnostics.posterior_hpd_check(log_prob_true, log_prob_sample, n_alpha=100)
            fig, ax = _coverage_axes("HPD")
            ax.plot(hpd_alpha, hpd_ecp)
            _save(fig, plot_dir, "2_posterior_hpd.png")
        except Exception as e:
            LOGGER.warning(f"HPD check failed ({type(e).__name__}: {e})")

    # test of accuracy with random points (TARP), full + marginals
    if tests["tarp"]:
        try:
            tarp_alpha, tarp_ecp, tarp_std = diagnostics.posterior_tarp_check(
                theta_true, theta_sample, n_bootstrap=100, n_alpha=50
            )
            fig, ax = _coverage_axes("TARP")
            ax.plot(tarp_alpha, tarp_ecp)
            ax.fill_between(
                tarp_alpha, tarp_ecp - 2 * tarp_std, tarp_ecp + 2 * tarp_std, alpha=0.3, label=r"2$\sigma$"
            )
            _save(fig, plot_dir, "2_posterior_tarp.png")
        except Exception as e:
            LOGGER.warning(f"TARP check failed ({type(e).__name__}: {e})")

        try:
            marginal_param_sets = [["Om"], ["s8"], ["w0"], ["Om", "s8"], ["Om", "s8", "w0"]]
            fig, ax = _coverage_axes("TARP (marginals)")
            for param_set in marginal_param_sets:
                if not all(p in params for p in param_set):
                    continue
                indices = [params.index(p) for p in param_set]
                m_alpha, m_ecp, m_std = diagnostics.posterior_tarp_check(
                    theta_true[:, indices], theta_sample[:, :, indices], n_bootstrap=100, n_alpha=50
                )
                (line,) = ax.plot(m_alpha, m_ecp, label=", ".join(param_set))
                ax.fill_between(m_alpha, m_ecp - 2 * m_std, m_ecp + 2 * m_std, alpha=0.3, color=line.get_color())
            ax.legend(fontsize=8)
            _save(fig, plot_dir, "2_posterior_tarp_marginals.png")
        except Exception as e:
            LOGGER.warning(f"TARP marginals check failed ({type(e).__name__}: {e})")

    # simulation-based calibration (SBC) -- needs sbi
    if tests["sbc"]:
        try:
            import torch
            from sbi.diagnostics import check_sbc
            from sbi.analysis import sbc_rank_plot

            ranks, dap_samples = diagnostics.run_sbc_precomputed(theta_true, theta_sample)
            num_posterior_samples = theta_sample.shape[0]
            check_stats = check_sbc(
                torch.from_numpy(ranks.astype(np.float32)),
                torch.from_numpy(theta_true),
                torch.from_numpy(dap_samples),
                num_posterior_samples=num_posterior_samples,
            )
            LOGGER.info(f"SBC Kolmogorov-Smirnov p-values = {check_stats['ks_pvals'].numpy()}")
            LOGGER.info(f"SBC c2st (ranks) = {check_stats['c2st_ranks'].numpy()}")
            LOGGER.info(f"SBC c2st (dap)   = {check_stats['c2st_dap'].numpy()}")

            # ~1000 sims over 100 bins gives ~10 noisy counts/bin; coarser bins make the histogram readable
            fig, ax = sbc_rank_plot(
                ranks=ranks, num_posterior_samples=num_posterior_samples, plot_type="hist", num_bins=20
            )
            _save(fig, plot_dir, "2_posterior_sbc.png")
        except ImportError:
            LOGGER.warning("sbi not available; skipping SBC check.")
        except Exception as e:
            LOGGER.warning(f"SBC check failed ({type(e).__name__}: {e})")


def run_likelihood_coverage_tests(
    grid_preds_true, grid_preds_sample, grid_cosmos, flow, plot_dir, tarp_kwargs=None, tests=None
):
    """Run the likelihood-level coverage diagnostics (HPD/EECP, TARP) and save plots as 1_likelihood_*.png.

    The likelihood-level analogue of run_coverage_tests: same plot helpers (_coverage_axes / _save), only
    the content differs -- these operate in summary space on x ~ p(x|theta) drawn from the flow vs the true
    simulated summaries, not on posterior samples of theta. ``tests`` is the enabled-test flag dict (see
    _test_flags); only the hpd/tarp flags apply at this level. Defaults to _DEFAULT_TESTS.
    """
    tests = tests or _DEFAULT_TESTS
    if tarp_kwargs is None:
        tarp_kwargs = {"n_bootstrap": 200, "n_alpha_bins": 50}

    # highest posterior density (HPD / EECP) in summary space
    if tests["hpd"]:
        try:
            hpd_alpha, hpd_ecp = diagnostics.plot_eecp_check(
                grid_preds_true, grid_preds_sample, grid_cosmos, flow, do_plot=False
            )
            fig, ax = _coverage_axes("HPD")
            ax.plot(hpd_alpha, hpd_ecp)
            _save(fig, plot_dir, "1_likelihood_hpd.png")
        except Exception as e:
            LOGGER.warning(f"likelihood HPD check failed ({type(e).__name__}: {e})")

    # test of accuracy with random points (TARP) in summary space
    if tests["tarp"]:
        try:
            tarp_alpha, tarp_ecp, tarp_std = diagnostics.plot_tarp_check(
                grid_preds_true, grid_preds_sample, grid_cosmos, do_plot=False, **tarp_kwargs
            )
            fig, ax = _coverage_axes("TARP")
            ax.plot(tarp_alpha, tarp_ecp)
            ax.fill_between(
                tarp_alpha, tarp_ecp - 2 * tarp_std, tarp_ecp + 2 * tarp_std, alpha=0.3, label=r"2$\sigma$"
            )
            _save(fig, plot_dir, "1_likelihood_tarp.png")
        except Exception as e:
            LOGGER.warning(f"likelihood TARP check failed ({type(e).__name__}: {e})")


def run_likelihood_coverage(flow, grid_preds, grid_cosmos, i_signal, flow_conf, n_likelihood_samples=100):
    """Orchestrate the likelihood-level coverage stage: sample p(x|theta) for the held-out mock
    observations and run the HPD (EECP) and TARP diagnostics, with plots saved as 1_likelihood_*.png under
    flow.model_dir/unblinding_plots.

    The likelihood-level counterpart of run_coverage; mirrors
    deep_lss_paper/paper_2/pre-unblinding/1_likelihood_coverage.ipynb so it runs automatically as part of
    run_inference.py. Uses the same held-out validation split (_held_out_split) as the posterior-level
    stage, so both levels judge calibration on the identical mock observations.
    """
    tests = _test_flags(flow_conf)
    if not (tests["hpd"] or tests["tarp"]):
        LOGGER.info("Likelihood coverage: hpd and tarp both disabled; skipping stage.")
        return

    x_vali, theta_vali = _held_out_split(flow, grid_preds, grid_cosmos, i_signal, flow_conf)
    x_true = np.asarray(x_vali.cpu())
    theta_true = np.asarray(theta_vali.cpu())

    n_likelihood_samples = flow_conf.get("diagnostics", {}).get("n_likelihood_samples", n_likelihood_samples)

    # thin to n_obs, matching sample_coverage_posteriors so the 1_ and 2_ plots cover the same mocks
    n_obs = min(flow_conf.get("diagnostics", {}).get("n_obs", 1000), x_true.shape[0])
    step = x_true.shape[0] // n_obs
    x_true = x_true[::step][:n_obs]
    theta_true = theta_true[::step][:n_obs]
    LOGGER.info(f"Likelihood coverage on {n_obs} held-out mock observations, {n_likelihood_samples} samples each")

    grid_preds_sample = flow.sample_likelihood(
        theta_true, n_samples=n_likelihood_samples, batch_size=10000, return_numpy=True
    )

    plot_dir = os.path.join(flow.model_dir, "unblinding_plots")
    os.makedirs(plot_dir, exist_ok=True)
    run_likelihood_coverage_tests(x_true, grid_preds_sample, theta_true, flow, plot_dir, tests=tests)


def lc2st_scores(samples, obs_pred, post_samples_star, conf_alpha=0.05, n_eval=10_000, seed=None):
    """Run the Local Classifier Two-Sample Test (l-C2ST) at one observation, following the sbi tutorial,
    and return its scores without plotting anything. Needs sbi.

    Split out of run_lc2st so that a caller which already has the arrays on disk -- the coverage samples
    and the observation's chain, e.g. a paper figure -- can obtain the same numbers without a flow, a GPU
    or the diagnostic plots. run_lc2st is the pipeline's wrapper around it.

    Args:
        samples: the coverage samples (mcmc_samples.h5) as loaded by sample_coverage_posteriors; only
            x_true, theta_true and theta_sample[0] are used, i.e. one posterior realization per
            calibration cosmology.
        obs_pred: the observation's network summary, of shape (n_summaries,).
        post_samples_star: the observation's posterior chain, of shape (n_samples, n_params).
        conf_alpha: significance level of the test.
        n_eval: number of posterior samples the classifier is evaluated on; the chain is subsampled
            down to this. Note the statistic is noticeably sensitive to which subsample is drawn.
        seed: seed for that subsample. None keeps the global numpy stream, i.e. an unseeded draw.

    Returns:
        dict: {probs_data, probs_null, T_data, T_null, p_value, reject, conf_alpha}.
    """
    import torch
    from sbi.diagnostics.lc2st import LC2ST

    # calibration set from the coverage samples; single posterior realization per calibration cosmology
    xs_star = np.asarray(obs_pred, dtype=np.float32)
    x_cal = samples["x_true"]
    theta_cal = samples["theta_true"]
    post_samples_cal = samples["theta_sample"][0]

    rng = np.random.default_rng(seed) if seed is not None else np.random
    i_rand = rng.choice(post_samples_star.shape[0], n_eval)
    post_samples_star = np.asarray(post_samples_star)[i_rand]

    xs_star = torch.from_numpy(xs_star)
    x_cal = torch.from_numpy(x_cal)
    theta_cal = torch.from_numpy(theta_cal)
    post_samples_cal = torch.from_numpy(post_samples_cal)
    post_samples_star = torch.from_numpy(post_samples_star.astype(np.float32))

    lc2st = LC2ST(thetas=theta_cal, xs=x_cal, posterior_samples=post_samples_cal, classifier="mlp", num_ensemble=1)
    # sbi's LC2ST drives its classifier-training tqdm bars off `verbosity` (disable=verbosity<1);
    # keep them only at debug level, consistent with the MCMC/diagnostics bars.
    lc2st_verbosity = 1 if LOGGER.islevel("debug") else 0
    lc2st.train_under_null_hypothesis(verbosity=lc2st_verbosity)
    lc2st.train_on_observed_data(verbosity=lc2st_verbosity)

    probs_data, _ = lc2st.get_scores(
        theta_o=post_samples_star, x_o=xs_star, return_probs=True, trained_clfs=lc2st.trained_clfs
    )
    T_data = lc2st.get_statistic_on_observed_data(theta_o=post_samples_star, x_o=xs_star)
    probs_null, T_null = lc2st.get_statistics_under_null_hypothesis(
        theta_o=post_samples_star, x_o=xs_star, return_probs=True
    )
    return {
        "probs_data": np.asarray(probs_data),
        "probs_null": np.asarray(probs_null),
        "T_data": float(T_data),
        "T_null": np.asarray(T_null),
        "p_value": float(lc2st.p_value(post_samples_star, xs_star)),
        "reject": bool(lc2st.reject_test(post_samples_star, xs_star, alpha=conf_alpha)),
        "conf_alpha": conf_alpha,
    }


def run_lc2st(samples, params, flow, obs_pred, obs_label, plot_dir, conf_alpha=0.05):
    """l-C2ST for one observation, plotted as the two unblinding diagnostics. Uses the coverage
    calibration set (x_true/theta_true/theta_sample) and the observation's posterior chain (loaded from
    chain_{obs_label}.npy if available, else resampled). Needs sbi."""
    from sbi.analysis.plot import pp_plot_lc2st

    chain_file = os.path.join(flow.model_dir, f"chain_{obs_label}.npy")
    if os.path.exists(chain_file):
        post_samples_star = np.load(chain_file)
    else:
        post_samples_star = np.asarray(flow.sample_posterior(np.asarray(obs_pred, dtype=np.float32), label=obs_label))

    scores = lc2st_scores(samples, obs_pred, post_samples_star, conf_alpha=conf_alpha)
    T_data, T_null = scores["T_data"], scores["T_null"]
    p_value, reject = scores["p_value"], scores["reject"]
    LOGGER.info(f"l-C2ST [{obs_label}]: p-value = {p_value:.3f}, reject = {reject}")

    # quantitative: observed statistic vs null distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(T_null, bins=50, density=True, alpha=0.5, label="Null")
    ax.axvline(T_data, color="red", label="Observed")
    for q in np.quantile(T_null, [0, 1 - conf_alpha]):
        ax.axvline(q, color="black", linestyle="--")
    ax.set(xlabel="test statistic", ylabel="density", title=f"{obs_label}: p-value = {p_value:.3f}, reject = {reject}")
    ax.legend(loc="upper right")
    _save(fig, plot_dir, f"2_posterior_l-C2ST_quant_{obs_label}.png")

    # qualitative: pp-plot
    fig, ax = plt.subplots(figsize=(12, 8))
    pp_plot_lc2st(
        probs=[scores["probs_data"]],
        probs_null=scores["probs_null"],
        conf_alpha=conf_alpha,
        labels=["classifier probabilities on observed data"],
        colors=["red"],
        ax=ax,
    )
    ax.set(title=obs_label)
    ax.legend()
    _save(fig, plot_dir, f"2_posterior_l-C2ST_qual_{obs_label}.png")


def run_coverage(
    flow,
    grid_preds,
    grid_cosmos,
    i_signal,
    flow_conf,
    params,
    obs_pred_dict=None,
    obs_label="DESy3",
    i_sobol=None,
    msfm_conf=None,
    i_noise=None,
):
    """Orchestrate the full posterior-coverage stage: GPU-batched sampling of held-out mock observations,
    then the coverage diagnostics, with plots saved under flow.model_dir/unblinding_plots.

    Mirrors deep_lss_paper/paper_2/pre-unblinding/2_posterior_coverage.ipynb so it runs automatically as
    part of run_inference.py --sample_posterior. i_sobol (row-aligned with grid_preds) and msfm_conf are
    required only when flow_conf enables diagnostics.prior_selection='wide'.
    """
    samples = sample_coverage_posteriors(
        flow, grid_preds, grid_cosmos, i_signal, flow_conf, i_sobol=i_sobol, msfm_conf=msfm_conf, i_noise=i_noise
    )

    plot_dir = os.path.join(flow.model_dir, "unblinding_plots")
    os.makedirs(plot_dir, exist_ok=True)

    tests = _test_flags(flow_conf)
    run_coverage_tests(samples, params, plot_dir, tests=tests)

    # l-C2ST is per-observation and needs the observation's summary; default to DES if present
    if tests["lc2st"]:
        if obs_pred_dict is not None and obs_label in obs_pred_dict:
            try:
                run_lc2st(samples, params, flow, obs_pred_dict[obs_label], obs_label, plot_dir)
            except ImportError:
                LOGGER.warning("sbi not available; skipping l-C2ST check.")
            except Exception as e:
                LOGGER.warning(f"l-C2ST check failed ({type(e).__name__}: {e})")
        else:
            LOGGER.info(f"Observation '{obs_label}' not available; skipping l-C2ST check.")

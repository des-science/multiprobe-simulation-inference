import os
import numpy as np
import matplotlib.pyplot as plt
from trianglechain import TriangleChain

import torch
from sbi.diagnostics.misspecification import calc_misspecification_mmd

from msfm.utils import files, logger
from msi.utils import input_output, plotting
from msi.flow_conductor.likelihood_flow import LikelihoodFlow
from msi.flow_conductor import architecture


LOGGER = logger.get_logger(__file__)


class PosteriorPredictiveChecks:
    """
    Class for running posterior predictive checks (PPC) for LSS probes.

    This class handles loading data, setting up normalizing flows, and running various checks to validate
    the posterior distribution obtained from inference.
    """

    def __init__(
        self,
        conf,
        cosmo_params=["Om", "s8", "w0"],
        seed=111,
        # data loading
        wl_pred_file=None,
        gc_pred_file=None,
        wl_flow_dir=None,
        gc_flow_dir=None,
    ):
        """
        Initialize the PosteriorPredictiveChecks object.

        Args:
            conf: Path to the configuration file or dictionary.
            cosmo_params: List of cosmological parameters.
            seed: Random seed for reproducibility.
        """

        self.conf = files.load_config(conf)
        self.cosmo_params = cosmo_params
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.wl_pred_file = wl_pred_file
        self.gc_pred_file = gc_pred_file

        self.wl_flow_dir = wl_flow_dir
        self.gc_flow_dir = gc_flow_dir

        if self.wl_pred_file:
            LOGGER.info("Loading weak lensing data")
            self.s_wl_grid, self.theta_wl_grid, self.wl_obs_dict = input_output.load_network_preds_simple(
                self.wl_pred_file
            )

            self.wl_params = cosmo_params.copy()
            self.wl_params += self.conf["analysis"]["params"]["ia"]["nla"]
            if self.conf["analysis"]["modelling"]["lensing"]["extended_nla"]:
                self.wl_params += self.conf["analysis"]["params"]["ia"]["tatt"]
            self.wl_cosmo_idx = [self.wl_params.index(p) for p in cosmo_params]

        if self.gc_pred_file:
            LOGGER.info("Loading galaxy clustering data")
            self.s_gc_grid, self.theta_gc_grid, self.gc_obs_dict = input_output.load_network_preds_simple(
                self.gc_pred_file
            )

            self.gc_params = cosmo_params.copy()
            self.gc_params += self.conf["analysis"]["params"]["bg"]["linear"]
            if self.conf["analysis"]["modelling"]["clustering"]["quadratic_biasing"]:
                self.gc_params += self.conf["analysis"]["params"]["bg"]["quadratic"]
            self.gc_cosmo_idx = [self.gc_params.index(p) for p in cosmo_params]

    def setup_flow(
        self, rep_probe, obs_probe, independent_cross=False, train_flow=False, flow_label="", fit_kwargs={}
    ):
        """
        Set up the normalizing flow for the posterior predictive checks.

        Args:
            rep_probe (str): The probe to be replicated (predicted), either 'lensing' or 'clustering'.
            obs_probe (str): The probe used for observation (conditioning), either 'lensing' or 'clustering'.
            independent_cross (bool): If True, treats cross-probe correlations as independent.
            train_flow (bool): If True, trains the flow from scratch.
            flow_label (str): Label for the flow model.
            fit_kwargs (dict): Additional keyword arguments for fitting the flow.
        """

        assert rep_probe in ["lensing", "clustering"]
        assert obs_probe in ["lensing", "clustering"]

        self.rep_probe = rep_probe
        self.obs_probe = obs_probe
        self.is_cross_probe = self.obs_probe != rep_probe
        self.independent_cross = independent_cross

        self.rep_abbrv = "wl" if rep_probe == "lensing" else "gc"
        self.obs_abbrv = "wl" if obs_probe == "lensing" else "gc"

        LOGGER.info(f"Conditioning on {self.obs_probe} and sampling in {self.rep_probe} summary space")

        if self.is_cross_probe:
            if self.rep_probe == "lensing":
                flow_dir = self.gc_flow_dir

                features_grid = self.s_wl_grid
                if independent_cross:
                    self.flow_dist = "p(s_wl | theta_cosmo)"
                    # only shared cosmo params: WL is insensitive to GC bias parameters
                    context_grid = self.theta_gc_grid[:, self.gc_cosmo_idx]
                else:
                    self.flow_dist = "p(s_wl | theta_gc, s_gc)"
                    context_grid = np.concatenate([self.theta_gc_grid, self.s_gc_grid], axis=-1)

            elif self.rep_probe == "clustering":
                flow_dir = self.wl_flow_dir

                features_grid = self.s_gc_grid
                if independent_cross:
                    self.flow_dist = "p(s_gc | theta_cosmo)"
                    # only shared cosmo params: GC is insensitive to WL IA parameters
                    context_grid = self.theta_wl_grid[:, self.wl_cosmo_idx]
                else:
                    self.flow_dist = "p(s_gc | theta_wl, s_wl)"
                    context_grid = np.concatenate([self.theta_wl_grid, self.s_wl_grid], axis=-1)

        else:
            if self.rep_probe == "lensing":
                self.flow_dist = "p(s_wl | theta_wl)"
                flow_dir = self.wl_flow_dir
                features_grid = self.s_wl_grid
                context_grid = self.theta_wl_grid

            elif self.rep_probe == "clustering":
                self.flow_dist = "p(s_gc | theta_gc)"
                flow_dir = self.gc_flow_dir
                features_grid = self.s_gc_grid
                context_grid = self.theta_gc_grid

        LOGGER.info(f"flow = {self.flow_dist}")
        self.context_grid = context_grid

        if self.is_cross_probe:
            flow_label += "ppc/cross"
            flow_label += f"_{self.rep_abbrv}_given_{self.obs_abbrv}"
            flow_label += independent_cross * "_independent"
        else:
            flow_label += "ppc/auto_"
            flow_label += self.obs_abbrv

        self.flow = LikelihoodFlow(
            params=[],
            conf=self.conf,
            embedding_net=architecture.get_context_embedding_net(context_grid.shape[-1]),
            base_dist=architecture.get_normal_dist(features_grid.shape[-1]),
            transform=architecture.get_sigmoids_transform(features_grid.shape[-1]),
            out_dir=flow_dir,
            label=flow_label,
            load_existing=not train_flow,
        )
        self.out_dir = self.flow.model_dir

        if train_flow:
            self.flow.fit(
                x=features_grid,
                theta=context_grid,
                batch_size=10_000,
                scheduler_type="cosine",
                save_model=True,
                **fit_kwargs,
            )

    def run_checks(
        self,
        # define observation
        obs_label=None,
        s_obs=None,
        theta_post=None,
        s_obs_rep=None,
        theta_post_rep=None,
        # samples
        n_samples_neural=100_000,
        n_samples_grid=1_000,
        k_highest_grid=None,
        # select checks
        plot_param_posterior=False,
        check_data_marginals=True,
        check_mmd=True,
        check_log_probs=True,
    ):
        """
        Run the requested posterior predictive checks.

        Args:
            obs_label (str): Label for the observation.
            s_obs (np.ndarray): Observed summary statistics.
            theta_post (np.ndarray): Posterior samples of parameters.
            s_obs_rep (np.ndarray): Observed summary statistics for the replicated probe.
            theta_post_rep (np.ndarray): Posterior samples for the replicated probe.
            n_samples_neural (int): Number of samples to draw from the neural posterior predictive.
            n_samples_grid (int): Number of samples to draw from the grid posterior predictive (importance sampling).
            k_highest_grid (int): Number of highest probability samples to select from the grid.
            plot_param_posterior (bool): Whether to plot the parameter posterior.
            check_data_marginals (bool): Whether to check data marginals.
            check_mmd (bool): Whether to check Maximum Mean Discrepancy (MMD).
            check_log_probs (bool): Whether to check log probabilities.
        """

        self._set_observation(obs_label, s_obs, theta_post, s_obs_rep, theta_post_rep)

        if plot_param_posterior:
            self._plot_param_posterior()

        self._sample_neural_posterior_predictive(n_samples=n_samples_neural)
        if not self.is_cross_probe:
            self._sample_grid_posterior_predictive(n_importance_samples=n_samples_grid, k_highest=k_highest_grid)

        if check_data_marginals:
            self._check_data_marginals()

        if check_mmd:
            self._check_mmd()

        if check_log_probs:
            self._check_log_probs()

    def _set_observation(self, obs_label=None, s_obs=None, theta_post=None, s_obs_rep=None, theta_post_rep=None):
        """Set up the observation data and configuration for the PPC."""

        self.obs_label = obs_label

        if self.obs_probe == "lensing":
            self.post_dist = "p(theta_wl | s_wl)"

            obs_model_dir = self.wl_flow_dir
            obs_dict = self.wl_obs_dict

            if self.is_cross_probe:
                self.s_prior = self.s_gc_grid
                rep_flow_dir = self.gc_flow_dir
                rep_obs_dict = self.gc_obs_dict
            else:
                self.s_prior = self.s_wl_grid

        elif self.obs_probe == "clustering":
            self.post_dist = "p(theta_gc | s_gc)"

            obs_model_dir = self.gc_flow_dir
            obs_dict = self.gc_obs_dict

            if self.is_cross_probe:
                self.s_prior = self.s_wl_grid
                rep_flow_dir = self.wl_flow_dir
                rep_obs_dict = self.wl_obs_dict
            else:
                self.s_prior = self.s_gc_grid

        LOGGER.info(f"post = {self.post_dist}")

        # obs_probe
        if s_obs is None:
            s_obs = obs_dict[obs_label]
        self.s_obs = s_obs

        if theta_post is None:
            theta_post = np.load(os.path.join(obs_model_dir, f"chain_{obs_label}.npy"))
        self.theta_post = theta_post

        # rep_probe
        if self.is_cross_probe:
            if s_obs_rep is None:
                s_obs_rep = rep_obs_dict[obs_label]

            if theta_post_rep is None:
                theta_post_rep = np.load(os.path.join(rep_flow_dir, f"chain_{obs_label}.npy"))
        else:
            s_obs_rep = s_obs
            theta_post_rep = theta_post

        self.s_obs_rep = s_obs_rep
        self.theta_post_rep = theta_post_rep

    def _plot_param_posterior(self):
        """Plot the parameter posteriors for the observation and replicated probe."""

        chains = [self.theta_post]
        labels = [self.obs_probe]
        if self.obs_probe == "lensing":
            params = [self.wl_params]
        elif self.obs_probe == "clustering":
            params = [self.gc_params]

        if self.is_cross_probe:
            chains.append(self.theta_post_rep)
            labels.append(self.rep_probe)
            if self.rep_probe == "lensing":
                params.append(self.wl_params)
            elif self.rep_probe == "clustering":
                params.append(self.gc_params)

        plotting.plot_chains(
            chains=chains,
            params=params,
            conf=self.conf,
            plot_labels=labels,
            obs_cosmo=None,
            out_dir=self.out_dir,
            file_label=self.obs_label,
        )

    def _sample_neural_posterior_predictive(self, n_samples=100_000):
        """Sample from the neural posterior predictive distribution."""

        # subsample the posterior
        i_star = self.rng.integers(0, self.theta_post.shape[0], n_samples)
        theta_star = self.theta_post[i_star]

        # sample the flow
        if self.is_cross_probe and not self.independent_cross:
            s_obs_star = np.repeat(np.atleast_2d(self.s_obs), n_samples, axis=0)
            context_star = np.concatenate([theta_star, s_obs_star], axis=-1)
        elif self.is_cross_probe and self.independent_cross:
            # marginalise over probe-specific nuisances by using only the shared cosmo columns
            obs_cosmo_idx = self.wl_cosmo_idx if self.obs_probe == "lensing" else self.gc_cosmo_idx
            context_star = theta_star[:, obs_cosmo_idx]
        else:
            context_star = theta_star

        LOGGER.info(f"Generating {n_samples} neural samples of {self.flow_dist} flow")
        LOGGER.timer.start("sampling")
        s_rep = self.flow.sample_likelihood(
            context_star,
            n_samples=1,
            batch_size=context_star.shape[0] // 10,
        )
        s_rep = np.squeeze(s_rep)
        LOGGER.info(f"Done sampling after {LOGGER.timer.elapsed('sampling')}")

        self.context_star = context_star
        self.s_rep = s_rep

    def _sample_grid_posterior_predictive(self, n_importance_samples=None, k_highest=None):
        """Sample from the grid posterior predictive using importance sampling or top-k selection."""
        # TODO for the cross-probe check, this is currently wrong: https://gemini.google.com/share/1e7a829ec98b
        # The weights should be proportional to p(theta|s_obs) ~ p(s_obs|theta) and not p(s_rep|theta, s_obs).
        # For the single probe, it doesn't make a difference as s_rep = s_obs.
        assert not self.is_cross_probe, "Grid PPC not implemented for cross-probe checks yet."

        log_probs_grid = self.flow.log_likelihood(
            np.repeat(np.atleast_2d(self.s_obs_rep), self.context_grid.shape[0], axis=0),
            self.context_grid,
        )
        log_probs_grid = log_probs_grid.cpu().numpy()
        log_probs_grid -= np.max(log_probs_grid)
        probs = np.exp(log_probs_grid)
        probs = probs / np.sum(probs)

        # effective sample size
        ess = 1 / np.sum(probs**2)
        LOGGER.info(f"Effective Sample Size (ESS) = {ess:.1f} out of {self.context_grid.shape[0]}")

        if (n_importance_samples is not None) and (k_highest is None):
            n_importance_samples = max(n_importance_samples, int(ess))
            LOGGER.info(f"Drawing {n_importance_samples} samples from the grid with importance weights")
            i_is = self.rng.choice(self.context_grid.shape[0], size=n_importance_samples, replace=True, p=probs)
            s_rep = self.s_prior[i_is]
            s_rep_unique = np.unique(s_rep, axis=0)
            LOGGER.info(f"Obtained {s_rep_unique.shape[0]} unique samples out of {n_importance_samples} samples")

        elif (k_highest is not None) and (n_importance_samples is None):
            LOGGER.info(f"Selecting the {k_highest} highest probability samples from the grid")
            i_sorted = np.argsort(probs)[-k_highest:]
            s_rep = self.s_prior[i_sorted]

        else:
            raise ValueError("Either n_importance_samples or k_highest must be specified, but not both")

        self.s_rep_grid = s_rep

    def _check_data_marginals(self, n_scatter=1_000):
        """Check and plot the marginal distributions of the data."""

        prior_label = r"$p(s_{" + self.rep_abbrv + r"})$"
        post_label = r"$p(s_{" + self.rep_abbrv + r"}|s_{" + self.obs_abbrv + r"}^{obs})$"
        post_label_sim = r"$p(s_{" + self.rep_abbrv + r"}|s_{" + self.obs_abbrv + r"}^{obs})$ (sims)"
        obs_label = r"$s_{" + self.rep_abbrv + r"}^{obs}$"

        tri = TriangleChain(
            show_legend=True,
            legend_fontsize=48,
            size=2,
            line_kwargs={"zorder": 0, "linewidths": 2},
            hist_kwargs={"zorder": 0, "lw": 2},
            scatter_kwargs={"s": 1, "marker": "o"},
        )

        def contour_or_scatter(tri, data, color, label):
            if data.shape[0] > n_scatter:
                tri.contour_cl(data, color=color, label=label)
            else:
                tri.scatter(data, color=color, label=label)

        contour_or_scatter(tri, self.s_prior, color="tab:blue", label=prior_label)
        contour_or_scatter(tri, self.s_rep, color="tab:orange", label=post_label)
        if not self.is_cross_probe:
            contour_or_scatter(tri, self.s_rep_grid, color="tab:green", label=post_label_sim)

        tri.scatter(
            np.atleast_2d(self.s_obs_rep),
            scatter_kwargs={"s": 200, "marker": "*", "zorder": 10},
            color="k",
            scatter_vline_1D=True,
            plot_histograms_1D=False,
            label=obs_label,
        )

        # only keep the last legend
        try:
            for legend in tri.fig.legends[:-1]:
                legend.remove()
        except AttributeError:
            pass

        tri.fig.suptitle(self.obs_label, fontsize=24, y=0.9)

        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_data_marginals.png")
        LOGGER.info(f"Saving data marginals plot to {plot_file}")
        tri.fig.savefig(plot_file, bbox_inches="tight", dpi=100)

    def _check_mmd(self):
        """Perform the Maximum Mean Discrepancy (MMD) check."""

        mmd_label = r"MMD($s_{" + self.rep_abbrv + r"}^{rep}$, $s_{" + self.rep_abbrv + r"}^{obs}$)"

        s_obs_rep = np.atleast_2d(self.s_obs_rep)

        p_val, (mmds_baseline, mmd) = calc_misspecification_mmd(
            x_obs=torch.from_numpy(s_obs_rep).float(),
            x=torch.from_numpy(self.s_rep).float(),
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(mmds_baseline.numpy(), bins=100, alpha=0.5, label="baseline")
        ax.axvline(mmd.item(), color="k", label=mmd_label)

        ax.set(
            xlabel="MMD", ylabel="Count", title=f"{self.obs_label}: Maximum Mean Discrepancy Check: p = {p_val:.4f}"
        )
        ax.legend()

        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_mmd_check.png")
        LOGGER.info(f"Saving MMD check plot to {plot_file}")
        fig.savefig(plot_file, bbox_inches="tight", dpi=100)

    def _check_log_probs(self):
        """Check the log probabilities of the observed and replicated summaries."""

        rep_label = r"$p(s_{" + self.rep_abbrv + r"}^{rep} | \theta^{\star}, s_{" + self.obs_abbrv + r"}^{obs})$"
        obs_label = r"$p(s_{" + self.rep_abbrv + r"}^{obs} | \theta^{\star}, s_{" + self.obs_abbrv + r"}^{obs})$"

        log_prob_rep = self.flow.log_likelihood(self.s_rep, self.context_star, return_numpy=True)
        log_prob_obs = self.flow.log_likelihood(
            np.repeat(np.atleast_2d(self.s_obs_rep), self.context_star.shape[0], axis=0),
            self.context_star,
            return_numpy=True,
        )
        p_val = np.mean(log_prob_rep < log_prob_obs)

        fig, ax = plt.subplots(figsize=(12, 6), ncols=2, sharex=True)

        ax[0].hist(log_prob_rep, bins=100, label=rep_label, alpha=0.5)
        ax[0].hist(log_prob_obs, bins=100, label=obs_label, alpha=0.5)
        ax[0].set(xlabel="log prob", ylabel="Count")
        ax[0].legend()

        ax[1].scatter(log_prob_obs, log_prob_rep, s=0.1, alpha=0.5)
        log_min = np.min([log_prob_obs, log_prob_rep])
        log_max = np.max([log_prob_obs, log_prob_rep])
        ax[1].plot([log_min, log_max], [log_min, log_max], color="k", linestyle="--")
        ax[1].set(xlabel="log prob obs", ylabel="log prob rep")
        ax[1].set_aspect("equal")

        fig.suptitle(f"{self.obs_label}: Log Probability Check: p = {p_val:.4f}")

        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_log_prob_check.png")
        LOGGER.info(f"Saving log probability check plot to {plot_file}")
        fig.savefig(plot_file, bbox_inches="tight", dpi=100)

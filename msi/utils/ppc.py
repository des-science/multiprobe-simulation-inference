import os
import numpy as np
import matplotlib.pyplot as plt
from trianglechain import TriangleChain

from msfm.utils import files, logger
from msi.utils import input_output, plotting
from msi.flow_conductor.likelihood_flow import LikelihoodFlow
from msi.flow_conductor import architecture


LOGGER = logger.get_logger(__file__)

_PROBE_ABBREVIATIONS = {
    "lensing": "wl",
    "clustering": "gc",
    "cross": "x",
    "combined": "wl+gc",
}


class PosteriorPredictiveChecks:
    """
    Class for running posterior predictive checks (PPC) for LSS probes.

    This class handles loading data, setting up normalizing flows, and running various checks to validate
    the posterior distribution obtained from inference.

    Probes are referred to generically as 'probe1' and 'probe2' (e.g. weak lensing and galaxy clustering).
    """

    def __init__(
        self,
        conf,
        cosmo_params=["Om", "s8", "w0"],
        seed=111,
        # probe names
        probe1_name=None,
        probe2_name=None,
        # data loading
        probe1_pred_file=None,
        probe2_pred_file=None,
        probe1_flow_dir=None,
        probe2_flow_dir=None,
    ):
        """
        Initialize the PosteriorPredictiveChecks object.

        Args:
            conf: Path to the configuration file or dictionary.
            cosmo_params: List of cosmological parameters.
            seed: Random seed for reproducibility.
            probe1_name: Name of probe 1. One of 'lensing', 'clustering', 'cross', 'combined'.
                Used for plot labels; shorthand abbreviations are taken from _PROBE_ABBREVIATIONS.
            probe2_name: Name of probe 2. Same options as probe1_name.
            probe1_pred_file: Path to the probe 1 predictions file.
            probe2_pred_file: Path to the probe 2 predictions file.
            probe1_flow_dir: Directory for the probe 1 flow model.
            probe2_flow_dir: Directory for the probe 2 flow model.
        """

        self.conf = files.load_config(conf)
        self.cosmo_params = cosmo_params
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.probe1_name = probe1_name
        self.probe2_name = probe2_name
        self.probe1_abbrv = _PROBE_ABBREVIATIONS[probe1_name] if probe1_name else None
        self.probe2_abbrv = _PROBE_ABBREVIATIONS[probe2_name] if probe2_name else None

        self.probe1_pred_file = probe1_pred_file
        self.probe2_pred_file = probe2_pred_file

        self.probe1_flow_dir = probe1_flow_dir
        self.probe2_flow_dir = probe2_flow_dir

        if self.probe1_pred_file:
            LOGGER.info(f"Loading {probe1_name} data")
            self.s_probe1_grid, self.theta_probe1_grid, self.probe1_obs_dict = input_output.load_network_preds_simple(
                self.probe1_pred_file
            )
            self.probe1_params = self._get_probe_params(probe1_name)
            self.probe1_cosmo_idx = [self.probe1_params.index(p) for p in cosmo_params]

        if self.probe2_pred_file:
            LOGGER.info(f"Loading {probe2_name} data")
            self.s_probe2_grid, self.theta_probe2_grid, self.probe2_obs_dict = input_output.load_network_preds_simple(
                self.probe2_pred_file
            )
            self.probe2_params = self._get_probe_params(probe2_name)
            self.probe2_cosmo_idx = [self.probe2_params.index(p) for p in cosmo_params]

    def _get_probe_params(self, probe_name):
        """Return the full parameter list for a probe: cosmo params + probe-specific nuisances."""
        params = self.cosmo_params.copy()
        if probe_name in ("lensing", "combined", "cross"):
            params += self.conf["analysis"]["params"]["ia"]["nla"]
            if self.conf["analysis"]["modelling"]["lensing"]["extended_nla"]:
                params += self.conf["analysis"]["params"]["ia"]["tatt"]
        if probe_name in ("clustering", "combined", "cross"):
            params += self.conf["analysis"]["params"]["bg"]["linear"]
            if self.conf["analysis"]["modelling"]["clustering"]["quadratic_biasing"]:
                params += self.conf["analysis"]["params"]["bg"]["quadratic"]

        LOGGER.info(f"Probe '{probe_name}' parameters: {params}")
        return params

    def setup_flow(
        self, rep_probe, obs_probe, independent_cross=False, train_flow=False, flow_label="", fit_kwargs={}
    ):
        """
        Set up the normalizing flow for the posterior predictive checks.

        Args:
            rep_probe (str): The probe to be replicated (predicted). Must be one of the names
                passed as probe1_name or probe2_name at construction time.
            obs_probe (str): The probe used for observation (conditioning). Same options.
            independent_cross (bool): If True, treats cross-probe correlations as independent.
            train_flow (bool): If True, trains the flow from scratch.
            flow_label (str): Label for the flow model.
            fit_kwargs (dict): Additional keyword arguments for fitting the flow.
        """

        assert rep_probe in [
            self.probe1_name,
            self.probe2_name,
        ], f"rep_probe must be one of {[self.probe1_name, self.probe2_name]}, got '{rep_probe}'"
        assert obs_probe in [
            self.probe1_name,
            self.probe2_name,
        ], f"obs_probe must be one of {[self.probe1_name, self.probe2_name]}, got '{obs_probe}'"

        self.rep_probe = "probe1" if rep_probe == self.probe1_name else "probe2"
        self.obs_probe = "probe1" if obs_probe == self.probe1_name else "probe2"

        self.is_cross_probe = self.obs_probe != self.rep_probe
        self.independent_cross = independent_cross

        self.rep_abbrv = self.probe1_abbrv if self.rep_probe == "probe1" else self.probe2_abbrv
        self.obs_abbrv = self.probe1_abbrv if self.obs_probe == "probe1" else self.probe2_abbrv
        self.rep_probe_name = self.probe1_name if self.rep_probe == "probe1" else self.probe2_name
        self.obs_probe_name = self.probe1_name if self.obs_probe == "probe1" else self.probe2_name

        LOGGER.info(f"Conditioning on {self.obs_probe_name} and sampling in {self.rep_probe_name} summary space")

        if self.is_cross_probe:
            flow_dir = getattr(self, f"{self.obs_probe}_flow_dir")
            features_grid = getattr(self, f"s_{self.rep_probe}_grid")
            if independent_cross:
                self.flow_dist = f"p(s_{self.rep_abbrv} | theta_cosmo)"
                # only shared cosmo params: rep probe is insensitive to obs probe nuisance parameters
                context_grid = getattr(self, f"theta_{self.obs_probe}_grid")[
                    :, getattr(self, f"{self.obs_probe}_cosmo_idx")
                ]
            else:
                self.flow_dist = f"p(s_{self.rep_abbrv} | theta_{self.obs_abbrv}, s_{self.obs_abbrv})"
                context_grid = np.concatenate(
                    [getattr(self, f"theta_{self.obs_probe}_grid"), getattr(self, f"s_{self.obs_probe}_grid")], axis=-1
                )
        else:
            self.flow_dist = f"p(s_{self.rep_abbrv} | theta_{self.rep_abbrv})"
            flow_dir = getattr(self, f"{self.rep_probe}_flow_dir")
            features_grid = getattr(self, f"s_{self.rep_probe}_grid")
            context_grid = getattr(self, f"theta_{self.rep_probe}_grid")

        LOGGER.info(f"flow = {self.flow_dist}")
        self.context_grid = context_grid

        if self.is_cross_probe:
            flow_label += "ppc/cross"
            flow_label += f"_{self.rep_abbrv}_given_{self.obs_abbrv}"
            flow_label += "_independent" if independent_cross else ""
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
        check_kernel=True,
        check_log_prob=True,
        check_mahalanobis=True,
        check_l2=True,
        check_l1=True,
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
            check_kernel (bool): Whether to run the kernel similarity outlier test.
            check_log_prob (bool): Whether to run the log-probability posterior predictive check.
            check_mahalanobis (bool): Whether to check the Mahalanobis distance.
            check_l2 (bool): Whether to check the mean L2 distance to the PPD.
            check_l1 (bool): Whether to check the mean L1 distance to the PPD.
        """

        self._set_observation(obs_label, s_obs, theta_post, s_obs_rep, theta_post_rep)

        if plot_param_posterior:
            self._plot_param_posterior()

        self._sample_neural_posterior_predictive(n_samples=n_samples_neural)
        if not self.is_cross_probe:
            self._sample_grid_posterior_predictive(n_importance_samples=n_samples_grid, k_highest=k_highest_grid)

        if check_data_marginals:
            self._check_data_marginals()

        if check_log_prob:
            self._check_one_sample(stat="log_prob")

        if check_kernel:
            self._check_one_sample(stat="kernel")

        if check_mahalanobis:
            self._check_one_sample(stat="mahalanobis")

        if check_l2:
            self._check_one_sample(stat="l2")

        if check_l1:
            self._check_one_sample(stat="l1")

    def _set_observation(self, obs_label=None, s_obs=None, theta_post=None, s_obs_rep=None, theta_post_rep=None):
        """Set up the observation data and configuration for the PPC."""

        self.obs_label = obs_label

        self.post_dist = f"p(theta_{self.obs_abbrv} | s_{self.obs_abbrv})"
        LOGGER.info(f"post = {self.post_dist}")

        obs_flow_dir = getattr(self, f"{self.obs_probe}_flow_dir")
        obs_dict = getattr(self, f"{self.obs_probe}_obs_dict")

        if self.is_cross_probe:
            self.s_prior = getattr(self, f"s_{self.rep_probe}_grid")
            rep_flow_dir = getattr(self, f"{self.rep_probe}_flow_dir")
            rep_obs_dict = getattr(self, f"{self.rep_probe}_obs_dict")
        else:
            self.s_prior = getattr(self, f"s_{self.obs_probe}_grid")

        # obs_probe
        if s_obs is None:
            s_obs = obs_dict[obs_label]
        self.s_obs = s_obs

        if theta_post is None:
            theta_post = np.load(os.path.join(obs_flow_dir, f"chain_{obs_label}.npy"))
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
        # only for plotting the parameter posterior
        self.theta_post_rep = theta_post_rep

    def _plot_param_posterior(self):
        """Plot the parameter posteriors for the observation and replicated probe."""

        chains = [self.theta_post]
        labels = [self.obs_probe_name]
        params = [getattr(self, f"{self.obs_probe}_params")]

        if self.is_cross_probe:
            chains.append(self.theta_post_rep)
            labels.append(self.rep_probe_name)
            params.append(getattr(self, f"{self.rep_probe}_params"))

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
            obs_cosmo_idx = getattr(self, f"{self.obs_probe}_cosmo_idx")
            context_star = theta_star[:, obs_cosmo_idx]
        else:
            context_star = theta_star

        LOGGER.info(f"Generating {n_samples} neural samples of {self.flow_dist} flow")
        LOGGER.timer.start("sampling")
        s_rep = self.flow.sample_likelihood(
            context_star,
            n_samples=1,
            batch_size=min(context_star.shape[0], 10_000),
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

    def _check_one_sample(self, stat, n_bootstrap=10_000, n_kernel_ref=5_000):
        """Generic one-sample test: is s_obs an outlier relative to the PPD?

        Null distribution: evaluate the same statistic on bootstrap draws from
        the PPD samples s_rep.  A small p-value means s_obs is extreme.

        Args:
            stat: 'mahalanobis', 'l1', 'l2', 'log_prob', or 'kernel'.
            n_bootstrap: Number of bootstrap draws for the null.
            n_kernel_ref: Size of the reference subsample (kernel only).
        """
        s_rep = self.s_rep  # (N, dim)
        s_obs = np.atleast_2d(self.s_obs_rep)  # (1, dim)

        if stat == "mahalanobis":
            mu = np.mean(s_rep, axis=0)
            cov = np.cov(s_rep, rowvar=False)
            cov_inv = np.linalg.pinv(cov)

            def compute_stat(x):
                diff = x - mu  # (M, dim)
                return np.einsum("...i,ij,...j->...", diff, cov_inv, diff)  # (M,)

            xlabel = "Mahalanobis distance²"
            file_tag = "mahalanobis_check"
            title_tag = "Mahalanobis Distance Check"
            stat_label = r"$D_M^2(s_{" + self.rep_abbrv + r"}^{obs})$"
            outlier_if_low = False

        elif stat in ("l1", "l2"):
            from scipy.spatial.distance import cdist

            cdist_metric = "cityblock" if stat == "l1" else "euclidean"
            norm_ord = 1 if stat == "l1" else 2

            def compute_stat(x):
                return np.mean(cdist(x, s_rep, metric=cdist_metric), axis=-1)  # (M,)

            xlabel = f"Mean L{norm_ord} distance to PPD"
            file_tag = f"{stat}_check"
            title_tag = f"L{norm_ord} Distance Check"
            stat_label = (
                r"$\bar{d}_{L"
                + str(norm_ord)
                + r"}(s_{"
                + self.rep_abbrv
                + r"}^{obs},\, s_{"
                + self.rep_abbrv
                + r"}^{rep})$"
            )
            outlier_if_low = False

        elif stat == "log_prob":

            def compute_stat(x, context):
                return self.flow.log_likelihood(x, context, return_numpy=True)

            file_tag = "log_prob_check"
            title_tag = "Log-Prob PPC"

        elif stat == "kernel":
            from scipy.spatial.distance import cdist

            n_ref = min(n_kernel_ref, s_rep.shape[0])
            i_ref = self.rng.integers(0, s_rep.shape[0], n_ref)
            s_ref = s_rep[i_ref]

            n_bw = min(2_000, n_ref)
            i_bw = self.rng.integers(0, n_ref, n_bw)
            sq_dists_bw = cdist(s_ref[i_bw], s_ref[i_bw], metric="sqeuclidean")
            bw2 = np.median(sq_dists_bw[np.triu_indices(n_bw, k=1)])
            if bw2 == 0:
                bw2 = 1.0
            LOGGER.info(f"Kernel bandwidth (squared): {bw2:.4f}")

            def compute_stat(x):
                sq_d = cdist(x, s_ref, metric="sqeuclidean")  # (M, n_ref)
                return np.mean(np.exp(-sq_d / bw2), axis=-1)  # high = similar = not outlier

            xlabel = "Mean kernel similarity"
            file_tag = "kernel_check"
            title_tag = "Kernel Similarity Check"
            stat_label = r"$\bar{k}(s_{" + self.rep_abbrv + r"}^{obs},\, s_{" + self.rep_abbrv + r"}^{rep})$"
            outlier_if_low = True

        else:
            raise ValueError(f"Unknown stat: {stat}")

        i_boot = self.rng.integers(0, s_rep.shape[0], n_bootstrap)

        if stat == "log_prob":
            # for each posterior draw theta_i, compare log p(s_obs | theta_i) with log p(s_rep_i | theta_i).
            # The p-value is the fraction of draws where the replicated data is more likely than the observed data
            # (element-wise).
            t_obs_array = compute_stat(np.repeat(s_obs, n_bootstrap, axis=0), self.context_star[i_boot])
            t_boot = compute_stat(s_rep[i_boot], self.context_star[i_boot])
            t_diff = t_boot - t_obs_array  # positive: rep more likely than obs
            p_val = np.mean(t_diff <= 0)  # fraction where obs is at least as likely as rep
        else:
            t_obs = compute_stat(s_obs)[0]
            t_boot = compute_stat(s_rep[i_boot])
            # p-value: fraction of null draws at least as extreme as t_obs
            p_val = np.mean(t_boot <= t_obs) if outlier_if_low else np.mean(t_boot >= t_obs)

        fig, ax = plt.subplots(figsize=(12, 6))
        if stat == "log_prob":
            # plot per-draw differences: rep vs obs log-prob. p-value = fraction left of 0.
            ax.hist(t_diff, bins=100, alpha=0.5, label=r"$\log p(s^{rep}|\theta_i) - \log p(s^{obs}|\theta_i)$")
            ax.axvline(0, color="k", linestyle="--", label=f"p = {p_val:.4f}")
            ax.set(
                xlabel=r"$\log p(s^{rep}|\theta) - \log p(s^{obs}|\theta)$",
                ylabel="Count",
                title=f"{self.obs_label}: {title_tag}: p = {p_val:.4f}",
            )
        else:
            ax.hist(t_boot, bins=100, alpha=0.5, label="null (PPD samples)")
            ax.axvline(t_obs, color="k", label=f"{stat_label} = {t_obs:.4f}")
            ax.set(xlabel=xlabel, ylabel="Count", title=f"{self.obs_label}: {title_tag}: p = {p_val:.4f}")
        ax.legend()

        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_{file_tag}.png")
        LOGGER.info(f"Saving {title_tag} plot to {plot_file}")
        fig.savefig(plot_file, bbox_inches="tight", dpi=100)

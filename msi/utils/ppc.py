import os
import numpy as np
import matplotlib.pyplot as plt
from trianglechain import TriangleChain

from msfm.utils import files, logger
from msi.utils import input_output, plotting, tensions
from msi.utils import flow as flow_utils
from msi.flow_conductor.likelihood_flow import LikelihoodFlow, LikelihoodFlowEnsemble

LOGGER = logger.get_logger(__file__)

_PROBE_ABBREVIATIONS = {"lensing": "wl", "clustering": "gc", "cross": "x", "combined": "wl+gc"}


def _join(*parts):
    """Join non-empty parts with a comma — used to build LaTeX subscripts like 'wl,maps'."""
    return ",".join(p for p in parts if p)


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
        # data type for each run (free-form, e.g. "maps" or "cls"). Used for plot annotations
        # and to disambiguate runs that share the same probe_name.
        probe1_data=None,
        probe2_data=None,
        # unique labels for each run — auto-derived from probe_name + data; pass explicitly only
        # when overriding the default, must differ between probe1 and probe2.
        probe1_label=None,
        probe2_label=None,
        # data loading
        probe1_pred_file=None,
        probe2_pred_file=None,
        probe1_flow_dir=None,
        probe2_flow_dir=None,
        shared_data=False,
        # flow architecture (shared by every setup_flow call on this instance)
        flow_conf=None,
        n_flows=1,
    ):
        """
        Initialize the PosteriorPredictiveChecks object.

        Args:
            conf: Path to the configuration file or dictionary.
            cosmo_params: List of cosmological parameters.
            seed: Random seed for reproducibility.
            probe1_name: Physical probe type for probe 1. One of 'lensing', 'clustering', 'cross',
                'combined'. Determines nuisance parameters and abbreviation.
            probe2_name: Physical probe type for probe 2. Same options as probe1_name.
            probe1_data: Summary-statistic / data-vector type for probe 1 (free-form string,
                typically 'maps' or 'cls'). Optional metadata used to (a) auto-derive a unique
                ``probe1_label`` when both probes share the same ``probe_name``, and (b) enrich
                plot titles, legends, and log lines so the maps-vs-Cls vs lensing-vs-clustering
                distinction is visible end-to-end. Pass ``None`` to fall back to probe-only labels.
            probe2_data: Same as ``probe1_data`` for probe 2.
            probe1_label: Unique identifier for probe 1 used in setup_flow. Defaults to
                ``f"{probe1_name}_{probe1_data}"`` when ``probe1_data`` is set, otherwise to
                ``probe1_name``. Pass explicitly only to override.
            probe2_label: Same as ``probe1_label`` for probe 2.
            probe1_pred_file: Path to the probe 1 predictions file.
            probe2_pred_file: Path to the probe 2 predictions file.
            probe1_flow_dir: Directory for the probe 1 flow model.
            probe2_flow_dir: Directory for the probe 2 flow model.
            shared_data: True when probe1 and probe2 are different summary statistics on the SAME
                physical data (e.g. lensing maps vs lensing Cls). In that case the conditional-
                independence assumption underlying ``independent_cross=True`` does not hold (the
                two summaries share cosmic variance and noise); ``setup_flow`` will refuse
                ``independent_cross=True`` for cross-probe runs.
            flow_conf: Flow config dict (``context_embedding`` / ``transform`` / ``training``
                blocks, e.g. ``configs/flow/maf.yaml``) defining the PPC flow architecture and
                training, consumed by ``flow_utils.build_flow_architecture`` /
                ``_extract_train_kwargs`` -- the same builders ``run_inference`` uses. ``None`` ->
                ``{}`` reproduces the library defaults.
            n_flows: Number of flows in the PPC ensemble. 1 (default) builds a single
                ``LikelihoodFlow``; >1 builds a ``LikelihoodFlowEnsemble`` of independently
                initialized members with the same architecture.
        """

        self.conf = files.load_config(conf)
        self.cosmo_params = cosmo_params
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.flow_conf = flow_conf or {}
        self.n_flows = n_flows

        self.probe1_name = probe1_name
        self.probe2_name = probe2_name
        self.probe1_data = probe1_data
        self.probe2_data = probe2_data
        self.probe1_label = probe1_label or self._default_label(probe1_name, probe1_data)
        self.probe2_label = probe2_label or self._default_label(probe2_name, probe2_data)
        if probe1_name and probe2_name:
            assert self.probe1_label != self.probe2_label, (
                f"probe1_label and probe2_label collide ('{self.probe1_label}'). Set probe*_data "
                "to disambiguate (e.g. data='maps' vs data='cls') or pass probe*_label explicitly."
            )
        self.probe1_abbrv = _PROBE_ABBREVIATIONS[probe1_name] if probe1_name else None
        self.probe2_abbrv = _PROBE_ABBREVIATIONS[probe2_name] if probe2_name else None

        self.probe1_pred_file = probe1_pred_file
        self.probe2_pred_file = probe2_pred_file

        self.probe1_flow_dir = probe1_flow_dir
        self.probe2_flow_dir = probe2_flow_dir
        self.shared_data = shared_data

        if self.probe1_pred_file:
            LOGGER.info(f"Loading {probe1_name} data")
            self.s_probe1_grid, self.theta_probe1_grid, self.probe1_obs_dict, _ = (
                input_output.load_network_preds_simple(self.probe1_pred_file)
            )
            self.probe1_real_idx = self._load_realization_idx(self.probe1_pred_file)
            self.s_probe1_grid, self.theta_probe1_grid, self.probe1_real_idx = tensions.align_rows(
                self.s_probe1_grid, self.theta_probe1_grid, self.probe1_real_idx
            )
            self.probe1_params = self._get_probe_params(probe1_name)
            self.probe1_cosmo_idx = [self.probe1_params.index(p) for p in cosmo_params]

        if self.probe2_pred_file:
            LOGGER.info(f"Loading {probe2_name} data")
            self.s_probe2_grid, self.theta_probe2_grid, self.probe2_obs_dict, _ = (
                input_output.load_network_preds_simple(self.probe2_pred_file)
            )
            self.probe2_real_idx = self._load_realization_idx(self.probe2_pred_file)
            self.s_probe2_grid, self.theta_probe2_grid, self.probe2_real_idx = tensions.align_rows(
                self.s_probe2_grid, self.theta_probe2_grid, self.probe2_real_idx
            )
            self.probe2_params = self._get_probe_params(probe2_name)
            self.probe2_cosmo_idx = [self.probe2_params.index(p) for p in cosmo_params]

        if self.probe1_pred_file and self.probe2_pred_file:
            self._assert_aligned_grids()

    @staticmethod
    def _load_realization_idx(pred_file):
        """Per-row ``(i_sobol, i_signal, i_noise)`` realization indices for the grid test set.

        Read in the same row order as ``load_network_preds_simple``'s concatenated grid arrays
        (the stored ``(n_cosmo, n_examples)`` index grids flatten C-order, matching the
        ``concatenate`` of the ``(n_cosmo, n_examples, dim)`` predictions). Used to sort each
        probe's grid into a canonical realization order so the two probes can be paired by row.
        """
        import h5py

        with h5py.File(pred_file, "r") as f:
            return np.stack([f[f"grid/{k}/test"][:].reshape(-1) for k in ("i_sobol", "i_signal", "i_noise")], axis=1)

    def _assert_aligned_grids(self):
        """Verify probe1 and probe2 grids are the SAME realizations in the same row order.

        Both grids are sorted into canonical ``(i_sobol, i_signal, i_noise)`` order in
        ``__init__`` (via ``tensions.align_rows``). Cross-probe ``setup_flow`` uses the joint
        conditional ``p(s_rep | theta_obs, s_obs)`` (the Doux et al. 2020 mode) and pairs row
        ``i`` of one probe with row ``i`` of the other. That joint only encodes the cross-probe
        data-level correlation (cosmic variance, shared noise / footprint) if row ``i`` is the
        *same sky realization* in both probes. ``evaluate_grid`` only sorts by ``i_sobol``, so two
        separately-evaluated runs can differ in their within-cosmology signal/noise ordering --
        checking cosmology alone would pass on mismatched realizations and silently train the flow
        on independent draws (collapsing the correlation). See memory
        ``project_tension_row_alignment_bug``.
        """
        idx1, idx2 = self.probe1_real_idx, self.probe2_real_idx
        assert idx1.shape == idx2.shape, (
            f"Probe grids have different shapes ({idx1.shape} vs {idx2.shape}); the two pred "
            "files must come from the same simulation grid (same train/test split)."
        )
        assert np.array_equal(idx1, idx2), (
            "Probe grids are not the same realizations after alignment: the (i_sobol, i_signal, "
            "i_noise) index arrays differ row by row. Cross-probe checks pair the two grids by "
            "row, so both pred files must contain the same set of realizations. Re-export them "
            "from the same simulation grid."
        )
        # cosmology then agrees by construction; verify cheaply to catch cosmo_params mislabeling.
        c1 = np.asarray(self.theta_probe1_grid)[:, self.probe1_cosmo_idx]
        c2 = np.asarray(self.theta_probe2_grid)[:, self.probe2_cosmo_idx]
        assert np.allclose(c1, c2), (
            "Cosmology columns disagree despite aligned realization indices; check that "
            "cosmo_params map to the correct columns in each probe's parameter list."
        )

    @staticmethod
    def _default_label(probe_name, data):
        if not probe_name:
            return None
        return f"{probe_name}_{data}" if data else probe_name

    def _setup_descriptor(self):
        """Human-readable descriptor of the current obs→rep setup for plot titles / logs.

        Examples:
            "auto: lensing/maps"
            "cross: lensing/maps → clustering/maps  (joint)"
            "cross: lensing/maps → lensing/cls  (joint, shared_data)"
        """

        def _fmt(probe_name, data):
            return f"{probe_name}/{data}" if data else probe_name

        obs = _fmt(self.obs_probe_name, self.obs_data)
        rep = _fmt(self.rep_probe_name, self.rep_data)
        if not self.is_cross_probe:
            return f"auto: {obs}"
        kind = "indep" if self.independent_cross else "joint"
        if self.shared_data:
            kind += ", shared_data"
        return f"cross: {obs} → {rep}  ({kind})"

    def _summ_subs(self, side):
        """LaTeX subscript for the summary on the obs/rep side, including data type if set."""
        if side == "obs":
            return _join(self.obs_abbrv, self.obs_data)
        return _join(self.rep_abbrv, self.rep_data)

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

    def setup_flow(self, rep_probe, obs_probe, independent_cross=False, retrain=False, flow_label="", fit_kwargs={}):
        """
        Set up the normalizing flow for the posterior predictive checks.

        Args:
            rep_probe (str): The probe to be replicated (predicted). Must be one of the names
                passed as probe1_name or probe2_name at construction time.
            obs_probe (str): The probe used for observation (conditioning). Same options.
            independent_cross (bool): Controls the cross-probe flow:

                - ``False`` (default): train the joint conditional ``p(s_rep | theta_obs, s_obs)``.
                  This encodes the cross-probe data-level correlation (cosmic variance, shared
                  noise / footprint) and is the standard Doux et al. 2020 cross-probe consistency
                  formulation — respecting probe correlations is the whole point of the test.
                  Use this for cross-probe consistency checks and for same-data summary-stat
                  comparisons (``shared_data=True``).
                - ``True``: train the marginal ``p(s_rep | theta_cosmo)``, treating the two
                  probes as conditionally independent given cosmology. This is a simplified
                  diagnostic that ignores cross-probe correlations; only appropriate when the
                  two probes really are independent measurements (different sky areas, different
                  physics). Refused when ``shared_data=True`` since shared-data summaries clearly
                  share cosmic variance and noise.

            retrain (bool): Force training from scratch even when a checkpoint exists. Default False:
                the flow is recovered from its checkpoint if one exists, and only trained when none
                is found.
            flow_label (str): Label for the flow model.
            fit_kwargs (dict): Overrides for the training kwargs derived from ``self.flow_conf``
                (``_extract_train_kwargs``); merged on top, so e.g. ``{"n_epochs": 50}`` shortens
                training without changing the architecture.
        """

        assert rep_probe in [
            self.probe1_label,
            self.probe2_label,
        ], f"rep_probe must be one of {[self.probe1_label, self.probe2_label]}, got '{rep_probe}'"
        assert obs_probe in [
            self.probe1_label,
            self.probe2_label,
        ], f"obs_probe must be one of {[self.probe1_label, self.probe2_label]}, got '{obs_probe}'"

        self.rep_probe = "probe1" if rep_probe == self.probe1_label else "probe2"
        self.obs_probe = "probe1" if obs_probe == self.probe1_label else "probe2"

        self.is_cross_probe = self.obs_probe != self.rep_probe
        self.independent_cross = independent_cross

        assert not (self.is_cross_probe and independent_cross and self.shared_data), (
            "independent_cross=True assumes the two probes are conditionally independent given "
            "cosmology, which does not hold when shared_data=True (different summary statistics "
            "on the same physical data share cosmic variance and noise). Use independent_cross=False."
        )

        self.rep_abbrv = self.probe1_abbrv if self.rep_probe == "probe1" else self.probe2_abbrv
        self.obs_abbrv = self.probe1_abbrv if self.obs_probe == "probe1" else self.probe2_abbrv
        self.rep_probe_name = self.probe1_name if self.rep_probe == "probe1" else self.probe2_name
        self.obs_probe_name = self.probe1_name if self.obs_probe == "probe1" else self.probe2_name
        self.rep_data = self.probe1_data if self.rep_probe == "probe1" else self.probe2_data
        self.obs_data = self.probe1_data if self.obs_probe == "probe1" else self.probe2_data

        LOGGER.info(f"Setup: {self._setup_descriptor()}")

        # Bind role → attribute once; all private methods use these names directly.
        self._obs_flow_dir = getattr(self, f"{self.obs_probe}_flow_dir")
        self._rep_flow_dir = getattr(self, f"{self.rep_probe}_flow_dir")
        self._s_obs_grid = getattr(self, f"s_{self.obs_probe}_grid")
        self._s_rep_prior = getattr(self, f"s_{self.rep_probe}_grid")
        self._theta_obs = getattr(self, f"theta_{self.obs_probe}_grid")
        self._obs_cosmo_idx = getattr(self, f"{self.obs_probe}_cosmo_idx")
        self._obs_obs_dict = getattr(self, f"{self.obs_probe}_obs_dict")
        self._rep_obs_dict = getattr(self, f"{self.rep_probe}_obs_dict") if self.is_cross_probe else None
        self._obs_params = getattr(self, f"{self.obs_probe}_params")
        self._rep_params = getattr(self, f"{self.rep_probe}_params")
        self.s_prior = self._s_rep_prior

        rep_subs = self._summ_subs("rep")
        obs_subs = self._summ_subs("obs")

        if self.is_cross_probe:
            flow_dir = self._obs_flow_dir
            features_grid = self._s_rep_prior
            if independent_cross:
                self.flow_dist = f"p(s_{{{rep_subs}}} | theta_cosmo)"
                # only shared cosmo params: rep probe is insensitive to obs probe nuisance parameters
                context_grid = self._theta_obs[:, self._obs_cosmo_idx]
            else:
                self.flow_dist = f"p(s_{{{rep_subs}}} | theta_{self.obs_abbrv}, s_{{{obs_subs}}})"
                context_grid = np.concatenate([self._theta_obs, self._s_obs_grid], axis=-1)
        else:
            self.flow_dist = f"p(s_{{{rep_subs}}} | theta_{self.rep_abbrv})"
            flow_dir = self._obs_flow_dir
            features_grid = self._s_rep_prior
            context_grid = self._theta_obs

        LOGGER.info(f"flow = {self.flow_dist}")
        self.context_grid = context_grid

        if self.is_cross_probe:
            flow_label += "ppc/cross"
            flow_label += f"_{self.rep_abbrv}_given_{self.obs_abbrv}"
            flow_label += "_independent" if independent_cross else ""
        else:
            flow_label += "ppc/auto_"
            flow_label += self.obs_abbrv

        # Build the flow from flow_conf using the same architecture/training builders as
        # run_inference (msi.utils.flow), so PPC and inference share one flow definition. n_flows>1
        # gives a LikelihoodFlowEnsemble of independently-initialized members (matching the ensemble
        # used for fast GPU MCMC in run_inference).
        feature_dim = features_grid.shape[-1]
        context_dim = context_grid.shape[-1]

        def _arch(which):
            # fresh embedding/transform per ensemble member (which: 0=embedding_net, 1=transform)
            return flow_utils.build_flow_architecture(feature_dim, context_dim, self.flow_conf)[which]

        if self.n_flows > 1:
            self.flow = LikelihoodFlowEnsemble(
                params=[],
                conf=self.conf,
                n_flows=self.n_flows,
                feature_dim=feature_dim,
                embedding_net_fn=lambda: _arch(0),
                transform_fn=lambda: _arch(1),
                out_dir=flow_dir,
                label=flow_label,
                load_existing=not retrain,
                torch_seed=self.flow_conf.get("seed", 7),
            )
        else:
            embedding_net, transform = flow_utils.build_flow_architecture(feature_dim, context_dim, self.flow_conf)
            self.flow = LikelihoodFlow(
                params=[],
                conf=self.conf,
                feature_dim=feature_dim,
                embedding_net=embedding_net,
                transform=transform,
                out_dir=flow_dir,
                label=flow_label,
                load_existing=not retrain,
                torch_seed=self.flow_conf.get("seed", 7),
            )
        self.out_dir = self.flow.model_dir

        # Default policy: recover the flow from its checkpoint when one exists, otherwise train.
        # ``retrain`` forces training even when a checkpoint is present. (When retrain is False and a
        # checkpoint is missing, the constructor above already fell back to fresh weights, so we train.)
        do_train = retrain or not os.path.exists(self.flow.model_file)
        if do_train:
            reason = "retrain requested" if retrain else f"no checkpoint at {self.flow.model_file}"
            LOGGER.info(f"Training PPC flow ({reason})")
            train_kwargs = flow_utils._extract_train_kwargs(self.flow_conf)
            train_kwargs.update(fit_kwargs)  # explicit per-call overrides win
            self.flow.fit(x=features_grid, theta=context_grid, save_model=True, **train_kwargs)
        else:
            LOGGER.info(f"Loaded existing PPC flow from {self.flow.model_file}")

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
        check_linf=True,
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
            check_linf (bool): Whether to check the max standardised deviation (L∞ norm).
        """

        self._set_observation(obs_label, s_obs, theta_post, s_obs_rep, theta_post_rep)

        if plot_param_posterior:
            self._plot_param_posterior()

        self._sample_neural_posterior_predictive(n_samples=n_samples_neural)
        if not self.is_cross_probe:
            self._sample_grid_posterior_predictive(n_importance_samples=n_samples_grid, k_highest=k_highest_grid)

        if check_data_marginals:
            # The marginals figure is a non-essential diagnostic; never let a plotting failure (e.g.
            # a trianglechain density-estimation edge case) abort the quantitative checks below or the
            # remaining runs in the loop.
            try:
                self._check_data_marginals()
            except Exception as e:
                LOGGER.warning(
                    f"Skipping data-marginals plot for {self.obs_label} ({self._setup_descriptor()}): "
                    f"{type(e).__name__}: {e}. Quantitative checks are unaffected."
                )

        if check_log_prob:
            self._check_log_prob()

        if check_kernel:
            self._check_one_sample(stat="kernel")

        if check_mahalanobis:
            self._check_one_sample(stat="mahalanobis")

        if check_l2:
            self._check_one_sample(stat="l2")

        if check_l1:
            self._check_one_sample(stat="l1")

        if check_linf:
            self._check_one_sample(stat="linf")

    def _set_observation(self, obs_label=None, s_obs=None, theta_post=None, s_obs_rep=None, theta_post_rep=None):
        """Set up the observation data and configuration for the PPC."""

        self.obs_label = obs_label

        self.post_dist = f"p(theta_{self.obs_abbrv} | s_{{{self._summ_subs('obs')}}})"
        LOGGER.info(f"post = {self.post_dist}")

        obs_flow_dir = self._obs_flow_dir
        obs_dict = self._obs_obs_dict

        if self.is_cross_probe:
            rep_flow_dir = self._rep_flow_dir
            rep_obs_dict = self._rep_obs_dict

        # obs_probe
        if s_obs is None:
            assert obs_label in obs_dict, (
                f"obs_label '{obs_label}' not found in {self.obs_probe_name} observations: "
                f"{sorted(obs_dict.keys())}"
            )
            s_obs = obs_dict[obs_label]
        self.s_obs = s_obs

        if theta_post is None:
            theta_post = np.load(os.path.join(obs_flow_dir, f"chain_{obs_label}.npy"))
        self.theta_post = theta_post

        # rep_probe
        if self.is_cross_probe:
            if s_obs_rep is None:
                assert obs_label in rep_obs_dict, (
                    f"obs_label '{obs_label}' not found in {self.rep_probe_name} observations: "
                    f"{sorted(rep_obs_dict.keys())}"
                )
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

        def _chain_label(probe_name, data):
            return f"{probe_name}/{data}" if data else probe_name

        chains = [self.theta_post]
        labels = [_chain_label(self.obs_probe_name, self.obs_data)]
        params = [self._obs_params]

        if self.is_cross_probe:
            chains.append(self.theta_post_rep)
            labels.append(_chain_label(self.rep_probe_name, self.rep_data))
            params.append(self._rep_params)

        plotting.plot_chains(
            chains=chains,
            params=params,
            conf=self.conf,
            plot_labels=labels,
            obs_cosmo=None,
            out_dir=self.out_dir,
            file_label=self.obs_label,
        )

    def _sample_neural(self, theta_post, n_samples, s_obs=None, log=True):
        """Pure neural posterior-predictive sampler: returns ``(s_rep, context_star)``.

        Draws ``s_rep ~ p(s | context)`` where ``context`` is built from ``theta_post`` (a
        posterior chain) exactly as the obs path does -- for cross-probe checks the obs summary
        ``s_obs`` is concatenated (joint) or only the cosmo columns are kept (independent). Holds
        no instance state, so it can be called repeatedly on mock posteriors during calibration
        without clobbering the obs-path ``self.s_rep`` / ``self.context_star``.
        """
        # subsample the posterior
        i_star = self.rng.integers(0, theta_post.shape[0], n_samples)
        theta_star = theta_post[i_star]

        # sample the flow
        if self.is_cross_probe and not self.independent_cross:
            s_obs_star = np.repeat(np.atleast_2d(s_obs), n_samples, axis=0)
            context_star = np.concatenate([theta_star, s_obs_star], axis=-1)
        elif self.is_cross_probe and self.independent_cross:
            # marginalise over probe-specific nuisances by using only the shared cosmo columns
            context_star = theta_star[:, self._obs_cosmo_idx]
        else:
            context_star = theta_star

        if log:
            LOGGER.info(f"Generating {n_samples} neural samples of {self.flow_dist} flow")
            LOGGER.timer.start("sampling")
        s_rep = self.flow.sample_likelihood(context_star, n_samples=1, batch_size=min(context_star.shape[0], 10_000))
        s_rep = np.squeeze(s_rep)
        if log:
            LOGGER.info(f"Done sampling after {LOGGER.timer.elapsed('sampling')}")

        # Normalizing-flow sampling can occasionally emit pathological draws: the inverse transform
        # overflows for extreme base-distribution samples, giving either non-finite values OR finite
        # but astronomically large ones (|s| ~ 1e150). A freshly retrained flow may do this where a
        # previously trained one did not. Both wreck the run: a NaN/inf or huge row breaks the
        # density-grid plot in _check_data_marginals (the huge values overflow PCA's X.T@X to NaN) and
        # poisons every statistic (Mahalanobis covariance, distance means). Drop such rows together
        # with their paired context (keeping s_rep and context_star aligned). The bound is a very
        # generous margin around the prior summary support (self.s_prior, real finite sims in the same
        # rep space), so only catastrophic flow failures are removed -- legitimate PPD tail draws sit
        # comfortably inside it and well-behaved flows lose nothing. A large bad fraction means the
        # flow is unusable and is surfaced loudly even inside the (quiet) calibration loop.
        s_rep_2d = s_rep.reshape(s_rep.shape[0], -1)
        lo = self.s_prior.min(axis=0)
        hi = self.s_prior.max(axis=0)
        span = np.maximum(hi - lo, np.finfo(s_rep.dtype).tiny)
        margin = 100.0  # keep draws within 100x the full prior span of either edge
        good = (
            np.isfinite(s_rep_2d).all(axis=1)
            & (s_rep_2d >= lo - margin * span).all(axis=1)
            & (s_rep_2d <= hi + margin * span).all(axis=1)
        )
        n_bad = int((~good).sum())
        if n_bad:
            frac = n_bad / s_rep.shape[0]
            if log or frac > 0.01:
                LOGGER.warning(
                    f"Dropping {n_bad}/{s_rep.shape[0]} ({frac:.2%}) pathological neural samples from "
                    f"{self.flow_dist} flow (non-finite or far outside the prior summary support)."
                )
            s_rep = s_rep[good]
            context_star = context_star[good]

        return s_rep, context_star

    def _sample_neural_posterior_predictive(self, n_samples=100_000):
        """Sample from the neural posterior predictive distribution (obs path; stores state)."""
        self.s_rep, self.context_star = self._sample_neural(self.theta_post, n_samples, s_obs=self.s_obs)

    def _grid_importance_indices(self, n_samples):
        """Importance-sample indices from the cosmology grid using p(s_obs | theta).

        Computes flow log-likelihoods at ``self.s_obs_rep`` across ``self.context_grid``, turns
        them into normalised weights, draws ``n_samples`` indices with replacement, and returns
        ``(i_picked, ess)``. The number of indices returned is capped at ``int(ESS)`` because
        drawing more than that just yields duplicates.
        """
        log_probs = (
            self.flow.log_likelihood(
                np.repeat(np.atleast_2d(self.s_obs_rep), self.context_grid.shape[0], axis=0), self.context_grid
            )
            .cpu()
            .numpy()
        )
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs)

        ess = 1.0 / np.sum(probs**2)
        LOGGER.info(f"Effective Sample Size (ESS) = {ess:.1f} out of {self.context_grid.shape[0]}")

        n_eff = max(1, int(ess))
        if n_samples > n_eff:
            LOGGER.info(f"Capping importance samples at int(ESS)={n_eff} (requested {n_samples}).")
            n_samples = n_eff

        LOGGER.info(f"Drawing {n_samples} samples from the grid with importance weights")
        i_picked = self.rng.choice(self.context_grid.shape[0], size=n_samples, replace=True, p=probs)
        n_unique = np.unique(i_picked).shape[0]
        LOGGER.info(f"Obtained {n_unique} unique samples out of {n_samples} samples")
        return i_picked, ess

    def _sample_grid_posterior_predictive(self, n_importance_samples=None, k_highest=None, ess_floor=500):
        """Sample from the grid posterior predictive using importance sampling or top-k selection.

        When the effective sample size (ESS) falls below ``ess_floor``, the simulation grid is
        too sparse near the posterior mode for the resulting empirical PPD to be meaningful;
        ``self.s_rep_grid`` is set to ``None`` and the caller should skip plotting/using it.
        """
        # TODO the importance weights below are only correct for the auto-probe case. They should be
        # proportional to p(theta | s_obs) ~ p(s_obs | theta), but _grid_importance_indices evaluates
        # p(s_rep | theta, s_obs). Those coincide only when s_rep == s_obs, i.e. rep == obs -- hence the
        # guard. Implementing the cross case needs a separate p(s_obs | theta) evaluation.
        assert not self.is_cross_probe, "Grid PPC not implemented for cross-probe checks yet."

        if (n_importance_samples is not None) and (k_highest is None):
            i_picked, ess = self._grid_importance_indices(n_importance_samples)
            if ess < ess_floor:
                LOGGER.warning(
                    f"Grid PPD ESS={ess:.1f} is below the floor of {ess_floor}; the simulation grid "
                    "is too sparse near the posterior mode for a reliable empirical PPD. Skipping the "
                    "grid-based PPD (s_rep_grid=None)."
                )
                self.s_rep_grid = None
                return
            self.s_rep_grid = self.s_prior[i_picked]

        elif (k_highest is not None) and (n_importance_samples is None):
            log_probs = (
                self.flow.log_likelihood(
                    np.repeat(np.atleast_2d(self.s_obs_rep), self.context_grid.shape[0], axis=0), self.context_grid
                )
                .cpu()
                .numpy()
            )
            log_probs -= np.max(log_probs)
            LOGGER.info(f"Selecting the {k_highest} highest probability samples from the grid")
            i_sorted = np.argsort(log_probs)[-k_highest:]
            self.s_rep_grid = self.s_prior[i_sorted]

        else:
            raise ValueError("Either n_importance_samples or k_highest must be specified, but not both")

    def _check_data_marginals(self, n_scatter=1_000, outlier_quantile=1e-3):
        """Check and plot the marginal distributions of the data."""

        n_s = self.s_prior.shape[1]

        rep_subs = self._summ_subs("rep")
        obs_subs = self._summ_subs("obs")

        prior_label = r"$p(s_{" + rep_subs + r"})$"
        post_label = r"$p(s_{" + rep_subs + r"}|s_{" + obs_subs + r"}^{obs})$"
        post_label_sim = r"$p(s_{" + rep_subs + r"}|s_{" + obs_subs + r"}^{obs})$ (sims)"
        obs_label_str = r"$s_{" + rep_subs + r"}^{obs}$"

        tri = TriangleChain(
            progress_bar=False,
            show_legend=True,
            legend_fontsize=24,
            size=2,
            line_kwargs={"zorder": 0, "linewidths": 2},
            hist_kwargs={"zorder": 0, "lw": 2},
            scatter_kwargs={"s": 1, "marker": "o"},
            params=[str(i) for i in range(n_s)],
            labels=[rf"$s^{{{i}}}_{{{rep_subs}}}$" for i in range(n_s)],
            ranges={
                str(i): (
                    np.quantile(self.s_prior[:, i], outlier_quantile),
                    np.quantile(self.s_prior[:, i], (1 - outlier_quantile)),
                )
                for i in range(n_s)
            },
        )

        def contour_or_scatter(tri, data, color, label):
            if data.shape[0] > n_scatter:
                tri.contour_cl(data, color=color, label=label)
            else:
                tri.scatter(data, color=color, label=label)

        contour_or_scatter(tri, self.s_prior, color="tab:blue", label=prior_label)
        contour_or_scatter(tri, self.s_rep, color="tab:orange", label=post_label)
        if not self.is_cross_probe and self.s_rep_grid is not None:
            contour_or_scatter(tri, self.s_rep_grid, color="tab:green", label=post_label_sim)

        tri.scatter(
            np.atleast_2d(self.s_obs_rep),
            scatter_kwargs={"s": 200, "marker": "*", "zorder": 10},
            color="k",
            scatter_vline_1D=True,
            plot_histograms_1D=False,
            label=obs_label_str,
        )

        # only keep the last legend
        try:
            for legend in tri.fig.legends[:-1]:
                legend.remove()
        except AttributeError:
            pass

        if tri.fig.legends:
            legend = tri.fig.legends[-1]
            legend._loc = 1  # 1 = 'upper right'
            legend.set_bbox_to_anchor((0.80, 0.80))
            for text in legend.get_texts():
                text.set_fontsize(24)
            for marker in legend.legend_handles:
                try:
                    marker.set_sizes([200])
                except AttributeError:
                    pass
                try:
                    marker.set_linewidth(4.0)
                except AttributeError:
                    pass

        # to fix the ugly tick labels
        import matplotlib.ticker as mticker

        _fmt = mticker.FuncFormatter(lambda x, _: f"{x:.4g}")
        for ax in tri.fig.axes:
            try:
                ss = ax.get_subplotspec()
                row = ss.rowspan.start
                col = ss.colspan.start
            except (AttributeError, TypeError):
                continue
            ax.xaxis.set_major_formatter(_fmt)
            ax.yaxis.set_major_formatter(_fmt)
            if row < n_s - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if col > 0:
                ax.tick_params(axis="y", labelleft=False)

        tri.fig.suptitle(f"{self.obs_label} — {self._setup_descriptor()}", fontsize=24, y=0.9)

        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_data_marginals.png")
        LOGGER.info(f"Saving data marginals plot to {plot_file}")
        tri.fig.savefig(plot_file, bbox_inches="tight", dpi=100)

    def _pval_log_prob(self, s_rep, s_obs_rep, context_star):
        """Pure log-prob PPC p-value (see ``_check_log_prob`` for the statistic's meaning).

        Uses the full PPD cloud directly: every ``(theta_i, s_rep_i)`` pair in ``s_rep`` /
        ``context_star`` contributes one paired draw -- no bootstrap subsampling. The pairs are
        already i.i.d. draws from ``p(s_rep | s_obs)``, so the plain mean over all of them is the
        lowest-variance estimate of ``P[T_rep >= T_obs]``; the only knob is the number of PPD
        samples (``sampling.n_samples_neural``).

        Returns ``(p_val, t_score, t_diff)`` where ``t_diff`` is the per-draw paired
        log-likelihood difference (for plotting) and ``t_score = median(t_diff)`` is a
        continuous discrepancy oriented so *larger = more extreme* (obs less likely than rep),
        used by the calibration's tie-robust ranking.
        """
        s_obs = np.atleast_2d(s_obs_rep)
        n = s_rep.shape[0]

        log_lik = lambda x, ctx: self.flow.log_likelihood(x, ctx, return_numpy=True)  # noqa: E731
        t_diff = log_lik(s_rep, context_star) - log_lik(
            np.repeat(s_obs, n, axis=0), context_star
        )  # positive: rep more likely than obs
        p_val = np.mean(t_diff <= 0)
        return p_val, float(np.median(t_diff)), t_diff

    def _check_log_prob(self):
        """Bayesian posterior predictive p-value via paired log-likelihood comparison.

        For each PPD draw i (all ``n_samples_neural`` of them -- no bootstrap), computes
            delta_i = log p(s_rep_i | theta_i) - log p(s_obs | theta_i)
        where theta_i ~ p(theta | s_obs) and s_rep_i ~ p(s | theta_i).
        p-value = fraction of draws where delta_i <= 0 (obs at least as likely as rep).

        Interpretation:
            - p ≈ 0.5 : good fit (obs is a typical draw under the model).
            - p → 0   : tension — the obs is in the lower tail of the predictive log-density,
                        i.e. less likely than rep draws under the same theta.
            - p → 1   : usually NOT a tension signal but a sign of posterior over-dispersion or
                        flow leakage (model assigns the obs higher density than its own samples).

        Note: this p-value is not uniform under the null (a known property of Bayesian PPD
        p-values: the data is used to fit the posterior and to evaluate the test, so p
        concentrates near 0.5 — small p is therefore conservative as a tension signal).

        Cross-probe caveat: when ``independent_cross=False`` the flow models the conditional
        ``p(s_rep | theta, s_obs)``, so the densities here are conditional on s_obs. This is a
        different statistic from the marginal-likelihood test in Doux et al. 2020 and the
        numerical p-values are not directly comparable.
        """
        p_val, _, t_diff = self._pval_log_prob(self.s_rep, self.s_obs_rep, self.context_star)

        rep_subs = self._summ_subs("rep")
        diff_label = (
            r"$\log p(s_{" + rep_subs + r"}^{rep}|\theta_i)" r" - \log p(s_{" + rep_subs + r"}^{obs}|\theta_i)$"
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(t_diff, bins=100, alpha=0.5, label=diff_label)
        ax.axvline(0, color="k", linestyle="--", label=f"p = {p_val:.4f}")
        ax.set(
            xlabel=diff_label,
            ylabel="Count",
            title=f"{self.obs_label}: Log-Prob PPC: p = {p_val:.4f}\n{self._setup_descriptor()}",
        )
        ax.legend()
        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_log_prob_check.png")
        LOGGER.info(f"Saving Log-Prob PPC plot to {plot_file}")
        fig.savefig(plot_file, bbox_inches="tight", dpi=100)

    def log_prob_ppc(self):
        """The log-prob PPC result for the current observation, as numbers rather than a plot.

        ``run_checks(check_log_prob=True)`` draws the histogram straight into the run directory;
        this returns the same quantities for a caller that draws them itself (the paper figure
        does). Call it after ``run_checks`` has set the observation state -- with every ``check_*``
        off if the pipeline's own PNGs are not wanted.

        Returns:
            tuple: ``(p_val, t_score, t_diff)`` as defined by ``_pval_log_prob`` -- the posterior
            predictive p-value ``P[t_diff <= 0]``, the median paired difference, and the per-draw
            differences the p-value is the tail mass of.
        """
        return self._pval_log_prob(self.s_rep, self.s_obs_rep, self.context_star)

    def _to_dev(self, a):
        """Move a numpy array onto the flow's torch device/dtype (lazy torch import).

        Used by the distance statistics so the (calibration-dominating) pairwise-distance work runs
        on the GPU via ``torch.cdist`` with the reduction done on-device -- only the small reduced
        vector is copied back to host.
        """
        import torch

        return torch.as_tensor(
            np.ascontiguousarray(a),
            dtype=getattr(self.flow, "floatx", torch.float32),
            device=getattr(self.flow, "device", "cpu"),
        )

    def _pval_one_sample(self, stat, s_rep, s_obs_rep, n_bootstrap=10_000, n_ref=5_000, log=True):
        """Generic one-sample test: is s_obs an outlier relative to the PPD? (pure helper)

        Null distribution: evaluate the same statistic on bootstrap draws from
        the PPD samples s_rep.  A small p-value means s_obs is extreme.

        For kernel/L1/L2/Linf the data are whitened by the ref-pool covariance
        (Cholesky-based; per-dimension standardisation is used as a fallback if
        the covariance is not positive-definite).  Whitening makes the metric
        isotropic in the PPD frame, so deviations along narrow correlated
        directions are not swamped by long axes of the cloud.  s_rep is split
        into non-overlapping halves so the reference cloud (defines the
        statistic) and the bootstrap pool (builds the null) are independent;
        the Mahalanobis covariance is also estimated from the ref pool only.

        Returns ``(p_val, t_score, info)``: ``t_score`` is oriented so *larger = more extreme*
        (``= t_obs`` for the high-tail stats, ``= -t_obs`` for kernel similarity), so the
        calibration can rank one continuous discrepancy uniformly across stats; ``info`` carries
        the arrays/labels the plotting wrapper needs.

        Args:
            stat: 'mahalanobis', 'l1', 'l2', 'linf', or 'kernel'.
            s_rep: PPD samples, shape (N, dim).
            s_obs_rep: observed summary for the replicated probe, shape (dim,) or (1, dim).
            n_bootstrap: Number of bootstrap draws for the null.
            n_ref: Reference subsample size for distance-based stats (kernel, L1, L2).
        """
        from scipy.linalg import solve_triangular

        s_obs = np.atleast_2d(s_obs_rep)  # (1, dim)
        n_rep = s_rep.shape[0]

        # Non-overlapping split: ref pool defines the statistic; boot pool builds the null.
        perm = self.rng.permutation(n_rep)
        i_ref_pool, i_boot_pool = perm[: n_rep // 2], perm[n_rep // 2 :]

        # Whiten using ref-pool stats (fall back to per-dim standardisation if Cholesky fails).
        s_mu = np.mean(s_rep[i_ref_pool], axis=0)
        cov_ref = np.cov(s_rep[i_ref_pool], rowvar=False)
        cov_ref = np.atleast_2d(cov_ref)
        try:
            jitter = 1e-8 * max(1.0, float(np.trace(cov_ref)) / cov_ref.shape[0])
            L = np.linalg.cholesky(cov_ref + jitter * np.eye(cov_ref.shape[0]))

            def _whiten(x):
                return solve_triangular(L, (x - s_mu).T, lower=True).T

            s_rep_n = _whiten(s_rep)
            s_obs_n = _whiten(s_obs)
        except np.linalg.LinAlgError:
            LOGGER.warning(
                "Cholesky decomposition of ref-pool covariance failed; falling back to per-dim standardisation."
            )
            s_std = np.std(s_rep[i_ref_pool], axis=0)
            s_std[s_std == 0] = 1.0
            s_rep_n = (s_rep - s_mu) / s_std
            s_obs_n = (s_obs - s_mu) / s_std

        s_ref_n = s_rep_n[i_ref_pool[: min(n_ref, len(i_ref_pool))]]
        rep_subs = self._summ_subs("rep")

        if stat == "mahalanobis":
            cov_inv = np.linalg.pinv(cov_ref)

            def compute_stat(x):
                diff = x - s_mu
                return np.einsum("...i,ij,...j->...", diff, cov_inv, diff)

            s_obs_eval, s_rep_eval = s_obs, s_rep
            outlier_if_high = True
            xlabel = "Mahalanobis distance²"
            file_tag = "mahalanobis_check"
            title_tag = "Mahalanobis Distance Check"
            stat_label = r"$D_M^2(s_{" + rep_subs + r"}^{obs})$"

        elif stat in ("l1", "l2"):
            import torch

            norm_ord = 1 if stat == "l1" else 2
            ref_t = self._to_dev(s_ref_n)

            def compute_stat(x):
                return torch.cdist(self._to_dev(x), ref_t, p=float(norm_ord)).mean(dim=-1).cpu().numpy()

            s_obs_eval, s_rep_eval = s_obs_n, s_rep_n
            outlier_if_high = True
            xlabel = f"Mean L{norm_ord} distance to PPD"
            file_tag = f"{stat}_check"
            title_tag = f"L{norm_ord} Distance Check"
            stat_label = (
                r"$\bar{d}_{L" + str(norm_ord) + r"}(s_{" + rep_subs + r"}^{obs},\, s_{" + rep_subs + r"}^{rep})$"
            )

        elif stat == "linf":

            def compute_stat(x):
                return np.max(np.abs(x), axis=-1)  # max standardised deviation across dims

            s_obs_eval, s_rep_eval = s_obs_n, s_rep_n
            outlier_if_high = True
            xlabel = r"$\max_j |s_j^{std}|$  (L∞ norm)"
            file_tag = "linf_check"
            title_tag = "L∞ Distance Check"
            stat_label = r"$\|s_{" + rep_subs + r"}^{obs}\|_\infty$"

        elif stat == "kernel":
            import torch

            ref_t = self._to_dev(s_ref_n)
            n_bw = min(2_000, s_ref_n.shape[0])
            d2_bw = torch.cdist(ref_t[:n_bw], ref_t[:n_bw], p=2) ** 2
            iu = torch.triu_indices(n_bw, n_bw, offset=1, device=d2_bw.device)
            bw2 = float(d2_bw[iu[0], iu[1]].median()) or 1.0
            if log:
                LOGGER.info(f"Kernel bandwidth (squared, normalised): {bw2:.4f}")

            def compute_stat(x):
                d2 = torch.cdist(self._to_dev(x), ref_t, p=2) ** 2
                return torch.exp(-d2 / bw2).mean(dim=-1).cpu().numpy()

            s_obs_eval, s_rep_eval = s_obs_n, s_rep_n
            outlier_if_high = False
            xlabel = "Mean kernel similarity"
            file_tag = "kernel_check"
            title_tag = "Kernel Similarity Check"
            stat_label = r"$\bar{k}(s_{" + rep_subs + r"}^{obs},\, s_{" + rep_subs + r"}^{rep})$"

        else:
            raise ValueError(f"Unknown stat: {stat}")

        i_boot = i_boot_pool[self.rng.integers(0, len(i_boot_pool), n_bootstrap)]
        t_obs = compute_stat(s_obs_eval)[0]
        t_boot = compute_stat(s_rep_eval[i_boot])
        p_val = np.mean(t_boot >= t_obs) if outlier_if_high else np.mean(t_boot <= t_obs)
        t_score = t_obs if outlier_if_high else -t_obs

        info = dict(
            t_obs=t_obs, t_boot=t_boot, xlabel=xlabel, title_tag=title_tag, file_tag=file_tag, stat_label=stat_label
        )
        return p_val, float(t_score), info

    def _check_one_sample(self, stat, n_bootstrap=10_000, n_ref=5_000):
        """Run ``_pval_one_sample`` on the obs PPD and plot the null histogram + obs marker."""
        p_val, _, info = self._pval_one_sample(stat, self.s_rep, self.s_obs_rep, n_bootstrap, n_ref)
        t_obs, t_boot = info["t_obs"], info["t_boot"]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(t_boot, bins=100, alpha=0.5, label="null (PPD samples)")
        ax.axvline(t_obs, color="k", label=f"{info['stat_label']} = {t_obs:.4f}")
        ax.set(
            xlabel=info["xlabel"],
            ylabel="Count",
            title=f"{self.obs_label}: {info['title_tag']}: p = {p_val:.4f}\n{self._setup_descriptor()}",
        )
        ax.legend()

        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_{info['file_tag']}.png")
        LOGGER.info(f"Saving {info['title_tag']} plot to {plot_file}")
        fig.savefig(plot_file, bbox_inches="tight", dpi=100)

    # ---- p-value calibration (Doux et al. 2020, Eq. 9) -----------------------------------------
    # Calibrate each raw PPC p-value (auto AND cross) against its null distribution over the consistent
    # wide-prior mock observations whose posteriors the inference coverage stage already sampled into
    # ``{obs_flow_dir}/mcmc_samples.h5``. The reported p̃ is the percentile of the observed raw p within
    # that null and is ~Uniform(0,1) under the null, so p̃≈0.5 means the observed p (even a saturated
    # p≈1) is exactly what consistent data produces (no tension). Cross-probe calibration pairs each obs
    # mock to the rep-probe summary of the same realization via the saved ``real_idx`` (see
    # _load_mock_posteriors).
    _CALIB_STATS = ("log_prob", "mahalanobis", "l2", "l1", "linf", "kernel")

    def _load_mock_posteriors(self):
        """Load the wide-prior mock observations + posteriors from the OBS probe's ``mcmc_samples.h5``.

        Returns ``(x_true, theta_sample, x_true_rep)`` or ``None`` (with a warning) when the file (or,
        for cross-probe, the ``real_idx`` dataset) is absent:

        * ``x_true``      -- (N, dim_obs) obs-probe summaries of the held-out wide-prior mocks.
        * ``theta_sample``-- (n_samp, N, n_params) obs-probe posteriors, same params order as ``theta_post``.
        * ``x_true_rep``  -- (N, dim_rep) summary scored against the predictive. For AUTO this is just
          ``x_true`` (rep == obs). For CROSS it is the REP probe's summary of the SAME sky realization,
          looked up from the rep grid (``self._s_rep_prior``) by the per-mock ``(i_sobol, i_signal,
          i_noise)`` saved in ``real_idx`` -- the data-level pairing that makes the cross null respect
          the probe correlation (cf. ``_assert_aligned_grids``).

        Dimensions are asserted (obs dim for ``x_true``, rep dim for ``x_true_rep``, obs params for
        ``theta_sample``) so a mismatched / PCA / multi-checkpoint preds file fails loudly.
        """
        import h5py

        path = os.path.join(self._obs_flow_dir, "mcmc_samples.h5")
        if not os.path.exists(path):
            LOGGER.warning(
                f"No mcmc_samples.h5 at {path}; skipping p-value calibration. Produce it by running "
                "inference with --sample_posterior (the coverage stage writes this file)."
            )
            return None

        with h5py.File(path, "r") as f:
            x_true = f["x_true"][:]
            theta_sample = f["theta_sample"][:]
            real_idx = f["real_idx"][:] if "real_idx" in f else None

        assert x_true.shape[1] == self._s_obs_grid.shape[1], (
            f"mock x_true summary dim {x_true.shape[1]} != obs-probe summary dim "
            f"{self._s_obs_grid.shape[1]}; mcmc_samples.h5 must come from the same summary space "
            "(no PCA / multi-checkpoint preds)."
        )
        assert theta_sample.shape[-1] == len(self._obs_params), (
            f"mock theta dim {theta_sample.shape[-1]} != n obs params {len(self._obs_params)} "
            f"({self._obs_params})."
        )
        assert theta_sample.shape[1] == x_true.shape[0], (
            f"mcmc_samples.h5 mock-count mismatch: x_true has {x_true.shape[0]} rows but "
            f"theta_sample has {theta_sample.shape[1]}."
        )

        if not self.is_cross_probe:
            x_true_rep = x_true
        else:
            if real_idx is None:
                LOGGER.warning(
                    f"mcmc_samples.h5 at {path} has no 'real_idx' dataset; cross-probe calibration needs "
                    "the per-mock (i_sobol, i_signal, i_noise) to pair the obs mock with the rep-probe "
                    "summary. Regenerate it by re-running inference --sample_posterior. Skipping."
                )
                return None
            # Pair each obs mock to the rep-probe summary of the SAME sky realization. The rep grid is
            # aligned to (i_sobol, i_signal, i_noise) and asserted identical to the obs grid in cross
            # mode (_assert_aligned_grids), so a value-keyed lookup is order-independent and exact.
            rep_real_idx = getattr(self, f"{self.rep_probe}_real_idx")
            pos_of = {tuple(int(v) for v in row): i for i, row in enumerate(rep_real_idx)}
            try:
                rep_pos = np.array([pos_of[tuple(int(v) for v in row)] for row in real_idx])
            except KeyError as e:
                raise AssertionError(
                    f"mock realization {e.args[0]} from mcmc_samples.h5 not found in the rep-probe grid; "
                    "the obs and rep pred files must come from the same simulation grid / split."
                )
            x_true_rep = self._s_rep_prior[rep_pos]
            assert (
                x_true_rep.shape[1] == self.s_prior.shape[1]
            ), f"paired rep summary dim {x_true_rep.shape[1]} != PPC rep dim {self.s_prior.shape[1]}."

        LOGGER.info(f"Loaded {x_true.shape[0]} mock posteriors from {path}")
        return x_true, theta_sample, x_true_rep

    def _calibration_pvals(self, s_rep, context_star, s_obs_rep, stats, n_bootstrap, n_ref):
        """Compute ``{stat: (p, t_score)}`` for one (s_rep, s_obs) pair using the pure helpers."""
        out = {}
        for stat in stats:
            if stat == "log_prob":
                # log_prob uses the full PPD cloud (no bootstrap); n_bootstrap applies only to the
                # distance/kernel stats below. Parity holds since both legs draw n_samples_neural.
                p, score, _ = self._pval_log_prob(s_rep, s_obs_rep, context_star)
            else:
                # quiet inside the calibration loop: the per-mock kernel-bandwidth line would
                # otherwise repeat once per mock and swamp the log.
                p, score, _ = self._pval_one_sample(stat, s_rep, s_obs_rep, n_bootstrap, n_ref, log=False)
            out[stat] = (float(p), float(score))
        return out

    def run_calibration(self, n_sim="all", n_samples_neural=10_000, n_bootstrap=2_000, n_ref=1_000, stats=None):
        """Doux Eq. 9 calibration of the PPC p-values for the current observation (auto AND cross).

        ``n_sim`` is the number of mock observations forming the null: ``"all"`` (default) uses every
        mock available in ``mcmc_samples.h5``; an int thins them by a ``linspace`` stride.

        Builds the null distribution of each raw statistic p-value over ``n_sim`` consistent
        wide-prior mocks (loaded from the obs probe's ``mcmc_samples.h5``), recomputes the observed p
        at the SAME ``(n_samples_neural, n_bootstrap, n_ref)`` for parity, and reports
        ``p̃ = mean(p_mock <= p_obs)`` plus a tie-robust continuous variant
        ``p̃_cont = mean(score_mock >= score_obs)``. Must be called after ``run_checks`` (so the
        observation state is set).

        For cross-probe setups the null reuses the same trained cross flow ``p(s_rep | theta_obs,
        s_obs)``: each mock draws ``s_rep`` from the obs mock's posterior + obs summary and scores the
        rep-probe summary of the SAME sky realization (paired via ``real_idx`` in
        ``_load_mock_posteriors``), so the calibrated cross p̃ respects the probe correlation.
        Requires ``real_idx`` in ``mcmc_samples.h5`` (re-run inference --sample_posterior); skipped
        with a warning otherwise.
        """
        stats = tuple(stats) if stats is not None else self._CALIB_STATS
        mocks = self._load_mock_posteriors()
        if mocks is None:
            return None
        x_true, theta_sample, x_true_rep = mocks
        N = x_true.shape[0]
        if isinstance(n_sim, str):
            if n_sim != "all":
                raise ValueError(f"n_sim must be an int or 'all', got {n_sim!r}")
            LOGGER.info(f"n_sim='all': using all {N} mocks available in mcmc_samples.h5")
            n_sim = N
        else:
            n_sim = min(int(n_sim), N)
        idx = np.unique(np.linspace(0, N - 1, n_sim).round().astype(int))
        n_sim = idx.size

        LOGGER.info(
            f"Calibration ({self.obs_label}, {self._setup_descriptor()}): {n_sim} mocks, "
            f"n_samples_neural={n_samples_neural}, n_bootstrap={n_bootstrap}, n_ref={n_ref}"
        )

        # obs leg, at the SAME reduced settings as the mocks (parity)
        s_rep_obs, ctx_obs = self._sample_neural(self.theta_post, n_samples_neural, s_obs=self.s_obs)
        obs = self._calibration_pvals(s_rep_obs, ctx_obs, self.s_obs_rep, stats, n_bootstrap, n_ref)

        # null leg: one mock at a time. (Sampling is NOT batched across mocks -- enflows batches the
        # num_samples dimension, not the context rows, so a single flow.sample call over all mocks'
        # contexts would invert the whole sigmoid flow at once and OOM; per-mock sampling of
        # n_samples_neural contexts is cheap and bounded.)
        null_p = {s: np.empty(n_sim) for s in stats}
        null_score = {s: np.empty(n_sim) for s in stats}
        LOGGER.timer.start("calibration")
        for k, j in enumerate(idx):
            # context from the obs mock (posterior + obs summary); score the rep-probe summary of the
            # same realization (x_true_rep[j] == x_true[j] for auto).
            s_rep_j, ctx_j = self._sample_neural(theta_sample[:, j, :], n_samples_neural, s_obs=x_true[j], log=False)
            pj = self._calibration_pvals(s_rep_j, ctx_j, x_true_rep[j], stats, n_bootstrap, n_ref)
            for s in stats:
                null_p[s][k], null_score[s][k] = pj[s]
            if (k + 1) % 50 == 0 or (k + 1) == n_sim:
                LOGGER.info(f"  calibration: {k + 1}/{n_sim} mocks ({LOGGER.timer.elapsed('calibration')})")

        summary = {}
        for s in stats:
            p_obs, score_obs = obs[s]
            p_tilde = float(np.mean(null_p[s] <= p_obs))
            p_tilde_cont = float(np.mean(null_score[s] >= score_obs))
            summary[s] = dict(p_obs=p_obs, p_tilde=p_tilde, p_tilde_continuous=p_tilde_cont)
            LOGGER.info(f"calibration[{s}]: p_obs={p_obs:.4f}  p̃={p_tilde:.4f}  p̃_cont={p_tilde_cont:.4f}")
            self._plot_calibration(s, null_p[s], p_obs, p_tilde, null_score[s], score_obs, p_tilde_cont, n_sim)

        self._save_calibration_summary(
            summary, dict(n_sim=n_sim, n_samples_neural=n_samples_neural, n_bootstrap=n_bootstrap, n_ref=n_ref)
        )
        return summary

    def _plot_calibration(self, stat, null_p, p_obs, p_tilde, null_score, score_obs, p_tilde_cont, n_sim):
        """Two-panel calibration figure for one statistic.

        Left: null distribution of the raw inner p-value over the mocks with the observed p marked;
        ``p̃`` is the mass at/below the line. Right: null distribution of the continuous discrepancy
        (oriented larger = more extreme) with the observed value marked; ``p̃_cont`` is the mass
        at/above the line — the tie-robust companion that avoids the raw-p saturation near 1.
        """
        bins = min(50, max(10, n_sim // 10))
        fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(18, 6))

        # left: raw inner-p null (p̃ = fraction at/below the obs line)
        ax_p.hist(null_p, bins=bins, range=(0, 1), alpha=0.5, color="tab:blue", label=f"null p ({n_sim} mocks)")
        ax_p.axvline(p_obs, color="k", label=f"p_obs = {p_obs:.4f}")
        ax_p.set(xlabel=f"raw {stat} p-value", ylabel="Count", title=f"p̃ = {p_tilde:.3f}  (rank of raw p)")
        ax_p.legend()

        # right: continuous-discrepancy null (p̃_cont = fraction at/above the obs line)
        ax_s.hist(null_score, bins=bins, alpha=0.5, color="tab:orange", label=f"null discrepancy ({n_sim} mocks)")
        ax_s.axvline(score_obs, color="k", label=f"t_obs = {score_obs:.4g}")
        ax_s.set(
            xlabel=f"{stat} discrepancy (larger = more extreme)",
            ylabel="Count",
            title=f"p̃_cont = {p_tilde_cont:.3f}  (rank of continuous discrepancy)",
        )
        ax_s.legend()

        fig.suptitle(f"{self.obs_label}: {stat} calibration — {self._setup_descriptor()}")
        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_{stat}_calibration.png")
        LOGGER.info(f"Saving calibration plot to {plot_file}")
        fig.savefig(plot_file, bbox_inches="tight", dpi=100)
        plt.close(fig)

    def _save_calibration_summary(self, summary, meta):
        """Write the per-statistic {p_obs, p_tilde, p_tilde_continuous} table to JSON."""
        import json

        meta = dict(meta, obs_label=self.obs_label, setup=self._setup_descriptor())
        out_file = os.path.join(self.out_dir, f"{self.obs_label}_calibration.json")
        with open(out_file, "w") as fp:
            json.dump({"meta": meta, "stats": summary}, fp, indent=2)
        LOGGER.info(f"Saved calibration summary to {out_file}")

    _PROBE_NAME_TO_CLS_FLAGS = {
        "lensing": {"with_lensing": True, "with_clustering": False, "with_cross_z": True, "with_cross_probe": False},
        "clustering": {
            "with_lensing": False,
            "with_clustering": True,
            "with_cross_z": True,
            "with_cross_probe": False,
        },
        "cross": {"with_lensing": False, "with_clustering": False, "with_cross_z": True, "with_cross_probe": True},
        "combined": {"with_lensing": True, "with_clustering": True, "with_cross_z": True, "with_cross_probe": True},
    }

    def _build_cls_obs(self, dlss_conf, cls_n_bins):
        """Build the observed (linear) rebinned Cls vector for ``self.obs_label`` via the
        catalog → maps → hard_rebinned-Cls pipeline — the same preprocessing the cls network is
        trained on (``deep_lss.utils.cls_preprocessing.preprocess_obs_hard_rebinned``).

        Only catalog-based labels (currently ``"DESy3"``) are supported. For mock labels that
        live only in the maps obs_dict (e.g. ``"bench_fidu_mean"``, ``"grid_*"``), the caller
        must pass ``cls_obs`` to ``check_cls_marginals`` directly — the underlying maps for
        those mocks are not co-located with the compressed-summary obs_dict.
        """
        # deferred imports: keep PPC usable when the catalog data / TF are not present
        from msfm.utils import catalog
        from deep_lss.utils import cls_preprocessing

        if dlss_conf is None:
            raise ValueError(
                "_build_cls_obs requires dlss_conf (with scale_cuts); pass it to "
                "check_cls_marginals or provide cls_obs explicitly."
            )

        if self.obs_label != "DESy3":
            raise ValueError(
                f"Auto-build of cls_obs is only implemented for obs_label='DESy3', "
                f"got '{self.obs_label}'. Pass cls_obs explicitly for mock observations."
            )

        flags = self._PROBE_NAME_TO_CLS_FLAGS[self.obs_probe_name]
        LOGGER.info(f"Auto-building cls_obs for obs_label='{self.obs_label}' with flags {flags}")

        wl_gamma_map, _ = catalog.build_metacal_map_from_cat(self.conf)
        gc_count_map = catalog.build_maglim_map_from_cat(self.conf)

        # Ask for the *linear* rebinned Cls (apply_log=False): the network input uses the sign-log
        # transform sign(x)*log(|x|+eps), which is non-injective and cannot be inverted to linear, so
        # we build the linear obs directly (same channels / flatten order as load_rebinned_cls_grid).
        obs = cls_preprocessing.preprocess_obs_hard_rebinned(
            wl_gamma_map=wl_gamma_map,
            gc_count_map=gc_count_map,
            msfm_conf=self.conf,
            dlss_conf=dlss_conf,
            cls_n_bins=cls_n_bins,
            apply_log=False,
            **flags,
        )
        return np.asarray(obs).reshape(-1)

    def check_cls_marginals(
        self,
        dlss_conf,
        base_dir,
        cls_n_bins,
        scales_name,
        cls_obs=None,
        grid=None,
        sample_set="test",
        k_top=None,
        apply_log=True,
        n_samples=5_000,
        percentiles=(16, 84),
        outer_percentiles=(2.5, 97.5),
        file_label="cls",
        x_label="ell bin (per z-pair)",
        log_y=None,
        n_traces=20,
    ):
        """Plot the posterior predictive distribution in Cls space (auto-probe only).

        Uses the ``hard_rebinned`` pipeline the cls network is trained on
        (``deep_lss.utils.cls_preprocessing``): loads the full rebinned Cls grid from the cache in
        ``base_dir`` (``cls/rebinned_nb{cls_n_bins}_{scales_name}.h5``), exactly aligns it to
        ``self.s_prior`` by the per-row ``(i_sobol, i_signal, i_noise)`` sky realization,
        importance-samples indices, and optionally applies ``log(|Cls|)``. The cached grid examples
        are already per-noise realizations, so (unlike the old soft-pruned path) no extra noise draw
        is added.

        Args:
            dlss_conf: dict (or path) for the deep-lss config; supplies ``scale_cuts`` matching the
                cache's ``scales_name`` (used for the obs rebinning and the per-pair ell axis).
            base_dir: data directory containing the ``cls/rebinned_*`` cache.
            cls_n_bins: number of ell bins per tomographic pair (matches the cache / training).
            scales_name: scales-config stem identifying the cache (e.g. ``"8wl,32gc"``).
            cls_obs: (n_cls_dims,) observed *linear* rebinned Cls vector. If None, built
                automatically via ``_build_cls_obs`` (only supports ``self.obs_label='DESy3'``).
            grid: optional preloaded cache as the 4-tuple
                ``(cls_full, real_idx_full, cosmos_full, cosmo_param_names)`` returned by
                ``cls_preprocessing.load_rebinned_cls_grid`` (the cache is ~GB; passing it lets a
                notebook iterate without re-reading it). If None, it is loaded internally.
            sample_set: which grid realizations form the PPD pool. The PPD is the astro-aware
                posterior predictive: each realization is importance-weighted by ``p(s_obs | theta)``
                with the FULL ``theta`` (cosmology + the per-signal Latin-hypercube astro nuisances)
                the flow was trained on, and drawn proportionally (capped at ``int(ESS)``).
                - ``"test"`` (default): only the test realizations, whose theta is the exact
                  context the flow was trained on (no parameter-column matching). ESS limited.
                - ``"all"``: every realization (train + test), giving ~5x more astro samples and a
                  higher ESS. The flow's train/val losses are close, so scoring train thetas is fine.
                  theta is taken from the cache cosmos, column-matched to the flow context and
                  verified against the test context_grid. Requires cosmos (auto-loaded, or pass a
                  4-tuple ``grid``).
            k_top: if set to an int, replace importance sampling with a deterministic TOP-K
                selection: the ``k_top`` realizations with the highest ``p(s_obs | theta)``. Gives
                ``k_top`` distinct sims and a smooth band even when the ESS collapses, but it is
                BIASED — it shows the spread of the best-fitting sims, not the posterior-predictive
                width (it drops the weight tails, so it understates the spread). Default None =
                importance sampling (the unbiased posterior predictive).
            apply_log: apply ``log(|Cls|)`` to both PPD samples and obs for the top panel. Default True.
            n_samples: number of importance draws (capped at int(ESS); drawing more just duplicates).
            percentiles: low/high percentiles for the inner shaded band.
            outer_percentiles: low/high percentiles for an outer (fainter) band, or None to draw a
                single band. Default ``(2.5, 97.5)`` pairs the 95% band with the 68% inner band.
            file_label: tag in the output filename.
            x_label: x-axis label.
            log_y: log scale on the y-axis. Defaults to ``not apply_log`` (linear when data
                are already log-transformed).
            n_traces: number of thin individual PPD lines to overlay under the band.
        """
        assert not self.is_cross_probe, "check_cls_marginals is implemented for auto-probe checks only."

        from deep_lss.utils import cls_preprocessing

        if log_y is None:
            log_y = not apply_log

        flags = self._PROBE_NAME_TO_CLS_FLAGS[self.obs_probe_name]

        if grid is None:
            LOGGER.info(f"Loading rebinned Cls grid (cls_n_bins={cls_n_bins}, scales={scales_name}) from {base_dir}")
            cls_full, real_idx_full, cosmos_full, cosmo_param_names = cls_preprocessing.load_rebinned_cls_grid(
                data_dir=base_dir,
                msfm_conf=self.conf,
                dlss_conf=dlss_conf,
                cls_n_bins=cls_n_bins,
                scales_name=scales_name,
                **flags,
            )
        else:
            cls_full, real_idx_full, cosmos_full, cosmo_param_names = grid
        # cls_full: (n_cosmo * n_examples, n_cls_dims) linear, bin-major / pair-minor flatten order.

        # Per-pair ell axis (sqrt-spaced bin centers, one block of cls_n_bins per selected pair)
        # plus each pair's (lmin, lmax) scale-cut range for the bottom-axis annotation.
        _pair_labels, ell_centers, ell_ranges = cls_preprocessing.get_rebinned_pair_info(
            self.conf, dlss_conf, cls_n_bins, **flags
        )
        n_pairs = ell_centers.shape[0]
        n_ell_per_pair = cls_n_bins
        n_dims = n_pairs * n_ell_per_pair
        ell_flat = ell_centers.reshape(-1)  # pair-major (pair, bin) -> (n_dims,)

        assert cls_full.shape[1] == n_dims, (
            f"rebinned Cls grid has {cls_full.shape[1]} columns but the probe selection implies "
            f"{n_pairs} pairs x {cls_n_bins} bins = {n_dims}."
        )

        # Reorder the flat vectors from the cache's bin-major / pair-minor layout to pair-major
        # (contiguous cls_n_bins-blocks per pair) so the flat-index panels and ell axis line up.
        def _to_pair_major(a):
            a = np.asarray(a)
            return a.reshape(a.shape[:-1] + (cls_n_bins, n_pairs)).swapaxes(-1, -2).reshape(a.shape[:-1] + (n_dims,))

        if cls_obs is None:
            cls_obs_raw = self._build_cls_obs(dlss_conf=dlss_conf, cls_n_bins=cls_n_bins)
        else:
            cls_obs_raw = np.atleast_1d(np.asarray(cls_obs)).reshape(-1)
        assert cls_obs_raw.shape[0] == n_dims, (
            f"cls_obs has {cls_obs_raw.shape[0]} elements but the probe selection implies {n_dims}; "
            "ensure probe/bin selection matches."
        )
        cls_obs_raw = _to_pair_major(cls_obs_raw)
        cls_obs = np.log(np.abs(cls_obs_raw)) if apply_log else cls_obs_raw.copy()

        # --- build the realization pool (sample_set), then importance-sample it ----------------------
        # Map each prediction-grid realization to its row in the loaded Cls grid by the
        # (i_sobol, i_signal, i_noise) key (the predictions were aligned on this in __init__).
        theta_obs = np.asarray(self._theta_obs)
        pos_of = {tuple(int(v) for v in row): i for i, row in enumerate(real_idx_full)}
        try:
            order = np.array([pos_of[tuple(int(v) for v in row)] for row in self.probe1_real_idx])
        except KeyError as e:
            raise AssertionError(
                f"realization {e.args[0]} from self.s_prior not found in the rebinned Cls grid; the "
                "predictions file and the Cls cache must come from the same simulation grid."
            )

        if sample_set == "test":
            # Only the test realizations; theta is exactly the flow's context_grid.
            cls_pool = cls_full[order]
            theta_pool = theta_obs
            pool_real_idx = self.probe1_real_idx
        elif sample_set == "all":
            # Every realization. theta comes from the cache cosmos, column-matched to the flow context
            # and VERIFIED against context_grid on the test realizations (guards parameter ordering).
            from msfm.utils import parameters as _msfm_params

            requested = set(_msfm_params.get_parameters(self._obs_params, self.conf))
            col_idx = [i for i, p in enumerate(cosmo_param_names) if p in requested]
            assert len(col_idx) == theta_obs.shape[1], (
                f"matched {len(col_idx)} cosmos columns but the flow context has {theta_obs.shape[1]}; "
                "parameter sets disagree."
            )
            theta_pool = cosmos_full[:, col_idx]
            cls_pool = cls_full
            pool_real_idx = real_idx_full
            assert np.allclose(theta_pool[order], theta_obs, rtol=1e-3, atol=1e-5), (
                "column-matched cosmos disagree with the flow context on the test realizations; "
                "parameter ordering mismatch — refusing to weight on misaligned theta."
            )
        else:
            raise ValueError(f"sample_set must be 'test' or 'all', got {sample_set!r}")

        # Astro-aware posterior predictive: weight each realization by p(s_obs | full theta) and draw
        # proportionally (capped at int(ESS); more just duplicates). Chunk the flow eval to bound memory.
        x_obs = np.atleast_2d(self.s_obs_rep)
        log_p_chunks = []
        for i in range(0, theta_pool.shape[0], 100_000):
            t = theta_pool[i : i + 100_000]
            log_p_chunks.append(self.flow.log_likelihood(np.repeat(x_obs, t.shape[0], axis=0), t).cpu().numpy())
        log_p = np.concatenate(log_p_chunks)
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        p /= np.sum(p)
        ess = float(1.0 / np.sum(p**2))

        # Parameter-space ESS: a theta-point's noise replicas share the same weight, so the row ESS
        # above is inflated by the noise multiplicity. Aggregate the row weights to unique
        # (i_sobol, i_signal) points to report how many distinct (cosmo+astro) points back the band.
        _tp_key = pool_real_idx[:, 0].astype(np.int64) * (int(pool_real_idx[:, 1].max()) + 1) + pool_real_idx[:, 1]
        _, _inv = np.unique(_tp_key, return_inverse=True)
        w_theta = np.bincount(_inv, weights=p)
        ess_theta = float(1.0 / np.sum(w_theta**2))
        n_theta = int(w_theta.size)

        if k_top is None:
            # Importance sampling: the unbiased posterior predictive. Draw realizations proportional
            # to p (capped at int(ESS); drawing more just duplicates).
            n_draws = min(int(n_samples), max(1, int(ess)))
            picked = self.rng.choice(theta_pool.shape[0], size=n_draws, replace=True, p=p)
            sel_label = "importance"
        else:
            # Top-k: the k realizations most consistent with the obs (highest p(s_obs|theta)).
            # Deterministic, k distinct sims (smooth band even at tiny ESS) but BIASED — it shows the
            # spread of the best-fit sims, not the posterior-predictive width.
            k = min(int(k_top), theta_pool.shape[0])
            picked = np.argsort(log_p)[-k:]
            n_draws = int(k)
            sel_label = f"top-{k}"

        cls_picked = _to_pair_major(cls_pool[picked])
        cls_picked_raw = cls_picked.copy()
        n_unique = int(np.unique(picked).size)
        LOGGER.info(
            f"Cls PPD (sample_set={sample_set}, {sel_label}): {n_draws} draws, {n_unique} unique sims, "
            f"ESS={ess:.1f}/{theta_pool.shape[0]} rows, ESS_theta={ess_theta:.1f}/{n_theta} pts "
            f"(noise multiplicity ~{ess / ess_theta:.1f})."
        )
        if n_unique < 50:
            LOGGER.warning(
                f"Only {n_unique} unique sims back the {self.obs_probe_name} Cls PPD; the grid is sparse "
                "near this posterior. Try sample_set='all' for more astro samples — but a small ESS is "
                "honest when the data tightly constrains theta."
            )

        if apply_log:
            cls_picked = np.log(np.abs(cls_picked))

        x = np.arange(n_dims)

        # PPD bands: inner = `percentiles`, optional outer = `outer_percentiles`. Listed outer-first
        # (fainter, drawn first) so the wide band sits under the narrow one.
        band_qs = [outer_percentiles, percentiles] if outer_percentiles is not None else [percentiles]
        band_alphas = [0.15, 0.30] if outer_percentiles is not None else [0.30]
        band_labels = [f"PPD [{q[0]:g}, {q[1]:g}]%" for q in band_qs]

        def _bands(samples):
            return [(np.percentile(samples, q[0], axis=0), np.percentile(samples, q[1], axis=0)) for q in band_qs]

        mid = np.median(cls_picked, axis=0)
        bands_log = _bands(cls_picked)

        cls_picked_lcl = ell_flat * cls_picked_raw
        mid_lcl = np.median(cls_picked_lcl, axis=0)
        bands_lcl = _bands(cls_picked_lcl)
        cls_obs_lcl = ell_flat * cls_obs_raw

        mid_raw = np.median(cls_picked_raw, axis=0)
        ppd_std = np.std(cls_picked_raw, axis=0)
        sig_obs = (cls_obs_raw - mid_raw) / np.where(ppd_std > 0, ppd_std, np.nan)

        # Robust residual: percentile of obs within the PPD per bin (empirical PPD CDF at obs),
        # bounded in [0, 1] and immune to the cross-z zero-crossings that blow up a relative diff.
        obs_rank = np.mean(cls_picked_raw < cls_obs_raw[None, :], axis=0)

        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True, constrained_layout=True)
        ax = axes[0]

        # --- panel 0: log(|Cl|) or Cl with optional log y-scale ---
        for (blo, bhi), alpha, lab in zip(bands_log, band_alphas, band_labels):
            ax.fill_between(x, blo, bhi, alpha=alpha, color="tab:orange", label=lab)
        ax.plot(x, mid, color="tab:orange", lw=1.5, label="PPD median")
        if n_traces > 0:
            ax.plot(x, cls_picked[: min(n_traces, cls_picked.shape[0])].T, color="tab:orange", alpha=0.2, lw=0.5)
        ax.plot(x, cls_obs, color="k", lw=1.5, label=f"obs ({self.obs_label})")

        if log_y:
            if np.any(cls_picked <= 0) or np.any(cls_obs <= 0):
                LOGGER.warning("Non-positive values in cls_picked / cls_obs; using linear y-scale.")
            else:
                ax.set_yscale("log")

        title = (
            f"{self.obs_label}: rebinned Cls PPD — {self._setup_descriptor()}, scales={scales_name}\n"
            f"samples={sample_set}, {sel_label}: {n_draws} draws, {n_unique} uniq, "
            f"ESS={ess:.1f} (ESS$_\\theta$={ess_theta:.1f}/{n_theta} pts)"
        )
        ax.set(ylabel=r"$\log C_\ell$" if apply_log else r"$C_\ell$", title=title)
        ax.yaxis.set_ticklabels([])
        ax.legend(fontsize=8)

        # --- panel 1: ℓ·Cl ---
        for (blo, bhi), alpha in zip(bands_lcl, band_alphas):
            axes[1].fill_between(x, blo, bhi, alpha=alpha, color="tab:orange")
        axes[1].plot(x, mid_lcl, color="tab:orange", lw=1.5)
        if n_traces > 0:
            axes[1].plot(
                x,
                (ell_flat * cls_picked_raw[: min(n_traces, cls_picked_raw.shape[0])]).T,
                color="tab:orange",
                alpha=0.2,
                lw=0.5,
            )
        axes[1].plot(x, cls_obs_lcl, color="k", lw=1.5)
        axes[1].set(ylabel=r"$\ell \, C_\ell$")
        axes[1].yaxis.set_ticklabels([])

        # --- panel 2: obs percentile within the PPD (robust residual) ---
        # 0.5 = obs at the PPD median; near 0/1 = obs in the PPD tail. The shaded band marks the
        # central 68% and the dashed lines the central 95%, so a point leaving them flags tension.
        axes[2].axhspan(0.16, 0.84, color="tab:orange", alpha=0.12, label="central 68%")
        axes[2].plot(x, obs_rank, color="tab:orange", lw=1.2)
        axes[2].axhline(0.5, color="k", lw=1.0)
        for q in (0.16, 0.84):
            axes[2].axhline(q, color="tab:orange", lw=0.6, ls=":")
        for q in (0.025, 0.975):
            axes[2].axhline(q, color="tab:red", lw=0.8, ls="--")
        axes[2].set(ylabel=r"obs percentile in PPD", ylim=(0, 1))

        # --- panel 3: significance / pull ---
        axes[3].plot(x, sig_obs, color="tab:orange", lw=1.0)
        axes[3].axhline(0, color="k", lw=1.2)
        axes[3].axhline(2, color="tab:red", lw=0.8, ls="--")
        axes[3].axhline(-2, color="tab:red", lw=0.8, ls="--")
        axes[3].axhline(1, color="tab:orange", lw=0.6, ls=":")
        axes[3].axhline(-1, color="tab:orange", lw=0.6, ls=":")
        axes[3].set(
            ylabel=r"$(C_\ell^\mathrm{obs} - \mathrm{med}_\mathrm{PPD})\,/\,\sigma_\mathrm{PPD}$", ylim=(-3, 3)
        )

        # --- segment boundaries ---
        for i_pair in range(1, n_pairs):
            xb = i_pair * n_ell_per_pair
            for a in axes:
                a.axvline(xb, color="gray", lw=0.6, ls="--", alpha=0.5)

        axes[3].set_xlabel(x_label)

        # --- pair labels (tomographic pair + its ell range) via a secondary top axis ---
        _tick_labels = [f"{lab}\n[{int(lo)}–{int(hi)}]" for lab, (lo, hi) in zip(_pair_labels, ell_ranges)]
        _ax_top = axes[0].twiny()
        _ax_top.set_xlim(0, n_dims)
        _ax_top.set_xticks([(k + 0.5) * n_ell_per_pair for k in range(n_pairs)])
        _ax_top.set_xticklabels(_tick_labels, fontsize=8, color="dimgray")
        _ax_top.tick_params(axis="x", which="both", length=0, pad=2)
        _ax_top.spines["top"].set_visible(False)
        _ax_top.yaxis.set_visible(False)

        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_{file_label}_marginals.png")
        LOGGER.info(f"Saving Cls marginals plot to {plot_file}")
        fig.savefig(plot_file, bbox_inches="tight", dpi=100)
        plt.close(fig)
        return plot_file

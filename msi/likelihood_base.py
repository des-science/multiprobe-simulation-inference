# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen

Wrapper around enflows to build a likelihood normalizing flow with training and sampling utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

from msi.utils import plotting, diagnostics
from msfm.utils import logger, prior, parameters

# NOTE: torch / msi.utils.torch_ensemble are imported lazily inside the batched-sampling methods below.
# LikelihoodBase is also subclassed by the TensorFlow GMM backend (LikelihoodGMM), which runs in an
# environment that need not have torch installed -- so this module must stay importable without torch.

LOGGER = logger.get_logger(__file__)


class LikelihoodBase(ABC):
    @abstractmethod
    def __init__(self, params, conf=None, out_dir=None, label=None, load_existing=False):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def sample_likelihood(self, theta_obs, n_samples, batch_size, return_numpy):
        pass

    @abstractmethod
    def log_likelihood(self, x, theta, return_numpy):
        pass

    @abstractmethod
    def sample_posterior(self, x_obs, n_samples, n_walkers, n_burnin_steps, label, device):
        pass

    @abstractmethod
    def _mcmc_log_posterior(self, theta_walkers, x_obs):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    # batched posterior (GPU) #########################################################################################

    def sample_posterior_batched(
        self,
        x_obs_batch,
        n_walkers=1_024,
        n_steps=1_000,
        n_burnin_steps=1_000,
        lambdaCDM=False,
        device=None,
        seed=12,
        use_validation_weights=True,
        compile_flow=True,
        method="ensemble",
    ):
        """Sample the posterior for a batch of observations at once with the GPU-batched torch ensemble
        sampler (msi.utils.torch_ensemble). The throughput-oriented counterpart of sample_posterior (which
        wraps emcee for a single observation): it runs the same Goodman & Weare stretch move but evaluates
        all observations in one batched forward pass per step.

        This driver is shared by LikelihoodFlow and LikelihoodFlowEnsemble; subclasses only provide
        _batched_log_likelihood_torch (the model's batched log p(x|theta)), _set_eval_device, and
        _get_ensemble_weights.

        Args:
            x_obs_batch (np.ndarray or torch.Tensor): Observations of shape (n_obs, n_features).
            n_walkers (int, optional): Number of walkers per chain (must be even). Defaults to 1024.
            n_steps (int, optional): Number of main-chain steps. Defaults to 1000.
            n_burnin_steps (int, optional): Number of burn-in steps. Defaults to 1000.
            lambdaCDM (bool, optional): If True, fix w0 = -1 and sample the reduced (w0-dropped) parameter
                space, mirroring the emcee sample_posterior's lambdaCDM mode. Defaults to False.
            device (str, optional): Device to run on. Defaults to None (self.device).
            seed (int, optional): RNG seed for reproducibility. Defaults to 12 (matching mcmc.py).
            use_validation_weights (bool, optional): For an ensemble, weight members by validation
                performance (no-op for a single flow). Defaults to True.
            compile_flow (bool, optional): On CUDA, allow Inductor-compiling the batched log-likelihood
                (``torch.compile``) to fuse the flow's many small pointwise kernels in the tight MCMC loop.
                Compiled and eager are timed on a few steps and the faster is used, so flows that do not
                benefit (or that Dynamo cannot trace) transparently stay eager. Set False to force eager
                (e.g. for debugging). Defaults to True.
            method (str, optional): Ensemble-only knob, ignored by a single flow (which always samples
                itself). LikelihoodFlowEnsemble overrides this method to interpret "ensemble" (one chain on
                the combined likelihood) vs "individual" (per-member chains, then pooled). Defaults to
                "ensemble".

        Returns:
            tuple(np.ndarray, np.ndarray): chain of shape (n_obs, n_steps * n_walkers, n_params) and its
                log-posterior of shape (n_obs, n_steps * n_walkers), as numpy arrays on the host. For
                lambdaCDM the chain is in the reduced (w0-dropped) space.
        """
        import torch
        from msi.utils import torch_ensemble

        if device is None:
            device = self.device

        x_obs_batch = torch.as_tensor(x_obs_batch, dtype=self.floatx, device=device)
        x_obs_batch = torch.atleast_2d(x_obs_batch)
        n_obs = x_obs_batch.shape[0]

        # the sampler walks the reduced (w0-dropped) space under lambdaCDM; w0 = -1 is reinserted only
        # to evaluate the model and the full-parameter prior, exactly as the emcee path does
        if lambdaCDM:
            i_w = self.params.index("w0")
            params_sample = [p for p in self.params if p != "w0"]
        else:
            i_w = None
            params_sample = self.params
        n_params = len(params_sample)

        self._set_eval_device(device)
        weights = self._get_ensemble_weights(use_validation_weights, device)

        # the prior is always evaluated in the full parameter space (after reinserting w0 if lambdaCDM)
        prior_data = prior.get_torch_prior_data(self.params, conf=self.conf, device=device, floatx=self.floatx)

        # initial walker positions: same recipe as mcmc.run_emcee, replicated for every observation.
        # Factored into a closure so the eager-fallback retry below can reproduce the exact same seeded
        # generator and initial walkers, making the fallback chain bit-for-bit what a non-compiled run gives.
        fiducials = torch.as_tensor(
            parameters.get_fiducials(params_sample, conf=self.conf), dtype=self.floatx, device=device
        )

        def make_initial_state():
            gen = torch.Generator(device=device).manual_seed(seed)
            theta0 = fiducials + 1e-3 * torch.randn(
                (n_obs, n_walkers, n_params), dtype=self.floatx, device=device, generator=gen
            )
            return gen, theta0

        def make_log_prob_fn(llf):
            def log_prob_fn(theta):
                if lambdaCDM:
                    # reinsert the fixed w0 = -1 column at its original index before the model/prior eval
                    w0_col = torch.full((*theta.shape[:-1], 1), -1.0, dtype=theta.dtype, device=theta.device)
                    theta = torch.cat([theta[..., :i_w], w0_col, theta[..., i_w:]], dim=-1)
                return self._batched_log_posterior_torch(
                    theta, x_obs_batch, prior_data, weights=weights, loglike_fn=llf
                )

            return log_prob_fn

        # On CUDA, optionally Inductor-compile the flow's batched log-likelihood. The spline/maf log_prob is a
        # long chain of small pointwise / searchsorted kernels whose per-launch overhead dominates the tight,
        # static-shape MCMC loop, so Inductor fusion is a large win there. But it is *not* a win for every
        # flow, so we pick between compiled and eager by a quick timed comparison (below) rather than assuming:
        #   - "reduce-overhead" (Inductor + CUDA graphs) is deliberately not used: graph capture fails for the
        #     coupling/affine flows ("storage data ptrs not allocated in pool"), and for the autograd-tracing
        #     lipschitz iResBlock log-det it captures but runs several times slower than eager.
        #   - plain Inductor (no CUDA graphs) is the candidate: big speedup for spline/maf, but for flows
        #     Dynamo cannot trace (sigmoid's no_analytic_inv SumOfSigmoids) or where fusion does not pay off
        #     (lipschitz) it either errors or is slower -- so we only keep it if it actually times faster.
        # donated_buffer must be disabled or compiling the autograd-tracing log-dets raises a donated-buffer
        # error; it is a no-op for the flows whose log_prob does not build a backward graph.
        use_compiled = compile_flow and "cuda" in str(device)
        loglike_fn = self._batched_log_likelihood_torch
        if use_compiled:
            import time
            import torch._functorch.config as _functorch_config

            _functorch_config.donated_buffer = False

            def _per_step_seconds(llf, n_warmup=3, n_timed=10):
                # representative per-evaluation wall time at the real sampling shape; warm-up triggers the
                # one-time compile / autograd-graph build so the timed loop measures steady state
                log_prob_fn = make_log_prob_fn(llf)
                _, theta = make_initial_state()
                for _ in range(n_warmup):
                    log_prob_fn(theta)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                for _ in range(n_timed):
                    log_prob_fn(theta)
                torch.cuda.synchronize(device)
                return (time.perf_counter() - t0) / n_timed

            try:
                compiled_fn = torch.compile(self._batched_log_likelihood_torch)
                t_compiled = _per_step_seconds(compiled_fn)
                t_eager = _per_step_seconds(self._batched_log_likelihood_torch)
                if t_compiled < t_eager:
                    loglike_fn = compiled_fn
                LOGGER.info(
                    f"[mcmc] log_prob per step: inductor {t_compiled * 1e3:.1f} ms vs eager "
                    f"{t_eager * 1e3:.1f} ms -> using {'inductor' if loglike_fn is compiled_fn else 'eager'}"
                )
            except Exception as e:
                # Dynamo cannot trace some transforms (sigmoid); fall back to eager and carry on
                LOGGER.warning(f"Inductor compilation unavailable ({type(e).__name__}: {e}); using eager")
                torch.compiler.reset()
                loglike_fn = self._batched_log_likelihood_torch

        generator, theta_0 = make_initial_state()
        try:
            chain, log_prob = torch_ensemble.run_ensemble_torch(
                make_log_prob_fn(loglike_fn),
                theta_0,
                n_steps=n_steps,
                n_burnin_steps=n_burnin_steps,
                generator=generator,
            )
        except Exception as e:
            # a compiled path that benchmarked fine can still error when it recompiles for the half-ensemble
            # shape mid-run; fall back to eager once. A genuine eager error (loglike_fn already eager) re-raises.
            if loglike_fn is self._batched_log_likelihood_torch:
                raise
            LOGGER.warning(f"Compiled MCMC sampling failed mid-run ({type(e).__name__}: {e}); retrying eager")
            torch.compiler.reset()
            generator, theta_0 = make_initial_state()
            chain, log_prob = torch_ensemble.run_ensemble_torch(
                make_log_prob_fn(self._batched_log_likelihood_torch),
                theta_0,
                n_steps=n_steps,
                n_burnin_steps=n_burnin_steps,
                generator=generator,
            )

        # restore the model to its original device
        self._set_eval_device(self.device)

        return chain, log_prob

    def _batched_log_posterior_torch(self, theta, x_obs, prior_data, weights=None, loglike_fn=None):
        """On-device batched log-posterior: the subclass's batched log-likelihood plus the hard top-hat
        prior (applied once here). theta is (n_obs, n_walkers, n_params); returns (n_obs, n_walkers) with
        -inf outside the flat analysis prior.

        ``loglike_fn`` lets the caller inject an alternative log-likelihood callable (e.g. a
        ``torch.compile``-wrapped version of ``_batched_log_likelihood_torch``); defaults to the eager
        method."""
        import torch

        if loglike_fn is None:
            loglike_fn = self._batched_log_likelihood_torch

        with torch.no_grad():
            log_like = loglike_fn(theta, x_obs, weights=weights)
            in_prior = prior.in_grid_prior_torch(theta, prior_data)
            log_post = torch.where(in_prior, log_like, torch.full_like(log_like, float("-inf")))
        return log_post

    def _set_eval_device(self, device):
        """Move the model(s) to ``device`` and switch to eval mode. Default works for an nn.Module-based
        single flow; the ensemble overrides it to loop over its members."""
        self.to(device)
        self.eval()

    def _get_ensemble_weights(self, use_validation_weights, device):
        """Per-member ensemble weights (or None for uniform / a single flow). Overridden by the ensemble."""
        return None

    # plotting ########################################################################################################

    def plot_contours(
        self,
        posterior_samples,
        # cosmetics
        scale_to_prior=True,
        group_params=True,
        density=False,
        # cosmo
        obs_point=None,
        obs_label="synthetic observation",
        with_des_chain=False,
        lambdaCDM=False,
        # output
        label=None,
    ):
        """
        Plot contours of the posterior samples.

        Args:
            samples (array-like): Samples from the posterior distribution.
            scale_to_prior (bool, optional): Whether to scale the plot to the prior distribution. Defaults to True.
            group_params (bool, optional): Whether to group cosmological and astrophysical parameters in the plot.
                Defaults to True.
            plot_fiducial (bool, optional): Whether to include the fiducial point in the plot. Defaults to True.
            fiducial_point (array-like, optional): Fiducial point to plot. Defaults to None.
            with_des_chain (bool, optional): Whether to include the DES chain in the plot. Defaults to False.
            label (str, optional): Additional label for the saved chain, for example to designate different
                observations. Defaults to None.
        """

        if lambdaCDM:
            label += "_lambdaCDM"
            params = [p for p in self.params if p != "w0"]
        else:
            params = self.params

        plotting.plot_chains(
            posterior_samples,
            params,
            self.conf,
            # file
            out_dir=self.model_dir,
            file_label=label,
            # cosmetics
            plot_labels=self.label,
            scale_to_prior=scale_to_prior,
            group_params=group_params,
            density=density,
            # cosmology
            obs_cosmo=obs_point,
            obs_label=obs_label,
            with_des_chain=with_des_chain,
        )

    def plot_diagnostics(
        self,
        grid_preds_true,
        grid_cosmos,
        # sampling
        n_cosmos=None,
        n_samples=100,
        batch_size=10000,
        # flags
        do_hist=False,
        do_dlss=False,
        do_eecp=True,
        do_tarp=True,
        tarp_kwargs=None,
        # output
        out_dir=None,
        prefix="",
    ):
        """
        Plot diagnostics of how well the likelihood p(x|theta) has been learned from the (samples of the) true
        distribution.

        Args:
            grid_preds_true (ndarray): Array of shape (n_cosmos, n_examples, n_summary) or (n_cosmos, n_summary) true
                predictions for each cosmology in the grid. These are used as the true baseline to compare to.
            grid_cosmos (ndarray): Array of shape (n_cosmos, n_params) of the cosmologies in the grid. This is used
                to condition the flow and sample from it.
            n_cosmos (int, optional): Number of cosmologies to select randomly from the grid. Defaults to None, then
                all cosmologies are used.
            n_samples (int, optional): Number of samples per cosmology. Defaults to 100.
            batch_size (int, optional): Batch size for sampling. Defaults to 4096.

        Returns:
            ndarray: Array of shape (n_cosmos, n_samples, n_summary) containing samples from the likelihood
            for the whole grid.
        """

        if tarp_kwargs is None:
            tarp_kwargs = {"n_bootstrap": 100, "n_alpha_bins": 20}

        assert grid_preds_true.shape[0] == grid_cosmos.shape[0], "n_cosmos must be the same for both arrays"
        assert grid_cosmos.ndim == 2, "grid_cosmos must have 2 dims containing (n_cosmos, n_params)"

        if out_dir is None:
            out_dir = self.model_dir
        os.makedirs(out_dir, exist_ok=True)

        if grid_preds_true.ndim == 2:
            LOGGER.warning(
                f"grid_preds_true.shape = {grid_preds_true.shape}, for sobol sequence + latin hypercube sampling"
            )
        elif grid_preds_true.ndim == 3:
            LOGGER.warning(f"grid_preds_true.shape = {grid_preds_true.shape}, for sobol sequence")
        else:
            raise ValueError(f"grid_preds_true.ndim = {grid_preds_true.ndim} not supported")

        if n_cosmos is not None:
            LOGGER.info(f"Selecting {n_cosmos} random cosmologies")
            random_indices = np.random.choice(grid_preds_true.shape[0], n_cosmos, replace=False)
            grid_preds_true = grid_preds_true[random_indices]
            grid_cosmos = grid_cosmos[random_indices]

        LOGGER.timer.start("sampling")
        LOGGER.info(f"Drawing samples from the likelihood")
        grid_preds_sample = self.sample_likelihood(
            grid_cosmos, n_samples=n_samples, batch_size=batch_size, return_numpy=True
        )
        LOGGER.info(f"Done drawing samples after {LOGGER.timer.elapsed('sampling')}")

        if do_hist:
            assert (
                grid_preds_true.ndim == 3
            ), "grid_preds_true must have 3 dims containing (n_cosmos, n_samples, n_summaries)"
            diagnostics.plot_histogram_check(
                grid_preds_true, grid_preds_sample, n_random_indices=10, out_dir=out_dir, prefix=prefix
            )
        if do_dlss:
            diagnostics.plot_deeplss_check(grid_preds_true, grid_preds_sample, out_dir=out_dir, prefix=prefix)
        if do_eecp:
            diagnostics.plot_eecp_check(
                grid_preds_true, grid_preds_sample, grid_cosmos, self, out_dir=out_dir, prefix=prefix
            )
        if do_tarp:
            diagnostics.plot_tarp_check(
                grid_preds_true, grid_preds_sample, grid_cosmos, out_dir=out_dir, prefix=prefix, **tarp_kwargs
            )

        # (n_cosmos, n_samples, n_summary)
        return grid_preds_sample, grid_preds_true, grid_cosmos

    def _plot_epochs(self, train_losses, vali_losses):
        """Produce a diagnostics plot of the loss curves after training has finished"""

        all_losses = np.concatenate([train_losses, vali_losses])

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(train_losses, label="training")
        ax.plot(vali_losses, label="validation")
        ax.set(
            xlabel="epoch", ylabel="loss", ylim=(np.nanquantile(all_losses, 0.01), np.nanquantile(all_losses, 0.99))
        )
        ax.grid(True)
        ax.legend()

        if self.model_dir is not None:
            fig.savefig(os.path.join(self.model_dir, "loss_curves.png"))

    # utils ###########################################################################################################

    def _setup_dirs(self, file_type):
        if self.model_dir is not None:
            self.model_file = os.path.join(self.model_dir, self.model_name + file_type)
        elif self.out_dir is not None and self.model_dir is None:
            if self.label is None:
                self.model_dir = os.path.join(self.out_dir, self.prefix + self.model_name + self.suffix)
            else:
                self.model_dir = os.path.join(self.out_dir, self.label, self.prefix + self.model_name + self.suffix)
            os.makedirs(self.model_dir, exist_ok=True)
            LOGGER.info(f"Set up the model directory {self.model_dir}")
            self.model_file = os.path.join(self.model_dir, self.model_name + file_type)
        else:
            self.model_dir = None
            self.model_file = None

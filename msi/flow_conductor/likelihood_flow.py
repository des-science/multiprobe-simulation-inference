# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen

Wrapper around enflows to build a likelihood normalizing flow with training and sampling utilities.
"""

import os
import numpy as np

import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split

from enflows.flows import Flow

from msi.likelihood_base import LikelihoodBase
from msi.utils import mcmc
from msi.flow_conductor import architecture
from msfm.utils import logger, files, prior

from msi.flow_conductor.pytorch import EarlyStopper, get_lr

LOGGER = logger.get_logger(__file__)


class _EnsembleLogProb(torch.nn.Module):
    """Thin adapter so torch.func.functional_call's forward dispatches to a member flow's ``log_prob``.

    Used only by LikelihoodFlowEnsemble's vmap path: stacking the members' parameters and vmapping this
    wrapper evaluates every member's batched log p(x|theta) in a single fused pass instead of a Python
    loop over members -- the win for large ensembles of small flows, where per-call launch overhead
    (not FLOPs) dominates the tight MCMC loop.
    """

    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, inputs, context):
        return self.flow.log_prob(inputs=inputs, context=context)


def _pool_chains(member_chains, weights=None, member_log_probs=None, rng=None):
    """Pool per-member posterior chains into one chain by drawing from the mixture sum_i w_i p_i(theta|x):
    member i contributes a fraction w_i of the output rows, selected at random from its own chain. With the
    ensemble weights this reproduces the (weighted) likelihood-level ensemble posterior -- exact under
    uniform weights, where it is an even split. The output keeps the same number of samples as one member,
    so it is shape-compatible with the 'ensemble' path's chain.

    Two layouts are supported, distinguished by ndim:
      - single observation:   chain (n_samples, n_params),        log_probs (n_samples,)
      - batched observations:  chain (n_obs, n_samples, n_params), log_probs (n_obs, n_samples)
    Returns the pooled chain, or (chain, log_probs) when member_log_probs is given.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_members = len(member_chains)
    first = np.asarray(member_chains[0])
    sample_axis = 1 if first.ndim == 3 else 0  # batched obs put the sample axis second
    n_samples = first.shape[sample_axis]

    if weights is None:
        weights = np.full(n_members, 1.0 / n_members)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    # integer per-member sample counts summing exactly to n_samples (largest-remainder rounding)
    counts = np.floor(weights * n_samples).astype(int)
    frac = weights * n_samples - counts
    for k in np.argsort(-frac)[: n_samples - int(counts.sum())]:
        counts[k] += 1

    pooled, pooled_lp = [], []
    for i, chain in enumerate(member_chains):
        chain = np.asarray(chain)
        s_i = chain.shape[sample_axis]
        sel = rng.choice(s_i, size=int(counts[i]), replace=counts[i] > s_i)
        pooled.append(np.take(chain, sel, axis=sample_axis))
        if member_log_probs is not None:
            pooled_lp.append(np.take(np.asarray(member_log_probs[i]), sel, axis=sample_axis))

    perm = rng.permutation(n_samples)  # interleave members so the pooled chain is not block-ordered
    chain_out = np.take(np.concatenate(pooled, axis=sample_axis), perm, axis=sample_axis)
    if member_log_probs is not None:
        return chain_out, np.take(np.concatenate(pooled_lp, axis=sample_axis), perm, axis=sample_axis)
    return chain_out


class LikelihoodFlow(Flow, LikelihoodBase):
    """Normalizing flow implementing a likelihood function p(x|theta), where x is some summary statistic vector and
    theta a vector of cosmological/astrophysical parameters to be constrained.

    The main purpose of the class is to wrap the FlowConductor library and provide a convenient interface for training
    and MCMC sampling from the posterior p(theta|x_obs), where x_obs is a summary corresponding to a (mock)
    observation.
    """

    model_name = "likelihood_flow"

    def __init__(
        self,
        params,
        conf=None,
        # output
        out_dir=None,
        model_dir=None,
        prefix="",
        suffix="",
        label=None,
        load_existing=True,
        # architecture
        feature_dim=None,
        embedding_net=None,
        base_dist=None,
        transform=None,
        # computational
        device=None,
        floatx=torch.float32,
        torch_seed=7,
    ):
        """
        Initialize the LikelihoodFlow object.

        Args:
            params (list): The cosmological and astrophysical parameters to be constrained. Note that the default
                architecture makes the assumption that the summary statistic has the same dimensionality as the number
                of parameters.
            conf (str, optional): The configuration file path. Defaults to None, then the default is loaded.
            out_dir (str, optional): The output directory path. Defaults to None, then no output is saved.
            label (str, optional): The label used in the saved filenames. Defaults to None.
            load_existing (bool, optional): Whether to load a model from disk if it exists. Defaults to True.
            embedding_net (nn.Module, optional): The context embedding network, taking in the theta. Defaults to None,
                then the default is loaded.
            base_dist (torch.distributions.Distribution, optional): The base distribution of the flow. Defaults to
                None, then the default is loaded.
            transform (nn.Module, optional): The transformation function of the flow. Defaults to None, then the
                default is loaded.
            device (str, optional): The device to evaluate the flow on. Defaults to None, then CUDA is used when
                available and otherwise the CPU.
            floatx (torch.dtype, optional): The default float type. Defaults to torch.float32.
        """

        self._init_kwargs = {
            "params": params,
            "conf": conf,
            "out_dir": out_dir,
            "model_dir": model_dir,
            "prefix": prefix,
            "suffix": suffix,
            "label": label,
            "feature_dim": feature_dim,
            "embedding_net": embedding_net,
            "base_dist": base_dist,
            "transform": transform,
            "device": device,
            "floatx": floatx,
            "torch_seed": torch_seed,
        }

        self.params = params
        self.conf = files.load_config(conf)

        self.out_dir = out_dir
        self.model_dir = model_dir
        self.prefix = prefix
        self.suffix = suffix
        self.label = label
        self._setup_dirs(".pt")

        # assume the summary statistic has the same dimension as the constrained parameters
        context_dim = len(params)

        # default architecture
        if embedding_net is None:
            embedding_net = architecture.get_context_embedding_net(context_dim)
            LOGGER.info(f"Using the default context embedding network:")
            LOGGER.info(type(embedding_net))
        if base_dist is None:
            base_dist = architecture.get_normal_dist(feature_dim)
            LOGGER.info(f"Using the default base distribution:")
            LOGGER.info(type(base_dist))
        if transform is None:
            transform = architecture.get_sigmoids_transform(feature_dim)
            LOGGER.info(f"Using the default transform:")
            LOGGER.info(type(transform))

        super(LikelihoodFlow, self).__init__(transform, base_dist, embedding_net=embedding_net)
        LOGGER.info(f"Initialized the normalizing flow")

        # device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.floatx = floatx
        self.torch_seed = torch_seed
        self.to(self.device)
        LOGGER.info(f"Running on device {self.device} with default float {self.floatx}")

        if load_existing:
            try:
                self.load()
            except FileNotFoundError:
                LOGGER.warning(f"Could not load the model from {self.model_file}")
        else:
            LOGGER.info(f"Initializing fresh weights")

    # training ########################################################################################################

    def fit(
        self,
        x,
        theta,
        n_epochs=100,
        batch_size=1000,
        vali_split=0.1,
        # optimizer
        learning_rate=1e-3,
        weight_decay=0.0,
        clip_by_global_norm=1.0,
        # learning rate scheduler
        scheduler_type=None,
        scheduler_kwargs=None,
        # early stopping
        n_patience_epochs=None,
        min_delta=1e-4,
        save_model=True,
        seed=None,
        group_ids=None,
        run_c2st=False,
        c2st_hidden_dim=64,
        c2st_n_epochs=50,
    ):
        """
        Fits the likelihood flow model to the given data and saves the resulting model.

        Args:
            x (torch.Tensor): The input features (summary statistics).
            theta (torch.Tensor): The input context (cosmological parameters).
            n_epochs (int, optional): The number of epochs to train for. Defaults to 100.
            batch_size (int, optional): The batch size for training and validation. Defaults to 1024.
            vali_split (float, optional): The validation split ratio. The validation set is used for early
                stopping. Defaults to 0.1.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.0.
            clip_by_global_norm (float, optional): The maximum gradient norm for gradient clipping. Defaults to
                100.0. When None, no clipping is applied.
            scheduler_type (str, optional): The type of learning rate scheduler to use. One of "plateau", "cosine" or
                None Defaults to None.
            scheduler_kwargs (dict, optional): Additional keyword arguments for the learning rate scheduler, which
                overwrite the defaults hardcoded in the function.
            n_patience_epochs (int, optional): The number of epochs to wait before early stopping. Defaults to 10.
            min_delta (float, optional): The minimum change in validation loss to consider as improvement for
                early stopping. Defaults to 0.05.
            save_model (bool, optional): Whether to save the model after training. Defaults to True.
            seed (int, optional): The seed for the random data split. Defaults to None, then self.torch_seed is used.
                Ignored when `group_ids` is given.
            group_ids (numpy.ndarray, optional): 1D array aligned row-for-row with `x`/`theta` (e.g. `i_signal`)
                used to make the train/validation split deterministic and group-aware -- see `_prepare_data`.
            run_c2st (bool, optional): Whether to run a Classifier Two-Sample Test on the validation set after
                training. Trains a small MLP to distinguish real validation pairs (x, theta) from flow-generated
                pairs (x_gen, theta). An accuracy close to 0.5 indicates the flow has learned the conditional
                distribution well. Defaults to False.
            c2st_hidden_dim (int, optional): Hidden layer size for the C2ST classifier MLP. Defaults to 64.
            c2st_n_epochs (int, optional): Number of epochs to train the C2ST classifier. Defaults to 50.
        """

        # copy to avoid mutating a shared default dict via setdefault below
        scheduler_kwargs = {} if scheduler_kwargs is None else dict(scheduler_kwargs)

        n_examples = x.shape[0]
        LOGGER.info(f"batch size = {batch_size} -> {n_examples // batch_size} steps per epoch for {n_epochs} epochs")

        self._prepare_data(x, theta, batch_size, vali_split, seed=seed, group_ids=group_ids)

        # optimizer
        self.clip_by_global_norm = clip_by_global_norm
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # learning rate scheduler
        if scheduler_type is None:
            LOGGER.info(f"Not using a learning rate scheduler")
        elif scheduler_type == "cosine":
            scheduler_kwargs.setdefault("eta_min", 1e-5)
            scheduler_kwargs.setdefault("T_max", n_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_kwargs)
            LOGGER.info(
                f"Using a cosine annealing scheduler with lr_min {scheduler_kwargs['eta_min']} and T_max {scheduler_kwargs['T_max']}"
            )
        elif scheduler_type == "exp":
            scheduler_kwargs.setdefault("gamma", 0.95)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, **scheduler_kwargs)
            LOGGER.info(
                f"Using an exponential decay scheduler with gamma {scheduler_kwargs['gamma']} resulting in "
                f"eta_min {(learning_rate*scheduler_kwargs['gamma']**n_epochs):.2E}"
            )
        elif scheduler_type == "plateau":
            LOGGER.info(f"Using a ReduceLROnPlateau scheduler")
            scheduler_kwargs.setdefault("min_lr", 1e-5)
            scheduler_kwargs.setdefault("mode", "min")
            scheduler_kwargs.setdefault("factor", 0.5)
            scheduler_kwargs.setdefault("patience", 4)
            scheduler_kwargs.setdefault("threshold", 1e-4)
            scheduler_kwargs.setdefault("threshold_mode", "rel")
            scheduler_kwargs.setdefault("cooldown", 1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler type {scheduler_type}")

        # early stopping
        if n_patience_epochs is not None:
            LOGGER.info(f"Using early stopping with patience {n_patience_epochs} and min delta {min_delta}")
            early_stopper = EarlyStopper(patience=n_patience_epochs, min_delta=min_delta, model=self)

        train_losses = []
        vali_losses = []
        pbar = LOGGER.progressbar(range(n_epochs), at_level="info", total=n_epochs)
        for i_epoch in pbar:
            train_loss = self._train_epoch()
            vali_loss = self._vali_epoch()

            if scheduler_type == "plateau":
                scheduler.step(vali_loss)
            elif scheduler_type in ["cosine", "exp"]:
                scheduler.step()

            if n_patience_epochs is not None and early_stopper.early_stop(vali_loss):
                LOGGER.info(f"Stopping early after {i_epoch} epochs")
                break

            pbar.set_description(f"lr: {get_lr(self.optimizer):.2E}, train: {train_loss:.2f}, vali: {vali_loss:.2f}")
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)

        self._plot_epochs(train_losses, vali_losses)
        if save_model:
            self.save()

        if run_c2st:
            # the model is already trained and saved above, so never let this diagnostic abort the run
            try:
                c2st_acc = self._run_c2st(hidden_dim=c2st_hidden_dim, n_epochs=c2st_n_epochs)
                LOGGER.info(f"C2ST accuracy: {c2st_acc:.4f} (ideal: 0.5, worst: 1.0)")
            except Exception as e:
                LOGGER.warning(f"C2ST failed ({type(e).__name__}: {e}); skipping (model already saved).")
                c2st_acc = float("nan")
            return {"train_loss": train_losses, "vali_loss": vali_losses, "c2st_accuracy": c2st_acc}

        return {"train_loss": train_losses, "vali_loss": vali_losses}

    def _run_c2st(self, n_epochs=50, hidden_dim=64, batch_size=256, test_fraction=0.3):
        """
        Conditional Classifier Two-Sample Test (C2ST).

        Trains a small binary MLP classifier to distinguish real validation pairs
        (x, theta) from flow-generated pairs (x_gen ~ p(x|theta), theta). The
        classifier receives the concatenation [x, theta] as input, enabling it
        to detect conditional distribution mismatches.

        An accuracy close to 0.5 indicates the flow has learned the conditional
        distribution well; an accuracy close to 1.0 indicates a poor fit.

        Requires that _prepare_data has been called (i.e. fit has been called
        at least once), so that self.vali_loader is available.

        Args:
            n_epochs (int, optional): Epochs to train the classifier. Defaults to 50.
            hidden_dim (int, optional): Hidden layer size of the classifier MLP. Defaults to 64.
            batch_size (int, optional): Batch size for classifier training. Defaults to 256.
            test_fraction (float, optional): Fraction held out as classifier test set.
                Defaults to 0.3.

        Returns:
            float: Classifier test-set accuracy (ideal: 0.5, worst: 1.0).
        """
        # Collect full validation set
        x_real_list, theta_list = [], []
        for x_batch, theta_batch in self.vali_loader:
            x_real_list.append(x_batch)
            theta_list.append(theta_batch)
        x_real = torch.cat(x_real_list, dim=0)  # (n, x_dim)
        theta_vali = torch.cat(theta_list, dim=0)  # (n, theta_dim)

        n_real = len(x_real)
        LOGGER.info(f"Running conditional C2ST with {n_real} real vs {n_real} flow samples ...")

        # Generate samples from the flow: sample(1, context=theta) -> (n, 1, x_dim) -> (n, x_dim)
        self.eval()
        with torch.no_grad():
            x_gen = self.sample(1, context=theta_vali).squeeze(1)  # (n, x_dim)

        # Drop non-finite generated samples. Flows with unbounded tails (e.g. the affine MAF, unlike
        # the linear-tailed RQ spline) can occasionally sample extreme/non-finite x; left in, these
        # poison the classifier (NaN predictions). Drop the same rows from the real set so the two
        # classes keep matched theta (this is a conditional C2ST) and balanced counts.
        finite_mask = torch.isfinite(x_gen).all(dim=1)
        n_finite = int(finite_mask.sum())
        if n_finite < n_real:
            LOGGER.warning(f"C2ST: dropping {n_real - n_finite}/{n_real} non-finite generated samples")
        if n_finite < 2:
            LOGGER.warning("C2ST: too few finite generated samples; skipping C2ST")
            return float("nan")
        x_real, x_gen, theta_c2st = x_real[finite_mask], x_gen[finite_mask], theta_vali[finite_mask]

        # Labels: 1 = real, 0 = generated
        labels_real = torch.ones(n_finite, 1, dtype=self.floatx, device=self.device)
        labels_gen = torch.zeros(n_finite, 1, dtype=self.floatx, device=self.device)

        # Classifier input: concat [x, theta] so context-dependent mismatches are detectable
        inp_real = torch.cat([x_real, theta_c2st], dim=1)
        inp_gen = torch.cat([x_gen, theta_c2st], dim=1)

        x_all = torch.cat([inp_real, inp_gen], dim=0)
        y_all = torch.cat([labels_real, labels_gen], dim=0)

        # Shuffle
        perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(self.torch_seed))
        x_all = x_all[perm]
        y_all = y_all[perm]

        # Train / test split for the classifier
        n_test = int(len(x_all) * test_fraction)
        n_clf_train = len(x_all) - n_test
        x_clf_train, x_clf_test = x_all[:n_clf_train], x_all[n_clf_train:]
        y_clf_train, y_clf_test = y_all[:n_clf_train], y_all[n_clf_train:]

        # 2-hidden-layer MLP classifier. Outputs raw logits (no final Sigmoid) and is trained with
        # BCEWithLogitsLoss, which is numerically stable -- a Sigmoid + BCELoss combination instead
        # triggers a device-side assert if any input is non-finite.
        input_dim = x_all.shape[1]
        classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        ).to(self.device)

        clf_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()

        clf_loader = DataLoader(TensorDataset(x_clf_train, y_clf_train), batch_size=batch_size, shuffle=True)

        classifier.train()
        for _ in range(n_epochs):
            for x_batch, y_batch in clf_loader:
                pred = classifier(x_batch)
                loss = criterion(pred, y_batch)
                clf_optimizer.zero_grad()
                loss.backward()
                clf_optimizer.step()

        # Evaluate on held-out classifier test set (logits > 0 <=> probability > 0.5)
        classifier.eval()
        with torch.no_grad():
            pred_test = classifier(x_clf_test)
            accuracy = ((pred_test > 0).float() == y_clf_test).float().mean().item()

        return accuracy

    def _prepare_data(self, x, theta, batch_size, vali_split, seed=None, group_ids=None):
        """
        Prepare the data for training and validation.

        Args:
            x (numpy.ndarray): The input features (summary statistics).
            theta (numpy.ndarray): The input context (cosmological parameters).
            batch_size (int): Batch size for training and validation.
            vali_split (float): Proportion of data to be used for validation.
            seed (int, optional): The seed for the random split. Defaults to None. Ignored when
                `group_ids` is given, since that split is fully deterministic.
            group_ids (numpy.ndarray, optional): 1D array aligned row-for-row with `x`/`theta`
                (e.g. `i_signal`). When given, the split is made deterministic and group-aware:
                the unique id values are sorted and partitioned into a train and a validation
                fraction (mirroring `_parse_cls_indices` in msi.utils.preprocessing), and every
                row is assigned to whichever set its group id falls into. This guarantees that no
                group (e.g. signal realization) appears in both the training and validation set,
                which a plain row-level random split cannot guarantee when groups have multiple
                rows (e.g. several noise realizations per signal). When omitted, falls back to
                the previous row-level `random_split` behaviour.

        Returns:
            None
        """

        if seed is None:
            seed = self.torch_seed

        x = torch.tensor(x, dtype=self.floatx, device=self.device)
        theta = torch.tensor(theta, dtype=self.floatx, device=self.device)

        dset = TensorDataset(x, theta)

        if group_ids is not None:
            unique_ids = np.unique(group_ids)
            split = int((1 - vali_split) * len(unique_ids))
            train_ids, vali_ids = unique_ids[:split], unique_ids[split:]
            train_idx = np.where(np.isin(group_ids, train_ids))[0]
            vali_idx = np.where(np.isin(group_ids, vali_ids))[0]
            LOGGER.info(
                f"Splitting by group id into {len(train_ids)} train / {len(vali_ids)} vali groups "
                f"({len(train_idx)} / {len(vali_idx)} rows)"
            )
            self.train_dset = Subset(dset, train_idx)
            self.vali_dset = Subset(dset, vali_idx)
        else:
            self.train_dset, self.vali_dset = random_split(
                dset, [1 - vali_split, vali_split], torch.Generator().manual_seed(seed)
            )

        self.train_loader = DataLoader(self.train_dset, batch_size, shuffle=True, drop_last=True)
        self.vali_loader = DataLoader(self.vali_dset, batch_size, shuffle=False, drop_last=True)

    def _train_epoch(self):
        """Train the model for one epoch."""

        self.train()

        epoch_loss = []
        for x, theta in self.train_loader:
            loss = -self.log_prob(inputs=x, context=theta).mean()
            epoch_loss.append(loss.item())

            # Backpropagation
            loss.backward()
            if self.clip_by_global_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_by_global_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        epoch_loss = np.mean(epoch_loss)

        return epoch_loss

    def _vali_epoch(self):
        """Evaluate the model on the validation set once."""

        self.eval()

        with torch.no_grad():
            epoch_loss = []
            for x, theta in self.vali_loader:
                loss = -self.log_prob(inputs=x, context=theta).mean()
                epoch_loss.append(loss.item())

        epoch_loss = np.mean(epoch_loss)

        return epoch_loss

    # likelihood ######################################################################################################

    def sample_likelihood(self, theta, n_samples=1000, batch_size=None, return_numpy=True):
        """
        Sample from the likelihood distribution p(x|theta). This can be done directly from the flow and doesn't need
        an MCMC sampler.

        Args:
            theta (Union[torch.Tensor, np.ndarray]): The theta values to condition on. This array/tensor can have more
                than one dimension.
            n_samples (int, optional): The number of samples to generate for each condition. Defaults to 1000.
            batch_size (int, optional): The batch size for generating samples. Defaults to None.
            return_numpy (bool, optional): Whether to return the samples as a numpy array instead of a pytorch tensor.
                Defaults to True.
            out_dir (str, optional): The directory to save the samples. Defaults to None.
            label (str, optional): The label for the saved samples. Defaults to None.

        Returns:
            torch.Tensor or numpy.ndarray: The generated samples of the same shape as theta_obs, except for an
                additional axis of length n_samples.
        """

        theta = torch.tensor(theta, dtype=self.floatx, device=self.device)

        self.eval()
        with torch.no_grad():
            samples = self.sample(n_samples, context=theta, batch_size=batch_size)

        if return_numpy:
            samples = samples.cpu().numpy()

        return samples

    def log_likelihood(self, x, theta, return_numpy=False, use_validation_weights=False):
        """Wrapper for the log_prob method of the base Flow. In most cases (e.g. for training and MCMC), the raw
        log_prob method is preferred.

        Args:
            x (Union[np.ndarray,torch.tensor]): Array/tensor containing the summary statistic. Possibly not 2
                dimensional, like shape (n_cosmos, n_examples, n_summary).
            theta (Union[np.ndarray,torch.tensor]): Array/tensor of the cosmological parameters. Same behavior as for x.
            return_numpy (bool, optional): Return numpy arrays instead of torch.tensors. Defaults to False.
            use_validation_weights (bool, optional): Dummy argument for compatibility with LikelihoodFlowEnsemble. Defaults to False.

        Returns:
            np.ndarray or torch.tensor: Non-normalized log probabilities.
        """

        x = torch.tensor(x, dtype=self.floatx, device=self.device)
        theta = torch.tensor(theta, dtype=self.floatx, device=self.device)

        # ravel all but the last dimension
        do_reshape = x.ndim > 2 or theta.ndim > 2
        if do_reshape:
            assert x.shape[:-1] == theta.shape[:-1], f"The feature dimension needs to be the same for x and theta"
            out_shape = x.shape[:-1]

            x_features = x.shape[-1]
            theta_features = theta.shape[-1]

            x = x.reshape(-1, x_features)
            theta = theta.reshape(-1, theta_features)

        with torch.no_grad():
            log_like = super().log_prob(x, context=theta)

        # bring into the original shape
        if do_reshape:
            log_like = log_like.reshape(out_shape)

        if return_numpy:
            log_like = log_like.cpu().numpy()

        return log_like

    # posterior #######################################################################################################

    def sample_posterior(
        self,
        x_obs,
        n_walkers=1_024,
        n_steps=1_000,
        n_burnin_steps=1_000,
        lambdaCDM=False,
        label=None,
        device=None,
        dont_save=False,
        method="ensemble",
        use_validation_weights=False,
    ):
        """
        Sample from the posterior distribution p(theta|x) using likelihood learned by the flow model and the flat
        analysis prior. The sampling is done using the emcee library, which runs on the CPU and in numpy.

        Args:
            x_obs (np.ndarray): The observation to condition the posterior on. It must have shape (n_features,) or
                (1, n_features).
            n_samples (int, optional): The number of samples to generate. Defaults to 512000.
            n_walkers (int, optional): The number of walkers in the MCMC chain. Defaults to 1024.
            n_burnin_steps (int, optional): The number of burn-in steps in the MCMC chain. Defaults to 100.
            label (str, optional): Additional label for the saved chain, for example to designate different
                observations. Defaults to None.
            device (str, optional): The device to use for computation, potentially override the initialized value since
                it can be advantageous to first train on GPU, but then run the whole MCMC chain on the CPU. Defaults to
                None.

        Returns:
            array-like: The generated samples from the likelihood flow model.
        """

        n_samples = n_steps * n_walkers

        if device is None:
            device = self.device

        x_obs = torch.tensor(x_obs, dtype=self.floatx, device=device)
        x_obs = torch.atleast_2d(x_obs)
        if x_obs.shape[0] == 1:
            LOGGER.info(f"Sampling the posterior from a single observation")
        else:
            LOGGER.info(f"Sampling the posterior from multiple observations")

        self.to(device)
        self.eval()

        if lambdaCDM:
            LOGGER.warning("lambdaCDM")
            label = (label or "") + "_lambdaCDM"
            i_w = self.params.index("w0")
            params = [p for p in self.params if p != "w0"]
        else:
            LOGGER.warning("wCDM")
            params = self.params

        def log_prob_fn(theta_walkers):
            if lambdaCDM:
                theta_walkers = np.insert(theta_walkers, i_w, -1.0, axis=1)
            return self._mcmc_log_posterior(theta_walkers, x_obs, device=device)

        chain = mcmc.run_emcee(
            log_prob_fn,
            params,
            conf=self.conf,
            out_dir=self.model_dir if not dont_save else None,
            label=label,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burnin_steps=n_burnin_steps,
        )

        # restore the flow to the original device
        self.to(self.device)

        return chain

    def _single_log_posterior(self, theta_walkers, x_obs, device="cuda"):
        """theta_walkers.shape = (n_walkers, theta_dim)"""
        assert x_obs.shape[0] == 1

        # FlowConductor doesn't broadcast the context, so we have to do it manually
        inputs = x_obs.repeat(theta_walkers.shape[0], 1)

        # override the default device
        context = torch.tensor(theta_walkers, dtype=self.floatx, device=device)

        with torch.no_grad():
            # evaluate the normalizing flow, for emcee the result must always be on the CPU in the end
            log_prob = self.log_prob(inputs=inputs, context=context).to("cpu").numpy()
            # log_prob = self.log_prob(inputs=context, context=inputs).to("cpu").numpy()

            # enforce the prior
            log_prob = prior.log_posterior(theta_walkers, log_prob, conf=self.conf, params=self.params)

        return log_prob

    def _mcmc_log_posterior(self, theta_walkers, x_obs, device="cuda"):
        """theta_walkers.shape = (n_walkers, theta_dim)"""

        assert x_obs.ndim == 2

        if x_obs.shape[0] == 1:
            log_prob = self._single_log_posterior(theta_walkers, x_obs, device=device)
        else:
            log_prob = np.zeros((theta_walkers.shape[0]))
            for x in x_obs:
                x = torch.atleast_2d(x)
                log_prob += self._single_log_posterior(theta_walkers, x, device=device)

        return log_prob

    def _batched_log_likelihood_torch(self, theta, x_obs, weights=None):
        """On-device batched flow log-likelihood log p(x|theta) for the torch ensemble sampler.

        Unlike _single_log_posterior (one observation, walkers batched, numpy round-trip per step), this
        keeps everything on the GPU and adds a leading observation axis so a single flow forward pass
        covers n_obs * n_walkers points. The hard prior is applied by the shared base wrapper
        (_batched_log_posterior_torch); ``weights`` is ignored for a single flow.

        Args:
            theta (torch.Tensor): Cosmological parameters, shape (n_obs, n_walkers, n_params), on device.
            x_obs (torch.Tensor): Observations, shape (n_obs, n_features), on device.

        Returns:
            torch.Tensor: Log-likelihood of shape (n_obs, n_walkers).
        """
        n_obs, n_walkers, n_params = theta.shape
        theta_flat = theta.reshape(n_obs * n_walkers, n_params)
        # broadcast each observation across its walkers (FlowConductor does not broadcast the context)
        x_flat = x_obs.unsqueeze(1).expand(-1, n_walkers, -1).reshape(n_obs * n_walkers, x_obs.shape[-1])
        return self.log_prob(inputs=x_flat, context=theta_flat).reshape(n_obs, n_walkers)

    # utils ###########################################################################################################

    def save(self):
        """Save the weights and initialization arguments of the model to disk."""

        if self.model_dir is not None:
            checkpoint = {"state_dict": self.state_dict(), "init_kwargs": self._init_kwargs}
            try:
                torch.save(checkpoint, self.model_file)
            except RuntimeError as e:
                # The checkpoint pickles init_kwargs, which holds the live transform/embedding_net modules.
                # Spectral-norm parametrized transforms (lipschitz iResBlocks) cannot be pickled
                # ("Serialization of parametrized modules is only supported through state_dict()"). Fall back
                # to a state_dict-only checkpoint, which load() already accepts. Such a checkpoint has no
                # init_kwargs, so from_checkpoint cannot reopen it -- acceptable for the lipschitz cross-check.
                LOGGER.warning(
                    f"Full checkpoint save failed ({type(e).__name__}: {e}); "
                    f"falling back to a state_dict-only checkpoint (from_checkpoint will not work for it)"
                )
                torch.save({"state_dict": self.state_dict()}, self.model_file)
            LOGGER.info(f"Saved the model to {self.model_file}")
        else:
            LOGGER.warning(f"Could not save the model, no output directory specified")

    def load(self):
        """Load the weights of the model from disk."""

        if self.device == "cpu":
            map_location = torch.device("cpu")
        else:
            map_location = None

        if self.model_dir is not None:
            loaded = torch.load(self.model_file, map_location=map_location, weights_only=False)
            if isinstance(loaded, dict) and "state_dict" in loaded:
                self.load_state_dict(loaded["state_dict"])
            else:
                self.load_state_dict(loaded)
            LOGGER.info(f"Loaded the model from {self.model_file}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_file=None,
        model_dir=None,
        out_dir=None,
        prefix="",
        suffix="",
        label=None,
        **kwargs_overrides,
    ):
        """
        Restore a completely initialized model from a checkpoint file.

        Args:
            checkpoint_file (str, optional): The path to the saved .pt file.
            model_dir (str, optional): The directory containing the model file.
            out_dir (str, optional): The base output directory.
            prefix (str, optional): Prefix for the model directory name.
            suffix (str, optional): Suffix for the model directory name.
            label (str, optional): Subdirectory label within out_dir.
            **kwargs_overrides: Optional arguments to override the ones saved in the checkpoint.

        Returns:
            LikelihoodFlow: The fully restored model.
        """
        if checkpoint_file is None:
            if model_dir is None and out_dir is not None:
                if label is None:
                    model_dir = os.path.join(out_dir, prefix + cls.model_name + suffix)
                else:
                    model_dir = os.path.join(out_dir, label, prefix + cls.model_name + suffix)

            if model_dir is not None:
                checkpoint_file = os.path.join(model_dir, f"{cls.model_name}.pt")

        if checkpoint_file is None:
            raise ValueError("Insufficient path arguments to determine checkpoint_file.")
        loaded = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

        if not isinstance(loaded, dict) or "init_kwargs" not in loaded:
            raise ValueError(f"The checkpoint at {checkpoint_file} does not contain the required 'init_kwargs'.")

        init_kwargs = loaded["init_kwargs"]
        init_kwargs.update(kwargs_overrides)

        model = cls(**init_kwargs, load_existing=True)

        return model


class LikelihoodFlowEnsemble(LikelihoodBase):
    """Ensemble of LikelihoodFlow models trained from different random initial conditions.

    This class creates and manages multiple LikelihoodFlow instances that share the same architecture
    but are trained from different random initial conditions. It provides methods for training the
    ensemble and sampling from the posterior using the ensemble average.
    """

    model_name = "ensemble_flow"

    def __init__(
        self,
        params,
        conf=None,
        n_flows=4,
        # output
        out_dir=None,
        model_dir=None,
        prefix="",
        suffix="",
        label=None,
        load_existing=True,
        # architecture
        feature_dim=None,
        embedding_net=None,
        base_dist=None,
        transform=None,
        embedding_net_fn=None,
        base_dist_fn=None,
        transform_fn=None,
        # computational
        device=None,
        floatx=torch.float32,
        torch_seed=7,
    ):
        """
        Initialize the EnsembleFlow object.

        Args:
            params (list): The cosmological and astrophysical parameters to be constrained.
            n_flows (int, optional): Number of flows in the ensemble. Defaults to 5.
            conf (str, optional): The configuration file path. Defaults to None.
            out_dir (str, optional): The output directory path. Defaults to None.
            model_dir (str, optional): The model directory path. Defaults to None.
            prefix (str, optional): The prefix used in the saved filenames. Defaults to "".
            suffix (str, optional): The suffix used in the saved filenames. Defaults to "".
            label (str, optional): The label used in the saved filenames. Defaults to None.
            load_existing (bool, optional): Whether to load models from disk if they exist. Defaults to True.
            embedding_net_fn (callable, optional): Function that returns a new embedding network. Defaults to None.
            base_dist_fn (callable, optional): Function that returns a new base distribution. Defaults to None.
            transform_fn (callable, optional): Function that returns a new transform. Defaults to None.
            device (str, optional): The device to evaluate flows on. Defaults to None.
            floatx (torch.dtype, optional): The default float type. Defaults to torch.float32.
            torch_seed (int, optional): Base random seed. Each flow gets seed + flow_idx. Defaults to 7.
        """

        self._init_kwargs = {
            "params": params,
            "conf": conf,
            "n_flows": n_flows,
            "out_dir": out_dir,
            "model_dir": model_dir,
            "prefix": prefix,
            "suffix": suffix,
            "label": label,
            "load_existing": False,
            "feature_dim": feature_dim,
            "embedding_net": embedding_net,
            "base_dist": base_dist,
            "transform": transform,
            "embedding_net_fn": embedding_net_fn,
            "base_dist_fn": base_dist_fn,
            "transform_fn": transform_fn,
            "device": device,
            "floatx": floatx,
            "torch_seed": torch_seed,
        }

        self.params = params
        self.n_flows = n_flows
        self.conf = files.load_config(conf)

        self.out_dir = out_dir
        self.model_dir = model_dir
        self.prefix = prefix
        self.suffix = suffix
        self.label = label
        self._setup_dirs(".pt")

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.floatx = floatx
        self.torch_seed = torch_seed
        # temperature applied to the negative-validation-loss softmax in _compute_validation_weights;
        # 1.0 reproduces the original weights, larger values flatten toward uniform. run_inference sets
        # this from the mcmc config before sampling.
        self.validation_weight_temperature = 1.0

        self.feature_dim = feature_dim
        # vmap fusion state for the batched sampler, (vmapped_fn, stacked_params, stacked_buffers) when the
        # members can be stacked and vmapped, else None (per-member loop). Built lazily in _set_eval_device.
        self._vmap_state = None
        self.use_vmap = True  # set False to force the per-member loop (debugging / unsupported flows)

        self.embedding_net_fn = embedding_net_fn
        self.base_dist_fn = base_dist_fn
        self.transform_fn = transform_fn

        # create ensemble of flows
        self.flows = []
        self.validation_losses = []
        for i in range(n_flows):
            flow_name = f"flow_{i}"
            flow_label = f"{label}_{flow_name}" if label else flow_name

            # create specific model directory for this flow
            flow_model_dir = None
            if self.model_dir is not None:
                flow_model_dir = os.path.join(self.model_dir, flow_name)
                os.makedirs(flow_model_dir, exist_ok=True)

            # Reload path: reconstruct each member from its own checkpoint, which carries that member's
            # own architecture in its init_kwargs. This is what makes a heterogeneous ensemble reloadable
            # (members differ in architecture, so we cannot rebuild them from a single shared config) and
            # avoids depending on the ensemble's own (unpicklable) factory closures.
            member_ckpt = (
                os.path.join(flow_model_dir, f"{LikelihoodFlow.model_name}.pt")
                if flow_model_dir is not None
                else None
            )
            if load_existing and member_ckpt is not None and os.path.exists(member_ckpt):
                try:
                    flow = LikelihoodFlow.from_checkpoint(model_dir=flow_model_dir)
                    self.flows.append(flow)
                    continue
                except (ValueError, FileNotFoundError) as e:
                    # e.g. a state_dict-only checkpoint (parametrized lipschitz transform) has no
                    # init_kwargs to rebuild from; surface a clear error rather than silently mis-loading.
                    raise RuntimeError(
                        f"Could not reload ensemble member from {member_ckpt}: {type(e).__name__}: {e}. "
                        "Heterogeneous/checkpoint reload is unsupported for members whose architecture "
                        "cannot be pickled (e.g. lipschitz)."
                    ) from e

            # get fresh architecture components for each flow. Each *_fn may be a single callable
            # (homogeneous ensemble: every member shares the same config) or a list of length
            # n_flows (heterogeneous ensemble: member i uses fn[i]).
            import copy

            def _member_fn(fn, idx):
                if isinstance(fn, (list, tuple)):
                    return fn[idx]
                return fn

            emb_fn_i = _member_fn(embedding_net_fn, i)
            base_fn_i = _member_fn(base_dist_fn, i)
            tr_fn_i = _member_fn(transform_fn, i)

            if emb_fn_i is not None:
                flow_embedding_net = emb_fn_i()
            else:
                flow_embedding_net = copy.deepcopy(embedding_net) if embedding_net is not None else None

            if base_fn_i is not None:
                flow_base_dist = base_fn_i()
            else:
                flow_base_dist = copy.deepcopy(base_dist) if base_dist is not None else None

            if tr_fn_i is not None:
                flow_transform = tr_fn_i()
            else:
                flow_transform = copy.deepcopy(transform) if transform is not None else None

            flow = LikelihoodFlow(
                params=params,
                conf=conf,
                out_dir=out_dir,
                model_dir=flow_model_dir,
                prefix=prefix,
                suffix=suffix,
                label=flow_label,
                load_existing=load_existing,
                feature_dim=feature_dim,
                embedding_net=flow_embedding_net,
                base_dist=flow_base_dist,
                transform=flow_transform,
                device=device,
                floatx=floatx,
                torch_seed=torch_seed + i,  # different seed for each flow
            )
            self.flows.append(flow)

        LOGGER.info(f"Initialized ensemble with {n_flows} flows on device {self.device}")

    def fit(
        self,
        x,
        theta,
        n_epochs=100,
        batch_size=1000,
        vali_split=0.1,
        learning_rate=1e-3,
        weight_decay=0.0,
        clip_by_global_norm=1.0,
        scheduler_type=None,
        scheduler_kwargs=None,
        n_patience_epochs=None,
        min_delta=1e-4,
        save_model=True,
        seed=None,
        group_ids=None,
        run_c2st=False,
        c2st_hidden_dim=64,
        c2st_n_epochs=50,
        member_train_kwargs=None,
    ):
        """
        Train all flows in the ensemble on the same data.

        Args:
            x (torch.Tensor): The input features (summary statistics).
            theta (torch.Tensor): The input context (cosmological parameters).
            n_epochs (int, optional): The number of epochs to train for. Defaults to 100.
            batch_size (int, optional): The batch size for training and validation. Defaults to 1000.
            vali_split (float, optional): The validation split ratio. Defaults to 0.1.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.0.
            clip_by_global_norm (float, optional): The maximum gradient norm for gradient clipping. Defaults to 1.0.
            scheduler_type (str, optional): The type of learning rate scheduler. Defaults to None.
            scheduler_kwargs (dict, optional): Additional kwargs for the scheduler. Defaults to {}.
            n_patience_epochs (int, optional): The number of epochs for early stopping. Defaults to None.
            min_delta (float, optional): The minimum change for early stopping. Defaults to 1e-4.
            save_model (bool, optional): Whether to save the models after training. Defaults to True.
            seed (int, optional): The seed for the random data split. Defaults to None, then each flow uses its own seed.
            group_ids (numpy.ndarray, optional): 1D array aligned row-for-row with `x`/`theta`, forwarded to each
                flow's `_prepare_data` to make the train/vali split deterministic and group-aware -- see
                `LikelihoodFlow.fit`. When given, every ensemble member gets the same split (independent of `seed`),
                so `_compute_validation_weights` comparisons across members are apples-to-apples.
            run_c2st (bool, optional): Whether to run a Classifier Two-Sample Test. Defaults to False.
            c2st_hidden_dim (int, optional): Hidden layer size for the C2ST classifier MLP. Defaults to 64.
            c2st_n_epochs (int, optional): Number of epochs to train the C2ST classifier. Defaults to 50.
            member_train_kwargs (list[dict], optional): Per-member training-kwarg overrides of length
                n_flows, used by a heterogeneous ensemble so each member trains with its own config's
                `training` block (e.g. differing weight_decay/scheduler). Keys not present fall back to
                the uniform arguments above. Defaults to None (every member uses the uniform args).
        """

        LOGGER.info(f"Training ensemble of {self.n_flows} flows")

        # Fused vmap lockstep training: train all (identical-architecture) members in a single vmapped
        # pass per batch on one GPU, instead of the sequential per-member loop below. Restricted to the
        # fixed-step regime (no early stopping) and deterministic schedules, since lockstep cannot
        # early-stop members independently. Heterogeneous members and flows that do not vmap fall back to
        # the sequential loop (either via these gates or the try/except). Members are written back only on
        # success, so a failed fused attempt leaves their initial weights untouched for the fallback.
        can_fuse = (
            self.use_vmap
            and member_train_kwargs is None
            and n_patience_epochs is None
            and scheduler_type in (None, "cosine", "exp")
            and self.n_flows >= 2
        )
        if can_fuse:
            try:
                return self._fit_fused(
                    x,
                    theta,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    vali_split=vali_split,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    clip_by_global_norm=clip_by_global_norm,
                    scheduler_type=scheduler_type,
                    scheduler_kwargs=scheduler_kwargs,
                    save_model=save_model,
                    seed=seed,
                    group_ids=group_ids,
                    run_c2st=run_c2st,
                    c2st_hidden_dim=c2st_hidden_dim,
                    c2st_n_epochs=c2st_n_epochs,
                )
            except Exception as e:
                LOGGER.warning(
                    f"Fused vmap training unavailable ({type(e).__name__}: {e}); falling back to sequential"
                )

        if member_train_kwargs is not None and len(member_train_kwargs) != self.n_flows:
            raise ValueError(
                f"member_train_kwargs has length {len(member_train_kwargs)}, expected n_flows={self.n_flows}"
            )

        # uniform defaults; per-member overrides (if any) are layered on top below
        base_kwargs = dict(
            n_epochs=n_epochs,
            batch_size=batch_size,
            vali_split=vali_split,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clip_by_global_norm=clip_by_global_norm,
            scheduler_type=scheduler_type,
            scheduler_kwargs=scheduler_kwargs,
            n_patience_epochs=n_patience_epochs,
            min_delta=min_delta,
            run_c2st=run_c2st,
            c2st_hidden_dim=c2st_hidden_dim,
            c2st_n_epochs=c2st_n_epochs,
        )

        self.validation_losses = []
        histories = []
        for i, flow in enumerate(self.flows):
            LOGGER.info(f"Training flow {i+1}/{self.n_flows}")
            flow_kwargs = dict(base_kwargs)
            if member_train_kwargs is not None:
                # absent keys fall back to the uniform args; explicit values (including None, e.g.
                # n_patience_epochs=None meaning "no early stopping") are respected.
                flow_kwargs.update(member_train_kwargs[i])
            history = flow.fit(
                x=x,
                theta=theta,
                save_model=save_model,
                seed=seed if seed is not None else self.torch_seed,
                group_ids=group_ids,
                **flow_kwargs,
            )
            final_vali_loss = flow._vali_epoch()
            self.validation_losses.append(final_vali_loss)
            histories.append(history)
            LOGGER.info(f"Flow {i+1} final validation loss: {final_vali_loss:.4f}")

        # log validation-based weights
        weights = self._compute_validation_weights()
        LOGGER.info(f"Validation-based weights: {weights}")

        return histories

    def _fit_fused(
        self,
        x,
        theta,
        n_epochs,
        batch_size,
        vali_split,
        learning_rate,
        weight_decay,
        clip_by_global_norm,
        scheduler_type,
        scheduler_kwargs,
        save_model,
        seed,
        group_ids,
        run_c2st,
        c2st_hidden_dim,
        c2st_n_epochs,
    ):
        """Train all (identically-structured) members in lockstep with a single vmapped pass per batch.

        The members' parameters are stacked along a leading ensemble axis; ``torch.func.vmap`` over
        ``grad`` evaluates every member's gradient on its own shuffled batch in one fused kernel, and a
        single Adam over the stacked tensors updates all members at once (Adam is elementwise, so the
        leading axis is just more parameters -> mathematically N independent optimizers). Each member
        keeps its own initial weights (the seed diversity from __init__) and its own per-epoch shuffle, so
        the main statistical change versus the sequential path is the shared, deterministic LR schedule
        (plus, for dropout configs, a different RNG draw order for the masks -- still i.i.d. Bernoulli(p)
        per element, just not bit-identical to the sequential loop's).

        Restricted to fixed-step training (no early stopping); the caller gates on that. Raises on
        unsupported settings (e.g. a plateau schedule, or BatchNorm) or any vmap/trace failure so ``fit``
        can fall back to the sequential loop with the members' weights untouched (write-back happens only
        at the end). Dropout is supported: training uses a dedicated meta base with dropout explicitly
        re-enabled and per-member-independent masks (``randomness="different"``); validation uses a
        separate, fully-eval meta base with dropout off, matching ``_vali_epoch``.
        """
        import copy
        import matplotlib.pyplot as plt
        from torch.func import stack_module_state, functional_call, grad_and_value, vmap
        from msi.flow_conductor.vmap_compat import patch_enflows_for_vmap

        patch_enflows_for_vmap()  # rewrite enflows' in-place log-det accumulation out-of-place for vmap

        if scheduler_type == "plateau":
            raise NotImplementedError("fused training does not support the val-loss-dependent plateau schedule")

        # The fused forward runs the base in eval mode (see below); dropout is handled correctly via a
        # dedicated train-mode meta base (below). BatchNorm is not: it would use running stats instead of
        # batch stats while "training". No current config enables it, so reject rather than silently
        # mistrain if that ever changes.
        for flow in self.flows:
            for m in flow.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    raise NotImplementedError("fused training runs the base in eval mode; BatchNorm is not supported")

        device = self.device
        floatx = self.floatx
        N = self.n_flows
        scheduler_kwargs = {} if scheduler_kwargs is None else dict(scheduler_kwargs)
        member_seed = seed if seed is not None else self.torch_seed

        # shared, group-aware train/vali split (identical for every member, as in the sequential path)
        self.flows[0]._prepare_data(x, theta, batch_size, vali_split, seed=member_seed, group_ids=group_ids)
        tr = self.flows[0].train_dset
        va = self.flows[0].vali_dset
        train_x = tr.dataset.tensors[0][tr.indices].to(device)
        train_theta = tr.dataset.tensors[1][tr.indices].to(device)
        val_x = va.dataset.tensors[0][va.indices].to(device)
        val_theta = va.dataset.tensors[1][va.indices].to(device)
        n_train = train_x.shape[0]
        steps_per_epoch = n_train // batch_size
        if steps_per_epoch < 1:
            raise ValueError(f"batch_size {batch_size} exceeds the {n_train} training rows")

        # ActNorm uses lazy, data-dependent initialization guarded by `if self.training and not
        # self.initialized` -- a tensor branch vmap cannot trace. Initialize every member up front with a
        # train-mode warmup forward (sets initialized=True and the data-dependent scale/shift), then run
        # the vmapped forward with the base in eval mode so that guard short-circuits on training=False and
        # is never traced (dropout is handled separately below via a dedicated train-mode meta base).
        warm_x, warm_theta = train_x[:batch_size], train_theta[:batch_size]
        with torch.no_grad():
            for flow in self.flows:
                flow.train()
                flow.log_prob(inputs=warm_x, context=warm_theta)

        # stack member params/buffers once; both meta bases below share these stacked tensors.
        wrappers = [_EnsembleLogProb(flow) for flow in self.flows]
        stacked_params, buffers = stack_module_state(wrappers)
        params = {k: v.detach().clone().requires_grad_(True) for k, v in stacked_params.items()}
        buffers = {k: v.detach() for k, v in buffers.items()}

        # Two meta bases, identical except for Dropout submodules' `.training` flag (a plain Python bool,
        # not a stacked/vmapped tensor, so it can differ per base without affecting the shared params/
        # buffers). base_eval keeps everything in eval mode -- this is what makes ActNorm's data-dependent
        # `if self.training and not self.initialized` branch short-circuit safely under vmap (see above).
        # base_train starts from that same safe state, then selectively flips just the Dropout submodules
        # back to training=True, re-enabling their random masking without touching ActNorm. Never call
        # `.train()` on a container here -- that recurses and would flip ActNorm back on too.
        base_eval = copy.deepcopy(wrappers[0]).to("meta")
        base_eval.eval()
        base_train = copy.deepcopy(base_eval)
        for m in base_train.modules():
            if isinstance(m, torch.nn.Dropout):
                m.training = True

        def compute_loss(p, b, xb, tb):
            return -functional_call(base_train, (p, b), args=(xb, tb)).mean()

        # randomness="different": each vmap lane (ensemble member) draws its own independent dropout mask
        # per call -- the documented torch.func pattern for dropout under vmap, matching what the
        # sequential per-member loop does naturally. base_eval never calls a random op (dropout is a
        # no-op in eval mode), so neg_sum_fn keeps the default randomness="error" as a tripwire.
        grad_fn = vmap(grad_and_value(compute_loss), in_dims=(0, 0, 0, 0), randomness="different")
        neg_sum_fn = vmap(
            lambda p, b, xb, tb: -functional_call(base_eval, (p, b), args=(xb, tb)).sum(), (0, 0, None, None)
        )

        optimizer = optim.Adam(list(params.values()), lr=learning_rate, weight_decay=weight_decay)
        if scheduler_type == "cosine":
            scheduler_kwargs.setdefault("eta_min", 1e-5)
            scheduler_kwargs.setdefault("T_max", n_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
        elif scheduler_type == "exp":
            scheduler_kwargs.setdefault("gamma", 0.95)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_kwargs)
        else:
            scheduler = None

        LOGGER.info(
            f"Fused vmap training of {N} members: {steps_per_epoch} steps/epoch x {n_epochs} epochs on {device}"
        )

        gen = torch.Generator(device=device).manual_seed(member_seed)
        train_hist = np.zeros((n_epochs, N))
        val_hist = np.zeros((n_epochs, N))

        def _val_loss_vec():
            with torch.no_grad():
                total = torch.zeros(N, device=device, dtype=floatx)
                n_seen = 0
                for s in range(0, val_x.shape[0], batch_size):
                    xb = val_x[s : s + batch_size]
                    tb = val_theta[s : s + batch_size]
                    total = total + neg_sum_fn(params, buffers, xb, tb)
                    n_seen += xb.shape[0]
            return (total / max(n_seen, 1)).detach().cpu().numpy()

        pbar = LOGGER.progressbar(range(n_epochs), at_level="info", total=n_epochs)
        for i_epoch in pbar:
            # independent per-member shuffle each epoch (preserves the data-order diversity of the
            # sequential path); shape (N, n_train)
            perms = torch.argsort(torch.rand(N, n_train, device=device, generator=gen), dim=1)
            epoch_loss = torch.zeros(N, device=device, dtype=floatx)
            for s in range(steps_per_epoch):
                idx = perms[:, s * batch_size : (s + 1) * batch_size]  # (N, batch_size)
                xb = train_x[idx]  # (N, batch_size, x_dim)
                tb = train_theta[idx]  # (N, batch_size, theta_dim)
                grads, losses = grad_fn(params, buffers, xb, tb)  # grads: dict of (N, *), losses: (N,)

                if clip_by_global_norm is not None:
                    # per-member global-norm clipping, matching torch.nn.utils.clip_grad_norm_ per member
                    sq = torch.zeros(N, device=device, dtype=floatx)
                    for g in grads.values():
                        sq = sq + g.reshape(N, -1).pow(2).sum(dim=1)
                    scale = (clip_by_global_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
                    for k in grads:
                        g = grads[k]
                        grads[k] = g * scale.view(N, *([1] * (g.dim() - 1)))

                for k, p in params.items():
                    p.grad = grads[k]
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_loss = epoch_loss + losses.detach()

            if scheduler is not None:
                scheduler.step()

            train_hist[i_epoch] = (epoch_loss / steps_per_epoch).cpu().numpy()
            val_hist[i_epoch] = _val_loss_vec()
            pbar.set_description(
                f"lr: {get_lr(optimizer):.2E}, train: {train_hist[i_epoch].mean():.2f}, "
                f"vali: {val_hist[i_epoch].mean():.2f}"
            )

        # write the trained stacked weights back into the member flows (strip the "flow." wrapper prefix).
        # Restrict to the member's persistent state_dict keys: stack_module_state reads named_buffers and
        # so includes non-persistent buffers (e.g. the base distribution's constant `_log_z`) that are
        # absent from state_dict() and would otherwise be flagged as unexpected on load.
        pfx = "flow."
        for i, flow in enumerate(self.flows):
            expected = set(flow.state_dict().keys())
            member_sd = {k[len(pfx):]: v[i].detach() for k, v in params.items() if k[len(pfx):] in expected}
            member_sd.update(
                {k[len(pfx):]: v[i].detach() for k, v in buffers.items() if k[len(pfx):] in expected}
            )
            incompatible = flow.load_state_dict(member_sd, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"member {i} state_dict mismatch on write-back "
                    f"(missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys})"
                )

        self.validation_losses = list(val_hist[-1])
        LOGGER.info(f"Fused training final per-member validation losses: {self.validation_losses}")
        weights = self._compute_validation_weights()
        LOGGER.info(f"Validation-based weights: {weights}")

        # combined loss curves (mean +/- member spread) for the ensemble
        if self.model_dir is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            epochs = np.arange(n_epochs)
            for hist, label in [(train_hist, "training"), (val_hist, "validation")]:
                ax.plot(epochs, hist.mean(axis=1), label=label)
                ax.fill_between(epochs, hist.min(axis=1), hist.max(axis=1), alpha=0.2)
            ax.set(xlabel="epoch", ylabel="loss")
            ax.grid(True)
            ax.legend()
            fig.savefig(os.path.join(self.model_dir, "loss_curves.png"))
            plt.close(fig)

        if save_model:
            self.save()

        if run_c2st:
            for i, flow in enumerate(self.flows):
                try:
                    acc = flow._run_c2st(hidden_dim=c2st_hidden_dim, n_epochs=c2st_n_epochs)
                    LOGGER.info(f"Flow {i+1} C2ST accuracy: {acc:.4f} (ideal: 0.5)")
                except Exception as e:
                    LOGGER.warning(f"C2ST for flow {i+1} failed ({type(e).__name__}: {e}); skipping")

        return [
            {"train_loss": list(train_hist[:, i]), "vali_loss": list(val_hist[:, i])} for i in range(N)
        ]

    def sample_likelihood(self, theta, n_samples=1000, batch_size=None, return_numpy=True):
        """
        Sample from the ensemble likelihood distribution. Samples are drawn from a randomly selected
        flow in the ensemble.

        Args:
            theta (Union[torch.Tensor, np.ndarray]): The theta values to condition on.
            n_samples (int, optional): The number of samples per flow. Defaults to 1000.
            batch_size (int, optional): The batch size for generating samples. Defaults to None.
            return_numpy (bool, optional): Whether to return as numpy array. Defaults to True.

        Returns:
            torch.Tensor or numpy.ndarray: The generated samples.
        """

        all_samples = []
        samples_per_flow = n_samples // self.n_flows

        for flow in self.flows:
            samples = flow.sample_likelihood(
                theta=theta,
                n_samples=samples_per_flow,
                batch_size=batch_size,
                return_numpy=return_numpy,
            )
            all_samples.append(samples)

        # concatenate on the samples axis (-2), NOT the leading cosmos axis: each member returns
        # (n_cosmos, samples_per_flow, x_dim), so the ensemble draw is (n_cosmos, n_samples, x_dim).
        if return_numpy:
            all_samples = np.concatenate(all_samples, axis=-2)
        else:
            all_samples = torch.cat(all_samples, dim=-2)

        return all_samples

    def log_likelihood(self, x, theta, return_numpy=False, use_validation_weights=False):
        """
        Compute the ensemble log likelihood as the log of the weighted mean of the exponentials
        (i.e., weighted log-sum-exp).

        Args:
            x (Union[np.ndarray, torch.tensor]): Summary statistics.
            theta (Union[np.ndarray, torch.tensor]): Cosmological parameters.
            return_numpy (bool, optional): Return numpy arrays. Defaults to False.
            use_validation_weights (bool, optional): Weight flows by validation performance. Defaults to False.

        Returns:
            np.ndarray or torch.tensor: Ensemble log likelihoods.
        """

        x = torch.tensor(x, dtype=self.floatx, device=self.device)
        theta = torch.tensor(theta, dtype=self.floatx, device=self.device)

        log_likes = []
        for flow in self.flows:
            log_like = flow.log_likelihood(x, theta, return_numpy=False)
            log_likes.append(log_like)

        log_likes = torch.stack(log_likes, dim=0)

        if use_validation_weights and len(self.validation_losses) == self.n_flows:
            weights = torch.tensor(self._compute_validation_weights(), dtype=self.floatx, device=self.device)
            # weighted log-sum-exp: log(sum_i w_i * exp(log_like_i))
            log_ensemble = torch.logsumexp(log_likes + torch.log(weights).unsqueeze(-1), dim=0)
        else:
            # unweighted log-mean-exp
            log_ensemble = torch.logsumexp(log_likes, dim=0) - np.log(self.n_flows)

        if return_numpy:
            log_ensemble = log_ensemble.cpu().numpy()

        return log_ensemble

    def sample_posterior(
        self,
        x_obs,
        n_walkers=1_024,
        n_steps=1_000,
        n_burnin_steps=1_000,
        lambdaCDM=False,
        label=None,
        device=None,
        dont_save=False,
        method="individual",
        use_validation_weights=False,
        store_individual_chains=False,
    ):
        """
        Sample from the posterior distribution p(theta|x).

        Args:
            x_obs (np.ndarray): The observation to condition on.
            n_walkers (int, optional): The number of walkers in the MCMC chain. Defaults to 1_024.
            n_steps (int, optional): The number of steps per walker. Defaults to 1_000.
            n_burnin_steps (int, optional): The number of burn-in steps. Defaults to 1_000.
            lambdaCDM (bool, optional): Whether to fix w0=-1 for LambdaCDM. Defaults to False.
            label (str, optional): Additional label for the saved chain. Defaults to None.
            device (str, optional): The device to use. Defaults to None.
            dont_save (bool, optional): Whether to skip saving the chain. Defaults to False.
            method (str, optional): Either "ensemble" to sample from the averaged posterior, or "individual"
                to sample from each flow individually. Defaults to "ensemble".
            use_validation_weights (bool, optional): If True, weight flows by their validation
                performance (lower loss = higher weight). For method="ensemble" this weights the combined
                likelihood; for method="individual" it weights how members are pooled. Defaults to False.
            store_individual_chains (bool, optional): method="individual" only. If True, also save each
                member's own chain (chain_{label}_flow_{i}.npy); otherwise only the pooled chain is saved.
                Defaults to False.

        Returns:
            np.ndarray: A single array of posterior samples. For method="individual" this is the pooled
                chain (drawn from the weighted mixture of the members), shape-matching the ensemble path.
        """

        n_samples = n_steps * n_walkers

        if device is None:
            device = self.device

        x_obs = torch.tensor(x_obs, dtype=self.floatx, device=device)
        x_obs = torch.atleast_2d(x_obs)
        if x_obs.shape[0] == 1:
            LOGGER.info(f"Sampling the posterior from a single observation")
        else:
            LOGGER.info(f"Sampling the posterior from multiple observations")

        # move all flows to the specified device
        for flow in self.flows:
            flow.to(device)
            flow.eval()

        # Handle lambdaCDM setup
        if lambdaCDM:
            LOGGER.warning("lambdaCDM")
            label = (label or "") + "_lambdaCDM"
            i_w = self.params.index("w0")
            params = [p for p in self.params if p != "w0"]
        else:
            LOGGER.warning("wCDM")
            params = self.params

        # Compute weights for ensemble method
        if method == "ensemble":
            if use_validation_weights and len(self.validation_losses) == self.n_flows:
                LOGGER.info("Using validation-weighted ensemble")
                weights = self._compute_validation_weights()
                LOGGER.info(f"Weights: {weights}")
            else:
                if use_validation_weights:
                    LOGGER.warning("Validation weights requested but not available. Using uniform weights.")
                weights = None

            # Create log probability function
            def log_prob_fn(theta_walkers):
                if lambdaCDM:
                    theta_walkers = np.insert(theta_walkers, i_w, -1.0, axis=1)
                return self._mcmc_log_posterior(theta_walkers, x_obs, device=device, weights=weights)

        if method == "ensemble":
            LOGGER.info(f"Sampling the posterior from the ensemble using method '{method}'")
            chain = mcmc.run_emcee(
                log_prob_fn,
                params,
                conf=self.conf,
                out_dir=self.model_dir if not dont_save else None,
                label=label,
                n_walkers=n_walkers,
                n_steps=n_steps,
                n_burnin_steps=n_burnin_steps,
            )

        elif method == "individual":
            LOGGER.info(f"Sampling individual posteriors from {self.n_flows} flows, then pooling")
            # pool with the same weights the ensemble would use (uniform unless validation-weighted)
            if use_validation_weights and len(self.validation_losses) == self.n_flows:
                weights_np = self._compute_validation_weights()
            else:
                if use_validation_weights:
                    LOGGER.warning("Validation weights requested but not available. Using uniform weights.")
                weights_np = None

            member_chains = []
            for i, flow in enumerate(self.flows):
                flow_label = f"{label}_flow_{i}" if label else f"flow_{i}"
                LOGGER.info(f"Sampling posterior from flow {i+1}/{self.n_flows}")
                flow_chain = flow.sample_posterior(
                    x_obs=x_obs.cpu().numpy(),
                    n_walkers=n_walkers,
                    n_steps=n_steps,
                    n_burnin_steps=n_burnin_steps,
                    lambdaCDM=lambdaCDM,
                    label=flow_label,
                    device=device,
                    dont_save=(dont_save or not store_individual_chains),
                )
                member_chains.append(flow_chain)

            # pooled chain matches the 'ensemble' output (single array saved as chain_{label}.npy)
            rng = np.random.default_rng(self.torch_seed)
            chain = _pool_chains(member_chains, weights=weights_np, rng=rng)
            if not dont_save and self.model_dir is not None:
                chain_file = os.path.join(self.model_dir, f"chain_{label}.npy" if label else "chain.npy")
                np.save(chain_file, chain)
                LOGGER.info(f"Saved pooled individual chain to {chain_file}")

        else:
            raise ValueError(f"Unknown method {method}. Choose either 'ensemble' or 'individual'.")

        # restore flows to original device
        for flow in self.flows:
            flow.to(self.device)

        return chain

    def _mcmc_log_posterior(self, theta_walkers, x_obs, device="cuda", weights=None):
        """
        Compute the ensemble log posterior for MCMC sampling.

        Args:
            theta_walkers (np.ndarray): Walker positions with shape (n_walkers, theta_dim).
            x_obs (torch.Tensor): Observations with shape (n_obs, feature_dim).
            device (str, optional): Device to use. Defaults to "cuda".
            weights (np.ndarray, optional): Weights for each flow. If None, uses uniform weights.

        Returns:
            np.ndarray: Log posterior values for each walker.
        """

        assert x_obs.ndim == 2

        # compute ensemble log likelihood
        log_likes = []
        for flow in self.flows:
            if x_obs.shape[0] == 1:
                log_like = flow._single_log_posterior(theta_walkers, x_obs, device=device)
            else:
                # posterior product over multiple independent observations
                log_like = np.zeros((theta_walkers.shape[0]))
                for x in x_obs:
                    x = torch.atleast_2d(x)
                    log_like += flow._single_log_posterior(theta_walkers, x, device=device)
            log_likes.append(log_like)

        # average log likelihoods (in log space: weighted or unweighted log-mean-exp)
        log_likes = np.stack(log_likes, axis=0)

        if weights is not None:
            # weighted log-sum-exp: log(sum_i w_i * exp(log_like_i))
            log_weights = np.log(weights).reshape(-1, 1)  # Shape: (n_flows, 1)
            log_ensemble = np.logaddexp.reduce(log_likes + log_weights, axis=0)
        else:
            # unweighted log-mean-exp
            log_ensemble = np.logaddexp.reduce(log_likes, axis=0) - np.log(self.n_flows)

        return log_ensemble

    # GPU-batched sampling hooks (the shared driver lives in LikelihoodBase.sample_posterior_batched) ##############

    def _set_eval_device(self, device):
        """Move every member flow to ``device`` and switch it to eval mode (the ensemble is not itself an
        nn.Module, so the base default would not work), then (re)build the vmap fusion state for the new
        device."""
        for flow in self.flows:
            flow.to(device)
            flow.eval()
        self._build_vmap_state(device)

    def _build_vmap_state(self, device):
        """Stack the members' parameters once so the per-step batched log-likelihood can evaluate all
        members in a single vmapped pass instead of a Python loop. Requires identically-structured members
        (homogeneous ensembles, --n_flows seed-clones); heterogeneous members and flows whose log_prob does
        not vmap (e.g. lipschitz iResBlocks, or sigmoid transforms Dynamo/functorch cannot trace) fall back
        to the loop via the trial evaluation below. Parameters are fixed during sampling, so stacking once
        and reusing every step is what makes this cheap. Leaves self._vmap_state = None to mean "use loop".
        """
        self._vmap_state = None
        if not self.use_vmap or self.n_flows < 2:
            return
        try:
            import copy
            from torch.func import stack_module_state, functional_call, vmap
            from msi.flow_conductor.vmap_compat import patch_enflows_for_vmap

            patch_enflows_for_vmap()  # make the enflows forward traceable under vmap (no-op if already done)

            wrappers = [_EnsembleLogProb(flow) for flow in self.flows]
            params, buffers = stack_module_state(wrappers)
            # meta-device base holds no real storage; functional_call injects the stacked params per call.
            # deepcopy first so moving to meta does not strip the real flow's weights.
            base = copy.deepcopy(wrappers[0]).to("meta")

            def _fmodel(p, b, inputs, context):
                return functional_call(base, (p, b), args=(inputs, context))

            vmapped = vmap(_fmodel, in_dims=(0, 0, None, None))

            # trial run at a tiny batch to catch flows that cannot be vmapped/traced before the hot loop
            x_dummy = torch.zeros(2, self.feature_dim, dtype=self.floatx, device=device)
            theta_dummy = torch.zeros(2, len(self.params), dtype=self.floatx, device=device)
            with torch.no_grad():
                out = vmapped(params, buffers, x_dummy, theta_dummy)
            assert out.shape == (self.n_flows, 2), f"unexpected vmap output shape {tuple(out.shape)}"

            self._vmap_state = (vmapped, params, buffers)
            LOGGER.info(f"Ensemble vmap fusion enabled for {self.n_flows} members on {device}")
        except Exception as e:
            LOGGER.warning(f"Ensemble vmap fusion unavailable ({type(e).__name__}: {e}); using per-member loop")
            self._vmap_state = None

    def _get_ensemble_weights(self, use_validation_weights, device):
        """Per-member weights for the batched posterior: validation-performance weights when available,
        else None (uniform). Same policy as the emcee sample_posterior / log_likelihood."""
        if use_validation_weights and len(self.validation_losses) == self.n_flows:
            return torch.tensor(self._compute_validation_weights(), dtype=self.floatx, device=device)
        if use_validation_weights:
            LOGGER.warning("Validation weights requested but not available; using uniform ensemble weights.")
        return None

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
        return_members=False,
    ):
        """GPU-batched posterior sampling for the ensemble, switched by ``method``:

          - "ensemble": one chain on the combined (vmap-fused, weighted log-mean-exp) likelihood -- the
            shared LikelihoodBase driver.
          - "individual": sample each member separately with the single-flow driver, then pool the chains
            with _pool_chains using the same ensemble weights (uniform -> even split). This yields the same
            posterior as "ensemble" under uniform weights, but with better per-chain mixing and trivial
            parallelism across members.

        Returns (chain, log_probs) of the same shape as the "ensemble" path. With return_members=True the
        "individual" path additionally returns the lists of per-member (chain, log_probs) so the caller can
        persist them (store_individual_chains).
        """
        if method == "ensemble":
            return super().sample_posterior_batched(
                x_obs_batch,
                n_walkers=n_walkers,
                n_steps=n_steps,
                n_burnin_steps=n_burnin_steps,
                lambdaCDM=lambdaCDM,
                device=device,
                seed=seed,
                use_validation_weights=use_validation_weights,
                compile_flow=compile_flow,
            )
        if method != "individual":
            raise ValueError(f"Unknown method {method!r}; choose 'ensemble' or 'individual'.")

        if device is None:
            device = self.device
        weights = self._get_ensemble_weights(use_validation_weights, device)
        weights_np = weights.cpu().numpy() if weights is not None else None

        member_chains, member_log_probs = [], []
        for i, flow in enumerate(self.flows):
            LOGGER.info(f"[individual] GPU-batched sampling of member {i + 1}/{self.n_flows}")
            c, lp = flow.sample_posterior_batched(
                x_obs_batch,
                n_walkers=n_walkers,
                n_steps=n_steps,
                n_burnin_steps=n_burnin_steps,
                lambdaCDM=lambdaCDM,
                device=device,
                seed=seed + i,  # distinct walker init per member
                use_validation_weights=False,  # a single flow has no members to weight
                compile_flow=compile_flow,
            )
            member_chains.append(c)
            member_log_probs.append(lp)

        rng = np.random.default_rng(seed)
        chain, log_probs = _pool_chains(member_chains, weights=weights_np, member_log_probs=member_log_probs, rng=rng)
        if return_members:
            return chain, log_probs, member_chains, member_log_probs
        return chain, log_probs

    def _batched_log_likelihood_torch(self, theta, x_obs, weights=None):
        """On-device batched ensemble log-likelihood: the (weighted) log-mean-exp over members of each
        member's batched log p(x|theta). When vmap fusion is available (self._vmap_state, set in
        _set_eval_device) all members are evaluated in a single fused pass; otherwise they are looped over
        one at a time. Either way peak GPU memory is ~a single flow and the hard prior is applied once by
        the shared base wrapper. theta is (n_obs, n_walkers, n_params); returns (n_obs, n_walkers)."""
        n_obs, n_walkers, n_params = theta.shape
        theta_flat = theta.reshape(n_obs * n_walkers, n_params)
        x_flat = x_obs.unsqueeze(1).expand(-1, n_walkers, -1).reshape(n_obs * n_walkers, x_obs.shape[-1])

        if self._vmap_state is not None:
            vmapped, params, buffers = self._vmap_state
            # single fused pass over all members -> (n_flows, n_obs * n_walkers)
            log_likes = vmapped(params, buffers, x_flat, theta_flat).reshape(self.n_flows, n_obs, n_walkers)
        else:
            log_likes = torch.stack(
                [flow.log_prob(inputs=x_flat, context=theta_flat).reshape(n_obs, n_walkers) for flow in self.flows],
                dim=0,
            )  # (n_flows, n_obs, n_walkers)

        if weights is not None:
            # weighted log-sum-exp: log(sum_i w_i * exp(log_like_i))
            return torch.logsumexp(log_likes + torch.log(weights).view(-1, 1, 1), dim=0)
        # unweighted log-mean-exp
        return torch.logsumexp(log_likes, dim=0) - np.log(self.n_flows)

    def _prepare_data(self, *args, **kwargs):
        """Reproduce the deterministic, group-aware train/vali split for coverage-test reconstruction.

        The split depends only on (x, theta, group_ids) and is identical across members, so we prepare the
        first flow and expose its datasets on the ensemble -- this makes the ensemble a drop-in for the
        single-flow split reconstruction in run_inference.run_coverage_sampling / _set_up_flow.
        """
        self.flows[0]._prepare_data(*args, **kwargs)
        self.vali_dset = self.flows[0].vali_dset
        self.train_dset = getattr(self.flows[0], "train_dset", None)

    def save(self):
        """Save all flows in the ensemble."""
        if self.model_dir is not None:
            # The architecture entries (factory closures or nn.Module instances) are not stored in the
            # ensemble checkpoint: closures are unpicklable, and each member already persists its own
            # architecture. from_checkpoint rebuilds members from their own flow_i checkpoints, so the
            # ensemble file only needs the scalar/structural init_kwargs plus the validation losses.
            arch_keys = {"embedding_net", "base_dist", "transform", "embedding_net_fn", "base_dist_fn", "transform_fn"}
            init_kwargs = {k: v for k, v in self._init_kwargs.items() if k not in arch_keys}
            checkpoint = {"init_kwargs": init_kwargs, "validation_losses": self.validation_losses}
            torch.save(checkpoint, self.model_file)

        for flow in self.flows:
            flow.save()
        LOGGER.info(f"Saved ensemble of {self.n_flows} flows")

    def load(self):
        """Load all flows in the ensemble."""
        if self.model_dir is not None:
            try:
                # we don't strictly need to load the init_kwargs, but we can verify it's there
                loaded = torch.load(self.model_file, map_location="cpu", weights_only=False)
            except FileNotFoundError:
                LOGGER.warning(f"Could not load the model from {self.model_file}")

        for flow in self.flows:
            flow.load()
        LOGGER.info(f"Loaded ensemble of {self.n_flows} flows")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_file=None,
        model_dir=None,
        out_dir=None,
        prefix="",
        suffix="",
        label=None,
        **kwargs_overrides,
    ):
        """
        Restore a completely initialized ensemble model from a checkpoint file.

        Args:
            checkpoint_file (str, optional): The path to the saved .pt file.
            model_dir (str, optional): The directory containing the model file.
            out_dir (str, optional): The base output directory.
            prefix (str, optional): Prefix for the model directory name.
            suffix (str, optional): Suffix for the model directory name.
            label (str, optional): Subdirectory label within out_dir.
            **kwargs_overrides: Optional arguments to override the ones saved in the checkpoint.

        Returns:
            LikelihoodFlowEnsemble: The fully restored model.
        """
        if checkpoint_file is None:
            if model_dir is None and out_dir is not None:
                if label is None:
                    model_dir = os.path.join(out_dir, prefix + cls.model_name + suffix)
                else:
                    model_dir = os.path.join(out_dir, label, prefix + cls.model_name + suffix)

            if model_dir is not None:
                checkpoint_file = os.path.join(model_dir, f"{cls.model_name}.pt")

        if checkpoint_file is None:
            raise ValueError("Insufficient path arguments to determine checkpoint_file.")

        try:
            loaded = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            # For backward compatibility where we might have saved individual flows but not the ensemble file itself
            # We can reconstruct it from the first flow if it exists
            if model_dir is not None:
                flow_0_file = os.path.join(model_dir, "flow_0", f"{LikelihoodFlow.model_name}.pt")
                if os.path.exists(flow_0_file):
                    LOGGER.warning(f"Could not find {checkpoint_file}, attempting to load from {flow_0_file}")
                    flow_loaded = torch.load(flow_0_file, map_location="cpu", weights_only=False)
                    if isinstance(flow_loaded, dict) and "init_kwargs" in flow_loaded:
                        loaded = {"init_kwargs": flow_loaded["init_kwargs"]}
                        # If the old save format had no n_flows in init_kwargs but there are directories
                        import glob

                        flow_dirs = glob.glob(os.path.join(model_dir, "flow_*"))
                        loaded["init_kwargs"]["n_flows"] = len(flow_dirs)
                    else:
                        raise FileNotFoundError(f"Missing {checkpoint_file} and cannot reconstruct from {flow_0_file}")
                else:
                    raise
            else:
                raise

        if not isinstance(loaded, dict) or "init_kwargs" not in loaded:
            raise ValueError(f"The checkpoint at {checkpoint_file} does not contain the required 'init_kwargs'.")

        init_kwargs = loaded["init_kwargs"]

        # update paths in init_kwargs to match the current loading context
        init_kwargs["model_dir"] = model_dir
        init_kwargs["out_dir"] = out_dir
        init_kwargs["prefix"] = prefix
        init_kwargs["suffix"] = suffix
        init_kwargs["label"] = label

        init_kwargs.update(kwargs_overrides)
        if "load_existing" in init_kwargs:
            del init_kwargs["load_existing"]

        model = cls(**init_kwargs, load_existing=True)

        if "validation_losses" in loaded:
            model.validation_losses = loaded["validation_losses"]

        return model

    def _compute_validation_weights(self):
        """
        Compute normalized weights based on validation losses.
        Lower validation loss = higher weight.
        Uses softmax of negative losses for numerical stability.

        Returns:
            np.ndarray: Normalized weights summing to 1.
        """
        if len(self.validation_losses) != self.n_flows:
            LOGGER.warning("Validation losses not available, using uniform weights")
            return np.ones(self.n_flows) / self.n_flows

        # convert losses to weights: lower loss = higher weight. The softmax is tempered by
        # validation_weight_temperature T: T=1 reproduces the original weights, larger T flattens
        # toward uniform (preventing a single slightly-better member from dominating the mixture).
        temperature = getattr(self, "validation_weight_temperature", 1.0)
        if temperature <= 0:
            raise ValueError(f"validation_weight_temperature must be > 0, got {temperature}")
        neg_losses = -np.array(self.validation_losses) / temperature
        neg_losses_shifted = neg_losses - np.max(neg_losses)
        weights = np.exp(neg_losses_shifted)
        weights = weights / np.sum(weights)

        return weights

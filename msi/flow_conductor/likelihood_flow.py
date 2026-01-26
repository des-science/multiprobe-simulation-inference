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
from torch.utils.data import TensorDataset, DataLoader, random_split

from enflows.flows import Flow

from msi.likelihood_base import LikelihoodBase
from msi.utils import mcmc
from msi.flow_conductor import architecture
from msfm.utils import logger, files, prior

from msi.flow_conductor.pytorch import EarlyStopper, get_lr

LOGGER = logger.get_logger(__file__)


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

        self.params = params
        self.conf = files.load_config(conf)

        self.out_dir = out_dir
        self.model_dir = model_dir
        self.label = label
        self._setup_dirs(".pt")

        # the summary statistic has the same dimension as the constrained parameters
        context_dim = len(params)

        if feature_dim is None:
            feature_dim = context_dim
            LOGGER.warning(f"Assuming that the feature/summary dimension is equal to the context/parameter dimension")

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
        scheduler_kwargs={},
        # early stopping
        n_patience_epochs=None,
        min_delta=1e-4,
        save_model=True,
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
        """

        self._prepare_data(x, theta, batch_size, vali_split)

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

    def _prepare_data(self, x, theta, batch_size, vali_split):
        """
        Prepare the data for training and validation.

        Args:
            x (numpy.ndarray): The input features (summary statistics).
            theta (numpy.ndarray): The input context (cosmological parameters).
            batch_size (int): Batch size for training and validation.
            vali_split (float): Proportion of data to be used for validation.

        Returns:
            None
        """

        x = torch.tensor(x, dtype=self.floatx, device=self.device)
        theta = torch.tensor(theta, dtype=self.floatx, device=self.device)

        dset = TensorDataset(x, theta)

        self.train_dset, self.vali_dset = random_split(
            dset, [1 - vali_split, vali_split], torch.Generator().manual_seed(self.torch_seed)
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

    def log_likelihood(self, x, theta, return_numpy=False):
        """Wrapper for the log_prob method of the base Flow. In most cases (e.g. for training and MCMC), the raw
        log_prob method is preferred.

        Args:
            x (Union[np.ndarray,torch.tensor]): Array/tensor containing the summary statistic. Possibly not 2
                dimensional, like shape (n_cosmos, n_examples, n_summary).
            theta (Union[np.ndarray,torch.tensor]): Array/tensor of the cosmological parameters. Same behavior as for x.
            return_numpy (bool, optional): Return numpy arrays instead of torch.tensors. Defaults to False.

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
        n_samples=1_024_000,
        n_walkers=1_024,
        n_burnin_steps=1_000,
        label=None,
        device=None,
        dont_save=False,
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

        chain = mcmc.run_emcee(
            lambda theta_walkers: self._mcmc_log_posterior(theta_walkers, x_obs, device=device),
            self.params,
            conf=self.conf,
            out_dir=self.model_dir if not dont_save else None,
            label=label,
            n_walkers=n_walkers,
            n_steps=int(np.ceil(n_samples / n_walkers)),
            n_burnin_steps=n_burnin_steps,
        )

        # there can be more samples than requested due the walkers
        chain = chain[:n_samples]

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

    # utils ###########################################################################################################

    def save(self):
        """Save the weights of the model to disk."""

        if self.model_dir is not None:
            torch.save(self.state_dict(), self.model_file)
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
            self.load_state_dict(torch.load(self.model_file, map_location=map_location))
            LOGGER.info(f"Loaded the model from {self.model_file}")


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
        n_flows=5,
        # output
        out_dir=None,
        model_dir=None,
        label=None,
        load_existing=True,
        # architecture
        feature_dim=None,
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
            label (str, optional): The label used in the saved filenames. Defaults to None.
            load_existing (bool, optional): Whether to load models from disk if they exist. Defaults to True.
            embedding_net_fn (callable, optional): Function that returns a new embedding network. Defaults to None.
            base_dist_fn (callable, optional): Function that returns a new base distribution. Defaults to None.
            transform_fn (callable, optional): Function that returns a new transform. Defaults to None.
            device (str, optional): The device to evaluate flows on. Defaults to None.
            floatx (torch.dtype, optional): The default float type. Defaults to torch.float32.
            torch_seed (int, optional): Base random seed. Each flow gets seed + flow_idx. Defaults to 7.
        """

        self.params = params
        self.n_flows = n_flows
        self.conf = files.load_config(conf)

        self.out_dir = out_dir
        self.model_dir = model_dir
        self.label = label
        self._setup_dirs("")

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.floatx = floatx
        self.torch_seed = torch_seed

        self.embedding_net_fn = embedding_net_fn
        self.base_dist_fn = base_dist_fn
        self.transform_fn = transform_fn

        # create ensemble of flows
        self.flows = []
        self.validation_losses = []
        for i in range(n_flows):
            flow_label = f"{label}_flow_{i}" if label else f"flow_{i}"

            # get fresh architecture components for each flow
            embedding_net = embedding_net_fn() if embedding_net_fn is not None else None
            base_dist = base_dist_fn() if base_dist_fn is not None else None
            transform = transform_fn() if transform_fn is not None else None

            flow = LikelihoodFlow(
                params=params,
                conf=conf,
                out_dir=out_dir,
                model_dir=model_dir,
                label=flow_label,
                load_existing=load_existing,
                feature_dim=feature_dim,
                embedding_net=embedding_net,
                base_dist=base_dist,
                transform=transform,
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
        scheduler_kwargs={},
        n_patience_epochs=None,
        min_delta=1e-4,
        save_model=True,
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
        """

        LOGGER.info(f"Training ensemble of {self.n_flows} flows")

        self.validation_losses = []
        for i, flow in enumerate(self.flows):
            LOGGER.info(f"Training flow {i+1}/{self.n_flows}")
            flow.fit(
                x=x,
                theta=theta,
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
                save_model=save_model,
            )
            final_vali_loss = flow._vali_epoch()
            self.validation_losses.append(final_vali_loss)
            LOGGER.info(f"Flow {i+1} final validation loss: {final_vali_loss:.4f}")

        # log validation-based weights
        weights = self._compute_validation_weights()
        LOGGER.info(f"Validation-based weights: {weights}")

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

        if return_numpy:
            all_samples = np.concatenate(all_samples, axis=0)
        else:
            all_samples = torch.cat(all_samples, dim=0)

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
        n_samples=1_024_000,
        n_walkers=1_024,
        n_burnin_steps=1_000,
        label=None,
        device=None,
        dont_save=False,
        method="ensemble",
        use_validation_weights=False,
    ):
        """
        Sample from the posterior distribution p(theta|x).

        Args:
            x_obs (np.ndarray): The observation to condition on.
            n_samples (int, optional): The number of samples to generate. Defaults to 1_024_000.
            n_walkers (int, optional): The number of walkers in the MCMC chain. Defaults to 1_024.
            n_burnin_steps (int, optional): The number of burn-in steps. Defaults to 1_000.
            label (str, optional): Additional label for the saved chain. Defaults to None.
            device (str, optional): The device to use. Defaults to None.
            dont_save (bool, optional): Whether to skip saving the chain. Defaults to False.
            method (str, optional): Either "ensemble" to sample from the averaged posterior, or "separate"
                to sample from each flow individually. Defaults to "ensemble".
            use_validation_weights (bool, optional): If True and method="ensemble", weight flows by their
                validation performance (lower loss = higher weight). Defaults to False.

        Returns:
            array-like or list: If method="ensemble", returns a single array of posterior samples.
                If method="separate", returns a list of arrays, one for each flow in the ensemble.
        """

        if device is None:
            device = self.device

        x_obs = torch.tensor(x_obs, dtype=self.floatx, device=device)
        x_obs = torch.atleast_2d(x_obs)

        # move all flows to the specified device
        for flow in self.flows:
            flow.to(device)
            flow.eval()

        if method == "ensemble":
            if use_validation_weights and len(self.validation_losses) == self.n_flows:
                LOGGER.info("Using validation-weighted ensemble")
                weights = self._compute_validation_weights()
                LOGGER.info(f"Weights: {weights}")
            else:
                if use_validation_weights:
                    LOGGER.warning("Validation weights requested but not available. Using uniform weights.")
                weights = None

            chain = mcmc.run_emcee(
                lambda theta_walkers: self._mcmc_log_posterior(theta_walkers, x_obs, device=device, weights=weights),
                self.params,
                conf=self.conf,
                out_dir=self.model_dir if not dont_save else None,
                label=label,
                n_walkers=n_walkers,
                n_steps=int(np.ceil(n_samples / n_walkers)),
                n_burnin_steps=n_burnin_steps,
            )
            chain = chain[:n_samples]

        elif method == "individual":
            LOGGER.info(f"Sampling individual posteriors from {self.n_flows} flows")
            chain = []
            for i, flow in enumerate(self.flows):
                flow_label = f"{label}_flow_{i}" if label else f"flow_{i}"
                LOGGER.info(f"Sampling posterior from flow {i+1}/{self.n_flows}")

                flow_chain = flow.sample_posterior(
                    x_obs=x_obs.cpu().numpy(),
                    n_samples=n_samples,
                    n_walkers=n_walkers,
                    n_burnin_steps=n_burnin_steps,
                    label=flow_label,
                    device=device,
                    dont_save=dont_save,
                )
                chain.append(flow_chain)

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

    def save(self):
        """Save all flows in the ensemble."""
        for flow in self.flows:
            flow.save()
        LOGGER.info(f"Saved ensemble of {self.n_flows} flows")

    def load(self):
        """Load all flows in the ensemble."""
        for flow in self.flows:
            flow.load()
        LOGGER.info(f"Loaded ensemble of {self.n_flows} flows")

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

        # convert losses to weights: lower loss = higher weight
        neg_losses = -np.array(self.validation_losses)
        neg_losses_shifted = neg_losses - np.max(neg_losses)
        weights = np.exp(neg_losses_shifted)
        weights = weights / np.sum(weights)

        return weights

    def _setup_dirs(self, ext):
        """Setup output directories (inherited from LikelihoodBase)."""
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

        if self.model_dir is None and self.out_dir is not None:
            self.model_dir = os.path.join(self.out_dir, "models")
            os.makedirs(self.model_dir, exist_ok=True)

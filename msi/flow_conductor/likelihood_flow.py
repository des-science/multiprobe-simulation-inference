# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen

Wrapper around enflows to build a likelihood normalizing flow with training and sampling utilities.
"""

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
        label=None,
        load_existing=True,
        # architecture
        embedding_net=None,
        base_dist=None,
        transform=None,
        # computational
        device=None,
        floatx=torch.float32,
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
        self.label = label
        self._setup_dirs(".pt")

        # the summary statistic has the same dimension as the constrained parameters
        context_dim = len(params)

        if (embedding_net is None) or (base_dist is None) or (transform is None):
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
        n_patience_epochs=10,
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
            LOGGER.info(f"Using a cosine annealing scheduler")
            scheduler_kwargs.setdefault("eta_min", 1e-5)
            scheduler_kwargs.setdefault("T_max", n_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_kwargs)
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
            elif scheduler_type == "cosine":
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
        train_dset, vali_dset = random_split(dset, [1 - vali_split, vali_split])

        self.train_loader = DataLoader(train_dset, batch_size, shuffle=True, drop_last=True)
        self.vali_loader = DataLoader(vali_dset, batch_size, shuffle=False, drop_last=True)

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

    def sample_posterior(self, x_obs, n_samples=512000, n_walkers=1024, n_burnin_steps=100, label=None, device=None):
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

        self.to(device)
        self.eval()

        chain = mcmc.run_emcee(
            lambda theta_walkers: self._mcmc_log_posterior(theta_walkers, x_obs, device=device),
            self.params,
            out_dir=self.model_dir,
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

    def _mcmc_log_posterior(self, theta_walkers, x_obs, device="cuda"):
        """theta_walkers.shape = (n_walkers, theta_dim)"""

        # FlowConductor doesn't broadcast the context, so we have to do it manually
        inputs = x_obs.repeat(theta_walkers.shape[0], 1)

        # override the default device
        context = torch.tensor(theta_walkers, dtype=self.floatx, device=device)

        with torch.no_grad():
            # evaluate the normalizing flow, for emcee the result must always be on the CPU in the end
            log_prob = self.log_prob(inputs=inputs, context=context).to("cpu").numpy()

            # enforce the prior
            log_prob = prior.log_posterior(theta_walkers, log_prob, params=self.params, conf=self.conf)

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

        if self.model_dir is not None:
            self.load_state_dict(torch.load(self.model_file))
            LOGGER.info(f"Loaded the model from {self.model_file}")


class LikelihoodFlowEnsemble(LikelihoodFlow):
    """Ensemble of likelihood flows."""

    def __init__(self):
        raise NotImplementedError("This class is not implemented yet")

    def log_prob(self):
        """See https://github.com/jfcrenshaw/pzflow/blob/268a52b8ef3e38d3e1a4f1364a0430a8148eed22/pzflow/flowEnsemble.py#L241C59-L241C59"""
        raise NotImplementedError("This method is not implemented yet")

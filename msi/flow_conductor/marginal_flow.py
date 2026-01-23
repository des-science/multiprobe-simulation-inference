# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2026
Author: Arne Thomsen

Simplified normalizing flow for approximating unconditional distributions p(x).
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from enflows.flows import Flow
from enflows.distributions.normal import StandardNormal
from enflows.transforms import CompositeTransform, MaskedSumOfSigmoidsTransform, ActNorm, SVDLinear

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class EarlyStopper:
    """Simple early stopping utility."""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class MarginalFlow(Flow):
    """Normalizing flow for approximating unconditional distributions p(x).

    A simplified flow class designed for learning unconditional distributions
    of moderate dimensionality (~10 dimensions). Subclasses enflows.flows.Flow
    and provides a convenient training interface with validation split and
    gradient clipping for stable optimization.

    Args:
        feature_dim (int): Dimensionality of the data distribution.
        n_transforms (int, optional): Number of transformation layers. Defaults to 5.
        hidden_features (int, optional): Hidden dimension for transformation networks.
            Defaults to 128.
        n_blocks (int, optional): Number of residual blocks per transformation.
            Defaults to 2.
        device (str, optional): Device to run on ('cuda' or 'cpu'). Defaults to auto-detect.
        floatx (torch.dtype, optional): Float precision. Defaults to torch.float32.
        torch_seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """

    def __init__(
        self,
        feature_dim,
        n_transforms=5,
        hidden_features=128,
        n_blocks=2,
        device=None,
        floatx=torch.float32,
        torch_seed=42,
    ):
        # Base distribution: standard normal
        base_dist = StandardNormal(shape=(feature_dim,))

        # Build transformation: stack of masked sum-of-sigmoids transforms
        transforms = []
        for _ in range(n_transforms):
            transforms.append(ActNorm(features=feature_dim))

            transforms.append(SVDLinear(features=feature_dim, num_householder=2))

            transforms.append(
                MaskedSumOfSigmoidsTransform(
                    features=feature_dim,
                    hidden_features=hidden_features,
                    num_blocks=n_blocks,
                    activation=torch.nn.functional.relu,
                )
            )
        transform = CompositeTransform(transforms)

        # Initialize Flow base class (no embedding network for unconditional flow)
        super(MarginalFlow, self).__init__(transform, base_dist)

        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.floatx = floatx
        self.torch_seed = torch_seed
        self.to(self.device)

        LOGGER.info(f"Initialized MarginalFlow with {feature_dim}D, {n_transforms} transforms")
        LOGGER.info(f"Running on device: {self.device}")

    def fit(
        self,
        x,
        n_epochs=100,
        batch_size=256,
        vali_split=0.1,
        learning_rate=1e-3,
        weight_decay=0.0,
        clip_by_global_norm=1.0,
        n_patience_epochs=None,
        min_delta=1e-4,
        plot_loss=True,
        use_scheduler=True,
        eta_min=1e-6,
    ):
        """
        Fit the flow to data with validation-based monitoring and gradient clipping.

        Args:
            x (np.ndarray or torch.Tensor): Training data of shape (n_samples, feature_dim).
            n_epochs (int, optional): Maximum number of training epochs. Defaults to 100.
            batch_size (int, optional): Batch size for training/validation. Defaults to 256.
            vali_split (float, optional): Fraction of data for validation. Defaults to 0.1.
            learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 1e-3.
            weight_decay (float, optional): L2 regularization weight. Defaults to 0.0.
            clip_by_global_norm (float, optional): Max gradient norm for clipping.
                Set to None to disable. Defaults to 1.0.
            n_patience_epochs (int, optional): Early stopping patience. Set to None to
                disable. Defaults to None.
            min_delta (float, optional): Minimum validation loss improvement for early
                stopping. Defaults to 1e-4.
            plot_loss (bool, optional): Whether to create and save a loss curve plot.
                Defaults to True.
            use_scheduler (bool, optional): Whether to use cosine annealing learning rate
                scheduler. Defaults to True.
            eta_min (float, optional): Minimum learning rate for cosine annealing.
                Defaults to 1e-6.

        Returns:
            dict: Training history with 'train_loss' and 'vali_loss' lists.
        """

        # Prepare data
        x = torch.tensor(x, dtype=self.floatx, device=self.device)
        dataset = TensorDataset(x)

        # Split into train and validation
        n_vali = int(len(dataset) * vali_split)
        n_train = len(dataset) - n_vali
        train_dset, vali_dset = random_split(
            dataset, [n_train, n_vali], torch.Generator().manual_seed(self.torch_seed)
        )

        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        vali_loader = DataLoader(vali_dset, batch_size=batch_size, shuffle=False)

        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler
        scheduler = None
        if use_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=eta_min)
            LOGGER.info(f"Using cosine annealing scheduler: lr {learning_rate:.2e} -> {eta_min:.2e}")

        # Early stopping
        early_stopper = None
        if n_patience_epochs is not None:
            early_stopper = EarlyStopper(patience=n_patience_epochs, min_delta=min_delta)
            LOGGER.info(f"Using early stopping with patience={n_patience_epochs}, " f"min_delta={min_delta}")

        # Training loop
        train_losses = []
        vali_losses = []

        pbar = LOGGER.progressbar(range(n_epochs), at_level="info", total=n_epochs)
        for epoch in pbar:
            # Training phase
            self.train()
            epoch_train_loss = []
            for (batch_x,) in train_loader:
                # Negative log likelihood
                loss = -self.log_prob(batch_x).mean()
                epoch_train_loss.append(loss.item())

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if clip_by_global_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_by_global_norm)

                optimizer.step()

            train_loss = np.mean(epoch_train_loss)
            train_losses.append(train_loss)

            # Validation phase
            self.eval()
            epoch_vali_loss = []
            with torch.no_grad():
                for (batch_x,) in vali_loader:
                    loss = -self.log_prob(batch_x).mean()
                    epoch_vali_loss.append(loss.item())

            vali_loss = np.mean(epoch_vali_loss)
            vali_losses.append(vali_loss)

            # Learning rate scheduler step
            if scheduler is not None:
                scheduler.step()

            # Progress bar update
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"Epoch {epoch+1}/{n_epochs} - "
                f"lr: {current_lr:.2e}, train_loss: {train_loss:.3f}, vali_loss: {vali_loss:.3f}"
            )

            # Early stopping check
            if early_stopper is not None and early_stopper.early_stop(vali_loss):
                LOGGER.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        LOGGER.info(
            f"Training complete. Final train_loss: {train_losses[-1]:.3f}, " f"vali_loss: {vali_losses[-1]:.3f}"
        )
        if plot_loss:
            self._plot_loss_curves(train_losses, vali_losses)
        return {"train_loss": train_losses, "vali_loss": vali_losses}

    def _plot_loss_curves(self, train_losses, vali_losses):
        """Plot and save training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", linewidth=2)
        plt.plot(vali_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Negative Log Likelihood", fontsize=12)
        plt.title("MarginalFlow Training History", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def sample(self, n_samples=1000, return_numpy=True):
        """
        Generate samples from the learned distribution.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 1000.
            return_numpy (bool, optional): Return as numpy array instead of torch tensor.
                Defaults to True.

        Returns:
            np.ndarray or torch.Tensor: Samples of shape (n_samples, feature_dim).
        """
        self.eval()
        with torch.no_grad():
            samples = super().sample(n_samples)

        if return_numpy:
            samples = samples.cpu().numpy()

        return samples

    def log_prob(self, x, return_numpy=False, no_grad=False):
        """
        Evaluate log probability of data under the learned distribution.

        Args:
            x (np.ndarray or torch.Tensor): Data of shape (..., feature_dim).
            return_numpy (bool, optional): Return as numpy array instead of torch tensor.
                Defaults to False.

        Returns:
            np.ndarray or torch.Tensor: Log probabilities of shape (...).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.floatx, device=self.device)

        if no_grad:
            self.eval()
            with torch.no_grad():
                log_p = super().log_prob(x)
        else:
            log_p = super().log_prob(x)

        if return_numpy:
            log_p = log_p.cpu().numpy()

        return log_p

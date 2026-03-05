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

from msi.utils import mcmc
from msfm.utils import logger, prior

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
        run_c2st=False,
        c2st_hidden_dim=64,
        c2st_n_epochs=50,
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
            run_c2st (bool, optional): Whether to run a Classifier Two-Sample Test on the
                validation set after training. The result is an MLP classifier accuracy;
                a value close to 0.5 indicates the flow has learned the distribution well.
                Defaults to False.
            c2st_hidden_dim (int, optional): Hidden layer size for the C2ST classifier MLP.
                Defaults to 64.
            c2st_n_epochs (int, optional): Number of epochs to train the C2ST classifier.
                Defaults to 50.

        Returns:
            dict: Training history with 'train_loss', 'vali_loss', and (if run_c2st=True)
                'c2st_accuracy' keys.
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

        history = {"train_loss": train_losses, "vali_loss": vali_losses}

        if run_c2st:
            # Collect full validation set as a single tensor
            vali_data = torch.cat([batch[0] for batch in vali_loader], dim=0)
            c2st_acc = self._run_c2st(vali_data, hidden_dim=c2st_hidden_dim, n_epochs=c2st_n_epochs)
            history["c2st_accuracy"] = c2st_acc

        return history

    def _run_c2st(self, vali_data, n_epochs=50, hidden_dim=64, batch_size=256, test_fraction=0.3):
        """
        Classifier Two-Sample Test (C2ST).

        Trains a small binary MLP classifier to distinguish between real
        validation samples and samples generated by the flow. An accuracy
        close to 0.5 indicates the flow has learned the data distribution well;
        an accuracy close to 1.0 indicates a poor fit.

        Args:
            vali_data (torch.Tensor): Validation samples of shape (n, feature_dim).
            n_epochs (int, optional): Epochs to train the classifier. Defaults to 50.
            hidden_dim (int, optional): Hidden layer size of the classifier. Defaults to 64.
            batch_size (int, optional): Batch size for classifier training. Defaults to 256.
            test_fraction (float, optional): Fraction held out as classifier test set.
                Defaults to 0.3.

        Returns:
            float: Classifier test-set accuracy (ideal: 0.5, worst: 1.0).
        """
        n_real = len(vali_data)
        LOGGER.info(f"Running C2ST with {n_real} real vs {n_real} flow samples")

        # Generate samples from the flow
        self.eval()
        with torch.no_grad():
            generated = super(MarginalFlow, self).sample(n_real).to(self.device)

        # Labels: 1 = real, 0 = generated
        labels_real = torch.ones(n_real, 1, dtype=self.floatx, device=self.device)
        labels_gen = torch.zeros(n_real, 1, dtype=self.floatx, device=self.device)

        x_all = torch.cat([vali_data, generated], dim=0)
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

        # Simple 2-hidden-layer MLP classifier
        feature_dim = vali_data.shape[1]
        classifier = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        ).to(self.device)

        clf_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        clf_loader = DataLoader(
            TensorDataset(x_clf_train, y_clf_train), batch_size=batch_size, shuffle=True
        )

        classifier.train()
        for _ in range(n_epochs):
            for x_batch, y_batch in clf_loader:
                pred = classifier(x_batch)
                loss = criterion(pred, y_batch)
                clf_optimizer.zero_grad()
                loss.backward()
                clf_optimizer.step()

        # Evaluate on held-out classifier test set
        classifier.eval()
        with torch.no_grad():
            pred_test = classifier(x_clf_test)
            accuracy = ((pred_test > 0.5).float() == y_clf_test).float().mean().item()

        LOGGER.info(f"C2ST accuracy: {accuracy:.4f} (ideal: 0.5, worst: 1.0)")
        return accuracy

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

    def sample_residual_posterior(
        self,
        x_obs,
        params_wl,
        params_gc,
        emulator_wl,
        emulator_gc,
        conf,
        n_samples=1_024_000,
        n_walkers=1_024,
        n_burnin_steps=1_000,
        device=None,
        out_dir=None,
        label=None,
    ):
        """
        Sample from the residual posterior p(theta_wl, theta_gc | x_obs) using the learned residual distribution
        p(x_obs - mu(theta)) and emulators for the mean predictions.

        This is useful to sample P(Delta theta) in https://arxiv.org/pdf/2105.03324 for correlated observables.

        Args:
            x_obs (np.ndarray or torch.Tensor): The observed data vector.
            paramsp_wl (list): List of parameter names for the weak lensing emulator.
            params_gc (list): List of parameter names for the galaxy clustering emulator.
            emulator_wl (nn.Module): Emulator network for weak lensing mean.
            emulator_gc (nn.Module): Emulator network for galaxy clustering mean.
            conf (dict): Configuration dictionary containing priors.
            n_samples (int, optional): Number of samples to generate. Defaults to 100,000.
            n_walkers (int, optional): Number of MCMC walkers. Defaults to 100.
            n_burnin_steps (int, optional): Number of burn-in steps. Defaults to 100.
            device (str, optional): Device to run computations on. Defaults to None.
            out_dir (str, optional): Directory to save chains. Defaults to None.
            label (str, optional): Label for saved files. Defaults to None.

        Returns:
            np.ndarray: MCMC chain of samples.
        """
        if device is None:
            device = self.device

        x_obs = torch.tensor(x_obs, dtype=self.floatx, device=device)
        if x_obs.ndim == 1:
            x_obs = x_obs.unsqueeze(0)
        assert x_obs.shape[0] == 1, "sample_residual_posterior only supports a single observation vector."

        # ensure eval mode
        self.to(device)
        self.eval()
        emulator_wl.to(device)
        emulator_wl.eval()
        emulator_gc.to(device)
        emulator_gc.eval()

        combined_params = params_wl + params_gc
        n_wl = len(params_wl)

        def _log_posterior(theta_walkers):
            # theta_walkers shape: (n_walkers, n_params)

            theta_wl = theta_walkers[:, :n_wl]
            theta_gc = theta_walkers[:, n_wl:]

            t_theta_wl = torch.tensor(theta_wl, dtype=self.floatx, device=device)
            t_theta_gc = torch.tensor(theta_gc, dtype=self.floatx, device=device)

            with torch.no_grad():
                mu_wl = emulator_wl.predict(t_theta_wl, device=device)
                mu_gc = emulator_gc.predict(t_theta_gc, device=device)
                mu_joint = torch.cat([mu_wl, mu_gc], dim=1)

                # x_obs: (1, data_dim), mu_joint: (n_walkers, data_dim)
                residual = x_obs - mu_joint

                # log probability of residual under the flow
                log_prob_total = self.log_prob(residual).cpu().numpy()

                # add prior
                in_prior_wl = prior.in_grid_prior(theta_wl, conf=conf, params=params_wl)
                in_prior_gc = prior.in_grid_prior(theta_gc, conf=conf, params=params_gc)

                in_prior = np.logical_and(in_prior_wl, in_prior_gc)
                log_prob_total = np.where(in_prior, log_prob_total, -np.inf)

            return log_prob_total

        chain = mcmc.run_emcee(
            _log_posterior,
            combined_params,
            conf=conf,
            out_dir=out_dir,
            label=label,
            n_walkers=n_walkers,
            n_steps=int(np.ceil(n_samples / n_walkers)),
            n_burnin_steps=n_burnin_steps,
        )

        return chain[:n_samples]

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

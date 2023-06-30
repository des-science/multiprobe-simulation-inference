# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen, Janis Fluri

Implements a Gaussian Process regrossor, which is used as an emulator. For details, see Section F in
https://arxiv.org/pdf/2107.09002.pdf. The GP used is from 
https://gpflow.github.io/GPflow/2.4.0/notebooks/advanced/varying_noise.html
and allows for varying input noise.
Note that this model is absent from
https://gpflow.github.io/GPflow/2.7.0/notebooks/advanced/varying_noise.html

Adapted from
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/gp_emulator.py
by Janis Fluri
"""

import numpy as np
import tensorflow as tf
import gpflow, pickle

from gpflow.optimizers import NaturalGradient
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

# tf.config.run_functions_eagerly(True)


class HeteroskedasticGaussian(gpflow.likelihoods.Likelihood):
    """
    Likelihood for varying noise amplitude in the data. For this exact implementation, see
    https://gpflow.github.io/GPflow/2.4.0/notebooks/advanced/varying_noise.html#Make-a-new-likelihood
    And for the base class
    https://gpflow.github.io/GPflow/2.7.0/_modules/gpflow/likelihoods/base.html#Likelihood
    """

    def __init__(self, **kwargs):
        # this likelihood expects a single latent function F, and two columns in the data matrix Y:
        super().__init__(latent_dim=1, observation_dim=2, **kwargs)

    # For gpflow 2.7.0, this signature instead of (self, F, Y) for 2.4.0
    def _log_prob(self, X, F, Y):
        # log_prob is used by the quadrature fallback of variational_expectations and predict_log_density.
        # Because variational_expectations is implemented analytically below, this is not actually needed,
        # but is included for pedagogical purposes.
        # Note that currently relying on the quadrature would fail due to https://github.com/GPflow/GPflow/issues/966
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        return gpflow.logdensities.gaussian(Y, F, NoiseVar)

    # For gpflow 2.7.0, this signature instead of (self, Fmu, Fvar, Y) for 2.4.0
    def _variational_expectations(self, X, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        Fmu = Fmu[:, 0]
        Fvar = Fvar[:, 0]
        return (
            -0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(NoiseVar) - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
        )

    # The following two methods are abstract in the base class and therefore need to be implemented even if not used.

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError


class VGP_Emu:
    # default types
    np_float = gpflow.default_float()
    if gpflow.default_float() is np.float32:
        tf_float = tf.float32
    else:
        tf_float = tf.float64

    np_int = gpflow.default_int()
    if gpflow.default_int() is np.int32:
        tf_int = tf.int32
    else:
        tf_int = tf.int64

    def __init__(
        self,
        X_init: np.ndarray,
        Y_init: np.ndarray,
        # preprocessing
        normalize_X: bool = True,
        normalize_Y: bool = True,
        Y_with_std: bool = True,
        Y_min_var: bool = 1e-3,
        # kernel
        kernel_type: str = "matern52",
        ARD: bool = True,
    ):
        """Class implementing a Gaussian Process emulator that can handle input data of per point variable certainty.
        The GP implements a function f(x) = y and is fitted on a set of initial points to interpolate.

        Args:
            X_init (np.ndarray): Initial X (typically cosmology parameters) coordinates to emulate of shape
                (n_cosmos, n_params).
            Y_init (np.ndarray): Initial Y coordinates to emulate of shape (n_cosmos, 2), where the last axis contains
                the function values (predicted means) and their variance.
            normalize_X (bool, optional): Whether the X coordinates are preprocessed according to the normalization
                strategy of https://arxiv.org/abs/1912.08806, where the X coordinates are rotated to their eigenvalues,
                scaled and stretched. Defaults to True.
            normalize_Y (bool, optional): Whether the Y coordinates are standardized to have zero mean and unit
                std. Even though the GP can learn a q_mu that is different from zero, it typically fails, so this
                option should usually be True. Defaults to True.
            Y_with_std (bool, optional): If set to False, the Y coordinates are only shifted to have mean zero, but the
                variance remains unchanged. Defaults to True, then the variance is scaled to be one.
            Y_min_var (bool, optional): Minimum allowed variance estimate (contained in Y_init[:,1]), choosing this too
                small leads to numerical instabilities. Defaults to 1e-3.
            kern (str, optional): Kernel type, one of matern52, exponential, and squaredexponential. Defaults to
                "matern52".
            ARD (bool, optional): Whether to include Auto Relevance Determination (ARD), which means that every
                dimension gets its own lengthscale parameter in the kernel. Defaults to True.

        Raises:
            ValueError: When an unknown kernel function is specified
        """
        # preprocessing constants
        self.normalize_X = normalize_X
        self.normalize_Y = normalize_Y
        self.Y_with_std = Y_with_std
        self.Y_min_var = Y_min_var

        # normalize X
        self.fit_X(X_init)
        X_init = self.transform_X(X_init)

        # normalize Y
        self.fit_Y(Y_init)
        Y_init = self.transform_Y(Y_init)

        # kernel
        self.input_dim = int(X_init.shape[-1])
        if ARD:
            lengthscales = [1.0 for _ in range(self.input_dim)]
        else:
            lengthscales = 1.0

        self.kernel_type = kernel_type.lower()
        if self.kernel_type == "matern52":
            self.kernel = gpflow.kernels.Matern52(lengthscales=lengthscales)
        elif self.kernel_type == "exponential":
            self.kernel = gpflow.kernels.Exponential(lengthscales=lengthscales)
        elif self.kernel_type == "squaredexponential":
            self.kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
        else:
            raise ValueError(f"Kernel {self.kernel_type} is unkown")
        self.lengthscale_shape = self.kernel.lengthscales.shape

        # likelihood
        self.likelihood = HeteroskedasticGaussian(input_dim=self.input_dim)

        # model
        self.model = gpflow.models.VGP(
            (X_init, Y_init),
            kernel=self.kernel,
            likelihood=self.likelihood,
            num_latent_gps=1,
        )

        # turn off training for the variational parameters as they are trained with natgrad
        gpflow.utilities.set_trainable(self.model.q_mu, False)
        gpflow.utilities.set_trainable(self.model.q_sqrt, False)

        self.print_summary()

    # general #########################################################################################################

    def __call__(self, X):
        X = self.transform_X(X)

        # only retrieve the values, not variances
        Y_val = self.model.predict_f(X)[0].numpy()

        Y_val = self.inv_transform_Y(Y_val)
        return Y_val

    def print_summary(self):
        """Print a summary of the (trained) model parameters."""
        gpflow.utilities.print_summary(self.model)

    def save_model(self, out_file: str):
        """Save the model as a pickle object.

        Args:
            out_file (str): Output filename.
        """
        with open(out_file, "wb") as f:
            pickle.dump(self, f, protocol=4)

    @classmethod
    def load_model(cls, in_file: str):
        """Load the model from a pickle object.

        Args:
            in_file (str): Input filename.
        """
        with open(in_file, "rb") as f:
            emu = pickle.load(f)

        return emu

    # preprocessing ###################################################################################################

    def fit_X(self, X: np.ndarray):
        """Find the parameters to normalize the X data according to https://arxiv.org/abs/1912.08806 and set them as
        object attributes.

        Args:
            X (np.ndarray): Array of shape (n_cosmos, n_params) to normalize
        """

        if self.normalize_X:
            # make the params linearly uncorrelated
            cov = np.cov(X, rowvar=False)
            # eigenvals and vecs
            _, v = np.linalg.eig(cov)
            # transpose instead of inverse for orthogonal rotation mattrix
            rot_mat = v.T
            # matrix product
            rot_X = np.einsum("ij,aj->ai", rot_mat, X)
            # mean over the params
            rot_mean = np.mean(rot_X, axis=0, keepdims=True)
            # std (ddof of np.cov for consistency)
            rot_std = np.std(rot_X, axis=0, keepdims=True, ddof=1)

        else:
            rot_mat = np.eye(self.input_dim)
            rot_mean = np.zeros((self.input_dim, self.input_dim))
            rot_std = np.ones((self.input_dim, self.input_dim))

        self.rot_mat = rot_mat
        self.rot_mean = rot_mean
        self.rot_std = rot_std

    def transform_X(self, X: np.ndarray):
        """Normalize the X parameters according to the fit.

        Args:
            X (np.ndarray): Array of shape (n_cosmos, n_params) to normalize.

        Returns:
            np.ndarray: The normalized X array.
        """
        # rotation via matrix multiplication
        rot_X = np.einsum("ij,aj->ai", self.rot_mat, X)
        # standardize
        X = (rot_X - self.rot_mean) / self.rot_std

        X = X.astype(self.np_float)
        return X

    def inv_transform_X(self, X: np.ndarray):
        """Remove the X normalization.

        Args:
            X (np.ndarray): Array of shape (n_cosmos, n_params) to normalize.

        Returns:
            np.ndarray: The non-normalized X array.
        """
        # inverse rotation via matrix multiplication
        X = X * self.rot_std + self.rot_mean
        # inverse rotation
        X = np.einsum("ij,aj->ai", self.rot_mat.T, X)

        X = X.astype(self.np_float)
        return X

    def fit_Y(self, Y: np.ndarray):
        """Find the parameters to standardize the Y data and set them as object attributes.

        Args:
            Y (np.ndarray): Array of shape (n_cosmos, 2) to normalize. Y[:,0] contains the values to be fit, Y[:,1] the
                corresponding variances.
        """
        Y_val, Y_var = Y[:, 0], Y[:, 1]

        if self.normalize_Y:
            mu_Y = np.mean(Y_val)

            if self.Y_with_std:
                sigma_Y = np.std(Y_val)
            else:
                sigma_Y = 1.0
        else:
            # don't do anything when standardizing
            mu_Y = 0.0
            sigma_Y = 1.0

        self.mu_Y = mu_Y
        self.sigma_Y = sigma_Y

    def transform_Y(self, Y: np.ndarray):
        """Standardize the Y parameters according to the fit.

        Args:
            Y (np.ndarray): Array of shape (n_cosmos, 2) to normalize. Y[:,0] contains the values to be fit, Y[:,1] the
                corresponding variances.

        Returns:
            np.ndarray: The normalized Y array.
        """
        Y_val, Y_var = Y[:, 0], Y[:, 1]

        # standardize the values
        Y_val = (Y_val - self.mu_Y) / self.sigma_Y

        # scale the variance estimates appropriately
        Y_var /= self.sigma_Y**2

        # ensure the minimum variance
        Y_var = np.maximum(Y_var, self.Y_min_var)

        # join the values and variances
        Y = np.stack([Y_val, Y_var], axis=1)

        Y = Y.astype(self.np_float)
        return Y

    def inv_transform_Y(self, Y: np.ndarray):
        """Standardize the Y parameters according to the fit.

        Args:
            Y (np.ndarray): Array of shape (n_cosmos, 2) or (n_cosmos, 1) to normalize. Y[:,0] contains the values to
            be fit, if present Y[:,1] the corresponding variances.

        Returns:
            np.ndarray: The non-standardized Y array.
        """
        assert Y.ndim == 2, f"Y of shape {Y.shape} needs to be two dimensional"

        # Y contains both values and variances
        if Y.shape[1] == 2:
            Y_val, Y_var = Y[:, 0], Y[:, 1]

            # remove standardization
            Y_val = Y_val * self.sigma_Y + self.mu_Y

            # scale variances
            Y_var *= self.sigma_Y**2

            Y = np.stack([Y_val, Y_var], axis=1)

        # Y only contains values
        elif Y.shape[1] == 1:
            Y = Y * self.sigma_Y + self.mu_Y

        else:
            raise NotImplementedError

        Y = Y.astype(self.np_float)
        return Y

    # optimization ####################################################################################################

    def fit_model(self, n_steps: int = 1000, learning_rate: float = 1e-4):
        """Optimizes the VGP model parameters.

        Args:
            n_steps (int, optional): Number of optimization steps to perform. Defaults to 1000.
            learning_rate (float, optional): Learning rate to be used in the Adam optimizer. Defaults to 1e-4.

        Returns:
            list: List containing the loss values obtained in the individual training steps.
        """

        @tf.function
        def loss_objective_closure():
            return self.model.training_loss()

        # like in https://gpflow.github.io/GPflow/2.4.0/notebooks/advanced/varying_noise.html#Put-it-together-with-Variational-Gaussian-Process-(VGP)
        natgrad = NaturalGradient(gamma=1.0)
        adam = tf.optimizers.Adam(learning_rate)

        losses = []
        with LOGGER.progressbar(range(n_steps), total=n_steps, at_level="info", desc=f"fit the GP") as pbar:
            for _ in pbar:
                # minimize
                natgrad.minimize(loss_objective_closure, [(self.model.q_mu, self.model.q_sqrt)])
                adam.minimize(loss_objective_closure, self.model.trainable_variables)

                # store and log results
                loss = loss_objective_closure().numpy()
                pbar.set_postfix(loss_val=loss, refresh=False)
                losses.append(loss)

        return losses

    def repeated_fit(
        self,
        # fit
        n_steps: int = 1000,
        learning_rate: float = 1e-4,
        # repeated
        n_restarts: int = 10,
        full_restart: bool = False,
        # initial guesses
        kernel_init: float = 1.0,
        kernel_noise: float = 0.1,
        kernel_min: float = 0.1,
    ):
        """Optimizes the model for a given number of restarts and chooses the best result.

        Args:
            n_steps (int, optional): Number of optimization steps to perform. Defaults to 1000.
            learning_rate (float, optional): Learning rate to be used in the Adam optimizer. Defaults to 1e-4.
            n_restarts (int, optional): Number of times the optimization is repeated. Defaults to 10.
            full_restart (bool, optional): Ignore the current kernel values for the next optimization. Defaults to
                False.
            kernel_init (float, optional): For a full restart, this value defines the mean of the Gaussian from which
                the initial values of the kernel parameters are sampled. Defaults to 1.0.
            kernel_noise (float, optional): Defines the scale of the Gaussian from which new initial values to optimize
                are sampled after every restart. Defaults to 0.1.
            kernel_min (float, optional): Minimal value of the kernel parameters (variance and lengthscale). Defaults
                to 0.1.
        """
        if not self.normalize_X or not self.normalize_Y:
            LOGGER.warning("Adjust the values of kernel_init and kernel_noise in accordance with the data")

        # number of restarts
        if n_restarts is None or n_restarts < 1:
            LOGGER.warning("Setting number of restarts to 1")
            n_restarts = self.np_int(1)
        else:
            n_restarts = self.np_int(n_restarts)

        kernel_min = tf.constant(kernel_min, dtype=self.tf_float)

        final_losses = []
        model_params = []
        # optimize the GP repeatedly
        for i in range(n_restarts):
            try:
                # sample random shifts to the parameters
                kernel_variance = tf.random.normal(shape=(), dtype=self.tf_float, stddev=kernel_noise)
                kernel_lengthscales = tf.random.normal(
                    shape=self.lengthscale_shape,
                    dtype=self.tf_float,
                    stddev=kernel_noise,
                )

                # set new starting values for this training run
                if full_restart:
                    kernel_variance += kernel_init
                    kernel_lengthscales += kernel_init
                    q_mu = tf.zeros_like(self.model.q_mu)
                    q_sqrt = tf.eye(self.model.q_sqrt.shape[-1], batch_shape=[1], dtype=self.tf_float)

                else:
                    # read in the previous parameters
                    previous_params = self._readout_params()

                    kernel_variance += previous_params[0]
                    kernel_lengthscales += previous_params[1]
                    q_mu = previous_params[2]
                    q_sqrt = previous_params[3]

                # ensure the lower bound
                kernel_variance = tf.maximum(kernel_variance, kernel_min)
                kernel_lengthscales = tf.maximum(kernel_lengthscales, kernel_min)

                # assign the values
                self._set_params((kernel_variance, kernel_lengthscales, q_mu, q_sqrt))

                # optimize
                losses = self.fit_model(n_steps, learning_rate)

                # append the final loss value
                final_losses.append(losses[-1])
                model_params.append(self._readout_params())

                LOGGER.info(f"Training run {i}: loss = {final_losses[-1]}")
            except:
                LOGGER.warning(f"Training run {i} failed")

        # set to minimum
        min_index = np.argmin(final_losses)
        LOGGER.info(f"Training run {min_index} was best, setting those parameters")
        self._set_params(model_params[min_index])

        self.print_summary()

    def _readout_params(self):
        """Helper function to read out the model parameters in a specific fixed ordering.

        Return:
            Tupel: All trainable model parameters.
        """

        params = (
            self.model.kernel.variance.numpy(),
            self.model.kernel.lengthscales.numpy(),
            self.model.q_mu.numpy(),
            self.model.q_sqrt.numpy(),
        )

        return params

    def _set_params(self, params: np.ndarray):
        """Helper function to set the model parameters.

        Args:
            params (Tuple): Tuple of arrays in the same ordering as in _readout_params.
        """

        # scale in y axis, see https://gpflow.github.io/GPflow/2.7.0/notebooks/getting_started/kernels.html#Kernel-parameters
        self.model.kernel.variance.assign(params[0])
        # scale in x axis
        self.model.kernel.lengthscales.assign(params[1])
        # variational parameters of the VGP
        self.model.q_mu.assign(params[2])
        self.model.q_sqrt.assign(params[3])

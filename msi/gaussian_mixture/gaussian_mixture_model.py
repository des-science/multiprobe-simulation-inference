# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2023
Author: Arne Thomsen, Tomasz Kacprzak

Adapted from https://github.com/tomaszkacprzak/deep_lss/blob/main/deep_lss/networks/likemdn.py by Tomasz Kacprzak
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os, warnings, random, pickle, keras_tuner

from scipy.stats import norm
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from msfm.utils import logger
from msi.utils.keras import EpochProgressCallback

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class ConditionalGMM:
    """
    Conditional Gaussian Mixture Model (GMM), modelling p(y|x), where len(x) == len(y).
    Code adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/MixtureNormal#methods_2
    """

    def __init__(self, nx, ny, out_dir=None, tune_hyperparams=False, restore_weights=False, **kwargs):
        """Constructor

        Args:
            nx (int): Dimensionality of x, the condition.
            ny (int): Dimensionality of y, the random variable.
            out_dir (str, optional): Where to store the checkpoints and TensorBoard summaries. Defaults to None,
                then neither are saved.
            tune_hyperparams (bool, optional): Whether to run a random hyperparameter search or only train a single
                model.
        """
        self.nx = nx
        self.ny = ny
        self.out_dir = out_dir
        self.tune_hyperparams = tune_hyperparams

        self.gmm_config = {}
        # probability
        self.gmm_config.setdefault("n_gaussians", 4)
        # network architecture
        self.gmm_config.setdefault("u_units", 256)
        self.gmm_config.setdefault("n_layers", 3)
        self.gmm_config.setdefault("activation", "relu")
        # optimization
        self.gmm_config.setdefault("learning_rate", 1e-3)
        self.gmm_config.setdefault("global_clipnorm", 1.0)
        self.gmm_config.setdefault("dropout_rate", 0.0)
        self.gmm_config.setdefault("x_noise_sigma", 0.0)
        self.gmm_config.setdefault("validation_frac", 0.2)
        # hyperparameter tuning
        self.gmm_config.setdefault("max_trials", 10)

        # update the gmm_config keys
        for k in set(kwargs.keys()) & set(self.gmm_config.keys()):
            self.gmm_config[k] = kwargs[k]

        self.scaler_x = None
        self.scaler_y = None

        # create network
        self.build_model()

        if out_dir is not None:
            self.checkpoint_dir = os.path.join(out_dir, "checkpoint")
            self.summary_dir = os.path.join(out_dir, "tensor_board")

            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.summary_dir, exist_ok=True)
        else:
            self.checkpoint_dir = None
            self.summary_dir = None

        if tune_hyperparams:
            assert out_dir is not None, f"The hyperparameters can only be tuned if a out_dir is passed"

            self.build_tuner()

        if restore_weights:
            self.load()

    def fit(self, x, y, epochs=1000, batch_size=1000):
        """Wrapper for model.fit from the Keras API. The GMM implements p(y|x).

        Args:
            x (np.ndarray): Conditioning of shape (n_samples, n_conditions).
            y (np.ndarray): Features to be modelled by the Gaussians of shape (n_samples, n_features).
            epochs (int, optional): Number of epochs to train for. Defaults to 10000.
            batch_size (int, optional): Batch size used during training. Defaults to 1024.

        Returns:
            history: A keras history object containing information on the training.
        """

        # check dimensions
        assert (
            x.ndim == 2 and y.ndim == 2
        ), "Something is wrong with the x and y array dimensions, they should be =2 for training"
        assert x.shape[1] == self.nx, f"x should have {self.nx} dimensions"
        assert y.shape[1] == self.ny, f"y should have {self.ny} dimensions"
        assert x.shape[0] == y.shape[0], "x and y should have the same number of points"

        # preprocessing
        self.set_scalers(x, y)
        x = self.scale_forward_x(x)
        y = self.scale_forward_y(y)

        callbacks = []

        # TensorBoard summary writer
        if self.summary_dir is not None:
            callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.summary_dir)
            callbacks.append(callback_tensorboard)

        # learning rate scheduler
        callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.75, patience=20, verbose=0, min_delta=1e-4, cooldown=10, min_lr=1e-6
        )
        callbacks.append(callback_reduce_lr)

        # early stopping
        callback_early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-5, patience=100, verbose=1, restore_best_weights=True
        )
        callbacks.append(callback_early_stop)

        # verbosity
        callback_verbose = EpochProgressCallback(epochs)
        callbacks.append(callback_verbose)

        # hyperparameter tuning
        if self.tune_hyperparams:
            self.tuner.search(
                x,
                y,
                validation_split=self.gmm_config["validation_frac"],
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
                callbacks=[callback_reduce_lr, callback_early_stop, callback_verbose],
                verbose=0,
            )

            # keep the best model
            models = self.tuner.get_best_models(num_models=1)
            self.model = models[0]

            LOGGER.info("\n")
            self.tuner.results_summary()
            LOGGER.info(f"Finished the hyperparameter search. The best model is kept")

            # TODO is there a way to recover the keras history object even in this case?
            history = None

        # training of a single model
        else:
            history = self.model.fit(
                x,
                y,
                validation_split=self.gmm_config["validation_frac"],
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                shuffle=True,
                callbacks=callbacks,
            )

            LOGGER.info(f"Finished training")

        # save the model
        if self.checkpoint_dir is not None:
            self.save()

        return history

    def set_scalers(self, x, y):
        eps = 1e-5
        self.scaler_x = RobustScaler()
        self.scaler_y = MinMaxScaler(feature_range=(eps, 1 - eps))
        self.scaler_x.fit(x)
        self.scaler_y.fit(y)

        LOGGER.info(f"Fitted the x and y scalers")

    def scale_forward_x(self, x):
        return self.scaler_x.transform(x)

    def scale_inverse_x(self, x):
        return self.scaler_x.inverse_transform(x)

    def scale_forward_y(self, y):
        return norm.ppf(self.scaler_y.transform(y))

    def scale_inverse_y(self, y):
        return self.scaler_y.inverse_transform(norm.cdf(y))

    def log_likelihood(self, x, y):
        """log likelihood p(y|x) in numpy, meant to be used together with an MCMC.

        Args:
            x (np.ndarray): Conditioning of shape (n_samples, n_conditions).
            y (np.ndarray): Features to be modelled by the Gaussians of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of shape (n_samples,) containing the log likelihoods.
        """
        x = self.scale_forward_x(x)
        y = self.scale_forward_y(y)

        return self.model(x).log_prob(y).numpy()

    def build_model(self):
        """Wrapper function to build the GMM model."""
        self.model = build_gmm_network(
            nx=self.nx,
            ny=self.ny,
            # probability
            n_gaussians=self.gmm_config["n_gaussians"],
            # architecture
            u_units=self.gmm_config["u_units"],
            n_layers=self.gmm_config["n_layers"],
            activation=self.gmm_config["activation"],
            # optimization
            dropout_rate=self.gmm_config["dropout_rate"],
            x_noise_sigma=self.gmm_config["x_noise_sigma"],
            learning_rate=self.gmm_config["learning_rate"],
            global_clipnorm=self.gmm_config["global_clipnorm"],
        )

        LOGGER.info(f"Created GMM model with n_params = {self.model.count_params()}")
        self.model.summary()

    def build_tuner(self):
        """Build the tuner to search the hyperparameter space"""

        # specify the fixed hyperparameters excluded from the search here
        hyper_model = HyperGMM(
            nx=self.nx,
            ny=self.ny,
            # probability
            n_gaussians=self.gmm_config["n_gaussians"],
            # architecture
            u_units=self.gmm_config["u_units"],
            n_layers=self.gmm_config["n_layers"],
            activation=self.gmm_config["activation"],
            # optimization
            global_clipnorm=self.gmm_config["global_clipnorm"],
        )

        self.tuner = keras_tuner.RandomSearch(
            hypermodel=hyper_model,
            objective="val_loss",
            max_trials=self.gmm_config["max_trials"],
            executions_per_trial=1,
            overwrite=True,
            directory=self.out_dir,
            project_name="tuner_{:x}".format(random.getrandbits(32)),
        )
        LOGGER.info("Created GMM tuner: \n")
        self.tuner.search_space_summary()

    def save(self):
        self.save_scalers()
        self.model.save_weights(os.path.join(self.checkpoint_dir, "GMM"), save_format="tf")
        LOGGER.info(f"Saved the model in {self.out_dir}")

    def save_scalers(self):
        with open(os.path.join(self.checkpoint_dir, "scalers.pkl"), "wb") as f:
            pickle.dump([self.scaler_x, self.scaler_y], f)

        # write_to_pickle(scaler_file, [self.scaler_x, self.scaler_y])

    def load(self):
        self.model.load_weights(os.path.join(self.checkpoint_dir, "GMM"))

        with open(os.path.join(self.checkpoint_dir, "scalers.pkl"), "rb") as f:
            self.scaler_x, self.scaler_y = pickle.load(f)

        # self.scaler_x, self.scaler_y = read_from_pickle()
        LOGGER.info(f"Loaded the network from {self.out_dir}")

    def sample(self, x, n_samples, batch_size=10000):
        """Generate samples y ~ p(y|x) given x.

        Args:
            x (np.ndarray): Values to condition on of shape (n_samples, n_conditions).
            n_samples (int): Number of samples to draw.
            batch_size (int, optional): Draw the samples batch wise. Defaults to 10000.

        Returns:
            np.ndarray: Samples of shape (n_samples, n_features)
        """
        assert self.scaler_x is not None, "the MDN has not been fit yet"
        assert self.scaler_y is not None, "the MDN has not been fit yet"

        n_batches = np.ceil(len(x) / batch_size)
        ys = []
        m = self.model(np.expand_dims(x[0, :], 0)).sample(sample_shape=1).shape[-1]
        n = x.shape[0]
        LOGGER.info(f"drawing {n} samples with {m} dims")
        for x_ in LOGGER.progressbar(
            np.array_split(x, n_batches), desc=f"drawing samples with batch_size={batch_size}", at_level="info"
        ):
            x_ = self.scale_forward_x(x_)
            ys_ = tf.squeeze(self.model(x_).sample(sample_shape=n_samples))
            ys_ = self.scale_inverse_y(ys_)
            ys.append(ys_)

        ys = np.concatenate(ys)
        return ys


class HyperGMM(keras_tuner.HyperModel):
    def __init__(self, **model_args):
        self.model_args = model_args

    def build(self, hp):
        # NOTE specify the hyperparameter search space here
        dropout_rate = hp.Float(name="dropout_rate", min_value=0, max_value=0.25, step=0.05, default=0)
        x_noise_sigma = hp.Float(name="x_noise_sigma", min_value=0, max_value=0.1, step=0.01, default=0)
        learning_rate = hp.Float(
            name="learning_rate", min_value=1e-6, max_value=1e-2, step=10, sampling="log", default=1e-3
        )

        return build_gmm_network(
            dropout_rate=dropout_rate, x_noise_sigma=x_noise_sigma, learning_rate=learning_rate, **self.model_args
        )

    def fit(self, hp, model, *args, **kwargs):
        # the training is very slow for small batch sizes
        # batch_size = hp.Int(name="batch_size", min_value=100, max_value=10000, step=10, sampling="log")
        # return model.fit(*args, batch_size=batch_size, **kwargs)

        return model.fit(*args, **kwargs)


def build_gmm_network(
    nx,
    ny,
    # probability
    n_gaussians,
    # network architecture
    u_units,
    n_layers,
    activation,
    # optimization
    dropout_rate,
    x_noise_sigma,
    learning_rate,
    global_clipnorm,
):
    """Build a Gaussian Mixture Model network that is able to learn p(y|x) from samples.

    Args:
        nx (int): Dimensionality of x.
        ny (int): Dimensionality of y.
        u_units (int): Number of neurons in the hidden layers.
        n_layers (int): Number of hidden layers (there's two more dense layers in the model than this).
        activation (int): Activation function to use.
        n_gaussians (int): Number of Gaussians in the mixture.
        dropout_rate (int): Dropout rate to use during training.
        x_noise_sigma (float): Standard deviatian of Gaussian noise added to the x conditionals.

    Returns:
        tf.keras.model: The model with correctly initialized weights.
    """

    model = tf.keras.Sequential(name="gaussian_mixture_model")
    model.add(tf.keras.layers.Input(shape=(nx,)))
    model.add(tf.keras.layers.GaussianNoise(x_noise_sigma))
    model.add(tf.keras.layers.Dense(u_units, input_dim=nx, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))

    for _ in range(n_layers):
        model.add(tf.keras.layers.Dense(u_units, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # number of parameters needed to build the final distribution layer
    gmm_param_size = tfp.layers.MixtureSameFamily.params_size(
        n_gaussians, component_params_size=tfp.layers.MultivariateNormalTriL.params_size(ny)
    )
    model.add(tf.keras.layers.Dense(gmm_param_size, activation=None))
    model.add(tfp.layers.MixtureSameFamily(n_gaussians, tfp.layers.MultivariateNormalTriL(ny)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=global_clipnorm)

    # maximize the log likelihood of y ~ p(y|x)
    model.compile(optimizer=optimizer, loss=lambda y, model: -model.log_prob(y))

    return model

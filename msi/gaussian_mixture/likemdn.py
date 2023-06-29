"""
Adapted from https://github.com/tomaszkacprzak/deep_lss/blob/main/deep_lss/networks/likemdn.py by Tomasz Kazcprac
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os, h5py, itertools, warnings, random, pickle, keras_tuner

from scipy.stats import norm
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from msfm.utils import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logging.get_logger(__file__)


class LikeMDN:
    """
    Conditional Mixtude Density Network (MDN), modelling p(y|x).
    Works for len(x)=len(y)
    Code adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/MixtureNormal#methods_2
    """

    def __init__(self, nx, ny, filename_checkpoint=None, **kwargs):
        """
        Constructor.
        :param nd: number of dimesnions
        """

        self.nx = nx
        self.ny = ny
        self.mdn_config = {}
        self.mdn_config.setdefault("n_gaussians", 1)
        self.mdn_config.setdefault("u_units", 128)
        self.mdn_config.setdefault("n_layers", 2)
        self.mdn_config.setdefault("activation", "relu")
        self.mdn_config.setdefault("learning_rate", 1e-2)
        self.mdn_config.setdefault("dropout_rate", 0)
        self.mdn_config.setdefault("input_noise_sig", 0)
        self.filename_checkpoint = filename_checkpoint

        # update nn_config keys
        for k in set(kwargs.keys()) & set(self.mdn_config.keys()):
            self.mdn_config[k] = kwargs[k]

        self.scaler_x = None
        self.scaler_y = None

        # create network
        self.build_model()
        # self.build_tuner()

    def fit(self, x, y, epochs=10000, verb=1, batch_size=100):
        self.set_scalers(x, y)
        self.save_scalers(self.filename_checkpoint)
        x = self.scale_forward_x(x)
        y = self.scale_forward_y(y)

        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        # check dimensions
        assert (
            x.ndim == 2 and y.ndim == 2
        ), "something wrong with y array dimensions, it should be =2 for likenet training"
        assert x.shape[1] == self.nx, f"x should have {self.nx} dimensions"
        assert y.shape[1] == self.ny, f"x should have {self.ny} dimensions"
        assert x.shape[0] == y.shape[0], "x and y should have the same number of points"

        callbacks = []

        # learning rate scheduler
        callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.75, patience=10, verbose=1, min_delta=0.0001, cooldown=5, min_lr=1e-6
        )
        callbacks.append(callback_reduce_lr)

        # early stopping
        callback_early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-5, patience=50, verbose=1, restore_best_weights=True
        )
        callbacks.append(callback_early_stop)

        # checkpointing
        if self.filename_checkpoint is not None:

            callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.filename_checkpoint,
                                                                     monitor='val_loss',
                                                                     verbose=1,
                                                                     save_best_only=True,
                                                                     save_weights_only=True,
                                                                     mode='auto',
                                                                     save_freq='epoch')
            callbacks.append(callback_checkpoint)

        validation_frac = 0.2
        steps_per_epoch = min(100, int((1 - validation_frac) * x.shape[0] // batch_size))
        LOGGER.info(f"fitting with {epochs} epoch with {steps_per_epoch} steps")
        LOGGER.info(f"input_noise_sig={self.mdn_config['input_noise_sig']}")

        # self.tuner.search(x, y,
        #                   validation_split=validation_frac,
        #                   # batch_size=batch_size,
        #                   epochs=epochs,
        #                   shuffle=True,
        #                   callbacks=[callback_reduce_lr, callback_early_stop],
        #                   verbose=verb)

        # models = self.tuner.get_best_models(num_models=1)
        # self.model = models[0]

        self.model.fit(
            x,
            y,
            validation_split=validation_frac,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verb,
            shuffle=True,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
        )

        LOGGER.info(f"finished fitting with {epochs} epochs")

    def set_scalers(self, x, y):
        eps = 1e-5
        self.scaler_x = RobustScaler()
        self.scaler_y = MinMaxScaler(feature_range=(eps, 1 - eps))
        self.scaler_x.fit(x)
        self.scaler_y.fit(y)

    def scale_forward_x(self, x):
        return self.scaler_x.transform(x)

    def scale_forward_y(self, y):
        return norm.ppf(self.scaler_y.transform(y))

    def scale_inverse_x(self, x):
        return self.scaler_x.inverse_transform(x)

    def scale_inverse_y(self, y):
        return self.scaler_y.inverse_transform(norm.cdf(y))

    def log_like(self, x, y):
        x = self.scale_forward_x(x)
        y = self.scale_forward_y(y)

        return self.model(x).log_prob(y).numpy()

    def build_model(self):
        self.model = build_mdn_network(
            nx=self.nx,
            ny=self.ny,
            u_units=self.mdn_config["u_units"],
            n_layers=self.mdn_config["n_layers"],
            activation=self.mdn_config["activation"],
            n_gaussians=self.mdn_config["n_gaussians"],
            dropout_rate=self.mdn_config["dropout_rate"],
            input_noise_sig=self.mdn_config["input_noise_sig"],
        )

        LOGGER.info(f"created MDN model with n_params={self.model.count_params()}")
        self.model.summary()

    def build_tuner(self):
        hmodel = hyper_model(
            nx=self.nx,
            ny=self.ny,
            u_units=self.mdn_config["u_units"],
            n_layers=self.mdn_config["n_layers"],
            n_gaussians=self.mdn_config["n_gaussians"],
            activation=self.mdn_config["activation"],
        )

        self.tuner = keras_tuner.RandomSearch(
            hypermodel=hmodel,
            objective="val_loss",
            max_trials=20,
            executions_per_trial=1,
            overwrite=True,
            directory=os.path.dirname(self.filename_checkpoint),
            project_name="tuner_{:x}".format(random.getrandbits(32)),
        )
        LOGGER.info("created MDN tuner: \n")
        self.tuner.search_space_summary()

    def save(self, file_out):
        self.model.save_weights(file_out)
        LOGGER.info(f"saved {file_out}")
        self.save_scalers(file_out)

    def save_scalers(self, file_out):
        write_to_pickle(get_filename_scalers(file_out), [self.scaler_x, self.scaler_y])

    def load(self, file_in):
        self.model.load_weights(file_in)
        self.scaler_x, self.scaler_y = read_from_pickle(get_filename_scalers(file_in))
        LOGGER.info(f"loaded {file_in}")

    def sample(self, x, n_samples, batch_size=10000):
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


def get_filename_scalers(file_base):
    return f"{file_base}_scalers.pkl"


def build_mdn_network(nx, ny, u_units, n_layers, activation, n_gaussians, dropout_rate, input_noise_sig):
    gmm_param_size = tfp.layers.MixtureSameFamily.params_size(
        n_gaussians, component_params_size=tfp.layers.MultivariateNormalTriL.params_size(ny)
    )

    model = tf.keras.Sequential(name="likemdn")
    model.add(tf.keras.layers.Input(shape=(nx,)))
    model.add(tf.keras.layers.GaussianNoise(input_noise_sig))
    model.add(tf.keras.layers.Dense(u_units, input_dim=nx, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))

    for _ in range(n_layers):
        model.add(tf.keras.layers.Dense(u_units, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(gmm_param_size, activation=None))

    model.add(tfp.layers.MixtureSameFamily(n_gaussians, tfp.layers.MultivariateNormalTriL(ny)))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=lambda y, model: -model.log_prob(y))

    return model


def build_mdn_network_tune(hp, nx, ny, activation, u_units, n_layers, n_gaussians):
    dropout_rate = hp.Float(name="dropout_rate", min_value=0, max_value=0.25, step=0.01)

    model = build_mdn_network(nx, ny, u_units, n_layers, activation, n_gaussians, dropout_rate=dropout_rate)

    return model


class hyper_model(keras_tuner.HyperModel):
    def __init__(self, **model_args):
        self.model_args = model_args

    def build(self, hp):
        dropout_rate = hp.Float(name="dropout_rate", min_value=0, max_value=0.25, step=0.01)

        return build_mdn_network(dropout_rate=dropout_rate, **self.model_args)

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Int(name="batch_size", min_value=10, max_value=10000, step=1, sampling="log")

        return model.fit(*args, batch_size=batch_size, **kwargs)
    
# utils from DeepLSS
def write_to_pickle(filepath, obj, compression='none', compression_kwargs={}):

    compression_kwargs.setdefault('compresslevel', 4)
    compression_types =  ['none', 'lzf', 'bz2', 'gzip']

    if compression.lower()=='none':
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    elif compression.lower()=='lzf':
        import lzf
        with lzf.open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    elif compression.lower()=='bz2':
        import bz2
        with bz2.open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    elif compression.lower()=='gzip':
        import gzip
        with gzip.open(filepath, 'wb', **compression_kwargs) as f:
            pickle.dump(obj, f)

    else: 
        raise Exception('uknown compression {}, use {}'.format(compression, str(formats)))
    LOGGER.info(f'wrote {filepath}')


def read_from_pickle(filepath, compression='none'):

    compression_types =  ['none', 'lzf', 'bz2', 'gzip']

    if compression.lower()=='none':
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
    elif compression.lower()=='lzf':
        import lzf
        with lzf.open(filepath, 'rb') as f:
            obj = pickle.load(f)
    elif compression.lower()=='bz2':
        import bz2
        with bz2.open(filepath, 'rb') as f:
            obj = pickle.load(f)
    elif compression.lower()=='gzip':
        import gzip
        with gzip.open(filepath, 'rb') as f:
            obj = pickle.load(f)
    else: 
        raise Exception('uknown compression {}, use {}'.format(compression, str(formats)))

    LOGGER.debug(f'read {filepath}')
    return obj


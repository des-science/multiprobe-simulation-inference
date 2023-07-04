# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2023
Author: Arne Thomsen
"""

import tensorflow as tf
from sklearn.preprocessing import RobustScaler

from msfm.utils import logger
from msi.utils.keras import EpochProgressCallback

LOGGER = logger.get_logger(__file__)


class DenseEmulator(tf.keras.Model):
    """Simple class to emulate a summary statistic throughout (cosmological) parameter space using dense layers. This
    implementation subclasses tf.keras.Model, such that the keras functional API cann be conveniently used in the call
    method."""

    def __init__(
        self,
        x_dim,
        y_dim,
        # architecture
        n_units=256,
        n_layers=5,
        activation="relu",
        # optimization
        learning_rate=1e-3,
        global_clipnorm=None,
        dropout_rate=0.2,
        **kwargs,
    ):
        """Initialize the model.

        Args:
            x_dim (int): Size of the input dimension.
            y_dim (int): Size of the output dimension.
            n_units (int, optional): Number of nodes in the hidden layers. Defaults to 256.
            n_layers (int, optional): Number of hidden layers. Defaults to 5.
            activation (str, optional): Activation function. Defaults to "relu".
            learning_rate (float, optional): Learning rate to be used in the Adam optimizer. Defaults to 1e-3.
            global_clipnorm (float, optional): Global gradient norm to clip in Adam. Defaults to None.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
            **kwargs: additional keyword arguments to be passed to the optimizer.
        """

        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_units = n_units
        self.n_layers = n_layers
        self.activation = activation
        self.dropout_rate = dropout_rate

        # preprocessing
        self.x_mean = 0.0
        self.x_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0

        # layers
        self.build_layers()
        self.summary()

        # compilation
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, global_clipnorm=global_clipnorm, **kwargs
        )
        self.compile(optimizer=self.optimizer, loss="mse")

    def build_layers(self):
        """Set up the layers according to the specified hyperparameters."""
        self.dense_layers = [
            tf.keras.layers.Dense(self.n_units, activation=self.activation) for _ in range(self.n_layers)
        ]
        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate) for _ in range(self.n_layers)]
        self.output_layer = tf.keras.layers.Dense(self.y_dim)

    def summary(self):
        """Network summary has to be called in this way since model.build() doesn't get along with the functional API."""
        x = tf.keras.Input(shape=(self.x_dim,))
        tf.keras.Model(inputs=x, outputs=self.call(x)).summary()

    def call(self, inputs):
        """The forward pass. This allows to include the rescaling of inputs and outputs within the network's graph.

        Args:
            inputs (tf.tensor): The input data of shape (None, x_dim), where the batch dimension has size None.

        Returns:
            outputs (tf.tensor): The output data of shape (None, y_dim).
        """
        x = self.scale_forward_x(inputs)

        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            x = dense_layer(x)
            x = dropout_layer(x)

        y = self.output_layer(x)
        output = self.scale_inverse_y(y)

        return output

    def fit(
        self,
        x,
        y,
        epochs=1000,
        batch_size=256,
        # validation
        validation_split=0.0,
        validation_data=None,
        # callbacks
        early_stopping_callback=False,
        learning_rate_callback=False,
        **kwargs,
    ):
        """Fit the network to the training set.

        Args:
            x (tf.tensor): Input features of shape (n_samples, x_dim).
            y (tf.tensor): Output labels of shape (n_samples, y_dim).
            epochs (int, optional): Number of epochs to train. Defaults to 1000.
            batch_size (int, optional): Batch size. Defaults to 256.
            validation_split (float, optional): Fraction of x and y to use as the validation set. Defaults to 0.0.
            validation_data (tuple, optional): Explicit validation set like (x_vali, y_vali). Defaults to None.
            early_stopping_callback (bool, optional): Whether to use the early stopping callback. Defaults to False,
                then it is excluded.
            learning_rate_callback (bool, optional): Whether to use the learning rate reduction on plateau callback.
                Defaults to False, then it is excluded.
            **kwargs: additional keyword arguments to be passed to the fit method.

        Returns:
            history: A keras history object containing information on the training procedure.
        """

        # preprocessing
        self.fit_scalers(x, y)

        callbacks = []

        # early stopping
        if early_stopping_callback:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", min_delta=1e-5, patience=100, verbose=1, restore_best_weights=True
                )
            )

        # learning rate
        if learning_rate_callback:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=10, verbose=0, min_delta=1e-4, cooldown=10, min_lr=1e-6
                )
            )

        # verbosity
        callbacks.append(EpochProgressCallback(epochs))

        # self.fit(
        history = super().fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )

        return history

    def fit_scalers(self, x, y):
        """Determine the mean and standard deviation of x and y to standardize them.

        Args:
            x (tf.tensor): Input features of shape (n_samples, x_dim).
            y (tf.tensor): Output labels of shape (n_samples, y_dim).
        """
        self.x_mean = tf.reduce_mean(x, axis=0)
        self.x_std = tf.math.reduce_std(x, axis=0)

        self.y_mean = tf.reduce_mean(y, axis=0)
        self.y_std = tf.math.reduce_std(y, axis=0)

        LOGGER.info(f"Preprocessing scalers have been successfully fit")

    def scale_forward_x(self, x):
        return (x - self.x_mean) / self.x_std

    def scale_inverse_x(self, x_scaled):
        return (x_scaled * self.x_std) + self.x_mean

    def scale_forward_y(self, y):
        return (y - self.y_mean) / self.y_std

    def scale_inverse_y(self, y_scaled):
        return (y_scaled * self.y_std) + self.y_mean


class DenseEmulatorDeprecated:
    """Simple class to emulate a summary statistic throughout (cosmological) parameter space using dense layers. This
    version doesn't subclass keras.Model and therefore functions slightly differently."""

    def __init__(
        self,
        x_dim,
        y_dim,
        # architecture
        n_units=256,
        n_layers=5,
        activation="relu",
        # optimization
        learning_rate=1e-3,
        global_clipnorm=None,
        dropout_rate=0.2,
        **kwargs,
    ):
        """Initialize the model.

        Args:
            x_dim (int): Size of the input dimension.
            y_dim (int): Size of the output dimension.
            n_units (int, optional): Number of nodes in the hidden layers. Defaults to 256.
            n_layers (int, optional): Number of hidden layers. Defaults to 5.
            activation (str, optional): Activation function. Defaults to "relu".
            learning_rate (float, optional): Learning rate to be used in the Adam optimizer. Defaults to 1e-3.
            global_clipnorm (float, optional): Global gradient norm to clip in Adam. Defaults to None.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
            **kwargs: additional keyword arguments to be passed to the optimizer.
        """

        self.scaler_x = None
        self.scaler_y = None

        # build the network
        self.model = tf.keras.Sequential()

        # input
        self.model.add(tf.keras.Input(shape=(x_dim,)))

        # hidden
        for _ in range(n_layers):
            self.model.add(tf.keras.layers.Dense(n_units, activation=activation))
            self.model.add(tf.keras.layers.Dropout(dropout_rate))

        # output
        self.model.add(tf.keras.layers.Dense(y_dim))
        self.model.summary()

        # compile the network
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=global_clipnorm, **kwargs)
        loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(
        self,
        x,
        y,
        epochs=1000,
        batch_size=256,
        # validation
        validation_split=0.0,
        validation_data=None,
        # callbacks
        early_stopping_callback=False,
        learning_rate_callback=False,
        **kwargs,
    ):
        """Fit the network to the training set.

        Args:
            x (tf.tensor): Input features of shape (n_samples, x_dim)
            y (tf.tensor): Output labels of shape (n_samples, y_dim)
            epochs (int, optional): Number of epochs to train. Defaults to 1000.
            batch_size (int, optional): Batch size. Defaults to 256.
            validation_split (float, optional): Fraction of x and y to use as the validation set. Defaults to 0.0.
            validation_data (tuple, optional): Explicit validation set like (x_vali, y_vali). Defaults to None.
            early_stopping_callback (bool, optional): Whether to use the early stopping callback. Defaults to False,
                then it is excluded.
            learning_rate_callback (bool, optional): Whether to use the learning rate reduction on plateau callback.
                Defaults to False, then it is excluded.

        Returns:
            history: A keras history object containing information on the training procedure.
        """

        # preprocessing
        self.set_scalers(x, y)

        x = self.scale_forward_x(x)
        y = self.scale_forward_y(y)

        if validation_data is not None:
            assert (type(validation_data) == tuple) and (len(validation_data) == 2)
            x_val = self.scale_forward_x(validation_data[0])
            y_val = self.scale_forward_y(validation_data[1])
            validation_data = (x_val, y_val)

        callbacks = []

        # early stopping
        if early_stopping_callback:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", min_delta=1e-5, patience=100, verbose=1, restore_best_weights=True
                )
            )

        # learning rate
        if learning_rate_callback:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=10, verbose=0, min_delta=1e-4, cooldown=10, min_lr=1e-6
                )
            )

        # verbosity
        callbacks.append(EpochProgressCallback(epochs))

        history = self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
            **kwargs,
        )

        return history

    def __call__(self, x):
        return self.model(x)

    def scaled_call(self, x):
        x = self.scale_forward_x(x)
        y_scaled = self.model(x)
        y = self.scale_inverse_y(y_scaled)

        return y

    def set_scalers(self, x, y):
        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()
        self.scaler_x.fit(x)
        self.scaler_y.fit(y)

        LOGGER.info(f"Fitted the x and y scalers")

    def scale_forward_x(self, x):
        return self.scaler_x.transform(x)

    def scale_inverse_x(self, x):
        return self.scaler_x.inverse_transform(x)

    def scale_forward_y(self, y):
        return self.scaler_y.transform(y)

    def scale_inverse_y(self, y):
        return self.scaler_y.inverse_transform(y)

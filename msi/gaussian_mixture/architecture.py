# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2024
Author: Arne Thomsen
"""

import tensorflow as tf
import tensorflow_probability as tfp


def get_gmm_layers(
    n_x,
    n_theta,
    # probability
    n_gaussians=4,
    # network architecture
    n_units=256,
    n_layers=4,
    activation="relu",
    # optimization
    dropout_rate=0.0,
    x_noise_sigma=0.0,
):
    """Build a Gaussian Mixture Model network that is able to learn the likelihood p(x|theta) from samples.

    Args:
        n_x (int): Dimensionality of the feature variable x.
        n_theta (int): Dimensionality of the conditioning variable theta.
        n_units (int): Number of neurons in the hidden layers.
        n_layers (int): Number of hidden layers (there's two more dense layers in the model than this).
        activation (int): Activation function to use.
        n_gaussians (int): Number of Gaussians in the mixture.
        dropout_rate (int): Dropout rate to use during training.
        x_noise_sigma (float): Standard deviatian of Gaussian noise added to the x conditionals.

    Returns:
        tf.keras.model: The model with correctly initialized weights.
    """

    layers = []
    layers.append(tf.keras.layers.Input(shape=(n_theta,)))
    layers.append(tf.keras.layers.GaussianNoise(x_noise_sigma))
    layers.append(tf.keras.layers.Dense(n_units, input_dim=n_theta, activation=activation))
    layers.append(tf.keras.layers.Dropout(dropout_rate))

    for _ in range(n_layers):
        layers.append(tf.keras.layers.Dense(n_units, activation=activation))
        layers.append(tf.keras.layers.Dropout(dropout_rate))

    # number of parameters needed to build the final distribution layer
    gmm_param_size = tfp.layers.MixtureSameFamily.params_size(
        n_gaussians, component_params_size=tfp.layers.MultivariateNormalTriL.params_size(n_x)
    )
    layers.append(tf.keras.layers.Dense(gmm_param_size, activation=None))
    layers.append(tfp.layers.MixtureSameFamily(n_gaussians, tfp.layers.MultivariateNormalTriL(n_x)))

    return layers

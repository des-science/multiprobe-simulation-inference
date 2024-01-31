# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen

Contains the components to build up conditional normalizng flows in FlowConductor.
"""

import torch

from enflows.distributions.normal import StandardNormal, DiagonalNormal, ConditionalDiagonalNormal
from enflows.transforms import (
    CompositeTransform,
    ActNorm,
    iResBlock,
    MaskedSumOfSigmoidsTransform,
    ConditionalSVDTransform,
)
from enflows.nn.nets import Sin, CSin, ResidualNet

default_context_embedding_dim = 16


def get_normal_dist(feature_dim, type="standard"):
    """Base distribution of the flow

    Args:
        feature_dim (int): The flow operates on vectors of this dimension.
        type (str, optional): The kind of normal distribution. Defaults to "standard".

    Raises:
        ValueError: If type is not one of "standard", "diagonal", "conditional_diagonal".

    Returns:
        enflows.distributions.base.Distribution: The base distribution of the flow.
    """

    if type == "standard":
        dist = StandardNormal(shape=(feature_dim,))
    elif type == "diagonal":
        dist = DiagonalNormal(shape=(feature_dim,))
    elif type == "conditional_diagonal":
        dist = ConditionalDiagonalNormal(shape=(feature_dim,))
    else:
        raise ValueError(f"Unknown distribution type {type}")

    return dist


def get_context_embedding_net(
    context_dim,
    context_embedding_dim=default_context_embedding_dim,
    hidden_dim=32,
    n_blocks=2,
    activation=torch.nn.functional.silu,
    dropout_probability=0.0,
    use_batch_norm=False,
):
    """
    Returns a context embedding network.

    Args:
        context_dim (int): The dimension of the input context.
        context_embedding_dim (int): The dimension of the output context embedding. Defaults to 16.
        hidden_dim (int, optional): The dimension of the hidden layers. Defaults to 32.
        num_blocks (int, optional): The number of residual blocks in the network. Defaults to 2.
        activation (function, optional): The activation function to use. Defaults to torch.nn.functional.silu.
        dropout_probability (float, optional): The probability of dropout. Defaults to 0.0.
        use_batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.

    Returns:
        torch.nn.Module: The context embedding network.
    """
    embedding_net = ResidualNet(
        in_features=context_dim,
        out_features=context_embedding_dim,
        hidden_features=hidden_dim,
        num_blocks=n_blocks,
        activation=activation,
        dropout_probability=dropout_probability,
        use_batch_norm=use_batch_norm,
    )

    return embedding_net


def get_lipschitz_transform(
    feature_dim, context_embedding_dim=default_context_embedding_dim, n_layers=3, hidden_dim=64, lipschitz_coeff=0.97
):
    """Returns a Lipschitz transform as used in
    https://github.com/FabricioArendTorres/FlowConductor/blob/b276174a2ecdd8c1c85e4fac2e47396a3f8997ea/examples/conditional_toy_2d.py
    NOTE that some of the hyperparameters are hardcoded and taken from that example.

    Args:
        feature_dim (int): The dimension of the input features.
        context_embedding_dim (int): The dimension of the context embedding. Defaults to 16.
        n_layers (int, optional): The number of layers in the transform. Defaults to 3.
        hidden_dim (int, optional): The dimension of the hidden layers. Defaults to 128.
        lipschitz_coeff (float, optional): The Lipschitz coefficient. Defaults to 0.97.

    Returns:
        CompositeTransform: A composite transform consisting of ActNorm and iResBlock layers.
    """

    densenet_factory = iResBlock.Factory()

    if feature_dim < 4:
        densenet_factory.set_logabsdet_estimator(brute_force=True)
    else:
        densenet_factory.set_logabsdet_estimator(brute_force=False, unbiased_estimator=True, trace_estimator="neumann")

    # hardcoded values from https://github.com/FabricioArendTorres/FlowConductor/blob/b276174a2ecdd8c1c85e4fac2e47396a3f8997ea/examples/conditional_toy_2d.py#L70
    densenet_factory.set_densenet(
        condition_input=True,
        condition_lastlayer=False,
        condition_multiplicative=True,
        # hyperparameters
        dimension=feature_dim,
        densenet_depth=3,
        densenet_growth=32,
        c_embed_hidden_sizes=(hidden_dim, hidden_dim, 10),
        m_embed_hidden_sizes=(hidden_dim, hidden_dim),
        activation_function=CSin(10),
        lip_coeff=lipschitz_coeff,
        context_features=context_embedding_dim,
    )

    transforms = []
    for _ in range(n_layers):
        transforms.append(ActNorm(feature_dim))
        transforms.append(densenet_factory.build())

    transform = CompositeTransform(transforms)

    return transform


def get_sigmoids_transform(
    feature_dim,
    context_embedding_dim=default_context_embedding_dim,
    n_layers=3,
    hidden_dim=64,
    svd_kwargs={},
    sigmoids_kwargs={},
):
    """Returns a transform consisting of a sequence of SVD and MaskedSumOfSigmoidsTransform layers.

    Args:
        feature_dim (int): The dimension of the input features.
        context_embedding_dim (int): The dimension of the context embedding. Defaults to 16.
        n_layers (int, optional): The number of layers in the transform. Defaults to 3.
        hidden_dim (int, optional): The dimension of the hidden layer. Defaults to 128.
        svd_kwargs (dict, optional): Keyword arguments for the ConditionalSVDTransform layer. Defaults to {}.
        sigmoids_kwargs (dict, optional): Keyword arguments for the MaskedSumOfSigmoidsTransform layer. Defaults to {}.

    Returns:
        CompositeTransform: The composite transform consisting of SVD and MaskedSumOfSigmoidsTransform layers.
    """

    svd_kwargs.setdefault("num_blocks", 2)
    svd_kwargs.setdefault("dropout_probability", 0.0)
    svd_kwargs.setdefault("activation", torch.nn.functional.relu)
    svd_kwargs.setdefault("use_batch_norm", False)

    sigmoids_kwargs.setdefault("n_sigmoids", 10)
    sigmoids_kwargs.setdefault("num_blocks", 2)
    sigmoids_kwargs.setdefault("dropout_probability", 0.0)
    sigmoids_kwargs.setdefault("activation", torch.nn.functional.relu)
    sigmoids_kwargs.setdefault("use_batch_norm", False)

    transforms = []
    for _ in range(n_layers):
        transforms.append(ActNorm(features=feature_dim))

        # this layer mixes the flow's dimensions
        transforms.append(
            ConditionalSVDTransform(
                features=feature_dim,
                hidden_features=hidden_dim,
                context_features=context_embedding_dim,
                **svd_kwargs,
            )
        )

        transforms.append(
            MaskedSumOfSigmoidsTransform(
                features=feature_dim,
                hidden_features=hidden_dim,
                context_features=context_embedding_dim,
                **sigmoids_kwargs,
            )
        )

    transform = CompositeTransform(transforms)

    return transform

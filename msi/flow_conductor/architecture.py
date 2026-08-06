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
from enflows.transforms.permutations import RandomPermutation
from enflows.transforms.lu import LULinear
from enflows.nn.nets import CSin, ResidualNet

from msi.flow_conductor.spline import RQSplineCouplingTransform, RQSplineAutoregressiveTransform
from msi.flow_conductor.maf import AffineAutoregressiveTransform

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


class StandardizedContextEmbedding(torch.nn.Module):
    """Fixed affine context standardization prepended to an embedding net.

    The flow's context (theta) enters the embedding net in physical units, whose per-parameter scales
    can differ by orders of magnitude (e.g. H0 ~ 70 vs n_Aia ~ 0.1 for the extended targets) -- the
    same conditioning problem as an unstandardized theta in the deep_lss VMIM head, one stage
    downstream. Call set_stats() once on the training thetas before fitting; the statistics are
    non-trainable buffers, so they persist through state_dict checkpoints and every context entry
    point (log_prob, sampling, MCMC) shares the identical transform. Until set_stats() is called the
    transform is the identity, reproducing the previous unstandardized behavior.
    """

    def __init__(self, embedding_net, context_dim):
        super().__init__()
        self.embedding_net = embedding_net
        self.register_buffer("context_shift", torch.zeros(context_dim))
        self.register_buffer("context_scale", torch.ones(context_dim))

    def set_stats(self, theta, eps=1e-8):
        theta = torch.as_tensor(theta, dtype=self.context_shift.dtype, device=self.context_shift.device)
        self.context_shift.copy_(theta.mean(dim=0))
        self.context_scale.copy_(theta.std(dim=0).clamp_min(eps))

    def forward(self, context):
        return self.embedding_net((context - self.context_shift) / self.context_scale)


def get_context_embedding_net(
    context_dim,
    context_embedding_dim=default_context_embedding_dim,
    hidden_dim=64,
    n_blocks=3,
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
    n_layers=4,
    hidden_dim=256,
    svd_kwargs=None,
    sigmoids_kwargs=None,
):
    """Returns a transform consisting of a sequence of SVD and MaskedSumOfSigmoidsTransform layers.

    Args:
        feature_dim (int): The dimension of the input features.
        context_embedding_dim (int): The dimension of the context embedding. Defaults to 16.
        n_layers (int, optional): The number of layers in the transform. Defaults to 4.
        hidden_dim (int, optional): The dimension of the hidden layer. Defaults to 256.
        svd_kwargs (dict, optional): Keyword arguments for the ConditionalSVDTransform layer. Defaults to {}.
        sigmoids_kwargs (dict, optional): Keyword arguments for the MaskedSumOfSigmoidsTransform layer. Defaults to {}.

    Returns:
        CompositeTransform: The composite transform consisting of SVD and MaskedSumOfSigmoidsTransform layers.
    """

    svd_kwargs = {} if svd_kwargs is None else dict(svd_kwargs)
    sigmoids_kwargs = {} if sigmoids_kwargs is None else dict(sigmoids_kwargs)

    svd_kwargs.setdefault("num_blocks", 2)
    svd_kwargs.setdefault("dropout_probability", 0.0)
    svd_kwargs.setdefault("activation", torch.nn.functional.relu)
    svd_kwargs.setdefault("use_batch_norm", False)

    sigmoids_kwargs.setdefault("n_sigmoids", 16)
    sigmoids_kwargs.setdefault("num_blocks", 3)
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


def get_spline_transform(
    feature_dim,
    context_embedding_dim=default_context_embedding_dim,
    n_layers=8,
    hidden_dim=128,
    num_bins=8,
    tail_bound=5.0,
    mask_type="coupling",
    num_blocks=2,
    dropout_probability=0.0,
    use_linear=True,
):
    """Returns a rational-quadratic neural spline transform implemented directly in PyTorch.

    This is a faster, smaller alternative to ``get_sigmoids_transform`` for the NLE setting, where
    the dominant cost is evaluating the flow's ``log_prob`` inside the MCMC posterior sampler -- the
    spline transforms have a single-pass ``log_prob``. Each layer is ``ActNorm`` -> a linear mixing
    layer (``LULinear`` if ``use_linear`` else ``RandomPermutation``) -> an RQ spline conditioner.

    Args:
        feature_dim (int): The dimension of the input features.
        context_embedding_dim (int): The dimension of the context embedding. Defaults to 16.
        n_layers (int, optional): The number of spline layers. Defaults to 8.
        hidden_dim (int, optional): Hidden width of the conditioner network. Defaults to 128.
        num_bins (int, optional): Number of spline bins. Defaults to 8.
        tail_bound (float, optional): The spline acts on [-tail_bound, tail_bound] and is the
            identity (linear tails) outside it. Defaults to 5.0.
        mask_type (str, optional): "coupling" (fast forward and inverse) or "autoregressive"
            (MAF-style: single-pass density, slow sampling). Defaults to "coupling".
        num_blocks (int, optional): Residual/MADE blocks in the conditioner. Defaults to 2.
        dropout_probability (float, optional): Conditioner dropout. Defaults to 0.0.
        use_linear (bool, optional): Use a learnable LU linear mix between layers instead of a
            fixed random permutation. Defaults to True.

    Returns:
        CompositeTransform: The composite RQ spline transform.
    """

    if mask_type not in ("coupling", "autoregressive"):
        raise ValueError(f"Unknown spline mask_type {mask_type!r}. Choose 'coupling' or 'autoregressive'.")

    transforms = []
    for i in range(n_layers):
        transforms.append(ActNorm(features=feature_dim))

        # mix the flow's dimensions between spline layers
        if use_linear:
            transforms.append(LULinear(feature_dim, identity_init=True))
        else:
            transforms.append(RandomPermutation(features=feature_dim))

        if mask_type == "coupling":
            transforms.append(
                RQSplineCouplingTransform(
                    feature_dim=feature_dim,
                    context_features=context_embedding_dim,
                    hidden_features=hidden_dim,
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                    num_blocks=num_blocks,
                    dropout_probability=dropout_probability,
                    mask_even=(i % 2 == 0),  # alternate the identity/transform split each layer
                )
            )
        else:
            transforms.append(
                RQSplineAutoregressiveTransform(
                    feature_dim=feature_dim,
                    context_features=context_embedding_dim,
                    hidden_features=hidden_dim,
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                    num_blocks=num_blocks,
                    dropout_probability=dropout_probability,
                )
            )

    transform = CompositeTransform(transforms)

    return transform


def get_maf_transform(
    feature_dim,
    context_embedding_dim=default_context_embedding_dim,
    n_layers=8,
    hidden_dim=128,
    num_blocks=2,
    dropout_probability=0.0,
    use_linear=True,
):
    """Returns a masked-affine autoregressive (MAF) transform implemented directly in PyTorch.

    The fastest expressive option for NLE: each layer's density (``log_prob``) is a single MADE pass
    plus an elementwise affine map -- no binning (no softmax / searchsorted) as in the RQ spline.
    Each layer is ``ActNorm`` -> a linear mixing layer (``LULinear`` if ``use_linear`` else
    ``RandomPermutation``, which also re-orders the autoregressive conditioning between layers) ->
    an affine autoregressive conditioner.

    Args:
        feature_dim (int): The dimension of the input features.
        context_embedding_dim (int): The dimension of the context embedding. Defaults to 16.
        n_layers (int, optional): The number of MAF layers. Defaults to 8.
        hidden_dim (int, optional): Hidden width of the MADE conditioner. Defaults to 128.
        num_blocks (int, optional): Residual blocks in the MADE conditioner. Defaults to 2.
        dropout_probability (float, optional): Conditioner dropout. Defaults to 0.0.
        use_linear (bool, optional): Use a learnable LU linear mix between layers instead of a fixed
            random permutation. Defaults to True.

    Returns:
        CompositeTransform: The composite MAF transform.
    """

    transforms = []
    for _ in range(n_layers):
        transforms.append(ActNorm(features=feature_dim))

        # mix the flow's dimensions (and re-order the autoregressive conditioning) between layers
        if use_linear:
            transforms.append(LULinear(feature_dim, identity_init=True))
        else:
            transforms.append(RandomPermutation(features=feature_dim))

        transforms.append(
            AffineAutoregressiveTransform(
                feature_dim=feature_dim,
                context_features=context_embedding_dim,
                hidden_features=hidden_dim,
                num_blocks=num_blocks,
                dropout_probability=dropout_probability,
            )
        )

    transform = CompositeTransform(transforms)

    return transform

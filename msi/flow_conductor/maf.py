# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2026
Author: Arne Thomsen

Pure-PyTorch masked-affine autoregressive flow (MAF; Papamakarios et al. 2017, arXiv:1705.07057),
implemented directly here as a faster alternative to the RQ neural spline for the neural likelihood
estimation (NLE) setting, where the dominant cost is evaluating the flow's ``log_prob`` inside the
MCMC posterior sampler.

The transform conforms to the ``enflows.transforms.Transform`` API -- ``forward``/``inverse`` each
return ``(outputs, logabsdet)`` with ``logabsdet`` of shape ``(batch,)`` -- so it slots into the same
``enflows.flows.Flow`` container as the spline transforms, with no change to ``LikelihoodFlow``'s
training / sampling / MCMC machinery. By the ``enflows.Flow`` convention ``forward`` maps data ->
noise (this is what ``log_prob`` uses) and ``inverse`` maps noise -> data (used only for sampling).

Versus the RQ spline, the affine transform has **no binning** -- no per-dimension ``softmax``,
``searchsorted``/``gather`` or per-bin transcendentals -- so its single-pass ``log_prob`` is the
cheapest expressive option here: density evaluation is one MADE forward plus an elementwise affine
map. Autoregressive sampling (``inverse``) costs ``feature_dim`` passes, which is irrelevant for NLE
since flow sampling is only used for likelihood-level diagnostics. The scale is constrained with
``softplus + epsilon`` (matching enflows' MaskedAffineAutoregressiveTransform) to keep it positive
and the training stable.
"""

import torch
from torch.nn import functional as F

from enflows.transforms.base import Transform
from enflows.transforms.made import MADE


class AffineAutoregressiveTransform(Transform):
    """Conditional masked-affine autoregressive (MAF) layer.

    A ``MADE`` conditioner emits a per-dimension affine ``(scale, shift)`` that depends only on the
    preceding dimensions and the embedded context. ``forward`` (density, the ``log_prob`` / MCMC hot
    path) is a single MADE pass; ``inverse`` (sampling) iterates ``feature_dim`` times to invert the
    triangular map.
    """

    def __init__(
        self,
        feature_dim,
        context_features,
        hidden_features=128,
        num_blocks=2,
        dropout_probability=0.0,
        activation=F.relu,
        epsilon=1e-3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.made = MADE(
            features=feature_dim,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=2,  # (unconstrained_scale, shift) per dimension
            activation=activation,
            dropout_probability=dropout_probability,
        )

    def _scale_shift(self, params):
        # MADE lays out the output_multiplier params contiguously per feature -> (batch, feature, 2)
        params = params.view(-1, self.feature_dim, 2)
        unconstrained_scale = params[..., 0]
        shift = params[..., 1]
        scale = F.softplus(unconstrained_scale) + self.epsilon  # positive, stable
        return scale, shift

    def forward(self, inputs, context=None):
        # single pass: density evaluation (the NLE / MCMC hot path)
        scale, shift = self._scale_shift(self.made(inputs, context))
        outputs = scale * inputs + shift
        logabsdet = torch.log(scale).sum(dim=-1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        # sequential inversion of the triangular autoregressive map (sampling only)
        outputs = torch.zeros_like(inputs)
        scale = None
        for _ in range(self.feature_dim):
            scale, shift = self._scale_shift(self.made(outputs, context))
            outputs = (inputs - shift) / scale
        logabsdet = -torch.log(scale).sum(dim=-1)
        return outputs, logabsdet

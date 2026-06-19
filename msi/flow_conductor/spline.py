# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2026
Author: Arne Thomsen

Pure-PyTorch rational-quadratic (RQ) neural spline transforms (Durkan et al. 2019,
arXiv:1906.04032), implemented directly here rather than via FlowConductor's monotonic
sum-of-sigmoids transforms. These are designed as a faster, smaller alternative for the
neural likelihood estimation (NLE) setting, where the dominant cost is evaluating the flow's
``log_prob`` inside the MCMC posterior sampler.

The transforms conform to the ``enflows.transforms.Transform`` API -- ``forward`` and
``inverse`` each return ``(outputs, logabsdet)`` with ``logabsdet`` of shape ``(batch,)`` --
so they slot directly into the existing ``enflows.flows.Flow`` container used by
``LikelihoodFlow`` with no changes to its training / sampling / MCMC machinery. By the
``enflows.Flow`` convention, ``forward`` maps data -> noise (this is what ``log_prob`` uses)
and ``inverse`` maps noise -> data (used only for sampling / diagnostics).

Two conditioners are provided:
- ``RQSplineCouplingTransform``: binary-mask coupling. Both ``forward`` and ``inverse`` are a
  single network pass, so it is fast to both evaluate and sample.
- ``RQSplineAutoregressiveTransform``: MAF-style. ``forward`` (density, the NLE hot path) is a
  single pass; ``inverse`` (sampling) costs ``feature_dim`` passes, which is irrelevant here
  since flow sampling is only used for likelihood-level diagnostics.
"""

import numpy as np
import torch
from torch.nn import functional as F

from enflows.transforms.base import Transform
from enflows.nn.nets import ResidualNet
from enflows.transforms.made import MADE

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

# The inverse (sampling) pass can optionally assert the RQ-spline discriminant is non-negative.
# That check calls ``.all()``, which forces a host-device sync each inverse pass and serializes
# the GPU during sampling-based diagnostics (and the autoregressive inverse runs it feature_dim
# times per layer). It is a correctness guard, not needed in the hot path, so it is off by default
# and gated here.
DEBUG_SPLINE_INVERSE = False


def _searchsorted(bin_locations, inputs):
    """Return the bin index i such that bin_locations[..., i] <= inputs < bin_locations[..., i+1].

    Uses ``torch.searchsorted`` (a single kernel, no large intermediate) rather than the
    ``sum(inputs[..., None] >= bin_locations)`` reduction, which materializes a
    ``(..., num_bins + 1)`` boolean tensor per spline layer -- a measurable cost inside the
    tight MCMC log_prob loop. The result is clamped to ``[0, num_bins - 1]`` so inputs landing
    exactly on the right edge (or, in the inverse pass, marginally outside) stay in range; the
    bin index is a discrete selector with no gradient in either formulation, so this is
    equivalent for both training and inference.
    """
    num_bins = bin_locations.shape[-1] - 1
    idx = torch.searchsorted(bin_locations.contiguous(), inputs.unsqueeze(-1), right=True).squeeze(-1) - 1
    return idx.clamp(0, num_bins - 1)


def _rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse,
    left,
    right,
    bottom,
    top,
    min_bin_width,
    min_bin_height,
    min_derivative,
):
    """Core monotonic RQ spline on the interval [left, right] -> [bottom, top].

    ``inputs`` has arbitrary leading shape ``(...,)``; the parameter tensors share that leading
    shape with an extra trailing bin axis, i.e. ``(..., num_bins)`` for widths/heights and
    ``(..., num_bins + 1)`` for the (already boundary-padded) derivatives. Returns
    ``(outputs, logabsdet)`` both of shape ``(...,)``.
    """
    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = _searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = _searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        if DEBUG_SPLINE_INVERSE:
            assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    """RQ spline with linear ("unconstrained") tails outside ``[-tail_bound, tail_bound]``.

    Inside the interval the monotonic RQ spline of ``_rational_quadratic_spline`` is applied;
    outside, the transform is the identity (zero log-determinant), which lets the flow act on
    unbounded inputs. ``unnormalized_derivatives`` carries the ``num_bins - 1`` interior knot
    derivatives and is boundary-padded here so the tails join the spline with unit slope.

    Shapes: ``inputs`` is ``(...,)``; ``unnormalized_widths``/``unnormalized_heights`` are
    ``(..., num_bins)``; ``unnormalized_derivatives`` is ``(..., num_bins - 1)``. Returns
    ``(outputs, logabsdet)`` both of shape ``(...,)``.
    """
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # pad the interior derivatives with the boundary value that yields unit-slope linear tails.
    # Pad both ends directly with that constant (functional, single op) rather than padding with 0
    # and then assigning in place -- the latter is two extra kernels and an in-place index_put that
    # can force a graph break under torch.compile.
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1), value=constant)

    # Evaluate the spline on the FULL tensor with static shapes and no data-dependent control flow,
    # then select the identity (linear tail) for out-of-interval elements via torch.where. This is
    # deliberately branch-free: an nflows-style implementation that boolean-masks the in-interval
    # elements (`inputs[mask]`, scatter-back) and guards with `if torch.any(mask)` incurs a
    # host-device sync per spline layer, which serializes the GPU and makes the spline's log_prob far
    # slower than a fully-vectorized transform (e.g. sum-of-sigmoids) inside a tight MCMC loop.
    # Out-of-interval inputs are clamped only so the core spline stays well-defined; their clamped
    # result is discarded by torch.where below, where the map is exactly the identity.
    inputs_clamped = torch.clamp(inputs, -tail_bound, tail_bound)
    spline_outputs, spline_logabsdet = _rational_quadratic_spline(
        inputs=inputs_clamped,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    outputs = torch.where(inside, spline_outputs, inputs)
    logabsdet = torch.where(inside, spline_logabsdet, torch.zeros_like(inputs))

    return outputs, logabsdet


class RQSplineCouplingTransform(Transform):
    """Conditional RQ neural spline coupling layer.

    Splits the ``feature_dim`` inputs into an identity half and a transform half via a fixed
    binary mask. The identity half, together with the embedded context, feeds a ``ResidualNet``
    conditioner that emits the spline parameters for the transform half. Both ``forward`` and
    ``inverse`` are a single conditioner pass.
    """

    def __init__(
        self,
        feature_dim,
        context_features,
        hidden_features=128,
        num_bins=8,
        tail_bound=5.0,
        num_blocks=2,
        dropout_probability=0.0,
        mask_even=True,
        activation=F.relu,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        # alternating binary mask: identity where mask==True, transformed where mask==False
        mask = torch.arange(feature_dim) % 2 == (0 if mask_even else 1)
        identity_idx = torch.where(mask)[0]
        transform_idx = torch.where(~mask)[0]
        if len(identity_idx) == 0 or len(transform_idx) == 0:
            raise ValueError(f"feature_dim={feature_dim} is too small to form a coupling split.")
        self.register_buffer("identity_idx", identity_idx)
        self.register_buffer("transform_idx", transform_idx)

        # number of spline parameters per transformed dimension: widths, heights, interior derivatives
        self.params_per_dim = 3 * num_bins - 1
        self.conditioner = ResidualNet(
            in_features=len(identity_idx),
            out_features=len(transform_idx) * self.params_per_dim,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
        )

    def _coupling(self, inputs, context, inverse):
        identity = inputs[:, self.identity_idx]
        transform = inputs[:, self.transform_idx]

        params = self.conditioner(identity, context)
        params = params.view(inputs.shape[0], len(self.transform_idx), self.params_per_dim)
        unnormalized_widths = params[..., : self.num_bins]
        unnormalized_heights = params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = params[..., 2 * self.num_bins :]

        transformed, logabsdet = unconstrained_rational_quadratic_spline(
            transform,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=inverse,
            tail_bound=self.tail_bound,
        )

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_idx] = identity
        outputs[:, self.transform_idx] = transformed

        return outputs, logabsdet.sum(dim=-1)

    def forward(self, inputs, context=None):
        return self._coupling(inputs, context, inverse=False)

    def inverse(self, inputs, context=None):
        return self._coupling(inputs, context, inverse=True)


class RQSplineAutoregressiveTransform(Transform):
    """Conditional masked-autoregressive (MAF-style) RQ neural spline layer.

    A ``MADE`` conditioner emits, for each dimension, the spline parameters that depend only on
    the preceding dimensions (and the embedded context). ``forward`` (density) is a single MADE
    pass; ``inverse`` (sampling) iterates ``feature_dim`` times to invert the triangular map.
    """

    def __init__(
        self,
        feature_dim,
        context_features,
        hidden_features=128,
        num_bins=8,
        tail_bound=5.0,
        num_blocks=2,
        dropout_probability=0.0,
        activation=F.relu,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.params_per_dim = 3 * num_bins - 1

        self.made = MADE(
            features=feature_dim,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self.params_per_dim,
            activation=activation,
            dropout_probability=dropout_probability,
        )

    def _spline(self, inputs, params, inverse):
        params = params.view(inputs.shape[0], self.feature_dim, self.params_per_dim)
        unnormalized_widths = params[..., : self.num_bins]
        unnormalized_heights = params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = params[..., 2 * self.num_bins :]
        return unconstrained_rational_quadratic_spline(
            inputs,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=inverse,
            tail_bound=self.tail_bound,
        )

    def forward(self, inputs, context=None):
        # single pass: density evaluation (the NLE / MCMC hot path)
        params = self.made(inputs, context)
        outputs, logabsdet = self._spline(inputs, params, inverse=False)
        return outputs, logabsdet.sum(dim=-1)

    def inverse(self, inputs, context=None):
        # sequential inversion of the triangular autoregressive map (sampling only)
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(self.feature_dim):
            params = self.made(outputs, context)
            outputs, logabsdet = self._spline(inputs, params, inverse=True)
        return outputs, logabsdet.sum(dim=-1)

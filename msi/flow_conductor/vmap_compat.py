# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""Runtime, behavior-preserving patches that make an enflows flow's forward/log_prob traceable under
``torch.func.vmap`` (used by the ensemble's fused vmap training and inference-time fusion).

The only enflows internal on the MAF forward path that vmap rejects is ``CompositeTransform._cascade``:
it seeds the log-det accumulator with ``inputs.new_zeros(batch_size)`` -- a fresh tensor that does NOT
carry the vmap batch dimension -- and then does an in-place ``total_logabsdet += logabsdet`` with a
batched value, which raises ``output with shape [B] doesn't match the broadcast shape [N, B]``. Rewriting
the accumulation out-of-place fixes it and is numerically identical for ordinary (non-vmap) calls, so the
patch is safe to leave applied process-wide.

(RQ-spline needs additional, structural changes -- branchless control flow + a vmap-safe searchsorted --
and is intentionally not handled here.)
"""

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

_PATCHED = False


def patch_enflows_for_vmap():
    """Idempotently patch enflows so a flow's forward/log_prob can be traced under vmap. Safe to call
    repeatedly; degrades to a no-op (and a warning) if the enflows layout changes, leaving the ensemble's
    existing per-member fallback to handle it."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        from enflows.transforms.base import CompositeTransform

        @staticmethod
        def _cascade(inputs, funcs, context):
            # identical to enflows' CompositeTransform._cascade except the log-det accumulation is
            # out-of-place (`= ... +` instead of `+=`), which vmap requires for the unbatched seed tensor.
            batch_size = inputs.shape[0]
            outputs = inputs
            total_logabsdet = inputs.new_zeros(batch_size)
            for func in funcs:
                outputs, logabsdet = func(outputs, context)
                total_logabsdet = total_logabsdet + logabsdet
            return outputs, total_logabsdet

        CompositeTransform._cascade = _cascade
        _PATCHED = True
        LOGGER.info("Patched enflows CompositeTransform._cascade for vmap compatibility")
    except Exception as e:
        LOGGER.warning(f"Could not patch enflows for vmap ({type(e).__name__}: {e}); fused/vmap paths will fall back")

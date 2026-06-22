"""
A PyTorch reimplementation of emcee's default affine-invariant ensemble sampler (the Goodman & Weare 2010
stretch move, ``emcee.moves.StretchMove`` with ``a = 2.0``), batched over an arbitrary number of
independent observations.

Why this exists: for posterior-level coverage testing we need to draw one MCMC posterior per held-out
observation (~1000 of them). emcee runs a sequential numpy loop and, even with ``vectorize=True``, only
batches over the ``n_walkers`` walkers of a single chain -- a tiny batch that leaves a GPU idle. Here the
leading ``n_obs`` axis lets a single normalizing-flow forward pass evaluate ``n_obs * n_walkers`` points at
once, which actually saturates a Grace-Hopper GPU. The move itself is identical to emcee's, so results
match emcee up to MCMC sampling noise (see run_mcmc_for_coverage_tests.py for the fidelity check).
"""

import numpy as np
import torch

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def _stretch_update(theta, log_prob, active, complement, log_prob_fn, a, generator):
    """Update one half of the ensemble in place against the other (complementary) half.

    Args:
        theta (torch.Tensor): Full ensemble, shape (n_obs, n_walkers, n_params).
        log_prob (torch.Tensor): Current log-posterior of the full ensemble, shape (n_obs, n_walkers).
        active (slice): Walker indices to update this sub-step.
        complement (slice): Walker indices used as the complementary set to propose from.
        log_prob_fn (callable): (n_obs, k, n_params) -> (n_obs, k) batched log-posterior.
        a (float): Stretch-move scale parameter (emcee default 2.0).
        generator (torch.Generator): RNG on theta's device.
    """
    n_obs, n_params = theta.shape[0], theta.shape[2]
    device, floatx = theta.device, theta.dtype

    s = theta[:, active, :]  # (n_obs, n_active, n_params)
    c = theta[:, complement, :]  # (n_obs, n_comp, n_params)
    lp_old = log_prob[:, active]  # (n_obs, n_active)

    n_active = s.shape[1]
    n_comp = c.shape[1]

    # pick a random complementary walker per active walker (uniform, with replacement), per observation
    idx = torch.randint(n_comp, (n_obs, n_active), device=device, generator=generator)  # (n_obs, n_active)
    c_sel = torch.gather(c, 1, idx.unsqueeze(-1).expand(-1, -1, n_params))  # (n_obs, n_active, n_params)

    # draw z ~ g(z) propto 1/sqrt(z) on [1/a, a], exactly as emcee.moves.StretchMove.get_proposal
    u = torch.rand((n_obs, n_active), device=device, dtype=floatx, generator=generator)
    zz = ((a - 1.0) * u + 1.0) ** 2 / a  # (n_obs, n_active)

    q = c_sel + zz.unsqueeze(-1) * (s - c_sel)  # proposal, (n_obs, n_active, n_params)

    lp_new = log_prob_fn(q)  # (n_obs, n_active)

    # Metropolis acceptance with the affine-invariant factor (n_params - 1) * ln(z)
    log_factor = (n_params - 1) * torch.log(zz)
    lnpdiff = log_factor + lp_new - lp_old
    log_u = torch.log(torch.rand((n_obs, n_active), device=device, dtype=floatx, generator=generator))
    accept = lnpdiff > log_u  # (n_obs, n_active); -inf (outside prior) never accepts

    theta[:, active, :] = torch.where(accept.unsqueeze(-1), q, s)
    log_prob[:, active] = torch.where(accept, lp_new, lp_old)


def run_ensemble_torch(
    log_prob_fn,
    theta_0,
    n_steps=1_000,
    n_burnin_steps=1_000,
    a=2.0,
    generator=None,
    progress=True,
):
    """Run the batched affine-invariant ensemble sampler.

    Args:
        log_prob_fn (callable): Vectorized batched log-posterior. Takes a tensor of shape
            (n_obs, k, n_params) and returns (n_obs, k). Must return -inf outside the prior.
        theta_0 (torch.Tensor): Initial walker positions, shape (n_obs, n_walkers, n_params), on the target
            device. ``n_walkers`` must be even (split into two half-ensembles, like emcee).
        n_steps (int, optional): Number of main-chain steps. Defaults to 1000.
        n_burnin_steps (int, optional): Number of burn-in steps (discarded). Defaults to 1000.
        a (float, optional): Stretch-move scale parameter. Defaults to 2.0 (emcee default).
        generator (torch.Generator, optional): RNG on theta_0's device. Defaults to None (a fresh one).
        progress (bool, optional): Log a progress bar for the main chain. Defaults to True.

    Returns:
        tuple(np.ndarray, np.ndarray): Flattened chain of shape (n_obs, n_steps * n_walkers, n_params) and
            log-posterior of shape (n_obs, n_steps * n_walkers), both as numpy arrays on the host. The
            ordering matches emcee's get_chain(flat=True): walkers are the fastest-varying axis within a
            step.
    """
    device = theta_0.device
    n_obs, n_walkers, n_params = theta_0.shape
    assert n_walkers % 2 == 0, f"n_walkers must be even, got {n_walkers}"

    if generator is None:
        generator = torch.Generator(device=device)

    half = n_walkers // 2
    first, second = slice(0, half), slice(half, n_walkers)

    theta = theta_0.clone()
    log_prob = log_prob_fn(theta)  # (n_obs, n_walkers)

    def _run(n, desc, store):
        # preallocate the host-side chain only for the kept (main-chain) phase
        if store:
            chain = np.empty((n, n_obs, n_walkers, n_params), dtype=np.float32)
            chain_lp = np.empty((n, n_obs, n_walkers), dtype=np.float32)
        steps = LOGGER.progressbar(range(n), desc=desc, at_level="debug") if progress else range(n)
        for step in steps:
            # update each half against the (current) other half, exactly as emcee's RedBlueMove
            _stretch_update(theta, log_prob, first, second, log_prob_fn, a, generator)
            _stretch_update(theta, log_prob, second, first, log_prob_fn, a, generator)
            if store:
                chain[step] = theta.cpu().numpy()
                chain_lp[step] = log_prob.cpu().numpy()
        if store:
            return chain, chain_lp
        return None, None

    LOGGER.info(f"Starting the burn in MCMC chain ({n_burnin_steps} steps) for {n_obs} observations")
    LOGGER.timer.start("torch_mcmc_burnin")
    _run(n_burnin_steps, "burn-in", store=False)
    LOGGER.info(f"[timing] burn-in ({n_burnin_steps} steps): {LOGGER.timer.elapsed('torch_mcmc_burnin')}")

    LOGGER.info(f"Starting the main MCMC chain ({n_steps} steps) for {n_obs} observations")
    LOGGER.timer.start("torch_mcmc_main")
    chain, chain_lp = _run(n_steps, "main chain", store=True)
    LOGGER.info(
        f"[timing] main chain ({n_steps} steps x {n_walkers} walkers x {n_obs} obs = "
        f"{n_steps * n_walkers * n_obs} log_prob evals): {LOGGER.timer.elapsed('torch_mcmc_main')}"
    )

    # (n_steps, n_obs, n_walkers, n_params) -> (n_obs, n_steps * n_walkers, n_params), walkers fastest
    chain = np.transpose(chain, (1, 0, 2, 3)).reshape(n_obs, n_steps * n_walkers, n_params)
    chain_lp = np.transpose(chain_lp, (1, 0, 2)).reshape(n_obs, n_steps * n_walkers)

    return chain, chain_lp

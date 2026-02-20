"""
Created June 2023
Author: Arne Thomsen

Utils to run the MCMC algorithm to get the chain to be plotted as the parameter constraints.
"""

import os
import numpy as np

import emcee
from emcee import EnsembleSampler

from msfm.utils import prior, parameters, logger

LOGGER = logger.get_logger(__file__)
np.random.seed(12)


def run_emcee(
    log_prob_fn, params, conf=None, out_dir=None, label=None, n_walkers=1024, n_steps=1000, n_burnin_steps=100, moves=None
):
    """Run the emcee EnsembleSampler to get a Markov Chain of samples from the distribution.

    TODO add support for Nautilus https://nautilus-sampler.readthedocs.io/en/stable/ in addition to emcee?

    Args:
        log_prob_fn (function): Vectorized function that takes in samples of shape (n_samples,) in parameter space and
            returns an array of corresponding log probabilities of shape (n_samples,)
        params (list): List of strings of the constrained cosmological parameters.
        out_dir (str, optional): Output directory to store the plot at. Defaults to None, then the plot is not saved.
        label (str, optional): Additional label for the file name, for example to designate different observations.
            Defaults to None.
        n_walkers (int, optional): Number of walkes to use in emcee, this determines the level of parallelization via
            vectorization. Defaults to 1024.
        n_steps (int, optional): Number of steps to run the chain for. Defaults to 1000.
        n_burnin_steps (int, optional): Number of steps to run the burn in chain for. Defaults to 100.
        moves (list, optional): List of emcee moves to use. Defaults to None, which uses the default moves. An example
            is [(emcee.moves.StretchMove(a=1.6), 0.75), (emcee.moves.WalkMove(), 0.25)].

    Returns:
        chain (np.ndarray): An array of shape (n_samples, n_params)
    """
    n_params = len(params)

    # initial points (spread around the fiducial parameters)
    theta_0 = np.random.normal(loc=parameters.get_fiducials(params, conf=conf), scale=1e-3, size=(n_walkers, n_params))

    LOGGER.info(f"Initial values in prior: {np.mean(prior.in_grid_prior(theta_0, conf=conf, params=params)*100):.1f}%")

    # sample burn in
    sampler = EnsembleSampler(n_walkers, n_params, log_prob_fn, vectorize=True, moves=moves)

    LOGGER.info(f"Starting the burn in MCMC chain ({n_burnin_steps} steps)")
    state = sampler.run_mcmc(theta_0, n_burnin_steps, progress=True)
    sampler.reset()

    # run the actual chain
    LOGGER.info(f"Starting the main MCMC chain ({n_steps} steps)")
    sampler.run_mcmc(state, n_steps, progress=True)

    chain = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)

    # there can be more samples than requested due the walkers
    n_samples = n_steps * n_walkers
    chain = chain[:n_samples]

    # get MAP
    MAP_params = chain[np.argmax(log_probs)]
    LOGGER.info(f"MAP parameters: " + str({p: np.round(v, 3) for p, v in zip(params, MAP_params)}))

    # save the result
    if out_dir is not None:
        chain_file = os.path.join(out_dir, f"chain_{label}.npy" if label else "chain.npy")
        log_probs_file = os.path.join(out_dir, f"log_probs_{label}.npy" if label else "log_probs.npy")

        np.save(chain_file, chain)
        np.save(log_probs_file, log_probs)

        LOGGER.info(f"Saved the MCMC chain to {chain_file}")
    else:
        LOGGER.warning(f"Not saving the MCMC chain")

    return chain

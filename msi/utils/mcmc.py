"""
Created June 2023
Author: Arne Thomsen

Utils to run the MCMC algorithm to get the chain to be plotted as the parameter constraints.
"""

import os
import numpy as np

from emcee import EnsembleSampler

from msfm.utils import prior, parameters, logger

LOGGER = logger.get_logger(__file__)
np.random.seed(12)


def run_emcee(log_prob, params, out_dir=None, label="temp", n_walkers=1024, n_steps=1000):
    """Run the emcee EnsembleSampler to get a Markov Chain of samples from the distribution.

    Args:
        log_prob (function): Vectorized function that takes in samples of shape (n_samples,) in parameter space and
            returns an array of corresponding log probabilities of shape (n_samples,)
        params (list): List of strings of the constrained cosmological parameters.
        out_dir (str, optional): Output directory to store the plot at. Defaults to None, then the plot is not saved.
        label (str, optional): Marks which inference method has been used. Defaults to "temp".
        n_walkers (int, optional): Number of walkes to use in emcee, this determines the level of parallelization via
            vectorization. Defaults to 1024.

    Returns:
        chain (np.ndarray): An array of shape (n_samples, n_params)
    """
    n_params = len(params)

    # initial points (spread around the fiducial parameters)
    theta_0 = np.random.normal(loc=parameters.get_fiducials(params), scale=1e-3, size=(n_walkers, n_params))
    LOGGER.info(f"Initial values in prior: {np.all(prior.in_grid_prior(theta_0, params=params))}")

    # sample burn in
    sampler = EnsembleSampler(n_walkers, n_params, log_prob, vectorize=True)
    state = sampler.run_mcmc(theta_0, 100)
    sampler.reset()

    # run the actual chain
    sampler.run_mcmc(state, n_steps, progress=True)

    # save the result
    chain = sampler.get_chain(flat=True)
    if out_dir is not None:
        np.save(os.path.join(out_dir, f"chain_{label}.npy"), chain)
    else:
        LOGGER.warning(f"Not saving the chain")

    return chain

# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen, Janis Fluri

Perform the ABC integral in equation (14) in https://arxiv.org/pdf/2107.09002.pdf to estimate the posterior.

Adapted from
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/probability.py
by Janis Fluri
"""

import numpy as np
import mpmath as mp
from multiprocessing import Pool, cpu_count

from msfm.utils import logger
from msi import kernels

LOGGER = logger.get_logger(__file__)


def _estimate_single_posterior(preds, fid_fisher, obs_pred, kernel, scale):
    """Estimates the ABC posterior and its uncertainty from parameter predictions.

    Args:
        preds (np.ndarray): 2D array of predictions with shape (n_examples, n_output).
        fid_fisher (np.ndarray): The (estimated) fisher matrix of the fiducial parameter with shape
            (n_output, n_output). If None, the inverse of the covariance of the predictions for the corresponding
            cosmology are used. Usually, this is estimated for the fiducial instead.
        obs_pred (np.ndarray): The prediction of the observation with shape (1, n_output).
        kernel (str, optional): The kernel to use for the ABC estimation, either sigmoid, gauss or logistic. Defaults
            to "sigmoid".
        scale (float, optional): The scale used for the kernel, aka h.

    Raises:
        ValueError: If an unknown kernel string designation is passed.

    Returns:
        np.float64, np.float64: Estimates of the log posterior and its variance like the ABC integral in equations (14)
        and (17) in https://arxiv.org/pdf/2107.09002.pdf.
    """

    # check kernel consistency
    if kernel == "sigmoid":

        def kernel(d):
            return kernels.sigmoid_kernel(d, scale=scale, use_mp=True, return_mp=True)

    elif kernel == "gauss":

        def kernel(d):
            return kernels.gaussian_kernel(d, scale=scale, use_mp=True, return_mp=True)

    elif kernel == "logistic":

        def kernel(d):
            return kernels.logistic_kernel(d, scale=scale, use_mp=True, return_mp=True)

    else:
        raise ValueError(f"Kernel type not understood {kernel}! kernel has to be either sigmoid, gauss or logistic.")

    with mp.workdps(100):
        # get the fisher matrix
        if fid_fisher is None:
            fid_fisher = np.linalg.inv(np.cov(preds, rowvar=False))

        # shape (n_examples, n_output), left side of the [vector^T * matrix * vector] multiplication below (14),
        # the einsum does this for all examples in one operation
        fisher_dist = np.einsum("ij,aj->ai", fid_fisher, preds - obs_pred)
        # shape (n_examples,), right side, sum runs over n_output
        fisher_dist = np.sum(fisher_dist * (preds - obs_pred), axis=1)
        fisher_dist = np.sqrt(fisher_dist)

        # evaluate the kernel, shape (n_examples)
        kernel_vals = kernel(fisher_dist)

        # posterior, like (14)
        mean = kernels.mp_mean(kernel_vals)
        # like (15)
        log_post = np.float64(mp.log(mean))

        # variance of the posterior, like between (16) and (17), with n_x = len(fisher_dist)
        log_var = kernels.mp_std(kernel_vals) ** 2 / len(fisher_dist)
        # like (17), add an eps here for stability

        log_var = np.float64(log_var / (mean**2 + 1e-99))

    return log_post, log_var


def estimate_grid_posterior(grid_preds, fid_fisher, obs_pred, kernel="sigmoid", scale=0.4, n_cpus=None):
    """Estimates the ABC posterior and its uncertainty from parameter predictions.

    Args:
        grid_preds (np.ndarray): 3D array of predictions (or the learned summary statistic) with shape
            (n_cosmos, n_examples_per_cosmo, n_output), where n_output is the size of the summary statistic.
        fid_fisher (np.ndarray): The (estimated) fisher matrix of the fiducial parameter with shape
            (n_output, n_output). If None, the inverse of the covariance of the predictions for the corresponding
            cosmology are used. Usually, this is estimated for the fiducial instead.
        obs_pred (np.ndarray): The prediction of the observation with shape (1, n_output).
        kernel (str, optional): The kernel to use for the ABC estimation, either sigmoid, gauss or logistic. Defaults
            to "sigmoid".
        scale (float, optional): The scale used for the kernel, aka h. Defaults to 0.4.
        n_cpus (int, optional): Number of parallel processes to use. Defaults to None, then all cores are used.

    Returns:
        np.ndarray, np.ndarray: Y_init is a 2D array of shape (n_cosmos, 2) containing the ABC log posterior estimates
        and their variance. These values are emulated by the Gaussian Process. normalized_post is an array of shape
        (n_cosmos,) and contains (non-logged) probabilities that sum to one.
    """
    if fid_fisher is None:
        LOGGER.warning(f"No Fisher matrix was passed, using the inverse of the covariance at the grid points.")

    # number of CPU cores
    if n_cpus is None:
        n_cpus = cpu_count()

    # make the obs 2D
    obs_pred = np.atleast_2d(obs_pred)

    # loop over the different cosmologies
    inputs = [(preds, fid_fisher, obs_pred, kernel, scale) for preds in grid_preds]

    # get the ABC estimates
    log_post_estimates = []
    log_var_estimates = []
    with mp.workdps(100):
        with Pool(processes=n_cpus) as pool:
            for log_post, log_var in pool.starmap(_estimate_single_posterior, inputs):
                log_post_estimates.append(log_post)
                log_var_estimates.append(log_var)

    # shape (n_cosmos,)
    log_post_estimates = np.asarray(log_post_estimates)
    log_var_estimates = np.asarray(log_var_estimates)

    # shape (n_cosmos, 2), used to initialize the Gaussian Process emulator
    Y_init = np.stack([log_post_estimates, log_var_estimates], axis=-1)

    # shape (n_cosmos,), normalize the posterior
    posterior = log_post_estimates
    posterior -= np.max(posterior)
    posterior = np.exp(posterior)
    posterior /= np.sum(posterior)

    return Y_init, posterior

# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen, Janis Fluri

The kernels implemented in this file use the mpmath library to employ arbitrarily precise floats.

Adapted from
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/kernels.py
by Janis Fluri
"""

import numpy as np
import mpmath as mp
import os

from msfm.utils import logger, input_output

LOGGER = logger.get_logger(__file__)

# load the config
file_dir = os.path.dirname(__file__)
repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
conf = os.path.join(repo_dir, "configs/dlss_config.yaml")
conf = input_output.read_yaml(conf)

kernel_min_val = conf["abc"]["kernel_min_val"]
LOGGER.info(f"Setting the minimum value of the kernel function to {kernel_min_val} for the mpmath library")


@mp.workdps(100)
def gaussian_kernel(d, scale=0.05, use_mp=False, return_mp=False):
    """Gaussian kernel with scale parameter (mean is zero).

    Args:
        d (np.ndarray): The input of the kernel (a distance, usually Fisher), might be numpy array but not mp matrix.
        scale (float, optional): The scale parameter, aka h. Defaults to 0.05.
        use_mp (bool, optional): Use arbitrary floating point precision using mpmath. Defaults to False.
        return_mp (bool, optional): If True, return mpf object. Defaults to False, then a numpy object is returned.

    Returns:
        np.ndarray, mp: The Gaussian kernel evaluated at d for the scale.
    """
    if not use_mp:
        if return_mp:
            LOGGER.warning(f"To return an mpf object, set use_mp=True. Returning numpy object")
        norm = np.sqrt(2 * np.pi) * scale
        chi = -0.5 * (d / scale) ** 2
        return np.maximum(np.exp(chi) / (norm), kernel_min_val)

    else:
        if isinstance(d, np.ndarray):
            is_array = True
            shape = d.shape
            # the array has to be 1D for the mpmath loop below
            d = d.ravel().astype(np.float64)
        else:
            is_array = False
            d = np.array([d]).astype(np.float64)

        res = []
        for dd in d:
            norm = mp.sqrt(2 * mp.pi) * scale
            chi = -0.5 * (mp.mpf(dd) / scale) ** 2
            res.append(mp.exp(chi) / norm)

        if is_array:
            if return_mp:
                return np.array(res).reshape(shape)
            else:
                return np.array(res, dtype=np.float64).reshape(shape)
        else:
            if return_mp:
                return res[0]
            else:
                return np.float64(res[0])


@mp.workdps(100)
def logistic_kernel(d, scale=0.05, use_mp=False, return_mp=False):
    """Logistic kernel with scale parameter.

    Args:
        d (np.ndarray): The input of the kernel (a distance, usually Fisher), might be numpy array but not mp matrix.
        scale (float, optional): The scale parameter, aka h. Defaults to 0.05.
        use_mp (bool, optional): Use arbitrary floating point precision using mpmath. Defaults to False.
        return_mp (bool, optional): If True, return mpf object. Defaults to False, then a numpy object is returned.

    Returns:
        np.ndarray, mp: The Gaussian kernel evaluated at d for the scale.
    """
    if not use_mp:
        if return_mp:
            LOGGER.warning(f"To return an mpf object, set use_mp=True. Returning numpy object")
        pos_exp = np.exp(d / scale)
        neg_exp = np.exp(-d / scale)
        return np.maximum(1.0 / (scale * (pos_exp + neg_exp + 2)), kernel_min_val)

    else:
        if isinstance(d, np.ndarray):
            is_array = True
            shape = d.shape
            # the array has to be 1D for the mpmath loop below
            d = d.ravel().astype(np.float64)
        else:
            is_array = False
            d = np.array([d]).astype(np.float64)

        res = []
        for dd in d:
            pos_exp = mp.exp(mp.mpf(dd) / scale)
            neg_exp = mp.exp(-mp.mpf(dd) / scale)
            res.append(1.0 / (scale * (pos_exp + neg_exp + 2.0)))

        if is_array:
            if return_mp:
                return np.array(res).reshape(shape)
            else:
                return np.array(res, dtype=np.float64).reshape(shape)
        else:
            if return_mp:
                return res[0]
            else:
                return np.float64(res[0])


@mp.workdps(100)
def sigmoid_kernel(d, scale=0.05, use_mp=False, return_mp=False):
    """Sigmoid kernel with scale parameter. Like in Section F of https://arxiv.org/pdf/2107.09002.pdf.

    Args:
        d (np.ndarray): The input of the kernel (a distance, usually Fisher), might be numpy array but not mp matrix.
        scale (float, optional): The scale parameter, aka h. Defaults to 0.05.
        use_mp (bool, optional): Use arbitrary floating point precision using mpmath. Defaults to False.
        return_mp (bool, optional): If True, return mpf object. Defaults to False, then a numpy object is returned.

    Returns:
        np.ndarray, mp: The Gaussian kernel evaluated at d for the scale.
    """

    if not use_mp:
        if return_mp:
            print("To return an mpf object, set use_mp=True, returning numpy object...")
        pos_exp = np.exp(d / scale)
        neg_exp = np.exp(-d / scale)
        return np.maximum(2.0 / (np.pi * scale * (pos_exp + neg_exp)), kernel_min_val)

    else:
        if isinstance(d, np.ndarray):
            is_array = True
            shape = d.shape
            # the array has to be 1D for the mpmath loop below
            d = d.ravel().astype(np.float64)
        else:
            is_array = False
            d = np.array([d]).astype(np.float64)

        res = []
        for dd in d:
            pos_exp = mp.exp(mp.mpf(dd) / scale)
            neg_exp = mp.exp(-mp.mpf(dd) / scale)
            res.append(2.0 / (mp.pi * scale * (pos_exp + neg_exp)))

        if is_array:
            if return_mp:
                return np.array(res).reshape(shape)
            else:
                return np.array(res, dtype=np.float64).reshape(shape)
        else:
            if return_mp:
                return res[0]
            else:
                return np.float64(res[0])


# mpmath helper functions #############################################################################################


@mp.workdps(100)
def mp_mean(arr):
    """Calculates the mean of the array of mpf values..

    Args:
        arr (mp.mpf): array of mp.mpf floats.

    Returns:
        mp.mpf: The sample mean.
    """
    arr = arr.ravel()
    N = arr.size

    res = mp.mpf(0.0)
    for a in arr:
        res = res + a

    return res / N


@mp.workdps(100)
def mp_std(arr, ddof=0):
    """Calculates the standard deviation of the array of mpf values.

    Args:
        arr (_type_): array of mp.mpf floats.
        ddof (int, optional): Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            here N represents the number of elements. Defaults to 0 (np.std convention).

    Returns:
        mp.mpf: The sample standard deviation.
    """
    arr = arr.ravel()
    N = arr.size

    # get the mean
    mean = mp_mean(arr)

    res = mp.mpf(0.0)
    for a in arr:
        res = res + (a - mean) ** 2

    return mp.sqrt(res / (N - ddof))

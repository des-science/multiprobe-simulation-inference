"""
Created June 2023
Author: Arne Thomsen

Utils to load MCMC chains to be plotted.
"""

import os, h5py

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def load_preds(base_dir, model_dir, n_steps=None, file_label=None):
    out_dir = os.path.join(base_dir, model_dir)

    if n_steps is None:
        preds_file = os.path.join(out_dir, f"preds.h5")
    elif file_label is None:
        preds_file = os.path.join(out_dir, f"preds_{n_steps}.h5")
    else:
        preds_file = os.path.join(out_dir, f"preds_{n_steps}_{file_label}.h5")

    with h5py.File(preds_file, "r") as f:
        # fiducial
        fidu_train_preds = f["fiducial/train/pred"][:]
        fidu_vali_preds = f["fiducial/vali/pred"][:]

        LOGGER.info(f"Array shapes:\n")
        LOGGER.info(f"fidu_train_preds =   {fidu_train_preds.shape}")
        LOGGER.info(f"fidu_vali_preds =    {fidu_vali_preds.shape}")

        # grid
        grid_preds = f["grid/pred"][:]
        grid_cosmos = f["grid/cosmo"][:]
        grid_sobol = f["grid/i_sobol"][:]

        LOGGER.info(f"grid_preds =         {grid_preds.shape}")
        LOGGER.info(f"grid_cosmos =        {grid_cosmos.shape}")
        LOGGER.info(f"grid_sobol =         {grid_cosmos.shape}")

    return fidu_train_preds, fidu_vali_preds, grid_preds, grid_cosmos, grid_sobol

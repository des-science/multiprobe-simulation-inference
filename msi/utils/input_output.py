"""
Created June 2023
Author: Arne Thomsen

Utils to load MCMC chains to be plotted.
"""

import os, h5py, yaml, glob
import numpy as np

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def get_abs_dir_repo():
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))

    return repo_dir


def load_msi_config():
    repo_dir = get_abs_dir_repo()
    conf_file = os.path.join(repo_dir, "configs/config.yaml")

    with open(conf_file, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    return conf


def load_preds(base_dir, model_dir, n_steps=None, file_label=None, preds_file=None):
    out_dir = os.path.join(base_dir, model_dir)

    if preds_file is None:
        if n_steps is None:
            preds_file = os.path.join(out_dir, f"preds.h5")
        elif file_label is None:
            preds_file = os.path.join(out_dir, f"preds_{n_steps}.h5")
        else:
            preds_file = os.path.join(out_dir, f"preds_{n_steps}_{file_label}.h5")
    else:
        preds_file = os.path.join(out_dir, preds_file)

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

def load_cls(fidu_dir, grid_dir):
    fidu_index = []
    fidu_cls = []
    grid_theta = []
    grid_cls = []

    n_bins = 4
    for i in range(n_bins):
        for j in range(n_bins):
            if i <= j:
                bin_num = f"{i+1}x{j+1}"
                LOGGER.info(f"Loading bin_num = {bin_num}")

                # load cls from .h5
                fidu_file_list = glob.glob(fidu_dir + f"/bin{bin_num}/WL_index_*_DESy3_fiducial_???.tfrecord.h5")
                LOGGER.info(f"found {len(fidu_file_list)} .h5 files")

                current_index = []
                current_cls = []

                for fidu_file in fidu_file_list:
                    with h5py.File(fidu_file, "r") as f:
                        current_index.append(f["power_spectrum"].attrs["index"][0,0])
                        current_cls.append(f["power_spectrum"][:])
                        
                current_index = np.asarray(current_index)
                current_cls = np.asarray(current_cls)
                
                # every example index must be unique
                assert len(np.unique(current_index)) == len(fidu_file_list)

                # sort
                i_sorted = np.argsort(current_index)
                current_index = current_index[i_sorted]
                current_cls = current_cls[i_sorted]
                
                # apply scale cut
                current_cls = current_cls[:,l_cut]
                
                # collect in lists
                fidu_index.append(current_index)
                fidu_cls.append(current_cls)

    # collect the different cosmologies
    fidu_index = np.stack(fidu_index, axis=-1)
    fidu_cls = np.concatenate(fidu_cls, axis=-1)

    print(f"\nfidu_index.shape = {fidu_index.shape}")
    print(f"fidu_cls.shape = {fidu_cls.shape}")

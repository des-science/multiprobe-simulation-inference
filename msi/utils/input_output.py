"""
Created June 2023
Author: Arne Thomsen

Utils to load MCMC chains to be plotted.
"""

import os, h5py, yaml, glob
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from msi.utils import plotting
from msfm.utils import logger, cross_statistics, parameters, files

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


""" neural network summary statistics """


def load_network_preds(base_dir, model_dir, n_steps=None, file_label=None, preds_file=None, return_training=False):
    out_dir = os.path.join(base_dir, model_dir)

    # build file name
    if preds_file is None:
        if n_steps is None:
            preds_file = os.path.join(out_dir, f"preds.h5")
        elif file_label is None:
            preds_file = os.path.join(out_dir, f"preds_{n_steps}.h5")
        else:
            preds_file = os.path.join(out_dir, f"preds_{n_steps}_{file_label}.h5")
    else:
        preds_file = os.path.join(out_dir, preds_file)

    h5_keys = [
        "fiducial/vali/pred",
        "fiducial/vali/i_example",
        "fiducial/vali/i_noise",
        "grid/pred",
        "grid/cosmo",
        "grid/i_example",
        "grid/i_noise",
        "grid/i_sobol",
    ]

    if return_training:
        h5_keys.append("fiducial/train/pred")

    with h5py.File(preds_file, "r") as f:
        LOGGER.info(f"Array shapes:")

        out_dict = {}
        for h5_key in h5_keys:
            out_dict[h5_key] = f[h5_key][:]
            LOGGER.info(f"{h5_key:<18} = {out_dict[h5_key].shape}")

    return out_dict


def get_reshaped_network_preds(base_dir, model_dir, n_steps=None, file_label=None, preds_file=None, n_params=None):
    file_dict = load_network_preds(base_dir, model_dir, n_steps=n_steps, file_label=file_label, preds_file=preds_file)

    fidu_preds = file_dict["fiducial/vali/pred"]
    grid_preds = file_dict["grid/pred"]
    grid_cosmos = file_dict["grid/cosmo"]

    # only relevant for the likelihood loss
    fidu_preds = fidu_preds[..., :n_params]
    grid_preds = grid_preds[..., :n_params]

    # combine the example and cosmology axes
    grid_preds = np.concatenate(grid_preds, axis=0)
    grid_cosmos = np.repeat(grid_cosmos, grid_preds.shape[0] // grid_cosmos.shape[0], axis=0)

    print("\n")
    LOGGER.info(f"Shapes after concatenation and selection:")
    LOGGER.info(f"fidu_preds  = {fidu_preds.shape}")
    LOGGER.info(f"grid_preds  = {grid_preds.shape}")
    LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")

    return fidu_preds, grid_preds, grid_cosmos


""" human defined summary statistics """


def load_human_summaries(base_dir, summary_type, file_label=None):
    assert summary_type in ["peaks", "cls"]

    if file_label is None:
        fidu_file = os.path.join(base_dir, f"fiducial_{summary_type}.h5")
        grid_file = os.path.join(base_dir, f"grid_{summary_type}.h5")
    else:
        fidu_file = os.path.join(base_dir, f"fiducial_{summary_type}_{file_label}.h5")
        grid_file = os.path.join(base_dir, f"grid_{summary_type}_{file_label}.h5")

    out_dict = {}

    fidu_keys = [summary_type, "i_example", "i_noise"]
    with h5py.File(fidu_file, "r") as f:
        LOGGER.info(f"Array shapes:")

        for h5_key in fidu_keys:
            dict_key = f"fiducial/{h5_key}"
            out_dict[dict_key] = f[h5_key][:]
            LOGGER.info(f"{dict_key:<18} = {out_dict[dict_key].shape}")

    grid_keys = [summary_type, "cosmo", "i_example", "i_noise", "i_sobol"]
    with h5py.File(grid_file, "r") as f:
        for h5_key in grid_keys:
            dict_key = f"grid/{h5_key}"
            out_dict[dict_key] = f[h5_key][:]
            LOGGER.info(f"{dict_key:<18} = {out_dict[dict_key].shape}")

    return out_dict


def get_reshaped_human_summaries(
    base_dir,
    summary_type,
    # file
    file_label=None,
    # configuration
    conf=None,
    params=None,
    # selection
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=True,
    # TODO scale cuts
    # i_min_scales=[],
    # i_max_scales=[],
    # plotting
    do_plot=True,
    # additional preprocessing
    apply_log=False,
    pca_components=None,
):
    file_dict = load_human_summaries(base_dir, summary_type, file_label=file_label)

    fidu_summs = file_dict[f"fiducial/{summary_type}"]
    grid_summs = file_dict[f"grid/{summary_type}"]
    grid_cosmos = file_dict["grid/cosmo"]

    # TODO
    # # apply the scale cuts
    # if summary_type == "cls":
    #     LOGGER.info(f"Applying scale cuts to the binned Cls")
    #     n_z = len(i_max_scales)
    #     assert n_z == len(i_min_scales), f"l_maxs and l_mins have different lengths"
    #     assert n_z * (n_z + 1) / 2 == fidu_summs.shape[-1], f"l_maxs and l_mins have wrong lengths"

    #     for i in range(n_z):
    #         for j in range(n_z):
    #             if i <= j:
    #                 i_min_scale = max(i_min_scales[i], i_min_scales[j])
    #                 i_max_scale = min(i_max_scales[i], i_max_scales[j])

    #                 fidu_summs[...,]

    bin_indices, bin_names = cross_statistics.get_cross_bin_indices(
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
    )

    # select the right auto and cross bins
    LOGGER.info(f"Using the bin indices {bin_indices}")
    LOGGER.info(f"With names {bin_names}")
    fidu_summs = fidu_summs[..., bin_indices]
    grid_summs = grid_summs[..., bin_indices]

    # select the right cosmological parameters
    conf = files.load_config(conf)
    all_params = parameters.get_parameters(None, conf)
    params = parameters.get_parameters(params, conf)

    param_indices = []
    for i, param in enumerate(all_params):
        if param in params:
            param_indices.append(i)
    grid_cosmos = grid_cosmos[..., param_indices]

    print("\n")
    LOGGER.info(f"Shapes after selection")
    LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
    LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")
    LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")

    # concatenate the bins along the last axis
    fidu_summs = np.concatenate([fidu_summs[..., i] for i in range(fidu_summs.shape[-1])], axis=-1)
    grid_summs = np.concatenate([grid_summs[..., i] for i in range(grid_summs.shape[-1])], axis=-1)

    if summary_type == "peaks":
        # concatenate the scales along the last axis
        fidu_summs = np.concatenate([fidu_summs[..., i, :] for i in range(fidu_summs.shape[-2])], axis=-1)
        grid_summs = np.concatenate([grid_summs[..., i, :] for i in range(grid_summs.shape[-2])], axis=-1)

    # concatenate the examples along the first axis
    grid_cosmos = np.repeat(grid_cosmos, repeats=grid_summs.shape[1], axis=0)
    grid_summs = np.concatenate([grid_summs[i, ...] for i in range(grid_summs.shape[0])], axis=0)

    print("\n")
    LOGGER.info("Shapes after concatenation")
    LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
    LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")
    LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")

    if do_plot:
        LOGGER.info(f"Plotting the selected raw {summary_type}")
        label = f"lensing={with_lensing},clustering={with_clustering},cross_z={with_cross_z},cross_probe={with_cross_probe}"

        if summary_type == "cls":
            plotting.plot_human_summary(
                fidu_summs,
                grid_summs,
                base_dir,
                label=label,
                bin_size=conf["analysis"]["power_spectra"]["n_bins"] - 1,
                bin_names=bin_names,
            )
        elif summary_type == "peaks":
            plotting.plot_human_summary(
                fidu_summs,
                grid_summs,
                base_dir,
                label=label,
                bin_size=conf["analysis"]["peak_statistics"]["n_bins"],
                bin_names=bin_names,
            )

    if apply_log:
        LOGGER.info(f"Taking the logarithm of the {summary_type}. Note that this only works for positive summaries")
        fidu_summs = np.log(fidu_summs)
        grid_summs = np.log(grid_summs)

    if pca_components is not None:
        print("\n")
        LOGGER.info(f"Scaling the {summary_type} to zero mean and unit variance")
        scaler = StandardScaler()
        grid_summs = scaler.fit_transform(grid_summs)
        fidu_summs = scaler.transform(fidu_summs)

        LOGGER.info(f"Applying PCA to compress to {pca_components} components")
        # PCA, is whitening a good idea?
        pca = PCA(n_components=pca_components, whiten=True)

        grid_summs = pca.fit_transform(grid_summs)
        fidu_summs = pca.transform(fidu_summs)
        LOGGER.info(f"Total explained variance = {np.sum(pca.explained_variance_ratio_)}")

        print("\n")
        LOGGER.info("Shapes after pre-processing")
        LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
        LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")
        LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")

    return fidu_summs, grid_summs, grid_cosmos


def load_virginia_cls(fidu_dir, grid_dir):
    LOGGER.warning(f"This function is deprecated!")

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
                        current_index.append(f["power_spectrum"].attrs["index"][0, 0])
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
                current_cls = current_cls[:, l_cut]

                # collect in lists
                fidu_index.append(current_index)
                fidu_cls.append(current_cls)

    # collect the different cosmologies
    fidu_index = np.stack(fidu_index, axis=-1)
    fidu_cls = np.concatenate(fidu_cls, axis=-1)

    print(f"\nfidu_index.shape = {fidu_index.shape}")
    print(f"fidu_cls.shape = {fidu_cls.shape}")

"""
Created June 2023
Author: Arne Thomsen

Utils to preprocess the raw network predictions and human defined summary statistics for density estimation. This for
example entails concatenating the example and cosmology axes.
"""

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from msi.utils import plotting, input_output
from msfm.utils import logger, cross_statistics, parameters, files

LOGGER = logger.get_logger(__file__)


def get_reshaped_network_preds(base_dir, model_dir, n_steps=None, file_label=None, preds_file=None, n_params=None):
    file_dict = input_output.load_network_preds(
        base_dir, model_dir, n_steps=n_steps, file_label=file_label, preds_file=preds_file
    )

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

    return fidu_preds, grid_preds, grid_cosmos, file_dict


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
    file_dict = input_output.load_human_summaries(base_dir, summary_type, file_label=file_label)

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

    return fidu_summs, grid_summs, grid_cosmos, file_dict

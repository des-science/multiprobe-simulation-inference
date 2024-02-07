# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2023
Author: Arne Thomsen

Utils to preprocess the raw network predictions and human defined summary statistics for density estimation. This for
example entails concatenating the example and cosmology axes.
"""

import os, re
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from msi.utils.sklearn import GeneralizedSklearnModel
from msi.utils import plotting, input_output
from msfm.utils import logger, cross_statistics, parameters, files, power_spectra

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
    concat_example_dim=True,
    do_plot=True,
    # selection
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=True,
    # CLs scale cuts
    l_mins=None,
    l_maxs=None,
    n_bins=None,
    # peaks scale selection
    scale_indices=None,
    # additional preprocessing
    apply_log=False,
    standardize=False,
    pca_components=None,
):
    assert summary_type in ["cls", "peaks"], "Only cls and peaks are supported"
    apply_cl_scale_cut = (l_mins is not None) and (l_maxs is not None) and (n_bins is not None)

    if summary_type == "cls":
        # apply scale cuts to the raw Cls
        if apply_cl_scale_cut:
            LOGGER.info(f"Applying scale cuts to the raw Cls")

            with np.printoptions(precision=1, suppress=True, floatmode="fixed"):
                LOGGER.info(f"l_mins = {l_mins}")
                LOGGER.info(f"l_maxs = {l_maxs}")

                bin_names = f"l_mins={np.array(l_mins)},l_maxs={np.array(l_maxs)},n_bins={n_bins}"
                bin_names = re.sub(r"\s+", ",", bin_names)

                fidu_file = os.path.join(base_dir, summary_type, f"fiducial_{bin_names}.npy")
                grid_file = os.path.join(base_dir, summary_type, f"grid_{bin_names}.npy")

            try:
                fidu_summs = np.load(fidu_file)
                grid_summs = np.load(grid_file)
                file_dict = input_output.load_human_summaries(
                    base_dir, summary_type, file_label=file_label, return_raw_cls=False
                )
                LOGGER.info(f"Loaded the binned Cls from {bin_names}")
            except FileNotFoundError:
                file_dict = input_output.load_human_summaries(
                    base_dir, summary_type, file_label=file_label, return_raw_cls=True
                )
                LOGGER.warning(f"Applying the scale cuts to the raw Cls, this takes a while and consumes a lot of RAM")
                LOGGER.timer.start("scale_cuts")
                fidu_summs, _ = power_spectra.bin_cls(file_dict["fiducial/cls/raw"], l_mins, l_maxs, n_bins)
                grid_summs, _ = power_spectra.bin_cls(file_dict["grid/cls/raw"], l_mins, l_maxs, n_bins)
                LOGGER.info(f"Done after {LOGGER.timer.elapsed('scale_cuts')}")

                np.save(fidu_file, fidu_summs)
                np.save(grid_file, grid_summs)
                LOGGER.info(f"Saved the binned Cls to {bin_names}")

        # load the pre-binned Cls
        else:
            LOGGER.info(f"Loading the pre-binned Cls")
            file_dict = input_output.load_human_summaries(
                base_dir, summary_type, file_label=file_label, return_raw_cls=False
            )
            fidu_summs = file_dict[f"fiducial/cls/binned"]
            grid_summs = file_dict[f"grid/cls/binned"]

    elif summary_type == "peaks":
        LOGGER.info(f"Loading the pre-binned peak statistics")
        
        if apply_cl_scale_cut:
            LOGGER.warning(f"The scale cuts are baked into the peak statistics, ignoring the l_mins, l_maxs, and n_bins arguments")

        file_dict = input_output.load_human_summaries(
            base_dir, summary_type, file_label=file_label, return_raw_cls=False
        )
        fidu_summs = file_dict[f"fiducial/{summary_type}"]
        grid_summs = file_dict[f"grid/{summary_type}"]

    else:
        raise ValueError

    grid_cosmos = file_dict["grid/cosmo"]

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
    LOGGER.info(f"Shapes after probe selection")
    LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
    LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")
    LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")

    # concatenate the bins along the last axis
    fidu_summs = np.concatenate([fidu_summs[..., i] for i in range(fidu_summs.shape[-1])], axis=-1)
    grid_summs = np.concatenate([grid_summs[..., i] for i in range(grid_summs.shape[-1])], axis=-1)

    # TODO implement scale selection
    if summary_type == "peaks":
        if scale_indices is None:
            assert fidu_summs.shape[-2] == grid_summs.shape[-2], "The number of scales must be the same for fiducial and grid"
            scale_indices = range(fidu_summs.shape[-2])

        # concatenate the scales along the last axis
        fidu_summs = np.concatenate([fidu_summs[..., i, :] for i in scale_indices], axis=-1)
        grid_summs = np.concatenate([grid_summs[..., i, :] for i in scale_indices], axis=-1)

        print("\n")
        LOGGER.info("Shapes after scale selection")
        LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
        LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")

    # concatenate the examples along the first axis
    if concat_example_dim and summary_type:
        # TODO this is due to how it's stored in the .h5 files and not super clean
        if summary_type == "cls":
            grid_cosmos = np.concatenate([grid_cosmos[i, ...] for i in range(grid_cosmos.shape[0])], axis=0)
        elif summary_type == "peaks":
            grid_cosmos = np.repeat(grid_cosmos, repeats=grid_summs.shape[1], axis=0)
        
        grid_summs = np.concatenate([grid_summs[i, ...] for i in range(grid_summs.shape[0])], axis=0)

        print("\n")
        LOGGER.info("Shapes after concatenation")
        LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
        LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")
        LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")

    if do_plot:
        assert concat_example_dim, f"Plotting only works if the examples are concatenated"

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

    if standardize:
        LOGGER.info(f"Scaling the {summary_type} to zero mean and unit variance")
        scaler = GeneralizedSklearnModel(StandardScaler())
        grid_summs = scaler.fit_transform(grid_summs)
        fidu_summs = scaler.transform(fidu_summs)

    if pca_components is not None:
        LOGGER.info(f"Applying PCA to compress to {pca_components} components")
        # whitening doesn't change the results much
        pca = GeneralizedSklearnModel(PCA(n_components=pca_components, whiten=True))
        grid_summs = pca.fit_transform(grid_summs)
        fidu_summs = pca.transform(fidu_summs)
        LOGGER.info(f"Total explained variance = {np.sum(pca.model.explained_variance_ratio_)}")

        print("\n")
        LOGGER.info("Shapes after pre-processing")
        LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
        LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")
        LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")

    return fidu_summs, grid_summs, grid_cosmos, file_dict

# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2023
Author: Arne Thomsen

Utils to preprocess the raw network predictions and human defined summary statistics for density estimation. This for
example entails concatenating the example and cosmology axes.
"""

import os, re
import numpy as np

from scipy.stats import binned_statistic
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from msfm.utils import logger, cross_statistics, parameters, files, power_spectra, observation, scales
from deep_lss.utils import configuration
from msi.utils.sklearn import GeneralizedSklearnModel
from msi.utils import plotting, input_output

LOGGER = logger.get_logger(__file__)


def get_reshaped_network_preds(
    base_dir, model_dir, n_steps=None, file_label=None, preds_file=None, n_params=None, n_perms_per_cosmo=None
):
    file_dict = input_output.load_network_preds(
        base_dir, model_dir, n_steps=n_steps, file_label=file_label, preds_file=preds_file
    )

    fidu_preds = file_dict["fiducial/vali/pred"]
    grid_preds = file_dict["grid/pred"]
    grid_cosmos = file_dict["grid/cosmo"]

    # only relevant for the likelihood loss
    fidu_preds = fidu_preds[..., :n_params]
    grid_preds = grid_preds[..., :n_params]

    # only take a subset of the permutations
    if n_perms_per_cosmo is not None:
        LOGGER.warning(f"Only taking the first {n_perms_per_cosmo} permutations per cosmology")
        LOGGER.warning(f"n_patches and n_noise are hard-coded here!")
        n_patches = 4
        n_noise = 3
        grid_preds = grid_preds[:, : (n_perms_per_cosmo * n_patches * n_noise), :]

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
    msfm_conf=None,
    dlss_conf=None,
    params=None,
    concat_example_dim=True,
    do_plot=True,
    # selection
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    # power spectra specific
    from_raw_cls=False,
    l_mins=None,
    l_maxs=None,
    n_bins=None,
    only_keep_bins=None,
    fixed_binning=False,
    # peaks specific
    scale_indices=None,
    # additional preprocessing
    apply_log=False,
    standardize=False,
    pca_components=None,
):
    assert summary_type in ["cls", "peaks"], "Only cls and peaks are supported"

    msfm_conf = files.load_config(msfm_conf)
    dlss_conf = configuration.load_deep_lss_config(dlss_conf)

    noise_cls = None

    if summary_type == "cls":
        if l_maxs is None:
            theta_fwhm = (
                dlss_conf["scale_cuts"]["lensing"]["theta_fwhm"] + dlss_conf["scale_cuts"]["clustering"]["theta_fwhm"]
            )
            l_maxs = scales.angle_to_ell(np.array(theta_fwhm), arcmin=dlss_conf["scale_cuts"]["arcmin"])
            LOGGER.info(f"Using l_maxs = {l_maxs} from the dlss config")
        if l_mins is None:
            l_mins = np.zeros_like(l_maxs)
            LOGGER.info(f"Using l_mins = {l_mins} by default (no smoothing)")
        if n_bins is None:
            n_bins = msfm_conf["analysis"]["power_spectra"]["n_bins"]
            LOGGER.info(f"Using n_bins = {n_bins} from the msfm config")

        # apply scale cuts to the raw Cls
        if from_raw_cls:
            LOGGER.warning(f"Applying scale cuts to the raw Cls, this is deprecated")

            assert (
                (l_mins is not None) and (l_maxs is not None) and (n_bins is not None)
            ), "The l_mins, l_maxs, and n_bins arguments must be provided"

            with np.printoptions(precision=1, suppress=True, floatmode="fixed"):
                LOGGER.info(f"l_mins = {l_mins}")
                LOGGER.info(f"l_maxs = {l_maxs}")

                bin_names = f"l_mins={np.array(l_mins)},l_maxs={np.array(l_maxs)},n_bins={n_bins},fixed_binning={fixed_binning}"
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
                fidu_summs, _ = power_spectra.smooth_and_bin_cls(
                    file_dict["fiducial/cls/raw"],
                    l_mins,
                    l_maxs,
                    n_bins,
                    n_side=msfm_conf["analysis"]["n_side"],
                    fixed_binning=fixed_binning,
                )
                grid_summs, _ = power_spectra.smooth_and_bin_cls(
                    file_dict["grid/cls/raw"],
                    l_mins,
                    l_maxs,
                    n_bins,
                    n_side=msfm_conf["analysis"]["n_side"],
                    fixed_binning=fixed_binning,
                )
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

            LOGGER.info(f"Applying scale cuts to the pre-binned Cls")

            # binning
            ells = np.arange(0, 3 * msfm_conf["analysis"]["n_side"])
            bins = power_spectra.get_cl_bins(
                msfm_conf["analysis"]["power_spectra"]["l_min"],
                msfm_conf["analysis"]["power_spectra"]["l_max"],
                msfm_conf["analysis"]["power_spectra"]["n_bins"],
            )

            # white noise
            noise_cls = input_output.load_cl_white_noise(base_dir)
            white_noise_sigma = (
                dlss_conf["scale_cuts"]["lensing"]["white_noise_sigma"]
                + dlss_conf["scale_cuts"]["clustering"]["white_noise_sigma"]
            )

            n_z = len(l_maxs)
            k = 0
            for i in range(n_z):
                for j in range(n_z):
                    if (i == j) or (i < j):
                        # the theta_fwhm formulation is entirely equivalent
                        # smoothing_fac = scales.gaussian_low_pass_factor_alm(
                        #     ells, theta_fwhm=max(theta_fwhm[i], theta_fwhm[j]), arcmin=dlss_conf["scale_cuts"]["arcmin"]
                        # )

                        # smoothing_fac = np.ones_like(ells, dtype=np.float32)
                        # if l_mins[i] is not None and l_mins[j] is not None:
                        #     smoothing_fac *= scales.gaussian_high_pass_factor_alm(
                        #         ells, l_min=max(l_mins[i], l_mins[j])
                        #     )
                        # if l_maxs[i] is not None and l_maxs[j] is not None:
                        #     smoothing_fac *= scales.gaussian_low_pass_factor_alm(ells, l_max=min(l_maxs[i], l_maxs[j]))

                        smoothing_fac = scales.gaussian_high_pass_factor_alm(ells, l_min=max(l_mins[i], l_mins[j]))
                        smoothing_fac *= scales.gaussian_low_pass_factor_alm(ells, l_max=min(l_maxs[i], l_maxs[j]))
                        smoothing_fac = smoothing_fac**2
                        smoothing_fac = binned_statistic(ells, smoothing_fac, statistic="mean", bins=bins)[0]

                        fidu_summs[..., k] *= smoothing_fac
                        grid_summs[..., k] *= smoothing_fac
                        noise_cls[..., k] *= white_noise_sigma[i] * white_noise_sigma[j]

                        k += 1

    elif summary_type == "peaks":
        LOGGER.info(f"Loading the pre-binned peak statistics")

        LOGGER.warning(
            f"The scale cuts are baked into the peak statistics, ignoring the l_mins, l_maxs, and n_bins arguments"
        )

        file_dict = input_output.load_human_summaries(
            base_dir, summary_type, file_label=file_label, return_raw_cls=False
        )
        fidu_summs = file_dict[f"fiducial/{summary_type}"]
        grid_summs = file_dict[f"grid/{summary_type}"]

    else:
        raise ValueError

    grid_cosmos = file_dict["grid/cosmo"]
    grid_i_sobols = file_dict["grid/i_sobol"]

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
    if noise_cls is not None:
        noise_cls = noise_cls[..., bin_indices]

    if only_keep_bins is not None:
        LOGGER.warning(f"Keeping only the first {only_keep_bins} bins")
        fidu_summs = fidu_summs[..., :only_keep_bins, :]
        grid_summs = grid_summs[..., :only_keep_bins, :]
        if noise_cls is not None:
            noise_cls = noise_cls[..., :only_keep_bins, :]

    # select the right cosmological parameters
    msfm_conf = files.load_config(msfm_conf)
    all_params = parameters.get_parameters(None, msfm_conf)
    params = parameters.get_parameters(params, msfm_conf)

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
    LOGGER.info(f"grid_i_sobols = {grid_i_sobols.shape}")

    # concatenate the bins along the last axis
    fidu_summs = np.concatenate([fidu_summs[..., i] for i in range(fidu_summs.shape[-1])], axis=-1)
    grid_summs = np.concatenate([grid_summs[..., i] for i in range(grid_summs.shape[-1])], axis=-1)
    if noise_cls is not None:
        noise_cls = np.concatenate([noise_cls[..., i] for i in range(noise_cls.shape[-1])], axis=-1)

    # TODO implement scale selection
    if summary_type == "peaks":
        if scale_indices is None:
            assert (
                fidu_summs.shape[-2] == grid_summs.shape[-2]
            ), "The number of scales must be the same for fiducial and grid"
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
            grid_i_sobols = np.concatenate([grid_i_sobols[i, ...] for i in range(grid_i_sobols.shape[0])], axis=0)
        elif summary_type == "peaks":
            grid_cosmos = np.repeat(grid_cosmos, repeats=grid_summs.shape[1], axis=0)

        grid_summs = np.concatenate([grid_summs[i, ...] for i in range(grid_summs.shape[0])], axis=0)

        print("\n")
        LOGGER.info("Shapes after concatenation")
        LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
        LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")
        LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")
        LOGGER.info(f"grid_i_sobols = {grid_i_sobols.shape}")

    if do_plot:
        assert concat_example_dim, f"Plotting only works if the examples are concatenated"

        LOGGER.info(f"Plotting the selected raw {summary_type}")
        label = f"lensing={with_lensing},clustering={with_clustering},cross_z={with_cross_z},cross_probe={with_cross_probe}"

        if summary_type == "cls":
            plotting.plot_human_summary(
                fidu_summs,
                grid_summs,
                os.path.join(base_dir, summary_type),
                label=label,
                bin_size=msfm_conf["analysis"]["power_spectra"]["n_bins"] - 1,
                bin_names=bin_names,
            )
        elif summary_type == "peaks":
            plotting.plot_human_summary(
                fidu_summs,
                grid_summs,
                os.path.join(base_dir, summary_type),
                label=label,
                bin_size=msfm_conf["analysis"]["peak_statistics"]["n_bins"],
                bin_names=bin_names,
            )

    grid_summs, scaler, pca = preprocess_human_summaries(
        grid_summs, apply_log, standardize=standardize, pca_components=pca_components
    )
    fidu_summs, _, _ = preprocess_human_summaries(
        fidu_summs, apply_log, standardize=standardize, pca_components=pca_components, scaler=scaler, pca=pca
    )
    if noise_cls is not None:
        noise_cls, _, _ = preprocess_human_summaries(
            noise_cls, apply_log, standardize=standardize, pca_components=pca_components, scaler=scaler, pca=pca
        )

    print("\n")
    LOGGER.info("Shapes after pre-processing")
    LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
    LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")
    LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")

    return fidu_summs, grid_summs, noise_cls, grid_cosmos, grid_i_sobols, file_dict, scaler, pca


def preprocess_human_summaries(
    summaries, apply_log=False, standardize=False, pca_components=None, scaler=None, pca=None
):
    if apply_log:
        LOGGER.info(f"Taking the logarithm of the absolute values.")
        summaries = np.log(np.abs(summaries))

    if standardize and scaler is None:
        LOGGER.info(f"Fitting the scaler to transform to zero mean and unit variance")
        scaler = GeneralizedSklearnModel(StandardScaler())
        summaries = scaler.fit_transform(summaries)
    elif isinstance(scaler, GeneralizedSklearnModel):
        LOGGER.info(f"Applying the scaler to transform to zero mean and unit variance")
        summaries = scaler.transform(summaries)

    if pca_components is not None and pca is None:
        LOGGER.info(f"Fitting PCA to compress to {pca_components} components")
        pca = GeneralizedSklearnModel(PCA(n_components=pca_components, whiten=False))
        summaries = np.nan_to_num(summaries)
        summaries = pca.fit_transform(summaries)
        LOGGER.info(f"Total explained variance = {np.sum(pca.model.explained_variance_ratio_)}")
    elif isinstance(pca, GeneralizedSklearnModel):
        LOGGER.info(f"Applying PCA to compress to {pca_components} components")
        summaries = np.nan_to_num(summaries)
        summaries = pca.transform(summaries)

    summaries = np.nan_to_num(summaries)

    return summaries, scaler, pca


def get_preprocessed_cl_observation(
    wl_gamma_map=None,
    gc_count_map=None,
    # configuration
    msfm_conf=None,
    dlss_conf=None,
    base_dir=None,
    from_raw_cls=False,
    # selection
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    # CLs scale cuts
    l_mins=None,
    l_maxs=None,
    n_bins=None,
    only_keep_bins=None,
    # additional preprocessing
    apply_log=False,
    standardize=False,
    pca_components=None,
    scaler=None,
    pca=None,
):
    msfm_conf = files.load_config(msfm_conf)
    dlss_conf = configuration.load_deep_lss_config(dlss_conf)

    if l_maxs is None:
        theta_fwhm = (
            dlss_conf["scale_cuts"]["lensing"]["theta_fwhm"] + dlss_conf["scale_cuts"]["clustering"]["theta_fwhm"]
        )
        l_maxs = scales.angle_to_ell(np.array(theta_fwhm), arcmin=dlss_conf["scale_cuts"]["arcmin"])
        LOGGER.info(f"Using l_maxs = {l_maxs} from the dlss config")
    if l_mins is None:
        l_mins = np.zeros_like(l_maxs, dtype=int)
        LOGGER.info(f"Using l_mins = {l_mins} by default (no smoothing)")

    _, obs_cl, _ = observation.forward_model_observation_map(
        wl_gamma_map=wl_gamma_map,
        gc_count_map=gc_count_map,
        conf=msfm_conf,
        apply_norm=True,
        with_padding=True,
        nest=False,
    )

    # apply the same transformations as in get_reshaped_human_summaries to an observation as put out by
    # msfm.observation.forward_model_observation_map
    if from_raw_cls:
        LOGGER.warning(f"Applying scale cuts to the raw Cls, this is deprecated")
        obs_cl, _ = power_spectra.smooth_and_bin_cls(
            obs_cl,
            l_mins_smoothing=l_mins,
            l_maxs_smoothing=l_maxs,
            n_bins=n_bins,
            n_side=msfm_conf["analysis"]["n_side"],
            with_cross=True,
            fixed_binning=False,
        )
    else:
        obs_cl, _ = power_spectra.smooth_and_bin_cls(
            obs_cl,
            l_mins_smoothing=l_mins,
            l_maxs_smoothing=l_maxs,
            with_cross=True,
            fixed_binning=True,
            n_bins=msfm_conf["analysis"]["power_spectra"]["n_bins"],
            l_min_binning=msfm_conf["analysis"]["power_spectra"]["l_min"],
            l_max_binning=msfm_conf["analysis"]["power_spectra"]["l_max"],
        )

    # like in get_reshaped_human_summaries
    if base_dir is not None:
        noise_cl = input_output.load_cl_white_noise(base_dir)[0]

        white_noise_sigma = (
            dlss_conf["scale_cuts"]["lensing"]["white_noise_sigma"]
            + dlss_conf["scale_cuts"]["clustering"]["white_noise_sigma"]
        )
        n_z = len(white_noise_sigma)
        k = 0
        for i in range(n_z):
            for j in range(n_z):
                if (i == j) or (i < j):
                    noise_cl[:, k] *= white_noise_sigma[i] * white_noise_sigma[j]
                    k += 1

        obs_cl += noise_cl
        LOGGER.info(f"Adding white noise to the observation")
    else:
        LOGGER.warning(f"Not adding white noise to the observation!")

    bin_indices, _ = cross_statistics.get_cross_bin_indices(
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
    )
    obs_cl = obs_cl[..., bin_indices]
    if only_keep_bins is not None:
        obs_cl = obs_cl[..., :only_keep_bins, :]

    # concatenate the bins along the last axis
    obs_cl = np.concatenate([obs_cl[..., i] for i in range(obs_cl.shape[-1])], axis=-1)

    obs_cl, _, _ = preprocess_human_summaries(
        obs_cl[np.newaxis],
        apply_log=apply_log,
        standardize=standardize,
        pca_components=pca_components,
        scaler=scaler,
        pca=pca,
    )

    plotting.plot_single_power_spectrum(
        obs_cl,
        bin_size=msfm_conf["analysis"]["power_spectra"]["n_bins"] - 1,
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
    )

    return obs_cl

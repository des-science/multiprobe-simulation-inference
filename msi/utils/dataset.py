import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from msfm.utils import cross_statistics
from msi.utils import preprocessing, plotting


def get_binned_power_spectra_dset(
    base_dir,
    # file
    file_label=None,
    # configuration
    msfm_conf=None,
    dlss_conf=None,
    params=None,
    train_test_split=0.8,
    n_examples_to_plot=10,
    cls_from_maps=False,
    # tf.data
    batch_size=2**12,
    shuffle_buffer=2**14,
    prefetch=3,
    num_parallel_calls=tf.data.AUTOTUNE,
    float_type=np.float32,
    # selection
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    with_gaussian_noise=True,
    bin_indices=None,
    # CLs scale cuts
    l_mins=None,
    l_maxs=None,
    theta_fwhms=None,
    white_noise_sigmas=None,
    n_bins=None,
    # additional preprocessing
    apply_log=True,
    standardize=False,
):
    fidu_cls, grid_cls, noise_cls, grid_cosmos, grid_i_sobols, file_dict, scaler, pca = (
        preprocessing.get_reshaped_human_summaries(
            base_dir,
            "cls",
            file_label=file_label,
            # configuration
            msfm_conf=msfm_conf,
            dlss_conf=dlss_conf,
            params=params,
            concat_example_dim=False,
            do_plot=False,
            # selection
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_cross_z=with_cross_z,
            with_cross_probe=with_cross_probe,
            bin_indices=bin_indices,
            # power spectra: scales
            from_raw_cls=False,
            l_mins=l_mins,
            l_maxs=l_maxs,
            theta_fwhms=theta_fwhms,
            white_noise_sigmas=white_noise_sigmas,
            n_bins=n_bins,
            cls_from_maps=cls_from_maps,
            # unlike the standardization, the logarithm is not linear and has to be applied as log(signal + noise), not
            # log(signal) + log(noise)
            apply_log=False,
            standardize=standardize,
        )
    )

    fidu_cls = fidu_cls.astype(float_type)
    grid_cls = grid_cls.astype(float_type)
    grid_cosmos = grid_cosmos.astype(float_type)
    noise_cls = noise_cls.astype(float_type)

    # split along the "examples per cosmo" axis
    i_split = int(train_test_split * grid_cls.shape[1])

    grid_cls_train = grid_cls[:, :i_split, :]
    grid_cls_test = grid_cls[:, i_split:, :]
    grid_cosmos_train = grid_cosmos[:, :i_split, :]
    grid_cosmos_test = grid_cosmos[:, i_split:, :]

    _concat_example_axis = lambda array: np.concatenate([array[i, ...] for i in range(array.shape[0])], axis=0)

    grid_cls_train = _concat_example_axis(grid_cls_train)
    grid_cls_test = _concat_example_axis(grid_cls_test)
    grid_cosmos_train = _concat_example_axis(grid_cosmos_train)
    grid_cosmos_test = _concat_example_axis(grid_cosmos_test)

    def _augmentations(example, noise):
        signal, label = example

        if with_gaussian_noise:
            signal += noise

        if apply_log:
            signal = tf.math.log(tf.math.abs(signal))

        signal = tf.where(tf.math.is_finite(signal), signal, tf.zeros_like(signal))

        return signal, label

    # create the datasets
    dset_noise = tf.data.Dataset.from_tensor_slices(noise_cls).cache().repeat().shuffle(shuffle_buffer)

    dset_train = (
        tf.data.Dataset.from_tensor_slices((grid_cls_train, grid_cosmos_train))
        .cache()
        .repeat()
        .shuffle(shuffle_buffer)
    )
    dset_train = (
        tf.data.Dataset.zip((dset_train, dset_noise))
        .batch(batch_size)
        .map(_augmentations, num_parallel_calls=num_parallel_calls, deterministic=False)
        .prefetch(prefetch)
    )

    dset_test = tf.data.Dataset.from_tensor_slices((grid_cls_test, grid_cosmos_test)).cache()
    dset_test = (
        tf.data.Dataset.zip((dset_test, dset_noise))
        .batch(batch_size)
        .map(_augmentations, num_parallel_calls=num_parallel_calls, deterministic=True)
        .prefetch(prefetch)
    )

    # add the white noise to the non-dataset cls too
    rng = np.random.default_rng()

    def _noise_and_log(cls):
        if with_gaussian_noise:
            cls += noise_cls[rng.integers(low=0, high=noise_cls.shape[0], size=cls.shape[0])]

        cls = preprocessing.preprocess_human_summaries(cls, apply_log=apply_log)[0]

        return cls

    fidu_cls = _noise_and_log(fidu_cls)
    grid_cls_train = _noise_and_log(grid_cls_train)
    grid_cls_test = _noise_and_log(grid_cls_test)

    plotting.plot_human_summary(
        fidu_cls,
        grid_cls_train,
        bin_size=msfm_conf["analysis"]["power_spectra"]["n_bins"] - 1,
        n_random_indices=n_examples_to_plot,
        yscale="linear",
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
    )

    out_dict = {
        "fidu/cls": fidu_cls,
        "grid/cls/train": grid_cls_train,
        "grid/cls/test": grid_cls_test,
        "grid/cosmos/train": grid_cosmos_train,
        "grid/cosmos/test": grid_cosmos_test,
    }

    return dset_train, dset_test, out_dict


def get_noisy_cls(
    base_dir,
    # file
    file_label=None,
    # configuration
    msfm_conf=None,
    dlss_conf=None,
    params=None,
    train_test_split=0.8,
    n_examples_to_plot=10,
    cls_from_maps=False,
    float_type=np.float32,
    # selection
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    with_gaussian_noise=True,
    bin_indices=None,
    # CLs scale cuts
    l_mins=None,
    l_maxs=None,
    theta_fwhms=None,
    white_noise_sigmas=None,
    n_bins=None,
    # additional preprocessing
    apply_log=True,
    standardize=False,
):
    fidu_cls, grid_cls, noise_cls, grid_cosmos, grid_i_sobols, file_dict, scaler, pca = (
        preprocessing.get_reshaped_human_summaries(
            base_dir,
            "cls",
            file_label=file_label,
            # configuration
            msfm_conf=msfm_conf,
            dlss_conf=dlss_conf,
            params=params,
            concat_example_dim=False,
            do_plot=False,
            # selection
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_cross_z=with_cross_z,
            with_cross_probe=with_cross_probe,
            bin_indices=bin_indices,
            # power spectra: scales
            from_raw_cls=False,
            l_mins=l_mins,
            l_maxs=l_maxs,
            theta_fwhms=theta_fwhms,
            white_noise_sigmas=white_noise_sigmas,
            n_bins=n_bins,
            cls_from_maps=cls_from_maps,
            # unlike the standardization, the logarithm is not linear and has to be applied as log(signal + noise), not
            # log(signal) + log(noise)
            apply_log=False,
            standardize=standardize,
        )
    )

    fidu_cls = fidu_cls.astype(float_type)
    grid_cls = grid_cls.astype(float_type)
    grid_cosmos = grid_cosmos.astype(float_type)
    noise_cls = noise_cls.astype(float_type)

    # split along the "examples per cosmo" axis
    i_split = int(train_test_split * grid_cls.shape[1])

    grid_cls_train = grid_cls[:, :i_split, :]
    grid_cls_test = grid_cls[:, i_split:, :]
    grid_cosmos_train = grid_cosmos[:, :i_split, :]
    grid_cosmos_test = grid_cosmos[:, i_split:, :]

    _concat_example_axis = lambda array: np.concatenate([array[i, ...] for i in range(array.shape[0])], axis=0)

    grid_cls_train = _concat_example_axis(grid_cls_train)
    grid_cls_test = _concat_example_axis(grid_cls_test)
    grid_cosmos_train = _concat_example_axis(grid_cosmos_train)
    grid_cosmos_test = _concat_example_axis(grid_cosmos_test)

    # add the white noise to the non-dataset cls too
    rng = np.random.default_rng()

    def _noise_and_log(cls):
        if with_gaussian_noise:
            cls += noise_cls[rng.integers(low=0, high=noise_cls.shape[0], size=cls.shape[0])]

        cls = preprocessing.preprocess_human_summaries(cls, apply_log=apply_log)[0]

        return cls

    fidu_cls = _noise_and_log(fidu_cls)
    grid_cls_train = _noise_and_log(grid_cls_train)
    grid_cls_test = _noise_and_log(grid_cls_test)

    plotting.plot_human_summary(
        fidu_cls,
        grid_cls_train,
        bin_size=msfm_conf["analysis"]["power_spectra"]["n_bins"] - 1,
        n_random_indices=n_examples_to_plot,
        yscale="linear",
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
    )

    out_dict = {
        "fidu/cls": fidu_cls,
        "grid/cls/train": grid_cls_train,
        "grid/cls/test": grid_cls_test,
        "grid/cosmos/train": grid_cosmos_train,
        "grid/cosmos/test": grid_cosmos_test,
    }

    return out_dict


def get_binned_power_spectra_dset_legacy(
    base_dir,
    # file
    file_label=None,
    # configuration
    msfm_conf=None,
    dlss_conf=None,
    params=None,
    train_test_split=0.8,
    n_examples_to_plot=10,
    # tf.data
    batch_size=2**12,
    shuffle_buffer=2**14,
    prefetch=3,
    float_type=np.float32,
    # selection
    with_lensing=True,
    with_clustering=True,
    # CLs scale cuts
    l_mins=None,
    l_maxs=None,
    n_bins=None,
    fixed_binning=False,
    # additional preprocessing
    apply_log=False,
    standardize=False,
    pca_components=None,
):
    fidu_cls, grid_cls, noise_cls, grid_cosmos, grid_i_sobols, file_dict, scaler, pca = (
        preprocessing.get_reshaped_human_summaries(
            base_dir,
            "cls",
            # file
            file_label=file_label,
            # configuration
            msfm_conf=msfm_conf,
            dlss_conf=dlss_conf,
            params=params,
            concat_example_dim=False,
            do_plot=False,
            # selection
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_cross_z=True,
            with_cross_probe=(with_lensing and with_clustering),
            # power spectra: scales
            from_raw_cls=True,
            l_mins=l_mins,
            l_maxs=l_maxs,
            n_bins=n_bins,
            fixed_binning=fixed_binning,
            # additional preprocessing
            apply_log=apply_log,
            standardize=standardize,
            pca_components=pca_components,
        )
    )

    fidu_cls = fidu_cls.astype(float_type)
    grid_cls = grid_cls.astype(float_type)
    grid_cosmos = grid_cosmos.astype(float_type)

    # split along the "examples per cosmo" axis
    i_split = int(train_test_split * grid_cls.shape[1])

    grid_cls_train = grid_cls[:, :i_split, :]
    grid_cls_test = grid_cls[:, i_split:, :]

    grid_cosmos_train = grid_cosmos[:, :i_split, :]
    grid_cosmos_test = grid_cosmos[:, i_split:, :]

    grid_cls_train = np.concatenate([grid_cls_train[i, ...] for i in range(grid_cls_train.shape[0])], axis=0)
    grid_cls_test = np.concatenate([grid_cls_test[i, ...] for i in range(grid_cls_test.shape[0])], axis=0)

    grid_cosmos_train = np.concatenate([grid_cosmos_train[i, ...] for i in range(grid_cosmos_train.shape[0])], axis=0)
    grid_cosmos_test = np.concatenate([grid_cosmos_test[i, ...] for i in range(grid_cosmos_test.shape[0])], axis=0)

    dset_train = tf.data.Dataset.from_tensor_slices((grid_cls_train, grid_cosmos_train))
    dset_train = dset_train.cache().repeat().shuffle(shuffle_buffer).batch(batch_size).prefetch(prefetch)

    dset_test = tf.data.Dataset.from_tensor_slices((grid_cls_test, grid_cosmos_test))
    dset_test = dset_test.cache().batch(batch_size).prefetch(prefetch)

    plotting.plot_human_summary(
        fidu_cls,
        grid_cls_train,
        bin_size=msfm_conf["analysis"]["power_spectra"]["n_bins"] - 1,
        n_random_indices=n_examples_to_plot,
        yscale="linear",
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=True,
        with_cross_probe=(with_lensing and with_clustering),
    )

    out_dict = {
        "fidu/cls": fidu_cls,
        "grid/cls/train": grid_cls_train,
        "grid/cls/test": grid_cls_test,
        "grid/cosmos/train": grid_cosmos_train,
        "grid/cosmos/test": grid_cosmos_test,
    }

    return dset_train, dset_test, out_dict, scaler, pca

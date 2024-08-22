import tensorflow as tf
import numpy as np

from msi.utils import preprocessing


def get_binned_power_spectra_dset(
    base_dir,
    # file
    file_label=None,
    # configuration
    conf=None,
    params=None,
    train_test_split=0.8,
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
    # additional preprocessing
    apply_log=False,
    standardize=False,
    pca_components=None,
):
    fidu_cls, grid_cls, grid_cosmos, grid_i_sobols, file_dict, scaler, pca = (
        preprocessing.get_reshaped_human_summaries(
            base_dir,
            "cls",
            # file
            file_label=file_label,
            # configuration
            conf=conf,
            params=params,
            concat_example_dim=False,
            do_plot=False,
            # selection
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_cross_z=True,
            with_cross_probe=(with_lensing and with_clustering),
            # power spectra: scales
            l_mins=l_mins,
            l_maxs=l_maxs,
            n_bins=n_bins,
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

    out_dict = {
        "fidu/cls": fidu_cls,
        "grid/cls/train": grid_cls_train,
        "grid/cls/test": grid_cls_test,
        "grid/cosmos/train": grid_cosmos_train,
        "grid/cosmos/test": grid_cosmos_test,
    }

    return dset_train, dset_test, out_dict

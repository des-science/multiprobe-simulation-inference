# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2026
Author: Arne Thomsen

Runs configs (configs/runs/<survey>/<version>/<subversion>/*.yaml): the trained y3-deep-lss runs an analysis
compares, under runs.<representation>.<probe>, each with the pred_dir holding its preds_<steps>.h5.

A pred_dir is either absolute, or relative to the runs directory of the config's dataset: the top-level key
dataset (e.g. desy6/v1/default) names msfm's configs/<dataset>.yaml, whose locations.yaml gives runs_dir
(msfm.utils.locations). A relative pred_dir without a dataset is an error.
"""

import os

import msfm
from msfm.utils import locations
from msfm.utils.input_output import read_yaml

MSFM_DIR = os.path.abspath(os.path.join(os.path.dirname(msfm.__file__), ".."))


def dataset_runs_dir(dataset):
    """The runs directory of a dataset, e.g. desy6/v1/default."""
    msfm_config = os.path.join(MSFM_DIR, "configs", f"{dataset}.yaml")
    if not os.path.isfile(msfm_config):
        raise FileNotFoundError(f"the dataset {dataset} has no msfm config {msfm_config}")
    return locations.get(msfm_config, "runs_dir")


def load_runs_config(path):
    """Reads a runs config, with every relative pred_dir resolved against its dataset's runs directory.

    Returns:
        dict: the config, absolute pred_dirs unchanged
    """
    conf = read_yaml(path)
    dataset = conf.get("dataset")
    runs_dir = dataset_runs_dir(dataset) if dataset is not None else None
    for representation, probes in (conf.get("runs") or {}).items():
        # a representation set to null is disabled
        for probe, run in (probes or {}).items():
            pred_dir = run["pred_dir"]
            if os.path.isabs(pred_dir):
                continue
            if runs_dir is None:
                raise ValueError(
                    f"{path}: runs.{representation}.{probe}.pred_dir {pred_dir!r} is relative, so the config needs "
                    "a dataset key"
                )
            run["pred_dir"] = os.path.join(runs_dir, pred_dir)
    return conf

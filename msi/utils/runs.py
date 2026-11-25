# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2026
Author: Arne Thomsen

Runs configs (configs/runs/<survey>/<version>/*.yaml): the trained y3-deep-lss runs an analysis compares, under
runs.<representation>.<probe>, each with the pred_dir holding its preds_<steps>.h5.

A pred_dir is either absolute, or relative to the runs directory of the config's dataset: the top-level key
dataset (e.g. desy6/v2) names msfm's configs/<dataset>/<version>.yaml, whose locations.yaml gives runs_dir
(msfm.utils.locations). A relative pred_dir without a dataset is an error.
"""

import os

from deep_lss.utils import configuration
from msfm.utils import locations
from msfm.utils.input_output import read_yaml


def dataset_runs_dir(dataset):
    """The runs directory of a dataset, e.g. desy6/v2."""
    msfm_config = locations.dataset_config(dataset)
    if not os.path.isfile(msfm_config):
        raise FileNotFoundError(f"the dataset {dataset} has no msfm config {msfm_config}")
    return locations.get(msfm_config, "runs_dir")


def load_run_configs(pred_dir):
    """The dlss_conf and msfm_conf a trained run saved in its configs.yaml (deep_lss configuration.load_run_configs)."""
    conf = configuration.load_run_configs(os.path.join(pred_dir, "configs.yaml"))
    return conf["dlss"], conf["msfm"]


def run_params(pred_dir):
    """The network target of a trained run, dlss.dset.training.params of the configs.yaml it saved."""
    if not os.path.isfile(os.path.join(pred_dir, "configs.yaml")):
        raise FileNotFoundError(f"{pred_dir} has no configs.yaml; give the run's params in the runs config instead")
    return list(load_run_configs(pred_dir)[0]["dset"]["training"]["params"])


def load_runs_config(path):
    """Reads a runs config, with every relative pred_dir resolved against its dataset's runs directory. A run
    without params takes those it was trained on (run_params); given, they override them.

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
            if not os.path.isabs(pred_dir):
                if runs_dir is None:
                    raise ValueError(
                        f"{path}: runs.{representation}.{probe}.pred_dir {pred_dir!r} is relative, so the config "
                        "needs a dataset key"
                    )
                run["pred_dir"] = os.path.join(runs_dir, pred_dir)
            if run.get("params") is None:
                run["params"] = run_params(run["pred_dir"])
    return conf

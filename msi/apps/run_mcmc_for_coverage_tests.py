# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Sample the MCMC posterior for N random observations from the CosmoGrid for posterior-level coverage testing. This is 
meant for CPU nodes.

example usage:

esub run_mcmc_for_coverage_tests.py \
    --preds_file=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/preds_400000.h5 \
    --flow_dir=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/400000_steps_likelihood_sigmoid_7/likelihood_flow \
    --mode=jobarray --function=all --tasks="0>1000" --n_jobs=1000 \
    --jobname=mcmc --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
"""

import numpy as np
import torch, os, argparse, warnings, h5py, yaml, time

from msfm.utils import logger

from msi.utils import preprocessing
from msi.flow_conductor import architecture
from msi.flow_conductor.likelihood_flow import LikelihoodFlow


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def resources(args):
    args = setup(args)

    if args.cluster == "perlmutter":
        # because of hyperthreading, there's a total of 256 threads per node
        resources = {
            "main_time": 0.2,
            "main_n_cores": 2,
            "main_memory": 1952,
            "main_scratch": 0,
            "merge_time": 0.1,
            "merge_n_cores": 8,
            "merge_memory": 1952,
            "merge_scratch": 0,
        }
    elif args.cluster == "euler":
        resources = {"main_time": 4, "main_memory": 4096, "main_n_cores": 8, "merge_memory": 4096, "merge_n_cores": 16}

    return resources


def setup(args):
    description = "evaluate the power spectra from the input pipelines"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "--preds_file",
        type=str,
        required=True,
        help="directory containing the predictions of the compression network",
    )
    parser.add_argument(
        "--flow_dir",
        type=str,
        required=True,
        help="directory containing the flow network",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="perlmutter",
        choices=("perlmutter", "euler"),
        help="the cluster to execute on",
    )
    parser.add_argument(
        "--max_sleep",
        type=int,
        default=60,
        help="set the maximal amount of time to sleep before copying to avoid clashes",
    )
    parser.add_argument(
        "--torch_seed",
        type=int,
        default=7,
        help="seed for the torch random number generator, used for the shape and Poisson noise",
    )
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    return args


def main(indices, args):
    args = setup(args)

    n_walkers = 128
    n_burnin_steps = 500
    n_samples = n_walkers * 1000

    if args.debug:
        args.max_sleep = 0
        n_burnin_steps = 10
        n_samples = n_walkers * 100
        LOGGER.warning("!!! debug mode !!!")
    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    model, x_true, theta_true = _set_up_flow(args)

    # subselect cosmologies
    n_cosmos = x_true.shape[0]
    n_cosmos_test = len(indices)
    x_true = x_true[:: x_true.shape[0] // n_cosmos_test]
    theta_true = theta_true[:: n_cosmos // n_cosmos_test]

    for index in indices:
        current_x_true = x_true[index]
        current_theta_true = theta_true[index]

        current_theta_samples = model.sample_posterior(
            current_x_true,
            n_walkers=n_walkers,
            n_burnin_steps=n_burnin_steps,
            n_samples=n_samples,
        )

        out_file = os.path.join(args.flow_dir, f"mcmc_samples_{index}.h5")
        with h5py.File(out_file, "w") as f:
            f.create_dataset("x_true", data=current_x_true)
            f.create_dataset("theta_true", data=current_theta_true)
            f.create_dataset("theta_samples", data=current_theta_samples)

        yield index


def _set_up_flow(args):
    """Like in the jupyter notebook where this flow has been trained"""

    # constants
    if "lensing" in args.preds_file:
        params = ["Om", "s8", "w0", "Aia", "n_Aia", "bta"]
    elif "clustering" in args.preds_file:
        params = ["Om", "s8", "w0", "bg1", "bg2", "bg3", "bg4"]
    elif "combined" in args.preds_file:
        params = ["Om", "s8", "w0", "Aia", "n_Aia", "bta", "bg1", "bg2", "bg3", "bg4"]
    else:
        raise ValueError(f"Unknown prediction file {args.preds_file}")

    model_dir = os.path.dirname(args.preds_file)
    with open(os.path.join(model_dir, "configs.yaml"), "r") as f:
        net_conf, dlss_conf, msfm_conf = list(yaml.load_all(f, Loader=yaml.FullLoader))

    _, grid_preds, grid_cosmos, _ = preprocessing.get_reshaped_network_preds(
        "",
        "",
        preds_file=args.preds_file,
        with_fidu=False,
    )

    # input dimensions
    x_dim = grid_preds.shape[-1]
    theta_dim = grid_cosmos.shape[-1]

    # NOTE this is hacky, the parameters entered here have to match the ones in the weights file

    # shared hyperparameters
    context_embedding_dim = 32

    embedding_net = architecture.get_context_embedding_net(
        context_dim=theta_dim,
        context_embedding_dim=context_embedding_dim,
        hidden_dim=64,
        n_blocks=3,
        dropout_probability=0.0,
        use_batch_norm=False,
    )

    base_dist = architecture.get_normal_dist(
        feature_dim=x_dim,
    )

    transform = architecture.get_sigmoids_transform(
        feature_dim=x_dim,
        context_embedding_dim=context_embedding_dim,
        n_layers=4,
        hidden_dim=256,
        svd_kwargs={},
        sigmoids_kwargs={
            "n_sigmoids": 16,
            "num_blocks": 3,
            "dropout_probability": 0.0,
        },
    )

    # transform = architecture.get_lipschitz_transform(
    #     feature_dim=x_dim,
    #     context_embedding_dim=context_embedding_dim,
    #     n_layers=4,
    #     hidden_dim=256,
    #     # hidden_dim=512,
    # )

    model = LikelihoodFlow(
        params,
        msfm_conf,
        embedding_net=embedding_net,
        base_dist=base_dist,
        transform=transform,
        model_dir=args.flow_dir,
        load_existing=True,
        device="cpu",
        torch_seed=args.torch_seed,
    )

    # get the correct random split of the validation data
    model._prepare_data(
        x=grid_preds,
        theta=grid_cosmos,
        batch_size=10000,
        vali_split=0.1,
    )

    x_vali = model.vali_dset.dataset.tensors[0][model.vali_dset.indices]
    theta_vali = model.vali_dset.dataset.tensors[1][model.vali_dset.indices]

    return model, x_vali, theta_vali


def merge(indices, args):
    args = setup(args)
    n_cosmos_test = len(indices)

    out_file = os.path.join(args.flow_dir, f"mcmc_samples.h5")
    with h5py.File(out_file, "w") as f_merged:
        for index in LOGGER.progressbar(indices, desc="merging files", at_level="info"):
            in_file = os.path.join(args.flow_dir, f"mcmc_samples_{index}.h5")
            with h5py.File(in_file, "r") as f_in:
                x_true = f_in["x_true"][:]
                theta_true = f_in["theta_true"][:]
                theta_samples = f_in["theta_samples"][:]

            if index == indices[0]:
                f_merged.create_dataset("x_true", shape=(n_cosmos_test, x_true.shape[0]), dtype=np.float32)
                f_merged.create_dataset("theta_true", shape=(n_cosmos_test, theta_true.shape[0]), dtype=np.float32)
                f_merged.create_dataset(
                    "theta_samples",
                    shape=(n_cosmos_test, theta_samples.shape[0], theta_samples.shape[1]),
                    dtype=np.float32,
                )
            f_merged["x_true"][index] = x_true
            f_merged["theta_true"][index] = theta_true
            f_merged["theta_samples"][index] = theta_samples
    LOGGER.info(f"Merged all files into {out_file}")

    # only remove the files after the above loop has terminated successfully
    for index in indices:
        in_file = os.path.join(args.flow_dir, f"mcmc_samples_{index}.h5")
        if os.path.exists(in_file):
            os.remove(in_file)
    LOGGER.info(f"Removed temporary files")

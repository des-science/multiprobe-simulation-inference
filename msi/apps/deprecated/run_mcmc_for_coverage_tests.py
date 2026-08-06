# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Sample the MCMC posterior for N random observations from the CosmoGrid for posterior-level coverage testing.

Two execution paths are supported:

1. Legacy esub job-array workflow on Perlmutter CPU nodes (one chain per task). See the examples below.

2. Direct-SLURM GPU workflow on Clariden (GH200), where the flow runs on cuda and the in-house batched
   ensemble sampler (msi.utils.torch_ensemble) evaluates many observations in a single forward pass. The
   script is invoked directly with python (no esub): each process samples one contiguous shard of the
   n_sims observations, then a final --merge process combines the per-index files. See
   y3-deep-lss/submissions/clariden/mcmc_coverage.sh (analogously placed under msi/submissions). Example:

   python run_mcmc_for_coverage_tests.py --preds_file=... --flow_dir=... \
       --n_sims=1000 --device=cuda --obs_batch=64 --n_shards=4 --shard_id=0
   # ... then once all shards finish:
   python run_mcmc_for_coverage_tests.py --preds_file=... --flow_dir=... --n_sims=1000 --merge

For a flow trained on combined maps+Cls summaries (run_inference.py --out_dir_2, see
y3-deep-lss/submissions/clariden/combined_inference.sh), also pass --preds_file_2 pointing at the
second model's preds_*.h5, so the held-out validation split can be faithfully reproduced.

example usage:

esub run_mcmc_for_coverage_tests.py \
    --preds_file=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/preds_400000.h5 \
    --flow_dir=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/400000_steps_likelihood_sigmoid_test_v4/likelihood_flow \
    --n_sims=1000 --n_jobs=1000 --device=cpu \
    --mode=jobarray --function=all --keep_submit_files \
    --jobname=mcmc_test --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


esub run_mcmc_for_coverage_tests.py \
    --preds_file=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/preds_400000.h5 \
    --flow_dir=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/400000_steps_likelihood_sigmoid_test_v4/likelihood_flow \
    --n_sims=1000 --n_jobs=1000 --device=cpu \
    --mode=jobarray --function=rerun_missing --keep_submit_files \
    --jobname=mcmc_test --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

"""

import numpy as np
import torch, os, argparse, warnings, h5py, time

from msfm.utils import logger

from msi.utils import flow as flow_utils
from msi.flow_conductor.likelihood_flow import LikelihoodFlow

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def get_tasks(args):
    args = setup(args)

    return list(range(args.n_sims))


def resources(args):
    args = setup(args)

    if args.cluster == "perlmutter":
        # because of hyperthreading, there's a total of 256 threads per node
        resources = {
            "main_time_per_index": 0.4,
            "main_n_cores": 2,
            "main_memory": 1952,
            "main_scratch": 0,
            "merge_time": 0.5,
            "merge_n_cores": 8,
            "merge_memory": 1952,
            "merge_scratch": 0,
        }
    elif args.cluster == "euler":
        resources = {"main_time": 4, "main_memory": 4096, "main_n_cores": 8, "merge_memory": 4096, "merge_n_cores": 16}
    else:
        # Clariden does not use esub; jobs are launched via the direct-SLURM __main__ entry point
        resources = {}

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
        "--preds_file_2",
        type=str,
        default=None,
        help="optional second model's predictions file; its summary is concatenated feature-wise "
        "with the primary one's, e.g. to reproduce a flow trained on combined maps+Cls summaries "
        "via run_inference.py's --out_dir_2. Must match what the flow at --flow_dir was trained on.",
    )
    parser.add_argument(
        "--flow_dir",
        type=str,
        required=True,
        help="directory containing the flow network",
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        default=1000,
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
        choices=("perlmutter", "euler", "clariden"),
        help="the cluster to execute on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="torch device for the flow and the batched sampler (e.g. 'cuda' on Clariden, 'cpu' "
        "for the legacy esub/Perlmutter CPU workflow)",
    )
    parser.add_argument(
        "--obs_batch",
        type=int,
        default=64,
        help="number of observations sampled together in a single batched forward pass; size to fit "
        "GPU memory (forward batch is obs_batch * n_walkers points)",
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=1,
        help="number of shards the n_sims observations are split into for direct SLURM execution",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="which contiguous shard (0..n_shards-1) this process handles for direct SLURM execution",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="direct-execution merge mode: combine all per-index files into mcmc_samples.h5 and exit",
    )
    parser.add_argument(
        "--max_sleep",
        type=int,
        default=60,
        help="set the maximal amount of time to sleep before copying to avoid clashes",
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

    n_walkers = 1024
    n_burnin_steps = 1_000
    n_steps = 1_000
    n_samples_out = 10_000

    if args.debug:
        args.max_sleep = 0
        n_burnin_steps = 10
        n_steps = 10
        LOGGER.warning("!!! debug mode !!!")
    # the GPU-batched sampler returns n_steps * n_walkers samples per observation
    n_samples_out = min(n_samples_out, n_steps * n_walkers)
    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    model, x_true_all, theta_true_all = _set_up_flow(args)

    # deterministically subselect cosmologies
    n_cosmos = x_true_all.shape[0]
    x_true_sub = x_true_all[:: n_cosmos // args.n_sims]
    theta_true_sub = theta_true_all[:: n_cosmos // args.n_sims]

    # process the (possibly sharded) observation indices in batches that share one forward pass
    indices = list(indices)
    for chunk_start in range(0, len(indices), args.obs_batch):
        chunk = indices[chunk_start : chunk_start + args.obs_batch]

        x_true_batch = x_true_sub[chunk]  # (n_chunk, n_features)
        theta_true_batch = theta_true_sub[chunk]  # (n_chunk, n_params)

        # GPU-batched sampling for the whole chunk at once
        chain, _ = model.sample_posterior_batched(
            x_true_batch,
            n_walkers=n_walkers,
            n_burnin_steps=n_burnin_steps,
            n_steps=n_steps,
            device=args.device,
        )  # chain: (n_chunk, n_steps * n_walkers, n_params)

        # raw flow log-likelihood of the true cosmology for each observation in the chunk (batched)
        log_prob_true_batch = model.log_likelihood(x_true_batch, theta_true_batch, return_numpy=True)

        for j, index in enumerate(chunk):
            x_true = np.asarray(x_true_batch[j].cpu() if hasattr(x_true_batch[j], "cpu") else x_true_batch[j])
            theta_true = np.asarray(
                theta_true_batch[j].cpu() if hasattr(theta_true_batch[j], "cpu") else theta_true_batch[j]
            )
            log_prob_true = np.atleast_1d(log_prob_true_batch[j])

            theta_sample = chain[j]
            # too many samples make the test slow and are not needed
            theta_sample = theta_sample[np.random.choice(theta_sample.shape[0], n_samples_out, replace=False)]

            # raw flow log-likelihood of the posterior samples (same observation repeated)
            x_repeated = np.repeat(x_true[None, :], theta_sample.shape[0], axis=0)
            log_prob_sample = model.log_likelihood(x_repeated, theta_sample, return_numpy=True)

            out_file = os.path.join(args.flow_dir, f"mcmc_samples_{index}.h5")
            with h5py.File(out_file, "w") as f:
                f.create_dataset("x_true", data=x_true)
                f.create_dataset("theta_true", data=theta_true)
                f.create_dataset("log_prob_true", data=log_prob_true)

                f.create_dataset("theta_sample", data=theta_sample)
                f.create_dataset("log_prob_sample", data=log_prob_sample)

            yield index


def _set_up_flow(args):
    """Restore the trained flow and reproduce its held-out validation split for coverage testing.

    LikelihoodFlow._prepare_data's train/validation split is made deterministic and grouped by
    signal realization (group_ids=i_signal): the unique signal ids are sorted and partitioned into
    a train and a validation fraction, so no signal realization -- regardless of its noise
    realizations -- appears in both sets. The split therefore depends only on the content and row
    order of (x, theta, i_signal), not on any random seed, so grid_preds/grid_cosmos/i_signal must
    be (re)built exactly as run_inference.py built them at training time -- hence reusing
    flow_utils.load_grid_summaries. This guarantees the mock observations drawn from the
    reconstructed validation set were seen neither by the compression network nor by the flow.
    """
    grid_preds, grid_cosmos, _, _, i_signal, _ = flow_utils.load_grid_summaries(args.preds_file, args.preds_file_2)

    model = LikelihoodFlow.from_checkpoint(model_dir=args.flow_dir, device=args.device)

    # get the correct signal-grouped split of the validation data
    model._prepare_data(
        x=grid_preds,
        theta=grid_cosmos,
        batch_size=10000,
        vali_split=0.1,
        group_ids=i_signal,
    )

    x_vali = model.vali_dset.dataset.tensors[0][model.vali_dset.indices]
    theta_vali = model.vali_dset.dataset.tensors[1][model.vali_dset.indices]
    LOGGER.info(f"Reconstructed {len(x_vali)} held-out validation examples for coverage testing")

    return model, x_vali, theta_vali


def merge(indices, args):
    args = setup(args)
    n_sims = args.n_sims

    out_file = os.path.join(args.flow_dir, f"mcmc_samples.h5")
    with h5py.File(out_file, "w") as f_merged:
        for index in LOGGER.progressbar(indices, desc="merging files", at_level="info"):
            try:
                in_file = os.path.join(args.flow_dir, f"mcmc_samples_{index}.h5")
                with h5py.File(in_file, "r") as f_in:
                    x_true = f_in["x_true"][:]
                    theta_true = f_in["theta_true"][:]
                    log_prob_true = f_in["log_prob_true"][:]

                    theta_sample = f_in["theta_sample"][:]
                    log_prob_sample = f_in["log_prob_sample"][:]

                if index == indices[0]:
                    f_merged.create_dataset("x_true", shape=(n_sims, x_true.shape[0]), dtype=np.float32)
                    f_merged.create_dataset("theta_true", shape=(n_sims, theta_true.shape[0]), dtype=np.float32)
                    f_merged.create_dataset("log_prob_true", shape=(n_sims), dtype=np.float32)

                    # shape as expected by TARP package
                    f_merged.create_dataset(
                        "theta_sample",
                        shape=(theta_sample.shape[0], n_sims, theta_sample.shape[1]),
                        dtype=np.float32,
                    )
                    f_merged.create_dataset("log_prob_sample", shape=(theta_sample.shape[0], n_sims), dtype=np.float32)

                f_merged["x_true"][index] = x_true
                f_merged["theta_true"][index] = theta_true
                f_merged["log_prob_true"][index] = log_prob_true

                # breakpoint()

                f_merged["theta_sample"][:, index] = theta_sample
                f_merged["log_prob_sample"][:, index] = log_prob_sample
            except (FileNotFoundError, TypeError):
                pass
    LOGGER.info(f"Merged all files into {out_file}")

    # only remove the files after the above loop has terminated successfully
    for index in indices:
        in_file = os.path.join(args.flow_dir, f"mcmc_samples_{index}.h5")
        if os.path.exists(in_file):
            os.remove(in_file)
    LOGGER.info(f"Removed temporary files")


def _run_direct(argv):
    """Direct-SLURM entry point for Clariden (esub is not available there).

    Each process either (a) handles one contiguous shard of the n_sims observations on a single GPU, or
    (b) with --merge, combines all per-index files into the final mcmc_samples.h5. The per-index file
    layout is identical to the esub workflow, so merge() is shared between the two paths.
    """
    parsed = setup(argv)

    all_indices = list(range(parsed.n_sims))

    if parsed.merge:
        merge(all_indices, argv)
        return

    shard = [int(i) for i in np.array_split(all_indices, parsed.n_shards)[parsed.shard_id]]
    LOGGER.info(
        f"Shard {parsed.shard_id}/{parsed.n_shards}: sampling {len(shard)} observations "
        f"(indices {shard[0]}..{shard[-1]}) on device {parsed.device}"
    )
    # main is a generator (esub contract); exhaust it to run the work
    for _ in main(shard, argv):
        pass


if __name__ == "__main__":
    import sys

    _run_direct(sys.argv[1:])

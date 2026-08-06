"""
DEPRECATED -- `from msi.apps import run_mcmc_for_coverage_tests` no longer resolves: that module
moved to msi/apps/deprecated/ when the coverage stage was folded into run_inference.py
--sample_posterior (msi/utils/coverage.py). Kept for the interactive-session recipe below.

Minimal interactive batched-sampling test. Start a GPU session, activate the env, then run this file
(or paste it). Tweak OBS_BATCH and re-run the last block to feel out memory / speed.

    srun -A a0158 --partition=normal -N1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G --uenv=pytorch/v2.9.1:v2 --view=default --pty bash
    source ~/dlss/torch_env/bin/activate
    cd /users/athomsen/dlss/repos/multiprobe-simulation-inference
    python submissions/clariden/tests/interactive.py
"""

import time
import torch
from argparse import Namespace

from msi.apps import run_mcmc_for_coverage_tests as cov

# --- variables ----------------------------------------------------------------------------------------
PREDS_FILE = "/users/athomsen/scratch/deep_lss/runs/v16/rot_in_place/cls/lensing/v26/preds_1000000.h5"
FLOW_DIR = "/users/athomsen/scratch/deep_lss/runs/v16/rot_in_place/cls/lensing/v26/likelihood_flow_1000000"
N_SIMS = 1000
OBS_BATCH = 1000
# ------------------------------------------------------------------------------------------------------

# load the flow on the GPU and reconstruct the held-out validation observations (once)
args = Namespace(preds_file=PREDS_FILE, preds_file_2=None, flow_dir=FLOW_DIR, device="cuda")
model, x_true_all, _ = cov._set_up_flow(args)
x_true_sub = x_true_all[:: x_true_all.shape[0] // N_SIMS]

# sample one batch of OBS_BATCH observations and time it
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
chain, _ = model.sample_posterior_batched(x_true_sub[:OBS_BATCH], n_burnin_steps=1000, n_steps=1000)
torch.cuda.synchronize()

print(
    f"OBS_BATCH={OBS_BATCH}: {time.perf_counter() - t0:.1f}s, "
    f"peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, chain shape {chain.shape}"
)

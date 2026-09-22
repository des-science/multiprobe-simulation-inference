#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=tension
#SBATCH --output=/users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/slurm/slurm-%j.out

# Posterior-tension analysis in two sequential stages with different environments:
#   1. torch_env: train emulators + residual flow, build the parameter-difference chains (stage A)
#   2. tensorflow: run tensiometer's flow estimator to assign the numerical tension value (stage B)
# Both stages loop internally over the run combinations and mock observations defined in the configs.

set -euo pipefail
ulimit -c 0

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MSI="$REPOS/multiprobe-simulation-inference"

# --- Overridable defaults ------------------------------------------------------------------------
#
# The three configs are derived from (VERSION, SUBVERSION, RUNS_NAME) so that switching datasets
# needs no edit to this file. Both layouts already follow the convention:
#   configs/runs/<VERSION>/<SUBVERSION>/<RUNS_NAME>.yaml   (msi, run definitions)
#   configs/<VERSION>/<SUBVERSION>.yaml                    (msfm, priors for the residual MCMC)
# Older v17 combinations stay reachable, e.g.
#   VERSION=v17 SUBVERSION=baseline RUNS_NAME=t1_v3 sbatch tension.sh
# Any of the three full paths can still be overridden directly.
VERSION="${VERSION:-v18}"
SUBVERSION="${SUBVERSION:-default}"
# Default is the production set. It was `v1` until 2026-09-08, when that file was renamed
# bench_v7.yaml -- see its header; `RUNS_NAME=bench_v7` still reaches it.
RUNS_NAME="${RUNS_NAME:-prod}"

RUNS_CONFIG="${RUNS_CONFIG:-$MSI/configs/runs/$VERSION/$SUBVERSION/$RUNS_NAME.yaml}"
TENSION_CONFIG="${TENSION_CONFIG:-$MSI/configs/tension/tension.yaml}"
MSFM_CONFIG="${MSFM_CONFIG:-$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml}"

# Both stages loop over every run pair the runs config implies, so wall clock scales with the
# number of combinations: ~2.3 min per pair in stage A, ~2.8 min per pair in stage B (4 flow fits).
# The header was 1 h until 2026-09-22 and that was never enough for a production set: the two v17
# t2_v3 jobs (12 pairs) both died there with stage B half done, and job 3328338 -- v18/default prod,
# the 21 combinations this file now defaults to -- took 1:58:29 (stage A 52:56, stage B 1:04:54).
# 3 h carries that with margin. A SMALLER set can be given less:
#   sbatch --time=01:00:00 tension.sh

LOG_DIR="$MSI/submissions/clariden/slurm"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLURM_JOB_ID}"

# --- stage A: difference chains (PyTorch / torch_env) ---------------------------------------------
# --cpu-bind=none is required for a step that sub-allocates inside an --exclusive batch step:
# without it the step asks to bind more CPUs than its own 72-CPU allocation holds and dies in
# seconds with "CPU binding outside of job step allocation", leaving a 0-byte stage log. Jobs
# 3478116/3478117 died exactly there on 2026-09-22; maps/rerun/inference.sh carries the same flag.
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
    --cpu-bind=none --uenv=pytorch/v2.9.1:v2 --view=default \
    --output="${LOG}_chains.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $MSI/msi/apps/run_tension_chains.py \
        --runs_config=\"$RUNS_CONFIG\" \
        --tension_config=\"$TENSION_CONFIG\" \
        --msfm_config=\"$MSFM_CONFIG\" \
        --device=cuda"


# --- stage B: numerical tension values (TensorFlow / tensorflow env) ------------------------------
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
    --environment=tensorflow --cpu-bind=none --gpu-bind=none \
    --output="${LOG}_values.log" \
    python $MSI/msi/apps/run_tension_values.py \
        --runs_config="$RUNS_CONFIG" \
        --tension_config="$TENSION_CONFIG"


# --- what actually landed -------------------------------------------------------------------------
# Neither stage fails on a missing input: stage B logs "Missing <kind> chain, skipping" and stage A
# catches plot errors, so exit 0 does NOT mean the analysis is complete. Count the outputs here.
# This is the check that was absent when two v17 pairs ended up with difference chains and no
# significance, and it costs nothing next to a 2 h job.
echo "=== tension summary (job ${SLURM_JOB_ID}) ==="
printf 'stage A  difference chains saved : %s\n' "$(grep -c 'Saved chain to' "${LOG}_chains.log" || true)"
printf 'stage A  plot failures           : %s\n' "$(grep -c 'plotting failed' "${LOG}_chains.log" || true)"
printf 'stage B  significances saved     : %s\n' "$(grep -c 'Saved tension results to' "${LOG}_values.log" || true)"
printf 'stage B  chains missing          : %s\n' "$(grep -c 'chain, skipping' "${LOG}_values.log" || true)"
grep -n 'chain, skipping\|plotting failed\|Traceback' "${LOG}_chains.log" "${LOG}_values.log" \
    || echo "no skips, plot failures or tracebacks"

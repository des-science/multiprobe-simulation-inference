#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
# Cold first run trains ~10 PPC flows AND runs per-mock calibration over all auto runs; that fits in
# ~6h. Re-runs are checkpoint-aware (flows recovered from disk) and much faster -- lower if needed.
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=ppc
#SBATCH --output=/users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/slurm/slurm-%j.out

# Posterior predictive checks in a single environment (torch_env): PosteriorPredictiveChecks trains
# a torch/enflows LikelihoodFlow and loads (backend-agnostic) saved inference chains, so no separate
# TensorFlow stage is needed. The app loops internally over the runs / comparisons / observations
# defined in the configs. Walltime scales with the number of runs (auto) and pairs (cross); adjust.

set -euo pipefail
ulimit -c 0

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MSI="$REPOS/multiprobe-simulation-inference"

# --- Overridable defaults ------------------------------------------------------------------------
#
# Same (VERSION, SUBVERSION, RUNS_NAME) interface as tension.sh, so switching datasets needs no edit
# to this file. Both layouts already follow the convention:
#   configs/runs/<VERSION>/<SUBVERSION>/<RUNS_NAME>.yaml   (msi, run definitions)
#   configs/<VERSION>/<SUBVERSION>.yaml                    (msfm, parameter definitions / priors)
# The msfm config must match the dataset the runs were trained on: v17 dropped bta, so pointing a
# v18 runs config at configs/v17/baseline.yaml fails the PPC's parameter-dimension asserts.
# Older combinations stay reachable, e.g.
#   VERSION=v17 SUBVERSION=baseline RUNS_NAME=t2_v3 sbatch ppc.sh
VERSION="${VERSION:-v18}"
SUBVERSION="${SUBVERSION:-default}"
# Default is the production set. It was `v1` until 2026-09-08, when that file was renamed
# bench_v7.yaml -- see its header; `RUNS_NAME=bench_v7` still reaches it.
RUNS_NAME="${RUNS_NAME:-prod}"

RUNS_CONFIG="${RUNS_CONFIG:-$MSI/configs/runs/$VERSION/$SUBVERSION/$RUNS_NAME.yaml}"
# ppc_quick.yaml trims everything but the Cls-space PPD; short enough for --partition=debug.
PPC_CONFIG="${PPC_CONFIG:-$MSI/configs/ppc/ppc.yaml}"
MSFM_CONFIG="${MSFM_CONFIG:-$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml}"

# Flow training is checkpoint-aware by default: each PPC flow is recovered from disk when one exists.
# Force a retrain from scratch (e.g. after an architecture change) with
#   RETRAIN_FLOWS=--retrain_flows sbatch ppc.sh
RETRAIN_FLOWS="${RETRAIN_FLOWS:-}"

# Keep stage logs in the submissions tree (do not pollute the scratch runs/ tree), like tension.sh.
LOG_DIR="$MSI/submissions/clariden/slurm"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLURM_JOB_ID}"

# --- posterior predictive checks (PyTorch / torch_env) -------------------------------------------
# --cpu-bind=none is required for a step that sub-allocates inside an --exclusive batch step:
# without it the step asks to bind more CPUs than its own 72-CPU allocation holds and dies in
# seconds with "CPU binding outside of job step allocation", leaving a 0-byte stage log. Jobs
# 3478116/3478117 died exactly there on 2026-09-22; maps/rerun/inference.sh carries the same flag.
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
    --cpu-bind=none --uenv=pytorch/v2.9.1:v2 --view=default \
    --output="${LOG}_ppc.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $MSI/msi/apps/run_ppc.py \
        --runs_config=\"$RUNS_CONFIG\" \
        --ppc_config=\"$PPC_CONFIG\" \
        --msfm_config=\"$MSFM_CONFIG\" \
        $RETRAIN_FLOWS \
        --device=cuda"


# --- what actually landed -------------------------------------------------------------------------
# run_ppc skips rather than fails when an input is missing -- the calibration returns None with a
# warning when mcmc_samples.h5 (or its real_idx) is absent, and the Cls-space PPD is disabled with a
# warning when its cache cannot be resolved. Exit 0 therefore does NOT mean every check ran, so
# count what landed instead of trusting the job state.
echo "=== ppc summary (job ${SLURM_JOB_ID}) ==="
printf 'auto runs processed     : %s\n' "$(grep -c '=== auto:' "${LOG}_ppc.log" || true)"
printf 'cross pairs processed   : %s\n' "$(grep -c '=== cross:' "${LOG}_ppc.log" || true)"
printf 'calibrations saved      : %s\n' "$(grep -c 'Saved calibration summary to' "${LOG}_ppc.log" || true)"
printf 'calibration nulls saved : %s\n' "$(grep -c 'Saved calibration null arrays to' "${LOG}_ppc.log" || true)"
printf 'warnings                : %s\n' "$(grep -c ' WAR ' "${LOG}_ppc.log" || true)"
grep -n 'skipping p-value calibration\|disabled:\|Traceback' "${LOG}_ppc.log" \
    || echo "no skipped calibrations, disabled checks or tracebacks"

#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=ppc
#SBATCH --output=/users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/slurm/slurm-%j.out

# Posterior predictive checks in a single environment (torch_env): PosteriorPredictiveChecks trains
# a torch/enflows LikelihoodFlow and loads (backend-agnostic) saved inference chains, so no separate
# TensorFlow stage is needed. The app loops internally over the runs / comparisons / observations
# defined in the configs. Walltime scales with the number of runs (auto) and pairs (cross); adjust.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MSI="$REPOS/multiprobe-simulation-inference"

# RUNS_CONFIG="$MSI/configs/runs/v8_v33.yaml"
RUNS_CONFIG="$MSI/configs/runs/v8_v33_extended.yaml"
PPC_CONFIG="$MSI/configs/ppc.yaml"
MSFM_CONFIG="$REPOS/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"

# Keep stage logs in the submissions tree (do not pollute the scratch runs/ tree), like tension.sh.
LOG_DIR="$MSI/submissions/clariden/slurm"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLURM_JOB_ID}"

# --- posterior predictive checks (PyTorch / torch_env) -------------------------------------------
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output="${LOG}_ppc.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $MSI/msi/apps/run_ppc.py \
        --runs_config=\"$RUNS_CONFIG\" \
        --ppc_config=\"$PPC_CONFIG\" \
        --msfm_config=\"$MSFM_CONFIG\" \
        --device=cuda"

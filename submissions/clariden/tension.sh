#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=tension
#SBATCH --output=/users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/slurm/slurm-%j.out

# Posterior-tension analysis in two sequential stages with different environments:
#   1. torch_env: train emulators + residual flow, build the parameter-difference chains (stage A)
#   2. tensorflow: run tensiometer's flow estimator to assign the numerical tension value (stage B)
# Both stages loop internally over the run combinations and mock observations defined in the configs.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MSI="$REPOS/multiprobe-simulation-inference"

RUNS_CONFIG="$MSI/configs/runs/v8_v33.yaml"
TENSION_CONFIG="$MSI/configs/tension.yaml"
MSFM_CONFIG="$REPOS/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"

LOG_DIR="$MSI/submissions/clariden/slurm"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLURM_JOB_ID}"

# --- stage A: difference chains (PyTorch / torch_env) ---------------------------------------------
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output="${LOG}_chains.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $MSI/msi/apps/run_tension_chains.py \
        --runs_config=\"$RUNS_CONFIG\" \
        --tension_config=\"$TENSION_CONFIG\" \
        --msfm_config=\"$MSFM_CONFIG\" \
        --device=cuda"

sleep 30

# --- stage B: numerical tension values (TensorFlow / tensorflow env) ------------------------------
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
    --environment=tensorflow --gpu-bind=none \
    --output="${LOG}_values.log" \
    python $MSI/msi/apps/run_tension_values.py \
        --runs_config="$RUNS_CONFIG" \
        --tension_config="$TENSION_CONFIG"

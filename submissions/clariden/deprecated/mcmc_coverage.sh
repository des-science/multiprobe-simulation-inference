#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=mcmc_coverage
#SBATCH --output=/users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/slurm/slurm-%j.out

# DEPRECATED -- does not run. It drives msi/apps/run_mcmc_for_coverage_tests.py, which has moved to
# msi/apps/deprecated/, and its paths are hardcoded to the retired v16/rot_in_place/cls/lensing/v26 run.
# Superseded by the coverage stage inside run_inference.py --sample_posterior (msi/utils/coverage.py),
# which samples the held-out mocks and writes mcmc_samples.h5 as part of the normal inference job.
# Kept for reference only.

# Posterior-level coverage test on Clariden (GH200) via the direct-SLURM GPU workflow:
# the flow runs on cuda and the in-house batched ensemble sampler (msi.utils.torch_ensemble) evaluates
# many observations per forward pass. The n_sims observations are split into N_SHARDS contiguous shards,
# one per GPU (4 per node), all launched in the background and joined with `wait`. A final --merge step
# then combines the per-index files into mcmc_samples.h5. This replaces the Perlmutter esub job array.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
APP="$REPOS/multiprobe-simulation-inference/msi/apps/run_mcmc_for_coverage_tests.py"

# --- run to test: point these at the trained flow's outputs -------------------------------------------
PREDS_FILE="/users/athomsen/scratch/deep_lss/runs/v16/rot_in_place/cls/lensing/v26/preds_1000000.h5"
FLOW_DIR="/users/athomsen/scratch/deep_lss/runs/v16/rot_in_place/cls/lensing/v26/likelihood_flow_1000000"
# for a combined maps+Cls flow also set: PREDS_FILE_2="/.../preds_*.h5" and add --preds_file_2 below
# ------------------------------------------------------------------------------------------------------

N_SIMS=1000
N_SHARDS=4         # one shard per GPU on a single node
OBS_BATCH=64       # observations per batched forward pass; lower if you hit GPU OOM

LOG_DIR="$FLOW_DIR/coverage_logs"
mkdir -p "$LOG_DIR"

echo "Starting coverage test: $N_SIMS observations across $N_SHARDS GPUs (obs_batch=$OBS_BATCH)"

for SHARD_ID in $(seq 0 $((N_SHARDS - 1))); do
    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
        --uenv=pytorch/v2.9.1:v2 --view=default \
        --output="$LOG_DIR/${SLURM_JOB_ID}_shard${SHARD_ID}.log" \
        bash -c "source ~/dlss/torch_env/bin/activate && python $APP \
            --preds_file=\"$PREDS_FILE\" \
            --flow_dir=\"$FLOW_DIR\" \
            --n_sims=$N_SIMS \
            --cluster=clariden \
            --device=cuda \
            --obs_batch=$OBS_BATCH \
            --n_shards=$N_SHARDS \
            --shard_id=$SHARD_ID" &
done
wait
echo "All shards finished; merging per-index files..."

# merge only reads the per-index h5 files (no flow / no GPU needed)
srun -N1 --ntasks-per-node=1 --cpus-per-task=72 --mem=110G \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output="$LOG_DIR/${SLURM_JOB_ID}_merge.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $APP \
        --preds_file=\"$PREDS_FILE\" \
        --flow_dir=\"$FLOW_DIR\" \
        --n_sims=$N_SIMS \
        --device=cpu \
        --merge"

echo "Coverage test complete: $FLOW_DIR/mcmc_samples.h5"

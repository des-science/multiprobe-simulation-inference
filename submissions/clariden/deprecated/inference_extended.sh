#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=ext_inference
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v16/rot_in_place/cls/lensing/lmax_1024/logs/slurm-%j.out

# DEPRECATED -- folded into y3-deep-lss/submissions/clariden/maps/rerun/inference.sh, which now takes
# EXTEND_PARAMS and LOAD_FLOW as environment variables instead of existing as a separate copy:
#   EXTEND_PARAMS=--extend_params VERSION=v16 SUBVERSION=rot_in_place \
#     OUTPUT=$MYSCRATCH/deep_lss/runs/v16/rot_in_place/cls/lensing MODEL_DIR=lmax_1024 \
#     sbatch --uenv-passthrough=ignore inference.sh
# Kept for reference only; its paths are hardcoded to the retired v16 lensing run.

# Extended-conditioning-vector inference (constraining-power decisive test): retrain the likelihood
# flow on the EXISTING network summaries with the conditioning vector extended by the implicitly
# marginalized grid parameters (ns, Ob, H0, bary_Mc, bary_nu; looked up per grid row via i_sobol --
# no summary recomputation), then sample the usual chains plus the reference-prior (Gower-Street
# family: near-delta ns/Obh2/H0, baryons fixed) DES variants for the apples-to-apples comparison
# with J24/G24/W26. Everything saves under ext_ensemble_flow_<steps>/, baseline flow untouched.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

OUT_DIR="$MYSCRATCH/deep_lss/runs/v16/rot_in_place/cls/lensing"
MODEL_NAME="lmax_1024"
FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

# Rerun the sampling stages on an already-trained extended flow (e.g. after a plotting fix) with
#   sbatch --export=ALL,LOAD_FLOW=--load_flow inference_extended.sh
LOAD_FLOW="${LOAD_FLOW:-}"

LOG="$OUT_DIR/$MODEL_NAME/logs/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$LOG")"

srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output="${LOG}_ext_inference.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
        --out_dir=\"$OUT_DIR\" \
        --model_name=\"$MODEL_NAME\" \
        --flow_config=\"$FLOW_CONFIG\" \
        --n_flows=4 \
        --extend_params \
        $LOAD_FLOW \
        --sample_posterior \
        --include_grid \
        --include_des \
        --include_mocks"

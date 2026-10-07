# Sourced by every script in this tree: roots, dataset defaults, the torch step.
# Source it by absolute path; sbatch runs a spool copy of the calling script.

ulimit -c 0  # a crashing task would otherwise fill the /users quota with a core dump

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"
STORE="/capstor/store/cscs/swissai/a0158/athomsen"
MSI="$REPOS/multiprobe-simulation-inference"
DEEP_LSS="$REPOS/y3-deep-lss"

# Every step here is a 1-GPU step, a quarter node. Inside another job's allocation (prod/maps.sh)
# the environment stays as that job set it: srun refuses a SLURM_CPUS_PER_TASK that contradicts
# its SLURM_TRES_PER_TASK, and the steps pass --cpus-per-task themselves.
[ -z "${SLURM_TRES_PER_TASK:-}" ] && export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=72 TF_NUM_INTRAOP_THREADS=72

VERSION="${VERSION:-v18}"
SUBVERSION="${SUBVERSION:-default}"
MSFM_CONFIG="${MSFM_CONFIG:-$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml}"

# DRYRUN=1 prints every srun instead of running it (works on the login node).
DRYRUN="${DRYRUN:-0}"
LOG_DIR="$MSI/submissions/clariden/slurm"
[ "$DRYRUN" = "1" ] || mkdir -p "$LOG_DIR"
SRUN="srun"
[ "$DRYRUN" = "1" ] && SRUN="echo srun"
SLURM_JOB_ID="${SLURM_JOB_ID:-dryrun}"

# One GPU inside an --exclusive node. --cpu-bind=none is required for a sub-allocating step, which
# otherwise dies at once with "CPU binding outside of job step allocation".
GPU_STEP=(-N1 -n1 --exclusive --gpus-per-task=1 --cpus-per-task=72 --cpu-bind=none)
TORCH_ENV=(--uenv=pytorch/v2.9.1:v2 --view=default)
TORCH_PY="$HOME/dlss/torch_env/bin/python"  # the venv's own interpreter, no activation needed

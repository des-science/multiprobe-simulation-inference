# Sourced by every script in this tree: roots, the torch step, and use_dataset for the scripts that read a dataset.
# Source it by absolute path; sbatch runs a spool copy of the calling script.

ulimit -c 0  # a crashing task would otherwise fill the /users quota with a core dump

REPOS="/users/athomsen/dlss/repos"
MSI="$REPOS/multiprobe-simulation-inference"
MSFM="$REPOS/multiprobe-simulation-forward-model"
DEEP_LSS="$REPOS/y3-deep-lss"

# Every step here is a 1-GPU step, a quarter node. Inside another job's allocation (prod/maps.sh)
# the environment stays as that job set it: srun refuses a SLURM_CPUS_PER_TASK that contradicts
# its SLURM_TRES_PER_TASK, and the steps pass --cpus-per-task themselves.
[ -z "${SLURM_TRES_PER_TASK:-}" ] && export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=72 TF_NUM_INTRAOP_THREADS=72

SUBVERSION="${SUBVERSION:-default}"

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

# use_dataset [runs] [data]: the dataset of VERSION, msfm's configs/<VERSION>/<SUBVERSION>.yaml (MSFM_CONFIG), e.g.
# VERSION=desy6/v1 or VERSION=desy3/v18. VERSION has no default, so a run never falls back to another survey's
# runs. The paths asked for come from the locations.yaml next to that config (msfm.utils.locations): runs sets
# RUNS_ROOT, data sets DATA_DIR. An environment variable overrides each, and then nothing is looked up.
use_dataset() {
    local datasets key file value need=()
    datasets=$(ls "$MSFM"/configs/*/*/locations.yaml 2>/dev/null | sed -E "s#^$MSFM/configs/(.*)/locations.yaml#\1#" | xargs)
    if [ -z "${VERSION:-}" ]; then
        echo "Set VERSION to a dataset of msfm's configs/: $datasets" >&2
        exit 1
    fi
    MSFM_CONFIG="${MSFM_CONFIG:-$MSFM/configs/$VERSION/$SUBVERSION.yaml}"
    if [ ! -f "$MSFM_CONFIG" ]; then
        echo "No msfm config $MSFM_CONFIG; the datasets are: $datasets" >&2
        exit 1
    fi
    for key in "$@"; do
        case $key in
            runs) [ -n "${RUNS_ROOT:-}" ] || need+=(runs_dir) ;;
            data) [ -n "${DATA_DIR:-}" ] || need+=(data_dir) ;;
            *) echo "use_dataset: unknown path $key" >&2; exit 1 ;;
        esac
    done
    [ ${#need[@]} -gt 0 ] || return 0

    file=$(mktemp)
    if [ "$DRYRUN" = "1" ]; then
        uenv run --view=default pytorch/v2.9.1:v2 -- "$TORCH_PY" -m msfm.utils.locations "$MSFM_CONFIG" > "$file"
    else
        srun "${GPU_STEP[@]}" "${TORCH_ENV[@]}" --output="$file" "$TORCH_PY" -m msfm.utils.locations "$MSFM_CONFIG"
    fi || { cat "$file" >&2; echo "Cannot resolve the locations of $MSFM_CONFIG." >&2; exit 1; }
    RUNS_ROOT="${RUNS_ROOT:-$(sed -n 's|^runs_dir=||p' "$file")}"
    DATA_DIR="${DATA_DIR:-$(sed -n 's|^data_dir=||p' "$file")}"
    rm -f "$file"
    for key in "${need[@]}"; do
        case $key in
            runs_dir) value=$RUNS_ROOT ;;
            data_dir) value=$DATA_DIR ;;
        esac
        if [ -z "$value" ]; then
            echo "The locations of $MSFM_CONFIG give no $key; set it in the environment." >&2
            exit 1
        fi
    done
}

#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=inference
#SBATCH --output=/users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/slurm/slurm-%j.out

# Likelihood-flow inference on the summaries of trained runs, one run per GPU. The defaults are the
# production settings; y3-deep-lss's prod/{maps,cls}.sh call this as their last stage. The six v18
# production runs (overwrites their chains):
#   RUNS="maps_gcnn/lensing/v1 maps_gcnn/clustering/v1 maps_gcnn/combined/v1
#         cls/lensing/v1 cls/clustering/v1 cls/combined/v1" sbatch inference.sh
#   LOAD_FLOW=--load_flow RUNS=... sbatch inference.sh           # resample an already trained flow
#   FLOW_LABEL=test FLOW_CONFIG=<yaml> RUNS=... sbatch inference.sh   # experiment beside the prod flow
# From inside a uenv session: env -u LD_LIBRARY_PATH sbatch --uenv-passthrough=ignore ...

source /users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/common.sh

: "${RUNS:?set RUNS to one or more <method>/<probe>/<run> dirs under RUNS_ROOT}"
RUNS_ROOT="${RUNS_ROOT:-$STORE/deep_lss/runs/$VERSION/$SUBVERSION}"
RUN_NUM="${RUN_NUM:-1}"  # names the log only

FLOW_CONFIG="${FLOW_CONFIG:-$MSI/configs/flow/maf.yaml}"
FLOW_CONFIGS="${FLOW_CONFIGS:-}"  # several configs: one heterogeneous ensemble, replaces FLOW_CONFIG
FLOW_LABEL="${FLOW_LABEL:-}"      # prefixes the flow dir, keeping an experiment off the prod flow
N_FLOWS="${N_FLOWS:-8}"           # rewrites the flow in place; the dir name omits the member count
EXTEND_PARAMS="${EXTEND_PARAMS:-}"  # e.g. "--extend_params ns Ob H0 bary_Mc"; overrides the flow config
LOAD_FLOW="${LOAD_FLOW:-}"        # --load_flow: sample an already trained flow

# ${VAR-default}: an explicitly empty value switches the stage off.
FLOW_MEMBERS="${FLOW_MEMBERS---sample_flow_members}"  # per-member DES chains, the blinding test
SAMPLE_POSTERIOR="${SAMPLE_POSTERIOR---sample_posterior}"
INCLUDE_OBS="${INCLUDE_OBS---include_grid --include_des --include_mocks}"

# --- Memory budget -----------------------------------------------------------------------------

# GH200 nodes report each GPU's HBM as memory too, so budget against the CPU NUMA nodes' DRAM.
host_dram_gb() {
    local total=0 n kb
    for n in /sys/devices/system/node/node*; do
        [ -n "$(cat "$n/cpulist" 2>/dev/null)" ] || continue
        kb=$(awk '/MemTotal/ {print $4; exit}' "$n/meminfo" 2>/dev/null)
        [ -n "$kb" ] && total=$((total + kb))
    done
    echo $((total / 1024 / 1024))
}
DRAM_GB=$(host_dram_gb)
[ "${DRAM_GB:-0}" -lt 64 ] && DRAM_GB=476

# Measured MaxRSS x 1.3 with the extended conditioning vector; the coverage stage dominates.
probe_need_gb() {
    case "$1" in
        lensing) echo 105 ;;
        clustering) echo 100 ;;
        *) echo 135 ;;
    esac
}

MAX_NEED=0
for RUN in $RUNS; do
    NEED=$(probe_need_gb "$(basename "$(dirname "$RUN")")")
    [ "$NEED" -gt "$MAX_NEED" ] && MAX_NEED=$NEED
done

GPUS_PER_NODE="${GPUS_PER_NODE:-$((DRAM_GB / MAX_NEED))}"
[ "$GPUS_PER_NODE" -gt 4 ] && GPUS_PER_NODE=4
[ "$GPUS_PER_NODE" -lt 1 ] && GPUS_PER_NODE=1
# Steps are carved out of the job's --mem, not out of host DRAM.
JOB_MEM_GB=$DRAM_GB
[ -n "${SLURM_MEM_PER_NODE:-}" ] && [ $((SLURM_MEM_PER_NODE / 1024)) -lt "$JOB_MEM_GB" ] \
    && JOB_MEM_GB=$((SLURM_MEM_PER_NODE / 1024))
STEP_MEM="${STEP_MEM:-$((JOB_MEM_GB / GPUS_PER_NODE))G}"

STEP_MEM_GB=${STEP_MEM%[Gg]}
case "$STEP_MEM_GB" in
    '' | *[!0-9]*) echo "STEP_MEM must be whole GB, e.g. 150G, got '$STEP_MEM'" >&2; exit 1 ;;
esac
if [ $((GPUS_PER_NODE * STEP_MEM_GB)) -gt "$DRAM_GB" ]; then
    echo "Refusing: $GPUS_PER_NODE x ${STEP_MEM_GB}G exceeds the ${DRAM_GB}G of host DRAM." >&2
    exit 1
fi
# A --mem above DRAM trades the clean per-step cgroup kill for a node-level OOM.
if [ $((${SLURM_MEM_PER_NODE:-0} / 1024)) -gt "$DRAM_GB" ]; then
    echo "Refusing: --mem=$((SLURM_MEM_PER_NODE / 1024))G exceeds the ${DRAM_GB}G of host DRAM." >&2
    exit 1
fi
echo "[budget] DRAM ${DRAM_GB}G, heaviest run ~${MAX_NEED}G: $GPUS_PER_NODE per node at $STEP_MEM"

# --- Inference, one run per GPU, in waves ------------------------------------------------------

FLOW_FLAGS="--flow_config=$FLOW_CONFIG"
[ -n "$FLOW_CONFIGS" ] && FLOW_FLAGS="--flow_configs $FLOW_CONFIGS"
[ -n "$FLOW_LABEL" ] && FLOW_FLAGS+=" --flow_label=$FLOW_LABEL"

infer_one() {
    local out_dir="$RUNS_ROOT/$(dirname "$1")" model=$(basename "$1")
    local log="$out_dir/$model/logs/${SLURM_JOB_ID}_${RUN_NUM}_mirrored_inference.log"
    [ "$DRYRUN" = "1" ] || mkdir -p "$(dirname "$log")"
    echo "[$(date +%T)] $1 -> $log"
    $SRUN "${GPU_STEP[@]}" --mem="$STEP_MEM" "${TORCH_ENV[@]}" --output="$log" \
        "$TORCH_PY" "$MSI/msi/apps/run_inference.py" \
            --out_dir="$out_dir" \
            --model_name="$model" \
            $FLOW_FLAGS \
            --n_flows="$N_FLOWS" \
            $EXTEND_PARAMS $LOAD_FLOW $FLOW_MEMBERS $SAMPLE_POSTERIOR $INCLUDE_OBS \
        || { echo "FAILED: $1, see $log" >&2; return 1; }
}

# Wait on every pid: a bare `wait` returns only the last one's status.
rc=0
pids=()
flush() {
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    pids=()
}
for RUN in $RUNS; do
    infer_one "$RUN" &
    pids+=($!)
    [ ${#pids[@]} -eq "$GPUS_PER_NODE" ] && flush
done
flush
exit "$rc"

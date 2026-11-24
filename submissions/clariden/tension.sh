#!/bin/bash
#SBATCH --account=lp184
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=tension
#SBATCH --output=/users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/slurm/slurm-%j.out

# Posterior tension over every pair in a runs config, configs/runs/<VERSION>/<RUNS_NAME>.yaml
# (VERSION has no default, RUNS_NAME defaults to prod; v18 prod has 21 pairs, ~2 h).
# Stage A builds the difference chains (torch), stage B their tension values (TensorFlow).
#   VERSION=desy3/v18 sbatch tension.sh
#   VERSION=desy3/v17 MSFM_CONFIG=$MSFM/configs/desy3/v17/baseline.yaml \
#       RUNS_CONFIG=$MSI/configs/runs/desy3/v17/baseline/t1_v3.yaml sbatch --time=01:00:00 tension.sh

set -euo pipefail
source /users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/common.sh
use_dataset

RUNS_NAME="${RUNS_NAME:-prod}"
RUNS_CONFIG="${RUNS_CONFIG:-$MSI/configs/runs/$VERSION/$RUNS_NAME.yaml}"
TENSION_CONFIG="${TENSION_CONFIG:-$MSI/configs/tension/tension.yaml}"

LOG="$LOG_DIR/${SLURM_JOB_ID}"

# --- Stage A: difference chains ----------------------------------------------------------------

$SRUN "${GPU_STEP[@]}" --mem=110G "${TORCH_ENV[@]}" --output="${LOG}_chains.log" \
    "$TORCH_PY" "$MSI/msi/apps/run_tension_chains.py" \
        --runs_config="$RUNS_CONFIG" \
        --tension_config="$TENSION_CONFIG" \
        --msfm_config="$MSFM_CONFIG" \
        --device=cuda

# --- Stage B: tension values -------------------------------------------------------------------

$SRUN "${GPU_STEP[@]}" --mem=110G --environment=tensorflow --gpu-bind=none \
    --output="${LOG}_values.log" \
    python "$MSI/msi/apps/run_tension_values.py" \
        --runs_config="$RUNS_CONFIG" \
        --tension_config="$TENSION_CONFIG"
[ "$DRYRUN" = "1" ] && exit 0

# Neither stage fails on a missing input, so count what landed.
echo "=== tension summary (job ${SLURM_JOB_ID}) ==="
printf 'stage A  difference chains saved : %s\n' "$(grep -c 'Saved chain to' "${LOG}_chains.log" || true)"
printf 'stage A  plot failures           : %s\n' "$(grep -c 'plotting failed' "${LOG}_chains.log" || true)"
printf 'stage B  significances saved     : %s\n' "$(grep -c 'Saved tension results to' "${LOG}_values.log" || true)"
printf 'stage B  chains missing          : %s\n' "$(grep -c 'chain, skipping' "${LOG}_values.log" || true)"
grep -n 'chain, skipping\|plotting failed\|Traceback' "${LOG}_chains.log" "${LOG}_values.log" \
    || echo "no skips, plot failures or tracebacks"

#!/bin/bash
#SBATCH --account=lp184
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=ppc
#SBATCH --output=/users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/slurm/slurm-%j.out

# Posterior predictive checks over every run and pair in a runs config, configs/runs/<VERSION>/<RUNS_NAME>.yaml
# (VERSION has no default, RUNS_NAME defaults to prod). A cold run trains ~10 PPC flows and takes ~6 h; re-runs
# recover the flows from disk. A DES Y3 generation with several configs per version names both files.
#   VERSION=desy3/v18 sbatch ppc.sh
#   VERSION=desy3/v17 MSFM_CONFIG=$MSFM/configs/desy3/v17/baseline.yaml \
#       RUNS_CONFIG=$MSI/configs/runs/desy3/v17/baseline/t2_v3.yaml sbatch ppc.sh
#   VERSION=desy3/v18 PPC_CONFIG=$MSI/configs/ppc/ppc_quick.yaml sbatch --partition=debug --time=00:30:00 ppc.sh

set -euo pipefail
source /users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/common.sh
use_dataset

RUNS_NAME="${RUNS_NAME:-prod}"
RUNS_CONFIG="${RUNS_CONFIG:-$MSI/configs/runs/$VERSION/$RUNS_NAME.yaml}"
PPC_CONFIG="${PPC_CONFIG:-$MSI/configs/ppc/ppc.yaml}"
RETRAIN_FLOWS="${RETRAIN_FLOWS:-}"  # --retrain_flows ignores the flows on disk

LOG="$LOG_DIR/${SLURM_JOB_ID}_ppc.log"

$SRUN "${GPU_STEP[@]}" --mem=110G "${TORCH_ENV[@]}" --output="$LOG" \
    "$TORCH_PY" "$MSI/msi/apps/run_ppc.py" \
        --runs_config="$RUNS_CONFIG" \
        --ppc_config="$PPC_CONFIG" \
        --msfm_config="$MSFM_CONFIG" \
        $RETRAIN_FLOWS \
        --device=cuda
[ "$DRYRUN" = "1" ] && exit 0

# run_ppc skips a check whose input is missing and still exits 0, so count what landed.
echo "=== ppc summary (job ${SLURM_JOB_ID}) ==="
printf 'auto runs processed     : %s\n' "$(grep -c '=== auto:' "$LOG" || true)"
printf 'cross pairs processed   : %s\n' "$(grep -c '=== cross:' "$LOG" || true)"
printf 'calibrations saved      : %s\n' "$(grep -c 'Saved calibration summary to' "$LOG" || true)"
printf 'calibration nulls saved : %s\n' "$(grep -c 'Saved calibration null arrays to' "$LOG" || true)"
printf 'warnings                : %s\n' "$(grep -c ' WAR ' "$LOG" || true)"
grep -n 'skipping p-value calibration\|disabled:\|Traceback' "$LOG" \
    || echo "no skipped calibrations, disabled checks or tracebacks"

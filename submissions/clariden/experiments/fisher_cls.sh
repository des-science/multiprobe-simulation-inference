#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=fisher_cls
#SBATCH --output=/users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/slurm/slurm-%j.out

# Fisher forecast for (Om, s8, w0) from the hard_rebinned Cls, per probe, on CPU (msi/dev/fisher).
# Runtime is ~linear in N_BINS (nb16 ~13 min). Output: runs/<version>/<subversion>/fisher_cls/<scales>/nb<N>/
#   sbatch fisher_cls.sh
#   VERSION=v17 SUBVERSION=baseline SCALES=lmax_1024 N_BINS=32 sbatch fisher_cls.sh

source /users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/common.sh
export OMP_NUM_THREADS=288 TF_NUM_INTRAOP_THREADS=288 CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2

SCALES="${SCALES:-lmax_1024}"  # y3-deep-lss configs/scales/
N_BINS="${N_BINS:-16}"
# Extra prior diagnostics beside the headline (noncosmo | none | all); may be empty.
PRIOR_VARIATIONS="${PRIOR_VARIATIONS-noncosmo,none}"

OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/fisher_cls/$SCALES"
LOG="$OUTPUT/nb$N_BINS/run-${SLURM_JOB_ID}.log"  # the app writes into nb<N_BINS>/ itself
[ "$DRYRUN" = "1" ] || mkdir -p "$(dirname "$LOG")"

$SRUN --environment=tensorflow --output="$LOG" \
    bash -c "source ~/dlss/tf_env/bin/activate && python $MSI/dev/fisher/fisher_cls.py \
        --data_dir=$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION \
        --msfm_config=$MSFM_CONFIG \
        --scales_config=$DEEP_LSS/configs/scales/$SCALES.yaml \
        --cls_n_bins=$N_BINS \
        --prior_variations='$PRIOR_VARIATIONS' \
        --out_dir=$OUTPUT"

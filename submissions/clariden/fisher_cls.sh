#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
# runtime scales ~linearly with N_BINS (measured on 8wl,32gc: nb8 7 min, nb16 13 min, nb32 25 min),
# so 30 min would leave nb32 almost no margin. 1 h covers nb32 comfortably and leaves room for nb64.
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=fisher_cls
#SBATCH --output=/users/athomsen/dlss/repos/multiprobe-simulation-inference/submissions/clariden/slurm/slurm-%j.out

# Fisher forecast for (Om, s8, w0) from the hard_rebinned Cls, separately for the lensing /
# clustering / combined probes (fixed + nuisance-marginalized). Pure numpy + TF-CPU (reads the
# fiducial TFRecords for the derivatives); no GPU. Runs in the TensorFlow environment / tf_env.
#
# Like training.sh, the dataset, scale cut and number of ell bins are chosen via environment
# variables (nothing is hardcoded in the analysis script's name):
#   VERSION=v17 SUBVERSION=baseline SCALES=lmax_1024 N_BINS=16 sbatch fisher_cls.sh
#
# Parameters fall in three classes: cosmo (Om,s8,w0 -- what we report), astro (the probe's physical
# astrophysical params: intrinsic alignments for lensing, galaxy biases for clustering) and nuisance
# (the barely-constrained rest: H0,Ob,ns,bary_Mc,bary_nu).
#
# The HEADLINE result has ONE knob: which params the model includes. The three forecast modes include
# a widening set -- fixed (cosmo only) / astro / astro_nuisance (astro + nuisance) -- and each one
# applies the analysis prior (analysis.grid.priors, a top-hat encoded as its variance-matched
# Gaussian sigma=(b-a)/sqrt(12)) to exactly the params it includes, marginalizes over the non-cosmo
# ones, and reports (Om,s8,w0). No prior knob to choose: model membership decides it.
#
# PRIOR_VARIATIONS is a comma-separated LIST of EXTRA diagnostics that deliberately DEcouple the
# prior from the model. They are free (same prior-free Fisher, only the final <=14x14 inversion is
# redone) and land in a prior_variations/ subfolder so the headline stays uncluttered. Default
# "noncosmo,none"; pass PRIOR_VARIATIONS="" for the headline only.
#   noncosmo  prior on astro+nuisance only, cosmo left prior-free -> the pure-data cosmo sigmas.
#             Worth reading next to the headline: the gap is how much of it is prior, not data.
#   none      prior-free; informationless directions (e.g. bary_Mc in clustering-only, whose
#             derivative sits at the Monte-Carlo noise floor) then blow the cosmo errors up.
#             A pathology demo, not a result.
#   all       re-emits the headline under a tagged name (normally redundant).
#
# Outputs mirror the input dataset layout. Each run (one N_BINS) gets its OWN nb<N> subfolder so an
# ell-bin sweep stays tidy:
#   runs/$VERSION/$SUBVERSION/fisher_cls/$SCALES/nb<N_BINS>/
#     fisher_<probe>.npz                         headline, untagged
#     fisher_ellipses_<mode>.png                 probes overlaid, one per marginalization mode
#                                                (mode = fixed / astro / astro_nuisance)
#     fisher_ellipses_probe_<probe>.png          modes overlaid, one per probe
#     run-<jobid>.log
#     prior_variations/                          same files per variation, tagged _prior_<mode>

# disable core dumps: a crashing TF/Python task can otherwise write a huge core file and fill quota
ulimit -c 0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
# keep TF on CPU and quiet
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"
MSI="$REPOS/multiprobe-simulation-inference"

# VERSION/SUBVERSION/SCALES may be overridden from the environment to point at a different dataset
# or scale cut (e.g. SCALES=8wl,32gc, SCALES=unsmoothed). Defaults = the v17/baseline lmax_1024 run.
VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"
SCALES="${SCALES:-lmax_1024}"
N_BINS="${N_BINS:-16}"
# variations only -- the headline (prior coupled to the model) is always produced. May be empty.
PRIOR_VARIATIONS="${PRIOR_VARIATIONS-noncosmo,none}"

MSFM_CONFIG="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml"
SCALES_CONFIG="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/fisher_cls/$SCALES"
# the analysis script appends its own nb<N_BINS> subfolder to --out_dir so each ell-bin run gets
# its own folder; keep the srun log in that same subfolder rather than the shared scale dir.
RUN_DIR="$OUTPUT/nb$N_BINS"
LOG="$RUN_DIR/run-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

srun --environment=tensorflow --output="${LOG}.log" \
    bash -c "source ~/dlss/tf_env/bin/activate && python $MSI/dev/fisher/fisher_cls.py \
        --data_dir=$INPUT \
        --msfm_config=$MSFM_CONFIG \
        --scales_config=$SCALES_CONFIG \
        --cls_n_bins=$N_BINS \
        --prior_variations='$PRIOR_VARIATIONS' \
        --out_dir=$OUTPUT"

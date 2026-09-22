# Round 5 confirmation on individual probes

Run `all_long` and `extended_long` for each of:

| Probe | Input under `scratch/runs/v18/default/maps_gcnn` | Summary dimension | Extended context dimension |
|---|---|---:|---:|
| lensing | `lensing/v1/preds_126000.h5` | 6 | 9 |
| clustering | `clustering/v1/preds_126500.h5` | 7 | 10 |

Prepared outputs are in each probe's `v1/flow_round5_probes_clean` directory.
Both arms use all simulations, eight MAF members, seed 7, 304 epochs and
5,168 updates/member. Architecture and sampling settings match the combined
round; the extension appends `ns, Ob, H0` with their wide analysis priors.
The baseline provides a matched comparison for each probe. All 1,000 mock
identities and their order match the combined round exactly.

The launcher is in `y3-deep-lss/submissions/clariden/experiments/coverage_probes.sh`.
It launches four concurrent one-GPU steps on one four-GPU node and scores both
completed pairs afterward. Each prepared manifest records hashes of inputs,
configuration and source code. Preparation and execution reject overwrites.
The generalized runner records the actual compressor checkpoint; old combined
manifests without that field retain their 229900 interpretation. Scorer TARP
labels now reflect the actual common parameter dimension.

Before production, review marginal coverage and PIT effect sizes, likelihood
checks, paired baseline differences and posterior tails for all three probes.
The automated no-rejection screen alone is insufficient. Check sampler mixing
using retained temporal chains and dispersed starts on selected mocks, including
the problematic combined tails, and repeat with an independent training seed.
These additional runs are separate from this four-arm submission. Reusing these
development mocks does not provide a fresh final validation set. Production
inference is deferred until these results and additional checks are reviewed.

## Results

Both pairs completed and were scored into each round's `scores.json`. `extended_long` improves
on `all_long` in EVERY HPD and TARP test, on both probes -- there is no test where the baseline
is closer to nominal:

| Test (KS D, smaller better) | lensing base | lensing ext | clustering base | clustering ext |
|---|---:|---:|---:|---:|
| HPD (joint, own dimension) | 0.0484 | **0.0308** | 0.0538 | **0.0232** |
| TARP joint | 0.0240 | 0.0320 | 0.0252 | 0.0272 |
| TARP Om | 0.0254 | **0.0230** | 0.0426 | **0.0184** |
| TARP s8 | 0.0424 | **0.0314** | 0.0156 | 0.0216 |
| TARP w0 | 0.0396 | **0.0252** | 0.0324 | **0.0298** |
| TARP (Om,s8,w0) | 0.0374 | **0.0214** | 0.0340 | **0.0256** |

The baseline HPD tests are the ones that come closest to rejecting (unadjusted p = 0.018
lensing, 0.0059 clustering); extended_long moves both to 0.29 and 0.65. Nothing rejects at
0.01 under the within-arm Holm screen in any of the four arms, which is exactly why that
screen must not be the acceptance criterion (ROUND5_REVIEW.md, finding 1).

Marginals, nominal 1.000 and 0.680:

| | RMSE / RMS posterior sd | | | equal-tail coverage at 68% | | |
|---|---:|---:|---:|---:|---:|---:|
| | Om | s8 | w0 | Om | s8 | w0 |
| lensing all_long | 1.094 | 1.085 | 1.005 | 0.650 | 0.657 | 0.681 |
| lensing extended_long | 1.023 | 0.997 | 0.986 | 0.685 | 0.688 | 0.689 |
| clustering all_long | 1.145 | 1.022 | 1.007 | 0.602 | 0.688 | 0.679 |
| clustering extended_long | 0.990 | 1.009 | 0.999 | 0.675 | 0.697 | 0.694 |

Paired cosmology-block bootstrap, `extended_long` minus `all_long`, Om equal-tail coverage at
68%: lensing +0.035 [0.018, 0.052], clustering +0.073 [0.054, 0.092]. Both exclude zero. The
gain concentrates in Om, which is also where the baseline was worst on both probes; s8 and w0
were already near nominal there and move within noise. On the combined probe all three moved
(+0.054, +0.051, +0.036), so the pattern is probe-dependent, not a fixed correction.

Note the TARP joint test does NOT improve on either probe (-0.005 and -0.002 in coverage, both
CIs spanning zero). Coverage was already nominal there in the baseline, so there was nothing
for the correction to recover; it is not evidence against the change.

This satisfies finding 5 of ROUND5_REVIEW.md for the probe axis. The seed axis, and the sampler
mixing check of finding 4, are still open.

## Execution notes

The initial job 3474356 was cancelled after detecting that Slurm inherited all
four GPUs into its first step, serializing the arms. Its partial outputs remain
in `flow_round5_probes`. The replacement explicitly requests four task slots in
the allocation and one GPU per node/task in each exact-sized step. It uses fresh
manifests and the cleaned source; retained training paths matched the original
source exactly on seeded CPU fixtures (weighted/unweighted, sequential/fused).

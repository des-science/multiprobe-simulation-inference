# Round 5 review — 2026-09-22

Reviewed completed outputs in
`/users/athomsen/dlss/storage/runs/v18/default/maps_gcnn/combined/v1/archive/flow_round5`.
No jobs submitted, checkpoints changed, or frozen source/configuration files edited
during this review. All repositories checked remain on main; changes are uncommitted.

## Execution and data integrity

- Job 3467116: all_long, projected_long, joint_long, conditional_long. All four
  fused training runs started at 18:52:29 on September 21.
- Job 3467122: all_short, wide_short, wide_long, extended_long. All four fused
  training runs started at 18:52:47. The eight arms ran concurrently across two
  jobs, not sequentially in one job.
- All eight completion markers exist; logs record successful CUDA training,
  posterior sampling, and diagnostics. No ERROR, Traceback, failure, fallback,
  or step-serialization message was found by the log search.
- All file hashes listed in the prepared manifest still match, including the
  prediction file, metadata, configurations, mock list, and pinned source files.
- Update counts match the design: 2448 for short, 5168 for long. Both backends
  were unit-tested before submission; the completed runs used fused training.
- The 1000 mock identities, original ten-dimensional truths, and observed
  summaries match exactly across all eight HDF5 outputs. Arrays contain
  10,000 samples per mock with ten or thirteen labeled parameters as intended.
- An independent one-in-ten sample check found finite values and no box-prior
  violations. This screen is not a full convergence or support audit; the
  completion validator separately checks finiteness of the full stored arrays.

## Coverage results

Equal-tail marginal coverage at nominal 68%, from the existing scores.json
(2500 uniformly subsampled stored draws per mock):

| Arm | Om | s8 | w0 | Joint HPD coverage at 68% |
|---|---:|---:|---:|---:|
| all_short | 61.0% | 61.1% | 65.0% | 69.2% |
| all_long | 61.3% | 62.5% | 65.0% | 65.3% |
| projected_long | 63.0% | 63.0% | 66.7% | 65.1% |
| joint_long | 66.2% | 66.8% | 68.2% | 65.4% |
| conditional_long | 64.8% | 67.2% | 67.3% | 65.1% |
| wide_short | 66.4% | 66.2% | 67.6% | 69.2% |
| wide_long | 66.1% | 66.3% | 68.0% | 65.1% |
| extended_long | 66.7% | 67.6% | 68.6% | 68.4% (13D) |

The extended joint HPD test is thirteen-dimensional, so its numerical difference
from the other arms is not a paired comparison of the same credible regions.
The TARP tests on the original ten parameters and their subsets are comparable.
The extended arm's 95% equal-tail coverage is 94.9%, 95.4%, 94.8% for Om,s8,w0.

Existing paired bootstrap intervals, in percentage points:

| Comparison | Om coverage change | s8 coverage change | w0 coverage change |
|---|---:|---:|---:|
| extended_long minus all_long | +5.4 [3.8, 7.1] | +5.1 [3.3, 6.9] | +3.6 [1.9, 5.2] |
| conditional_long minus all_long | +3.5 [1.8, 5.3] | +4.7 [2.9, 6.6] | +2.3 [0.6, 3.9] |
| joint_long minus projected_long | +3.2 [1.7, 4.7] | +3.8 [2.3, 5.4] | +1.5 [0.0, 3.0] |

These intervals describe the paired mock comparison, not variability over training
seeds or all dependence from reused simulations. They are approximate for this
Sobol truth design.

## Main interpretation

The nuisance-prior explanation has substantially stronger support than before:
correcting the nuisance distribution improves coverage both while retaining the
original retained-parameter design (conditional arm) and after flattening it
(joint versus projected). Explicit conditioning provides a third, independent
implementation route with similar improvements. This does not establish that
all estimator errors have been removed or uniquely identify their mechanism.

Longer training alone does not fix the cosmological marginal coverage. Full-joint
weighting and wide-only training give nearly identical marginal coverage at a
matched update budget. There is consequently no evidence here that discarding
the narrow simulations is necessary.

The user's expectation of weak nuisance constraints is consistent with the
extended posteriors: the median posterior SD divided by the SD of the wide
uniform prior is 0.934 for ns, 0.966 for Ob, and 0.966 for H0 (computed using
2500 saved draws per mock). Weak marginal constraints do not imply that their
prior can be changed without affecting the cosmological posterior.

## Findings that prevent declaring production readiness

1. **The automated no-rejection flag is not an acceptance criterion.** Every arm
   passes `no_rejection_at_0.01_approx`, including the visibly under-covering
   baseline. Its seven-test Holm screen includes HPD/TARP but not the marginal
   PIT tests. Preserve the original report and distinguish this prespecified
   screen from conclusions based on the complete diagnostic evidence. Choosing
   the largest p-value or reporting all eight as calibrated would be misleading.

2. **A few broad tails dominate variance-based diagnostics.** In all_long,
   joint_long and wide_long, the ten largest per-mock s8 variances contribute
   approximately 25%, 22% and 25% of the total (one-in-ten sample screen). For
   extended_long the contribution is about 3%. For Sobol ID 90, whose true s8
   is 0.4547, joint_long places 32.48% of the full 10,000 stored samples above
   s8=0.8; extended_long places none there. This is a concrete posterior to
   inspect, not evidence by itself that the high-s8 mass is mathematically
   spurious. A small aggregate RMSE/RMS-SD can conceal this behavior.

3. **Residual rank discrepancies remain.** The full-sample SBC log for the
   extended arm reports an unadjusted w0 KS p-value around 0.0041. Its subsampled
   PIT calculation gives D=0.0466 and p approximately 0.025, illustrating that
   threshold decisions are also sensitive to finite posterior sampling. These
   p-values are approximate and not independent evidence across diagnostics;
   do not silently translate them into a new post-selection acceptance rule.

4. **MCMC convergence is not established by the saved outputs.** Posterior draws
   are randomly subsampled after flattening; temporal and walker identities,
   acceptance rates and convergence diagnostics are not saved. The common
   initialization is a tight cloud around the fiducial cosmology, especially
   tight relative to H0's range. Completed jobs and finite samples cannot prove
   that every observation's posterior was explored. Broad/multiple-mode cases
   deserve focused checks using retained chain structure and dispersed starts.

5. **Validation remains limited to one compression model/probe and one ensemble
   seed set.** These are the same development mocks used for model selection,
   and the split holds out signal realizations, not cosmologies. The other
   probes and fresh confirmation remain outstanding.

## Recommendation

Use extended_long as the leading candidate for confirmation, with conditional_long
as the strongest standard-context alternative worth investigating. Do not promote
either to all production runs yet. First examine the broad-tail mocks and sampler
convergence, then confirm with another ensemble seed and the remaining probes
under an explicitly fixed diagnostic decision rule. No new runs were launched
as part of this review.

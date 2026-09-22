# Round 5: likelihood target versus training budget

Input: `/users/athomsen/dlss/scratch/runs/v18/default/maps_gcnn/combined/v1/preds_229900.h5`.
The combined round has completed; see ROUND5_REVIEW.md. Individual-probe confirmation
is described in ROUND5_PROBES.md.
The compression network stays fixed. No DES observation is sampled.

## Outcome

`extended_long` won and is now the production default: `configs/flow/maf.yaml` carries
`extend_params: [ns, Ob, H0]`, which its header documents. It was confirmed against the
`all_long` baseline on all three probes, improving every HPD and TARP test on lensing and
clustering and landing within 0.03 of nominal on every marginal. It is the only arm that
keeps all 175,000 training rows: reweighting costs effective sample size (joint ESS/N 0.52,
conditional 0.35) and wide-only training discards half the simulations for no measured gain
(`joint_long - wide_long` is null on Om, s8 and w0).

Promoted on the user's decision, ahead of two checks ROUND5_REVIEW.md asks for and which are
still open: sampler mixing on the extended chains (item 4 -- under-mixing broadens a posterior
in the same direction as the improvement, and the saved draws are subsampled after flattening,
so it cannot be checked post hoc), and an independent ensemble seed (item 5, `--seed 107`).
`conditional_long` remains the fallback if mixing turns out to explain the extended arm: it
samples the same ten dimensions as the baseline, so it carries no such confound, and it reaches
RMSE/RMS-sd 1.021 / 0.987 / 1.004 on the combined probe.

The `training.train_prior` weighting modes below are retained as the experimental controls that
established the mechanism. Production does not use them.

## Questions and arms

Distinguish a mismatch in implicit nuisance marginalization from effects of
optimization and training design. These experiments cannot by themselves prove
that either explains all remaining coverage errors.

All arms use eight MAF members, the baseline four layers, 128 hidden units,
dropout 0.1, embedding dimension 16, batch size 10,000, uniform member weights,
and cosine learning-rate decay. The extended arm has 13 conditioning parameters
but still ten summary features; its embedding remains larger than its context.

| Arm | Training target/design | Epochs | Updates/member |
|---|---|---:|---:|
| all_short | Original mixture, standard context | 144 | 2,448 |
| all_long | Original mixture, standard context | 304 | 5,168 |
| projected_long | Flatten Om, s8, w0 only | 304 | 5,168 |
| joint_long | Correct the full six-coordinate mixture to the wide prior | 304 | 5,168 |
| conditional_long | Correct ns, Ob, H0 conditional on Om, s8, w0; retain original theta density | 304 | 5,168 |
| wide_short | Existing id_param < 1250 selection | 306 | 2,448 |
| wide_long | Same wide-only selection | 646 | 5,168 |
| extended_long | Original mixture; explicitly condition on ns, Ob, H0 and sample their wide priors | 304 | 5,168 |

The 175,000 full-grid training rows give 17 complete batches per epoch; the
87,500 wide-grid rows give eight. Budgets are multiples of 136, permitting exact
update-count matches without modifying either optimizer backend. Cosine decay
spans each full run; its epoch-wise discretization and the data-reuse frequency
still differ between full and wide runs. Short/long pairs test training duration
with a complete learning-rate schedule, not continuation of the short checkpoint.
The old 300-epoch results are historical context, not exactly matched controls.

Primary comparisons:

1. **conditional_long vs all_long:** change nuisance marginalization while
   retaining the population theta design. This is the most direct test of the
   nuisance explanation, though finite-sample weights also change gradient noise.
2. **joint_long vs projected_long:** both flatten the retained cosmological
   parameter density; only joint weighting corrects the omitted nuisances.
3. **joint_long vs wide_long:** same intended marginalized likelihood and update
   budget, using all rows versus the wide subset. Context standardization and
   data-dependent initialization remain fitted by the existing implementation;
   their finite-sample differences are additional estimator effects.
4. **all_short/long and wide_short/long:** determine whether the improvement
   from wide-only training depends on its smaller optimization budget.
5. **extended_long vs all_long:** independent route to the wide nuisance prior,
   at the cost of a harder conditional-density/interpolation problem. Failure
   of this arm alone does not rule out the nuisance explanation.

No thinning, theta jitter, post-hoc calibration, or narrower context bottleneck.

## Grid audit and weight definitions

The published construction continues the wide Sobol sequence and rejects points
outside the narrow six-dimensional box; it does not rescale a second sequence.
See [CosmoGridV1, section 2 and table 2](https://arxiv.org/html/2209.04662v2#S2).
The design is treated as a 50/50 mixture of the two normalized component densities.
Other nuisance coordinates are assumed to follow the same conditional design in
both components; bary_Mc and bary_nu retain their wide ranges in both halves.

Let `t = (Om,s8,w0)` and `e = (ns,Ob,H0)`. With `r` the simulation design and
`pi` the wide analysis prior, the unnormalized weights are:

```
projected   pi(t) / r(t)
joint       pi(t,e) / r(t,e)
conditional [pi(t,e) / r(t,e)] / [pi(t) / r(t)]
```

The third preserves `r(t)` while replacing `r(e|t)` by the wide nuisance prior.
All weights are normalized to mean one, and every row is retained in these arms.
Membership is determined from all necessary coordinates, including for points
originating from the wide component. Metadata is joined to rows by `sobol_index`.
The old `training.train_prior: reweight` implementation was removed during cleanup;
`reweight_projected` is the explicit deterministic control. Historical runs retain
their saved configurations.

Deterministic quadrature uses the actual Om-s8 hull and the w0 cut. The independent
nuisance interval ratios then give the six-dimensional volume ratio. On this input:

| Quantity | Value |
|---|---:|
| Narrow/wide volume in t | 0.326010529 |
| Narrow/wide volume in (t,e) | 0.021130312 |
| Projected inside/outside weight | 0.245858175 |
| Joint inside/outside weight | 0.020693061 |
| Projected row ESS / N | 0.6649 |
| Joint row ESS / N | 0.5169 |
| Conditional row ESS / N | 0.3530 |

Row ESS describes weighting only, not the number of independent simulations.
Equal updates do not imply equal gradient variance or equal effective information.
An improved weighted fit would not alone prove the causal mechanism; corroboration
across the conditional, joint and extended arms is more informative.

The handoff's geometric counts were incorrect: 400 of the first 1,250 points fall
inside the projected box, but only 19 inside the full box. Nine points with
id_param 1250..1258 lie outside the full narrow box. The user judged these
negligible: preserve the existing coverage/selection convention and record the
exceptions rather than silently redefining it. The 50/50 density ratio is thus
the documented-design approximation, not a reconstruction of every grid edit.

## Paired diagnostics and interpretation

Preparation freezes the existing selection of 1,000 wide-grid held-out mocks in
`mock_ids.npy`. Every identity is unique, including the cosmology. Training uses
signal groups 64..77; coverage uses groups 78..79. Both likelihood and posterior
coverage must match those exact identities, including their order. Missing or
training-set identities raise errors. This also fixes the previous discrepancy
where likelihood-level plots sampled the full mixture while posterior plots
selected the wide grid. New selection behavior is opt-in through `mock_ids_file`.

The original MCMC settings remain 1,024 walkers, 1,000 burn-in and 1,000 retained
steps, with 10,000 stored samples per mock. Sampling and subsampling seeds are
fixed. Likelihood checks draw 1,000 samples per mock rather than the old 100.
Each arm must produce all required likelihood/posterior HPD and TARP plots before
the launcher writes its `.complete` marker. Parameter names are saved in the HDF5
attributes, which prevents interpreting extended chains with the old ten labels.

`score_coverage_round` adds numerical comparisons using 2,500 samples per mock:

- HPD ranks, with dimension reported (13 for the extended arm, ten otherwise).
- TARP on the common ten parameters, Om, s8, w0, (Om,s8), and (Om,s8,w0).
  Independent references use fixed seeds 17, 29, 43 and fixed wide-prior box
  scaling, so all arms use the same reference geometry. Seed 17 is the primary
  test; the others check sensitivity, not opportunities to choose a passing seed.
  These numerical curves need not equal the historical plots, which use TARP's
  sample-dependent min-max normalization and bootstrap reference draws.
- Equal-tail coverage at 68/90/95/99%, PIT KS distances, RMSE, and RMS posterior
  standard deviation for each of the original ten parameters.
- Mean log likelihood on the same wide mocks, with paired differences only for
  arms sharing the conditioning vector. The extended conditional density is a
  different target and its log likelihood must not be ranked against the others.
- Paired cosmology-block bootstrap intervals for changes in coverage, squared
  error and posterior variance. Never compare joint HPD coverage differences
  across the ten- and thirteen-dimensional arms.

The seven primary tests (HPD plus six TARP groups) receive a within-arm Holm
adjustment at 0.01 as a **screen**, not a production acceptance guarantee. KS
p-values are approximate because the truths form a Sobol design and posterior
draws are finite and correlated. The bootstrap likewise approximates uncertainty
and cannot account for all dependence from reused simulation volumes. Check
effect sizes and reference stability, not largest p-value. No regional subset
of true parameters is tested against an unjustified uniform-rank null.

Do not select on RMSE/uncertainty alone or on averages hiding a failed marginal.
For any promising setup, inspect sampler mixing and confirm with an independent
ensemble seed (prepare another directory with `--seed 107`), then apply one fixed
configuration to the other probes and input types. These are development mocks,
not a fresh final validation sample; seed repetition is not new mock evidence.
The round alone cannot establish all-probe production readiness.

## Prepare, review, launch, score

From `/users/athomsen/dlss/repos`, preparation in the existing environment:

```bash
.claude/bin/plot.sh python -m msi.apps.coverage_round prepare \
  --output /users/athomsen/dlss/scratch/runs/v18/default/maps_gcnn/combined/v1/flow_round5
```

Preparation refuses an existing output directory. It writes eight complete YAML
configs, copies the run config, links the prediction file, saves mock identities,
and records input/config/code SHA256 hashes and the grid audit in `manifest.json`.
Each arm checks these hashes before training and refuses existing results. After
a failure, prepare a new directory or explicitly inspect/remove the failed arm's
outputs and `.started` marker; there is no automatic destructive retry.

Review the command for one arm without training:

```bash
.claude/bin/plot.sh python -m msi.apps.coverage_round run \
  --round /users/athomsen/dlss/scratch/runs/v18/default/maps_gcnn/combined/v1/flow_round5 \
  --arm joint_long --dry-run
```

When ready to launch (not executed during preparation), as two single-wave jobs
on two nodes, four arms each. Both write into the same prepared round directory;
arm sets are disjoint, job ids namespace the logs, and the per-arm `.started`
marker is created `O_EXCL`, so the two cannot collide.

```bash
R=/users/athomsen/dlss/scratch/runs/v18/default/maps_gcnn/combined/v1/flow_round5
S=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/experiments/coverage_round.sh

# job 1 -- the target question: does correcting the implicit nuisance
# marginalization fix coverage, and is the projected correction enough?
ROUND_DIR=$R ARMS="all_long projected_long joint_long conditional_long" \
  sbatch --job-name=round5_target "$S"

# job 2 -- the budget question, plus the route that conditions explicitly on the
# previously marginalized nuisance parameters.
ROUND_DIR=$R ARMS="all_short wide_short wide_long extended_long" \
  sbatch --job-name=round5_budget "$S"
```

Each submission allocates one four-GPU node and runs its four arms as one wave,
one GPU per arm, with a two-hour ceiling. Four arms per job is the wave width,
so a fifth name would silently start a second wave.

The two-hour ceiling leaves headroom for queue-node variability. Cost is dominated by the
posterior-coverage MCMC, which is the same in every arm: a round-4 combined arm
spent 11:57 there (4:13 burn-in, 4:13 chain, the rest tests and plots) against
1:35 of flow training, for 13:40 total. Training scales with the update budget,
so the long arms add about two minutes, and raising `n_likelihood_samples` from
100 to 1000 turns a one-second stage into ten. Do not trim this to half an hour:
there is no mid-arm checkpoint, so a TIMEOUT loses all four arms of that job and
leaves `.started` markers that must be cleared by hand before a retry. Check concurrent `squeue -s -u "$USER"` steps when first launched;
CPU dry runs do not verify Slurm allocation geometry. Per-arm logs and failures
are retained, and any failed arm makes that submission return a nonzero status.

Every primary comparison is paired through `mock_ids.npy`, not through the job,
so the split costs nothing: comparisons 1 and 2 close inside job 1, 5 inside
job 2, and 3 and 4 across the two once both have finished.

After completion:

```bash
.claude/bin/plot.sh python -m msi.apps.score_coverage_round \
  --round /users/athomsen/dlss/scratch/runs/v18/default/maps_gcnn/combined/v1/flow_round5 \
  --output /users/athomsen/dlss/scratch/runs/v18/default/maps_gcnn/combined/v1/flow_round5/scores.json
```

Tests (small CPU fixtures, not experiments on these summaries):

`OMP_NUM_THREADS=1` is required on a login node: the autograd engine otherwise spawns one
thread per core and the integration test dies with `RuntimeError: Resource temporarily
unavailable`, which looks like a logic failure and is not one.

```bash
OMP_NUM_THREADS=1 .claude/bin/plot.sh python -m unittest discover -s multiprobe-simulation-inference/tests
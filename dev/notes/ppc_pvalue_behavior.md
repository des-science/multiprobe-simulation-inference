# Why raw PPC p-values behave the way they do (distance stats → ~1, log-prob → ~0.5)

**Context.** Running the posterior predictive checks (PPC, `msi/utils/ppc.py`, driven by
`msi/apps/run_ppc.py`), the *uncalibrated* distance-based statistics (Mahalanobis, L1/L2, L∞, kernel)
come out with p-values very close to **1**, while the *uncalibrated* `log_prob` check sits near
**0.5**. Calibrating (Doux et al. 2020, Eq. 9) maps the distance-stat p ≈ 1 back to p̃ ≈ 0.5 but barely
moves `log_prob`. Example run where this was seen:
`runs/v16/rot_in_place/maps/clustering/v8_cls/ensemble_flow_600000/ppc/auto_gc/likelihood_flow`.

The two families of test are constructed differently, and that explains the gap.

## Distance stats sit at p ≈ 1: the observation is *too central* for the marginal PPD cloud

In [`_pval_one_sample`](../../msi/utils/ppc.py) (around L855): `t_obs = stat(s_obs)`, the null
`t_boot = stat(s_rep)` over PPD draws, and (for "outlier if high") `p = mean(t_boot ≥ t_obs)`.
So **p ≈ 1 means `s_obs` is more central (smaller distance-to-cloud / smaller D²) than nearly every
PPD draw.**

The cause is the variance budget of the *marginal* PPD. The cloud `s_rep` is drawn by
[`_sample_neural`](../../msi/utils/ppc.py) (L559): `θ ~ p(θ|s_obs)`, then `s_rep ~ p(s|θ)`. Its spread
is therefore

```
Var(PPD) = intrinsic/noise scatter  +  posterior-parameter scatter
```

but the observation is displaced from the cloud centre by only the **noise** piece
(`s_obs ≈ signal(θ_true) + noise`; cloud centre ≈ `signal` at the posterior mean). So `s_obs` lives at
radius ~`σ_noise`, while a typical PPD draw lives at radius ~`√(σ_noise² + σ_post²) > σ_noise`. The
data point is *inside* the shell the replicates occupy.

In Mahalanobis terms: `E[D²(s_rep)] ≈ d` (χ²_d-like, concentrated near the summary dimension `d`),
whereas `E[D²(s_obs)] ≈ d · σ_noise²/σ_PPD² < d`. Hence `t_obs` falls in the low tail → `p → 1`.

This is **not** "great fit" — it is the expected null behaviour. It gets *more* extreme when:
- the posterior contributes a large share of the PPD width (informative summaries — e.g. clustering), and
- the summary dimension `d` is larger (concentration of measure sharpens the centre-vs-shell gap).

Eq.-9 calibration ([`run_calibration`](../../msi/utils/ppc.py), L1074) removes exactly this systematic
bias: consistent wide-prior mocks also give p ≈ 1, so the observed p ≈ 1 maps to p̃ ≈ 0.5.

## log_prob is already ≈ 0.5: a balanced, matched-θ comparison

[`_pval_log_prob`](../../msi/utils/ppc.py) (L776) is a different construction:

```
δ_i = log p(s_rep_i | θ_i) − log p(s_obs | θ_i),   θ_i ~ p(θ|s_obs),  s_rep_i ~ p(s|θ_i)
p   = mean(δ_i ≤ 0)            # fraction where s_obs is at least as likely as the replicate
```

Two things make it self-centering:

1. **Matched θ, conditional (not marginal) density.** Both terms use the *same* `p(·|θ_i)`, the
   conditional, which carries only the noise scatter — *not* the inflated posterior spread. There is no
   centre-vs-shell radius mismatch because the comparison never touches the marginal cloud geometry.
2. **Exchangeability under a good fit.** At `θ_i` near `θ_true`, `s_rep_i` is a noise draw from
   `p(s|θ_i)` and `s_obs` is *also* ≈ a noise draw from the same density, so `log p(s_rep_i|θ_i)` and
   `log p(s_obs|θ_i)` are draws of the same random variable → `δ_i` symmetric about 0 → `p ≈ 0.5`.

So `log_prob` compares data vs replicate *on equal footing*, scored by the flow's actual learned
density; the distance stats compare one fixed central point against the spread of a wider cloud.
That balance is why `log_prob` needs little calibration and why calibration barely moves it.

## Takeaways for interpretation

- A raw distance-stat p near 1 is the **null expectation**, not evidence of fit quality. Only the
  **calibrated p̃** is meaningful for those statistics.
- `log_prob` is close to a proper posterior-predictive density check, so its raw value is already
  roughly interpretable — though still *conservative* (concentrated near 0.5, not exactly uniform),
  because the data is used both to fit the posterior and to evaluate the test.
- In this non-Gaussian setting, lean on `log_prob` (and `kernel`) as **primary** discrepancies and
  treat the distance stats as **corroborating** — always via p̃. See the related note below.

## Related: is Mahalanobis even meaningful here? (no Gaussianity)

Separate but connected point. The Mahalanobis test stays **valid** without Gaussianity because the null
is the *empirical* bootstrap distribution of D² over PPD draws (plus Eq.-9 calibration) — not a χ²
table. Gaussianity affects **power** and **interpretability**, not validity:
- D² uses only the first two moments (μ, Σ), so it is blind to skew, heavy tails, and multimodality
  (a point in a zero-density "hole" between modes can have an ordinary D²).
- Do **not** read raw D² as χ²/"n-sigma" — that reading needs Gaussianity. Use p̃ only.
- The shape-aware statistics in the suite are `log_prob` (uses the flow's full density) and `kernel`
  (MMD-like). Mahalanobis is the fast, elliptical-if-Gaussian cross-check and the closest analog to
  Doux et al.'s χ² — but Doux earns the χ² interpretation via an analytic Gaussian likelihood with known
  covariance, which this SBI setting does not have (hence the empirical calibration).

## Code references
- `_pval_one_sample` (distance/kernel/Mahalanobis stats + bootstrap null) — `msi/utils/ppc.py` ~L855
- `_pval_log_prob` (paired matched-θ log-density comparison) — `msi/utils/ppc.py` ~L776
- `_sample_neural` (marginal PPD sampler: θ ~ posterior, then s_rep ~ flow) — `msi/utils/ppc.py` ~L559
- `run_calibration` (Doux Eq. 9 calibration of raw p → p̃) — `msi/utils/ppc.py` ~L1074

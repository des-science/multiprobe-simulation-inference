# Y6 v1 inference vs our Y3 v18: differences

Dev-phase note (2026-11-21). Repo docs must not point here.

msi runs the same code on both surveys. What it does depends on the run's saved configs: the msfm
config (`survey.name`, priors, fiducials) and the dlss config (the parameter vector). This list
records every place where a Y6 inference differs from the Y3 v18 one, so that no difference is
silent. Kinds in the table:

- **upstream**: follows from msfm or y3-deep-lss;
- **decision**: a stated user decision;
- **method**: the same quantity, derived another way.

Y3 side: `configs/runs/v18/default/prod.yaml`, `configs/flow/maf.yaml`, arXiv:2511.04681.
Y6 side: the msfm `configs/desy6/v1/default.yaml` as saved in each run's `configs.yaml`, and
`plan.md` (this folder). The forward-model differences are in msfm
`dev/notes/desy6/differences_to_y3.md`.

## Identical (the inference)

- Flow: `configs/flow/maf.yaml`, the `N_FLOWS=8` ensemble of `inference.sh`, conditioned on the
  extended vector (`extend_params: [ns, Ob, H0]`, looked up in the shared CosmoGridV1 metainfo).
- Sampling: torch-batched affine-invariant MCMC with the same walker, step and thinning settings.
- Priors: the msfm grid boxes. The `grid:` blocks of the two configs differ only in the bias
  names (`bg2` and the unsampled `rg` out, `bg5`, `bg6` in, all `[0.8, 3.0]`); cosmology, baryon,
  IA boxes and the Om-s8 border are the same.
- Coverage: wide-prior selection, 1000 held-out mocks, HPD/TARP/SBC/l-C2ST.
- Grid geometry: 2500 cosmologies × 80 examples (20 perms × 4 patches).
- DES chain variants: wCDM, lambdaCDM, w0 > -1, w0 > -1 + NLA, and for extended flows the
  `_refpriors` pair (D1, below).
- PPC and tension: the same `ppc.yaml` and `tension.yaml`; the Cls-space PPD is off in both.

## Differences

| # | item | Y3 v18 | Y6 v1 | kind | note |
|---|---|---|---|---|---|
| 1 | observation label | `DESy3`, `DESy3_no_sys`, ... (literal in msi) | `DESy6`, `DESy6_no_sys`, ..., from `survey.name` | upstream | y3-deep-lss `c4e353b` writes the keys; msi selects `<survey>` and `<survey>_*`. Chains are `chain_DESy6*.npy` |
| 2 | lens bias parameters | `bg1`-`bg4` | `bg1, bg3, bg4, bg5, bg6`, named by survey bin | upstream | msfm D3 / y3-deep-lss D5; the KP drops bin 2. Combined summary 11 parameters (Y3 10), clustering 8 (Y3 7) |
| 3 | key-project overlay | DES Y3 1x2pt/2x2pt/3x2pt chains via `with_des_chain` | none; `with_des_chain` raises | decision | the overlay maps `bg1-4` onto the Y3 chain's `b1-4`. Off in every production path on Y3 too |

## Shared on purpose

- **D1** (2026-11-21): the `_refpriors` chains use `REF_GAUSSIAN_PRIORS`, the near-delta
  ns (0.9649 ± 0.0063), Obh2 (0.02237 ± 0.00015) and H0 (70.22 ± 2.45) Gaussians of the DES Y3
  SBI papers (Jeffrey+24, Gatti+24, Williamson+26), on Y6 as on Y3. Extra chains; the main
  chain is unaffected.

## Not there yet

- The Y6 runs config (`configs/runs/desy6/v1/default/prod.yaml`), so no Y6 PPC or tension.
- Mock observations: no fiducial-family mocks (no fiducial projection) and no Buzzard-like
  validation mock, so no mock-contamination test and no recovery test. Both stages skip without
  failing.

## Plumbing (no effect on the inference)

- **D2** (2026-11-21): `--include_des` with no `<survey>` observation now logs a warning, on Y3
  too (before: nothing). The outputs are unchanged.
- `VERSION=desy6/v1` resolves `MSFM_CONFIG` and the runs-config path in `ppc.sh`/`tension.sh`.

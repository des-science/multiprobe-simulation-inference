# DES Y6 port of msi: plan

Dev-phase plan, started 2026-11-21 on branch `desy6`. Repo docs must not point here. The
upstream counterparts are msfm `dev/notes/desy6/plan.md` and y3-deep-lss
`dev/notes/desy6/plan.md`. The model differences to Y3 are in `differences_to_y3.md` (this
folder).

## Goal and rules

A Y6 run trained by y3-deep-lss on branch `desy6` gets its DES chains, plots, PPC and tension
from msi. y3-deep-lss writes the real-data predictions under msfm's `survey.name`
(`obs/preds/DESy6`, y3-deep-lss `c4e353b`; the `_no_sys`/`_no_psi_rot` test variants are opt-in
outside Y3 since `175a2cb`). Until this port it blocked its own
infer stage (`require_y3_inference`), because msi selected only `DESy3*`, so a Y6 inference
ended with no DES chain and no error.

Rules (the same as upstream):

- **Y3 stays bit-identical.** Every change falls back to the Y3 behaviour when `survey.name` is
  absent, and a Y3 regression proves it (Verification).
- **Nothing from Y3 enters Y6 unless stated.** Intended reuse is listed in `differences_to_y3.md`.
- **Every `sbatch` waits for the user's OK.**

## 1. What is survey-agnostic already (sweep of 2026-11-21)

- The parameters come from the run's own dlss config, and the msfm config comes from the run
  dir's `configs.yaml` (`run_inference.py`). A Y6 run carries `survey.name: DESy6` and its bias
  list into msi by itself.
- Bin counts come from `len(z_bins)` (`preprocessing.py`).
- The DES variant chains (lambdaCDM, w0 > -1, NLA, reference priors) and the per-chain plots key
  on `"des" in key.lower()` (`observations.py`), which matches `DESy6`.
- The hardcoded grid geometry (`(2500, 80, ...)` in `coverage_round.py`, `n_patches = 4` in
  `preprocessing.py`) has the same values in the Y6 config: 2500 cosmologies × 20 perms,
  4 patches, 5 noise realizations.
- `files.meta_info` (CosmoGridV1 metainfo, for `extend_params` and the wide-prior selection) is
  a shared file that the Y6 config lists explicitly.
- `inference.sh` is survey-agnostic (y3-deep-lss passes `RUNS_ROOT`). `ppc.sh` and `tension.sh`
  derive `MSFM_CONFIG` and `configs/runs/$VERSION/...` from `VERSION`, so `VERSION=desy6/v1`
  resolves both.

## 2. Work

1. **Observation label from `survey.name`** (done 2026-11-21).
   - `observations.survey_name(msfm_conf)`, which falls back to `DESy3` when the key is absent.
     All 15 Y3 v16/v17/v18 configs on msfm `main` name `DESy3`; `v17/fisher_bmode` has no
     survey block and falls back to it.
   - `get_des_observations(obs_pred_dict, survey)`, with the survey passed on from
     `collect_observations`. Finding no `<survey>` key logs a warning and continues (D2).
   - `run_inference.py`: `--flow_member_obs` defaults to `[survey]`. Before, `inference.sh`
     passed `--sample_flow_members` without the flag, so on Y6 that stage would have skipped
     with only a warning. `coverage.run_coverage` gets `obs_label=survey` for its l-C2ST.
   - `run_ppc.py` (Cls hook) and `ppc._build_cls_obs`: compare against the survey of the loaded
     msfm config. `cls_marginals` is off in `ppc.yaml`, so this is for consistency.
   - `run_fom_percentile.py`: `--obs_label` defaults to the runs config's first `obs_labels`
     entry (`DESy3` for `v18/default/prod.yaml`).
2. **Bias names `bg5`, `bg6`** (done 2026-11-21, D3).
   - `plotting.param_label_dict`: `bg5`, `bg6`. Without them every Y6 clustering/combined triangle
     plot raised a KeyError (`plot_chains`), and so did the tension plot (`tensions.shared_labels`).
   - `plot_chains(with_des_chain=True)` raises unless the survey is `DESy3`. The overlay loads the
     Y3 key-project chain with its `bg1-4` mapping (`chains.py`), so it must never be drawn over a
     Y6 posterior. No production path sets it.
3. **Docs** (done 2026-11-21). A DES Y3/Y6 paragraph in the README; `differences_to_y3.md`.
4. **Y6 runs config** `configs/runs/desy6/v1/default/prod.yaml`, once the first Y6 production
   runs exist: params `[..., bg1, bg3, bg4, bg5, bg6]`, `obs_labels: [{label: DESy6, lambdaCDM: false}]`,
   `scales_config` = y3-deep-lss `configs/scales/desy6/8wl,32gc.yaml`.

**Deliberately unchanged.**

- `coverage_round.py` `supported_params`: an experiment tool that already raises "Unsupported
  summary parameters" on Y6.
- `shifted_posterior.ASTRO_PARAMS_BY_PROBE` and `chains.py`: Y3 paper figures and the Y3
  key-project chain only.
- `preprocessing.py` `n_patches`/`n_noise`: the Y6 geometry is the same.
- `DEFAULT_MSFM_CONFIG` in `run_ppc.py`/`run_tension_chains.py`: the scripts always pass it.
- `common.sh` `STORE`: waits for the Globus move; y3-deep-lss passes the Y6 `RUNS_ROOT`.

**Hand-offs (not edited from msi).**

- y3-deep-lss: `require_y3_inference` (`submissions/clariden/common.sh`) can go once this branch
  is in. That is y3-deep-lss's step; work is ongoing there.
- msfm `dev/notes/desy6/plan.md` section 6: point the msi row here.

## 3. Decisions

- **D1** (2026-11-21): the `_refpriors` chains (near-delta ns/Obh2/H0 Gaussians of the DES Y3
  SBI papers, `REF_GAUSSIAN_PRIORS` in `observations.py`) also run on Y6. Extra chains; the main
  chain is unaffected. An intended reuse, listed in `differences_to_y3.md` and the README.
- **D2** (2026-11-21): `--include_des` with no `<survey>` observation warns and continues, as
  the mock path does for a missing label.
- **D3** (2026-11-21): bias names stay msfm's (`bg1, bg3-bg6`), as y3-deep-lss D5.

## 4. Verification

On the login node, `uenv run --view=default pytorch/v2.9.1:v2 -- ~/dlss/torch_env/bin/python`.
Read-only on every run.

1. **Y3 regression.**
   - The frozen v18 runs of `v18/default/prod.yaml`: old selector (literal `DESy3`) vs new
     (`survey.name` of the saved `configs.yaml`) give identical key lists; the survey resolves
     to `DESy3`.
   - `--flow_member_obs` resolves to `["DESy3"]`.
   - `python -m unittest discover tests`, before and after. The baseline at the branch point has
     one known error: `test_coverage_round_integration` reads msfm `configs/v18/default.yaml`,
     which msfm's `desy6` checkout moved to `configs/desy3/`.
2. **Y6 cases.**
   - A synthetic preds dict with `DESy6*` keys and the Y6 config selects every `DESy6*` key.
   - The same config with only `DESy3*` keys warns and selects nothing.
   - Labels resolve for `[Om, s8, w0, Aia, n_Aia, bta, bg1, bg3, bg4, bg5, bg6]`.
   - `with_des_chain=True` on the Y6 config raises.
3. **Y6 smoke inference**, after y3-deep-lss's Y6 smoke training (their verification step 4)
   writes a `preds_*.h5`. `inference.sh` on debug with `RUNS_ROOT` under `claude/`,
   `N_FLOWS=2`. Expected: `chain_DESy6*.npy` (wCDM and its variants), `chain_DESy6_flow_{m}.npy`,
   triangle plots with `bg5`/`bg6`, l-C2ST on `DESy6`. The smoke run carries only `DESy6`.
   **Passed 2026-11-21** (job 5416298, debug, 14:27, on y3-deep-lss job 5416191's
   `claude/bench/desy6_deep_lss_smoke/maps_gcnn_nocls/combined/smoke`): all 8 `chain_DESy6*` files
   (wCDM 14 columns = 11 + ns/Ob/H0, the variants one fewer per fixed parameter, both
   `_refpriors`), both member chains, all finite; grid contours label `bg5`/`bg6`; l-C2ST ran on
   `DESy6`; mock contamination skipped (no mocks). Flow config
   `claude/configs/msi_desy6_smoke/maf_smoke.yaml` = `maf.yaml` with `batch_size 256` and
   `thin 1`: `maf.yaml`'s batch of 10000 exceeds the 1680 smoke training rows, which gives 0 steps
   per epoch and a NaN loss (job 5416269). Production has ~200k rows.
   Seen in passing, not Y6: with `extend_params` the triangle's parameter groups are cut by
   count (`[cosmo, ia, bg]` sizes) while ns/Ob/H0 sit at the end, so the boxes split 6 | 3 | 5
   instead of by kind. Cosmetic, on Y3 too.

## 5. Later

- PPC and tension on Y6 need the Y6 runs config (item 4) and at least two Y6 runs.
- Y6 key-project chains as an overlay (public on the DES data server, not wired in).
- Mocks: the benchmark projections exist, the msfm mocks from them are not built yet, and there is
  no Y6 validation mock.

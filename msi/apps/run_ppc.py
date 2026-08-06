"""
Posterior predictive checks (PyTorch / torch_env), packaged from
``deep_lss_paper/paper_2/pre-unblinding/4_posterior_predictive_checks.ipynb``.

For the runs and comparisons defined in a runs config (e.g. ``configs/runs/v17/baseline/t2_v3.yaml``), this
script runs two families of posterior predictive checks via
``msi.utils.ppc.PosteriorPredictiveChecks``:

* **auto** (single-probe goodness-of-fit): for each run, ``rep == obs`` -- is the observed summary
  a typical draw from its own posterior predictive ``p(s | s_obs)``?
* **cross** (cross-probe consistency, Doux et al. 2020): for each run pair, both directions of
  ``p(s_rep | theta_obs, s_obs)`` -- does one probe's posterior predict the other probe's data,
  respecting their data-level correlation?

All plots and trained PPC flows are written by the class under ``{flow_dir}/ppc/...`` (co-located
with the model), analogous to the tension outputs. A single environment is needed: the PPC trains a
torch/enflows ``LikelihoodFlow`` and only loads (backend-agnostic) saved inference chains.
"""

import argparse
import os

from msfm.utils import files as msfm_files
from msfm.utils import logger
from msfm.utils.input_output import read_yaml
from msi.utils import tensions
from msi.utils.ppc import PosteriorPredictiveChecks

LOGGER = logger.get_logger(__file__)

DEFAULT_MSFM_CONFIG = "/users/athomsen/dlss/repos/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"


def setup():
    parser = argparse.ArgumentParser(description="Posterior predictive checks (torch).")
    parser.add_argument("--runs_config", required=True, help="YAML defining runs, comparisons and obs_labels.")
    parser.add_argument("--ppc_config", required=True, help="YAML defining method hyperparameters.")
    parser.add_argument(
        "--msfm_config",
        default=DEFAULT_MSFM_CONFIG,
        help="msfm forward-model config providing parameter definitions / priors.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--retrain_flows",
        action="store_true",
        help="Force retraining the PPC flows from scratch. By default each flow is recovered from its "
        "checkpoint when one exists, and only trained when none is found.",
    )
    return parser.parse_args()


def iter_runs(runs_conf):
    """Yield each run dict augmented with its ``data``, ``probe`` and ``flow_name`` (mirrors
    ``tensions._run``), iterating ``runs[data_representation][probe]``."""
    default_flow_name = runs_conf.get("flow_name", "likelihood_flow")
    for data, probes in runs_conf["runs"].items():
        if not probes:  # allow `cls: Null` (or empty) to drop a whole data representation
            continue
        for probe, entry in probes.items():
            run = dict(entry)
            run["data"] = data
            run["probe"] = probe
            run.setdefault("flow_name", default_flow_name)
            yield run


def pred_file(run):
    return os.path.join(run["pred_dir"], f"preds_{run['steps']}.h5")


def flow_dir(run):
    return os.path.join(run["pred_dir"], f"{run['flow_name']}_{run['steps']}")


def make_ppc(conf_path, cosmo_params, seed, run_1, run_2=None, shared_data=False, flow_conf=None, n_flows=1):
    """Construct a PosteriorPredictiveChecks for one run (auto) or a run pair (cross)."""
    kwargs = dict(
        conf=conf_path,
        cosmo_params=cosmo_params,
        seed=seed,
        probe1_name=run_1["probe"],
        probe1_data=run_1["data"],
        probe1_pred_file=pred_file(run_1),
        probe1_flow_dir=flow_dir(run_1),
        flow_conf=flow_conf,
        n_flows=n_flows,
    )
    if run_2 is not None:
        kwargs.update(
            probe2_name=run_2["probe"],
            probe2_data=run_2["data"],
            probe2_pred_file=pred_file(run_2),
            probe2_flow_dir=flow_dir(run_2),
            shared_data=shared_data,
        )
    return PosteriorPredictiveChecks(**kwargs)


def resolve_rebinned_cls_cache(cls_conf, scales_config):
    """Resolve everything the Cls-space PPD needs: ``(base_dir, cls_n_bins, scales_name, dlss_conf)``.

    The scale cut is a property of the RUNS (the network was trained with it), so ``scales_config``
    (the path to the scales YAML, e.g. ``configs/scales/8wl,32gc.yaml``) comes from the runs config.
    Its stem gives ``scales_name`` and its ``scale_cuts`` block defines the obs rebinning / ell axis.
    ``base_dir`` (data location) comes from ``cls_conf`` (the ppc config); ``cls_n_bins`` is read from
    the matching cache's HDF5 attr (authoritative), with an optional ``cls_conf['cls_n_bins']`` to
    disambiguate if several bin counts exist for the same scales.
    """
    import glob
    import h5py

    base_dir = cls_conf["base_dir"]
    scales_name = os.path.splitext(os.path.basename(scales_config))[0]

    scales_yaml = read_yaml(scales_config)
    assert "scale_cuts" in scales_yaml, f"{scales_config} has no 'scale_cuts' block."
    dlss_conf = {"scale_cuts": scales_yaml["scale_cuts"]}

    cls_dir = os.path.join(base_dir, "cls")
    pattern = os.path.join(cls_dir, f"rebinned_nb*_{scales_name}.h5")
    caches = sorted(glob.glob(pattern))
    if not caches:
        raise FileNotFoundError(f"no rebinned Cls cache matching {pattern}; build it first (cls precache).")

    want_nb = cls_conf.get("cls_n_bins")
    candidates = []
    for path in caches:
        with h5py.File(path, "r") as f:
            nb, sn = int(f.attrs["cls_n_bins"]), str(f.attrs["scales_name"])
        assert sn == scales_name, f"cache {path} has scales_name attr {sn!r} != {scales_name!r}"
        if want_nb is None or nb == want_nb:
            candidates.append((path, nb))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one rebinned Cls cache matching {pattern} (got {len(candidates)}: "
            f"{[c[0] for c in candidates]}); set cls_marginals.cls_n_bins to disambiguate."
        )
    cache_path, cls_n_bins = candidates[0]

    LOGGER.info(
        f"Cls PPD config: cache={cache_path}, scales_name={scales_name} (from {scales_config}), "
        f"cls_n_bins={cls_n_bins}"
    )
    return base_dir, cls_n_bins, scales_name, dlss_conf


def main():
    args = setup()

    runs_conf = read_yaml(args.runs_config)
    ppc_conf = read_yaml(args.ppc_config)
    # validate the msfm config path early (the class loads it again per PPC instance)
    msfm_files.load_config(args.msfm_config)

    cosmo_params = ppc_conf.get("cosmo_params", ["Om", "s8", "w0"])
    seed = ppc_conf.get("seed", 111)
    independent_cross = ppc_conf.get("independent_cross", False)
    cross_pairs = ppc_conf.get("cross_pairs", None)
    cross_same_probe = ppc_conf.get("cross_same_probe", True)
    n_flows = ppc_conf.get("n_flows", 1)
    flow_conf = read_yaml(ppc_conf["flow_config"]) if ppc_conf.get("flow_config") else {}
    flow_fit = ppc_conf.get("flow_fit", {}) or {}  # optional per-call training overrides
    sampling = ppc_conf.get("sampling", {})
    checks = ppc_conf.get("checks", {})
    cls_conf = ppc_conf.get("cls_marginals", {}) or {}
    calib_conf = ppc_conf.get("calibration", {}) or {}

    skip_probes = set(ppc_conf.get("skip_probes", []) or [])

    # LambdaCDM observations are not yet supported: a lambdaCDM inference chain drops w0, so theta_post
    # would no longer match the PPC flow's context dimensionality (built from the full-parameter grid).
    # Until that is handled end-to-end, fail loudly rather than silently loading the (full-w0)
    # `chain_{label}.npy` for a `lambdaCDM: true` entry.
    obs_entries = tensions.iter_observations(runs_conf)
    lcdm = [label for label, use_lambdaCDM in obs_entries if use_lambdaCDM]
    assert not lcdm, (
        f"obs_labels with lambdaCDM=true are not supported by run_ppc.py yet: {lcdm}. The lambdaCDM "
        "chain drops w0 and would mismatch the PPC flow context; remove the flag or extend the PPC "
        "to load chain_{label}_lambdaCDM.npy and a w0-marginalised context."
    )
    observations = [label for label, _ in obs_entries]
    LOGGER.info(f"observations: {observations}")
    if skip_probes:
        LOGGER.info(f"skipping probes: {sorted(skip_probes)}")

    def run_all_checks(ppc, post_hook=None):
        for obs_label in observations:
            LOGGER.info(f"run_checks: obs_label={obs_label}")
            ppc.run_checks(obs_label=obs_label, **sampling, **checks)
            if post_hook is not None:
                post_hook(ppc, obs_label)

    # ---- Doux Eq. 9 calibration: shared setup for auto and cross --------------------------------
    # Needs the obs probe's mcmc_samples.h5 (auto + cross) and, for cross, its `real_idx` dataset to
    # pair the obs mock with the rep-probe summary. The coverage stage (run_inference --sample_posterior)
    # writes real_idx for standard single-file runs; cross calibration auto-skips (run_calibration warns
    # and returns None) for runs without it, so no separate gate is needed.
    calib_enabled = calib_conf.get("enabled", False)
    calib_kwargs = {k: calib_conf[k] for k in ("n_sim", "n_samples_neural", "n_bootstrap", "n_ref") if k in calib_conf}
    # Calibrate exactly the statistics enabled in `checks` (single source of truth), so e.g.
    # check_l1: false drops L1 from the null loop too. Defaults mirror run_checks (all on) when absent.
    _stat_flag = {
        "log_prob": "check_log_prob",
        "mahalanobis": "check_mahalanobis",
        "l2": "check_l2",
        "l1": "check_l1",
        "linf": "check_linf",
        "kernel": "check_kernel",
    }
    calib_kwargs["stats"] = [s for s, f in _stat_flag.items() if checks.get(f, True)]

    def calib_hook(ppc, obs_label, _kwargs=calib_kwargs):
        ppc.run_calibration(**_kwargs)

    # ---- Cls-space posterior predictive (rebinned): resolve the shared cache config once --------
    # Runs for both maps runs (project the maps posterior into rebinned-Cls space) and cls runs
    # (their native data vector). The hard_rebinned cache is probe/data independent, so one
    # resolution serves every run; disabled with a warning if no cache is found.
    cls_marginals_cfg = None
    if cls_conf.get("enabled", False):
        scales_config = runs_conf.get("scales_config")
        if scales_config is None:
            LOGGER.warning(
                "Cls-space PPD disabled: the runs config has no 'scales_config' (the scale cut the "
                "network was trained with, e.g. configs/scales/8wl,32gc.yaml)."
            )
        else:
            try:
                cls_marginals_cfg = resolve_rebinned_cls_cache(cls_conf, scales_config)
            except (FileNotFoundError, ValueError, AssertionError, KeyError) as e:
                LOGGER.warning(f"Cls-space PPD disabled: {e}")

    # ---- auto: single-probe goodness-of-fit -----------------------------------------------------
    if ppc_conf.get("run_auto", True):
        for run in iter_runs(runs_conf):
            if run["probe"] in skip_probes:
                continue
            ident = tensions.get_identifier(run["pred_dir"])
            LOGGER.info(f"=== auto: {ident} ===")
            ppc = make_ppc(args.msfm_config, cosmo_params, seed, run, flow_conf=flow_conf, n_flows=n_flows)
            ppc.setup_flow(
                rep_probe=ppc.probe1_label,
                obs_probe=ppc.probe1_label,
                retrain=args.retrain_flows,
                fit_kwargs=flow_fit,
            )

            # post-check hooks (auto-probe), each run right after the per-observation state is set
            # by run_checks (so no checks are repeated).
            hooks = []

            # optional rebinned-Cls-space PPD (DESy3 only). The diagnostic importance-samples the
            # grid with this run's flow and shows the corresponding hard_rebinned Cls data vector --
            # for maps runs it projects the maps posterior into Cls space; for cls runs it is the
            # native data vector. Config comes from the shared cache (resolved above), so it does not
            # depend on the run's own configs.yaml.
            if cls_marginals_cfg is not None:
                _cls_sample_set = cls_conf.get("sample_set", "test")
                _cls_k_top = cls_conf.get("k_top", None)

                def cls_hook(ppc, obs_label, _cfg=cls_marginals_cfg, _sample_set=_cls_sample_set, _k_top=_cls_k_top):
                    if obs_label != "DESy3":
                        return
                    base_dir, cls_n_bins, scales_name, dlss_conf = _cfg
                    ppc.check_cls_marginals(
                        dlss_conf=dlss_conf,
                        base_dir=base_dir,
                        cls_n_bins=cls_n_bins,
                        scales_name=scales_name,
                        sample_set=_sample_set,
                        k_top=_k_top,
                    )

                hooks.append(cls_hook)

            # optional Doux Eq. 9 p-value calibration (needs the run's mcmc_samples.h5)
            if calib_enabled:
                hooks.append(calib_hook)

            def post_hook(ppc, obs_label, _hooks=hooks):
                for h in _hooks:
                    h(ppc, obs_label)

            run_all_checks(ppc, post_hook if hooks else None)

    # ---- cross: cross-probe consistency (both directions) ---------------------------------------
    if ppc_conf.get("run_cross", True):
        for designation, run_1, run_2 in tensions.build_combinations(runs_conf):
            if run_1["probe"] in skip_probes or run_2["probe"] in skip_probes:
                continue
            same_probe = run_1["probe"] == run_2["probe"]
            # The same-probe (maps-vs-cls) shared-data family is gated by `cross_same_probe`;
            # the disjoint-probe family is filtered by `cross_pairs`.
            if same_probe and not cross_same_probe:
                LOGGER.info(f"skipping same-probe (maps-vs-cls) cross {designation} (cross_same_probe=false)")
                continue
            if (not same_probe) and (cross_pairs is not None):
                pair = {run_1["probe"], run_2["probe"]}
                if not any(pair == set(cp) for cp in cross_pairs):
                    LOGGER.info(f"skipping cross pair {designation} (not in cross_pairs)")
                    continue

            LOGGER.info(f"=== cross: {designation} (shared_data={same_probe}) ===")
            ppc = make_ppc(
                args.msfm_config,
                cosmo_params,
                seed,
                run_1,
                run_2,
                shared_data=same_probe,
                flow_conf=flow_conf,
                n_flows=n_flows,
            )

            # Calibrate the cross p-values too (both directions); auto-skips when real_idx is missing.
            cross_post = calib_hook if calib_enabled else None

            for rep_probe, obs_probe in (
                (ppc.probe2_label, ppc.probe1_label),
                (ppc.probe1_label, ppc.probe2_label),
            ):
                ppc.setup_flow(
                    rep_probe=rep_probe,
                    obs_probe=obs_probe,
                    independent_cross=independent_cross,
                    retrain=args.retrain_flows,
                    fit_kwargs=flow_fit,
                )
                run_all_checks(ppc, cross_post)


if __name__ == "__main__":
    main()

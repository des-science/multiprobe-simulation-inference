"""
Posterior predictive checks (PyTorch / torch_env), packaged from
``deep_lss_paper/paper_2/pre-unblinding/4_posterior_predictive_checks.ipynb``.

For the runs and comparisons defined in a runs config (e.g. ``configs/runs/v8_v33.yaml``), this
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

DEFAULT_MSFM_CONFIG = (
    "/users/athomsen/dlss/repos/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
)


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
    return parser.parse_args()


def iter_runs(runs_conf):
    """Yield each run dict augmented with its ``data``, ``probe`` and ``flow_name`` (mirrors
    ``tensions._run``), iterating ``runs[data_representation][probe]``."""
    default_flow_name = runs_conf.get("flow_name", "likelihood_flow")
    for data, probes in runs_conf["runs"].items():
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


def main():
    args = setup()

    runs_conf = read_yaml(args.runs_config)
    ppc_conf = read_yaml(args.ppc_config)
    # validate the msfm config path early (the class loads it again per PPC instance)
    msfm_files.load_config(args.msfm_config)

    cosmo_params = ppc_conf.get("cosmo_params", ["Om", "s8", "w0"])
    seed = ppc_conf.get("seed", 111)
    train_flow = ppc_conf.get("train_flow", True)
    independent_cross = ppc_conf.get("independent_cross", False)
    cross_pairs = ppc_conf.get("cross_pairs", None)
    n_flows = ppc_conf.get("n_flows", 1)
    flow_conf = read_yaml(ppc_conf["flow_config"]) if ppc_conf.get("flow_config") else {}
    flow_fit = ppc_conf.get("flow_fit", {}) or {}  # optional per-call training overrides
    sampling = ppc_conf.get("sampling", {})
    checks = ppc_conf.get("checks", {})
    cls_conf = ppc_conf.get("cls_marginals", {}) or {}

    skip_probes = set(ppc_conf.get("skip_probes", []) or [])

    observations = [label for label, _ in tensions.iter_observations(runs_conf)]
    LOGGER.info(f"observations: {observations}")
    if skip_probes:
        LOGGER.info(f"skipping probes: {sorted(skip_probes)}")

    def run_all_checks(ppc, post_hook=None):
        for obs_label in observations:
            LOGGER.info(f"run_checks: obs_label={obs_label}")
            ppc.run_checks(obs_label=obs_label, **sampling, **checks)
            if post_hook is not None:
                post_hook(ppc, obs_label)

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
                train_flow=train_flow,
                fit_kwargs=flow_fit,
            )

            # optional Cls-space PPD (auto-probe + DESy3 only); runs right after the DESy3
            # observation state is set by run_checks, so no checks are repeated.
            cls_hook = None
            run_configs = os.path.join(run["pred_dir"], "configs.yaml")
            if cls_conf.get("enabled", False) and os.path.exists(run_configs):
                def cls_hook(ppc, obs_label, _run_configs=run_configs):
                    if obs_label != "DESy3":
                        return
                    from deep_lss.utils import configuration

                    # Derive the dlss_conf (dset + scale_cuts) from the run's own saved
                    # configs.yaml, so the Cls PPD uses the exact scale cuts the network was
                    # trained with (mirrors run_inference's config loading).
                    dlss_conf = configuration.load_run_configs(_run_configs)["dlss"]
                    ppc.check_cls_marginals(
                        dlss_conf=dlss_conf,
                        base_dir=cls_conf["base_dir"],
                    )

            run_all_checks(ppc, cls_hook)

    # ---- cross: cross-probe consistency (both directions) ---------------------------------------
    if ppc_conf.get("run_cross", True):
        for designation, run_1, run_2 in tensions.build_combinations(runs_conf):
            if run_1["probe"] in skip_probes or run_2["probe"] in skip_probes:
                continue
            same_probe = run_1["probe"] == run_2["probe"]
            # cross_pairs filters only the disjoint-probe family; the same-probe (maps-vs-cls)
            # shared-data family is meaningful for every probe and is never filtered.
            if (not same_probe) and (cross_pairs is not None):
                pair = {run_1["probe"], run_2["probe"]}
                if not any(pair == set(cp) for cp in cross_pairs):
                    LOGGER.info(f"skipping cross pair {designation} (not in cross_pairs)")
                    continue

            LOGGER.info(f"=== cross: {designation} (shared_data={same_probe}) ===")
            ppc = make_ppc(
                args.msfm_config, cosmo_params, seed, run_1, run_2,
                shared_data=same_probe, flow_conf=flow_conf, n_flows=n_flows,
            )

            for rep_probe, obs_probe in (
                (ppc.probe2_label, ppc.probe1_label),
                (ppc.probe1_label, ppc.probe2_label),
            ):
                ppc.setup_flow(
                    rep_probe=rep_probe,
                    obs_probe=obs_probe,
                    independent_cross=independent_cross,
                    train_flow=train_flow,
                    fit_kwargs=flow_fit,
                )
                run_all_checks(ppc)


if __name__ == "__main__":
    main()

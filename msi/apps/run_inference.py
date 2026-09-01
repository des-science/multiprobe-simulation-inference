import argparse
import os

import yaml

from msfm.utils import files as msfm_files
from msfm.utils import logger
from msfm.utils.input_output import read_yaml
from deep_lss.utils import configuration
from msi.flow_conductor.likelihood_flow import LikelihoodFlow, LikelihoodFlowEnsemble
from msi.utils import extended_params
from msi.utils import flow as flow_utils
from msi.utils import observations
from msi.utils import coverage
from msi.utils import prior_predictive
from msi.utils import mock_contamination

LOGGER = logger.get_logger(__file__)


def _load_configs(pred_dir, msfm_config_path, dlss_config_path):
    """Load msfm_conf and dlss_conf from either explicit paths or pred_dir/configs.yaml.

    configs.yaml is either the new single-doc nested format ({net|mlp, dlss, loss, data,
    msfm, run}) or a legacy multi-doc stream ending in [..., dlss_conf, msfm_conf] (3-doc
    maps or 4-doc Cls format).
    """
    if msfm_config_path and dlss_config_path:
        msfm_conf = msfm_files.load_config(msfm_config_path)
        dlss_conf = read_yaml(dlss_config_path)
    else:
        configs_path = os.path.join(pred_dir, "configs.yaml")
        try:
            conf = configuration.load_run_configs(configs_path)
            dlss_conf, msfm_conf = conf["dlss"], conf["msfm"]
        except ValueError:
            # Fallback for legacy 4-document Cls streams
            with open(configs_path) as f:
                docs = list(yaml.load_all(f, Loader=yaml.FullLoader))
            dlss_conf, msfm_conf = docs[-2], docs[-1]
    return dlss_conf, msfm_conf


def _dump_flow_config(model_dir, flow_conf, source):
    """Write a single re-loadable flow_config.yaml into model_dir."""
    if not model_dir:
        LOGGER.warning("flow has no model_dir; skipping flow config copy.")
        return
    out_file = os.path.join(model_dir, "flow_config.yaml")
    try:
        with open(out_file, "w") as f:
            f.write(f"# flow config for this run (source: {source or 'hardcoded defaults'})\n")
            yaml.safe_dump(flow_conf, f, default_flow_style=False, sort_keys=False)
        LOGGER.info(f"Saved flow config to {out_file}")
    except Exception as e:
        LOGGER.warning(f"Could not save flow config ({type(e).__name__}: {e})")


def _save_flow_config(flow, flow_conf, flow_config_path, flow_confs=None, flow_config_paths=None):
    """Write a re-loadable copy of the flow config(s) for run comparison.

    For a single flow / homogeneous ensemble, dump flow_conf into the (ensemble) model_dir. For a
    heterogeneous ensemble (flow_confs given), additionally dump each member's own config into its
    flow_i/ subdirectory so the run is self-describing per member. CLI choices like n_flows/flow_label/
    n_steps are already encoded in the directory name, so only the flow config(s) need saving here.
    """
    _dump_flow_config(getattr(flow, "model_dir", None), flow_conf, flow_config_path)

    if flow_confs is not None and hasattr(flow, "flows"):
        n_configs = len(flow_confs)
        for i, member in enumerate(flow.flows):
            src = flow_config_paths[i % n_configs] if flow_config_paths else None
            _dump_flow_config(getattr(member, "model_dir", None), flow_confs[i % n_configs], src)


def setup():
    parser = argparse.ArgumentParser(
        description="Normalizing flow inference on network summary statistics (maps or Cls)."
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="model")
    parser.add_argument(
        "--out_dir_2",
        default=None,
        help="Optional second model's out_dir; its summary is concatenated feature-wise with the "
        "primary model's summary, e.g. to combine a maps-level and a Cls-level model.",
    )
    parser.add_argument("--model_name_2", default="model")
    parser.add_argument("--n_steps_2", type=int, default=None)
    # Optional explicit config overrides (Cls path); falls back to pred_dir/configs.yaml
    parser.add_argument("--msfm_config", default=None)
    parser.add_argument("--dlss_config", default=None)
    parser.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="Prediction file step count; auto-detects the largest preds_*.h5 if omitted.",
    )
    parser.add_argument(
        "--n_steps_multi",
        nargs="+",
        type=int,
        default=None,
        help="Combine predictions from these specific training-step counts (feature-wise concatenation).",
    )
    parser.add_argument(
        "--n_steps_all",
        action="store_true",
        help="Combine predictions from ALL preds_*.h5 files found in the model directory.",
    )
    parser.add_argument(
        "--pca_compress",
        action="store_true",
        help="After concatenating multi-step summaries, apply PCA to compress back to single-run dimensionality.",
    )
    parser.add_argument(
        "--flow_config",
        default=None,
        help="Path to flow YAML config; uses hardcoded defaults if omitted.",
    )
    parser.add_argument(
        "--flow_configs",
        nargs="+",
        default=None,
        help="Paths to multiple flow YAML configs for a HETEROGENEOUS LikelihoodFlowEnsemble: one member "
        "per listed config (mixing e.g. sigmoid/spline/maf architectures gives more diversity than "
        "seed-only clones). --n_flows then acts as a per-config replication factor (total members = "
        "n_configs * n_flows). Mutually exclusive with --flow_config. The first config supplies the "
        "shared mcmc/diagnostics settings.",
    )
    parser.add_argument(
        "--load_flow",
        action="store_true",
        help="Load existing flow checkpoint instead of training a new one.",
    )
    parser.add_argument(
        "--n_flows",
        type=int,
        default=1,
        help="Train/load a LikelihoodFlowEnsemble with this many independently-initialized flows "
        "instead of a single LikelihoodFlow. 1 = current behavior. Must match the trained "
        "checkpoint when used with --load_flow.",
    )
    parser.add_argument(
        "--flow_label",
        default="",
        help="Prefix for the flow checkpoint directory, e.g. 'larger' saves to "
        "pred_dir/larger_likelihood_flow_{n_steps}/. Useful when comparing multiple "
        "flow configs on the same prediction file.",
    )
    parser.add_argument(
        "--extend_params",
        nargs="*",
        default=None,
        help="Condition the flow on an EXTENDED parameter vector: append these CosmoGrid grid parameters "
        "(recorded per grid cosmology in the metainfo, looked up via i_sobol -- no summary recomputation) "
        "to the training params, e.g. 'ns Ob H0 bary_Mc bary_nu' (also the default when the flag is given "
        "without values). The otherwise implicit wide flat marginalization over these parameters then "
        "becomes explicit and controllable at MCMC time: DES observations automatically get additional "
        "reference-prior chains (near-delta ns/Obh2/H0 Gaussians + fixed baryons, the Gower-Street-family "
        "analysis choices of the DES Y3 SBI papers; see msi.utils.observations). Unless --flow_label is "
        "given, it defaults to 'ext' so the extended flow does not clobber the baseline checkpoint.",
    )
    parser.add_argument(
        "--mcmc_backend",
        choices=("emcee", "torch_batched"),
        default="torch_batched",
        help="MCMC sampler backend. 'emcee' is the established per-observation CPU-loop sampler. "
        "'torch_batched' (default) uses the GPU-batched affine-invariant ensemble sampler "
        "(msi.utils.torch_ensemble), which samples all observations in one batched forward pass -- "
        "much faster on a GPU when there are several observations. Supported for both a single "
        "LikelihoodFlow and a LikelihoodFlowEnsemble (--n_flows>1).",
    )
    parser.add_argument(
        "--sample_posterior",
        action="store_true",
        help="Additionally run a posterior-level coverage test: sample the posterior for the held-out "
        "mock observations (count from the flow config's diagnostics.n_obs) in a single GPU-batched "
        "pass and write mcmc_samples.h5 for TARP. Requires --mcmc_backend=torch_batched (works for both "
        "a single LikelihoodFlow and an ensemble); otherwise warns and skips.",
    )
    parser.add_argument(
        "--sample_flow_members",
        action="store_true",
        help="Additionally sample each ensemble member's OWN posterior for the DES observation(s), saved as "
        "chain_DESy3_flow_{m}.npy next to the ensemble chain. This is the ensemble-convergence / "
        "out-of-distribution pre-unblinding test: the members agree on data from the training distribution, "
        "so disagreement on the real data means the summary sits outside it. The ensemble chains are NOT "
        "affected -- unlike the mcmc.method='individual' + mcmc.store_individual_chains config pair, which "
        "also replaces every production chain with a pooled-mixture one. Needs --n_flows>1, --include_des "
        "and --mcmc_backend=torch_batched; otherwise warns and skips.",
    )
    parser.add_argument(
        "--flow_member_obs",
        nargs="+",
        default=["DESy3"],
        help="Observation labels for --sample_flow_members. Defaults to the primary DESy3 data vector; the "
        "systematics variants (DESy3_no_sys, DESy3_no_psi_rot, ...) are separate observations and have to "
        "be named to be included.",
    )
    observations.add_obs_args(parser)
    return parser.parse_args()


def main():
    args = setup()

    is_multi = args.n_steps_multi is not None or args.n_steps_all
    if args.n_steps_multi is not None and args.n_steps_all:
        raise ValueError("--n_steps_multi and --n_steps_all are mutually exclusive.")

    LOGGER.timer.start("flow")

    if args.flow_config and args.flow_configs:
        raise ValueError("--flow_config and --flow_configs are mutually exclusive.")

    # Heterogeneous ensemble: one member per config in --flow_configs. The first config supplies the
    # shared mcmc/diagnostics/coverage settings used by the rest of main(); flow_confs drives the
    # per-member architecture/training in build_flow.
    flow_confs = [read_yaml(p) for p in args.flow_configs] if args.flow_configs else None
    is_hetero = flow_confs is not None
    flow_conf = flow_confs[0] if is_hetero else (read_yaml(args.flow_config) if args.flow_config else {})

    # an ensemble is used for >1 seed-clone members or for any heterogeneous config list
    is_ensemble = args.n_flows > 1 or is_hetero

    # extended conditioning vector: [] (flag without values) means the default extension set. Default the
    # flow label so the extended flow trains/saves alongside -- not over -- the baseline checkpoint.
    if args.extend_params is not None and len(args.extend_params) == 0:
        args.extend_params = list(extended_params.DEFAULT_EXTEND_PARAMS)
    if args.extend_params and not args.flow_label:
        args.flow_label = "ext"
        print("--extend_params: defaulting --flow_label to 'ext'")

    prefix = f"{args.flow_label}_" if args.flow_label else ""

    # Load the held-out grid/observation summaries and resolve the checkpoint label. The two paths
    # differ only in how summaries are loaded and how the checkpoint suffix is built; the tail
    # (config loading + load/build of the flow) is shared.
    if is_multi:
        pred_dir = os.path.join(args.out_dir, args.model_name)
        if args.n_steps_all:
            steps_list = flow_utils.find_all_n_steps(pred_dir)
            if not steps_list:
                raise FileNotFoundError(f"No preds_*.h5 found in {pred_dir}")
            print(f"Using all steps: {steps_list}")
        else:
            steps_list = sorted(args.n_steps_multi)
        pred_files = [os.path.join(pred_dir, f"preds_{s}.h5") for s in steps_list]

        grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal, i_sobol = (
            flow_utils.load_grid_summaries_multi(pred_files, pca_compress=args.pca_compress)
        )
        # multi-checkpoint / PCA grids are reordered+concatenated, so a single row-aligned i_noise is not
        # available; cross-probe PPC calibration is unsupported for these (real_idx simply not written).
        i_noise = None

        steps_str = "_".join(str(s) for s in steps_list)
        n_steps_label = f"multi_{steps_str}" + ("_pca" if args.pca_compress else "")
        suffix = f"_{n_steps_label}"
    else:
        pred_dir, pred_file, n_steps = flow_utils.resolve_pred_file(args.out_dir, args.model_name, args.n_steps)

        pred_file_2 = None
        if args.out_dir_2:
            _, pred_file_2, _ = flow_utils.resolve_pred_file(args.out_dir_2, args.model_name_2, args.n_steps_2)

        grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal, i_sobol = flow_utils.load_grid_summaries(
            pred_file, pred_file_2
        )
        # Per-row i_noise (single-file path leaves grid rows in native order, so the pred-file index is
        # row-aligned). Threaded into the coverage stage so mcmc_samples.h5 carries the full realization
        # triplet for cross-probe PPC calibration. Unavailable when combining two pred files.
        i_noise = None if pred_file_2 else flow_utils.load_grid_indices(pred_file)[2]

        n_steps_label = n_steps
        suffix = f"_{n_steps}" if n_steps is not None else ""

    dlss_conf, msfm_conf = _load_configs(pred_dir, args.msfm_config, args.dlss_config)
    params = dlss_conf["dset"]["training"]["params"]

    if args.extend_params:
        # Append the recorded grid values of the extension parameters to every grid row (via i_sobol) and
        # to the observation truth points; the flow below then conditions on the extended vector.
        extend = [p for p in args.extend_params if p not in params]
        table = extended_params.load_grid_param_table(msfm_conf)
        grid_cosmos = extended_params.extend_grid_cosmos(grid_cosmos, i_sobol, extend, msfm_conf, table=table)
        obs_cosmo_dict = extended_params.extend_obs_cosmo_dict(obs_cosmo_dict, params, extend, msfm_conf, table=table)
        params = params + extend
        print(f"Extended conditioning vector: {params}")

    if args.load_flow:
        print("Loading flow from checkpoint...")
        flow_cls = LikelihoodFlowEnsemble if is_ensemble else LikelihoodFlow
        flow = flow_cls.from_checkpoint(out_dir=pred_dir, prefix=prefix, suffix=suffix)
        if list(flow.params) != list(params):
            raise ValueError(
                f"Loaded flow was trained on params {flow.params} but the current run expects {params}; "
                "make sure --extend_params (and its values) match the trained checkpoint."
            )
    else:
        flow = flow_utils.build_flow(
            params,
            msfm_conf,
            pred_dir,
            n_steps_label,
            grid_preds,
            grid_cosmos,
            flow_conf,
            prefix=prefix,
            i_signal=i_signal,
            n_flows=args.n_flows,
            flow_confs=flow_confs,
        )

    LOGGER.info(f"[timing] flow {'loaded' if args.load_flow else 'trained'}: {LOGGER.timer.elapsed('flow')}")

    _save_flow_config(flow, flow_conf, args.flow_config, flow_confs=flow_confs, flow_config_paths=args.flow_configs)

    mcmc_conf = flow_conf.get("mcmc", {})
    # ensemble member weighting: tempered softmax of negative validation losses. Set before any
    # sampling so the deep _get_ensemble_weights -> _compute_validation_weights calls pick it up.
    # No-op for a single LikelihoodFlow (attribute simply unused).
    if hasattr(flow, "validation_weight_temperature"):
        flow.validation_weight_temperature = mcmc_conf.get("validation_weight_temperature", 1.0)

    # prior-level visualization (cheap, no MCMC): corner plot of the held-out grid summaries colored by
    # S8, i.e. the marginal distribution the likelihood flow has to learn the density of.
    try:
        LOGGER.timer.start("prior_predictive")
        prior_predictive.run_prior_predictive(flow, grid_preds, grid_cosmos, params, flow_conf)
        LOGGER.info(f"[timing] prior predictive: {LOGGER.timer.elapsed('prior_predictive')}")
    except Exception as e:
        print(f"ERROR: prior predictive stage failed ({type(e).__name__}: {e})")

    # likelihood-level coverage stage (cheap, no MCMC): HPD/TARP on x ~ p(x|theta) for the held-out mocks,
    try:
        LOGGER.timer.start("likelihood_coverage")
        coverage.run_likelihood_coverage(flow, grid_preds, grid_cosmos, i_signal, flow_conf)
        LOGGER.info(f"[timing] likelihood coverage: {LOGGER.timer.elapsed('likelihood_coverage')}")
    except Exception as e:
        print(f"ERROR: likelihood coverage stage failed ({type(e).__name__}: {e})")

    # mirror the eval-side default: no --mock_labels -> sample every mock in the prediction file
    if args.include_mocks and args.mock_labels is None:
        args.mock_labels = observations.discover_mock_labels(obs_pred_dict)
        LOGGER.info(f"Auto-discovered {len(args.mock_labels)} mock(s): {args.mock_labels}")

    obs_dict = observations.collect_observations(args, obs_pred_dict, obs_cosmo_dict, params, msfm_conf)
    try:
        LOGGER.timer.start("mcmc_total")
        observations.run_mcmc(
            flow,
            obs_dict,
            n_walkers=mcmc_conf.get("n_walkers", 1024),
            n_steps=mcmc_conf.get("n_steps", 1000),
            n_burnin_steps=mcmc_conf.get("n_burnin_steps", 1000),
            method=mcmc_conf.get("method", "ensemble"),
            use_validation_weights=mcmc_conf.get("use_validation_weights", True),
            backend=args.mcmc_backend,
            store_individual_chains=mcmc_conf.get("store_individual_chains", False),
        )
        LOGGER.info(f"[timing] all MCMC chains ({len(obs_dict)} observations): {LOGGER.timer.elapsed('mcmc_total')}")
    except Exception as e:
        print(f"ERROR: run_mcmc failed ({type(e).__name__}: {e})")

    # per-member DES chains for the ensemble-convergence test (independent of the pooled chains above)
    if args.sample_flow_members:
        try:
            LOGGER.timer.start("flow_members")
            observations.run_member_mcmc(
                flow,
                obs_dict,
                n_walkers=mcmc_conf.get("n_walkers", 1024),
                n_steps=mcmc_conf.get("n_steps", 1000),
                n_burnin_steps=mcmc_conf.get("n_burnin_steps", 1000),
                obs_labels=args.flow_member_obs,
                backend=args.mcmc_backend,
            )
            LOGGER.info(f"[timing] per-member DES chains: {LOGGER.timer.elapsed('flow_members')}")
        except Exception as e:
            print(f"ERROR: per-member sampling stage failed ({type(e).__name__}: {e})")

    # mock-contamination comparison (uses the per-mock chains from run_mcmc; independent of --sample_posterior)
    if flow_conf.get("diagnostics", {}).get("tests", {}).get("mock_contamination", False):
        try:
            LOGGER.timer.start("mock_contamination")
            mock_contamination.run_mock_contamination(flow, params, msfm_conf, flow_conf)
            LOGGER.info(f"[timing] mock contamination: {LOGGER.timer.elapsed('mock_contamination')}")
        except Exception as e:
            print(f"ERROR: mock contamination stage failed ({type(e).__name__}: {e})")

    if args.sample_posterior:
        if args.mcmc_backend != "torch_batched" or not hasattr(flow, "sample_posterior_batched"):
            LOGGER.warning("--sample_posterior requires --mcmc_backend=torch_batched; skipping.")
        else:
            try:
                LOGGER.timer.start("coverage")
                coverage.run_coverage(
                    flow,
                    grid_preds,
                    grid_cosmos,
                    i_signal,
                    flow_conf,
                    params,
                    obs_pred_dict,
                    i_sobol=i_sobol,
                    msfm_conf=msfm_conf,
                    i_noise=i_noise,
                )
                LOGGER.info(f"[timing] coverage sampling + tests: {LOGGER.timer.elapsed('coverage')}")
            except Exception as e:
                print(f"ERROR: coverage stage failed ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()

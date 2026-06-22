"""
Stage A (PyTorch / torch_env) of the posterior-tension analysis.

For each configured run pair and mock observation, this script produces the two
parameter-difference chains used to quantify the tension between two analysis setups:

* ``diff_chain_uncorrelated`` -- ``tensiometer.mcmc_tension.parameter_diff_chain`` applied to the
  two *independently* sampled per-probe inference chains (getdist/numpy only; no TensorFlow).
* ``diff_chain_correlated``   -- the shared-parameter shift of the *joint* residual posterior
  ``p(theta_1, theta_2 | x_obs)``, sampled with ``MarginalFlow.sample_residual_posterior`` using
  per-probe emulators and a residual normalizing flow trained on the held-out grid residuals.

Both chains are saved (as ``.npy``) into each run's ``{flow_name}_{steps}`` directory, and a
parameter-shift triangle plot is written to each run's ``unblinding_plots`` subdirectory. Each is
emitted in two parametrizations -- native ``sigma8`` and ``S8 = sigma8 * sqrt(Om / 0.3)`` (suffix
``_S8``) -- where S8 is a cheap post-hoc reparametrization of the shared emulators / flow / MCMC.

The numerical tension significance is assigned afterwards by ``run_tension_values.py`` (tf_env).

Refactor of ``deep_lss_paper/paper_2/pre-unblinding/5a_parameter_tension.ipynb``.
"""

import argparse
import os

import numpy as np

from msfm.utils import files as msfm_files
from msfm.utils import logger
from msfm.utils.input_output import read_yaml
from msi.flow_conductor.marginal_flow import MarginalFlow
from msi.utils import tensions

LOGGER = logger.get_logger(__file__)

DEFAULT_MSFM_CONFIG = (
    "/users/athomsen/dlss/repos/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
)


def setup():
    parser = argparse.ArgumentParser(description="Tension stage A (torch): build parameter-difference chains.")
    parser.add_argument("--runs_config", required=True, help="YAML defining runs, comparisons and obs_labels.")
    parser.add_argument("--tension_config", required=True, help="YAML defining method hyperparameters.")
    parser.add_argument(
        "--msfm_config",
        default=DEFAULT_MSFM_CONFIG,
        help="msfm forward-model config providing the parameter priors for the residual MCMC.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_residual_flow(residuals, flow_conf, device):
    """Train the residual normalizing flow on the (pair-specific) concatenated residuals."""
    residual_flow = MarginalFlow(
        feature_dim=residuals.shape[-1],
        n_transforms=flow_conf.get("n_transforms", 5),
        hidden_features=flow_conf.get("hidden_features", 128),
        n_blocks=flow_conf.get("n_blocks", 2),
        device=device,
    )
    residual_flow.fit(
        x=residuals,
        n_epochs=flow_conf.get("n_epochs", 50),
        batch_size=flow_conf.get("batch_size", 10_000),
        vali_split=flow_conf.get("vali_split", 0.1),
        learning_rate=flow_conf.get("learning_rate", 1e-3),
        clip_by_global_norm=flow_conf.get("clip_by_global_norm", 1.0),
        plot_loss=False,
        run_c2st=flow_conf.get("run_c2st", True),
    )
    return residual_flow


def save_chain(samples, out_dirs, filename):
    """Save a chain (as ``.npy``) into every output directory."""
    for odir in out_dirs:
        os.makedirs(odir, exist_ok=True)
        path = os.path.join(odir, filename)
        np.save(path, samples)
        LOGGER.info(f"Saved chain to {path}")


def main():
    args = setup()

    runs_conf = read_yaml(args.runs_config)
    tension_conf = read_yaml(args.tension_config)
    msfm_conf = msfm_files.load_config(args.msfm_config)

    emu_conf = tension_conf.get("emulator", {})
    flow_conf = tension_conf.get("residual_flow", {})
    mcmc_conf = tension_conf.get("mcmc", {})
    uncorr_conf = tension_conf.get("uncorrelated", {})

    combinations = tensions.build_combinations(runs_conf)
    observations = tensions.iter_observations(runs_conf)
    # LambdaCDM drops w0 -> different emulator input dim, so emulators are keyed by (run, lambdaCDM).
    lambda_values = sorted({use_lambdaCDM for _, use_lambdaCDM in observations})
    LOGGER.info(f"{len(combinations)} combination(s) x {len(observations)} observation(s) to process")

    # tensiometer.mcmc_tension.param_diff is pure getdist/numpy -- importing it does NOT pull in
    # TensorFlow (only flow_parameter_shift does), so the uncorrelated chain is built here.
    from getdist.mcsamples import MCSamples
    from tensiometer.mcmc_tension.param_diff import parameter_diff_chain

    # An emulator depends only on its own run (+ lambdaCDM/S8 processing), not on the other run in
    # the pair, and align_rows + a fixed random_state make it identical across pairs. The same run
    # appears in many combinations, so cache per-run data loads and emulators and only train the
    # (pair-specific) residual flow inside the loop. Keyed by pred_dir, which is unique per run.
    data_cache = {}  # pred_dir -> (preds, cosmos_raw, obs_dict, real_idx), row-aligned
    emu_cache = {}   # (pred_dir, use_lambdaCDM) -> (emu, val_idx, cosmos_processed, params_processed)

    def get_data(run):
        key = run["pred_dir"]
        if key not in data_cache:
            preds, cosmos_raw, obs_dict, real_idx = tensions.load_probe_data(run)
            preds, cosmos_raw, real_idx = tensions.align_rows(preds, cosmos_raw, real_idx)
            data_cache[key] = (preds, cosmos_raw, obs_dict, real_idx)
        return data_cache[key]

    def get_emu(run, use_lambdaCDM):
        key = (run["pred_dir"], use_lambdaCDM)
        if key not in emu_cache:
            preds, cosmos_raw, _, _ = get_data(run)
            # emulators/flow/MCMC always run in native sigma8 space; S8 = f(Om, sigma8) is applied as
            # a cheap post-hoc reparametrization of the resulting difference chains (both are emitted).
            cosmos, params = tensions.process_cosmologies(cosmos_raw, run["params"], use_lambdaCDM, False)
            LOGGER.info(f"Training emulator: {tensions.get_identifier(run['pred_dir'])} (lambdaCDM={use_lambdaCDM})")
            emu, val_idx = tensions.train_emu(preds, cosmos, emu_conf, device=args.device)
            emu_cache[key] = (emu, val_idx, cosmos, params)
        return emu_cache[key]

    for designation, run_1, run_2 in combinations:
        LOGGER.info(f"=== {designation} ===")
        out_dirs = tensions.out_dirs_for(run_1, run_2)
        orig_params_1 = list(run_1["params"])
        orig_params_2 = list(run_2["params"])

        preds_1, _, obs_dict_1, real_idx_1 = get_data(run_1)
        preds_2, _, obs_dict_2, real_idx_2 = get_data(run_2)
        assert np.array_equal(real_idx_1, real_idx_2), "grid realizations are not row-aligned between the two runs"

        for use_lambdaCDM in lambda_values:
            obs_for_group = [label for label, lcdm in observations if lcdm == use_lambdaCDM]
            _, lambdaCDM_string = tensions.string_suffixes(False, use_lambdaCDM)
            LOGGER.info(f"--- lambdaCDM={use_lambdaCDM}: {obs_for_group} ---")

            emu_1, val_idx_1, cosmos_1, params_1 = get_emu(run_1, use_lambdaCDM)
            emu_2, val_idx_2, cosmos_2, params_2 = get_emu(run_2, use_lambdaCDM)
            assert np.array_equal(val_idx_1, val_idx_2), "emulators used different validation splits"
            vali_idx = val_idx_1

            # residual flow is pair-specific: fit on the concatenated validation residuals
            residuals = np.concatenate(
                [
                    preds_1[vali_idx] - emu_1.predict(cosmos_1[vali_idx], device=args.device),
                    preds_2[vali_idx] - emu_2.predict(cosmos_2[vali_idx], device=args.device),
                ],
                axis=1,
            )
            residual_flow = build_residual_flow(residuals, flow_conf, args.device)

            shared = tensions.shared_params(params_1, params_2)
            # s8 -> S8 is an in-place reparametrization, so the shared-parameter indices are the same
            # for both the sigma8 and the S8 chains; only the column values and labels differ.
            shared_idx_1 = [params_1.index(p) for p in shared]
            shared_idx_2 = [len(params_1) + params_2.index(p) for p in shared]
            n1 = len(params_1)

            for obs_label in obs_for_group:
                LOGGER.info(f"Processing obs_label: {obs_label}")

                # load the two independent inference chains once (native sigma8 space)
                raw_1 = np.load(tensions.chain_path(run_1, obs_label, lambdaCDM_string))
                raw_2 = np.load(tensions.chain_path(run_2, obs_label, lambdaCDM_string))

                # run the (expensive) joint residual-posterior MCMC once in native sigma8 space
                joint_obs = np.concatenate([obs_dict_1[obs_label], obs_dict_2[obs_label]], axis=0)
                residual_samples = residual_flow.sample_residual_posterior(
                    x_obs=joint_obs,
                    params_wl=params_1,
                    params_gc=params_2,
                    emulator_wl=emu_1,
                    emulator_gc=emu_2,
                    conf=msfm_conf,
                    device=args.device,
                    n_walkers=mcmc_conf.get("n_walkers", 1024),
                    n_steps=mcmc_conf.get("n_steps", 10_000),
                    n_burnin_steps=mcmc_conf.get("n_burnin_steps", 1_000),
                )

                # emit both the sigma8 and the S8 parametrization; S8 = f(Om, sigma8) is a cheap
                # post-hoc reparametrization, so the emulators / flow / MCMC above are shared.
                for use_S8 in (False, True):
                    S8_string, _ = tensions.string_suffixes(use_S8, use_lambdaCDM)

                    # --- uncorrelated: parameter difference of the two independent chains ---
                    chain_1, names_1 = tensions.process_cosmologies(raw_1, orig_params_1, use_lambdaCDM, use_S8)
                    chain_2, names_2 = tensions.process_cosmologies(raw_2, orig_params_2, use_lambdaCDM, use_S8)
                    samples_1 = MCSamples(samples=chain_1, names=names_1)
                    samples_2 = MCSamples(samples=chain_2, names=names_2)
                    diff_uncorrelated = parameter_diff_chain(
                        samples_1, samples_2, boost=uncorr_conf.get("boost", 10)
                    )
                    save_chain(
                        diff_uncorrelated.samples,
                        out_dirs,
                        tensions.diff_chain_filename(
                            "uncorrelated", designation, obs_label, S8_string, lambdaCDM_string
                        ),
                    )

                    # --- correlated: shared-parameter shift of the joint residual posterior ---
                    # reparametrize the (native) MCMC samples per run, then difference shared params
                    proc_1, names_1c = tensions.process_cosmologies(residual_samples[:, :n1], params_1, False, use_S8)
                    proc_2, names_2c = tensions.process_cosmologies(residual_samples[:, n1:], params_2, False, use_S8)
                    joint_samples = np.concatenate([proc_1, proc_2], axis=1)
                    diff_correlated = joint_samples[:, shared_idx_1] - joint_samples[:, shared_idx_2]
                    save_chain(
                        diff_correlated,
                        out_dirs,
                        tensions.diff_chain_filename(
                            "correlated", designation, obs_label, S8_string, lambdaCDM_string
                        ),
                    )

                    # --- triangle plot of both difference chains ---
                    try:
                        plot_files = [
                            os.path.join(odir, "unblinding_plots",
                                         tensions.plot_filename(designation, obs_label, S8_string, lambdaCDM_string))
                            for odir in out_dirs
                        ]
                        tensions.plot_diff_chain(
                            diff_uncorrelated.samples,
                            labels=tensions.shared_labels(names_1c, names_2c),
                            title=f"{obs_label}: {designation}",
                            out_files=plot_files,
                            diff_chain_2=diff_correlated,
                            label_1=r"$\mathcal{L}_1(\theta_1) \mathcal{L}_2(\theta_2)$",
                            label_2=r"$\mathcal{L}(\theta_1, \theta_2)$",
                        )
                        LOGGER.info(f"Saved plots: {plot_files}")
                    except Exception as e:
                        LOGGER.warning(f"plotting failed ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()

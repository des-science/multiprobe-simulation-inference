"""Prepare or execute one arm of the summary coverage experiment.

Preparation is CPU-only metadata inspection, never model training. Run as
``python -m msi.apps.coverage_round prepare --output /path/to/new/round``.
``run --round /path/to/round --arm joint_long`` executes one prepared arm.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import yaml

from msi.utils.grid_weighting import COSMO_PARAMS, design_weights, lookup_coordinates


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PREDS = Path("/users/athomsen/dlss/scratch/runs/v18/default/maps_gcnn/combined/v1/preds_229900.h5")
ARMS = {
    "all_short": ("all", "short", []),
    "all_long": ("all", "long", []),
    "projected_long": ("reweight_projected", "long", []),
    "joint_long": ("reweight_joint", "long", []),
    "conditional_long": ("reweight_conditional", "long", []),
    "wide_short": ("wide", "short", []),
    "wide_long": ("wide", "long", []),
    "extended_long": ("all", "long", ["ns", "Ob", "H0"]),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(preds, output, seed=7, arms=None):
    """Freeze inputs, mock identities, update budgets and complete arm configs."""
    from msfm.utils.prior import NARROW_GRID_BOX
    import msfm

    preds, output = Path(preds).absolute(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {output}")
    with open(preds.parent / "configs.yaml") as f:
        run = yaml.safe_load(f)
    msfm_conf, params = run["msfm"], run["dlss"]["dset"]["training"]["params"]
    supported_params = [
        ["Om", "s8", "w0", "Aia", "n_Aia", "bta"],
        ["Om", "s8", "w0", "bg1", "bg2", "bg3", "bg4"],
        ["Om", "s8", "w0", "Aia", "n_Aia", "bta", "bg1", "bg2", "bg3", "bg4"],
    ]
    if params not in supported_params:
        raise ValueError(f"Unsupported summary parameters: {params}")
    selected_arms = list(ARMS) if arms is None else list(arms)
    if not selected_arms or len(set(selected_arms)) != len(selected_arms) or any(a not in ARMS for a in selected_arms):
        raise ValueError("Select distinct known arms")
    if not preds.stem.startswith("preds_") or not preds.stem[6:].isdigit():
        raise ValueError("Expected a preds_<checkpoint>.h5 filename")
    checkpoint = int(preds.stem[6:])
    meta = Path(msfm.__file__).resolve().parents[1] / msfm_conf["files"]["meta_info"]
    with h5py.File(meta) as f:
        table = f["parameters/grid"][:]
    with h5py.File(preds) as f:
        ids = np.column_stack([f[f"grid/{k}/test"][:].reshape(-1) for k in ("i_sobol", "i_signal", "i_noise")])
        theta = f["grid/cosmos/test"][:].reshape(-1, len(params))
        summary_shape = f["grid/preds/test"].shape
    if summary_shape != (2500, 80, len(params)):
        raise ValueError(f"Unexpected prediction shape {summary_shape}; recheck the experimental design")
    coords = lookup_coordinates(table, ids[:, 0])
    if not np.allclose(coords[:, :3], theta[:, :3], rtol=1e-5, atol=1e-6):
        raise ValueError("Metadata and prediction cosmological parameters disagree")
    if len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError("Duplicate realization identities")
    if not np.array_equal(np.unique(ids[:, 0]), np.sort(table["sobol_index"])):
        raise ValueError("Predictions must contain the complete grid for this design")
    _, counts = np.unique(ids[:, 0], return_counts=True)
    if len(np.unique(counts)) != 1:
        raise ValueError("Unequal examples per cosmology change the assumed grid mixture")
    wide_set = table["sobol_index"][table["id_param"] < 1250]
    wide = np.isin(ids[:, 0], wide_set)
    signals = np.unique(ids[:, 1])
    train = np.isin(ids[:, 1], signals[:int(0.9 * len(signals))])
    batch_size = 10000
    steps = {"all": int(train.sum()) // batch_size, "wide": int((train & wide).sum()) // batch_size}
    multiple = steps["all"] * steps["wide"] // math.gcd(steps["all"], steps["wide"])
    budgets = {"short": math.ceil(2400 / multiple) * multiple, "long": math.ceil(5100 / multiple) * multiple}
    available = ids[~train & wide]
    available = available[np.lexsort((available[:, 2], available[:, 1], available[:, 0]))]
    mocks = available[::len(available) // 1000][:1000]
    if len(mocks) != 1000 or len(np.unique(mocks[:, 0])) != 1000:
        raise ValueError("Expected 1000 distinct coverage cosmologies")

    audit = {}
    priors = msfm_conf["analysis"]["grid"]["priors"]
    for mode in ("projected", "joint", "conditional"):
        weights, info = design_weights(coords, priors, NARROW_GRID_BOX, mode)
        for subset, mask in (("train", train), ("validation", ~train)):
            w = weights[mask]
            info[f"{subset}_ess_fraction"] = float(w.sum() ** 2 / (w @ w) / len(w))
        audit[mode] = info
    bounds = np.array([NARROW_GRID_BOX[p] for p in COSMO_PARAMS])
    unique_coords = lookup_coordinates(table, table["sobol_index"])
    inside = ((unique_coords >= bounds[:, 0]) & (unique_coords <= bounds[:, 1])).all(axis=1)
    audit["narrow_component_outside_box_ids"] = table["id_param"][(table["id_param"] >= 1250) & ~inside].tolist()
    audit["wide_component_inside_6d_box"] = int(inside[table["id_param"] < 1250].sum())
    audit["mixture_assumption"] = "Documented 50/50 design; retain nine boundary-label exceptions (user accepted)."

    with open(REPO / "configs/flow/coverage/base8.yaml") as f:
        base = yaml.safe_load(f)
    base["seed"] = seed
    base["training"].update(group_by="signal", grid_wide_fraction=0.5)
    base["diagnostics"].update(
        mock_ids_file=str(output / "mock_ids.npy"), sampling_seed=12, subsample_seed=17,
        n_likelihood_samples=1000,
    )
    # Store the entire setup before a job starts. Input files are linked read-only in usage;
    # every result lands beneath this new directory, never in the original run.
    output.mkdir(parents=True)
    (output / preds.name).symlink_to(preds)
    (output / "configs.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
    np.save(output / "mock_ids.npy", mocks)
    (output / "arms").mkdir()
    manifest = {
        "checkpoint": checkpoint, "prediction_file": str(preds), "metadata_file": str(meta), "seed": seed, "n_flows": 8,
        "base_params": params, "prior_intervals": {p: priors[p] for p in params + ["ns", "Ob", "H0"]},
        "training_rows": int(train.sum()), "wide_training_rows": int((train & wide).sum()),
        "steps_per_epoch": steps, "budgets": budgets, "audit": audit, "arms": {}, "hashes": {},
    }
    for name in selected_arms:
        mode, budget, extension = ARMS[name]
        config = copy.deepcopy(base)
        updates = budgets[budget]
        epochs = updates // steps["wide" if mode == "wide" else "all"]
        config["training"].update(train_prior=mode, n_epochs=epochs, expected_updates=updates)
        config_path = output / "arms" / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        manifest["arms"][name] = {"config": str(config_path), "extend_params": extension,
                                  "n_epochs": epochs, "updates": updates}
    tracked = [preds, meta, output / "configs.yaml", output / "mock_ids.npy",
               *sorted((output / "arms").glob("*.yaml")),
               REPO / "msi/apps/coverage_round.py", REPO / "msi/apps/run_inference.py",
               REPO / "msi/utils/grid_weighting.py", REPO / "msi/utils/flow.py",
               REPO / "msi/utils/coverage.py", REPO / "msi/flow_conductor/likelihood_flow.py",
               REPO / "msi/apps/score_coverage_round.py", REPO / "msi/likelihood_base.py",
               REPO / "msi/flow_conductor/architecture.py", REPO / "msi/utils/extended_params.py",
               REPO / "msi/utils/torch_ensemble.py",
               Path(msfm.__file__).resolve().parent / "utils/prior.py"]
    manifest["hashes"] = {str(p): sha256(p) for p in tracked}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "steps": steps, "budgets": budgets, "audit": audit,
                      "arms": manifest["arms"]}, indent=2))
    return manifest


def run_arm(round_dir, arm, dry_run=False):
    round_dir = Path(round_dir).absolute()
    with open(round_dir / "manifest.json") as f:
        manifest = json.load(f)
    for path, expected in manifest["hashes"].items():
        if sha256(path) != expected:
            raise ValueError(f"Prepared input or code changed: {path}. Prepare a fresh round directory.")
    settings = manifest["arms"][arm]
    checkpoint = manifest.get("checkpoint", 229900)
    result = round_dir / f"{arm}_ensemble_flow_{checkpoint}"
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite existing arm {result}")
    command = [sys.executable, "-m", "msi.apps.run_inference", "--out_dir", str(round_dir.parent),
               "--model_name", round_dir.name, "--n_steps", str(checkpoint), "--n_flows", str(manifest["n_flows"]),
               "--flow_config", settings["config"], "--flow_label", arm, "--sample_posterior"]
    if settings["extend_params"]:
        command += ["--extend_params", *settings["extend_params"]]
    print(json.dumps(command), flush=True)
    if dry_run:
        return
    # An exclusive marker prevents two launchers from training the same arm concurrently.
    with open(round_dir / f"{arm}.started", "x") as f:
        f.write(json.dumps(command) + "\n")
    subprocess.run(command, check=True)
    # run_inference catches diagnostic failures; a successful process is not sufficient.
    with h5py.File(result / "mcmc_samples.h5") as f:
        if not np.array_equal(f["real_idx"][:], np.load(round_dir / "mock_ids.npy")):
            raise ValueError("Output coverage identities do not match the prepared experiment")
        expected_params = manifest["base_params"] + settings["extend_params"]
        if list(f.attrs["params"]) != expected_params:
            raise ValueError("Output parameter labels differ from experiment")
        if f["theta_sample"].shape != (10000, 1000, len(expected_params)):
            raise ValueError("Incomplete posterior sample array")
        for key in ("theta_sample", "log_prob_sample", "theta_true", "log_prob_true"):
            for start in range(0, len(f[key]), 100):
                if not np.isfinite(f[key][start:start + 100]).all():
                    raise ValueError(f"Nonfinite output in {key}")
    for name in ("1_likelihood_hpd.png", "1_likelihood_tarp.png", "2_posterior_hpd.png",
                 "2_posterior_tarp.png", "2_posterior_tarp_marginals.png"):
        if not (result / "unblinding_plots" / name).is_file():
            raise FileNotFoundError(f"Missing diagnostic plot: {name}")
    (round_dir / f"{arm}.complete").write_text("Validated required outputs\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--preds", type=Path, default=DEFAULT_PREDS)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=7)
    prep.add_argument("--arms", nargs="+", choices=list(ARMS))
    run = subs.add_parser("run")
    run.add_argument("--round", type=Path, required=True)
    run.add_argument("--arm", choices=list(ARMS), required=True)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.preds, args.output, args.seed, args.arms)
    else:
        run_arm(args.round, args.arm, args.dry_run)


if __name__ == "__main__":
    main()

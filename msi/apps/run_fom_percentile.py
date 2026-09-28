"""
Percentile of the observed figure of merit among the FoMs of the held-out coverage mocks.

For every run in a runs config, compares FoM = det[Cov(Om, S8)]^(-1/2) of the observed chain with that of each
mock posterior in ``mcmc_samples.h5`` (written by ``run_inference.py --sample_posterior``). Both are read from the
run's ``{flow_name}_{steps}`` directory, and the result is written there as ``fom_percentile_{obs_label}.yaml``.
Numpy and h5py only, so it runs on a login node.
"""

import argparse
import os

import h5py
import numpy as np
import yaml

from msfm.utils import logger
from msfm.utils.input_output import read_yaml
from msi.utils import diagnostics, tensions

LOGGER = logger.get_logger(__file__)


def setup():
    parser = argparse.ArgumentParser(description="Percentile of the observed FoM among the coverage mocks.")
    parser.add_argument("--runs_config", required=True, help="YAML defining the runs, as for run_tension_chains.py.")
    parser.add_argument("--obs_label", default="DESy3", help="Observed chain, chain_<obs_label>.npy.")
    parser.add_argument("--representations", nargs="+", default=None, help="Subset of runs_config['runs'] keys.")
    return parser.parse_args()


def main():
    args = setup()
    runs_conf = read_yaml(args.runs_config)
    flow_name = runs_conf.get("flow_name", "likelihood_flow")

    for rep, probes in runs_conf["runs"].items():
        if not probes or (args.representations is not None and rep not in args.representations):
            continue
        for probe, run in probes.items():
            run = dict(run, flow_name=run.get("flow_name", flow_name))
            flow_dir = os.path.dirname(tensions.chain_path(run, args.obs_label, ""))

            chain_obs, params_obs = tensions.load_chain(run, args.obs_label, "", run["params"])
            with h5py.File(os.path.join(flow_dir, "mcmc_samples.h5"), "r") as f:
                params_sample = [p.decode() if isinstance(p, bytes) else p for p in f.attrs["params"]]
                theta_sample = f["theta_sample"][:]

            fom_obs, fom_mocks, percentile = diagnostics.FoM_percentile(
                chain_obs, params_obs, theta_sample, params_sample
            )
            results = {
                "obs_label": args.obs_label,
                "param_set": ["Om", "S8"],
                "fom_obs": float(fom_obs),
                "percentile": float(percentile),
                "n_mocks": int(fom_mocks.size),
                "fom_mocks_quantiles": {q: float(np.quantile(fom_mocks, q / 100)) for q in (2.5, 16, 50, 84, 97.5)},
            }
            out_file = os.path.join(flow_dir, f"fom_percentile_{args.obs_label}.yaml")
            with open(out_file, "w") as f:
                yaml.safe_dump(results, f, default_flow_style=False, sort_keys=False)
            LOGGER.info(
                f"{rep} {probe}: FoM {fom_obs:.1f}, percentile {percentile:.1f} among {fom_mocks.size} mocks"
                f" (median {results['fom_mocks_quantiles'][50]:.1f}) -> {out_file}"
            )


if __name__ == "__main__":
    main()

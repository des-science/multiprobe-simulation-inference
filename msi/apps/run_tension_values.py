"""
Stage B (TensorFlow / tf_env) of the posterior-tension analysis.

Minimal companion to ``run_tension_chains.py``: it loads the parameter-difference chains produced
by stage A and uses tensiometer's normalizing-flow estimator (``flow_parameter_shift``, the only
TensorFlow-dependent step) to assign a numerical n-sigma tension significance to each.

Results are written as a small YAML file into each run's ``likelihood_flow_{steps}`` directory,
co-located with the chains they were computed from.

Refactor of ``deep_lss_paper/paper_2/pre-unblinding/5b_parameter_tension.ipynb``.
"""

import argparse
import os

import numpy as np
import yaml

from msfm.utils import logger
from msfm.utils.input_output import read_yaml
from msi.utils import tensions

LOGGER = logger.get_logger(__file__)


def setup():
    parser = argparse.ArgumentParser(description="Tension stage B (tf): assign numerical tension significance.")
    parser.add_argument("--runs_config", required=True, help="YAML defining runs, comparisons and obs_labels.")
    parser.add_argument("--tension_config", required=True, help="YAML defining method hyperparameters.")
    return parser.parse_args()


def main():
    args = setup()

    runs_conf = read_yaml(args.runs_config)
    tension_conf = read_yaml(args.tension_config)

    tensiometer_conf = tension_conf.get("tensiometer", {})
    flow_conf = tensiometer_conf.get("flow", {})
    run_kde = tensiometer_conf.get("run_kde", False)
    kde_conf = tensiometer_conf.get("kde", {})

    combinations = tensions.build_combinations(runs_conf)
    observations = tensions.iter_observations(runs_conf)
    LOGGER.info(f"{len(combinations)} combination(s) x {len(observations)} observation(s) to process")

    for designation, run_1, run_2 in combinations:
        LOGGER.info(f"=== {designation} ===")
        out_dirs = tensions.out_dirs_for(run_1, run_2)
        in_dir = out_dirs[0]  # chains are duplicated across out_dirs; read from the first

        for obs_label, use_lambdaCDM in observations:
            # stage A emits both the sigma8 and the S8 parametrization of every difference chain
            for use_S8 in (False, True):
                S8_string, lambdaCDM_string = tensions.string_suffixes(use_S8, use_lambdaCDM)

                results = {"designation": designation, "obs_label": obs_label, "use_lambdaCDM": use_lambdaCDM,
                           "use_S8": use_S8}
                for kind in ("uncorrelated", "correlated"):
                    chain_file = os.path.join(
                        in_dir, tensions.diff_chain_filename(kind, designation, obs_label, S8_string, lambdaCDM_string)
                    )
                    if not os.path.exists(chain_file):
                        LOGGER.warning(f"Missing {kind} chain, skipping: {chain_file}")
                        continue

                    diff_chain = np.load(chain_file)
                    LOGGER.info(f"Loaded {kind} chain from {chain_file}")

                    flow_result = tensions.sigma_tension_flow(diff_chain, flow_conf)
                    LOGGER.info(f"{kind} flow: n_sigma = {flow_result['n_sigma']:.3f} (P = {flow_result['P']:.5f})")
                    results[f"{kind}_flow"] = flow_result

                    if run_kde:
                        kde_result = tensions.sigma_tension_kde(diff_chain, kde_conf)
                        LOGGER.info(f"{kind} kde: n_sigma = {kde_result['n_sigma']:.3f}")
                        results[f"{kind}_kde"] = kde_result

                filename = tensions.tension_filename(designation, obs_label, S8_string, lambdaCDM_string)
                for odir in out_dirs:
                    os.makedirs(odir, exist_ok=True)
                    out_file = os.path.join(odir, filename)
                    with open(out_file, "w") as f:
                        yaml.safe_dump(results, f, default_flow_style=False, sort_keys=False)
                    LOGGER.info(f"Saved tension results to {out_file}")


if __name__ == "__main__":
    main()

"""Score a prepared coverage round on identical mocks, with paired comparisons.

KS p-values are approximate diagnostics: finite correlated MCMC draws and a Sobol
truth design do not satisfy the exact iid continuous-rank KS null. Do not rank
models by p-value. Primary TARP references use seed 17; two fixed repetitions
measure reference sensitivity and must not be selected after inspecting results.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import kstest


REFERENCE_SEEDS = (17, 29, 43)


def tarp_ranks(samples, truth, bounds, seed):
    """Euclidean distance ranks with fixed prior scaling and independent references."""
    bounds = np.asarray(bounds, dtype=np.float64)
    scale = bounds[:, 1] - bounds[:, 0]
    if np.any(scale <= 0):
        raise ValueError("Invalid TARP scaling intervals")
    ref = np.random.default_rng(seed).uniform(size=truth.shape)
    target = np.sum(((truth - bounds[:, 0]) / scale - ref) ** 2, axis=-1)
    count = np.zeros(len(truth), dtype=np.int64)
    for start in range(0, len(samples), 128):
        distance = np.sum(((samples[start:start + 128] - bounds[:, 0]) / scale - ref) ** 2, axis=-1)
        count += np.sum(distance < target[None, :], axis=0)
    return count / len(samples)


def rank_metrics(rank):
    alpha = np.linspace(0, 1, 101)
    ecp = np.mean(rank[:, None] <= alpha[None, :], axis=0)
    d, p = kstest(rank, "uniform")
    return {"ks_D": float(d), "ks_p_approx": float(p), "ecp": ecp.tolist(),
            "coverage_68": float(np.mean(rank <= .68))}


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    result = np.empty(len(pvalues))
    result[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return result


def score(round_dir, output, allow_partial=False):
    root = Path(round_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    checkpoint = manifest.get("checkpoint", 229900)
    joint_name = f"joint{len(manifest['base_params'])}"
    expected_ids = np.load(root / "mock_ids.npy", allow_pickle=False)
    results, per_mock = {}, {}
    for arm, settings in manifest["arms"].items():
        filename = root / f"{arm}_ensemble_flow_{checkpoint}/mcmc_samples.h5"
        if not (root / f"{arm}.complete").exists():
            if allow_partial:
                continue
            raise FileNotFoundError(f"Arm is not complete: {arm}")
        with h5py.File(filename) as f:
            if not np.array_equal(f["real_idx"][:], expected_ids):
                raise ValueError(f"Unpaired coverage mocks: {arm}")
            params = list(f.attrs["params"])
            truth = f["theta_true"][:].astype(np.float64)
            samples = f["theta_sample"][::4].astype(np.float64)
            log_like_true = f["log_prob_true"][:].astype(np.float64)
            hpd = (f["log_prob_sample"][::4] > log_like_true[None, :]).mean(axis=0)
        if params != manifest["base_params"] + settings["extend_params"]:
            raise ValueError(f"Unexpected parameter labels: {arm}")
        metrics = {"hpd_dimension": len(params), "tests": {"hpd": rank_metrics(hpd)}, "marginals": {}}
        metrics["wide_mean_log_likelihood"] = float(log_like_true.mean())
        per_mock[arm] = {"hpd_cov68": (hpd <= .68).astype(float), "log_likelihood": log_like_true}
        groups = [(joint_name, manifest["base_params"]), ("Om", ["Om"]), ("s8", ["s8"]),
                  ("w0", ["w0"]), ("Om_s8", ["Om", "s8"]), ("Om_s8_w0", ["Om", "s8", "w0"])]
        for name, group in groups:
            cols = [params.index(p) for p in group]
            bounds = [manifest["prior_intervals"][p] for p in group]
            ranks = [tarp_ranks(samples[:, :, cols], truth[:, cols], bounds, seed) for seed in REFERENCE_SEEDS]
            metrics["tests"][f"tarp_{name}"] = rank_metrics(ranks[0])
            metrics["tests"][f"tarp_{name}"]["reference_repeat_D"] = [rank_metrics(r)["ks_D"] for r in ranks]
            per_mock[arm][f"tarp_{name}_cov68"] = (ranks[0] <= .68).astype(float)
        adjusted = holm_adjust([m["ks_p_approx"] for m in metrics["tests"].values()])
        for item, p in zip(metrics["tests"].values(), adjusted):
            item["holm_p_approx"] = float(p)
        metrics["no_rejection_at_0.01_approx"] = bool(np.all(adjusted >= .01))
        for p in manifest["base_params"]:
            j = params.index(p)
            draws = samples[:, :, j]
            mu, variance = draws.mean(axis=0), draws.var(axis=0)
            error2 = (mu - truth[:, j]) ** 2
            pit = (draws < truth[:, j]).mean(axis=0)
            marginal = {"rmse": float(np.sqrt(error2.mean())),
                        "rms_posterior_sd": float(np.sqrt(variance.mean())),
                        "rmse_over_rms_sd": float(np.sqrt(error2.mean() / variance.mean())),
                        "mean_error": float(np.mean(mu - truth[:, j])),
                        "pit_ks_D": float(kstest(pit, "uniform").statistic)}
            for level in (.68, .90, .95, .99):
                covered = (np.abs(pit - .5) <= level / 2).astype(float)
                marginal[f"equal_tail_cov_{level}"] = float(covered.mean())
                if level == .68:
                    per_mock[arm][f"{p}_equal_tail_cov68"] = covered
            per_mock[arm][f"{p}_squared_error"] = error2
            per_mock[arm][f"{p}_posterior_variance"] = variance
            metrics["marginals"][p] = marginal
        results[arm] = metrics
        print(f"{arm:18s} HPD D={metrics['tests']['hpd']['ks_D']:.4f} "
              f"TARP {joint_name} D={metrics['tests'][f'tarp_{joint_name}']['ks_D']:.4f}", flush=True)

    comparisons = {}
    pairs = [("joint_long", "projected_long"), ("conditional_long", "all_long"),
             ("joint_long", "wide_long"), ("all_long", "all_short"),
             ("wide_long", "wide_short"), ("extended_long", "all_long")]
    # Resample complete cosmology blocks, preserving the A/B pairing for every metric.
    cosmos, inverse = np.unique(expected_ids[:, 0], return_inverse=True)
    rng = np.random.default_rng(91)
    resamples = rng.integers(len(cosmos), size=(1000, len(cosmos)))
    counts = np.bincount(inverse)
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        comparison = {}
        for name in per_mock[a]:
            if name in ("hpd_cov68", "log_likelihood") and results[a]["hpd_dimension"] != results[b]["hpd_dimension"]:
                continue
            delta = per_mock[a][name] - per_mock[b][name]
            sums = np.bincount(inverse, weights=delta)
            boot = sums[resamples].sum(axis=1) / counts[resamples].sum(axis=1)
            comparison[name] = {"mean_delta_A_minus_B": float(delta.mean()),
                                "paired_bootstrap_95": np.quantile(boot, [.025, .975]).tolist()}
        comparisons[f"{a} minus {b}"] = comparison
    report = {"reference_seeds": REFERENCE_SEEDS, "tarp_scaling": "fixed wide-prior box intervals",
              "alpha": np.linspace(0, 1, 101).tolist(), "arms": results, "paired_comparisons": comparisons,
              "caution": "KS/Holm and bootstrap uncertainty are approximate for Sobol truths and correlated MCMC. "
                         "No-rejection is a screening result, not proof of accuracy. All arms share development mocks."}
    Path(output).write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    score(args.round, args.output, args.allow_partial)


if __name__ == "__main__":
    main()

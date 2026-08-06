"""
Utilities for the blinded "shifted posterior" comparison plots.

Refactored from ``deep_lss_paper/paper_2/pre-unblinding/6b_shifted_posterior.ipynb`` (map-level vs.
Cls-level posterior comparison against the DES Y3 chain) and ``6_shifted_posterior.ipynb`` (the
training-step convergence diagnostic).

Blinding convention: each chain is shifted by ``-find_MAP(chain) + fiducial`` so its peak sits at the
fiducial cosmology. This is only valid when the chain does not press a prior boundary (the MAP is
then unreliable), so :func:`load_shifted_chain` returns ``None`` for a pressing chain and the caller
simply omits it -- per chain, not per probe.

Run bookkeeping is driven by a runs config (e.g. ``configs/runs/v17/baseline/t2_v3.yaml``); a run is the dict
``runs[data_type][probe] = {params, pred_dir, steps, [convergence_steps]}`` augmented with the
top-level ``flow_name``.
"""

import os

import numpy as np

from msfm.utils import logger, parameters, prior
from msi.utils import diagnostics, plotting

LOGGER = logger.get_logger(__file__)

# Astrophysical nuisance parameters to display per probe (curated subset of each probe's params).
ASTRO_PARAMS_BY_PROBE = {
    "lensing": ["Aia"],
    "clustering": ["bg1", "bg2", "bg3", "bg4"],
    "combined": ["Aia", "bg1", "bg2", "bg3", "bg4"],
    "2x2pt": ["Aia", "bg1", "bg2", "bg3", "bg4"],
    "cross": ["Aia", "bg1", "bg2", "bg3", "bg4"],
}


# ----------------------------------------------------------------------------------------
# run / path bookkeeping
# ----------------------------------------------------------------------------------------
def get_run(runs_conf, data_type, probe):
    """Return the run dict for ``(data_type, probe)`` augmented with ``flow_name``, or ``None``.

    Mirrors ``msi.utils.tensions`` so the same runs config (e.g. configs/runs/v17/baseline/t2_v3.yaml) drives
    both the tension analysis and these plots.
    """
    probes = runs_conf.get("runs", {}).get(data_type, {})
    if probe not in probes:
        return None
    run = dict(probes[probe])
    run["data"] = data_type
    run["probe"] = probe
    run.setdefault("flow_name", runs_conf.get("flow_name", "likelihood_flow"))
    return run


def flow_dir(run, steps=None):
    """Flow checkpoint directory ``{pred_dir}/{flow_name}_{steps}`` (defaults to the run's steps)."""
    steps = run["steps"] if steps is None else steps
    return os.path.join(run["pred_dir"], f"{run.get('flow_name', 'likelihood_flow')}_{steps}")


def unblinding_plot_dir(run):
    """The run's ``unblinding_plots`` directory (not created here)."""
    return os.path.join(flow_dir(run), "unblinding_plots")


def reset_dir(path):
    """Empty (or create) a directory. For the quick plot aggregation only -- not archival storage."""
    import shutil

    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    LOGGER.info(f"reset aggregation dir {path}")


# ----------------------------------------------------------------------------------------
# chain loading + blinding (MAP -> fiducial shift), with per-chain prior skip
# ----------------------------------------------------------------------------------------
def _load_chain_slice(run, obs_label, test_params, steps=None):
    """Load ``chain_{obs}.npy`` + ``log_probs_{obs}.npy`` and slice to ``test_params``."""
    fdir = flow_dir(run, steps)
    chain_file = os.path.join(fdir, f"chain_{obs_label}.npy")
    LOGGER.info(f"loading {chain_file}")
    chain = np.load(chain_file)
    log_probs = np.load(os.path.join(fdir, f"log_probs_{obs_label}.npy"))

    # "S8" is not a sampled parameter; derive it from (Om, sigma8) when requested.
    def column(p):
        if p == "S8":
            om = chain[:, run["params"].index("Om")]
            s8 = chain[:, run["params"].index("s8")]
            return plotting.sigma8_to_S8(sigma8=s8, Om=om)
        return chain[:, run["params"].index(p)]

    return np.stack([column(p) for p in test_params], axis=1), log_probs


def load_shifted_chain(run, obs_label, test_params, msfm_conf):
    """Load a chain and shift it to the fiducial, or return ``None`` if it presses the prior.

    Logs the per-pair FoM, then checks the prior boundary in the ``test_params`` subspace. A pressing
    chain cannot be reliably MAP-shifted, so it is skipped (returns ``None``) instead of aborting the
    whole figure.
    """
    chain, log_probs = _load_chain_slice(run, obs_label, test_params)

    for i, p1 in enumerate(test_params):
        for j, p2 in enumerate(test_params):
            if i > j:
                LOGGER.warning(f"FoM_({p1},{p2}) = {int(diagnostics.FoM_from_chain(chain, test_params, p1, p2))}")

    any_pressing = prior.assess_prior_boundary(chain, params=test_params, conf=msfm_conf, blinded=True)
    if any_pressing:
        LOGGER.warning(
            f"skipping {run['data']}/{run['probe']} {test_params}: prior boundary hit, MAP shift unreliable"
        )
        return None

    chain = chain - plotting.find_MAP(chain, log_probs, test_params, test_params)
    chain = chain + parameters.get_fiducials(test_params, msfm_conf)
    return chain


def compute_foms(run, obs_label, test_params):
    """Per-pair FoM dict from the raw chain (FoM is covariance-based, so shift-invariant)."""
    chain, _ = _load_chain_slice(run, obs_label, test_params)
    foms = {}
    for i, p1 in enumerate(test_params):
        for j, p2 in enumerate(test_params):
            if i > j:
                foms[(p1, p2)] = diagnostics.FoM_from_chain(chain, test_params, p1, p2)
    return foms


def compute_fom(run, obs_label, param_set):
    """Single N-d figure of merit over ``param_set`` (shift-invariant; ``"S8"`` derived if asked)."""
    chain, _ = _load_chain_slice(run, obs_label, param_set)
    return diagnostics.FoM_from_chain_nd(chain, param_set, param_set)


# ----------------------------------------------------------------------------------------
# comparison plot (maps vs. cls vs. DES), saved to every dir in plot_dirs
# ----------------------------------------------------------------------------------------
def plot_comparison(
    plot_params, obs_label, probe, msfm_conf, chains, ref, plot_dirs, des=None, use_harmonic=False, agg_dir=None
):
    """Overlay the (non-``None``) shifted chains and the DES chain, saving to each ``plot_dirs`` entry.

    Args:
        chains (dict): ``{"maps": arr|None, "cls": arr|None}`` -- already fiducial-shifted.
        ref (np.ndarray): fiducial values for ``plot_params`` (drawn as axlines).
        des (tuple|None): ``(chain, weights, foms, params)`` from ``chains.load_and_shift_des_chain``.
        plot_dirs (list): directories to save the PNG into (e.g. both maps and Cls unblinding_plots).
        agg_dir (str|None): optional flat aggregation dir; a probe-qualified copy is also saved there.
    """
    from trianglechain import TriangleChain

    des_chain = des_weights = des_params = des_foms = None
    if des is not None:
        des_chain, des_weights, des_foms, des_params = des

    if des_chain is None and all(c is None for c in chains.values()):
        LOGGER.warning(f"nothing to plot for {probe} {plot_params} (all chains skipped); no figure written")
        return

    tri = TriangleChain(
        names=plot_params,
        labels=[plotting.param_label_dict[p] for p in plot_params],
        ranges=dict(zip(plot_params, parameters.get_prior_intervals(plot_params, msfm_conf))),
        fill=False,
        show_legend=True,
        axlines_kwargs={"linestyle": "--"},
        progress_bar=False,
    )

    # DES first so it renders behind our chains
    if des_chain is not None:
        des_label = "DES Y3 2x2pt harmonic" if use_harmonic else "DES Y3 KP"
        tri.contour_cl(des_chain, prob=des_weights, names=des_params, label=des_label, color="k", fill=False)
    if chains.get("cls") is not None:
        tri.contour_cl(chains["cls"], names=plot_params, label="cls")
    if chains.get("maps") is not None:
        tri.contour_cl(chains["maps"], names=plot_params, label="maps")
    tri.axlines(ref[np.newaxis], names=plot_params, color="k", label="fiducial")

    # title: per-chain FoMs
    title_lines = [probe]
    for label in ("maps", "cls"):
        chain = chains.get(label)
        if chain is None:
            continue
        parts = [
            f"FoM({p1},{p2}) {label}={int(diagnostics.FoM_from_chain(chain, plot_params, p1, p2))}"
            for i, p1 in enumerate(plot_params)
            for j, p2 in enumerate(plot_params)
            if i > j
        ]
        if parts:
            title_lines.append("  |  ".join(parts))
    if des_foms:
        des_fom_label = "2x2pt harmonic" if use_harmonic else "kp"
        title_lines.append("  |  ".join(f"FoM({p1},{p2}) {des_fom_label}={v}" for (p1, p2), v in des_foms.items()))
    tri.fig.suptitle("\n".join(title_lines), fontsize=9)

    fname = f"6_comparison_{'_'.join(plot_params)}_{obs_label}.png"
    for pdir in plot_dirs:
        os.makedirs(pdir, exist_ok=True)
        out_file = os.path.join(pdir, fname)
        tri.fig.savefig(out_file, bbox_inches="tight", dpi=100)
        LOGGER.info(f"saved {out_file}")

    # probe-qualified copy in the flat aggregation dir (per-model dirs disambiguate by path; the
    # flat dir does not, so the probe must be in the filename to avoid collisions across probes)
    if agg_dir is not None:
        os.makedirs(agg_dir, exist_ok=True)
        agg_file = os.path.join(agg_dir, f"6_comparison_{probe}_{'_'.join(plot_params)}_{obs_label}.png")
        tri.fig.savefig(agg_file, bbox_inches="tight", dpi=100)
        LOGGER.info(f"saved {agg_file}")


# ----------------------------------------------------------------------------------------
# training-step convergence (chains shifted jointly with a single MAP anchor)
# ----------------------------------------------------------------------------------------
def load_shifted_chain_at_steps(run, obs_label, test_params, steps_list, msfm_conf, ref_steps=None):
    """Load the chain at each training step, all shifted by ONE common (reference-step) MAP.

    Comparing posteriors across training steps is a convergence diagnostic, not a physically
    meaningful re-blinding, so a single ``map_shift`` (from ``ref_steps``, default the largest step)
    is applied to every step. Returns ``(chains_by_step, fidu)``.
    """
    if ref_steps is None:
        ref_steps = max(steps_list)

    ref_chain, ref_log_probs = _load_chain_slice(run, obs_label, test_params, steps=ref_steps)
    map_shift = plotting.find_MAP(ref_chain, ref_log_probs, test_params, test_params)
    fidu = parameters.get_fiducials(test_params, msfm_conf)

    chains = {}
    for steps in steps_list:
        chain, _ = _load_chain_slice(run, obs_label, test_params, steps=steps)
        chains[steps] = chain - map_shift + fidu
    return chains, fidu


def plot_convergence(run, obs_label, test_params, steps_list, msfm_conf, plot_dir=None, ref_steps=None, agg_dir=None):
    """Triangle overlay of the posterior across training steps, jointly shifted, + fiducial axlines."""
    import matplotlib.pyplot as plt
    from trianglechain import TriangleChain

    chains, fidu = load_shifted_chain_at_steps(run, obs_label, test_params, steps_list, msfm_conf, ref_steps)

    cmap = plt.cm.viridis_r
    colors = [cmap(i / max(len(steps_list) - 1, 1)) for i in range(len(steps_list))]

    tri = TriangleChain(
        names=test_params,
        labels=[plotting.param_label_dict[p] for p in test_params],
        ranges=dict(zip(test_params, parameters.get_prior_intervals(test_params, msfm_conf))),
        fill=False,
        show_legend=True,
        axlines_kwargs={"linestyle": "--"},
        progress_bar=False,
    )
    for (steps, chain), color in zip(chains.items(), colors):
        tri.contour_cl(chain, names=test_params, label=f"{steps // 1000}k steps", color=color)
    tri.axlines(fidu[np.newaxis], names=test_params, color="k", label="fiducial")

    if plot_dir is None:
        plot_dir = os.path.join(run["pred_dir"], "convergence_plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_file = os.path.join(plot_dir, f"6_convergence_{'_'.join(test_params)}_{obs_label}.png")
    tri.fig.savefig(out_file, bbox_inches="tight", dpi=100)
    LOGGER.info(f"saved {out_file}")

    # data_type/probe-qualified copy in the flat aggregation dir (avoids collisions across runs)
    if agg_dir is not None:
        os.makedirs(agg_dir, exist_ok=True)
        tag = f"{run['data']}_{run['probe']}"
        agg_file = os.path.join(agg_dir, f"6_convergence_{tag}_{'_'.join(test_params)}_{obs_label}.png")
        tri.fig.savefig(agg_file, bbox_inches="tight", dpi=100)
        LOGGER.info(f"saved {agg_file}")

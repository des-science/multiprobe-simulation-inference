"""
Utilities for the posterior-tension analysis between two analysis setups.

Refactored from ``deep_lss_paper/paper_2/pre-unblinding/5a_parameter_tension.ipynb``
(PyTorch: emulators + residual flow, produces the parameter-difference chains) and
``5b_parameter_tension.ipynb`` (TensorFlow: tensiometer significance estimation).

The two analysis stages live in different environments (torch_env vs. tf_env), so this
module deliberately performs *no* top-level torch / tensorflow / tensiometer / trianglechain
imports.  Every framework-specific helper imports its dependencies lazily, so both
``run_tension_chains.py`` (torch) and ``run_tension_values.py`` (tf) can import the shared
filename / path helpers without dragging in the other framework.
"""

import os
import itertools

import numpy as np


# ----------------------------------------------------------------------------------------
# run / combination bookkeeping (pure python, importable from either environment)
# ----------------------------------------------------------------------------------------
def get_identifier(pred_dir):
    """Short, human-readable id for a run derived from its prediction directory.

    e.g. ``.../maps/clustering/v6_cls`` -> ``maps_clustering``.
    """
    parts = pred_dir.rstrip("/").split("/")
    return f"{parts[-3]}_{parts[-2]}" if len(parts) >= 3 else parts[-1]


def make_designation(run_1, run_2):
    """Unique ``{id_1}_vs_{id_2}`` designation for a run pair (matches the notebooks)."""
    return f"{get_identifier(run_1['pred_dir'])}_vs_{get_identifier(run_2['pred_dir'])}"


def build_combinations(runs_conf):
    """Auto-enumerate all run pairs to analyze from the runs config.

    The runs config organizes runs as ``runs[data_representation][probe] = {params, pred_dir,
    steps}``.  Two families of comparisons are built (each gated by a ``comparisons`` flag):

    * ``probes``: within each data representation, every unordered pair of probes
      (run_1, run_2 in config dict order).
    * ``data``: for every probe present in >= 2 data representations, every unordered pair of
      data representations (run_1 = earlier-listed representation, e.g. maps; run_2 = cls).

    Returns
    -------
    list of (designation, run_1, run_2)
        ``run_1`` / ``run_2`` are shallow copies of the config entries, each augmented with its
        ``data`` and ``probe`` keys.  Copies are returned so callers may process them without
        mutating the shared config.
    """
    # Drop data representations set to Null/empty in the config (e.g. `cls: Null` for a maps-only run),
    # so a single runs config can be sliced to just maps or just Cls without editing the comparisons.
    runs = {data: probes for data, probes in runs_conf["runs"].items() if probes}
    comparisons = runs_conf.get("comparisons", {})

    # Flow checkpoint dir name: "likelihood_flow" for a single LikelihoodFlow, "ensemble_flow" for a
    # LikelihoodFlowEnsemble (n_flows>1). Settable globally (runs_conf["flow_name"]) or per-run.
    default_flow_name = runs_conf.get("flow_name", "likelihood_flow")

    def _run(data, probe):
        run = dict(runs[data][probe])
        run["data"] = data
        run["probe"] = probe
        run.setdefault("flow_name", default_flow_name)
        return run

    combinations = []

    if comparisons.get("probes", False):
        for data, probes in runs.items():
            for probe_1, probe_2 in itertools.combinations(probes.keys(), 2):
                run_1, run_2 = _run(data, probe_1), _run(data, probe_2)
                combinations.append((make_designation(run_1, run_2), run_1, run_2))

    if comparisons.get("data", False):
        data_types = list(runs.keys())
        all_probes = {p for probes in runs.values() for p in probes}
        for probe in all_probes:
            present = [d for d in data_types if probe in runs[d]]
            for data_1, data_2 in itertools.combinations(present, 2):
                run_1, run_2 = _run(data_1, probe), _run(data_2, probe)
                combinations.append((make_designation(run_1, run_2), run_1, run_2))

    return combinations


def iter_observations(runs_conf):
    """Flatten the ``obs_labels`` config into ``[(label, use_lambdaCDM), ...]``.

    Each entry may be a plain string (``use_lambdaCDM`` defaults to False) or a mapping
    ``{label: ..., lambdaCDM: ...}``.  ``use_lambdaCDM`` is per-observation because the
    LambdaCDM DESy3 fit is a distinct inference chain (``chain_DESy3_lambdaCDM.npy``).
    """
    observations = []
    for entry in runs_conf["obs_labels"]:
        if isinstance(entry, str):
            observations.append((entry, False))
        else:
            observations.append((entry["label"], bool(entry.get("lambdaCDM", False))))
    return observations


# ----------------------------------------------------------------------------------------
# filename / path conventions (must stay byte-identical to the notebooks)
# ----------------------------------------------------------------------------------------
def string_suffixes(use_S8, use_lambdaCDM):
    """Return the ``(S8_string, lambdaCDM_string)`` filename suffixes used by the notebooks."""
    return ("_S8" if use_S8 else "", "_lambdaCDM" if use_lambdaCDM else "")


def out_dirs_for(run_1, run_2, create=True):
    """The ``{flow_name}_{steps}`` output directory for each run (one per run).

    ``flow_name`` defaults to ``likelihood_flow`` but is ``ensemble_flow`` for runs whose flow was
    trained as a ``LikelihoodFlowEnsemble``; set it per-run or globally in the runs config.
    """
    out_dirs = []
    for run in (run_1, run_2):
        flow_name = run.get("flow_name", "likelihood_flow")
        odir = os.path.join(run["pred_dir"], f"{flow_name}_{run['steps']}")
        if create:
            os.makedirs(odir, exist_ok=True)
        out_dirs.append(odir)
    return out_dirs


def diff_chain_filename(kind, designation, obs_label, S8_string, lambdaCDM_string):
    """Filename for a saved parameter-difference chain (``kind`` in {correlated, uncorrelated})."""
    assert kind in ("correlated", "uncorrelated"), kind
    return f"diff_chain_{kind}_{designation}_{obs_label}{S8_string}{lambdaCDM_string}.npy"


def tension_filename(designation, obs_label, S8_string, lambdaCDM_string):
    """Filename for the numerical tension results (written by the tf stage)."""
    return f"tension_{designation}_{obs_label}{S8_string}{lambdaCDM_string}.yaml"


def plot_filename(designation, obs_label, S8_string, lambdaCDM_string):
    """Filename for the parameter-shift triangle plot."""
    return f"5_parameter_shifts_{designation}_{obs_label}{S8_string}{lambdaCDM_string}.png"


def chain_path(probe_dict, obs_label, lambdaCDM_string):
    """Path to a per-probe inference MCMC chain (note: no S8 suffix; S8 is applied post-load)."""
    flow_name = probe_dict.get("flow_name", "likelihood_flow")
    return os.path.join(
        probe_dict["pred_dir"],
        f"{flow_name}_{probe_dict['steps']}",
        f"chain_{obs_label}{lambdaCDM_string}.npy",
    )


# ----------------------------------------------------------------------------------------
# data loading + row alignment + cosmology processing (numpy / h5py only)
# ----------------------------------------------------------------------------------------
def load_probe_data(probe_dict):
    """Load grid predictions, cosmologies, observation dict and realization indices for a run.

    The realization indices ``(i_sobol, i_signal, i_noise)`` are needed to pair rows across two
    runs whose dataset pipelines order the within-cosmology realizations differently.
    """
    import h5py

    from msi.utils import input_output

    pred_path = os.path.join(probe_dict["pred_dir"], f"preds_{probe_dict['steps']}.h5")
    preds, cosmos, obs_dict, _ = input_output.load_network_preds_simple(pred_path)

    with h5py.File(pred_path, "r") as f:
        real_idx = np.stack(
            [f[f"grid/{key}/test"][:].reshape(-1) for key in ("i_sobol", "i_signal", "i_noise")],
            axis=1,
        )

    return preds, cosmos, obs_dict, real_idx


def align_rows(preds, cosmos, real_idx):
    """Sort rows into canonical ``(i_sobol, i_signal, i_noise)`` order.

    ``evaluate_grid`` only sorts by ``i_sobol``, so the within-cosmology ordering of the
    signal/noise realizations depends on the dataset pipeline (maps vs. cls runs differ). The
    joint residuals are only correlated correctly if row ``i`` of both runs is the same
    realization.  See memory ``project_tension_row_alignment_bug``.
    """
    order = np.lexsort((real_idx[:, 2], real_idx[:, 1], real_idx[:, 0]))
    return preds[order], cosmos[order], real_idx[order]


def process_cosmologies(cosmos, params, use_lambdaCDM, use_S8):
    """Apply LambdaCDM filtering (drop ``w0``) and/or ``sigma8 -> S8`` conversion.

    Pure function: returns ``(cosmos_processed, params_processed)`` and does not mutate inputs.
    """
    cosmos_processed = cosmos.copy()
    params_processed = list(params)

    if use_lambdaCDM and "w0" in params_processed:
        w0_idx = params_processed.index("w0")
        cosmos_processed = np.delete(cosmos_processed, w0_idx, axis=-1)
        params_processed.remove("w0")

    if use_S8:
        from msi.utils.plotting import sigma8_to_S8

        s8_idx = params_processed.index("s8")
        Om_idx = params_processed.index("Om")
        cosmos_processed[:, s8_idx] = sigma8_to_S8(sigma8=cosmos_processed[:, s8_idx], Om=cosmos_processed[:, Om_idx])
        params_processed[s8_idx] = "S8"

    return cosmos_processed, params_processed


def shared_params(params_1, params_2):
    """Cosmological parameters shared by both runs (preserving run_1 order)."""
    return [p for p in params_1 if p in params_2]


def shared_labels(params_1, params_2):
    r"""TriangleChain labels for the parameter *shifts*, e.g. ``$\Delta \Omega_m$``."""
    from msi.utils.plotting import param_label_dict

    return [r"$\Delta " + param_label_dict[p][1:] for p in shared_params(params_1, params_2)]


# ----------------------------------------------------------------------------------------
# torch stage helpers (lazy torch import)
# ----------------------------------------------------------------------------------------
def train_emu(grid_preds, grid_cosmos, emu_conf, device="cuda"):
    """Train an MLP emulator predicting the network summary from cosmology.

    Returns ``(emulator, val_idx)``.  ``emu_conf['random_state']`` must be shared between the
    two probes so that the two emulators use the *same* validation split (the residuals are
    only paired correctly when ``val_idx_1 == val_idx_2``).
    """
    from msi.utils.mlp import MLP

    emu = MLP(
        input_dim=grid_cosmos.shape[-1],
        hidden_dims=list(emu_conf.get("hidden_dims", [512, 512])),
        output_dim=grid_preds.shape[-1],
        dropout=emu_conf.get("dropout", 0.1),
        use_layer_norm=emu_conf.get("use_layer_norm", True),
    )
    _, val_idx = emu.fit(
        X=grid_cosmos,
        y=grid_preds,
        num_epochs=emu_conf.get("num_epochs", 30),
        batch_size=emu_conf.get("batch_size", 1000),
        learning_rate=emu_conf.get("learning_rate", 1e-3),
        clip_grad_norm=emu_conf.get("clip_grad_norm", 1.0),
        validation_split=emu_conf.get("validation_split", 0.5),
        plot_history=False,
        device=device,
        random_state=emu_conf.get("random_state", 17),
    )
    return emu, val_idx


# ----------------------------------------------------------------------------------------
# tensorflow stage helpers (lazy tensorflow / tensiometer import)
# ----------------------------------------------------------------------------------------
def format_tensiometer(results):
    """Convert a tensiometer ``(P, P_low, P_hi)`` shift result into numbers + a LaTeX string."""
    from tensiometer.utilities import stats_utilities as utilities

    shift_P, shift_low, shift_hi = results
    n_sigma = float(utilities.from_confidence_to_sigma(shift_P))
    n_sigma_p = float(utilities.from_confidence_to_sigma(shift_hi) - utilities.from_confidence_to_sigma(shift_P))
    n_sigma_m = float(utilities.from_confidence_to_sigma(shift_P) - utilities.from_confidence_to_sigma(shift_low))
    return {
        "P": float(shift_P),
        "P_low": float(shift_low),
        "P_hi": float(shift_hi),
        "n_sigma": n_sigma,
        "n_sigma_p": n_sigma_p,
        "n_sigma_m": n_sigma_m,
        "sigma_str": rf"${n_sigma:.3f}^{{+{n_sigma_p:.3f}}}_{{-{n_sigma_m:.3f}}}\;\sigma$",
    }


def sigma_tension_flow(diff_chain, flow_conf):
    """Tension significance of a parameter-difference chain via tensiometer's normalizing flow.

    The flow's training cost scales with the chain length (tensiometer uses the whole chain as a
    batch / shuffle buffer), so ``flow_conf['max_samples']`` optionally caps how many samples the
    flow trains on.  The full chain is still saved on disk for plotting; only this density estimate
    is subsampled.  A few x10^5 - 10^6 samples is ample for the low-tension regime.
    """
    import tensorflow as tf
    from getdist.mcsamples import MCSamples
    from tensiometer import mcmc_tension

    if isinstance(diff_chain, np.ndarray):
        max_samples = flow_conf.get("max_samples")
        if max_samples is not None and len(diff_chain) > max_samples:
            rng = np.random.default_rng(flow_conf.get("subsample_seed", 0))
            idx = rng.choice(len(diff_chain), size=int(max_samples), replace=False)
            diff_chain = diff_chain[idx]
        diff_chain = MCSamples(samples=diff_chain)

    results, _ = mcmc_tension.flow_parameter_shift(
        diff_chain,
        feedback=flow_conf.get("feedback", 2),
        pop_size=flow_conf.get("pop_size", 1),
        epochs=flow_conf.get("epochs", 20),
        learning_rate=flow_conf.get("learning_rate", 1e-3),
        steps_per_epoch=flow_conf.get("steps_per_epoch", 100),
        hidden_units=list(flow_conf.get("hidden_units", [64, 64, 64, 64])),
        activation=tf.math.asinh,
    )
    return format_tensiometer(results)


def sigma_tension_kde(diff_chain, kde_conf):
    """Tension significance via tensiometer's KDE estimator (optional cross-check)."""
    from getdist.mcsamples import MCSamples
    from tensiometer import mcmc_tension

    if isinstance(diff_chain, np.ndarray):
        diff_chain = MCSamples(samples=diff_chain)
    diff_chain = diff_chain.copy()
    diff_chain.thin(kde_conf.get("thin_factor", 10))

    results = mcmc_tension.kde_parameter_shift(
        diff_chain,
        scale=kde_conf.get("scale", "MISE"),
        method=kde_conf.get("method", "neighbor_elimination"),
        feedback=kde_conf.get("feedback", 1),
    )
    return format_tensiometer(results)


# ----------------------------------------------------------------------------------------
# plotting (lazy trianglechain import)
# ----------------------------------------------------------------------------------------
def plot_diff_chain(
    diff_chain_1,
    labels,
    title,
    out_files,
    diff_chain_2=None,
    random_sign_flip=True,
    label_1="",
    label_2="",
    n_sigma_str_1="",
    n_sigma_str_2="",
    dpi=100,
):
    """Triangle plot of one or two parameter-difference chains, saved to ``out_files``.

    Mirrors ``plot_diff_chain`` in the notebooks.  ``random_sign_flip`` applies a shared random
    sign per parameter (the difference direction is arbitrary).
    """
    from getdist.mcsamples import MCSamples
    from trianglechain import TriangleChain

    if isinstance(diff_chain_1, MCSamples):
        diff_chain_1 = diff_chain_1.samples
    if isinstance(diff_chain_2, MCSamples):
        diff_chain_2 = diff_chain_2.samples

    if random_sign_flip:
        signs = np.random.choice([-1, 1], size=diff_chain_1.shape[1])
        diff_chain_1 = diff_chain_1 * signs
        if diff_chain_2 is not None:
            diff_chain_2 = diff_chain_2 * signs

    tri = TriangleChain(
        labels=labels,
        fill=True,
        grid=True,
        axlines_kwargs={"linestyle": "--", "lw": 1},
        de_kwargs={"levels": [0.68, 0.95, 0.997]},
        show_legend=True,
        progress_bar=False,
    )
    tri.contour_cl(diff_chain_1, label=label_1 + n_sigma_str_1)
    if diff_chain_2 is not None:
        tri.contour_cl(diff_chain_2, label=label_2 + n_sigma_str_2)
    tri.axlines(np.zeros((1, len(labels))), color="k")
    tri.fig.suptitle(title, fontsize=24, y=0.95)

    for out_file in out_files:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        tri.fig.savefig(out_file, bbox_inches="tight", dpi=dpi)

    return tri

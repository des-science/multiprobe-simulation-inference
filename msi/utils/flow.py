import glob
import os
import re

import h5py
import numpy as np
import torch
import yaml

from msi.flow_conductor import architecture
from msi.flow_conductor.likelihood_flow import LikelihoodFlow, LikelihoodFlowEnsemble
from msi.utils import input_output


def _find_latest_n_steps(pred_dir):
    matches = glob.glob(os.path.join(pred_dir, "preds_*.h5"))
    steps = []
    for f in matches:
        m = re.search(r"preds_(\d+)\.h5$", f)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def find_all_n_steps(pred_dir):
    """Return a sorted list of all training-step counts with existing preds_*.h5 files."""
    matches = glob.glob(os.path.join(pred_dir, "preds_*.h5"))
    steps = []
    for f in matches:
        m = re.search(r"preds_(\d+)\.h5$", f)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def _fit_pca(x, n_components):
    """Fit PCA via covariance eigendecomposition. Returns (mean, components) where
    components has shape (n_components, n_features), ordered by descending variance."""
    mean = x.mean(axis=0)
    cov = np.cov((x - mean).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx[:n_components]].T  # (n_components, n_features)
    return mean, components


def load_grid_indices(pred_file):
    """Load and flatten the (i_sobol, i_signal, i_noise) triplet identifying each grid/test row."""
    with h5py.File(pred_file, "r") as f:
        i_sobol = f["grid/i_sobol/test"][:]
        i_signal = f["grid/i_signal/test"][:]
        i_noise = f["grid/i_noise/test"][:]
    if i_sobol.ndim == 2:
        i_sobol, i_signal, i_noise = (np.concatenate(arr, axis=0) for arr in (i_sobol, i_signal, i_noise))
    return i_sobol, i_signal, i_noise


def resolve_pred_file(out_dir, model_name, n_steps=None):
    """Resolve a trained model's prediction file, auto-detecting the latest preds_*.h5 if n_steps is omitted.

    Returns:
        tuple: (pred_dir, pred_file, n_steps)
    """
    pred_dir = os.path.join(out_dir, model_name)
    if n_steps is None:
        n_steps = _find_latest_n_steps(pred_dir)
        if n_steps is not None:
            print(f"Auto-detected n_steps={n_steps}")
        else:
            print("No preds_*.h5 found; falling back to preds.h5")
    pred_file = (
        os.path.join(pred_dir, f"preds_{n_steps}.h5") if n_steps is not None else os.path.join(pred_dir, "preds.h5")
    )
    return pred_dir, pred_file, n_steps


def load_grid_summaries(pred_file, pred_file_2=None):
    """Load network summary statistics for flow training/evaluation, optionally combining two models.

    Loads grid_preds/grid_cosmos/obs_pred_dict/obs_cosmo_dict from pred_file and -- if pred_file_2 is
    given -- row-aligns both grids on (i_sobol, i_signal, i_noise) and concatenates their summaries
    feature-wise, e.g. to combine a maps-level and a Cls-level model. Sharing this between
    run_inference.py (which trains/loads the flow) and run_mcmc_for_coverage_tests.py (which
    reproduces the flow's held-out validation split for coverage testing) guarantees both build
    grid_preds/grid_cosmos identically -- same content, same row order -- which the latter relies on
    to faithfully reproduce the split.

    Also returns i_signal and i_sobol, both row-aligned with grid_preds/grid_cosmos. i_signal lets
    callers group rows by signal realization -- e.g. to build a deterministic, signal-id-grouped flow
    train/vali split that never places different noise realizations of the same signal in both sets
    (see LikelihoodFlow._prepare_data's group_ids argument). i_sobol identifies the CosmoGrid cosmology
    per row, used to select the wide-grid (analysis-prior) cosmologies for posterior coverage (see
    msi.utils.coverage.wide_prior_sobol_indices).

    Returns:
        tuple: (grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal, i_sobol)
    """
    print(f"Loading predictions from: {pred_file}")
    grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict = input_output.load_network_preds_simple(pred_file)
    i_sobol, i_signal, _ = load_grid_indices(pred_file)

    if pred_file_2:
        print(f"Loading second predictions from: {pred_file_2}")
        grid_preds_2, grid_cosmos_2, obs_pred_dict_2, _ = input_output.load_network_preds_simple(pred_file_2)

        # the two prediction files generally store the same held-out grid examples in different
        # orders, so align them onto a common (i_sobol, i_signal, i_noise) ordering before comparing
        print("Aligning the two grids by (i_sobol, i_signal, i_noise)...")
        i_sobol, i_signal, i_noise = load_grid_indices(pred_file)
        i_sobol_2, i_signal_2, i_noise_2 = load_grid_indices(pred_file_2)

        order = np.lexsort((i_noise, i_signal, i_sobol))
        order_2 = np.lexsort((i_noise_2, i_signal_2, i_sobol_2))
        grid_preds, grid_cosmos = grid_preds[order], grid_cosmos[order]
        grid_preds_2, grid_cosmos_2 = grid_preds_2[order_2], grid_cosmos_2[order_2]
        i_signal = i_signal[order]
        i_sobol = i_sobol[order]

        aligned = (
            np.array_equal(i_sobol, i_sobol_2[order_2])
            and np.array_equal(i_signal, i_signal_2[order_2])
            and np.array_equal(i_noise[order], i_noise_2[order_2])
        )
        if not aligned or not np.allclose(grid_cosmos, grid_cosmos_2):
            raise ValueError(
                "Cannot align the two prediction files' grid examples by (i_sobol, i_signal, i_noise); "
                "cannot concatenate their summaries row-wise. Make sure both models were trained on "
                "the same dataset version and train/eval grid split."
            )

        print("Concatenating summaries from the two models...")
        grid_preds = np.concatenate([grid_preds, grid_preds_2], axis=-1)
        obs_pred_dict = {
            label: np.concatenate([obs_pred_dict[label], obs_pred_dict_2[label]], axis=-1)
            for label in obs_pred_dict.keys() & obs_pred_dict_2.keys()
        }

    return grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal, i_sobol


def load_grid_summaries_multi(pred_files, pca_compress=False):
    """Load and concatenate summary statistics from multiple prediction files feature-wise.

    Row-aligns all files on (i_sobol, i_signal, i_noise) before concatenating, then optionally
    applies PCA to compress the combined data vector back to single-run dimensionality.

    Args:
        pred_files: List of paths to preds_*.h5 files (e.g. for different training-step checkpoints).
        pca_compress: If True, fit PCA on the concatenated grid_preds and project both grid_preds
            and obs_pred_dict down to the same dimensionality as a single file's summaries.

    Returns:
        tuple: (grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal, i_sobol) — same
            signature as load_grid_summaries.
    """
    all_grid_preds, all_grid_cosmos, all_obs_pred_dicts, all_obs_cosmo_dicts, all_indices = [], [], [], [], []

    for pf in pred_files:
        print(f"Loading predictions from: {pf}")
        gp, gc, opd, ocd = input_output.load_network_preds_simple(pf)
        all_grid_preds.append(gp)
        all_grid_cosmos.append(gc)
        all_obs_pred_dicts.append(opd)
        all_obs_cosmo_dicts.append(ocd)
        all_indices.append(load_grid_indices(pf))

    sort_orders = [np.lexsort((idx[2], idx[1], idx[0])) for idx in all_indices]

    ref_sobol, ref_signal, ref_noise = (arr[sort_orders[0]] for arr in all_indices[0])
    ref_cosmos = all_grid_cosmos[0][sort_orders[0]]

    print("Aligning prediction files by (i_sobol, i_signal, i_noise)...")
    for i in range(1, len(pred_files)):
        s_sobol, s_signal, s_noise = (arr[sort_orders[i]] for arr in all_indices[i])
        aligned = (
            np.array_equal(ref_sobol, s_sobol)
            and np.array_equal(ref_signal, s_signal)
            and np.array_equal(ref_noise, s_noise)
            and np.allclose(ref_cosmos, all_grid_cosmos[i][sort_orders[i]])
        )
        if not aligned:
            raise ValueError(
                f"Prediction file {pred_files[i]} has mismatched grid examples; "
                "make sure all files were evaluated on the same dataset version and grid split."
            )

    sorted_preds = [gp[order] for gp, order in zip(all_grid_preds, sort_orders)]
    print("Concatenating summaries feature-wise...")
    grid_preds = np.concatenate(sorted_preds, axis=-1)
    grid_cosmos = ref_cosmos
    i_signal = ref_signal
    i_sobol = ref_sobol

    common_keys = set.intersection(*[set(opd.keys()) for opd in all_obs_pred_dicts])
    obs_pred_dict = {key: np.concatenate([opd[key] for opd in all_obs_pred_dicts], axis=-1) for key in common_keys}
    obs_cosmo_dict = all_obs_cosmo_dicts[0]

    if pca_compress:
        n_components = all_grid_preds[0].shape[-1]
        print(f"Applying PCA compression: {grid_preds.shape[-1]} → {n_components} components")
        mean, components = _fit_pca(grid_preds, n_components)
        grid_preds = (grid_preds - mean) @ components.T
        obs_pred_dict = {key: (val - mean) @ components.T for key, val in obs_pred_dict.items()}

    return grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal, i_sobol


def build_flow_architecture(x_dim: int, theta_dim: int, flow_conf: dict):
    """Build embedding net and transform from a config dict.

    Defaults match default.yaml / the original notebook values, so passing an empty
    dict reproduces the standard architecture.

    Two transform families are supported via ``transform.type``:
    - ``"sigmoids"`` (default): ConditionalSVD + MaskedSumOfSigmoids layers.
    - ``"lipschitz"``: Lipschitz-constrained iResBlocks (architecturally independent,
      useful as a cross-check when diagnosing posterior instability).

    Returns:
        tuple: (embedding_net, transform) ready to pass to LikelihoodFlow.
    """
    emb_conf = flow_conf.get("context_embedding", {})
    ctx_emb_dim = emb_conf.get("dim", 32)

    embedding_net = architecture.get_context_embedding_net(
        context_dim=theta_dim,
        context_embedding_dim=ctx_emb_dim,
        hidden_dim=emb_conf.get("hidden_dim", 64),
        n_blocks=emb_conf.get("n_blocks", 3),
        dropout_probability=emb_conf.get("dropout_probability", 0.0),
        use_batch_norm=emb_conf.get("use_batch_norm", False),
    )

    tr_conf = flow_conf.get("transform", {})
    transform_type = tr_conf.get("type", "sigmoids")

    if transform_type == "sigmoids":
        sig_conf = tr_conf.get("sigmoids", {})
        transform = architecture.get_sigmoids_transform(
            feature_dim=x_dim,
            context_embedding_dim=ctx_emb_dim,
            n_layers=tr_conf.get("n_layers", 4),
            hidden_dim=tr_conf.get("hidden_dim", 256),
            svd_kwargs={},
            sigmoids_kwargs={
                "n_sigmoids": sig_conf.get("n_sigmoids", 16),
                "num_blocks": sig_conf.get("num_blocks", 3),
                "dropout_probability": sig_conf.get("dropout_probability", 0.0),
            },
        )
    elif transform_type == "lipschitz":
        lip_conf = tr_conf.get("lipschitz", {})
        transform = architecture.get_lipschitz_transform(
            feature_dim=x_dim,
            context_embedding_dim=ctx_emb_dim,
            n_layers=tr_conf.get("n_layers", 8),
            hidden_dim=tr_conf.get("hidden_dim", 128),
            lipschitz_coeff=lip_conf.get("lipschitz_coeff", 0.97),
        )
    elif transform_type == "spline":
        sp_conf = tr_conf.get("spline", {})
        transform = architecture.get_spline_transform(
            feature_dim=x_dim,
            context_embedding_dim=ctx_emb_dim,
            n_layers=tr_conf.get("n_layers", 8),
            hidden_dim=tr_conf.get("hidden_dim", 128),
            num_bins=sp_conf.get("num_bins", 8),
            tail_bound=sp_conf.get("tail_bound", 5.0),
            mask_type=sp_conf.get("mask_type", "coupling"),
            num_blocks=sp_conf.get("num_blocks", 2),
            dropout_probability=sp_conf.get("dropout_probability", 0.0),
            use_linear=sp_conf.get("use_linear", True),
        )
    elif transform_type == "maf":
        maf_conf = tr_conf.get("maf", {})
        transform = architecture.get_maf_transform(
            feature_dim=x_dim,
            context_embedding_dim=ctx_emb_dim,
            n_layers=tr_conf.get("n_layers", 8),
            hidden_dim=tr_conf.get("hidden_dim", 128),
            num_blocks=maf_conf.get("num_blocks", 2),
            dropout_probability=maf_conf.get("dropout_probability", 0.0),
            use_linear=maf_conf.get("use_linear", True),
        )
    else:
        raise ValueError(
            f"Unknown transform type: {transform_type!r}. Choose 'sigmoids', 'lipschitz', 'spline', or 'maf'."
        )

    return embedding_net, transform


def validate_training_config(flow_conf):
    """Reject removed experiment options before loading data or fitting a flow."""
    training = flow_conf.get("training", {})
    removed = {"theta_jitter", "group_design", "group_bootstrap"}.intersection(training)
    if removed:
        raise ValueError(f"Removed experimental training options: {sorted(removed)}")
    if training.get("group_by", "signal") not in ("signal", "cosmology"):
        raise ValueError("training.group_by must be signal or cosmology")
    train_prior_mode(flow_conf)


def _extract_train_kwargs(flow_conf: dict) -> dict:
    """Pull the LikelihoodFlow.fit training arguments out of a flow config's ``training`` block,
    applying the same defaults as the single-flow path."""
    validate_training_config(flow_conf)
    train_conf = flow_conf.get("training", {})
    return dict(
        n_epochs=train_conf.get("n_epochs", 100),
        batch_size=train_conf.get("batch_size", 10_000),
        vali_split=train_conf.get("vali_split", 0.1),
        learning_rate=train_conf.get("learning_rate", 1e-3),
        weight_decay=train_conf.get("weight_decay", 0.0),
        scheduler_type=train_conf.get("scheduler_type", "cosine"),
        scheduler_kwargs=train_conf.get("scheduler_kwargs", None),
        n_patience_epochs=train_conf.get("n_patience_epochs", None),
        min_delta=train_conf.get("min_delta", 1e-4),
        run_c2st=train_conf.get("run_c2st", True),
    )


def resolve_extend_params(cli_value, flow_confs):
    """Settle the flow's extended conditioning vector, and say whether the CLI asked for it.

    The production vector lives in the flow config (``extend_params``); ``--extend_params`` overrides
    it per invocation, with the bare flag meaning ``DEFAULT_EXTEND_PARAMS``. The second return value
    is what run_inference uses to decide whether to default ``--flow_label`` to 'ext': an experiment
    run from the CLI must not overwrite the baseline checkpoint it is measured against, whereas the
    configured vector IS the baseline and owns the unprefixed directory.

    Returns:
        tuple: (list of parameter names, True if it came from the command line).
    """
    from msi.utils.extended_params import DEFAULT_EXTEND_PARAMS

    if cli_value is not None:
        return list(cli_value) if cli_value else list(DEFAULT_EXTEND_PARAMS), True
    configured = [list(conf.get("extend_params") or []) for conf in flow_confs]
    if any(c != configured[0] for c in configured[1:]):
        raise ValueError(f"Flow configs disagree about extend_params: {configured}")
    return configured[0], False


TRAIN_PRIOR_MODES = ("all", "wide", "reweight_projected", "reweight_joint", "reweight_conditional")


def train_prior_mode(flow_conf: dict):
    """Select all rows, the wide subset, or a deterministic grid-weighting control.

    These controls support the coverage experiments. Production keeps every row and
    conditions on the nuisance parameters explicitly, via the flow config's extend_params.
    """
    v = flow_conf.get("training", {}).get("train_prior", "all")
    if v not in TRAIN_PRIOR_MODES:
        raise ValueError(f"training.train_prior must be one of {TRAIN_PRIOR_MODES}, got {v!r}")
    return v


def full_grid_train_weights(i_sobol, grid_cosmos, params, msfm_conf, flow_conf):
    """Weights using metadata coordinates, including nuisances omitted from context.

    Modes share a deterministic volume calculation and design-mixture fraction.
    """
    from msi.utils import extended_params, grid_weighting
    from msfm.utils.prior import NARROW_GRID_BOX

    table = extended_params.load_grid_param_table(msfm_conf)
    coords = grid_weighting.lookup_coordinates(table, i_sobol)
    for j, p in enumerate(grid_weighting.COSMO_PARAMS):
        if p in params and not np.allclose(coords[:, j], grid_cosmos[:, params.index(p)], rtol=1e-5, atol=1e-6):
            raise ValueError(f"Prediction/metadata mismatch for {p}; cannot assign grid weights")
    weights, audit = grid_weighting.design_weights(
        coords,
        msfm_conf["analysis"]["grid"]["priors"],
        NARROW_GRID_BOX,
        mode=train_prior_mode(flow_conf)[len("reweight_") :],
        wide_fraction=flow_conf.get("training", {}).get("grid_wide_fraction", 0.5),
    )
    print(f"Grid weighting audit: {audit}")
    return weights


def restrict_to_wide_prior(grid_preds, grid_cosmos, i_signal, i_sobol, i_noise, msfm_conf):
    """Drop every grid row whose cosmology belongs to the narrow Sobol half.

    The flow and both coverage stages use the same filtered rows. This halves the
    cosmologies and changes updates per epoch, so comparisons must match optimizer
    updates rather than only epoch counts.

    Returns:
        tuple: the same arrays masked to the wide half (``i_noise`` passes through as None if unset).
    """
    from msi.utils.coverage import wide_prior_sobol_indices

    wide = np.asarray(list(wide_prior_sobol_indices(msfm_conf)))
    keep = np.isin(np.asarray(i_sobol).reshape(-1), wide)
    n_cos = len(np.unique(np.asarray(i_sobol)[keep]))
    print(
        f"Restricting training grid to the wide prior: {keep.sum()} of {len(keep)} rows, "
        f"{n_cos} of {len(np.unique(i_sobol))} cosmologies"
    )
    if not keep.any():
        raise ValueError("training.train_prior='wide' kept no rows; check the metainfo file")
    out = [grid_preds[keep], grid_cosmos[keep], np.asarray(i_signal)[keep], np.asarray(i_sobol)[keep]]
    out.append(None if i_noise is None else np.asarray(i_noise)[keep])
    return tuple(out)


def resolve_group_ids(flow_conf: dict, i_signal, i_sobol=None, msfm_conf=None):
    """Return row-aligned groups for a signal or cosmology holdout.

    Cosmology grouping reserves evenly spaced wide-grid cosmologies for validation,
    then ranks the ids so the common sorted-group split selects exactly that set.
    Use the same returned groups for fitting and both coverage stages. This tests
    interpolation to unseen cosmologies but yields fewer distinct coverage truths
    than the signal split; their coverage p-values are not directly comparable.
    """
    group_by = flow_conf.get("training", {}).get("group_by", "signal")
    if group_by == "signal":
        return i_signal
    if group_by != "cosmology":
        raise ValueError(f"training.group_by must be 'signal' or 'cosmology', got {group_by!r}")

    if i_sobol is None or msfm_conf is None:
        raise ValueError("training.group_by='cosmology' needs both i_sobol and msfm_conf")

    from msi.utils.coverage import wide_prior_sobol_indices

    i_sobol = np.asarray(i_sobol).reshape(-1)
    present = np.unique(i_sobol)
    vali_split = flow_conf.get("training", {}).get("vali_split", 0.1)
    # mirror _prepare_data's arithmetic exactly, so the partition below is the one it will make
    n_hold = len(present) - int((1 - vali_split) * len(present))

    wide = np.intersect1d(present, np.asarray(list(wide_prior_sobol_indices(msfm_conf))))
    if n_hold > len(wide):
        raise ValueError(
            f"vali_split={vali_split} would hold out {n_hold} cosmologies but only {len(wide)} of the "
            f"{len(present)} present are wide-prior; the coverage stage keeps only the wide half, so a "
            f"larger held-out set cannot be filled from it"
        )
    held = wide[np.linspace(0, len(wide) - 1, n_hold).round().astype(int)] if n_hold else np.array([], int)

    rank = np.empty(present.max() + 1, dtype=np.int64)
    train_cos = np.setdiff1d(present, held)
    rank[train_cos] = np.arange(len(train_cos))
    rank[held] = len(train_cos) + np.arange(len(held))
    print(
        f"Grouping the flow split by cosmology: {len(train_cos)} train / {len(held)} held-out "
        f"cosmologies (all held-out are wide-prior)"
    )
    return rank[i_sobol]


def refuse_incompatible_retrain(flow, flow_conf):
    """Refuse to retrain over a directory whose flow used a different conditioning vector.

    Training saves with ``exist_ok=True``, so a re-run would leave the previous analysis's chains,
    PPC and tension outputs beside a flow that no longer matches them, and nothing downstream
    re-checks. Compares the saved ``flow_config.yaml``, not the checkpoint, because the v18
    production flows were pruned once their chains existed. No saved config means no conflict.
    """
    saved_file = os.path.join(flow.model_dir, "flow_config.yaml")
    if not os.path.exists(saved_file):
        return
    try:
        with open(saved_file) as f:
            saved = list(yaml.safe_load(f).get("extend_params") or [])
    except Exception as e:  # noqa: BLE001 -- an unreadable config is not a conflict
        print(f"WARNING: could not read {saved_file} ({type(e).__name__}: {e}); not checking")
        return
    current = list(flow_conf.get("extend_params") or [])
    if saved != current:
        raise ValueError(
            f"{flow.model_dir} holds a flow trained with extend_params={saved or '[]'}, but this run "
            f"would train with {current or '[]'} and overwrite it in place, leaving the chains, PPC and "
            f"tension outputs beside it stale. Pass --flow_label to train alongside it, or move the "
            f"old directory away."
        )


def build_flow(
    params,
    msfm_conf,
    pred_dir,
    n_steps,
    grid_preds,
    grid_cosmos,
    flow_conf: dict,
    prefix: str = "",
    group_ids=None,
    seed=None,
    n_flows=1,
    flow_confs=None,
    row_weights=None,
):
    """Build, train, plot diagnostics, and return a LikelihoodFlow or LikelihoodFlowEnsemble.

    Args:
        params: List of cosmological parameter names.
        msfm_conf: Forward-model config dict (passed to LikelihoodFlow).
        pred_dir: Output directory for checkpoints and plots.
        n_steps: Training-step label appended to saved filenames.
        grid_preds: Array of shape (N, x_dim) — network summary statistics.
        grid_cosmos: Array of shape (N, theta_dim) — cosmological parameters.
        flow_conf: Flow config dict (keys: context_embedding, transform, training,
            diagnostics). Use {} or read_yaml(path) to populate. Ignored when ``flow_confs``
            is given (heterogeneous mode), except its ``seed`` is used as the base seed.
        prefix: Prepended to the saved model directory name, e.g. ``"larger_"`` →
            ``pred_dir/larger_likelihood_flow_{n_steps}/``. Useful when comparing
            multiple flow configs on the same prediction file.
        group_ids: Row-aligned groups from resolve_group_ids (signal or cosmology).
            Pass the same array to both coverage stages to reproduce the training split.
        seed: Optional torch seed for weight init and the (group-aware) train/vali split.
            Defaults to None, then flow_conf.get("seed", 7) is used.
        n_flows: If 1 (default), build a single LikelihoodFlow (current behavior). If >1,
            build a homogeneous LikelihoodFlowEnsemble of n_flows independently-initialized
            flows. When ``flow_confs`` is given, ``n_flows`` is instead a per-config
            replication factor (see below).
        flow_confs: Optional list of flow config dicts for a *heterogeneous* ensemble. When
            given, build one ensemble member per (config, replica) pair: total members =
            ``len(flow_confs) * n_flows``, member i uses ``flow_confs[i % len(flow_confs)]``
            (replicas of a config differ only by seed). Each member trains with its own
            config's ``training`` block. Defaults to None (homogeneous behavior above).
        row_weights: Optional array of shape (N,), row-aligned with grid_preds/grid_cosmos, weighting
            each row's contribution to the training and validation loss. Built by
            ``full_grid_train_weights`` for the weighting controls. Defaults to None
            (every row weighted equally).

    Returns:
        LikelihoodFlow or LikelihoodFlowEnsemble: Trained flow(s) with saved checkpoint(s).
    """
    x_dim = grid_preds.shape[-1]
    theta_dim = grid_cosmos.shape[-1]

    base_conf = flow_confs[0] if flow_confs else flow_conf
    # Experiment configs can assert the actual update budget without changing
    # historical epoch-based training or either training backend.
    training = base_conf.get("training", {})
    if "expected_updates" in training:
        from msi.flow_conductor.likelihood_flow import group_split_indices

        if group_ids is None or flow_confs:
            raise ValueError("expected_updates requires a homogeneous, group-split, unresampled training set")
        n_train = len(group_split_indices(group_ids, training.get("vali_split", 0.1))[0])
        actual = (n_train // training.get("batch_size", 10000)) * training["n_epochs"]
        if actual != training["expected_updates"]:
            raise ValueError(f"Training budget mismatch: {actual} != {training['expected_updates']}")
        print(f"Verified optimizer budget: {actual} updates per member")
    if seed is None:
        seed = base_conf.get("seed", 7)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    suffix = f"_{n_steps}" if n_steps is not None else ""

    member_train_kwargs = None

    if flow_confs:
        # Heterogeneous ensemble: one member per (config, replica). Build per-member architecture
        # factories and per-member training kwargs, so members differ in architecture and each trains
        # with its own config. Default-arg binding (c=conf) captures the right config in each closure.
        n_configs = len(flow_confs)
        total = n_configs * n_flows
        member_confs = [flow_confs[i % n_configs] for i in range(total)]
        embedding_fns = [(lambda c=c: build_flow_architecture(x_dim, theta_dim, c)[0]) for c in member_confs]
        transform_fns = [(lambda c=c: build_flow_architecture(x_dim, theta_dim, c)[1]) for c in member_confs]
        member_train_kwargs = [_extract_train_kwargs(c) for c in member_confs]
        flow = LikelihoodFlowEnsemble(
            params,
            msfm_conf,
            n_flows=total,
            feature_dim=x_dim,
            embedding_net_fn=embedding_fns,
            transform_fn=transform_fns,
            out_dir=pred_dir,
            prefix=prefix,
            suffix=suffix,
            load_existing=False,
            torch_seed=seed,
        )
        print(f"Fitting heterogeneous ensemble of {total} flows ({n_configs} configs x {n_flows} replicas)...")
    elif n_flows > 1:
        flow = LikelihoodFlowEnsemble(
            params,
            msfm_conf,
            n_flows=n_flows,
            feature_dim=x_dim,
            embedding_net_fn=lambda: build_flow_architecture(x_dim, theta_dim, flow_conf)[0],
            transform_fn=lambda: build_flow_architecture(x_dim, theta_dim, flow_conf)[1],
            out_dir=pred_dir,
            prefix=prefix,
            suffix=suffix,
            load_existing=False,
            torch_seed=seed,
        )
        print(f"Fitting ensemble of {n_flows} flows...")
    else:
        embedding_net, transform = build_flow_architecture(x_dim, theta_dim, flow_conf)
        flow = LikelihoodFlow(
            params,
            msfm_conf,
            feature_dim=x_dim,
            embedding_net=embedding_net,
            transform=transform,
            out_dir=pred_dir,
            prefix=prefix,
            suffix=suffix,
            load_existing=False,
            torch_seed=seed,
        )
        print("Fitting flow...")

    refuse_incompatible_retrain(flow, base_conf)

    fit_kwargs = _extract_train_kwargs(base_conf)
    if member_train_kwargs is not None:
        fit_kwargs["member_train_kwargs"] = member_train_kwargs
    flow.fit(
        x=grid_preds,
        theta=grid_cosmos,
        save_model=True,
        group_ids=group_ids,
        row_weights=row_weights,
        **fit_kwargs,
    )

    # likelihood-level coverage (HPD/EECP, TARP) is run as an explicit stage in run_inference.py
    # (coverage.run_likelihood_coverage), parallel to the posterior-level stage, so it also covers the
    # --load_flow path and shares the held-out split and plot helpers with the posterior tests.

    return flow

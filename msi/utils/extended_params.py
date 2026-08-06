"""Extend the flow's conditioning vector by grid parameters that are implicitly marginalized.

The CosmoGridV1 grid varies more parameters than the standard 6-dim inferred vector
(Om, s8, w0, Aia, n_Aia, bta): ns, Ob, H0 and -- for the baryonified grid -- bary_Mc,
bary_nu take a different value at every Sobol point. A flow conditioned only on the 6-dim
vector therefore implicitly marginalizes the others with the full flat grid prior. The
helpers here look those recorded values up per grid row (via i_sobol) so the flow can be
retrained on the extended conditioning vector WITHOUT recomputing the network summaries,
and the marginalization can instead be controlled at MCMC time (see the reference-prior
variant in msi.utils.observations and the fixed_params / gaussian_priors arguments of
sample_posterior_batched).

bary_Mc is stored raw (~1e12..1e15) in the metainfo but the config prior/fiducial use
log10 (12..15); the lookup converts to log10 to match the config convention.
"""

import os

import h5py
import numpy as np

import msfm
from msfm.utils import logger, parameters

LOGGER = logger.get_logger(__file__)

# parameters stored in log10 in the config (priors/fiducials) but raw in the metainfo grid table --
# single source of truth in msfm.utils.parameters (shared with the label-gather conversions)
_LOG10_PARAMS = parameters.LOG10_PARAMS

DEFAULT_EXTEND_PARAMS = ["ns", "Ob", "H0", "bary_Mc", "bary_nu"]


def load_grid_param_table(msfm_conf):
    """Load the CosmoGridV1 grid parameter table (one row per grid cosmology, all columns named
    like the config parameters, log10 applied where the config convention is log10)."""
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(msfm.__file__), ".."))
    meta_info_file = os.path.join(repo_dir, msfm_conf["files"]["meta_info"])
    with h5py.File(meta_info_file, "r") as f:
        table = f["parameters/grid"][:]
    return table


def _column(table, param):
    values = np.asarray(table[param], dtype=np.float64)
    if param in _LOG10_PARAMS:
        values = np.log10(values)
    return values.astype(np.float32)


def extend_grid_cosmos(grid_cosmos, i_sobol, extend_params, msfm_conf, table=None):
    """Append the recorded grid values of ``extend_params`` to every row of ``grid_cosmos``.

    Args:
        grid_cosmos: (N, n_params) array of the stored per-row parameters.
        i_sobol: (N,) Sobol-sequence indices (the ``sobol_index`` column of the metainfo grid
            table, NOT the row index), row-aligned with grid_cosmos.
        extend_params: list of metainfo column names to append, e.g. ["ns", "Ob", "H0",
            "bary_Mc", "bary_nu"].
        msfm_conf: forward-model config dict (for the metainfo path).
        table: optional preloaded output of load_grid_param_table.

    Returns:
        (N, n_params + n_extend) array.
    """
    if table is None:
        table = load_grid_param_table(msfm_conf)

    sobol_to_row = {int(s): i for i, s in enumerate(table["sobol_index"])}
    rows = np.array([sobol_to_row[int(s)] for s in np.asarray(i_sobol)])

    extra = np.stack([_column(table, p)[rows] for p in extend_params], axis=-1)
    LOGGER.info(
        f"Extended grid_cosmos by {extend_params}: {grid_cosmos.shape} -> "
        f"{(grid_cosmos.shape[0], grid_cosmos.shape[1] + len(extend_params))}"
    )
    return np.concatenate([grid_cosmos.astype(np.float32), extra], axis=-1)


def extend_obs_cosmo_dict(obs_cosmo_dict, params, extend_params, msfm_conf, table=None):
    """Append ``extend_params`` values to every observation's true-parameter vector.

    Grid observations are matched to their metainfo row by their (Om, s8, w0) values (unique
    per grid cosmology); anything that does not match a grid row (e.g. fiducial benchmark
    mocks) falls back to the config fiducials. Only used for plotting truth points, so the
    fiducial fallback is exact for the fiducial mocks and irrelevant for DES data (no cosmo).
    """
    if table is None:
        table = load_grid_param_table(msfm_conf)

    match_params = [p for p in ("Om", "s8", "w0") if p in params]
    match_idx = [params.index(p) for p in match_params]
    match_cols = np.stack([_column(table, p) for p in match_params], axis=-1)  # (n_grid, n_match)
    extend_cols = np.stack([_column(table, p) for p in extend_params], axis=-1)  # (n_grid, n_extend)

    fiducial = msfm_conf["analysis"]["fiducial"]
    fid_extra = np.array([fiducial[p] for p in extend_params], dtype=np.float32)

    extended = {}
    for label, cosmo in obs_cosmo_dict.items():
        cosmo = np.asarray(cosmo, dtype=np.float32)
        dist = np.abs(match_cols - cosmo[match_idx]).max(axis=-1)
        i_min = int(np.argmin(dist))
        extra = extend_cols[i_min] if dist[i_min] < 1e-4 else fid_extra
        extended[label] = np.concatenate([cosmo, extra])
    return extended

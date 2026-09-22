"""Importance weights for the documented wide + narrow CosmoGrid design.

Unlike weights computed from the flow context alone, these use all six recorded
cosmological coordinates, even when some are implicitly marginalized by the flow.
The 50/50 mixture is a design approximation: the current metadata has nine points
at id_param 1250..1258 outside the narrow box. Audit this rather than silently
inferring mixture membership from geometric membership.
"""

import numpy as np
from scipy.integrate import quad
from scipy.spatial import ConvexHull


COSMO_PARAMS = ("Om", "s8", "w0", "ns", "Ob", "H0")


def volume_fractions(priors, narrow_box):
    """Integrate the cut 3D volumes; factor out independent nuisance intervals.

    Uses the same convex hull and w0 >= 1/(Om-1)+0.01 as in_grid_prior.
    No noisy six-dimensional Monte Carlo volume estimate is needed.
    """
    equations = ConvexHull(priors["Om_s8_border_points"]).equations

    def volume(box):
        om_lo, om_hi = box["Om"]
        s_lo, s_hi = box["s8"]
        w_lo, w_hi = box["w0"]
        if not om_lo < om_hi < 1:
            raise ValueError("Expected 0 < Om interval width with upper bound < 1")

        def area(om):
            lower, upper = s_lo, s_hi
            for a, b, c in equations:
                if abs(b) < 1e-14:
                    if a * om + c > 1e-12:
                        return 0.0
                elif b > 0:
                    upper = min(upper, -(a * om + c) / b)
                else:
                    lower = max(lower, -(a * om + c) / b)
            return max(0.0, upper - lower) * max(0.0, w_hi - max(w_lo, 1 / (om - 1) + 0.01))

        points = [p[0] for p in priors["Om_s8_border_points"]]
        for w in (w_lo, w_hi):
            if w != 0.01:
                points.append(1 + 1 / (w - 0.01))
        points = sorted({p for p in points if om_lo < p < om_hi})
        return quad(area, om_lo, om_hi, points=points, epsabs=1e-10, limit=200)[0]

    for p in COSMO_PARAMS:
        lo, hi = narrow_box[p]
        if not priors[p][0] <= lo < hi <= priors[p][1]:
            raise ValueError(f"Narrow interval for {p} is not nested in the analysis prior")
    f3 = volume(narrow_box) / volume(priors)
    f6 = f3 * np.prod([
        (narrow_box[p][1] - narrow_box[p][0]) / (priors[p][1] - priors[p][0])
        for p in COSMO_PARAMS[3:]
    ])
    if not 0 < f6 <= f3 <= 1:
        raise ValueError(f"Invalid narrow/wide volume fractions: {f3}, {f6}")
    return float(f3), float(f6)


def design_weights(coords, priors, narrow_box, mode, wide_fraction=0.5):
    """Return mean-one row weights and an audit for a (N, 6) coordinate array.

    projected: pi(theta) / r(theta)
    joint: pi(theta, eta) / r(theta, eta)
    conditional: pi(eta | theta) / r(eta | theta) = joint / projected

    Here theta's grid coordinates are Om,s8,w0 and eta=(ns,Ob,H0).
    Other parameters must have the same conditional design in both components.
    Weights apply to *every* point in a region, including wide-component points.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 6 or len(coords) == 0 or not np.isfinite(coords).all():
        raise ValueError("Expected a nonempty finite (N, 6) grid coordinate array")
    if mode not in ("projected", "joint", "conditional"):
        raise ValueError(f"Unknown grid weighting mode {mode!r}")
    if not 0 < wide_fraction < 1:
        raise ValueError("wide_fraction must lie strictly between zero and one")
    f3, f6 = volume_fractions(priors, narrow_box)
    bounds = np.array([narrow_box[p] for p in COSMO_PARAMS])
    inside = (coords >= bounds[:, 0]) & (coords <= bounds[:, 1])
    inside3, inside6 = inside[:, :3].all(axis=1), inside.all(axis=1)
    ratio = (1 - wide_fraction) / wide_fraction
    projected = 1 / (1 + inside3 * ratio / f3)
    joint = 1 / (1 + inside6 * ratio / f6)
    raw = {"projected": projected, "joint": joint, "conditional": joint / projected}[mode]
    weights = raw / raw.mean()
    return weights, {
        "mode": mode, "wide_fraction": wide_fraction,
        "volume_fraction_3d": f3, "volume_fraction_6d": f6,
        "inside_3d": int(inside3.sum()), "inside_6d": int(inside6.sum()),
        "joint_inside_outside_ratio": float(1 / (1 + ratio / f6)),
        "projected_inside_outside_ratio": float(1 / (1 + ratio / f3)),
        "row_ess_fraction": float(weights.sum() ** 2 / (weights @ weights) / len(weights)),
        "weight_min": float(weights.min()), "weight_max": float(weights.max()),
    }


def lookup_coordinates(table, i_sobol):
    """Join by Sobol identity, never by row position or id_param."""
    keys = np.asarray(table["sobol_index"])
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Duplicate sobol_index in grid metadata")
    order = np.argsort(keys)
    requested = np.asarray(i_sobol).reshape(-1)
    pos = np.searchsorted(keys[order], requested)
    if np.any(pos >= len(keys)) or not np.array_equal(keys[order][pos], requested):
        raise ValueError("Prediction contains a Sobol index absent from grid metadata")
    rows = order[pos]
    return np.column_stack([table[p][rows] for p in COSMO_PARAMS])


def match_mock_rows(available, requested):
    """Match exact (Sobol, signal, noise) identities, retaining requested order."""
    available, requested = np.asarray(available), np.asarray(requested)
    if available.ndim != 2 or available.shape[1] != 3 or requested.ndim != 2 or requested.shape[1] != 3:
        raise ValueError("Mock identities must have shape (N, 3)")
    if len(np.unique(available, axis=0)) != len(available) or len(np.unique(requested, axis=0)) != len(requested):
        raise ValueError("Duplicate mock identity")
    lookup = {tuple(row): i for i, row in enumerate(available)}
    try:
        return np.array([lookup[tuple(row)] for row in requested], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Requested coverage mock is not held out: {exc.args[0]}") from exc

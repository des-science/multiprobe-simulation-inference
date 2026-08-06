# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Fisher forecast for (Om, s8, w0) from the hard_rebinned angular power spectra of a forward-model
dataset, separately for the lensing, clustering and combined probes. The forward-model version and
the scale cut are chosen via --msfm_config and --scales_config (nothing is hardcoded here).

Ingredients (neither is pre-computed in the hard_rebinned scheme):
  * Covariance  -- the signal+noise fiducial realizations in
                   {data_dir}/cls/fiducial_cls.h5 : cls/raw, rebinned to hard_rebinned.
  * Derivatives -- the finite-difference perturbation Cls (cl_{label}) stored inside the
                   fiducial .tfrecords ({data_dir}/tfrecords/fiducial/*.tfrecord).

The data vector is the hard_rebinned Cls the Cls MLP trains on: per (i<=j) tomographic pair,
raw per-ell Cls are averaged into cls_n_bins (=16) sqrt-spaced bins with a per-pair scale cut
[l_min, l_max] read from the chosen scales config (e.g. lmax_1024 -> l_min=30, l_max=1024 for all
bins; 8wl,32gc -> per-z-bin cuts). Rebinning reuses
deep_lss.utils.cls_preprocessing._build_bin_weights_all_pairs; probe selection reuses
msfm.utils.cross_statistics.get_cross_bin_indices. The Fisher construction mirrors
deep_lss/deprecated/estimators.py (F = J^T C^-1 J, central finite differences).

The headline result has ONE knob: which parameters the model includes. Each of the three forecast
modes (fixed / astro / astro_nuisance) includes a widening set, priors exactly the params it
includes, marginalizes over the non-cosmo ones and reports (Om, s8, w0). Decoupling the prior from
the model is a diagnostic, not a result, so those variations are emitted separately under
prior_variations/ (see --prior_variations) and the headline dir stays uncluttered.

Meant to run under the TensorFlow environment on a compute node (reads TFRecords, imports
msfm/deep_lss). CPU-only; no GPU needed.
"""

import os
import glob
import time
import argparse
import textwrap

import yaml
import h5py
import numpy as np
import tensorflow as tf

from msfm.utils import files, cross_statistics, tfrecords, logger, parameters
from deep_lss.utils.cls_preprocessing import _build_bin_weights_all_pairs

LOGGER = logger.get_logger(__file__)

# cosmological parameters we forecast; the fixed-nuisance Fisher uses exactly these
# Every model parameter falls in exactly one of three classes (per probe):
#   cosmo    -- what we ultimately constrain and report; never marginalized away.
#   astro    -- the probe's physical astrophysical parameters: intrinsic alignments (Aia, n_Aia) for
#               lensing, galaxy biases (bg1-4) for clustering. Constrained, and correlated with cosmo.
#   nuisance -- the remaining, barely/unconstrained parameters (H0, Ob, ns, bary_Mc, bary_nu). Not
#               listed explicitly: the class is DERIVED per probe as "constrained, non-cosmo,
#               non-astro" so it stays correct if the config's parameter set changes. Parameters the
#               probe's forward model does not touch at all (e.g. galaxy bias for lensing-only) have
#               an exactly-zero derivative and are dropped rather than counted as nuisance.
COSMO_PARAMS = ["Om", "s8", "w0"]

# probe definitions: selection flags for get_cross_bin_indices + the probe's astro parameter set
# (matching the y3-deep-lss _nla probe configs for the v17 standard-NLA dataset).
PROBES = {
    "lensing": dict(
        with_lensing=True,
        with_clustering=False,
        with_cross_z=True,
        with_cross_probe=False,
        astro=["Aia", "n_Aia"],
    ),
    "clustering": dict(
        with_lensing=False,
        with_clustering=True,
        with_cross_z=True,
        with_cross_probe=False,
        astro=["bg1", "bg2", "bg3", "bg4"],
    ),
    "combined": dict(
        with_lensing=True,
        with_clustering=True,
        with_cross_z=True,
        with_cross_probe=True,
        astro=["Aia", "n_Aia", "bg1", "bg2", "bg3", "bg4"],
    ),
}


# analysis-prior treatments. These are NOT alternative datasets: the Fisher is prior-free and the
# prior is a diagonal add at the final (<=14x14) inversion, so every mode is recoverable from the
# saved F for free -- run several rather than committing to one.
#   all       -- prior on every parameter (cosmo + astro + nuisance)
#   noncosmo  -- prior on the non-cosmo parameters (astro + nuisance); cosmo stays pure data
#   none      -- no prior at all
PRIOR_MODES = ["all", "noncosmo", "none"]

# The HEADLINE forecast couples the prior to the model: whatever parameters a mode includes carry the
# analysis prior, and the ones it excludes are held fixed (a delta-function prior) so contribute no
# prior term. Because _cosmo_cov restricts F and F_prior to the same sub-block, "prior on all params"
# expresses exactly that coupling -- there is one knob (which params are in the model), not two.
# The other PRIOR_MODES deliberately break the coupling and are diagnostics, emitted under
# prior_variations/ so they do not clutter the headline result.
HEADLINE_PRIOR = "all"

# the three forecast modes, in report/plot order. Machine keys are underscore-safe; the human titles
# use the "+" the user thinks in. Each mode marginalizes over a widening parameter set:
#   fixed          -- astro AND nuisance held fixed -> cosmo-only Fisher block
#   astro          -- marginalize over the probe's astro parameters
#   astro_nuisance -- marginalize over astro AND nuisance (every constrained non-cosmo parameter)
MODES = ["fixed", "astro", "astro_nuisance"]
MODE_TITLE = {
    "fixed": "astro + nuisance fixed (cosmo only)",
    "astro": "astro marginalized",
    "astro_nuisance": "astro + nuisance marginalized",
}


def build_prior_fisher(params, conf, which):
    """Diagonal prior Fisher F_prior encoding the analysis prior, + the equivalent sigma per param.

    The analysis prior (msfm.utils.prior.in_grid_prior) is a hard TOP-HAT: the box intervals in
    analysis.grid.priors, plus an Om-s8 convex hull and a w0(Om) threshold. A Fisher forecast is
    Gaussian by construction, so the box is encoded as its VARIANCE-MATCHED Gaussian: a uniform
    [a,b] has variance (b-a)^2/12, hence F_prior_ii = 12/(b-a)^2. Intervals come from
    msfm.utils.parameters.get_prior_intervals (the repo's single source of truth), not hardcoded.

    The hull and w0 threshold are non-Gaussian and cosmo-space; they can only cut probability off
    the box, i.e. only TIGHTEN the constraint, so omitting them keeps the forecast conservative.

    Units: the prior interval and the derivative step must share units. They do -- the config is
    log10 throughout for bary_Mc (prior [12,15], fiducial 13.82, step 0.1), matching the log10
    derivative from the tfrecords. (Marginalization alone is invariant under a constant rescaling
    of a nuisance, but a PRIOR is not, so this is the one place bary_Mc's log10 convention bites.)

    which: "all"       -- prior on every parameter (most faithful to the analysis; note it also
                          tightens cosmo, so part of the reported sigma is prior, not data)
           "noncosmo"  -- prior on the non-cosmo parameters (astro + nuisance) only; cosmo sigmas
                          stay pure data constraints
           "none"      -- no prior (Fisher is then singular along informationless directions)
    """
    intervals = parameters.get_prior_intervals(params, conf)  # (n_params, 2), config order
    sigma = (intervals[:, 1] - intervals[:, 0]) / np.sqrt(12.0)
    F_prior = np.zeros((len(params), len(params)))
    for i, p in enumerate(params):
        if which == "none" or (which == "noncosmo" and p in COSMO_PARAMS):
            continue
        F_prior[i, i] = 1.0 / sigma[i] ** 2
    return F_prior, dict(zip(params, sigma))


def all_params_from_config(perturbations):
    """Every param with a finite-difference perturbation in the dataset, cosmo first then the rest in
    config order. This is the full set we compute derivatives for; the non-cosmo ones split into the
    astro and nuisance classes per probe (e.g. Om,s8,H0,Ob,ns,w0,bary_*,Aia,n_Aia,bg1-4)."""
    rest = [p for p in perturbations if p not in COSMO_PARAMS]
    return list(COSMO_PARAMS) + rest


def _pert_labels(params):
    """The tfrecord field stems: 'fiducial' + delta_{param}_{m,p} for each param."""
    labels = ["fiducial"]
    for p in params:
        labels += [f"delta_{p}_m", f"delta_{p}_p"]
    return labels


def _flatten_pairs(arr, bin_indices):
    """Select tomographic pairs and flatten (n_bins, n_pairs) bin-major, exactly like
    cls_preprocessing.preprocess_obs_hard_rebinned: arr[..., :, bin_indices].reshape(..., -1)."""
    sub = arr[..., :, bin_indices]
    return sub.reshape(sub.shape[:-2] + (-1,))


def load_scale_cuts(scales_config, n_z_lensing, n_z_clustering):
    """Per-z-bin (l_min, l_max), lensing bins then clustering bins, from a scales yaml."""
    with open(scales_config) as f:
        sc = yaml.safe_load(f)["scale_cuts"]
    l_min = list(sc["lensing"]["l_min"]) + list(sc["clustering"]["l_min"])
    l_max = list(sc["lensing"]["l_max"]) + list(sc["clustering"]["l_max"])
    assert len(l_min) == len(l_max) == n_z_lensing + n_z_clustering
    return l_min, l_max


def compute_rebinned_covariance(fiducial_h5, W, chunk=4000):
    """Rebin fiducial_cls.h5:cls/raw (N,1536,36) -> (N,16,36) with weight tensor W, chunked."""
    with h5py.File(fiducial_h5, "r") as f:
        raw = f["cls/raw"]
        n, n_ell, n_pairs = raw.shape
        assert (n_ell, n_pairs) == (W.shape[0], W.shape[2]), (raw.shape, W.shape)
        Xb = np.empty((n, W.shape[1], n_pairs), dtype=np.float64)
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            block = raw[i0:i1].astype(np.float64)  # (c,1536,36)
            Xb[i0:i1] = np.einsum("nlc,lkc->nkc", block, W, optimize=True)
            LOGGER.info(f"covariance rebin: {i1}/{n}")
    return Xb  # (N, 16, 36)


def compute_rebinned_derivative_means(tfr_files, labels, W, n_noise, n_pairs, n_z_lensing, n_z_clustering):
    """Stream fiducial tfrecords, decode only the cl_{label} fields, and accumulate the mean
    over all (signal x noise) of the rebinned (16,36) Cls, per label.

    Returns: means (dict label -> (16,36)), n_seen (int, number of signal x noise realizations)."""
    n_bins, n_ell = W.shape[1], W.shape[0]
    # running sum of rebinned Cls per label
    sums = {lab: np.zeros((n_bins, n_pairs), dtype=np.float64) for lab in labels}
    n_seen = 0

    def _parse(ex):
        # pass ALL shapes explicitly (like msfm.fiducial_pipeline): the parser's internal
        # get_cross_bin_indices does range(n_z_metacal + n_z_maglim), which needs python ints,
        # not the record's shape tensors, inside a graph-mode .map.
        return tfrecords.parse_inverse_fiducial(
            ex,
            labels,
            range(n_noise),
            None,  # n_pix (maps not returned)
            n_z_lensing,  # n_z_metacal
            n_z_clustering,  # n_z_maglim
            n_noise,
            n_ell,  # n_cls (the ell axis)
            n_pairs,  # n_z_cross
            True,  # with_lensing
            True,  # with_clustering
            False,  # return_maps
            True,  # return_cls
        )

    # Each fiducial record is ~0.5 GB (maps interleaved with the Cls we want), so bound the reader
    # concurrency: AUTOTUNE on a 288-core node spawns hundreds of in-flight ~1 GB parses and OOMs.
    # The job is IO-bound, so a handful of parallel reads already saturates the filesystem.
    ds = tf.data.TFRecordDataset(tfr_files, num_parallel_reads=8)
    ds = ds.map(_parse, num_parallel_calls=4).prefetch(4)

    t0 = time.time()
    n_ex = 0
    for out in ds:
        for lab in labels:
            cl = out[f"cl_{lab}"].numpy()  # (n_noise, 1536, 36)
            reb = np.einsum("nlc,lkc->nkc", cl.astype(np.float64), W, optimize=True)  # (n_noise,16,36)
            sums[lab] += reb.sum(axis=0)
        n_seen += n_noise
        n_ex += 1
        if n_ex % 200 == 0:
            LOGGER.info(f"derivatives: {n_ex} examples ({n_seen} realizations), {time.time()-t0:.0f}s")

    means = {lab: sums[lab] / n_seen for lab in labels}
    LOGGER.info(f"derivatives: done, {n_ex} examples, {n_seen} realizations in {time.time()-t0:.0f}s")
    return means, n_seen


def build_derivative_grid(means, params, steps):
    """Central finite differences per param -> dict param -> (16,36) rebinned derivative.
    Mirrors deep_lss/utils/delta_loss.py and estimators.py: (plus - minus)/(2*step)."""
    d = {}
    for p in params:
        d[p] = (means[f"delta_{p}_p"] - means[f"delta_{p}_m"]) / (2.0 * steps[p])
    return d


def hartlap(n_sim, d):
    """Hartlap (2007) debiasing factor for an inverse sample covariance."""
    return (n_sim - d - 2.0) / (n_sim - 1.0)


def fisher_forecast(Xb, dgrid, probe_flags, params_full, n_z_lensing, n_z_clustering):
    """For one probe: build C from the rebinned realizations, D from the derivative grid,
    and return the Fisher matrix over params_full plus diagnostics."""
    bin_indices, _ = cross_statistics.get_cross_bin_indices(
        n_z_lensing=n_z_lensing,
        n_z_clustering=n_z_clustering,
        with_lensing=probe_flags["with_lensing"],
        with_clustering=probe_flags["with_clustering"],
        with_cross_z=probe_flags["with_cross_z"],
        with_cross_probe=probe_flags["with_cross_probe"],
    )

    # data vector: covariance realizations and derivatives, same flatten order
    X = _flatten_pairs(Xb, bin_indices)  # (N, d)
    D = np.stack([_flatten_pairs(dgrid[p], bin_indices) for p in params_full], axis=0)  # (n_par, d)

    n_sim, d = X.shape
    C = np.cov(X, rowvar=False)  # (d, d)

    # The data vector mixes lensing Cls (~1e-8) and clustering Cls (~counts), so C spans ~14 orders
    # of magnitude and cond(C) is huge purely from scale. The Fisher F = D^T C^-1 D is invariant under
    # per-feature rescaling d -> d/s, so invert the CORRELATION matrix R = C/(s s^T) instead: same
    # answer, but cond(R) reflects only true degeneracies and the inverse is numerically stable.
    s = np.sqrt(np.diag(C))
    R = C / np.outer(s, s)
    cond_C = np.linalg.cond(C)
    cond_R = np.linalg.cond(R)
    # positive-definite check on the well-scaled correlation matrix
    np.linalg.cholesky(R)
    h = hartlap(n_sim, d)
    Rinv = h * np.linalg.inv(R)

    Dw = D / s[None, :]  # whitened derivatives
    F = Dw @ Rinv @ Dw.T  # == h * D C^-1 D^T, mirrors estimators.py F = J^T C^-1 J
    return dict(
        F=F, C=C, D=D, bin_indices=np.asarray(bin_indices), d=d, n_sim=n_sim, cond=cond_C, cond_R=cond_R, hartlap=h
    )


def s8_jacobian(fid, params3):
    """Gradient of S8 = s8 * sqrt(Om/0.3) wrt (Om, s8, w0) at the fiducial."""
    Om, s8 = fid["Om"], fid["s8"]
    g = np.zeros(3)
    g[params3.index("Om")] = s8 / (2.0 * np.sqrt(0.3 * Om))
    g[params3.index("s8")] = np.sqrt(Om / 0.3)
    # dS8/dw0 = 0
    return g


def stable_inv(F):
    """Invert a Fisher matrix after whitening by its diagonal. Params span very different scales
    (e.g. step H0=2.0 vs Om=0.01), so raw inv(F) is ill-conditioned; F_ij/sqrt(F_ii F_jj) is not,
    and the result is exact (a similarity transform, not an approximation)."""
    d = np.sqrt(np.diag(F))
    Ft = F / np.outer(d, d)
    return np.linalg.inv(Ft) / np.outer(d, d)


def _cosmo_cov(F_all, F_prior, params_all, use_params):
    """Cosmo (Om,s8,w0) covariance for one forecast mode: restrict the full Fisher to use_params,
    add the prior, invert (marginalizing over any astro/nuisance params in use_params), and take the
    cosmo block. With use_params == COSMO_PARAMS this is the fixed (cosmo-only) forecast.

    Params NOT in use_params are held fixed, i.e. they get a delta-function prior, so no prior term
    of theirs enters -- the prior is restricted to the same sub-block as the Fisher."""
    idx = [params_all.index(p) for p in use_params]
    F_sub = F_all[np.ix_(idx, idx)] + F_prior[np.ix_(idx, idx)]
    cov_sub = stable_inv(F_sub)
    cidx = [use_params.index(p) for p in COSMO_PARAMS]
    return cov_sub[np.ix_(cidx, cidx)]


def _cov_stats(cov, fid):
    """sigma / correlation / S8 / FoM summary for a 3x3 cosmo covariance."""
    sig = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sig, sig)
    g = s8_jacobian(fid, COSMO_PARAMS)
    sig_S8 = float(np.sqrt(g @ cov @ g))
    fom = {
        "Om_s8": float(np.linalg.det(cov[np.ix_([0, 1], [0, 1])]) ** -0.5),
        "Om_w0": float(np.linalg.det(cov[np.ix_([0, 2], [0, 2])]) ** -0.5),
        "s8_w0": float(np.linalg.det(cov[np.ix_([1, 2], [1, 2])]) ** -0.5),
        "Om_s8_w0": float(np.linalg.det(cov) ** -0.5),
    }
    return dict(cov=cov, sigma=dict(zip(COSMO_PARAMS, sig)), sigma_S8=sig_S8, corr=corr, fom=fom)


def classify_params(F_all, params_all, probe_astro):
    """Split this probe's parameters into the three classes (see the module header), from the full
    Fisher's diagonal. Returns (astro, nuisance, dropped) as ordered lists.

    * astro    -- the probe's physical astro parameters (probe_astro), intersected with the
                  constrained set (a defensive no-op: a probe's astro params always carry info).
    * nuisance -- constrained, non-cosmo, non-astro (H0, Ob, ns, bary_Mc, bary_nu). Derived, not
                  hardcoded, so it tracks the config's parameter set.
    * dropped  -- non-cosmo parameters with an EXACTLY-zero derivative for this probe (e.g. galaxy
                  bias in a lensing-only data vector -- the forward model never applies it). Not part
                  of the probe's model, so they are reported rather than carried as empty rows/cols.
    """
    diagF = np.diag(F_all)
    scale = diagF.max()
    noncosmo = [p for p in params_all if p not in COSMO_PARAMS]
    constrained = [p for p in noncosmo if diagF[params_all.index(p)] > 1e-12 * scale]
    astro = [p for p in probe_astro if p in constrained]
    nuisance = [p for p in constrained if p not in astro]
    dropped = [p for p in noncosmo if p not in constrained]
    return astro, nuisance, dropped


def summarize(F_all, F_prior, params_all, probe_astro, fid):
    """Return the fixed / astro / astro_nuisance cosmo forecasts from the full Fisher, plus the
    parameter classification (astro / nuisance / dropped) for this probe.

    * fixed          -- astro AND nuisance held fixed (cosmo-only Fisher block).
    * astro          -- marginalize over the probe's astro parameters.
    * astro_nuisance -- marginalize over astro AND nuisance, i.e. every constrained non-cosmo param.

    Note on near-zero (as opposed to exactly-zero) derivatives, e.g. bary_Mc for clustering, where
    baryonic feedback does not measurably move galaxy clustering at l<~300: marginalizing over such
    a direction does NOT decouple from cosmology. The correction to the cosmo Fisher,
    (D_cos C^-1 e)^2 / (e C^-1 e), is scale-invariant in the derivative e, so as e -> 0 it still
    projects out a whole (essentially random) direction of cosmological information. That is a
    numerical pathology, not physics, and it is why the prior matters: a nuisance the data cannot
    constrain becomes prior-dominated instead of running away. See prior_variations/*_prior_none."""
    astro, nuisance, dropped = classify_params(F_all, params_all, probe_astro)

    use = {
        "fixed": list(COSMO_PARAMS),
        "astro": list(COSMO_PARAMS) + astro,
        "astro_nuisance": list(COSMO_PARAMS) + astro + nuisance,
    }
    out = {}
    for name in MODES:
        cov = _cosmo_cov(F_all, F_prior, params_all, use[name])
        stats = _cov_stats(cov, fid)
        stats["marg_params"] = [p for p in use[name] if p not in COSMO_PARAMS]
        out[name] = stats
    return out, dict(astro=astro, nuisance=nuisance, dropped=dropped)


# Gaussian confidence ellipses, standard 2D cosmology convention: "1/2 sigma" means the contour
# enclosing 68%/95% of the 2D probability, i.e. delta-chi2 = 2.30 / 6.17 for 2 dof. Note this is NOT
# the delta-chi2 = 1 ellipse: the 1-sigma ellipse below extends to 1.52 sigma along each axis, so the
# reported (marginalized, 1D) sigmas are not the half-widths of these contours.
ELLIPSE_LEVELS = [(np.sqrt(2.30), 1.6, 1.0), (np.sqrt(6.17), 1.0, 0.55)]  # (scale, lw, alpha)
ELLIPSE_NOTE = "1,2 sigma ellipses (68%,95% of 2D)"


def confidence_ellipse(ax, mean, cov2, **kw):
    """Draw the 68% and 95% (1- and 2-sigma) Gaussian confidence ellipses of a 2x2 covariance."""
    import matplotlib.patches as mpatches

    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for scale, lw, alpha in ELLIPSE_LEVELS:
        w, h = 2 * scale * np.sqrt(vals)
        ax.add_patch(mpatches.Ellipse(mean, w, h, angle=theta, fill=False, lw=lw, alpha=alpha, **kw))


def _triangle(entries, fid, out_png, suptitle):
    """Triangle plot of the (Om,s8,w0) Gaussian forecasts in `entries`, a list of
    (label, cov, color) -- one curve/ellipse per entry. Both plot flavours below are just
    different groupings of the same per-(probe,mode) covariances: overlay probes at fixed mode,
    or overlay modes at fixed probe."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pnames = COSMO_PARAMS
    fidv = [fid[p] for p in pnames]
    npar = len(pnames)

    fig, axes = plt.subplots(npar, npar, figsize=(9, 9))
    for i in range(npar):
        for j in range(npar):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                for label, cov, color in entries:
                    s = np.sqrt(cov[i, i])
                    xs = np.linspace(fidv[i] - 4 * s, fidv[i] + 4 * s, 200)
                    ax.plot(xs, np.exp(-0.5 * ((xs - fidv[i]) / s) ** 2), color=color, label=label)
                ax.axvline(fidv[i], color="k", lw=0.6, ls=":")
                if i == 0:
                    ax.legend(fontsize=7, loc="upper right")
            else:
                for label, cov, color in entries:
                    sub = cov[np.ix_([j, i], [j, i])]
                    confidence_ellipse(ax, (fidv[j], fidv[i]), sub, edgecolor=color)
                ax.plot(fidv[j], fidv[i], "k+", ms=8)
            if i == npar - 1:
                ax.set_xlabel(pnames[j])
            if j == 0 and i > 0:
                ax.set_ylabel(pnames[i])
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    LOGGER.info(f"saved plot -> {out_png}")


def _prior_note(prior_mode, headline=False):
    """Human description of the prior treatment for a plot title. The headline result couples the
    prior to the model (every included param gets it), so it is described as such rather than by the
    machine mode name -- it is the default, not one option among three."""
    if headline:
        return "analysis prior on every included param"
    return {
        "all": "analysis prior on all params",
        "noncosmo": "analysis prior on astro+nuisance (cosmo free)",
        "none": "NO prior",
    }[prior_mode]


def make_plot(results, fid, out_png, mode, scales_name, n_bins, prior_mode, headline=False):
    """Triangle plot overlaying the three probes for ONE forecast mode (fixed / astro /
    astro_nuisance). The mode is stated in the title and encoded in the filename."""
    colors = {"lensing": "C0", "clustering": "C1", "combined": "C2"}
    entries = [(probe, res["summary"][mode]["cov"], colors[probe]) for probe, res in results.items()]
    _triangle(
        entries,
        fid,
        out_png,
        f"Fisher forecast -- {MODE_TITLE[mode]}\n"
        f"hard_rebinned {scales_name}, {n_bins} ell bins/pair, "
        f"{_prior_note(prior_mode, headline)}, {ELLIPSE_NOTE}",
    )


def make_probe_plot(results, fid, out_png, probe, scales_name, n_bins, prior_mode, headline=False):
    """Triangle plot overlaying the three forecast modes for ONE probe: how much the contours open
    up as the marginalization widens (fixed -> astro -> astro+nuisance). The legend spells out which
    params each mode marginalizes over, since the astro/nuisance split is probe-dependent."""
    res = results[probe]
    colors = {"fixed": "C3", "astro": "C0", "astro_nuisance": "C2"}
    entries = []
    for mode in MODES:
        s = res["summary"][mode]
        marg = s["marg_params"]
        # astro_nuisance lists up to 11 params -- wrap so the legend stays inside the panel
        names = textwrap.fill(",".join(marg), 34) if marg else "none"
        entries.append((f"{MODE_TITLE[mode]}\n  ({names})", s["cov"], colors[mode]))
    dropped = res["classes"]["dropped"]
    dropped_note = f"\n(no info for this probe, dropped: {','.join(dropped)})" if dropped else ""
    _triangle(
        entries,
        fid,
        out_png,
        f"Fisher forecast -- {probe}: marginalization comparison\n"
        f"hard_rebinned {scales_name}, {n_bins} ell bins/pair, "
        f"{_prior_note(prior_mode, headline)}, {ELLIPSE_NOTE}{dropped_note}",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="/iopsstor/scratch/cscs/athomsen/deep_lss/data/v17/baseline")
    ap.add_argument(
        "--msfm_config",
        default="/users/athomsen/dlss/repos/multiprobe-simulation-forward-model/configs/v17/baseline.yaml",
    )
    ap.add_argument("--scales_config", default="/users/athomsen/dlss/repos/y3-deep-lss/configs/scales/lmax_1024.yaml")
    ap.add_argument("--cls_n_bins", type=int, default=16)
    ap.add_argument(
        "--prior_variations",
        default="noncosmo,none",
        help="comma-separated list of EXTRA, decoupled prior treatments to report as "
        "diagnostics in the prior_variations/ subfolder. The headline result always "
        "uses the coupled prior (= 'all': every included param carries the analysis "
        "prior) and needs no flag. Each variation is free -- it reuses the same "
        "prior-free Fisher and only redoes the final <=14x14 inversion. Choices: "
        "'noncosmo' = prior on astro+nuisance only, so cosmo sigmas are pure data "
        "constraints (shows how much of the headline is prior); 'none' = prior-free "
        "(informationless directions blow up -- a pathology demo, not a result); "
        "'all' = re-emit the headline under a tagged name. Pass '' for headline only.",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="base dir; the run's own nb<N> subfolder is appended under it. Default mirrors "
        "the dataset layout: {data_dir with /data/->/runs/}/fisher_cls/{scale}",
    )
    args = ap.parse_args()

    prior_modes = [m.strip() for m in args.prior_variations.split(",") if m.strip()]
    unknown = [m for m in prior_modes if m not in PRIOR_MODES]
    assert not unknown, f"unknown prior mode(s) {unknown}, choose from {PRIOR_MODES}"

    scales_name = os.path.splitext(os.path.basename(args.scales_config))[0]
    if args.out_dir is None:
        # mirror the input dataset layout: .../runs/<version>/<subversion>/fisher_cls/<scale>
        args.out_dir = os.path.join(args.data_dir.replace("/data/", "/runs/"), "fisher_cls", scales_name)
    # one folder per run (a run == one ell-bin count), so the ell-bin sweep does not share a dir.
    # The prior modes all come from a single run's Fisher, so they stay as a filename tag within it.
    args.out_dir = os.path.join(args.out_dir, f"nb{args.cls_n_bins}")
    os.makedirs(args.out_dir, exist_ok=True)
    LOGGER.info(f"scale={scales_name}  out_dir={args.out_dir}")
    LOGGER.info(f"headline prior='{HEADLINE_PRIOR}' (coupled); variations={prior_modes or 'none'}")
    conf = files.load_config(args.msfm_config)

    n_z_lensing = len(conf["survey"]["metacal"]["z_bins"])
    n_z_clustering = len(conf["survey"]["maglim"]["z_bins"])
    n_side = conf["analysis"]["n_side"]
    n_ell = 3 * n_side
    n_pairs = (n_z_lensing + n_z_clustering) * (n_z_lensing + n_z_clustering + 1) // 2
    n_noise = conf["analysis"]["fiducial"]["n_noise_per_signal"]
    fid = {p: conf["analysis"]["fiducial"][p] for p in COSMO_PARAMS}

    perturbations = conf["analysis"]["fiducial"]["perturbations"]
    params_all = all_params_from_config(perturbations)
    labels = _pert_labels(params_all)
    steps = {p: perturbations[p] for p in params_all}
    LOGGER.info(f"params={params_all}")
    LOGGER.info(f"steps={steps}")

    # analysis prior, variance-matched from the config's top-hat intervals. The equivalent sigmas
    # are a property of the config intervals alone (the mode only decides which ones are applied),
    # so build them once here for the diagnostics below.
    _, prior_sigma = build_prior_fisher(params_all, conf, "all")
    LOGGER.info("prior-equivalent sigma per param: " + ", ".join(f"{p}={prior_sigma[p]:.4g}" for p in params_all))

    # scale cuts + rebinning weight tensor (reused from deep_lss)
    l_min_z, l_max_z = load_scale_cuts(args.scales_config, n_z_lensing, n_z_clustering)
    LOGGER.info(f"l_min_z={l_min_z} l_max_z={l_max_z}")
    W = _build_bin_weights_all_pairs(n_ell, args.cls_n_bins, n_z_lensing, n_z_clustering, l_min_z, l_max_z)

    # (1) covariance realizations, rebinned
    fiducial_h5 = os.path.join(args.data_dir, "cls", "fiducial_cls.h5")
    Xb = compute_rebinned_covariance(fiducial_h5, W)

    # (2) derivative means from the fiducial tfrecords
    tfr_files = sorted(glob.glob(os.path.join(args.data_dir, "tfrecords", "fiducial", "*.tfrecord")))
    LOGGER.info(f"reading {len(tfr_files)} fiducial tfrecord shards")
    means, n_seen = compute_rebinned_derivative_means(
        tfr_files, labels, W, n_noise, n_pairs, n_z_lensing, n_z_clustering
    )
    dgrid = build_derivative_grid(means, params_all, steps)

    # sanity: lensing kappa-kappa auto (pair 0) derivative wrt s8 should be positive
    ds8_00 = dgrid["s8"][:, 0]
    LOGGER.info(f"sanity dCl(kk,bin0)/ds8 sign: {np.sign(np.nanmean(ds8_00))} (expect +)")

    # (3) per-probe Fisher. This is the expensive, PRIOR-FREE part, so it is computed once per probe
    # and reused for every prior mode below; the prior only ever enters the final sub-block inversion.
    fishers = {}
    for probe, flags in PROBES.items():
        fp = fisher_forecast(Xb, dgrid, flags, params_all, n_z_lensing, n_z_clustering)
        fishers[probe] = fp
        LOGGER.info(
            f"[{probe}] d={fp['d']} n_sim={fp['n_sim']} cond(C)={fp['cond']:.2e} "
            f"cond(R)={fp['cond_R']:.2e} hartlap={fp['hartlap']:.4f}"
        )
        # parameter classification (prior-independent -- from the Fisher diagonal only), so compute
        # and log it once per probe here rather than repeating it for every prior mode.
        astro, nuisance, dropped = classify_params(fp["F"], params_all, PROBES[probe]["astro"])
        pclass = {p: "cosmo" for p in COSMO_PARAMS}
        pclass.update({p: "astro" for p in astro})
        pclass.update({p: "nuisance" for p in nuisance})
        pclass.update({p: "dropped" for p in dropped})
        LOGGER.info(f"  [{probe}] cosmo={COSMO_PARAMS} astro={astro} nuisance={nuisance} dropped={dropped}")
        # Which params does the DATA constrain vs which ride on the prior? Compare the Fisher's own
        # information to the prior's: F_pp * sigma_prior^2 << 1 means prior-dominated. Also compare
        # F_pp against the Monte-Carlo noise floor of the derivative: the mean over n_seen paired
        # realizations has residual noise, and if the derivative is pure noise then
        # E[F_pp] ~ d/(2*n_seen*step^2) (an upper bound -- seed pairing suppresses it further).
        # Prior-independent: it is a statement about the data, so it is logged once, not per mode.
        for p in params_all:
            i = params_all.index(p)
            F_pp = fp["F"][i, i]
            null = fp["d"] / (2.0 * n_seen * steps[p] ** 2)
            note = (
                "NO DATA INFO (not in this probe's model)"
                if F_pp <= 1e-12 * np.diag(fp["F"]).max()
                else (
                    "at/below MC-noise floor -> prior-dominated"
                    if F_pp < null
                    else "prior-dominated" if F_pp * prior_sigma[p] ** 2 < 1.0 else "data-dominated"
                )
            )
            LOGGER.info(
                f"  [{probe}] {p:<9} [{pclass[p]:>8}] F_pp={F_pp:11.4g}  "
                f"F_pp/noise_null={F_pp/null:9.1f}  "
                f"F_pp*sig_prior^2={F_pp * prior_sigma[p] ** 2:11.4g}  {note}"
            )

    # (4) forecasts, report and plots.
    #
    # HEADLINE (out_dir, untagged filenames): the prior is COUPLED to the model -- every parameter a
    # forecast includes carries the analysis prior, and only the cosmo params are reported (astro /
    # nuisance are marginalized over). This needs no special code path: _cosmo_cov restricts F and
    # F_prior to the same sub-block, so prior=all IS that coupling -- each mode automatically gets
    # the prior on exactly the params it includes, and a param held fixed contributes no prior term.
    #
    # VARIATIONS (prior_variations/, tagged): deliberately DEcouple the prior from the model
    # (cosmo left prior-free, or no prior anywhere). Diagnostics for how much of a headline sigma is
    # prior rather than data -- kept out of the headline dir so it stays uncluttered.
    runs = [(HEADLINE_PRIOR, args.out_dir, "", True)]
    if prior_modes:
        var_dir = os.path.join(args.out_dir, "prior_variations")
        os.makedirs(var_dir, exist_ok=True)
        runs += [(m, var_dir, f"_prior_{m}", False) for m in prior_modes]

    for prior_mode, odir, tag, headline in runs:
        F_prior, _ = build_prior_fisher(params_all, conf, prior_mode)
        LOGGER.info(f"=== {'HEADLINE' if headline else 'variation'}: prior={prior_mode} -> {odir} ===")
        results = {}
        for probe, fp in fishers.items():
            summary, classes = summarize(fp["F"], F_prior, params_all, PROBES[probe]["astro"], fid)
            results[probe] = dict(fisher=fp, summary=summary, classes=classes, params_full=params_all)
            LOGGER.info(
                f"[{probe}] astro_nuisance marg over {summary['astro_nuisance']['marg_params']} "
                f"(dropped, no info: {classes['dropped']})"
            )
            np.savez(
                os.path.join(odir, f"fisher_{probe}{tag}.npz"),
                F=fp["F"],
                C=fp["C"],
                D=fp["D"],
                bin_indices=fp["bin_indices"],
                params_full=np.array(params_all),
                cond=fp["cond"],
                cond_R=fp["cond_R"],
                hartlap=fp["hartlap"],
                F_prior=F_prior,
                prior_mode=prior_mode,
                prior_sigma=np.array([prior_sigma[p] for p in params_all]),
                astro_params=np.array(classes["astro"]),
                nuisance_params=np.array(classes["nuisance"]),
                dropped_params=np.array(classes["dropped"]),
                cov_fixed=summary["fixed"]["cov"],
                cov_astro=summary["astro"]["cov"],
                cov_astro_nuisance=summary["astro_nuisance"]["cov"],
            )

        # report + plots: one triangle per forecast mode (probes overlaid), and one per probe
        # (marginalization modes overlaid)
        print_report(results, fid, n_seen, scales_name, args.cls_n_bins, prior_mode, headline)
        for mode in MODES:
            out_png = os.path.join(odir, f"fisher_ellipses_{mode}{tag}.png")
            make_plot(results, fid, out_png, mode, scales_name, args.cls_n_bins, prior_mode, headline)
        for probe in results:
            out_png = os.path.join(odir, f"fisher_ellipses_probe_{probe}{tag}.png")
            make_probe_plot(results, fid, out_png, probe, scales_name, args.cls_n_bins, prior_mode, headline)
    LOGGER.info(
        f"headline outputs in {args.out_dir}"
        + (f"; prior variations in {args.out_dir}/prior_variations" if prior_modes else "")
    )


def print_report(results, fid, n_seen, scales_name, n_bins, prior_mode, headline=False):
    lines = []
    lines.append("=" * 90)
    lines.append(
        f"{'HEADLINE' if headline else 'PRIOR VARIATION (diagnostic)'} -- "
        f"Fisher forecast (Om, s8, w0), hard_rebinned {scales_name}, {n_bins} ell bins/pair, "
        f"v17/baseline"
    )
    lines.append(f"fiducial: Om={fid['Om']}, s8={fid['s8']}, w0={fid['w0']}; " f"derivative realizations={n_seen}")
    lines.append(
        f"prior: {_prior_note(prior_mode, headline)} (analysis.grid.priors top-hat, "
        f"variance-matched to a Gaussian: sigma=(b-a)/sqrt(12))"
    )
    if headline:
        lines.append(
            "       the prior is coupled to the model: each mode below priors exactly the " "params it includes."
        )
        lines.append(
            "       NOTE it also tightens cosmo, so part of each sigma/FoM is prior, not "
            "data. See prior_variations/*_prior_noncosmo for the pure-data cosmo sigmas."
        )
    lines.append(
        "       parameter classes: cosmo (reported) / astro (probe's IA or bias) / "
        "nuisance (barely-constrained rest) / dropped (not in this probe's model)"
    )
    lines.append("=" * 90)
    for probe, res in results.items():
        fp = res["fisher"]
        cl = res["classes"]
        lines.append(
            f"\n### {probe}  (data vector d={fp['d']}, N_sim={fp['n_sim']}, "
            f"cond(C)={fp['cond']:.2e}, cond(R)={fp['cond_R']:.2e}, Hartlap={fp['hartlap']:.4f})"
        )
        lines.append(
            f"     astro={cl['astro']}  nuisance={cl['nuisance']}"
            + (f"  dropped={cl['dropped']}" if cl["dropped"] else "")
        )
        for mode in MODES:
            s = res["summary"][mode]
            marg = s["marg_params"]
            tag = "(none)" if not marg else ",".join(marg)
            lines.append(f"  {mode:>16} [{tag}]:")
            sig = s["sigma"]
            lines.append(
                f"        sig(Om)={sig['Om']:.4f}  sig(s8)={sig['s8']:.4f}  "
                f"sig(w0)={sig['w0']:.4f}  sig(S8)={s['sigma_S8']:.4f}"
            )
            lines.append(
                f"        FoM: Om-s8={s['fom']['Om_s8']:.1f}  "
                f"Om-w0={s['fom']['Om_w0']:.1f}  s8-w0={s['fom']['s8_w0']:.1f}  "
                f"3D={s['fom']['Om_s8_w0']:.1f}"
            )
        # cosmo error degradation from marginalization, relative to the fixed (cosmo-only) forecast
        sf = res["summary"]["fixed"]["sigma"]
        for mode in ("astro", "astro_nuisance"):
            sm = res["summary"][mode]["sigma"]
            ratio = {p: sm[p] / sf[p] for p in COSMO_PARAMS}
            lines.append(
                f"     degradation ({mode}/fixed): "
                f"Om={ratio['Om']:.2f}x  s8={ratio['s8']:.2f}x  w0={ratio['w0']:.2f}x"
            )
    report = "\n".join(lines)
    print(report)
    return report


if __name__ == "__main__":
    main()

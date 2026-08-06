# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Overlay the Gaussian Fisher forecast on the actual (non-Gaussian) SBI posterior chains, per probe,
in the (Om, s8, w0) plane. This is the sanity check that the Fisher forecast and the real analysis
have comparable constraining power -- if the ellipse and the chain contour are wildly different
sizes, something is wrong with one of them.

For each probe (lensing / clustering / combined) it draws a triangle plot overlaying:
  * the Fisher forecast -- cov_astro_nuisance from fisher_<probe>.npz (the headline mode). This is
    the RIGHT Fisher mode to compare against: the network infers only cosmo + astro, while the
    nuisances (H0, Ob, ns, bary_Mc, bary_nu) are varied in the training grid but never predicted, so
    the chain posterior on (Om, s8, w0) is implicitly marginalized over BOTH astro and nuisance --
    which is exactly what astro_nuisance marginalizes.
  * the chain -- the first three columns (Om, s8, w0) of chain_fiducial_bench_mean.npy, whose
    ordering is the run's grid.params [Om, s8, w0, <astro...>]; taking the cosmo columns marginalizes
    over astro automatically.
Both are drawn at 1 and 2 sigma (68% / 95% of the 2D probability).

IMPORTANT the chain and the Fisher must come from the SAME scale cut or the sizes are not
comparable. The v3_mlp Cls runs use 8wl,32gc (verified from their configs.yaml scale_cuts:
lensing l_max=[589,863,1159,1382], clustering l_max=[133,195,255,305]), so point --fisher_dir at the
8wl,32gc Fisher output.

matplotlib only (no TF / deep_lss / scipy import), so it is cheap -- run it in the tf_env (which has
matplotlib) after the Fisher job has written its npz. The contour maths is numpy-only.
"""

import os
import argparse

import numpy as np

# fiducial (matches the msfm v17/baseline config) and the reported cosmo params, in chain-column order
COSMO_PARAMS = ["Om", "s8", "w0"]
FIDUCIAL = {"Om": 0.26, "s8": 0.84, "w0": -1.0}

# per probe: which Fisher npz to read and which chain to overlay. The chain path is completed with
# --chain_base / --chain_tag / --chain_flow so a different run set can be pointed at without edits.
PROBES = ["lensing", "clustering", "combined"]

# Gaussian 2-dof confidence levels, same convention as fisher_cls.py: 68% / 95% of the 2D
# probability are delta-chi2 = 2.30 / 6.17. For the Fisher ellipse these are analytic; for the chain
# they become the sample-density thresholds enclosing 68% / 95% of the samples.
LEVELS = [(0.68, np.sqrt(2.30)), (0.95, np.sqrt(6.17))]


def _gauss_blur_2d(H, sigma):
    """Separable Gaussian blur of a 2D histogram, numpy only (avoids a scipy dependency)."""
    if sigma <= 0:
        return H
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    H = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, H)
    H = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, H)
    return H


def _chain_levels(H):
    """Density thresholds of a (smoothed) 2D histogram that enclose 68% / 95% of the samples.
    Standard corner/getdist recipe: sort the histogram descending, take the cumulative mass, and
    read off the height at which it first reaches each target fraction. Returned ascending (so the
    95% threshold, which is the lower height, comes first) as matplotlib.contour requires."""
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    heights = []
    for frac, _ in LEVELS:
        idx = int(np.searchsorted(csum, frac))
        heights.append(flat[min(idx, len(flat) - 1)])
    # ascending + de-duplicated (a degenerate 1-sample-wide contour can collide)
    return sorted(set(heights))


def _draw_fisher_ellipse(ax, mean, cov2, color):
    import matplotlib.patches as mpatches

    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for (_, scale), lw, alpha in zip(LEVELS, (1.7, 1.0), (1.0, 0.6)):
        w, h = 2 * scale * np.sqrt(vals)
        ax.add_patch(
            mpatches.Ellipse(mean, w, h, angle=theta, fill=False, edgecolor=color, lw=lw, alpha=alpha, ls="--")
        )


def _draw_chain_contour(ax, x, y, color, bins, smooth):
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    H = _gauss_blur_2d(H, smooth)
    levels = _chain_levels(H)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    # H is (nx, ny) with x along axis 0 -> transpose for contour's (row=y, col=x) convention
    ax.contour(xc, yc, H.T, levels=levels, colors=color, linewidths=[1.0, 1.7][: len(levels)], linestyles="-")


def make_comparison(probe, cov_fisher, chain_cosmo, out_png, scales_name, n_bins, recenter=False):
    """Triangle plot of the Fisher ellipse vs the chain contour in (Om, s8, w0).

    recenter=False -- honest view: Fisher at the fiducial, chain at its own mean, so any projection
                      offset between the two is visible along with the size.
    recenter=True  -- pure size comparison: the chain is shifted so its mean sits on the fiducial
                      (where the Fisher ellipse already is), removing the offset so only extent/shape
                      differ. The shift is a rigid translation -- the chain's size and shape are
                      untouched.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    npar = len(COSMO_PARAMS)
    fidv = [FIDUCIAL[p] for p in COSMO_PARAMS]
    c_fish, c_chain = "C3", "C0"
    if recenter:
        chain_cosmo = chain_cosmo - chain_cosmo.mean(axis=0) + np.array(fidv)

    fig, axes = plt.subplots(npar, npar, figsize=(8.5, 8.5))
    for i in range(npar):
        for j in range(npar):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                # 1D marginals: chain histogram + Fisher Gaussian (mean = fiducial), both normalised
                xs_chain = chain_cosmo[:, i]
                ax.hist(xs_chain, bins=120, density=True, histtype="step", color=c_chain, lw=1.4)
                s = np.sqrt(cov_fisher[i, i])
                xs = np.linspace(fidv[i] - 4 * s, fidv[i] + 4 * s, 200)
                ax.plot(
                    xs,
                    np.exp(-0.5 * ((xs - fidv[i]) / s) ** 2) / (s * np.sqrt(2 * np.pi)),
                    color=c_fish,
                    ls="--",
                    lw=1.4,
                )
                ax.axvline(fidv[i], color="k", lw=0.6, ls=":")
                ax.set_yticks([])
            else:
                _draw_chain_contour(ax, chain_cosmo[:, j], chain_cosmo[:, i], c_chain, bins=80, smooth=1.2)
                sub = cov_fisher[np.ix_([j, i], [j, i])]
                _draw_fisher_ellipse(ax, (fidv[j], fidv[i]), sub, c_fish)
                ax.plot(fidv[j], fidv[i], "k+", ms=8)
            if i == npar - 1:
                ax.set_xlabel(COSMO_PARAMS[j])
            if j == 0 and i > 0:
                ax.set_ylabel(COSMO_PARAMS[i])

    chain_lbl = "SBI chain (v3_mlp), re-centered on fiducial" if recenter else "SBI chain (v3_mlp)"
    handles = [
        mlines.Line2D([], [], color=c_fish, ls="--", label="Fisher (astro+nuisance marg.)"),
        mlines.Line2D([], [], color=c_chain, ls="-", label=chain_lbl),
    ]
    # park the legend in the empty upper-right panel so it never collides with the suptitle
    axes[0, npar - 1].legend(handles=handles, loc="center", fontsize=10, frameon=False)
    ctitle = "size comparison (chain re-centered on fiducial)" if recenter else "as-measured (own centers)"
    fig.suptitle(
        f"Fisher vs SBI chain -- {probe}: {ctitle}\n"
        f"hard_rebinned {scales_name}, {n_bins} ell bins/pair, 1,2 sigma (68%,95% of 2D)"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"saved {out_png}")


def summarize_sizes(probe, cov_fisher, chain_cosmo):
    """Print the marginal 1-sigma widths side by side so 'similar sizes' is a number, not just a
    picture. Chain sigma = sample std of each cosmo column (marginal over astro)."""
    sig_f = np.sqrt(np.diag(cov_fisher))
    sig_c = chain_cosmo.std(axis=0)
    print(f"\n[{probe}] marginal 1-sigma (Fisher astro_nuisance  vs  chain sample std):")
    for k, p in enumerate(COSMO_PARAMS):
        print(
            f"    {p:<3} Fisher={sig_f[k]:8.4f}  chain={sig_c[k]:8.4f}  "
            f"ratio(chain/Fisher)={sig_c[k] / sig_f[k]:5.2f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fisher_dir",
        required=True,
        help="the Fisher headline output dir (…/fisher_cls/<scale>/nb<N>) holding "
        "fisher_<probe>.npz. Must be the SAME scale cut as the chains (8wl,32gc).",
    )
    ap.add_argument(
        "--chain_base",
        default="/users/athomsen/scratch/deep_lss/runs/v17/baseline/cls",
        help="root of the per-probe Cls SBI runs",
    )
    ap.add_argument("--chain_tag", default="v3_mlp", help="run tag under <chain_base>/<probe>/")
    ap.add_argument(
        "--chain_flow", default="ensemble_flow_1000000", help="flow subdir holding chain_fiducial_bench_mean.npy"
    )
    ap.add_argument("--chain_file", default="chain_fiducial_bench_mean.npy")
    ap.add_argument(
        "--out_dir", default=None, help="where to write the comparison plots; default = <fisher_dir>/chain_comparison"
    )
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.fisher_dir, "chain_comparison")
    os.makedirs(out_dir, exist_ok=True)

    # scale / bin count for the titles, recovered from the fisher_dir path (…/<scale>/nb<N>)
    n_bins = os.path.basename(args.fisher_dir.rstrip("/")).replace("nb", "")
    scales_name = os.path.basename(os.path.dirname(args.fisher_dir.rstrip("/")))

    for probe in PROBES:
        npz_path = os.path.join(args.fisher_dir, f"fisher_{probe}.npz")
        chain_path = os.path.join(args.chain_base, probe, args.chain_tag, args.chain_flow, args.chain_file)
        if not os.path.exists(npz_path):
            print(f"[{probe}] SKIP: no Fisher npz at {npz_path}")
            continue
        if not os.path.exists(chain_path):
            print(f"[{probe}] SKIP: no chain at {chain_path}")
            continue

        cov_fisher = np.load(npz_path, allow_pickle=True)["cov_astro_nuisance"]
        chain = np.load(chain_path)
        assert chain.shape[1] >= 3, f"{probe} chain has <3 columns: {chain.shape}"
        chain_cosmo = chain[:, :3].astype(np.float64)  # [Om, s8, w0]

        summarize_sizes(probe, cov_fisher, chain_cosmo)
        # both flavours: as-measured (own centers, shows offset) and re-centered (pure size compare)
        make_comparison(
            probe,
            cov_fisher,
            chain_cosmo,
            os.path.join(out_dir, f"fisher_vs_chain_{probe}.png"),
            scales_name,
            n_bins,
            recenter=False,
        )
        make_comparison(
            probe,
            cov_fisher,
            chain_cosmo,
            os.path.join(out_dir, f"fisher_vs_chain_{probe}_centered.png"),
            scales_name,
            n_bins,
            recenter=True,
        )

    print(f"\ncomparison plots in {out_dir}")


if __name__ == "__main__":
    main()

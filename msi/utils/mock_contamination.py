import glob
import os

import numpy as np
import seaborn as sns

from msfm.utils import logger
from msi.utils import plotting

LOGGER = logger.get_logger(__file__)

# Mock labels starting with these prefixes are not "contaminated systematics" mocks and are excluded
# from the comparison: cosmo_* are the large per-cosmology grid, Buzzard is a separate N-body suite.
_EXCLUDE_PREFIXES = ("cosmo", "Buzzard")

# {sigma, enclosed-probability level} pairs and {parametrization} variants for the four plots.
_SIGMA_LEVELS = [([0.235], "0.3sigma"), ([0.68], "1sigma")]
_PARAM_VARIANTS = [(False, ["Om", "s8", "w0"], "sigma8"), (True, ["Om", "S8", "w0"], "S8")]

# Auto axis ranges scale with the plotted contour level: a 2D contour enclosing probability `level`
# reaches `sqrt(-2 ln(1-level))` marginal sigmas on each axis (0.73 sigma at the 0.3-sigma level,
# 1.51 sigma at the 1-sigma level). We take the union over mocks of center +/- (radius + margin)*sigma,
# so the tight 0.3-sigma plot zooms in hard while the 1-sigma plot stays wider, and `margin` sigmas of
# headroom keep every mock's (smoothed) contour clear of the frame.
_RANGE_MARGIN = 1.0


def _plot_columns(chain, params, params_plot, use_S8):
    """Return the chain restricted to params_plot, in plot space (s8 -> S8 when requested)."""
    cols = []
    for p in params_plot:
        if p == "S8":
            cols.append(plotting.sigma8_to_S8(chain[:, params.index("s8")], chain[:, params.index("Om")]))
        else:
            cols.append(chain[:, params.index(p)])
    return np.column_stack(cols)


def _auto_ranges(chains, params, params_plot, use_S8, level):
    """Tight per-parameter axis ranges (keyed by plotted param name) for one contour `level`.

    Well-constrained params (Om, s8/S8) zoom in to a few sigma around the contour; unconstrained ones
    (w0) stay wide. The union across mocks plus the sigma margin guarantees no contour is clipped.
    """
    radius = np.sqrt(-2.0 * np.log(1.0 - level))
    half = radius + _RANGE_MARGIN
    cols = [_plot_columns(c, params, params_plot, use_S8) for c in chains]
    ranges = {}
    for j, p in enumerate(params_plot):
        los = [x[:, j].mean() - half * x[:, j].std() for x in cols]
        his = [x[:, j].mean() + half * x[:, j].std() for x in cols]
        ranges[p] = [min(los), max(his)]
    return ranges


def _load_mock_chains(model_dir, fidu_label):
    """Discover the per-mock {label}_mean chains saved by run_mcmc and split off the fiducial baseline.

    Returns (fidu_chain, fidu_log_probs, contam_labels, contam_chains). The fiducial entries are None
    when its chain is absent. Cosmo-grid and Buzzard chains are skipped.
    """
    fidu_chain, fidu_log_probs = None, None
    contam_labels, contam_chains = [], []

    suffix = "_mean.npy"
    for chain_file in sorted(glob.glob(os.path.join(model_dir, "chain_*_mean.npy"))):
        label = os.path.basename(chain_file)[len("chain_") : -len(suffix)]
        if label.startswith(_EXCLUDE_PREFIXES):
            continue
        chain = np.load(chain_file)
        if label == fidu_label:
            fidu_chain = chain
            lp_file = os.path.join(model_dir, f"log_probs_{label}_mean.npy")
            if os.path.exists(lp_file):
                fidu_log_probs = np.load(lp_file)
            else:
                LOGGER.warning(f"Missing {lp_file}; fiducial MAP marker will be skipped.")
        else:
            contam_labels.append(label)
            contam_chains.append(chain)
    return fidu_chain, fidu_log_probs, contam_labels, contam_chains


def run_mock_contamination(flow, params, msfm_conf, flow_conf, fidu_label="fiducial_bench"):
    """Overlay the posterior contours of every non-cosmo mock against the fiducial baseline.

    Mirrors deep_lss_paper/paper_2/pre-unblinding/3_mock_contamination.ipynb so it runs automatically
    as part of run_inference.py. Reads the per-mock {label}_mean chains saved by run_mcmc from
    flow.model_dir and writes four plots into flow.model_dir/unblinding_plots: the {sigma8, S8}
    parametrizations at the {0.3, 1} sigma contour levels. Mocks are labelled by the stem baked into
    their *_obs_maps.h5 filename. The fiducial mock is drawn as a black baseline with its MAP marker.
    """
    plot_dir = os.path.join(flow.model_dir, "unblinding_plots")
    os.makedirs(plot_dir, exist_ok=True)

    fidu_chain, fidu_log_probs, contam_labels, contam_chains = _load_mock_chains(flow.model_dir, fidu_label)

    if fidu_chain is None and not contam_chains:
        LOGGER.info("Mock contamination: no non-cosmo mock chains found; skipping stage.")
        return

    # fiducial first so it is the black baseline, then the contaminated mocks in the colorblind palette
    if fidu_chain is not None:
        chains = [fidu_chain, *contam_chains]
        plot_labels = ["fiducial", *contam_labels]
        colors = ["k", *sns.color_palette("colorblind", len(contam_chains))]
        linestyles = ["-", *["--"] * len(contam_chains)]
    else:
        LOGGER.warning(f"Fiducial chain '{fidu_label}' not found; overlaying contaminated mocks only.")
        chains = list(contam_chains)
        plot_labels = list(contam_labels)
        colors = list(sns.color_palette("colorblind", len(contam_chains)))
        linestyles = ["--"] * len(contam_chains)

    LOGGER.info(
        f"Mock contamination: fiducial={'yes' if fidu_chain is not None else 'no'}, "
        f"contaminated={contam_labels}"
    )

    for use_S8, params_plot, ptag in _PARAM_VARIANTS:
        for levels, stag in _SIGMA_LEVELS:
            ranges = _auto_ranges(chains, params, params_plot, use_S8, levels[0])
            de_kwargs = {"levels": levels, "smoothing_parameter1D": 0.2, "smoothing_parameter2D": 0.2}
            tri = plotting.plot_chains(
                chains=chains,
                params=params,
                conf=msfm_conf,
                plot_labels=plot_labels,
                use_S8=use_S8,
                group_params=True,
                colors=colors,
                ranges=ranges,
                scale_to_prior=False,
                include_prior=False,
                params_plot=params_plot,
                obs_cosmo=None,
                linestyles=linestyles,
                fills=False,
                show_legend=True,
                tri_kwargs={
                    "de_kwargs": de_kwargs,
                    "fill": False,
                    "scatter_kwargs": {"s": 50, "marker": "x", "zorder": 299},
                },
            )

            # fiducial MAP marker (convert the chain's s8 column to S8 for the S8 version, as in the notebook)
            if fidu_chain is not None and fidu_log_probs is not None:
                fchain = fidu_chain.copy()
                fparams = list(params)
                if use_S8:
                    s8i = params.index("s8")
                    fchain[:, s8i] = plotting.sigma8_to_S8(fchain[:, s8i], fchain[:, params.index("Om")])
                    fparams[s8i] = "S8"
                fidu_MAP = dict(
                    zip(params_plot, plotting.find_MAP(fchain, fidu_log_probs, fparams, params_plot, percentile=1))
                )
                tri.scatter(fidu_MAP, color="k", label="fiducial MAP")

            plot_file = os.path.join(plot_dir, f"3_mock_contamination_{ptag}_{stag}.png")
            tri.fig.savefig(plot_file, bbox_inches="tight", dpi=100)
            LOGGER.info(f"Saved {plot_file}")

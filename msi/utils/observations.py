import os

import numpy as np

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

# Restricted-w0 DES variant: w0 stays a free, sampled parameter but its flat prior is tightened to the
# non-phantom half, w0 > -1 (lower bound raised to -1, upper kept at the config value). Run automatically
# for every DES observation as a third chain alongside the wCDM and lambdaCDM (w0 = -1) chains.
W0_GT_M1_PRIOR = (-1.0, None)
W0_SUFFIX = "_w0gt-1"

# Combined restricted DES variant: w0 > -1 AND NLA (bta = 0). Run automatically for every DES
# observation of a probe that has bta among its inferred params (lensing / 2x2pt / combined). bta is
# dropped from the sampled space and fixed to 0 (delta-NLA/TATT -> standard NLA); the chain is thus in
# the reduced (bta-dropped) space. clustering has no bta, so this variant is skipped there.
NLA_SUFFIX = "_nla"

# Reference-prior DES variant, only possible for a flow conditioned on the EXTENDED parameter vector
# (run_inference --extend_params): replaces the implicit wide flat CosmoGrid marginalization of
# ns / Obh2 / H0 with the near-delta Gaussians shared by the DES Y3 SBI reference papers (the Gower
# Street analysis family: Jeffrey+24 2403.02314, Gatti+24 2405.10881, Williamson+26). Run
# automatically for every DES observation when ns/Ob/H0 are among the flow's params, as w0 > -1 + NLA
# and lambdaCDM + NLA chains -- the closest apples-to-apples analogues to the references' wCDM and
# LCDM results.
# Baryons are fixed at the fiducial only if the flow is also conditioned on them, which the
# production config is not; the references do not marginalize them at all.
REF_GAUSSIAN_PRIORS = {
    "ns": (0.9649, 0.0063),
    "Obh2": (0.02237, 0.00015),  # derived Ob * (H0/100)^2
    "H0": (70.22, 2.45),
}
REF_PRIOR_SUFFIX = "_refpriors"

# Mocks sampled as a product over their stack by default. Only the Buzzard flock: its 15 members are
# independent survey realizations, while the fiducial-family stacks (80 rows) would give an
# 80-fold-area posterior that answers no question in the paper.
MOCK_PRODUCT_DEFAULT = ("buzzard",)

# Fixed-extension mock variant, the default of the Buzzard recovery test: every chain of a matching
# mock (its _mean, each realization and the _stack product) is sampled once more with the flow's
# extension parameters (ns, Ob, H0) fixed at the mock's own truth. The data barely constrain them,
# and in the 15-fold product they otherwise drift into the prior edge along the Om-ns-H0
# degeneracy and drag Om with them. Written as chain_{key}_fixedext.npy beside the free chain, which
# stays what every other figure reads. Needs an extended flow and a truth for each fixed parameter.
FIXED_EXT_SUFFIX = "_fixedext"
MOCK_FIXED_EXT_DEFAULT = ("buzzard",)


def _ref_prior_kwargs(flow):
    """Sampler kwargs for the reference-prior variant, or None when the flow is not conditioned on the
    extended parameter vector. Baryon fiducials are read from the run's own msfm config."""
    if not all(p in flow.params for p in ("ns", "Ob", "H0")):
        return None
    fiducial = flow.conf["analysis"]["fiducial"]
    fixed = {p: fiducial[p] for p in ("bary_Mc", "bary_nu") if p in flow.params}
    return {"gaussian_priors": REF_GAUSSIAN_PRIORS, "fixed_params": fixed}


def des_variants(flow):
    """The DES posterior variants a run samples, as one ordered table.

    Each entry is ``(suffix, model_kwargs, variant_label)``. ``suffix`` names the chain file, and is
    what the batched path appends itself; ``variant_label`` is the only part the emcee path has to be
    told, because ``LikelihoodFlow.sample_posterior`` derives the lambdaCDM / w0gt-1 / nla suffix
    from the kwargs itself, in this same order. The first entry is the unrestricted wCDM chain.

    One table for both sampler backends and for the per-member stage, so a variant cannot exist on
    one path and not another -- it used to be spelled out once per backend.
    """
    params = getattr(flow, "params", [])
    # bta is not always sampled (v17 fixed it), and `nla` is a no-op without it -- so the suffix must
    # drop `_nla` too, or the batched path would name a file the emcee path spells differently.
    nla = NLA_SUFFIX if "bta" in params else ""

    variants = [
        ("", {}, None),
        ("_lambdaCDM", {"lambdaCDM": True}, None),
        (W0_SUFFIX, {"w0_prior": W0_GT_M1_PRIOR}, None),
    ]
    if "bta" in params:
        variants.append((f"{W0_SUFFIX}{nla}", {"w0_prior": W0_GT_M1_PRIOR, "nla": True}, None))

    # Extended-vector flows additionally get the reference-prior (Gower-Street-family) chains: the
    # w0 > -1 + NLA and lambdaCDM + NLA models with near-delta ns/Obh2/H0 Gaussians and baryons fixed
    # at the fiducial, matching the analysis choices of the DES Y3 SBI reference papers.
    ref_kwargs = _ref_prior_kwargs(flow)
    if ref_kwargs is not None:
        variants += [
            (
                f"{W0_SUFFIX}{nla}{REF_PRIOR_SUFFIX}",
                {"w0_prior": W0_GT_M1_PRIOR, "nla": True, **ref_kwargs},
                REF_PRIOR_SUFFIX,
            ),
            (
                f"_lambdaCDM{nla}{REF_PRIOR_SUFFIX}",
                {"lambdaCDM": True, "nla": True, **ref_kwargs},
                REF_PRIOR_SUFFIX,
            ),
        ]
    return variants


def add_obs_args(parser, mock_labels_default=None):
    """Add observation inclusion flags to an argument parser (all default off)."""
    parser.add_argument("--include_grid", action="store_true")
    parser.add_argument("--n_grid_examples", type=int, default=16)
    parser.add_argument("--include_des", action="store_true")
    parser.add_argument("--include_mocks", action="store_true")
    parser.add_argument(
        "--mock_labels",
        nargs="+",
        default=mock_labels_default,
        help="mock labels to sample; if omitted, every mock in the prediction file is used "
        "(see discover_mock_labels)",
    )
    parser.add_argument(
        "--mock_realizations",
        action="store_true",
        help="also sample each individual realization in {label}_stack as its OWN observation "
        "(one chain per realization, not a product over likelihoods); default samples only "
        "the {label}_mean summary (one chain per mock).",
    )
    parser.add_argument(
        "--mock_product",
        nargs="*",
        default=list(MOCK_PRODUCT_DEFAULT),
        metavar="SUBSTRING",
        help="mocks whose label contains one of these substrings also get their whole {label}_stack "
        "sampled as ONE product-likelihood posterior (MacCrann+2018), chain_{label}_stack.npy: the "
        "N-fold-area joint posterior of the Buzzard recovery test, which the {label}_mean chain is NOT. "
        "Default: the Buzzard flock only. Bare --mock_product (no values) switches it off.",
    )
    parser.add_argument(
        "--mock_fixed_ext",
        nargs="*",
        default=list(MOCK_FIXED_EXT_DEFAULT),
        metavar="SUBSTRING",
        help="mocks whose label contains one of these substrings also get every chain sampled with the "
        f"extension parameters fixed at the mock's truth, chain_{{key}}{FIXED_EXT_SUFFIX}.npy (the Buzzard "
        "recovery test). Default: the Buzzard flock only. Bare --mock_fixed_ext switches it off.",
    )


def discover_mock_labels(obs_pred_dict):
    """All mock labels in a preds file: those with BOTH {L}_mean and {L}_stack.

    That mean+stack pair is the structural signature written only by evaluate_obs_mocks /
    evaluate_mock_cls, so the observation sources stay cleanly disjoint by structure (not by
    name): grid (grid_*) and DES (DESy3*) have neither key. The Buzzard flock is a mock like any
    other and is picked up here. Only the {L}_mean summary is sampled (one chain per mock); the
    _stack is the discovery signal, sampled only under --mock_realizations.
    """
    suf = "_stack"
    return sorted(
        k[: -len(suf)] for k in obs_pred_dict if k.endswith(suf) and f"{k[: -len(suf)]}_mean" in obs_pred_dict
    )


def _cosmo_dict(params, cosmo_arr):
    return {str(p): v for p, v in zip(params, cosmo_arr)}


def get_grid_observations(obs_pred_dict, obs_cosmo_dict, params, n_examples=16):
    obs_dict = {}
    for label in sorted(k for k in obs_pred_dict if k.startswith("grid_"))[:n_examples]:
        cosmo = _cosmo_dict(params, obs_cosmo_dict[label]) if label in obs_cosmo_dict else None
        obs_dict[label] = {"pred": obs_pred_dict[label], "cosmo": cosmo}
    return obs_dict


def get_des_observations(obs_pred_dict):
    obs_dict = {}
    for label in sorted(k for k in obs_pred_dict if k == "DESy3" or k.startswith("DESy3_")):
        obs_dict[label] = {"pred": obs_pred_dict[label], "cosmo": None}
    return obs_dict


def _fixed_ext(label, cosmo, fixed_ext_params):
    """``{param: truth}`` for the fixed-extension variant of mock ``label``, or None when it has no truth
    for one of them (a NaN, or no cosmo at all)."""
    if not fixed_ext_params:
        return None
    fixed = {p: float(cosmo[p]) for p in fixed_ext_params if cosmo is not None and p in cosmo}
    if len(fixed) < len(fixed_ext_params) or not np.all(np.isfinite(list(fixed.values()))):
        LOGGER.warning(f"{label}: no truth for all of {fixed_ext_params} ({fixed}); no {FIXED_EXT_SUFFIX} chains")
        return None
    return fixed


def get_mock_observations(
    obs_pred_dict,
    obs_cosmo_dict,
    params,
    obs_labels,
    include_realizations=False,
    product_match=(),
    fixed_ext_match=(),
    fixed_ext_params=(),
):
    """Mock observations keyed by chain name. An entry of a mock matching ``fixed_ext_match`` also
    carries ``fixed_params``, the truth of ``fixed_ext_params``, for its FIXED_EXT_SUFFIX chain."""
    obs_dict = {}
    for label in obs_labels:
        full_label = f"{label}_mean"
        if full_label not in obs_pred_dict:
            print(f"Warning: '{full_label}' not found in predictions, skipping.")
            continue
        cosmo = _cosmo_dict(params, obs_cosmo_dict[label]) if label in obs_cosmo_dict else None
        n_before = len(obs_dict)
        obs_dict[full_label] = {"pred": obs_pred_dict[full_label], "cosmo": cosmo}

        # The whole stack as ONE observation, i.e. the product over its per-realization likelihoods.
        # Keeping all rows under a single key is what makes it a product: both backends sum the log
        # likelihood over the rows of obs["pred"] (emcee in _mcmc_log_posterior, torch_batched via
        # obs_index), so this is the N-fold-area joint posterior rather than a posterior at the mean.
        # For an ensemble each row enters through the ensemble density, prod_r mean_m p_m(x_r|theta).
        if any(s in label for s in product_match):
            stack_label = f"{label}_stack"
            if stack_label not in obs_pred_dict:
                print(f"Warning: '{stack_label}' not found in predictions, skipping the product.")
            else:
                obs_dict[stack_label] = {"pred": obs_pred_dict[stack_label], "cosmo": cosmo}

        # Optionally add each stack realization as its own single-row observation (separate chain,
        # not a product likelihood). Keys are {label}_{i}, which do not end in "_mean" and so are
        # excluded from the mock-contamination plot (which uses only the {label}_mean chains).
        if include_realizations:
            stack_label = f"{label}_stack"
            if stack_label not in obs_pred_dict:
                print(f"Warning: '{stack_label}' not found in predictions, skipping realizations.")
            else:
                for i, row in enumerate(obs_pred_dict[stack_label]):
                    obs_dict[f"{label}_{i}"] = {"pred": row, "cosmo": cosmo}

        fixed = _fixed_ext(label, cosmo, fixed_ext_params) if any(s in label for s in fixed_ext_match) else None
        if fixed is not None:
            for key in list(obs_dict)[n_before:]:
                obs_dict[key]["fixed_params"] = fixed
    return obs_dict


def collect_observations(args, obs_pred_dict, obs_cosmo_dict, params, msfm_conf):
    """Build obs_dict from CLI args and loaded prediction dictionaries."""
    obs_dict = {}
    if args.include_grid:
        obs_dict.update(get_grid_observations(obs_pred_dict, obs_cosmo_dict, params, args.n_grid_examples))
    if args.include_des:
        obs_dict.update(get_des_observations(obs_pred_dict))
    if args.include_mocks:
        obs_dict.update(
            get_mock_observations(
                obs_pred_dict,
                obs_cosmo_dict,
                params,
                args.mock_labels,
                include_realizations=getattr(args, "mock_realizations", False),
                product_match=getattr(args, "mock_product", MOCK_PRODUCT_DEFAULT),
                fixed_ext_match=getattr(args, "mock_fixed_ext", MOCK_FIXED_EXT_DEFAULT),
                fixed_ext_params=[p for p in (getattr(args, "extend_params", None) or []) if p in params],
            )
        )
    return obs_dict


def _can_batch(flow, obs_dict, backend):
    """The GPU-batched sampler covers a single LikelihoodFlow or a LikelihoodFlowEnsemble, including
    multi-row (product-likelihood) observations; any other flow type falls back to the emcee loop."""
    if backend != "torch_batched":
        return False
    if not hasattr(flow, "sample_posterior_batched"):
        print("mcmc_backend=torch_batched unavailable for this flow type; using emcee.")
        return False
    return True


def _stack_rows(obs_dict, keys):
    """All summary rows of ``keys`` in one array, plus the obs_index mapping each row to its key's
    position, so a multi-row observation is sampled as one product-likelihood chain."""
    rows = [np.atleast_2d(obs_dict[k]["pred"]) for k in keys]
    obs_index = np.repeat(np.arange(len(keys)), [len(r) for r in rows])
    return np.concatenate(rows, axis=0), obs_index


def _save_member_chains(flow, keys, member_chains, member_log_probs, variant_suffix=""):
    """Persist each ensemble member's own batched chain, for store_individual_chains and for
    run_member_mcmc. member_chains[i] is that member's (n_obs, n_samples, n_params) array, keyed by
    observation order."""
    if flow.model_dir is None:
        return
    for m, (chains_m, lps_m) in enumerate(zip(member_chains, member_log_probs)):
        for i, key in enumerate(keys):
            np.save(os.path.join(flow.model_dir, f"chain_{key}{variant_suffix}_flow_{m}.npy"), chains_m[i])
            np.save(os.path.join(flow.model_dir, f"log_probs_{key}{variant_suffix}_flow_{m}.npy"), lps_m[i])


def _run_mcmc_batched(
    flow,
    obs_dict,
    n_walkers,
    n_steps,
    n_burnin_steps,
    thin=1,
    use_validation_weights=True,
    method="ensemble",
    store_individual_chains=False,
):
    """Sample every observation's wCDM posterior in a single GPU-batched run, then save each chain in
    the same location/format as the emcee path (mcmc.run_emcee) and reproduce its contour plots.

    method ("ensemble" | "individual") is forwarded to sample_posterior_batched; "individual" pools the
    per-member chains but returns the same (n_obs, n_samples, n_params) layout, so saving/plotting below is
    method-agnostic. With store_individual_chains the per-member chains are additionally saved."""
    keys = list(obs_dict.keys())
    x_batch, obs_index = _stack_rows(obs_dict, keys)  # (n_rows, n_features), n_rows >= n_obs
    want_members = store_individual_chains and method == "individual" and hasattr(flow, "flows")

    print(f"\nGPU-batched sampling of {len(keys)} observations (method={method})")
    result = flow.sample_posterior_batched(
        x_batch,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burnin_steps=n_burnin_steps,
        thin=thin,
        use_validation_weights=use_validation_weights,
        method=method,
        obs_index=obs_index,
        **({"return_members": True} if want_members else {}),
    )
    if want_members:
        chains, log_probs, member_chains, member_log_probs = result
        _save_member_chains(flow, keys, member_chains, member_log_probs)
    else:
        chains, log_probs = result

    for i, key in enumerate(keys):
        obs = obs_dict[key]
        if flow.model_dir is not None:
            np.save(os.path.join(flow.model_dir, f"chain_{key}.npy"), chains[i])
            np.save(os.path.join(flow.model_dir, f"log_probs_{key}.npy"), log_probs[i])

        if obs["cosmo"] is not None and "des" not in key.lower():
            flow.plot_contours(
                chains[i], obs_point=obs["cosmo"], obs_label=key, label=key, with_des_chain=False, density=True
            )

    # DES observations additionally get every restricted-model variant in the des_variants table
    # (lambdaCDM, w0 > -1, +NLA, reference priors). Each variant batches all DES observations
    # together, so this doesn't degenerate into slow one-at-a-time chains.
    des_keys = [k for k in keys if "des" in k.lower()]
    if des_keys:
        x_des, des_index = _stack_rows(obs_dict, des_keys)
        for suffix, model_kwargs, _ in des_variants(flow)[1:]:
            print(f"\nGPU-batched sampling of variant '{suffix}' for {len(des_keys)} DES obs (method={method})")
            result_v = flow.sample_posterior_batched(
                x_des,
                n_walkers=n_walkers,
                n_steps=n_steps,
                n_burnin_steps=n_burnin_steps,
                thin=thin,
                use_validation_weights=use_validation_weights,
                method=method,
                obs_index=des_index,
                **model_kwargs,
                **({"return_members": True} if want_members else {}),
            )
            if want_members:
                chains_v, log_probs_v, member_chains_v, member_log_probs_v = result_v
                _save_member_chains(flow, des_keys, member_chains_v, member_log_probs_v, variant_suffix=suffix)
            else:
                chains_v, log_probs_v = result_v
            for i, key in enumerate(des_keys):
                if flow.model_dir is not None:
                    np.save(os.path.join(flow.model_dir, f"chain_{key}{suffix}.npy"), chains_v[i])
                    np.save(os.path.join(flow.model_dir, f"log_probs_{key}{suffix}.npy"), log_probs_v[i])

    # fixed-extension mock chains, batched per set of fixed values (one per mock truth)
    groups = {}
    for key in keys:
        if obs_dict[key].get("fixed_params"):
            groups.setdefault(tuple(sorted(obs_dict[key]["fixed_params"].items())), []).append(key)
    for fixed, fixed_keys in groups.items():
        x_fix, fix_index = _stack_rows(obs_dict, fixed_keys)
        print(f"\nGPU-batched sampling of {len(fixed_keys)} mock chains with {dict(fixed)} fixed (method={method})")
        chains_f, log_probs_f = flow.sample_posterior_batched(
            x_fix,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burnin_steps=n_burnin_steps,
            thin=thin,
            use_validation_weights=use_validation_weights,
            method=method,
            obs_index=fix_index,
            fixed_params=dict(fixed),
        )
        for i, key in enumerate(fixed_keys):
            if flow.model_dir is not None:
                np.save(os.path.join(flow.model_dir, f"chain_{key}{FIXED_EXT_SUFFIX}.npy"), chains_f[i])
                np.save(os.path.join(flow.model_dir, f"log_probs_{key}{FIXED_EXT_SUFFIX}.npy"), log_probs_f[i])


def run_member_mcmc(
    flow,
    obs_dict,
    n_walkers=1024,
    n_steps=1000,
    n_burnin_steps=1000,
    thin=1,
    obs_labels=("DESy3",),
    backend="torch_batched",
):
    """Sample each ensemble member's OWN posterior for the DES observation(s).

    This is the ensemble-convergence test of the blinding strategy: the members agree by
    construction on data drawn from the training distribution, so a disagreement on the real data is
    evidence that the summary lies outside it (the flows do not extrapolate).

    Writes ``chain_{obs}_flow_{m}.npy`` and ``log_probs_{obs}_flow_{m}.npy`` beside the ensemble's
    own ``chain_{obs}.npy`` and touches nothing else. In particular the pooled chain that
    ``method="individual"`` returns is **discarded rather than saved**, so the production chains keep
    the ensemble-likelihood definition -- which is the whole reason this is a separate stage instead
    of the ``mcmc.method``/``store_individual_chains`` config pair.

    Only the unrestricted wCDM model is sampled, i.e. no model kwargs, which is the baseline entry
    of ``des_variants``: the restricted-prior variants answer a different question, and each one
    would multiply the cost by ``n_flows``.
    """
    if not hasattr(flow, "flows"):
        LOGGER.warning("--sample_flow_members needs a LikelihoodFlowEnsemble (--n_flows>1); skipping.")
        return
    # The emcee store_individual_chains path saves into flow_{m}/ and writes no member log_probs, so
    # it is not an equivalent fallback; the batched path is the one supported layout.
    if backend != "torch_batched" or not hasattr(flow, "sample_posterior_batched"):
        LOGGER.warning("--sample_flow_members requires --mcmc_backend=torch_batched; skipping.")
        return

    keys = [k for k in obs_labels if k in obs_dict]
    for missing in [k for k in obs_labels if k not in obs_dict]:
        LOGGER.warning(f"--sample_flow_members: '{missing}' is not among the sampled observations; skipping it.")
    if not keys:
        LOGGER.warning("--sample_flow_members: no requested observation present (need --include_des); skipping.")
        return

    x_batch = np.concatenate([np.atleast_2d(obs_dict[k]["pred"]) for k in keys], axis=0)
    print(f"\nGPU-batched per-member sampling of {len(keys)} observation(s) over {flow.n_flows} flows")
    _, _, member_chains, member_log_probs = flow.sample_posterior_batched(
        x_batch,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burnin_steps=n_burnin_steps,
        thin=thin,
        use_validation_weights=False,  # a member posterior is its own, unweighted
        method="individual",
        return_members=True,
    )
    _save_member_chains(flow, keys, member_chains, member_log_probs)
    LOGGER.info(f"Saved {flow.n_flows} per-member chains for {keys} in {flow.model_dir}")


def run_mcmc(
    flow,
    obs_dict,
    n_walkers=1024,
    n_steps=1000,
    n_burnin_steps=1000,
    thin=1,
    method="ensemble",
    use_validation_weights=True,
    backend="emcee",
    store_individual_chains=False,
):
    if _can_batch(flow, obs_dict, backend):
        _run_mcmc_batched(
            flow,
            obs_dict,
            n_walkers,
            n_steps,
            n_burnin_steps,
            thin=thin,
            use_validation_weights=use_validation_weights,
            method=method,
            store_individual_chains=store_individual_chains,
        )
        return

    # store_individual_chains is only meaningful for a LikelihoodFlowEnsemble's "individual" method; a
    # single LikelihoodFlow ignores both (its sample_posterior has no such args).
    extra = {} if not hasattr(flow, "flows") else {"store_individual_chains": store_individual_chains}
    for key, obs in obs_dict.items():
        print(f"\nStarting with mock observation {key}")
        posterior_samples = flow.sample_posterior(
            obs["pred"],
            label=key,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burnin_steps=n_burnin_steps,
            thin=thin,
            method=method,
            use_validation_weights=use_validation_weights,
            **extra,
        )
        if obs["cosmo"] is not None and "des" not in key.lower():
            flow.plot_contours(
                posterior_samples,
                obs_point=obs["cosmo"],
                obs_label=key,
                label=key,
                with_des_chain=False,
                density=True,
            )
        if "des" in key.lower():
            # same variant table as the batched path; sample_posterior derives the matching filename
            # suffix from model_kwargs itself, so only variant_label has to be passed through
            for suffix, model_kwargs, variant_label in des_variants(flow)[1:]:
                print(f"\nStarting variant '{suffix}' run for {key}")
                flow.sample_posterior(
                    obs["pred"],
                    label=key,
                    n_walkers=n_walkers,
                    n_steps=n_steps,
                    n_burnin_steps=n_burnin_steps,
                    thin=thin,
                    variant_label=variant_label,
                    method=method,
                    use_validation_weights=use_validation_weights,
                    **model_kwargs,
                    **extra,
                )
        if obs.get("fixed_params"):
            print(f"\nStarting fixed-extension run for {key} with {obs['fixed_params']} fixed")
            flow.sample_posterior(
                obs["pred"],
                label=key,
                n_walkers=n_walkers,
                n_steps=n_steps,
                n_burnin_steps=n_burnin_steps,
                thin=thin,
                fixed_params=obs["fixed_params"],
                variant_label=FIXED_EXT_SUFFIX,
                method=method,
                use_validation_weights=use_validation_weights,
                **extra,
            )

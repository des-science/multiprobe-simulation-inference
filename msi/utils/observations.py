import os

import numpy as np

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
# Street analysis family: Jeffrey+24 2403.02314, Gatti+24 2405.10881, Williamson+26), and fixes
# baryonification at the fiducial (the references do not marginalize baryons). Run automatically for
# every DES observation when ns/Ob/H0 are among the flow's params, as w0 > -1 + NLA and lambdaCDM +
# NLA chains -- the closest apples-to-apples analogues to the references' wCDM and LCDM results.
REF_GAUSSIAN_PRIORS = {
    "ns": (0.9649, 0.0063),
    "Obh2": (0.02237, 0.00015),  # derived Ob * (H0/100)^2
    "H0": (70.22, 2.45),
}
REF_PRIOR_SUFFIX = "_refpriors"


def _ref_prior_kwargs(flow):
    """Sampler kwargs for the reference-prior variant, or None when the flow is not conditioned on the
    extended parameter vector. Baryon fiducials are read from the run's own msfm config."""
    if not all(p in flow.params for p in ("ns", "Ob", "H0")):
        return None
    fiducial = flow.conf["analysis"]["fiducial"]
    fixed = {p: fiducial[p] for p in ("bary_Mc", "bary_nu") if p in flow.params}
    return {"gaussian_priors": REF_GAUSSIAN_PRIORS, "fixed_params": fixed}


def add_obs_args(parser, mock_labels_default=None):
    """Add observation inclusion flags to an argument parser (all default off)."""
    parser.add_argument("--include_grid", action="store_true")
    parser.add_argument("--n_grid_examples", type=int, default=16)
    parser.add_argument("--include_des", action="store_true")
    parser.add_argument("--include_buzzard", action="store_true")
    parser.add_argument("--buzzard_labels", nargs="+", default=["Buzzard_mean"])
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


def discover_mock_labels(obs_pred_dict):
    """All mock labels in a preds file: those with BOTH {L}_mean and {L}_stack.

    That mean+stack pair is the structural signature written only by evaluate_obs_benchmark /
    evaluate_mock_cls, so the three observation sources stay cleanly disjoint by structure (not by
    name): grid (grid_*) and DES (DESy3*) have neither key; Buzzard writes only Buzzard_mean (no
    _stack) and is excluded too. Only the {L}_mean summary is sampled (one chain per mock); the
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


def get_buzzard_observations(obs_pred_dict, obs_cosmo_dict, params, labels):
    obs_dict = {}
    for label in labels:
        if label not in obs_pred_dict:
            print(f"Warning: '{label}' not found in predictions, skipping.")
            continue
        cosmo = _cosmo_dict(params, obs_cosmo_dict[label]) if label in obs_cosmo_dict else None
        obs_dict[label] = {"pred": obs_pred_dict[label], "cosmo": cosmo}
    return obs_dict


def get_mock_observations(obs_pred_dict, obs_cosmo_dict, params, obs_labels, include_realizations=False):
    obs_dict = {}
    for label in obs_labels:
        full_label = f"{label}_mean"
        if full_label not in obs_pred_dict:
            print(f"Warning: '{full_label}' not found in predictions, skipping.")
            continue
        cosmo = _cosmo_dict(params, obs_cosmo_dict[label]) if label in obs_cosmo_dict else None
        obs_dict[full_label] = {"pred": obs_pred_dict[full_label], "cosmo": cosmo}

        # Optionally add each stack realization as its own single-row observation (separate chain,
        # not a product likelihood). Keys are {label}_{i}, which do not end in "_mean" and so are
        # excluded from the mock-contamination plot (which uses only the {label}_mean chains).
        if include_realizations:
            stack_label = f"{label}_stack"
            if stack_label not in obs_pred_dict:
                print(f"Warning: '{stack_label}' not found in predictions, skipping realizations.")
                continue
            for i, row in enumerate(obs_pred_dict[stack_label]):
                obs_dict[f"{label}_{i}"] = {"pred": row, "cosmo": cosmo}
    return obs_dict


def collect_observations(args, obs_pred_dict, obs_cosmo_dict, params, msfm_conf):
    """Build obs_dict from CLI args and loaded prediction dictionaries."""
    obs_dict = {}
    if args.include_grid:
        obs_dict.update(get_grid_observations(obs_pred_dict, obs_cosmo_dict, params, args.n_grid_examples))
    if args.include_des:
        obs_dict.update(get_des_observations(obs_pred_dict))
    if args.include_buzzard:
        obs_dict.update(get_buzzard_observations(obs_pred_dict, obs_cosmo_dict, params, args.buzzard_labels))
    if args.include_mocks:
        obs_dict.update(
            get_mock_observations(
                obs_pred_dict,
                obs_cosmo_dict,
                params,
                args.mock_labels,
                include_realizations=getattr(args, "mock_realizations", False),
            )
        )
    return obs_dict


def _can_batch(flow, obs_dict, backend):
    """The GPU-batched sampler covers a single LikelihoodFlow or a LikelihoodFlowEnsemble, with one summary
    vector per observation (it treats the leading axis as independent observations). Anything else -- a flow
    type without sample_posterior_batched, or an observation that bundles several summary rows into one
    product-likelihood posterior -- transparently falls back to the emcee loop."""
    if backend != "torch_batched":
        return False
    if not hasattr(flow, "sample_posterior_batched"):
        print("mcmc_backend=torch_batched unavailable for this flow type; using emcee.")
        return False
    if any(np.atleast_2d(obs["pred"]).shape[0] != 1 for obs in obs_dict.values()):
        print("mcmc_backend=torch_batched: some observations bundle multiple summary rows; using emcee.")
        return False
    return True


def _save_member_chains(flow, keys, member_chains, member_log_probs, variant_suffix=""):
    """Persist each ensemble member's own batched chain alongside the pooled one (store_individual_chains).
    member_chains[i] is that member's (n_obs, n_samples, n_params) array, keyed by observation order."""
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
    x_batch = np.concatenate([np.atleast_2d(obs_dict[k]["pred"]) for k in keys], axis=0)  # (n_obs, n_features)
    want_members = store_individual_chains and method == "individual" and hasattr(flow, "flows")

    print(f"\nGPU-batched sampling of {len(keys)} observations (method={method})")
    result = flow.sample_posterior_batched(
        x_batch,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burnin_steps=n_burnin_steps,
        use_validation_weights=use_validation_weights,
        method=method,
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

    # DES observations additionally get a lambdaCDM (w0 = -1) posterior, like the emcee path; batch them
    # together so this doesn't degenerate into slow one-at-a-time chains
    des_keys = [k for k in keys if "des" in k.lower()]
    if des_keys:
        print(f"\nGPU-batched LambdaCDM sampling of {len(des_keys)} DES observation(s) (method={method})")
        x_des = np.concatenate([np.atleast_2d(obs_dict[k]["pred"]) for k in des_keys], axis=0)
        result_l = flow.sample_posterior_batched(
            x_des,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burnin_steps=n_burnin_steps,
            lambdaCDM=True,
            use_validation_weights=use_validation_weights,
            method=method,
            **({"return_members": True} if want_members else {}),
        )
        if want_members:
            chains_l, log_probs_l, member_chains_l, member_log_probs_l = result_l
            _save_member_chains(flow, des_keys, member_chains_l, member_log_probs_l, variant_suffix="_lambdaCDM")
        else:
            chains_l, log_probs_l = result_l
        for i, key in enumerate(des_keys):
            if flow.model_dir is not None:
                np.save(os.path.join(flow.model_dir, f"chain_{key}_lambdaCDM.npy"), chains_l[i])
                np.save(os.path.join(flow.model_dir, f"log_probs_{key}_lambdaCDM.npy"), log_probs_l[i])

        # DES observations additionally get a restricted-w0 (w0 > -1) posterior, with w0 still a free
        # sampled parameter (full-dimensional chain), batched together like the lambdaCDM block above
        print(
            f"\nGPU-batched restricted-w0 (w0 > -1) sampling of {len(des_keys)} DES observation(s) (method={method})"
        )
        result_w = flow.sample_posterior_batched(
            x_des,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burnin_steps=n_burnin_steps,
            w0_prior=W0_GT_M1_PRIOR,
            use_validation_weights=use_validation_weights,
            method=method,
            **({"return_members": True} if want_members else {}),
        )
        if want_members:
            chains_w, log_probs_w, member_chains_w, member_log_probs_w = result_w
            _save_member_chains(flow, des_keys, member_chains_w, member_log_probs_w, variant_suffix=W0_SUFFIX)
        else:
            chains_w, log_probs_w = result_w
        for i, key in enumerate(des_keys):
            if flow.model_dir is not None:
                np.save(os.path.join(flow.model_dir, f"chain_{key}{W0_SUFFIX}.npy"), chains_w[i])
                np.save(os.path.join(flow.model_dir, f"log_probs_{key}{W0_SUFFIX}.npy"), log_probs_w[i])

        # DES observations of an IA probe additionally get the combined w0 > -1 AND NLA (bta = 0)
        # posterior: the closest analogue to a standard DES-Y3-like model. bta is dropped from the
        # sampled space, so this chain is in the reduced (bta-dropped) parameter ordering.
        if "bta" in getattr(flow, "params", []):
            suffix_wn = f"{W0_SUFFIX}{NLA_SUFFIX}"
            print(
                f"\nGPU-batched w0 > -1 + NLA (bta = 0) sampling of {len(des_keys)} "
                f"DES observation(s) (method={method})"
            )
            result_wn = flow.sample_posterior_batched(
                x_des,
                n_walkers=n_walkers,
                n_steps=n_steps,
                n_burnin_steps=n_burnin_steps,
                w0_prior=W0_GT_M1_PRIOR,
                nla=True,
                use_validation_weights=use_validation_weights,
                method=method,
                **({"return_members": True} if want_members else {}),
            )
            if want_members:
                chains_wn, log_probs_wn, member_chains_wn, member_log_probs_wn = result_wn
                _save_member_chains(flow, des_keys, member_chains_wn, member_log_probs_wn, variant_suffix=suffix_wn)
            else:
                chains_wn, log_probs_wn = result_wn
            for i, key in enumerate(des_keys):
                if flow.model_dir is not None:
                    np.save(os.path.join(flow.model_dir, f"chain_{key}{suffix_wn}.npy"), chains_wn[i])
                    np.save(os.path.join(flow.model_dir, f"log_probs_{key}{suffix_wn}.npy"), log_probs_wn[i])

        # Extended-vector flows additionally get the reference-prior (Gower-Street-family) chains: the
        # w0 > -1 + NLA and lambdaCDM + NLA models with near-delta ns/Obh2/H0 Gaussians and baryons fixed
        # at the fiducial, matching the analysis choices of the DES Y3 SBI reference papers.
        ref_kwargs = _ref_prior_kwargs(flow)
        if ref_kwargs is not None:
            for model_kwargs, suffix_ref in (
                ({"w0_prior": W0_GT_M1_PRIOR, "nla": True}, f"{W0_SUFFIX}{NLA_SUFFIX}{REF_PRIOR_SUFFIX}"),
                ({"lambdaCDM": True, "nla": True}, f"_lambdaCDM{NLA_SUFFIX}{REF_PRIOR_SUFFIX}"),
            ):
                print(
                    f"\nGPU-batched reference-prior sampling ({suffix_ref}) of {len(des_keys)} "
                    f"DES observation(s) (method={method})"
                )
                result_r = flow.sample_posterior_batched(
                    x_des,
                    n_walkers=n_walkers,
                    n_steps=n_steps,
                    n_burnin_steps=n_burnin_steps,
                    use_validation_weights=use_validation_weights,
                    method=method,
                    **model_kwargs,
                    **ref_kwargs,
                    **({"return_members": True} if want_members else {}),
                )
                if want_members:
                    chains_r, log_probs_r, member_chains_r, member_log_probs_r = result_r
                    _save_member_chains(flow, des_keys, member_chains_r, member_log_probs_r, variant_suffix=suffix_ref)
                else:
                    chains_r, log_probs_r = result_r
                for i, key in enumerate(des_keys):
                    if flow.model_dir is not None:
                        np.save(os.path.join(flow.model_dir, f"chain_{key}{suffix_ref}.npy"), chains_r[i])
                        np.save(os.path.join(flow.model_dir, f"log_probs_{key}{suffix_ref}.npy"), log_probs_r[i])


def run_mcmc(
    flow,
    obs_dict,
    n_walkers=1024,
    n_steps=1000,
    n_burnin_steps=1000,
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
            print(f"\nStarting LambdaCDM run for {key}")
            flow.sample_posterior(
                obs["pred"],
                label=key,
                n_walkers=n_walkers,
                n_steps=n_steps,
                n_burnin_steps=n_burnin_steps,
                lambdaCDM=True,
                method=method,
                use_validation_weights=use_validation_weights,
                **extra,
            )
            print(f"\nStarting restricted-w0 (w0 > -1) run for {key}")
            flow.sample_posterior(
                obs["pred"],
                label=key,
                n_walkers=n_walkers,
                n_steps=n_steps,
                n_burnin_steps=n_burnin_steps,
                w0_prior=W0_GT_M1_PRIOR,
                method=method,
                use_validation_weights=use_validation_weights,
                **extra,
            )
            if "bta" in getattr(flow, "params", []):
                print(f"\nStarting w0 > -1 + NLA (bta = 0) run for {key}")
                flow.sample_posterior(
                    obs["pred"],
                    label=key,
                    n_walkers=n_walkers,
                    n_steps=n_steps,
                    n_burnin_steps=n_burnin_steps,
                    w0_prior=W0_GT_M1_PRIOR,
                    nla=True,
                    method=method,
                    use_validation_weights=use_validation_weights,
                    **extra,
                )
            ref_kwargs = _ref_prior_kwargs(flow)
            if ref_kwargs is not None:
                for model_kwargs in ({"w0_prior": W0_GT_M1_PRIOR, "nla": True}, {"lambdaCDM": True, "nla": True}):
                    print(f"\nStarting reference-prior run ({model_kwargs}) for {key}")
                    flow.sample_posterior(
                        obs["pred"],
                        label=key,
                        n_walkers=n_walkers,
                        n_steps=n_steps,
                        n_burnin_steps=n_burnin_steps,
                        variant_label=REF_PRIOR_SUFFIX,
                        method=method,
                        use_validation_weights=use_validation_weights,
                        **model_kwargs,
                        **ref_kwargs,
                        **extra,
                    )

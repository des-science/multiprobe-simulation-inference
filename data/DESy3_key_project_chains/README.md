# DES Y3 key project chains

Public DES Y3 real-space chains, renamed to `<cosmo_model>/<ia_model>/<probes>[_variant][+external].txt`
so they can be indexed like `msi.utils.chains` does (`[cosmo_model][ia_model][probes]`). The files are
byte-identical to the release (md5 below). Downloaded 2026-10-08 from
`https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/`.

Only this README is tracked in git; download the chain files from the release paths below.

This directory supersedes `../DESy3_chains/`, which is kept unchanged. In that older tree,
`LambdaCDM/tatt/` holds copies of the **wCDM** chains, and the ΛCDM TATT chains are missing.

## Releases

- `tatt/` and `external/` come from the key paper (DES Collaboration 2022, arXiv:2105.13549),
  [Y3key-products](https://des.ncsa.illinois.edu/releases/y3a2/Y3key-products). The fiducial IA model is
  TATT (columns `a1 a2 alpha1 alpha2 bias_ta`). maglim lenses, shear ratios (`smallratio`) included.
- `nla/` comes from the extensions paper (DES Collaboration 2023, arXiv:2207.05766),
  [Y3key-extensions](https://des.ncsa.illinois.edu/releases/y3a2/Y3key-extensions). NLA IA (`a1 alpha1`),
  which is the IA model of our forward model. The release has no NLA chain for 1x2pt or 2x2pt.

## Naming

| Token | Meaning |
|---|---|
| `1x2pt` | cosmic shear ξ± + shear ratios |
| `2x2pt` | galaxy clustering w(θ) + galaxy-galaxy lensing γt + shear ratios |
| `3x2pt` | all three |
| `_opt` | ΛCDM-optimized scale cuts (the cosmic-shear papers' fiducial, arXiv:2105.13543/13544) |
| `_fixednu` | neutrino mass fixed, Ωνh² = 0.00083 (3 degenerate species, Σmν ≈ 0.077 eV). ΛCDM only |
| `_NLbias` | nonlinear galaxy bias. Bias columns are `b1E_sig8sq_bin*`, not `b1..b4` |
| `+BAO+RSD+SN` | + BOSS DR12, eBOSS DR16 (LRG, ELG, QSO, Lyα), MGS, 6dFGS, Pantheon |
| `+Planck+BAO+RSD+SN` | the same + Planck 2018 |
| `external/Planck.txt` | Planck 2018 TT,TE,EE+lowE alone, without CMB lensing ("Planck CMB" of arXiv:2105.13549 Sec. VII). DES priors, m_ν varied |

The column names are in each file's first line. The last column is the sample weight. ΛCDM chains have no
`w` column. `SIGMA_8` is a derived column, so S8 is computed from it.

## Files

| File | Release path | md5 |
|---|---|---|
| `LambdaCDM/tatt/1x2pt.txt` | `chains/chain_1x2pt_lcdm_SR_maglim.txt` | `bbcb3a7037e42c94b3273f55bbbd1dfa` |
| `LambdaCDM/tatt/1x2pt_opt.txt` | `chains/chain_1x2pt_lcdm_SR_maglim_opt.txt` | `b1d14eb30f4253392393bea37c1673d5` |
| `LambdaCDM/tatt/2x2pt.txt` | `chains/chain_2x2pt_lcdm_SR_maglim.txt` | `29fe26de45acfec620deda5cf90f3bb0` |
| `LambdaCDM/tatt/2x2pt_NLbias.txt` | `chains/chain_2x2pt_lcdm_SR_maglim_NLbias.txt` | `3ab126db5fa776ab1a3d416442ae166f` |
| `LambdaCDM/tatt/3x2pt.txt` | `chains/chain_3x2pt_lcdm_SR_maglim.txt` | `b85ce9de8fae3e08f94e2f5e2b020de5` |
| `LambdaCDM/tatt/3x2pt_NLbias.txt` | `chains/chain_3x2pt_lcdm_SR_maglim_NLbias.txt` | `1114b28ea8a03b3408af5783bad30598` |
| `LambdaCDM/tatt/3x2pt_fixednu.txt` | `chains/chain_3x2pt_fixednu_lcdm.txt` | `82b00441e0d0b1f7c56a2d5104d88e90` |
| `LambdaCDM/tatt/3x2pt_opt.txt` | `chains/chain_3x2pt_lcdm_SR_maglim_opt.txt` | `5c5995d68c9adc18e061cecff1e3ebb1` |
| `LambdaCDM/nla/3x2pt.txt` | `y3a2_beyond_lcdm/chains/…ini.d3_l_nla_realy3dat.txt` | `f2f3301b757d7b3ec9a7a41ed0b49fba` |
| `LambdaCDM/nla/3x2pt+BAO+RSD+SN.txt` | `y3a2_beyond_lcdm/chains/…ini.d3-brs_l_nla_realy3dat.txt` | `7a62e9ec483151a43ba9909fc4cee4ac` |
| `LambdaCDM/nla/3x2pt+Planck+BAO+RSD+SN.txt` | `y3a2_beyond_lcdm/chains/…ini.d3-pbrs_l_nla_realy3dat.txt` | `7f4464fe51eed4b4b8fe5d6ffa8467fd` |
| `LambdaCDM/external/Planck.txt` | `chains/chain_p-TTTEEE-lowE_lcdm.txt` | `6fd279ce37a55757b724c58367edfb8a` |
| `wCDM/tatt/1x2pt.txt` | `chains/chain_1x2pt_wcdm_SR_maglim.txt` | `f63bb0fbc8361ffe9274e41df6c90cad` |
| `wCDM/tatt/2x2pt.txt` | `chains/chain_2x2pt_wcdm_SR_maglim.txt` | `54cd9a03c7fd6f6bb4d0feb809c578c9` |
| `wCDM/tatt/2x2pt_NLbias.txt` | `chains/chain_2x2pt_wcdm_SR_maglim_NLbias.txt` | `7effcfc939e7052c12008db42698aa53` |
| `wCDM/tatt/3x2pt.txt` | `chains/chain_3x2pt_wcdm_SR_maglim.txt` | `b34909ff32e53b25703aef9c2a3b12a0` |
| `wCDM/tatt/3x2pt_NLbias.txt` | `chains/chain_3x2pt_wcdm_SR_maglim_NLbias.txt` | `1e4f995ee3418725bcfb385082b60b17` |
| `wCDM/nla/3x2pt.txt` | `y3a2_beyond_lcdm/chains/…ini.d3_w_nla_realy3dat.txt` | `eca88b54ba83949077f21a7f141e1675` |
| `wCDM/nla/3x2pt+BAO+RSD+SN.txt` | `y3a2_beyond_lcdm/chains/…ini.d3-brs_w_nla_realy3dat.txt` | `a46552bf099ebe918394221d46051a8b` |
| `wCDM/nla/3x2pt+Planck+BAO+RSD+SN.txt` | `y3a2_beyond_lcdm/chains/…ini.d3-pbrs_w_nla_realy3dat.txt` | `462ba96d2dd36c191d53a673b0173e54` |
| `wCDM/external/Planck.txt` | `chains/chain_p-TTTEEE-lowE_wcdm.txt` | `d408eacecef3bafd647ba9e727edd8c7` |

`…` stands for `chain_2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits.scales-ml_3x2pt_8_6_0.5_v0.40`.
Verify with `md5sum` against this table.

`3x2pt_fixednu` is the only fixed-neutrino chain in the release. Its file name has no lens tag, but its
header points at the maglim data vector (`2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits`).
Added 2026-10-09.

## Not included

The release also has redMaGiC-lens chains, other DES+external combinations
(`all`, `des_all`, `lowz`, `highzh0`, `planck_des`), the beyond-wCDM extensions (w0wa, Neff, Ωk, Σ/μ,
binned growth), and the CMB-lensing 5x2pt/6x2pt chains. None of them is needed for this paper.

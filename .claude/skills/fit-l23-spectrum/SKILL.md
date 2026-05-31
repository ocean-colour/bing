---
name: fit-l23-spectrum
description: Load a Loisel et al. 2023 (L23) synthetic spectrum, fit it with BING, and compare retrieved IOPs and Chl/Sdg/Y to truth. Use when the user asks to validate against L23, test on synthetic data, or benchmark a model on the Loisel dataset.
---

# Fit a Loisel et al. 2023 synthetic spectrum

## When to use this skill

- "Validate this model on L23"
- "Fit Loisel 2023 spectrum 170"
- "Compare retrieved a_ph to truth"
- Benchmarking new absorption/backscattering models

## What L23 gives you

For each of 3320 indices, the dataset (`ocpy.hydrolight.loisel23.load_ds(4, 0)`) provides simulated Rrs **and** the true IOPs that generated it:

| Quantity | Symbol | Notes |
|---|---|---|
| Wavelengths | `wave` | 400–700 nm (Hydrolight grid) |
| Remote sensing reflectance | `Rrs` | sr⁻¹ |
| Total absorption | `a` | m⁻¹ |
| Phytoplankton absorption | `aph` | m⁻¹ |
| CDOM absorption | `ag` | m⁻¹ |
| Detrital absorption | `ad` | m⁻¹ |
| a_dg (dissolved + detrital) | `adg = ag + ad` | derived |
| Total backscattering | `bb` | m⁻¹ |
| Non-water backscattering | `bbnw` | m⁻¹ |
| Chlorophyll | `Chl` | from `aph(440)/0.05582` |
| Slope of a_dg | `Sdg` | fit by `functions.fit_Sdg` |
| Lee Y | `Y` | Lee+2002 band-ratio formula |

`load_one_l23(idx)` returns all of these in a dict.

## Single-spectrum fit (one call)

The high-level wrapper does everything:

```python
from bing.parameters import standard
from bing.fitting import l23

p = standard.expb_pow(
    satellite='PACE',
    nsteps=40000,
    nburn=2000,
    scl_noise=0.02,      # 2% relative noise, or 'PACE'/'MODIS_Aqua'/'SeaWiFS'/'SBG'
    add_noise=True,
    variable_Gordon=False,
    include_Raman=False,
)

chains, models, prep, idx, extras = l23.fit_one(p, idx=170)
```

`prep['odict']` holds the L23 truth. `extras` has `wave, obs_Rrs, varRrs, Chl, Y` for plotting/saving.

## Compare retrieval to truth

```python
import numpy as np
from bing import evaluate

a_med, bb_med, a_lo, a_hi, bb_lo, bb_hi, Rrs_pred, sigRrs = \
    evaluate.reconstruct_from_chains(models, chains,
                                     rt_dict=l23.rt_defs.rt_dict_from_p(p),
                                     perc=(5, 95))

# Truth on the model wavelength grid (interpolate)
from bing.preproc import convert_to_satwave
true_a   = convert_to_satwave(prep['odict']['true_wave'],
                              prep['odict']['a'],
                              models[0].wave)
true_bb  = convert_to_satwave(prep['odict']['true_wave'],
                              prep['odict']['bb'],
                              models[1].wave)

# Bias and dispersion in log space (typical for IOPs)
def report(name, fit, lo, hi, truth):
    bias = np.log10(fit) - np.log10(truth)
    print(f"{name}  log10 bias: mean={bias.mean():+.3f}  std={bias.std():.3f}")
    in_band = (truth >= lo) & (truth <= hi)
    print(f"{name}  truth in 90% CI: {in_band.mean():.1%}")

report("a (440-700)", a_med, a_lo, a_hi, true_a)
report("bb (440-700)", bb_med, bb_lo, bb_hi, true_bb)
```

A healthy ExpBricaud+Pow fit at PACE noise level should yield log10 bias < 0.05 and ~90% coverage of truth by the 90% credible interval.

## Decompose retrieved a_dg vs a_ph (truth comparison)

```python
from bing.evaluate import thin_burn_chains

flat = thin_burn_chains(chains)  # (nsamples, nparam)
a_dg, a_ph = models[0].eval_anw(
    flat[..., :models[0].nparam], retsub_comps=True)

i440 = np.argmin(np.abs(models[0].wave - 440))
print(f"a_dg(440)  fit = {np.median(a_dg[:, i440]):.4f}  "
      f"truth = {prep['odict']['adg'][np.argmin(np.abs(prep['odict']['true_wave']-440))]:.4f}")
print(f"a_ph(440)  fit = {np.median(a_ph[:, i440]):.4f}  "
      f"truth = {prep['odict']['aph'][np.argmin(np.abs(prep['odict']['true_wave']-440))]:.4f}")
```

## Recover Chl, Sdg, Y from chains

```python
# Chl (only if the model fitted it — ExpBricaud has Aph at index 2)
if 'Aph' in models[0].pnames:
    iAph = models[0].pnames.index('Aph')
    Chl_fit = 10**np.median(flat[:, iAph]) / 0.05582
    print(f"Chl  fit = {Chl_fit:.3f}  truth = {prep['odict']['Chl']:.3f}")

# Sdg
if 'Sdg' in models[0].pnames:
    iSdg = models[0].pnames.index('Sdg')
    Sdg_fit = np.median(flat[:, iSdg])
    print(f"Sdg  fit = {Sdg_fit:.4f}  truth = {prep['odict']['Sdg']:.4f}")

# beta / Y (backscattering slope is in models[1])
if 'beta' in models[1].pnames:
    ibeta = models[1].pnames.index('beta')
    j = models[0].nparam + ibeta
    print(f"beta  fit = {np.median(flat[:, j]):.3f}  L23 Y = {prep['odict']['Y']:.3f}")
```

## Batch over the whole L23 dataset

```python
l23.batch_fit(p, n_batch=5, n_cores=15,
              out_dir='papers/bing_2.0/Analysis/Fits/')
```

This writes one NPZ per index using `chain_filename(p, idx=idx, path=out_dir)`. Filenames encode model/satellite/noise/UV-cutoff conventions — see `chain_filename` in [bing/fitting/l23.py](../../../bing/fitting/l23.py).

After the run, aggregate with:

```python
l23.process_all(p, outfile='papers/bing_2.0/Analysis/results_summary.csv',
                n_cores=15)
```

## Recommended figures

Truth-vs-fit panels are most informative when stratified by water type:

1. Scatter `bbp(440)_fit` vs `bbp(440)_truth` colored by `Chl_truth` (or by index)
2. log–log scatter `a_ph(440)_fit` vs `a_ph(440)_truth`
3. Residual spectra `(a_fit - a_truth) / a_truth` averaged over Chl bins
4. Coverage diagnostic: fraction of wavelengths where `a_lo ≤ a_truth ≤ a_hi`, target = `perc[1] - perc[0]` (e.g., 90%)

[plot-bing-fit](../plot-bing-fit/SKILL.md) handles panel (3) and (4) via `plotting.show_fits`.

## Common pitfalls

- **`p.satellite='L23'` keeps the native Hydrolight grid**; any other value (`'PACE'`, `'MODIS'`, etc.) interpolates Rrs to that grid and noise is sampled accordingly.
- **`include_Raman=True` requires the L23 `f_a` / `f_bb` interpolators**, which `prep_one_l23` builds for you — but if you call lower-level functions yourself, build them from `ds.Lambda` and `ds.a/ds.bb`.
- **Chl/Y in the truth dict** are derived, not stored directly in L23; `load_one_l23` recomputes them every time.
- **`process_one` references `anly_utils_20` and `bbw_440`** — those imports live in the analysis dir, not in BING. Use `process_one` only inside `papers/bing_2.0/Analysis/` or supply your own loader.
- **Output filename convention** is encoded by `chain_filename`; don't roll your own or `process_all` won't find them.

## Related skills

- [run-bing-fit](../run-bing-fit/SKILL.md) — lower-level fitting if you don't want the L23 wrapper
- [batch-fit-argo](../batch-fit-argo/SKILL.md) — parallel-fitting pattern that mirrors `l23.batch_fit`
- [plot-bing-fit](../plot-bing-fit/SKILL.md) — figures
- [diagnose-mcmc](../diagnose-mcmc/SKILL.md) — convergence checks

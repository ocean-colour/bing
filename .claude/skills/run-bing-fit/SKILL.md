---
name: run-bing-fit
description: Run the canonical end-to-end MCMC fit for a single Rrs spectrum in BING — configure models, set priors, do a least-squares initial guess, run emcee, and reconstruct IOPs with uncertainties. Use whenever the user asks to fit a spectrum, run an MCMC on Rrs, or invoke the standard BING pipeline.
---

# End-to-end BING fit for one Rrs spectrum

## When to use this skill

- "Fit this Rrs spectrum"
- "Run an MCMC on this data"
- "Get a, bb from this Rrs"
- Building a new analysis script that mirrors the bing_fit_Rrs CLI

## Required inputs

- `wave`: wavelengths (np.ndarray, shape `(nwave,)`)
- `Rrs`: observed remote sensing reflectance (np.ndarray, `(nwave,)` in sr⁻¹)
- `varRrs`: per-wavelength variance (np.ndarray, `(nwave,)` in sr⁻²). If unavailable, use `(0.02 * Rrs)**2` or call `bing.noise.scale_noise`.
- Optional: `Chl` (chlorophyll, mg m⁻³) and `Y` (backscattering slope) for models that need them.

## Canonical script

```python
import numpy as np
from bing.parameters import standard
from bing.models import utils as model_utils
from bing.priors import priors as bing_priors
from bing.fitting import chisq_fit, inference as bing_inf
from bing.rt import defs as rt_defs
from bing import evaluate

# ---------- 1. Configuration ----------
p = standard.expb_pow(
    satellite='PACE',
    nsteps=40000,
    nburn=2000,
    variable_Gordon=False,
    include_Raman=False,
)
rt_dict = rt_defs.rt_dict_from_p(p)

# ---------- 2. Initialize models on the data wavelength grid ----------
models = model_utils.init(p.model_names, wave)
bing_priors.set_standard_priors(models, p)

# ---------- 3. Set Chl / Y if model needs them ----------
# Required if ExpBricaud (uses_Chl=True) or Lee (uses_basis_params=True).
model_utils.init_other_bits(models, Chl=Chl, Y=Y, Rrs=Rrs)

# ---------- 4. Initial guess via least-squares ----------
# Bounds come from the prior dicts (only 'uniform'/'log_uniform' supported here).
low, high = [], []
for prior in models[0].priors.priors + models[1].priors.priors:
    low.append(prior.pmin)
    high.append(prior.pmax)
bounds = (np.array(low), np.array(high))

# p0 in linear space, then log10 the params whose prior is log_*
p0_a = models[0].init_guess(np.maximum(Rrs * 5.0, 1e-4))  # crude fallback
p0_b = models[1].init_guess(np.maximum(Rrs * 0.1, 1e-4))
p0 = np.concatenate([np.atleast_1d(p0_a), np.atleast_1d(p0_b)])
ii = 0
for ss in [0, 1]:
    for prior in models[ss].priors.priors:
        if prior.flavor.startswith('log'):
            p0[ii] = np.log10(max(p0[ii], 1e-6))
        ii += 1

items = (Rrs, varRrs, p0, 0)
p0_best, cov, _ = chisq_fit.fit(items, models, rt_dict, bounds=bounds)

# ---------- 5. MCMC refinement ----------
pdict = bing_inf.init_mcmc(models, nsteps=p.nsteps, nburn=p.nburn)
pdict['Chl'] = np.array([Chl if Chl is not None else 0.0])
pdict['Y']   = np.array([Y   if Y   is not None else 0.0])

chains, _ = bing_inf.fit_one(
    (Rrs, varRrs, p0_best, 0),
    models=models, pdict=pdict, chains_only=True, rt_dict=rt_dict,
)
# chains has shape (nsteps, nwalkers, nparam)

# ---------- 6. Analysis ----------
stats = evaluate.calc_stats(chains, names=models[0].pnames + models[1].pnames)
a, bb, a_lo, a_hi, bb_lo, bb_hi, Rrs_pred, sigRrs = \
    evaluate.reconstruct_from_chains(models, chains, rt_dict, perc=(5, 95))

# Print headline IOPs
for name, med in zip(stats['names'], stats['med']):
    p_lo = stats[f"p{int(np.round(5)):02d}"] if False else None  # keys are 'p05','p95'
print("a(440):  {:.4f}  [{:.4f}, {:.4f}]".format(
    a[np.argmin(np.abs(wave - 440))],
    a_lo[np.argmin(np.abs(wave - 440))],
    a_hi[np.argmin(np.abs(wave - 440))]))
```

## Picking a standard combination

Pre-configured combos are in [bing/parameters/standard.py](../../../bing/parameters/standard.py):

| Combo | Models | Use when |
|---|---|---|
| `expb_pow()` | ExpBricaud + Pow | Default for PACE / mixed waters |
| `expbf_pow()` | ExpBricaudFree + Pow | Chl is uncertain, fit it as a parameter |
| `giop()` | GIOP + Lee | Reproduce GIOP-like behavior (Werdell 2013) |
| `gsm()` | GSM + GSM | Garver-Siegel-Maritorena baseline |
| `k2b()` | Bricaud + Cst | Phyto-dominated waters with flat bb |

All accept `**kwargs` that flow into the `p_ntuple.gen` config; common keys: `satellite`, `nsteps`, `nburn`, `scl_noise`, `wv_min`, `wv_max`, `variable_Gordon`, `include_Raman`, `add_noise`, `beta`, `set_Sdg`, `Sdg`, `sSdg`.

## Sanity checks

Before trusting any fit:

```python
# 1. Initial least-squares should already be close
pred0 = chisq_fit.fit_func(wave, *p0_best, models=models, rt_dict=rt_dict)
print("LM χ²ν:", np.sum((pred0 - Rrs)**2 / varRrs) / (wave.size - p0_best.size))
# 2. MCMC should accept ~15–50% of proposals; if much lower, priors are too wide
# 3. log_prob at p0_best must be finite
from bing.fitting.inference import log_prob
print("log_prob(p0_best):", log_prob(p0_best, models, Rrs, varRrs, rt_dict))
```

If `log_prob` is `-inf`, see [debug-priors](../debug-priors/SKILL.md).

## Output for downstream analysis

Save with `bing.fitting.l23.save_chains` (also works for non-L23 spectra):

```python
from bing.fitting.l23 import save_chains
save_chains(
    chains, 0, outfile='my_fit.npz',
    extras=dict(wave=wave, obs_Rrs=Rrs, varRrs=varRrs,
                Chl=Chl or 0., Y=Y or 0.),
)
```

NPZ contains: `chains`, `idx`, plus everything in `extras`.

## CLI alternative

The same workflow runs from the shell:

```bash
bing_fit_Rrs spectrum.csv ExpBricaud,Pow --outroot fit --satellite PACE --fit_method mcmc
```

CSV format: `wave,Rrs,sigRrs[,anw,bbnw]`. See [bing/scripts/fit_Rrs.py](../../../bing/scripts/fit_Rrs.py).

## Common pitfalls

- **`init_other_bits` skipped** → ExpBricaud throws because `self.aph_star` is undefined; Lee throws because `self._shape` is undefined.
- **`pdict['Chl']` / `pdict['Y']` is `None`** → `fit_one` indexes into them; pass at least a zero array of the right length.
- **`rt_dict` missing** → all forward-model calls now require it (added during the Raman/fluorescence refactor); pass `rt_defs.rt_dict_from_p(p)`.
- **Wave grid mismatch** → `models` and `Rrs` must share `wave`. If you have hyperspectral data and a satellite grid, use [satellite-band-prep](../satellite-band-prep/SKILL.md) first.
- **Tight priors near initial guess** → `log_prob = -inf` at p0; widen priors before MCMC.

## Related skills

- [diagnose-mcmc](../diagnose-mcmc/SKILL.md) — check chain convergence
- [debug-priors](../debug-priors/SKILL.md) — fix stuck chains
- [plot-bing-fit](../plot-bing-fit/SKILL.md) — visualize the result
- [satellite-band-prep](../satellite-band-prep/SKILL.md) — prep hyperspectral input
- [inelastic-rrs](../inelastic-rrs/SKILL.md) — add Raman/fluorescence to the forward model

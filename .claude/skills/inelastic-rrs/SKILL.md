---
name: inelastic-rrs
description: Add Raman scattering and/or chlorophyll fluorescence to the BING Rrs forward model. Use when the user asks to include Raman, add fluorescence, or switch from the elastic-only Gordon model. Covers when to use which calc_Rrs_with_* variant, parameter additions, and the rt_dict flags that turn each correction on.
---

# Inelastic radiative transfer in BING

## When to use this skill

- "Include Raman scattering in the fit"
- "Add chlorophyll fluorescence to the forward model"
- "Use `calc_Rrs_with_raman` / `calc_Rrs_with_fluorescence`"
- Investigating residuals around 685 nm (fluorescence) or in the green/red (Raman)

## Three forward models in `bing.rt.rrs`

| Function | Includes | When to use |
|---|---|---|
| `calc_Rrs(a, bb, in_G1=None, in_G2=None, a_ex=None, bb_ex=None, bb_R=None)` | Elastic Gordon ± Raman if `a_ex/bb_ex/bb_R` supplied | Default and most flexible |
| `calc_elastic_Rrs(a, bb, in_G1=None, in_G2=None)` | Pure elastic (Gordon 1988) | Quick checks, no inelastic processes |
| `calc_Rrs_with_fluorescence(...)` / `calc_Rrs_with_raman(...)` | Convenience wrappers | One-shot calls for plotting/diagnostics |

The MCMC pipeline uses the first form. Whether Raman is included is controlled by `rt_dict['include_Raman']`.

## Turning Raman on for a fit

1. Set `include_Raman=True` in your `standard.*` parameters:

```python
from bing.parameters import standard
p = standard.expb_pow(
    satellite='PACE',
    nsteps=40000,
    include_Raman=True,    # ← here
    variable_Gordon=True,  # often paired with Raman for hyperspectral fits
)
```

2. `l23.prep_one_l23` (or your own prep) builds `a_ex`, `bb_ex` and reads `bb_R` from `models[1].bb_R`. These are the IOPs **at the Raman excitation wavelengths** `model.wave_ex`, computed by the base model `__init__`.

3. `rt_dict = rt_defs.rt_dict_from_p(p)` produces `{'variable_Gordon': True, 'include_Raman': True}`. `inference.log_prob` and `evaluate.calc_Rrs_from_models` read this dict.

4. No extra free parameters — Raman uses the absorption/backscattering posterior you're already fitting, evaluated at `wave_ex`.

## Adding chlorophyll fluorescence

Fluorescence is **not yet wired into `log_prob`** — `calc_Rrs_from_models` does not call the fluorescence machinery. You'd add it either:

- **Outside the fit**: compute and subtract it from `Rrs_observed` before fitting, treating it as a known additive term.
- **Inside `evaluate.calc_Rrs_from_models`**: extend that function (and the prior list) with a fluorescence-quantum-yield parameter.

The building blocks live in [bing/rt/chl_fl.py](../../../bing/rt/chl_fl.py):

```python
from bing.rt import chl_fl
from bing.rt.rrs import calc_Rrs_with_fluorescence, calc_fluorescence_spectrum

# Standalone fluorescence Rrs contribution from Chl and IOPs
Rrs_fl = calc_Rrs_with_fluorescence(
    a, bb, Chl,           # m^-1, m^-1, mg m^-3
    wave,                  # nm
    quantum_yield=0.005,
)

# Or shape only:
shape = calc_fluorescence_spectrum(wave, peak_nm=685., fwhm_nm=25.)
```

`calc_fluorescence_line_height`, `calc_R_fluorescence`, and `fluorescence_backscattering_coeff` give you the low-level pieces.

## Variable Gordon coefficients

Either correction is usually paired with wavelength-dependent G₁(λ), G₂(λ):

```python
from bing.rt.rrs import wave_dependent_gordon
G1, G2 = wave_dependent_gordon(wave)
# Must be attached to both models:
models[0].G1 = G1; models[0].G2 = G2
models[1].G1 = G1; models[1].G2 = G2
```

`rt_dict['variable_Gordon']=True` then makes `calc_Rrs_from_models` use them. If you forget to set `G1/G2` on the models, you get `ValueError: Need to set model G1, G2 for variable Gordon`.

## Manual Rrs build (no MCMC)

```python
from bing.rt import rrs, raman

# Elastic only
Rrs_e = rrs.calc_elastic_Rrs(a, bb)

# Elastic + Raman, hand-built
a_ex   = a_model.eval_a_ex(a_params)
bb_ex  = bb_model.eval_bb_ex(bb_params)
bb_R   = bb_model.bb_R
Rrs_eR = rrs.calc_Rrs(a, bb, in_G1=None, in_G2=None,
                      a_ex=a_ex, bb_ex=bb_ex, bb_R=bb_R)

# Convenience version
Rrs_eR2 = rrs.calc_Rrs_with_raman(a, bb, a_ex, bb_ex, bb_R)
```

## Sanity comparison

Plot elastic-only vs elastic+Raman residuals to see the wavelength regions affected:

```python
import matplotlib.pyplot as plt
plt.plot(wave, Rrs_eR - Rrs_e)
plt.axhline(0, color='k', lw=0.5)
plt.xlabel("wavelength [nm]"); plt.ylabel("ΔRrs from Raman [sr⁻¹]")
```

Typical Raman contribution: positive, ~1–3% of Rrs in the 500–600 nm window for clear water; smaller for chl-rich water.

## Adding fluorescence as a free parameter (sketch)

If you want to fit a quantum yield, extend the absorption model:

```python
class aNWExpBricaudFlu(aNWExpBricaud):
    name = 'ExpBricaudFlu'
    nparam = 4                        # add logQ_fl
    pnames = ['Adg', 'Sdg', 'Aph', 'logQ_fl']

    def eval_anw(self, params, **kwargs):
        # discard the 4th parameter for absorption (still log10 stored)
        return super().eval_anw(params[..., :3], **kwargs)
```

Then extend `evaluate.calc_Rrs_from_models` to read `Q_fl = 10**params[..., -1]` and call `calc_Rrs_with_fluorescence(... quantum_yield=Q_fl)`. **This is real surgery — don't do it casually**; the change touches both least-squares and MCMC paths.

## Common pitfalls

- **`include_Raman=True` but `wave_ex` not initialized** → most likely you instantiated a model with a non-standard wavelength grid that puts `wave_ex` outside any reference range. Check `models[0].wave_ex` exists and is finite.
- **Variable Gordon without setting G1/G2 on both models** → `ValueError`. The absorption model owns G1/G2 conceptually, but the bbnw model must also have them set so its `eval_a_ex` / `eval_bb_ex` calls work.
- **Mixing elastic Gordon coefficients (constants) with `variable_Gordon=True`** → you'll silently double-count if you also set `in_G1=A_Rrs` somewhere. Either let `models[0].G1` flow through, or pass `in_G1=in_G2=None`.
- **Fluorescence added by hand to observed Rrs** → fine, but the Bayesian posterior then ignores the Q_fl uncertainty. Document this if you do it.

## Related skills

- [run-bing-fit](../run-bing-fit/SKILL.md) — flip `include_Raman` / `variable_Gordon` in the standard pipeline
- [satellite-band-prep](../satellite-band-prep/SKILL.md) — match excitation/emission grids
- [plot-bing-fit](../plot-bing-fit/SKILL.md) — diagnose where Raman/fluorescence matter in residuals

## Reference files

- [bing/rt/rrs.py](../../../bing/rt/rrs.py)
- [bing/rt/raman.py](../../../bing/rt/raman.py)
- [bing/rt/chl_fl.py](../../../bing/rt/chl_fl.py)
- [bing/evaluate.py](../../../bing/evaluate.py) (`calc_Rrs_from_models`, `reconstruct_from_chains`)

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

This page describes BING's own (`rt_backend='gordon'`) inelastic wiring —
`bing.rt.raman` / `bing.rt.chl_fl`, composed via `calc_Rrs`. This is BING's
**legacy** inelastic path, kept for the `gordon` backend and backward
compatibility. It is not the recommended path for new work — see below.

## Recommended inelastic path: `robust.rt.inelastic`

The project has adopted `robust.rt.inelastic` (from the sibling `robust`
package) as the recommended inelastic (Raman + fluorescence) path going
forward, reachable from BING by setting `rt_dict['rt_backend'] =
'robust_ztt'` or `'robust_hybrid'` (not `'robust_baseline'`, which is
elastic-only and raises `ValueError` if combined with
`include_Raman`/`include_Chl_fl` — see
[run-bing-fit](../run-bing-fit/SKILL.md#rt-backend-selection)). It is wired
end-to-end and exercised by the test suite (M0-M4 of the `rob_rt`
integration). **New work should prefer `robust_ztt`/`robust_hybrid` over
`bing.rt.raman`/`bing.rt.chl_fl`** (per project decision — see
`claude_prompts/RT/rob_rt_prompt_6.md` Q1/Q5); the legacy `gordon`-backend
modules documented above are being kept only for backward compatibility and
should not be the starting point for new inelastic work.

**Be clear-eyed about the measured accuracy gap — this recommendation does
not erase it.** Measured on real L23 data (`nb/RT/rob_rt_coding_5.ipynb`
§§3-4, a real production `Ed` on both sides), the robust-vs-BING agreement
on the inelastic terms is:

- **Raman**: max 11.4% / mean 5.6% (worst at 400 nm)
- **Fluorescence**: max 9.2% / mean 5.8% (mismatched-`Ed` case, as M4
  ships) to max 18.5% / mean 7.2% (`Ed`-matched case — larger, not
  smaller)

Both are several **orders of magnitude** outside this project's working
tolerance for cross-backend agreement (`rtol <= 5e-4`), and the gap is not
numerical noise — float32-vs-float64 alone costs ~2.3e-7 relative, four
orders of magnitude smaller than the measured disagreement. The root cause
is a genuine physics-composition difference: `robust`'s Raman/fluorescence
kernels interpolate/clamp the emission-grid IOP spectrum to stand in for
the excitation-wavelength IOPs, rather than re-evaluating the parametric
`a`/`bb` models at the true excitation grid the way BING's own legacy
Gordon+Raman/fluorescence path does. For contrast, the *elastic* path
agrees tightly (`robust_baseline` vs `gordon`, ~2.3e-7 relative) — the
disagreement above is specific to the inelastic terms.

What this means in practice:

- Use `rt_backend='robust_ztt'`/`'robust_hybrid'` for new inelastic work —
  this is the project's adopted path, and it keeps the forward model
  internally consistent with `robust`'s elastic terms (see
  `docs/design/rob_rt_design.md` §3.1).
- If your work specifically depends on the most physically faithful
  Raman/fluorescence terms BING can currently produce — e.g. you are
  diagnosing or reproducing the numbers above — `rt_backend='gordon'`
  remains available and is the measured-accurate choice for that narrow
  purpose. It is not otherwise the recommended path.
- Do not assume `robust`'s inelastic terms match BING's legacy physics to
  within the project's usual `5e-4` tolerance — they don't, by 1-2 orders
  of magnitude, for the reason above. That gap is expected and understood,
  not a bug.

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

## Adding CDOM fluorescence (robust backends only)

The third inelastic term, `rt_dict['include_CDOM_fl']` — robust's analytic Hawes et al. (1992) FA7 kernel (`robust.rt.cdom_fl`). There is **no Gordon-path equivalent**: `include_CDOM_fl=True` with `rt_backend='gordon'` (or `'robust_baseline'`) raises at fit setup.

```python
rt_dict = dict(rt_defs.rt_dict_from_p(p),
               rt_backend='robust_ztt',
               include_CDOM_fl=True,
               cdom_fraction=0.8)   # a_cdom = 0.8 * a_dg  (see below)
```

- **`a_cdom = cdom_fraction * a_dg` is a fixed-fraction proxy, not a retrieval.** The kernel's source term is pure CDOM absorption; BING's `a_dg` is CDOM *plus* detritus, and no a-model splits them. The 0.8 default is a project decision (JXP, 2026-09-05; `claude_prompts/rt_tests.md` Q32) recorded in `bing.rt.defs.CDOM_FRACTION_DEFAULT`. The Rrs increment is essentially linear in the fraction.
- Requires an a-model with a separable `a_dg` (`has_a_dg`: ExpBricaud family, GIOP, GSM, ExpNMF); anything else raises, naming the class.
- The kernel amplitude `robust.rt.CDOMFl.scale` is held **fixed at 1.0** (`bing.evaluate.CDOM_FL_SCALE`) — never fitted.
- Emission is broad and featureless across the blue–green (~465–570 nm), not a 685 nm line.
- robust flags its own CDOM-fl term as analytic-only and **not yet validated** against HydroLight truth (its M5).

See [radiative_transfer.rst](../../../docs/radiative_transfer.rst) (`rt-cdom-fluorescence`) for the full write-up.

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

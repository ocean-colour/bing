---
name: add-bbnw-model
description: Scaffold a new non-water backscattering (bb_nw) model class in bing/models/bbnw.py, register it in init_model, attach a default prior dict, and add a unit test. Use when the user asks to add a new bb_nw parameterization or extend the backscattering model family.
---

# Add a new non-water backscattering model

## When to use this skill

- User says "add a bb_nw model", "implement <Name> for backscattering", "new particle backscattering law"
- A new size-distribution model or wavelength-dependent slope formulation needs adding

## API contract

The base class is `bbNWModel` in [bing/models/bbnw.py](../../../bing/models/bbnw.py). All subclasses MUST:

1. Set class attributes: `name` (str), `nparam` (int), `pnames` (list of str), `log_params` (list of bool), and `uses_basis_params` (bool).
2. Implement `_eval_bbnw(params, wave)` returning shape `(nsample, nwave)`, and evaluate on the `wave` you are given.
3. Amplitude parameters are in log10 space. Exponents stay linear.
4. Use `self.wave`, `self.bb_w` (set by base `__init__`). Don't recompute water backscattering.
5. If the spectral shape depends on an externally-supplied basis (e.g., Lee's Y from band ratios), set `uses_basis_params = True` and implement `set_basis_func(Y)` plus `eval_basis_func(wave)`.
6. Optionally implement `init_guess(bbnw_observed)` returning a 1D guess in linear space.

The pivot reference wavelength convention in BING is **600 nm** for the Power-law model. GSM uses 443 nm, and `Pow2`'s mineral term uses 700 nm on a separate `pivot_min` attribute. Keep `pivot` a scalar — analysis scripts read `models[1].pivot` and assume "the wavelength at which `params[0]` is the amplitude".

## Three contracts that bite

**1. Implement `_eval_bbnw`, not `eval_bbnw`.**
The public `eval_bbnw(params, wave=None)` lives on the base class: it resolves `wave=None` to `self.wave` and delegates to your `_eval_bbnw(params, wave)`. So `wave` is never None in your method — and you must use it rather than `self.wave`, because `eval_bb_ex` passes the Raman *excitation* grid. The base `_eval_bbnw` raises `NotImplementedError`, so forgetting it fails loudly rather than silently.

**2. `init_guess` returns LINEAR amplitudes; the caller log10s by prior flavor.**
Both `bing.fitting.l23` and `ioptics.run` convert exactly the slots whose prior `flavor` starts with `log`. So `pnames` order must match the prior-dict order, each amplitude needs a `log_*` prior and each exponent a linear one. Both failure directions are silent: an exponent given a `log_uniform` prior starts at `log10(0)`, and an amplitude given a `uniform` prior is never converted yet `eval_bbnw` still applies `10**`.

Also: **never seed a parameter at exactly 0**. The MCMC walker ball is scaled per parameter, and a 0 seed used to yield zero inter-walker spread — a dimension that never moves for the whole run. There is now an absolute floor in `init_walkers`, but a nonzero seed is still the right habit.

**3. Chi-squared bounds come from the prior dicts' `pmin`/`pmax`.**
Use `uniform` or `log_uniform`. A `gaussian` prior has no `pmin`, and IOPtics turns the resulting `None` into a silent NaN bound. Don't use a `RatioPrior` across bb parameters either: an extra prior beyond `nparam` makes IOPtics' log-mask longer than p0 and raises `IndexError`.

Nothing else validates prior count, so the base class does: `check_priors()` raises if `len(priors) != nparam`, on construction and from `set_standard_priors`.

## Scaffold

Add to [bing/models/bbnw.py](../../../bing/models/bbnw.py):

```python
class bbNWYourName(bbNWModel):
    """One-line description: bb_nw(λ) = <equation>.

    References
    ----------
    - Author et al. (YEAR). Title. Journal X, pages.
    """
    name = 'YourName'                 # must match the eval_bbnw branch
    nparam = 2
    pnames = ['Bnw', 'beta']          # log10 amplitude, linear exponent
    log_params = [True, False]        # read by bing.plotting
    uses_basis_params = False
    pivot = 600.0                     # reference wavelength [nm]

    # prior_dicts MUST default to None (the bb base class makes it
    # positional, unlike the a-models) so tests can construct directly
    def __init__(self, wave, prior_dicts=None):
        bbNWModel.__init__(self, wave, prior_dicts)

    def _eval_bbnw(self, params, wave):
        """bb_nw on the GIVEN grid; shape (nsample, nwave).

        Use `wave`, not self.wave -- eval_bb_ex passes the Raman
        excitation wavelengths.
        """
        return functions.powerlaw(wave, params, pivot=self.pivot)

    def init_guess(self, bbnw_observed):
        """Linear-space starting guess; caller log10s the log slots."""
        i600 = np.argmin(np.abs(self.wave - self.pivot))
        return np.array([max(bbnw_observed[i600], 1e-4), 1.0])
```

`functions.powerlaw`/`constant`/`gen_basis` already return
`(nsample, nwave)` for both 1-D and 2-D (chain-shaped) parameters, and
they work on a *slice* of a wider parameter array — which is how the
two-component models sum two terms:

```python
    def _eval_bbnw(self, params, wave):      # bbNWPow2
        return (functions.powerlaw(wave, params[..., 0:2],
                                   pivot=self.pivot_min) +
                functions.powerlaw(wave, params[..., 2:4],
                                   pivot=self.pivot))
```

## Register in the factory

The `init_model` entry is now the *only* place the base class needs to
know about your model (evaluation is polymorphic). Add to `model_dict` in
[bing/models/bbnw.py](../../../bing/models/bbnw.py):

```python
model_dict = {
    ...,
    'YourName': bbNWYourName,
}
```

## Lee-style models that need a basis function

If the exponent comes from an external estimator (e.g., band ratio of Rrs):

```python
class bbNWYourLee(bbNWModel):
    name = 'YourLee'
    nparam = 1
    pnames = ['Bnw']
    log_params = [True]
    uses_basis_params = True
    pivot = 600.

    def __init__(self, wave, prior_dicts=None):
        bbNWModel.__init__(self, wave, prior_dicts)
        self.Y = None

    def set_basis_func(self, Y):
        """Called by model_utils.init_other_bits before fitting."""
        self.Y = Y
        self.basis_func = self.eval_basis_func()   # cache on self.wave

    def eval_basis_func(self, wave=None):
        """The shape on ANY grid -- this is what makes Raman correct."""
        if self.Y is None:
            raise ValueError('set_basis_func(Y) first')
        wave = self.wave if wave is None else wave
        return (self.pivot / wave)**self.Y
```

and the evaluation, which for a single-basis model is already provided by
the base class as `_eval_basis_bbnw` (this is what `Lee` and `GSM` do):

```python
    def _eval_bbnw(self, params, wave):
        return self._eval_basis_bbnw(params, wave)
```

⚠ Do **not** evaluate the cached `self.basis_func` there. It is
built on `self.wave`, so `eval_bb_ex` would then add pure-water
backscattering on the *excitation* grid to particle backscattering on the
*emission* grid — silently, because the two grids have equal length.

`model_utils.init_other_bits(models, Y=..., Rrs=...)` calls `set_basis_func(Y)` automatically when `uses_basis_params == True`. If your absorption model is `GIOP`, the dispatcher computes `Y` from Rrs via `ocpy.iop.zlee.Y_from_Rrs`.

## Add a default combo to standard.py

Not optional in practice: `set_standard_priors` falls back to
`log_uniform(-6, 5)` for **every** bb parameter when `bpriors` is absent,
which is wrong for linear exponents (they would then be log10'd on the p0
path and allowed to roam over [-6, 5]).

```python
def expb_yourname(**kwargs):
    apriors = [dict(flavor='log_uniform', pmin=-6, pmax=5)] * 3
    apriors[1] = dict(flavor='uniform', pmin=0.01, pmax=0.02)
    bpriors = [dict(flavor='log_uniform', pmin=-6, pmax=5)] * 2
    bpriors[1] = dict(flavor='uniform', pmin=0., pmax=2.)
    params = dict(model_names=['ExpBricaud', 'YourName'],
                  apriors=apriors, bpriors=bpriors,
                  sSdg=0.002, set_Sdg=False)
    params.update(kwargs)
    return p_ntuple.gen(**params)
```

## Add a test

`bing/tests/test_bbnw.py` already exists — extend it, and copy the
patterns already there (parameterisation over models, the `bb_bounds` and
`log10_by_flavor` helpers that mirror the fitters' own logic):

```python
import numpy as np
import pytest
from bing.models import bbnw as bing_bbnw

wave = np.arange(400, 701, 5.)

def test_yourname_init_and_eval():
    model = bing_bbnw.init_model('YourName', wave)
    assert model.nparam == 2

    # 1D
    bb_nw = model.eval_bbnw(np.array([-2.5, 1.0]))
    assert bb_nw.shape == (1, wave.size)
    assert np.all(bb_nw > 0)

    # 2D (chains)
    chains = np.array([[-2.5, 1.0], [-2.0, 1.2]])
    bb_nw = model.eval_bbnw(chains)
    assert bb_nw.shape == (2, wave.size)

def test_yourname_with_priors():
    pdicts = [{'flavor': 'log_uniform', 'pmin': -6, 'pmax': 5},
              {'flavor': 'uniform',     'pmin': 0., 'pmax': 3.}]
    model = bing_bbnw.init_model('YourName', wave, pdicts)
    assert model.priors.priors[1].flavor == 'uniform'
```

Run:
```bash
pytest bing/tests/test_bbnw.py -v
```

## Common pitfalls

Ordered by how much time they cost when they happen.

- **Ignoring the `wave` argument** in `_eval_bbnw` → Raman fits silently
  mix the emission and excitation grids. Never use `self.wave` there.
- **Overriding `eval_bbnw` instead of `_eval_bbnw`** → you lose the
  base class's grid resolution (and any future shared checks). Override
  the private one.
- **Prior flavor disagreeing with `log_params`** → p0 starts in the wrong
  space, silently. An exponent with a `log_uniform` prior starts at
  `log10(0)`; an amplitude with a `uniform` prior is never converted
  though `eval_bbnw` still applies `10**`.
- **Seeding a parameter at exactly 0 in `init_guess`** → historically
  froze that MCMC dimension for the whole run. Floored now, but still
  seed slightly off zero.
- **`uses_basis_params=True` but no `set_basis_func`** → `init_other_bits`
  raises `AttributeError` at MCMC start.
- **A `gaussian` (or `ratio`) prior on a bb parameter** → no `pmin`, so
  the chi-squared bounds break (NaN in IOPtics, `KeyError` in BING).
- **`bpriors` of the wrong length** → `check_priors()` now raises; before
  that guard it silently log10'd the wrong p0 slots.
- **Calling `self.priors.gen_bounds()`** (the `anw.py` `init_guess`
  idiom) → raises for any flavor but `uniform`. Build bounds from the
  prior dicts instead.
- **`super().__init__(wave)`** → `TypeError`; the base signature is
  `(wave, prior_dicts)`.
- **Reinitialized `bb_w`** → don't; the base class loads it from L23.

## Verification checklist

- [ ] `_eval_bbnw` is defined on your class (the base one raises)
- [ ] `init_model('YourName', wave)` succeeds, and with no `prior_dicts`
- [ ] Shape contract holds for 1D and 2D params
- [ ] `eval_bbnw(p, wave=model.wave_ex)` differs from `eval_bbnw(p)`, and
      `eval_bb_ex(p) == bb_w_ex + eval_bbnw(p, wave=wave_ex)`
- [ ] `log_params` matches the factory's prior flavors, element by element
- [ ] `init_guess` returns linear amplitudes, none of them exactly 0, and
      the log10'd result lands inside the prior bounds
- [ ] A wrong-length `bpriors` raises (`check_priors`)
- [ ] If `uses_basis_params=True`: `model_utils.init_other_bits(models, Y=1.0)` succeeds without error
- [ ] `pytest bing/tests/test_bbnw.py` passes
- [ ] Round-trip through `calc_Rrs` produces finite values
- [ ] If the model adds parameters: check convergence with the scipy
      default `maxfev` as well as a raised one

## Related skills

- [add-anw-model](../add-anw-model/SKILL.md) — companion for absorption
- [run-bing-fit](../run-bing-fit/SKILL.md) — end-to-end fit
- [debug-priors](../debug-priors/SKILL.md)

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

1. Set class attributes: `name` (str), `nparam` (int), `pnames` (list of str), and `uses_basis_params` (bool).
2. Implement `eval_bbnw(params)` returning shape `(nsample, nwave)`. Use `np.atleast_2d`.
3. Amplitude parameters are in log10 space. Exponents stay linear.
4. Use `self.wave`, `self.bb_w` (set by base `__init__`). Don't recompute water backscattering.
5. If the spectral shape depends on an externally-supplied basis (e.g., Lee's Y from band ratios), set `uses_basis_params = True` and implement `set_basis_func(Y)`.
6. Optionally implement `init_guess(bbnw_observed)` returning a 1D guess in linear space.

The pivot reference wavelength convention in BING is **600 nm** for the Power-law model. GSM uses 443 nm. Pick whatever the literature for your model uses, but document it on `self.pivot`.

## Scaffold

Add to [bing/models/bbnw.py](../../../bing/models/bbnw.py):

```python
class bbNWYourName(bbNWModel):
    """One-line description: bb_nw(λ) = <equation>.

    References
    ----------
    - Author et al. (YEAR). Title. Journal X, pages.
    """
    name = 'YourName'
    nparam = 2
    pnames = ['logBnw', 'beta']  # log10 amplitude, exponent
    uses_basis_params = False
    pivot = 600.0  # reference wavelength [nm]

    def __init__(self, wave, prior_dicts=None):
        super().__init__(wave)
        if prior_dicts is not None:
            self.priors = bing_priors.Priors(prior_dicts)

    def eval_bbnw(self, params, **kwargs):
        params = np.atleast_2d(params)
        Bnw  = 10**params[:, 0:1]
        beta = params[:, 1:2]
        wave = self.wave[np.newaxis, :]
        bb_nw = Bnw * (self.pivot / wave)**beta
        return bb_nw  # (nsample, nwave)

    def init_guess(self, bbnw_observed):
        i600 = np.argmin(np.abs(self.wave - self.pivot))
        return np.array([max(bbnw_observed[i600], 1e-4), 1.0])
```

## Register in the factory

Add to `model_dict` in `init_model` in [bing/models/bbnw.py](../../../bing/models/bbnw.py):

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
    pnames = ['logBnw']
    uses_basis_params = True

    def __init__(self, wave, prior_dicts=None):
        super().__init__(wave)
        if prior_dicts is not None:
            self.priors = bing_priors.Priors(prior_dicts)

    def set_basis_func(self, Y):
        """Called by model_utils.init_other_bits before MCMC."""
        self.Y = Y
        self._shape = (600.0 / self.wave)**Y  # cache (nwave,)

    def eval_bbnw(self, params, **kwargs):
        params = np.atleast_2d(params)
        Bnw = 10**params[:, 0:1]
        return Bnw * self._shape[np.newaxis, :]
```

`model_utils.init_other_bits(models, Y=..., Rrs=...)` calls `set_basis_func(Y)` automatically when `uses_basis_params == True`. If your absorption model is `GIOP`, the dispatcher computes `Y` from Rrs via `ocpy.iop.zlee.Y_from_Rrs`.

## Add a default combo to standard.py (optional)

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

Create or extend `bing/tests/test_bbnw.py` (mirror of `test_anw.py`):

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

- **`uses_basis_params=True` but no `set_basis_func`** → `init_other_bits` will raise `AttributeError` at MCMC start.
- **Forgot to clamp negative exponents** → set the second prior to `uniform(0, 2)`, not `log_uniform`, so the slope can't go negative if that's unphysical for your particle assumption.
- **Wrong pivot** → small bias that masquerades as parameter drift in chains; document `self.pivot` on the class.
- **`np.atleast_2d` missing** → MCMC walker initialization passes a 1D vector and the call crashes.
- **Reinitialized `bb_w`** → don't; the base class loads it via `ocpy.water.scattering`.

## Verification checklist

- [ ] `init_model('YourName', wave)` succeeds
- [ ] Shape contract holds for 1D and 2D params
- [ ] If `uses_basis_params=True`: `model_utils.init_other_bits(models, Y=1.0)` succeeds without error
- [ ] `pytest bing/tests/test_bbnw.py` passes
- [ ] Round-trip through `calc_Rrs` produces finite values

## Related skills

- [add-anw-model](../add-anw-model/SKILL.md) — companion for absorption
- [run-bing-fit](../run-bing-fit/SKILL.md) — end-to-end fit
- [debug-priors](../debug-priors/SKILL.md)

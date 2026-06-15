---
name: add-anw-model
description: Scaffold a new non-water absorption (a_nw) model class in bing/models/anw.py, register it in init_model, attach a default prior dict, and add a unit test. Use when the user asks to add a new absorption model, implement a new a_nw parameterization, or extend the absorption model family.
---

# Add a new non-water absorption model

## When to use this skill

- User says "add a new a_nw model", "implement <Name> for absorption", "extend bing/models/anw.py with …"
- A model from the literature needs to be added (e.g., Bricaud variant, Chase variant, hybrid CDOM+phyto)

## API contract (read carefully before writing code)

The base class is `aNWModel` in [bing/models/anw.py](../../../bing/models/anw.py). All subclasses MUST:

1. Set class attributes: `name` (str), `nparam` (int), `pnames` (list of str), and `uses_Chl` (bool, default `False`).
2. Implement `eval_anw(params)` returning shape `(nsample, nwave)`. Use `np.atleast_2d(params)` first.
3. Respect the log10 convention: amplitude parameters arrive in log10 space; convert with `10**`. Slopes/exponents stay linear.
4. Use `self.wave`, `self.a_w` (already set by base `__init__`). Don't recompute water absorption.
5. If chlorophyll-dependent, override or call `set_aph(Chl)` in `__init__` and set `uses_Chl = True`.
6. Optionally implement `init_guess(anw_observed)` returning a first guess in linear (not log10) space — `prep_one_l23` will log10-convert any prior whose flavor starts with `log`.

## Scaffold

Add this to [bing/models/anw.py](../../../bing/models/anw.py), placing the class alphabetically among existing aNW classes:

```python
class aNWYourName(aNWModel):
    """One-line description: a_nw(λ) = <equation>.

    References
    ----------
    - Author et al. (YEAR). Title. Journal X, pages.
    """
    name = 'YourName'
    nparam = 2
    pnames = ['logA', 'S']  # log10 amplitude, slope in nm^-1
    uses_Chl = False

    def __init__(self, wave, prior_dicts=None):
        super().__init__(wave)
        if prior_dicts is not None:
            self.priors = bing_priors.Priors(prior_dicts)

    def eval_anw(self, params, **kwargs):
        params = np.atleast_2d(params)
        A = 10**params[:, 0:1]          # (nsample, 1)
        S = params[:, 1:2]              # (nsample, 1) linear
        # Broadcast over wavelengths (1, nwave)
        wave = self.wave[np.newaxis, :]
        a_nw = A * np.exp(-S * (wave - 440.0))
        return a_nw  # (nsample, nwave)

    def init_guess(self, anw_observed):
        """Optional: return a 1D initial guess in LINEAR space."""
        i440 = np.argmin(np.abs(self.wave - 440.0))
        return np.array([max(anw_observed[i440], 1e-4), 0.015])
```

## Register in the factory

Add the name to the `model_dict` inside `init_model` in [bing/models/anw.py](../../../bing/models/anw.py):

```python
model_dict = {
    ...,
    'YourName': aNWYourName,
}
```

## Add a default prior in standard.py (only if it gets its own combo)

If this model deserves a pre-configured combination, add a function to [bing/parameters/standard.py](../../../bing/parameters/standard.py):

```python
def yourname_pow(**kwargs):
    apriors = [dict(flavor='log_uniform', pmin=-6, pmax=5)] * 2
    apriors[1] = dict(flavor='uniform', pmin=0.005, pmax=0.025)  # slope
    bpriors = [dict(flavor='log_uniform', pmin=-6, pmax=5)] * 2
    bpriors[1] = dict(flavor='uniform', pmin=0., pmax=2.)
    params = dict(model_names=['YourName', 'Pow'],
                  apriors=apriors, bpriors=bpriors)
    params.update(kwargs)
    return p_ntuple.gen(**params)
```

## Add a test

Add to [bing/tests/test_anw.py](../../../bing/tests/test_anw.py):

```python
def test_yourname_init_and_eval():
    wave = np.arange(400, 701, 5.)
    model = bing_anw.init_model('YourName', wave)
    assert model.nparam == 2
    assert model.pnames == ['logA', 'S']

    # Single spectrum (1D params)
    a_nw = model.eval_anw(np.array([-1.0, 0.015]))
    assert a_nw.shape == (1, wave.size)
    assert np.all(a_nw > 0)

    # Batched (chains-shaped)
    chains = np.array([[-1.0, 0.015], [-1.2, 0.018]])
    a_nw = model.eval_anw(chains)
    assert a_nw.shape == (2, wave.size)

def test_yourname_with_priors():
    wave = np.arange(400, 701, 5.)
    pdicts = [{'flavor': 'log_uniform', 'pmin': -6, 'pmax': 5},
              {'flavor': 'uniform',     'pmin': 0.005, 'pmax': 0.025}]
    model = bing_anw.init_model('YourName', wave, pdicts)
    assert model.priors.priors[0].flavor == 'log_uniform'
    assert model.priors.priors[1].flavor == 'uniform'
```

Run:
```bash
pytest bing/tests/test_anw.py::test_yourname_init_and_eval -v
```

## Chlorophyll-dependent variants

If your model contains a `a_ph(Chl)` term (Bricaud-like):

```python
class aNWYourBricaud(aNWModel):
    uses_Chl = True

    def __init__(self, wave, prior_dicts=None):
        super().__init__(wave)
        self.fix_Chl = False  # True ⇒ Chl is fixed, not fitted
        if prior_dicts is not None:
            self.priors = bing_priors.Priors(prior_dicts)

    def set_aph(self, Chl):
        """Called by model_utils.init_other_bits before MCMC."""
        # Bricaud (1995) interpolation already set up at module top
        self.aph_star = f_b1998_A(self.wave) * Chl**(1 - f_b1998_E(self.wave))
        # Store anything else you need

    def eval_anw(self, params, **kwargs):
        params = np.atleast_2d(params)
        # use self.aph_star here
        ...
```

The `model_utils.init_other_bits(models, Chl=..., Y=..., Rrs=...)` call in the MCMC pipeline will dispatch to `set_aph(Chl)` automatically when `uses_Chl == True`.

## Common pitfalls (cross-ref [CLAUDE.md](../../../CLAUDE.md#common-pitfalls))

- **Forgot `np.atleast_2d`** → `eval_anw` crashes on 1D MCMC walker input.
- **Returned 1D** → breaks the `(nsample, nwave)` contract that `evaluate.reconstruct_from_chains` depends on.
- **Used linear amplitude params** → priors are log_uniform by default; sampler will hit `-inf` for any positive value > 5.
- **Reinitialized `self.a_w`** → don't; the base class already loads it from IOCCG via `ocpy.water.absorption`.
- **Used self.wave - λ0 without np.newaxis** → broadcasting will fail when params is `(nsample, 1)`.

## Verification checklist

After scaffolding:

- [ ] `init_model('YourName', wave)` returns an instance without error
- [ ] `eval_anw(params_1d)` returns shape `(1, nwave)`
- [ ] `eval_anw(params_2d)` returns shape `(nsample, nwave)`
- [ ] `pytest bing/tests/test_anw.py -v` passes
- [ ] A full round-trip through `bing.rt.rrs.calc_Rrs(model.eval_a(p), bb_model.eval_bb(bp))` yields finite Rrs

## Related skills

- [add-bbnw-model](../add-bbnw-model/SKILL.md) — companion for backscattering
- [run-bing-fit](../run-bing-fit/SKILL.md) — end-to-end fit once your model is registered
- [debug-priors](../debug-priors/SKILL.md) — when chains won't move

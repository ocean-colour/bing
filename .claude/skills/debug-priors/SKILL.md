---
name: debug-priors
description: Triage BING fits where log_prob returns -inf, MCMC chains won't move, or priors silently reject the initial guess. Walks through the most common causes (log10 vs linear mismatch, RatioPrior indices, missing set_aph/set_basis_func, prior tighter than the initial guess) and how to fix each. Use when the user says MCMC is stuck, chains aren't moving, or priors keep rejecting.
---

# Triage a stuck/-inf MCMC fit

## When to use this skill

- `log_prob` returns `-np.inf` at the initial guess
- MCMC acceptance is essentially zero
- All walkers stuck at the same point or sitting at a prior boundary
- "Prior rejected my initial guess"

## First: confirm log_prob is finite at p0

```python
from bing.fitting.inference import log_prob

print("log_prob(p0):", log_prob(p0, models, Rrs, varRrs, rt_dict))
```

If `-inf`, the failure is upstream of MCMC and the diagnostic table below applies. If finite but small, the fit will run but may converge slowly — see [diagnose-mcmc](../diagnose-mcmc/SKILL.md).

## Decision tree

### 1. Does `models[*].priors.calc(p_*)` already return `-inf`?

```python
ap = p0[:models[0].nparam]
bp = p0[models[0].nparam:]
print("a-prior:",  models[0].priors.calc(ap))   # should be 0 (log_uniform/uniform) or finite (gaussian)
print("bb-prior:", models[1].priors.calc(bp))
```

If `-inf`, one of your parameters violates a prior. Check each prior:

```python
for i, (name, prior) in enumerate(zip(
        models[0].pnames + models[1].pnames,
        models[0].priors.priors + models[1].priors.priors)):
    val = p0[i]
    if prior.flavor in ('uniform', 'log_uniform'):
        in_range = prior.pmin <= val <= prior.pmax
        print(f"{name}: {val:+.3f}  prior=[{prior.pmin:+.3f}, {prior.pmax:+.3f}]  ok={in_range}")
```

**Most common cause: log10 vs linear mismatch.**
- Priors with `flavor='log_uniform'` expect parameters **already in log10**. Adg=0.5 (linear) violates pmin=-6 / pmax=5 only if you forgot to log10 it first.
- Priors with `flavor='uniform'` are linear (typical for slopes Sdg, beta).

Fix: log10 the parameters whose prior starts with `log` before constructing `p0`. The L23 prep does this:

```python
ii = 0
for ss in [0, 1]:
    for prior in models[ss].priors.priors:
        if prior.flavor.startswith('log'):
            p0[ii] = np.log10(max(p0[ii], 1e-6))
        ii += 1
```

### 2. Does the forward model crash inside log_prob?

`log_prob` catches NaNs from the forward model and returns `-inf`. Symptoms:
- Priors are individually fine (return 0)
- `log_prob` still returns `-inf`

Diagnose by calling the forward model directly:

```python
from bing import evaluate
try:
    Rrs_pred = evaluate.calc_Rrs_from_models(
        models[0], ap, models[1], bp, rt_dict)
    print("Rrs_pred has NaN?", np.isnan(Rrs_pred).any())
    print("Rrs_pred range:", Rrs_pred.min(), Rrs_pred.max())
except Exception as e:
    print("Forward model crashed:", e)
```

Common causes:

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: 'aNWExpBricaud' object has no attribute 'aph_star'` | Forgot `set_aph(Chl)` | Call `model_utils.init_other_bits(models, Chl=...)` before fitting |
| `AttributeError: ...has no attribute '_shape'` (Lee) | Forgot `set_basis_func(Y)` | Same: `init_other_bits(models, Y=...)` |
| `ValueError: Need to set model G1, G2 for variable Gordon` | `rt_dict['variable_Gordon']=True` but `models[*].G1` is `None` | Either turn it off or set `G1, G2 = bing.rt.rrs.wave_dependent_gordon(wave)` and attach to both models |
| `Rrs_pred` is all NaN | `a + bb = 0` at some wavelength (parameter underflow) | Widen lower prior bound or improve initial guess |
| Negative Rrs | Sign error in your new model | Check the equation; Gordon's formula assumes a, bb > 0 |

### 3. Is `RatioPrior` configured wrong?

`RatioPrior` enforces `params[i0] ≈ ratio * params[i1]` (in linear space, after `10**`). Wrong indices silently constrain the wrong pair.

```python
# Configure: enforce 10**logAdg ≈ 5 * (10**logAph), with σ=20% of Adg
ratio_prior = dict(
    flavor='ratio',
    ratio=5.0,
    sigma=0.2,
    i0=0,   # index of Adg in the COMBINED params vector
    i1=2,   # index of Aph
)
```

Verify indices match `models[0].pnames + models[1].pnames` ordering:

```python
all_names = models[0].pnames + models[1].pnames
print("Ratio: enforce", all_names[ratio_prior['i0']], "≈",
      ratio_prior['ratio'], "*", all_names[ratio_prior['i1']])
```

`RatioPrior.calc` receives the **full** params array (not a per-parameter scalar) and is processed by `Priors.calc` as a trailing entry. Add it to the model **after** the per-parameter priors:

```python
models[0].priors.add_prior(ratio_prior)
```

### 4. Is the initial guess outside a tightened prior?

Common scenario: you tightened `Sdg`'s uniform prior to `(0.014, 0.018)` for stability, but your auto-init still picks `0.013`. Result: instant rejection.

```python
# Clamp p0 inside priors before fitting
for i, prior in enumerate(models[0].priors.priors + models[1].priors.priors):
    if prior.flavor in ('uniform', 'log_uniform'):
        if p0[i] < prior.pmin: p0[i] = prior.pmin + 1e-4
        if p0[i] > prior.pmax: p0[i] = prior.pmax - 1e-4
```

This is a safety net, not a fix — if the data really pulls `Sdg` outside the prior, your prior is too narrow.

### 5. Are walkers initialized in a degenerate region?

`run_emcee` initializes walkers as `p0 * (1 ± 0.01)`. If any parameter is exactly 0 (e.g., `beta = 0` initial guess for the Pow exponent), the perturbation is also 0 and all walkers start at the same point.

Fix: perturb non-zero p0, or replace zero with a small value before letting `run_emcee` do its replication:

```python
p0 = np.where(np.abs(p0) < 1e-3, 0.5, p0)  # avoid exact zeros in linear params
```

## Quick triage script

```python
import numpy as np
from bing.fitting.inference import log_prob
from bing import evaluate

def triage(p0, models, Rrs, varRrs, rt_dict):
    # 1. Individual priors
    ap = p0[:models[0].nparam]; bp = p0[models[0].nparam:]
    lp_a = models[0].priors.calc(ap)
    lp_b = models[1].priors.calc(bp)
    print(f"prior_a = {lp_a:+.3f}, prior_b = {lp_b:+.3f}")
    if not np.isfinite(lp_a) or not np.isfinite(lp_b):
        all_names = models[0].pnames + models[1].pnames
        all_priors = models[0].priors.priors + models[1].priors.priors
        for n, pr, v in zip(all_names, all_priors, p0):
            print(f"  {n}={v:+.3f} flavor={pr.flavor}",
                  f"range=[{getattr(pr,'pmin',None)}, {getattr(pr,'pmax',None)}]")
        return

    # 2. Forward model
    try:
        Rrs_p = evaluate.calc_Rrs_from_models(models[0], ap, models[1], bp, rt_dict)
        print(f"Rrs_pred min/max = {Rrs_p.min():.3e} / {Rrs_p.max():.3e}, "
              f"NaN={np.isnan(Rrs_p).any()}")
    except Exception as e:
        print(f"forward crashed: {type(e).__name__}: {e}")
        return

    # 3. Full log_prob
    lp = log_prob(p0, models, Rrs, varRrs, rt_dict)
    print(f"log_prob = {lp:+.3f}")
```

## Common pitfalls (cross-link)

- See [run-bing-fit](../run-bing-fit/SKILL.md) for the canonical p0 → log10 pattern.
- See [add-anw-model](../add-anw-model/SKILL.md) / [add-bbnw-model](../add-bbnw-model/SKILL.md) for the shape and log10 contracts that your custom model must obey.
- See [diagnose-mcmc](../diagnose-mcmc/SKILL.md) for what to do once log_prob is finite but chains still misbehave.

## Reference files

- [bing/fitting/inference.py](../../../bing/fitting/inference.py) — `log_prob`
- [bing/priors/priors.py](../../../bing/priors/priors.py) — prior classes and `set_standard_priors`
- [bing/models/utils.py](../../../bing/models/utils.py) — `init_other_bits` (Chl, Y, GIOP-from-Rrs)
- [bing/evaluate.py](../../../bing/evaluate.py) — `calc_Rrs_from_models`

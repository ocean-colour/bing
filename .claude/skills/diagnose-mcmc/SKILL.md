---
name: diagnose-mcmc
description: Diagnose MCMC chain health from a BING fit — convergence via integrated autocorrelation time, acceptance fraction, prior boundary pile-ups, walker spread, and corner-plot inspection. Use when the user asks "is this fit converged", "the posterior looks weird", or "chains are stuck".
---

# Diagnose MCMC chain health

## When to use this skill

- User asks "is this converged?", "did the MCMC behave?", "the posterior looks bimodal", "chains look stuck"
- Before publishing or before downstream analysis that assumes a clean posterior

## What to compute

For a chains array with shape `(nsteps, nwalkers, nparam)`:

| Diagnostic | Target | What it tells you |
|---|---|---|
| Acceptance fraction | 0.2–0.5 per walker | Too low → priors too wide or step size too large; too high → posterior too narrow |
| Integrated autocorr time τ | nsteps ≳ 50·τ | Effective sample size = nsteps·nwalkers / τ |
| Walker spread (post-burn) | All walkers overlap in 1D histograms | If one walker sits at the prior edge, it never moved |
| Prior boundary pile-up | <1% of post-burn samples within ε of pmin/pmax | Indicates a too-narrow prior |
| ESS per parameter | ≥ 1000 | Below ~200 the percentiles are noisy |

## Diagnostic script

```python
import numpy as np
import matplotlib.pyplot as plt

# chains: shape (nsteps, nwalkers, nparam), from inference.fit_one
nsteps, nwalkers, nparam = chains.shape
pnames = models[0].pnames + models[1].pnames

# ---------- 1. Trace plots (visual convergence check) ----------
fig, axes = plt.subplots(nparam, 1, figsize=(8, 1.6 * nparam), sharex=True)
for i, ax in enumerate(np.atleast_1d(axes)):
    ax.plot(chains[:, :, i], alpha=0.3, lw=0.5)
    ax.set_ylabel(pnames[i])
axes[-1].set_xlabel("step")
plt.tight_layout()
plt.savefig("trace.png", dpi=120)

# ---------- 2. Acceptance — only available from the sampler, not from chains alone.
# Rerun with chains_only=False to get it:
# sampler, idx = inference.fit_one(..., chains_only=False)
# print(sampler.acceptance_fraction)  # one value per walker

# ---------- 3. Integrated autocorrelation time ----------
import emcee
try:
    # emcee expects (nsteps, nwalkers, nparam), which is what fit_one returns
    tau = emcee.autocorr.integrated_time(chains, tol=50)
    print("τ per parameter:", tau)
    print("Rule of thumb: nsteps should be > 50·max(τ) =", 50 * tau.max())
    ess = (nsteps * nwalkers) / tau
    print("Effective sample size:", ess.astype(int))
except emcee.autocorr.AutocorrError as e:
    print("τ estimate unreliable — chain too short:", e)

# ---------- 4. Walker pile-up at prior bounds ----------
from bing.evaluate import thin_burn_chains
flat = thin_burn_chains(chains, burn=nsteps // 4)  # (nsamples, nparam)
all_priors = models[0].priors.priors + models[1].priors.priors
eps = 1e-3
for i, (name, prior) in enumerate(zip(pnames, all_priors)):
    if prior.flavor not in ('uniform', 'log_uniform'):
        continue
    frac_lo = np.mean(flat[:, i] < prior.pmin + eps * (prior.pmax - prior.pmin))
    frac_hi = np.mean(flat[:, i] > prior.pmax - eps * (prior.pmax - prior.pmin))
    if frac_lo > 0.01 or frac_hi > 0.01:
        print(f"⚠ {name}: {frac_lo:.1%} at pmin, {frac_hi:.1%} at pmax — widen prior")

# ---------- 5. Walker disagreement ----------
# If any walker's mean is >3σ from the global mean, it's likely stuck.
walker_means = chains[nsteps // 4:].mean(axis=0)  # (nwalkers, nparam)
global_mean  = walker_means.mean(axis=0)
global_std   = walker_means.std(axis=0)
for i, name in enumerate(pnames):
    bad = np.abs(walker_means[:, i] - global_mean[i]) > 3 * global_std[i]
    if bad.any():
        print(f"⚠ {name}: walker(s) {np.where(bad)[0]} appear stuck")
```

## Corner plot

```python
from bing import plotting
plotting.corner_plot(chains, models=models, outfile='corner.png', show=False)
```

This handles burn-in (default 7000 steps) and overlays 5/95 percentiles as dashed lines. If `show_log=True` (default) parameters are shown in log10 space as fitted.

## Reading the results

| Symptom | Likely cause | Action |
|---|---|---|
| `τ > nsteps/50` for every parameter | Sampler hasn't converged | Increase `nsteps` (try 80000) or `nburn` (try 5000) |
| One walker at prior edge | Bad initial draw | Increase initial-perturbation in `run_emcee` (currently ±1%) or restart |
| Multi-modal posterior | Real or numerical degeneracy | Inspect physically — could be a sign that the data don't constrain a parameter |
| Acceptance < 0.05 | Priors way too wide / proposal too aggressive | Tighten priors; rerun |
| Acceptance > 0.7 | Step size too small | Usually not a real problem in emcee (it's auto-tuned), but check for very narrow priors |
| Pile-up at pmin/pmax | Prior is constraining the posterior | Widen the relevant prior in `standard.py` |
| `log_prob = -inf` everywhere | Initial guess violates prior or model crashes | Run [debug-priors](../debug-priors/SKILL.md) |

## When to trust the posterior

A fit is publishable when:

- [ ] `acceptance ∈ [0.2, 0.5]` for all walkers
- [ ] `nsteps > 50 · max(τ)` for all parameters
- [ ] No pile-up at prior bounds (>1% of samples within ε)
- [ ] All walkers agree in marginal histograms (visual check)
- [ ] χ²ν of median fit is O(1) (use [plot-bing-fit](../plot-bing-fit/SKILL.md) to check)
- [ ] Posterior is unimodal OR the bimodality is physically interpreted

## Reading `calc_stats` output

```python
stats = evaluate.calc_stats(chains, names=pnames, perc=(14, 86))
# stats['med'][i]   → median of parameter i in fitted (log10) space
# stats['p14'][i]   → 14th percentile
# stats['p86'][i]   → 86th percentile
```

To report in linear space for an amplitude parameter:
```python
linear_med = 10**stats['med'][i]
linear_lo  = 10**stats['p14'][i]
linear_hi  = 10**stats['p86'][i]
```

## Common pitfalls

- **Don't call `np.mean` on raw chains** before burn-in removal; that mean is biased.
- **Don't use `np.std` as the uncertainty** — the posterior is rarely Gaussian; always use percentiles.
- **Don't run `thin_burn_chains` twice** — it permanently flattens the walker dimension.
- **`emcee.autocorr.integrated_time` raises** when the chain is too short; this is informative, not a crash.

## Related skills

- [run-bing-fit](../run-bing-fit/SKILL.md) — produced the chains
- [debug-priors](../debug-priors/SKILL.md) — fix the underlying problem
- [plot-bing-fit](../plot-bing-fit/SKILL.md) — visual sanity check on the result

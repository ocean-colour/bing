---
name: plot-bing-fit
description: Make standardized BING figures — three-panel Rrs/a_nw/bb_nw with credible bands, decomposed a_dg vs a_ph plots, and corner plots — using bing.plotting. Use when the user asks to plot a fit, make a figure for a paper, or visualize MCMC posteriors.
---

# Standardized BING figures

## When to use this skill

- "Plot this fit"
- "Make a figure for the paper showing the spectral fit"
- "Show the corner plot"
- "Compare retrieved a_dg/a_ph to truth"

## Three plotting entry points in `bing.plotting`

| Function | Output | Use |
|---|---|---|
| `show_fits(models, chains_or_params, rt_dict, ...)` | 3-panel Rrs + a_nw + bb_nw with credible bands | Default for "show me the fit" |
| `show_anw_fits(models, prep_chains, ...)` | a_dg / a_ph decomposition with bands | When you care about CDOM vs phytoplankton |
| `corner_plot(chains, models, ...)` | Full corner plot with 5/95% lines | MCMC diagnostics |

All use STIX fonts (publication quality) and accept `outfile` to save as PNG/PDF.

## Three-panel spectral fit

```python
from bing import plotting
from bing.rt import defs as rt_defs
from bing.parameters import standard

p = standard.expb_pow(satellite='PACE')
rt_dict = rt_defs.rt_dict_from_p(p)

# chains: shape (nsteps, nwalkers, nparam) from inference.fit_one
# OR a 1D params vector from chisq_fit.fit (function auto-detects via ndim)
axes, model_Rrs = plotting.show_fits(
    models, chains, rt_dict,
    ex_a_params=Chl,                                     # required for Bricaud-based abs models
    ex_bb_params=Y,                                      # required for Lee-based bb models
    Rrs_true=dict(wave=wave, spec=Rrs_obs, var=varRrs),  # data + variance → χ²ν shown on Rrs panel
    anw_true=dict(wave=wave_truth, spec=anw_truth),      # optional; plotted as black dots
    bbnw_true=dict(wave=wave_truth, spec=bbnw_truth),    # optional
    xqaa=None,                                           # optional XQAA overlay: {'wave','anw','bbnw'}
    perc=(5, 95),
    log_Rrs=True,                                        # Rrs panel y-axis
    log_abb=False,                                       # a/bb panels y-axis
    show_params=False,                                   # print fitted values inside Rrs panel
    outfile='figures/fit_idx170.png',
    show=False,
)
```

`axes` is `[ax_anw, ax_bb, ax_Rrs]`. `model_Rrs` is the median predicted Rrs on the model grid.

If `Rrs_true['var']` is provided, the Rrs panel label includes the reduced χ² of the median fit — quick sanity check that the model isn't biased.

## a_dg / a_ph decomposition

For ExpBricaud (and its descendants), the absorption model can split the total non-water absorption into dissolved+detrital and phytoplankton:

```python
from bing.evaluate import thin_burn_chains

prep_chains = thin_burn_chains(chains)  # (nsamples, nparam)

ax = plotting.show_anw_fits(
    models, prep_chains,
    anw_true=dict(wave=wave_truth,
                  a_dg=adg_truth, a_ph=aph_truth),  # optional
    perc=(5, 95),
    adg_clr='blue', aph_clr='green',
    outfile='figures/anw_decomp.png',
)
```

Prints `a_ph(440)` fit ± uncertainty (and truth, if given) to stdout — handy for table values.

Only works if the absorption model implements `eval_anw(..., retsub_comps=True)` returning `(a_dg, a_ph)`. ExpBricaud, ExpBricaudFix, ExpBricaudFree, and GIOP do; plain `Exp` does not.

## Corner plot

```python
fig = plotting.corner_plot(
    chains,
    models=models,
    show_log=True,         # parameters in log10 space as fitted
    outfile='figures/corner.png',
    show=False,
)
```

Burn-in of 7000 steps is applied internally. 5/95% lines are overplotted on the 1D marginals.

`show_log=False` converts to linear space first — only do this if you've checked it makes physical sense for each parameter (slopes/exponents are already linear, amplitudes go through `10**`).

## Multi-panel: stacking three fits side-by-side

```python
import matplotlib.pyplot as plt

fig, axes_grid = plt.subplots(3, 3, figsize=(14, 10))

for col, idx in enumerate([170, 1000, 2500]):
    chains_i, models_i, prep_i, _, extras_i = l23.fit_one(p, idx=idx)
    rt_dict_i = rt_defs.rt_dict_from_p(p)
    # show_fits builds its own figure; for a grid, call lower-level plotters directly
    # or save individual figures and combine in a vector editor.
```

`show_fits` always creates its own figure via `gridspec.GridSpec(1, 3)`. If you need a grid, either save individual PNGs and tile externally, or fork `show_fits` to accept an `axes` argument.

## Choosing band percentiles

| `perc=` | Meaning |
|---|---|
| `(5, 95)` | 90% credible interval (default for `reconstruct_from_chains` and `show_fits`) |
| `(14, 86)` | ~1σ Gaussian equivalent (default for `calc_stats`) |
| `(2.5, 97.5)` | 95% — wider band for conservative figures |
| `(25, 75)` | Interquartile — emphasizes the bulk of the posterior |

Pick once and use consistently across all figures in a paper.

## Font sizes & saving for publication

`mpl.rcParams['font.family'] = 'stixgeneral'` is set at module import. To override:

```python
import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})
```

For vector output:

```python
plotting.show_fits(..., outfile='figures/fit.pdf')   # vector
plotting.show_fits(..., outfile='figures/fit.png')   # 300 dpi by default
```

## Common pitfalls

- **Passing a 1D least-squares solution into `show_fits` without `ex_a_params`/`ex_bb_params`** → ExpBricaud or Lee will crash inside `reconstruct_chisq_fits`. Always pass Chl (for ExpBricaud) and Y (for Lee) even if `None`-equivalent.
- **`log_Rrs=False`** raises immediately — not implemented; the Rrs panel only works with log y.
- **`show_log=False` on linear-space corner plot** can show negative or zero values that look broken; that's just `10**param` for parameters whose log10 went strongly negative. Set sensible xlim before saving.
- **Truth dicts with different wavelength grids than `models[0].wave`** are plotted as-is (with `ko` markers), so they may extend beyond the retrieval x-range. Trim before plotting if undesired.
- **`corner_plot` burn=7000** is hard-coded; if your chain has fewer than 7000 steps, it'll silently return an empty plot. Use `chains[burn:]` of your choosing before calling, or fork the function.

## Quick checklist before publishing a figure

- [ ] `χ²ν` shown on Rrs panel is `O(1)` (no severe over/under-fitting)
- [ ] Credible bands enclose the truth (if simulating) at ~90% over wavelength
- [ ] Font sizes legible at print width
- [ ] Saved as both PNG (for previews) and PDF (for typeset paper)
- [ ] Color choices are colorblind-safe and distinguish a_dg/a_ph/bb_nw clearly

## Related skills

- [run-bing-fit](../run-bing-fit/SKILL.md) — produces the chains
- [diagnose-mcmc](../diagnose-mcmc/SKILL.md) — sanity-check before publishing
- [fit-l23-spectrum](../fit-l23-spectrum/SKILL.md) — provides truth dicts for the `*_true` arguments

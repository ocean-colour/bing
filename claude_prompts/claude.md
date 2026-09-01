# Improving the Claude experinece for BING

## CLAUDE.md

1. Modify the existing CLAUDE.md file to guide development of BING.

## Skills

Suggested project-specific skills for BING. Each entry maps to a folder under
`.claude/skills/<slug>/SKILL.md` (with optional `scripts/` and `references/`
subfolders, per the [Anthropic Agent Skills spec](https://github.com/anthropics/skills/tree/main/spec)).
The community organizes skills the same way — see
[ianhi/scientific-python-skills](https://github.com/ianhi/scientific-python-skills)
for a single-`.md`-per-library layout, and
[anthropics/skills](https://github.com/anthropics/skills) for the official
multi-asset layout.

### BING-specific skills

1. **`add-anw-model`** — Scaffold a new absorption model.
   - *Trigger*: "add a new a_nw model", "implement <ModelName> for absorption".
   - *Contents*: class skeleton (subclass `aNWModel`, set `nparam`, `pnames`, `uses_Chl`); registration in `bing/models/anw.py::init_model`; matching prior dict template; minimal test in `bing/tests/test_anw.py`; reminder of the log10 / `(nsample, nwave)` contracts.

2. **`add-bbnw-model`** — Scaffold a new backscattering model.
   - *Trigger*: "add a backscattering model", "new bb_nw".
   - *Contents*: subclass `bbNWModel`, set `nparam`/`pnames`/`uses_basis_params`, wire into `init_model`, add to `bing/parameters/standard.py` if it should be part of a combo, write a `eval_bbnw` returning `(nsample, nwave)`.

3. **`run-bing-fit`** — Canonical end-to-end MCMC fit for a single spectrum.
   - *Trigger*: "fit this Rrs", "run an MCMC on …".
   - *Contents*: walk through `standard.expb_pow` → `model_utils.init` → set priors → `chisq_fit.fit` initial guess → `inference.init_mcmc` → `inference.fit_one` → `evaluate.calc_stats` / `reconstruct_from_chains`. Includes a checklist of "did you set `set_aph(Chl)` / `set_basis_func(Y)`?"

4. **`diagnose-mcmc`** — Chain-health diagnostics.
   - *Trigger*: "is this fit converged?", "weird posterior", "chains stuck".
   - *Contents*: integrated autocorrelation time via emcee, acceptance fraction targets, prior boundary pile-up checks, log10-vs-linear sanity check, ess per walker, corner-plot template.

5. **`fit-l23-spectrum`** — Loisel 2023 synthetic-truth workflow.
   - *Trigger*: "test against L23", "validate model on Loisel".
   - *Contents*: `bing.fitting.l23.load_one_l23` usage, comparing fitted IOPs vs. truth (`a`, `bb`, `Chl`, `Sdg`, `Y`), batch loop over the dataset, recommended figures (truth-vs-fit scatter, residual spectra).

6. **`inelastic-rrs`** — Adding Raman scattering and/or chlorophyll fluorescence.
   - *Trigger*: "include Raman", "add fluorescence", "use `calc_Rrs_with_*`".
   - *Contents*: when to use `bing.rt.rrs.calc_Rrs` vs. `calc_Rrs_with_raman` vs. `calc_Rrs_with_fluorescence`; required parameters and their priors; how the fluorescence line shape works; cross-reference [bing/rt/chl_fl.py](../bing/rt/chl_fl.py) and [bing/rt/raman.py](../bing/rt/raman.py).

7. **`satellite-band-prep`** — Prep hyperspectral Rrs for PACE/MODIS/SeaWiFS/SBG.
   - *Trigger*: "simulate a PACE spectrum", "downsample to MODIS bands".
   - *Contents*: `bing.preproc` interpolation patterns, `ocpy.satellites.{pace,modis,seawifs}.gen_noise_vector`, sanity checks on band centers, and the wavelength-grid pitfall from CLAUDE.md.

8. **`batch-fit-argo`** — Parallel batch fitting for Argo BGC matchups.
   - *Trigger*: "fit all Argo profiles", "run the biomass pipeline".
   - *Contents*: `inference.fit_batch` with `n_cores`, output NPZ layout, resumable iteration over `matched_argo_bgc_profiles_bbp.csv`, and the [papers/biomass/](../papers/biomass/) directory layout.

9. **`plot-bing-fit`** — Standardized spectral plots with credible bands.
   - *Trigger*: "plot this fit", "make a figure for the paper".
   - *Contents*: use `bing.plotting`, percentile shading (5/95), data-vs-model residual panel, log-y for a/bb, layout for stacked Rrs + a_nw + bb_nw panels.

10. **`debug-priors`** — Triage stuck/empty chains.
    - *Trigger*: "log_prob is -inf", "MCMC won't move", "prior rejected".
    - *Contents*: enumerate causes (log10 vs linear range, `RatioPrior` numerator/denominator order, missing `set_aph` / `set_basis_func`, prior tighter than initial guess), with a short checklist that maps each symptom to a fix.

11. **`add-paper-analysis`** — Lay out a new `papers/<topic>/` directory.
    - *Trigger*: "start a new analysis", "new paper directory".
    - *Contents*: standard subfolders (`Analysis/`, `Figures/`, `Data/`), conventions for keeping reusable code in `bing/` not `papers/`, and a template `README.md`.

### Community / general-purpose skills worth installing

These aren't BING-specific but pair well with this workflow:

- **[ianhi/scientific-python-skills](https://github.com/ianhi/scientific-python-skills)** — `xarray.md`, `zarr.md`, `icechunk.md`. Relevant because PACE L2/L3 products, Argo BGC profiles, and the L23 synthetic dataset are often consumed as xarray/zarr.
- **[K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills)** — 140 science skills incl. matplotlib, pandas, scipy idioms; useful for the analysis side of `papers/`.
- **[anthropics/skills](https://github.com/anthropics/skills)** — official document skills (`docx`, `pdf`, `pptx`, `xlsx`); handy for assembling figures and tables into manuscripts under [papers/](../papers/).
- **[travisvn/awesome-claude-skills](https://github.com/travisvn/awesome-claude-skills)** and **[ComposioHQ/awesome-claude-skills](https://github.com/ComposioHQ/awesome-claude-skills)** — curated indexes to browse for further additions.
- **[honnibal/claude-skills](https://github.com/honnibal/claude-skills)** — small, opinionated set worth reading as a style reference for writing your own SKILL.md.

### SKILL.md frontmatter template (Anthropic spec)

```markdown
---
name: add-anw-model
description: Scaffold a new absorption (a_nw) model class in bing/models/anw.py with matching prior dict, init_model registration, and unit test.
---

# Add a new a_nw model

(Body: instructions, code skeletons, links to CLAUDE.md sections, common
pitfalls. Keep under ~30KB so the whole skill fits in context when loaded.)
```

## Prompts

1. Perform the 1st command under CLAUDE.md
2. Given your understanding of the code base, provide a list of suggested skills to add for Claude.  Provide the list in the Skills section above.  Include examples from GitHub that the community has generated too. 
3. Please proceed to generate/install of the skills in the Skills section above.
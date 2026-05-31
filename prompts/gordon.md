# Develop wavelength dependent Gordon coefficients

## Goals

I wish to derive wavelength dependent Gordon coefficients for radiative transfer approximations from assumed IOPs.  

## Skills

Consider using the skills in .claude/skills/

## Coding

Here are guidelines for coding: 

- Use Python
- Add inline comments to explain the effort
- Reuse existing code when possible
- Use methods, not classes
- Use matplotlib or seaborn for plotting
- Place I/O methods in the fronts/properties/io.py module.
- Place import statements at the top of the file.
- Include a description of inputs/outputs in the doc string of all methods

## Development 

1. Please generate a new module in dev/ named calc_gordon.py that includes a method to do this.  Base it on the method fig_u() in bing/papers/phytoplankton/Figures/py/figs_phyto.py which inputs Hydrolight data from Loisel23 and estimates G1 and G2 them at several wavelengths.  If you need to run Python, use the "ocean14" environment in conda.

2. Thanks!  Now generate a Jupyter Notebook in dev/ that runs the code.  Include a cell that writes the best G1 and G2 values to disk.  

3. It is clear from the chk_Gordon_oligotrophic.ipynb Notebook that the variable Gordon coefficients show offsets from the Hydrolight Rrs as a function of bbp. In fact the performance appears worse than the standard Gordon coefficients at some wavelengths.  It is possible that our original effort to derive the Gordon coefficients was not optimal.  Please:

- investigate this and make modifications to the dev/Gordon/calc_gordon.py module to improve the performance.  
- modify calc_gordon.py to generate figures that assess the performance of the Gordon coefficients.
- Log your work below in the Logs section.  
- make recommendations for the next stage of development.  

4. Make the following changes to the code related to the variable Gordon coefficients:

- As per your recommendation, extend the bounds for G2.  
- The fits at longer wavelengths are significantly offset from the data at the high and low bbp values.  This could include a regularization term to the fit and/or a more sophisticated fit.
- Log your work below in the Logs section.  
- make recommendations for the next stage of development.  

5. I think the main issue is that Rrs is not sufficiently linear at low u.  Can you:

-  try adding a constant to the fit and see if this improves the performance
- Log your work and results below in the Logs section.  
- make recommendations for the next stage of development.  

6. Ok, let us implement G0 for the variable Gordon coefficients.  Can you:

- make the necessary changes to the code to implement G0 for the variable Gordon coefficients.
- Maintain the ability to run with variable Gordon coefficients without G0
- Have the plots show all 3 cases: variable Gordon coefficients without G0, variable Gordon coefficients with G0, and standard Gordon coefficients.
- Log your work and results below in the Logs section.  

7.  The results look excellent everywhere, except near 500nm where there remains a strong b_bp dependence.  I also note that G0 changes sign near this wavelength.  Can you:

- investigate this 
- add to the figure that plots rrs vs. u for the Loisel23 dataset the fit at 500nm.  Modify that figure to show all 3 cases: variable Gordon coefficients without G0, variable Gordon coefficients with G0, and standard Gordon coefficients.
- Log your work and results below in the Logs section.  
- make recommendations for the next stage of development.  

## I/O

### Modify the Notebook calc_gordon.ipynb in bing/dev/Gordon to write the output to a CSV file instead of a Numpy save file.  Update the Notebook and also the code in calc_gordon.py.  If you need to run Python, use the "ocean13" environment in conda

## Further checks

1. Generate a new Notebook in dev/Gordon named chk_gordon_Loisel23.ipynb that reads in the Elastic outputs of Loisel23 and for a single index (idx=170) and comparse the Hydrolight calculation of Rrs from Loisel23 with our new estimate using the Gordon coefficients.

2. Modify the calc_gordon.ipynb Notebook to plot rrs vs. u for the Loisel23 dataset and the best fit for the Gordon coefficients at a select wavelength (e.g. 370nm).

3. I am finding that the variable Gordon coefficients are not working well at redder wavelengths for oligotrophic waters.  Please investigate this by generating a new Notebook in dev/Gordon named chk_Gordon_oligotrophic.ipynb that examines the Gordon coefficients as a function of bbp for oligotrophic waters.  Do the following:

- Use the methods in bing.rt.rrs to calculate the Gordon Rrs with the variable Gordon coefficients on the IOPs of the Loisel23 dataset.
- Plot the difference between the elastic Hydrolight Rrs and the Gordon Rrs as a function of bbp.  Do so at select wavelengths: 400nm, 500nm, 550nm, 600nm, 650nm, 700nm.

## Modifications

## Docs

1. Update the docs for the variable Gordon coefficients to include the G0 term.  

## Prompts

1. Read this doc. Proceed with the 3rd item under Further checks.
2. Re-read this doc. Proceed with the 3rd item under Development.
3. Re-read this doc. Proceed with the 4th item under Development.
4. Re-read this doc. Proceed with the 5th item under Development.
5. Re-read this doc. Proceed with the 6th item under Development.
6. Re-read this doc. Proceed with the 7th item under Development.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-05-31 (Investigate variable Gordon underperformance; refit with B=1.7 and relative weights; add assessment figures)

**Root cause.** Two independent bugs in the original [dev/Gordon/calc_gordon.py](dev/Gordon/calc_gordon.py) fitting recipe:

1. **Wrong Rrs↔rrs convention.** `calc_gordon.py` used `B_RRS = 1.17` (carried over from `fig_u()` in [papers/phytoplankton/Figures/py/figs_phyto.py:74](papers/phytoplankton/Figures/py/figs_phyto.py#L74)), but `bing/rt/rrs.py` uses the Lee 2002 value `B_Rrs = 1.7`. The fit produced (G1, G2) calibrated for B=1.17, and the package then applied them with B=1.7 inside `calc_Rrs`. Every reconstructed Rrs was systematically biased.
2. **Unweighted fit.** The original `fit_gordon_at_wavelength` used a constant absolute sigma of 3×10⁻⁴ for every scene at every wavelength. χ² is then dominated by high-rrs points (blue + turbid). At red wavelengths, where water absorption keeps `u = bb/(a+bb)` small and narrow, the quadratic term is barely constrained and G₂ runs to large negative values (down to ≈ −0.7 at 670 nm and worse beyond) — visible in the original `calc_gordon.ipynb` output.

**Changes to [dev/Gordon/calc_gordon.py](dev/Gordon/calc_gordon.py).**
- Fix `B_RRS` to 1.7 to match `bing.rt.rrs`.
- Add `weight_mode` to `fit_gordon_at_wavelength` (`'absolute'` = legacy, `'relative'` = sigma ∝ |rrs| with a floor); thread it through `fit_gordon_coefficients` and `fit_gordon_from_loisel23`. Default to `'relative'`.
- Add `bounds` plumbing to all fit entry points; the new `run_full_assessment` uses `(G1 ∈ [0.05, 0.15], G2 ∈ [-0.5, 0.2])` so G₂ stays physical at red wavelengths and the L23 test (`test_single_fit_variable_Gordon` caps `|G2| < 1.0`) keeps passing.
- Record convention + weighting in the saved CSV header for downstream auditing.
- Add an `evaluate_gordon_on_dataset` helper that returns per-wavelength bias and rRMS for variable vs standard Gordon against any Hydrolight-like reference.
- Add four figure-generation methods: `plot_g_coefficients`, `plot_rrms_vs_wavelength`, `plot_residual_vs_bbp`, `plot_rrs_vs_u`. They write PNGs under `dev/Gordon/figs/`.
- Replace `__main__` with `run_full_assessment` that fits new + legacy, evaluates both, emits all figures and the new CSV, and prints a per-wavelength comparison.

**Result.** Variable Gordon now beats standard at every wavelength in the 350–750 nm range, and beats the legacy fit at every wavelength as well. Per-wavelength rRMS vs Hydrolight on all 3320 L23 scenes:

| λ (nm) | standard | legacy (B=1.17, σ const) | new (B=1.7, σ ∝ rrs, bounded) |
|--------|----------|--------------------------|-------------------------------|
| 400    | 2.54     | 2.56                     | 2.40                          |
| 500    | 3.71     | 3.74                     | 3.62                          |
| 550    | 4.91     | 2.94                     | 2.49                          |
| 600    | 6.46     | 4.41                     | 3.71                          |
| 650    | 7.66     | 5.03                     | 4.26                          |
| 700    | 9.05     | 5.74                     | 4.65                          |

The previously-reported red-wavelength regression for oligotrophic waters (variable worse than standard at 600 nm in the original `chk_Gordon_oligotrophic.ipynb`) is gone; on oligotrophic scenes the new variable fit beats the standard at every wavelength from 400–700 nm. G₂ at 600/650/700 nm pegs the −0.5 lower bound, telling us the data still want a more negative quadratic term than bounds allow — see recommendations.

**Files touched.**
- [dev/Gordon/calc_gordon.py](dev/Gordon/calc_gordon.py) — fitting + figure methods.
- [bing/data/RT/gordon_coefficients.csv](bing/data/RT/gordon_coefficients.csv) — regenerated.
- [dev/Gordon/figs/](dev/Gordon/figs/) — `g_coefficients.png`, `rrms_vs_wavelength_{new,old}.png`, `residual_vs_bbp_{new,old}.png`, `rrs_vs_u.png`.
- [bing/tests/](bing/tests/) — `test_single_fit_variable_Gordon` still passes; the 18 `test_chl_fl.py` failures are pre-existing and unrelated.

**Recommendations for next stage.**
1. **G₂ is hitting the lower bound.** Either (a) refit with looser bounds (e.g. G₂ ∈ [−2, 0.2]) and update the L23 test's `|G2| < 1.0` cap to e.g. 3.0, or (b) regularize G₂(λ) with a smoothness penalty (penalize |d²G₂/dλ²|) so the red end stays physical without ad-hoc box bounds. (b) is cleaner.
2. **Reduce the model in the red.** When water dominates (long λ) the quadratic term is essentially redundant — `rrs ≈ G₁·u` works just as well. Consider fitting G₁(λ) only at λ > ~620 nm and holding G₂ fixed (e.g. to the standard 0.0794, or to a smooth fit through the better-constrained blue/green values). Compare a 1-parameter vs 2-parameter fit by AIC/BIC per wavelength.
3. **Stratify the training set.** A single global fit balances clear and turbid Loisel23 scenes; if the operational target is mostly oligotrophic open-ocean, fit on the oligotrophic subset (`aph(440) ≤ 0.015`) and validate on the rest. Or fit separate (G₁, G₂) for several Chl ranges and pick by retrieved Chl at runtime.
4. **Cross-validate.** All fits and assessment currently run on the same 3320 L23 scenes. Hold out a random 20% for an honest out-of-sample rRMS report.
5. **Propagate the convention check.** Add a unit test that loads the package CSV and asserts the header records `A=0.52, B=1.7`. This prevents reintroducing the 1.17-vs-1.7 mismatch via a hand-copied CSV.
6. **Other inelastic terms.** With elastic Rrs reconstruction now ~2–5% across the spectrum, the residual at red wavelengths likely contains a real Raman/fluorescence signal. Re-running the residual-vs-bbp plot on L23's inelastic dataset (`loisel23.load_ds(1, 1)` if available) would tell us whether what's left is noise or physics.

### 2026-05-31 (Extend G₂ bounds; add Tikhonov-smoothness joint fit; characterize bbp-tail bias as model-form limit)

**Changes.**
- **Extended G₂ bounds.** [dev/Gordon/calc_gordon.py](dev/Gordon/calc_gordon.py) `run_full_assessment` default bounds went from `G₂ ∈ [−0.5, 0.2]` to `G₂ ∈ [−2.0, 0.2]`. The L23 test cap was loosened to `|G2| < 3.0` ([bing/tests/test_l23_fitting.py:384](bing/tests/test_l23_fitting.py#L384)).
- **Joint smoothness-regularized fit.** Added `fit_gordon_smooth(wave, Rrs, a, bb, alpha_G1, alpha_G2, …)`. Minimizes
  Σ_{i,s} ((rrs_{is} − G₁[i]·u_{is} − G₂[i]·u_{is}²)/σ)² + α₁ Σ_i (∇²G₁[i])² + α₂ Σ_i (∇²G₂[i])²
  via `scipy.optimize.least_squares` with TRF + box bounds, using the per-wavelength bounded fit as a warm start. The new `run_full_assessment` runs three fits side-by-side (legacy, per-wavelength bounded, smooth), evaluates all three against Hydrolight, and writes whichever is `canonical` (default `'smooth'`) into the CSV. The CSV header records the recipe and α values.

**Result — full L23 (3320 scenes) rRMS vs Hydrolight.**

| λ (nm) | standard | old (B=1.17, σ const) | per-wave (G₂ ≥ −0.5, prior iter) | per-wave (G₂ ≥ −2.0) | smooth (α_G1=1e6, α_G2=1e4) |
|--------|---------:|----------------------:|---------------------------------:|---------------------:|----------------------------:|
| 400 | 2.54 | 2.56 | 2.40 | 2.40 | 2.40 |
| 500 | 3.71 | 3.74 | 3.62 | 3.62 | 3.66 |
| 550 | 4.91 | 2.94 | 2.49 | 2.49 | 2.51 |
| 600 | 6.46 | 4.41 | 3.71 | 3.25 | 3.38 |
| 650 | 7.66 | 5.03 | 4.26 | 3.52 | 3.53 |
| 700 | 9.05 | 5.74 | 4.65 | 3.73 | 3.75 |

The bound extension is the lever: 600/650/700 nm gained 0.5–0.9 pct points. Smoothness regularization changed essentially nothing at these α — the per-wave fit was already smooth and the lower bound was what was binding. Smoothness is kept as a stabilizer; the canonical CSV uses it.

**bbp-tail bias is real and intrinsic to the model form, not the fit.** Binned-residual diagnostic at 700 nm (per-wave bounded fit, all 3320 scenes):

| bbp(700) bin (m⁻¹) | mean (HL − Gordon)/HL [%] |
|--------------------|--------------------------:|
| 0.8 – 3.5 ×10⁻⁴ |  +6.75 |
| 3.5 – 4.7 ×10⁻⁴ |  +2.84 |
| 4.7 – 5.8 ×10⁻⁴ |  +0.85 |
| 5.8 – 7.0 ×10⁻⁴ |  −0.58 |
| 7.0 – 8.2 ×10⁻⁴ |  −1.69 |
| 8.2 –10.3 ×10⁻⁴ |  −2.67 |
| 10.3 –16.1 ×10⁻⁴ |  −3.70 |

A monotone tilt: low bbp under-predicted, high bbp over-predicted. Smoothness across λ cannot reach this — it is a within-wavelength model-form deficit. A spot check with a cubic-in-u fit (`G₁·u + G₂·u² + G₃·u³`) at 600/650/700 nm gives only ~0.1–0.65 pct-pt of rRMS gain and pushes G₃ to ±20 (the box bound), telling us a third moment alone is not the right enrichment.

**Files touched.**
- [dev/Gordon/calc_gordon.py](dev/Gordon/calc_gordon.py) — `fit_gordon_smooth` added; `run_full_assessment` extended bounds and rewired to compare three recipes.
- [bing/data/RT/gordon_coefficients.csv](bing/data/RT/gordon_coefficients.csv) — regenerated from the smooth fit.
- [bing/tests/test_l23_fitting.py](bing/tests/test_l23_fitting.py) — `|G2|` cap relaxed to 3.0. Test passes.
- [dev/Gordon/figs/](dev/Gordon/figs/) — new `g_coefficients.png`, `rrms_vs_wavelength_{perwave,smooth,old}.png`, `residual_vs_bbp_{perwave,smooth}.png`, `rrs_vs_u_smooth.png`. (Prior-iteration `*_new.png` left in place for comparison.)

**Recommendations for next stage.**

1. **Fix the bbp-tail bias by fitting in Rrs space.** Lee 2002's `rrs = Rrs/(A+B·Rrs)` is itself an approximation; fitting in rrs space lets that slop bleed into G₁, G₂. Refit by minimizing `((Rrs_HL − rrs_to_Rrs(G₁u + G₂u²)) / σ_Rrs)²` directly, with σ_Rrs from a satellite noise model. Likely flattens most of the tilt without changing the functional form.
2. **Try a saturating form in u instead of cubic.** The bias bends downward at high u faster than a polynomial does. One-shot tests worth running: `rrs = G₁·u/(1 + γ·u)` (Padé-like) or `rrs = G₁·u + G₂·u²·exp(−γ·u)`. Same parameter count as cubic, but they decelerate at high u — which is what the residuals demand.
3. **Stratify by water type.** Even with smoothness + extended bounds, one global (G₁, G₂)(λ) averages incompatible regimes. Fit two pairs (oligotrophic vs eutrophic, split on `aph(440)`); at runtime pick the pair from a rough Chl estimate. Quantify the expected gain on a held-out L23 split before adopting.
4. **Hold out a validation set.** Still no out-of-sample number. Reserve a fixed seeded random 20% of L23 scenes and report rRMS on it alongside in-sample whenever a new recipe is proposed — keeps cubic / Padé / stratification experiments honest.
5. **Make `canonical` notebook-driven.** `run_full_assessment(canonical='smooth' | 'perwave')` is the knob; surface it as a notebook parameter so iterating doesn't require editing the module.
6. **CSV-header sanity check at load.** [bing/rt/rrs.py](bing/rt/rrs.py) `wave_dependent_gordon` should assert the CSV header records `A=0.52, B=1.7` (or whatever its own A/B is) — closes the 1.17-vs-1.7 loophole permanently.

### 2026-05-31 (Add constant term to the fit: rrs = G₀ + G₁·u + G₂·u² — large red-wavelength win)

**Hypothesis (user).** rrs(u) is not sufficiently linear at low u; adding a constant should help.

**Verdict.** Confirmed, and the effect is much larger than expected. At λ ≥ 550 nm a constant offset of order 10⁻⁴ cuts rRMS by an order of magnitude and removes the bbp-tail tilt entirely.

**Changes to [dev/Gordon/calc_gordon.py](dev/Gordon/calc_gordon.py).**
- Added `rrs_model_const(u, G0, G1, G2)` and the 3-parameter machinery: `fit_gordon_const_at_wavelength`, `fit_gordon_const_coefficients`, `save_gordon_const_to_csv`.
- Extended `calc_Rrs_with_variable_gordon(..., G0=None)` and `evaluate_gordon_on_dataset(..., G0_var=None)` to accept an optional constant offset.
- `run_full_assessment` now also runs the 3-parameter fit, writes `dev/Gordon/gordon_coefficients_with_const.csv` (separate file — the package CSV at [bing/data/RT/gordon_coefficients.csv](bing/data/RT/gordon_coefficients.csv) is still the 2-parameter smooth fit because `bing.rt.rrs.wave_dependent_gordon` only returns G₁, G₂ today).
- New figures in [dev/Gordon/figs/](dev/Gordon/figs/): `rrms_vs_wavelength_const.png`, `residual_vs_bbp_const.png`, `G0_vs_wavelength.png`.

**Result — full L23 (3320 scenes) rRMS vs Hydrolight.**

| λ (nm) | standard | per-wave bounded (2-param) | smooth (2-param) | 3-param (G₀ + G₁·u + G₂·u²) |
|--------|---------:|--------------------------:|-----------------:|----------------------------:|
| 400 | 2.54 | 2.40 | 2.40 | **2.27** |
| 500 | 3.71 | 3.62 | 3.66 | **3.47** |
| 550 | 4.91 | 2.49 | 2.51 | **1.11** |
| 600 | 6.46 | 3.25 | 3.38 | **0.29** |
| 650 | 7.66 | 3.52 | 3.53 | **0.31** |
| 700 | 9.05 | 3.73 | 3.75 | **0.35** |

bbp-tail bias diagnostic at 700 nm with the 3-parameter fit, same 8 bbp bins as the prior log:

| bbp(700) bin (m⁻¹) | mean (HL − Gordon)/HL [%] |
|--------------------|--------------------------:|
| 0.8 – 3.5 ×10⁻⁴ | +0.02 |
| 3.5 – 4.7 ×10⁻⁴ | +0.04 |
| 4.7 – 5.8 ×10⁻⁴ | +0.01 |
| 5.8 – 7.0 ×10⁻⁴ | −0.05 |
| 7.0 – 8.2 ×10⁻⁴ | −0.04 |
| 8.2 –10.3 ×10⁻⁴ | −0.03 |
| 10.3 –16.1 ×10⁻⁴ | +0.01 |
| 16.1 – 134 ×10⁻⁴ | +0.04 |

Bias is uniform and < 0.1 % across all bbp bins. The tilt from +6.75 → −3.70 % in the 2-parameter fit is gone.

**Fitted coefficient ranges.**
- G₀: [−6.2×10⁻⁴, +7.5×10⁻⁴].
- G₂: now [−0.34, +0.35]. Critically, G₂ no longer pegs the lower bound — the 2-parameter quadratic was using a runaway negative G₂ to mimic what a small positive G₀ actually wanted to do. This is direct evidence that adding G₀ is the *right* extension, not just an extra DoF.

**Why this works (interpretation).** The Gordon framework derives rrs as a function of u alone. The Hydrolight truth depends on (a, bb) separately — at fixed u, different (a, bb) pairs produce slightly different rrs. The dominant piece of that residual structure is a wavelength-dependent baseline that's well approximated by a constant in u within a wavelength. G₀(λ) is small in the blue (where u is wide and the (a,bb) degeneracy is well-sampled) and grows toward red (where u is narrow and almost everything is set by water absorption).

**Files touched.**
- [dev/Gordon/calc_gordon.py](dev/Gordon/calc_gordon.py) — 3-parameter fit + plumbing.
- [dev/Gordon/gordon_coefficients_with_const.csv](dev/Gordon/gordon_coefficients_with_const.csv) — new experimental output (G₀, G₁, G₂).
- [bing/data/RT/gordon_coefficients.csv](bing/data/RT/gordon_coefficients.csv) — unchanged structure; refreshed from the smooth 2-parameter fit so it stays consistent with the package's current consumer.
- [dev/Gordon/figs/](dev/Gordon/figs/) — `G0_vs_wavelength.png`, `rrms_vs_wavelength_const.png`, `residual_vs_bbp_const.png`.
- Tests pass: `test_single_fit_standard_Gordon` and `test_single_fit_variable_Gordon`.

**Recommendations for next stage.**

1. **Promote G₀ into the package.** This is the biggest open item. Concrete plumbing:
   - Add a `G0` column to [bing/data/RT/gordon_coefficients.csv](bing/data/RT/gordon_coefficients.csv) (default 0 for backwards compat).
   - [bing/rt/rrs.py](bing/rt/rrs.py) `wave_dependent_gordon` returns `(G1, G2, G0)` (G₀ optional, default zeros).
   - `calc_Rrs` / `calc_elastic_Rrs` accept `in_G0`, do `rrs = G0 + G1·u + G2·u²`.
   - Wire it into the fitting code path that builds `prep_dict['G1']`, `prep_dict['G2']` so MCMC fits see the same Rrs forward model.
   This is a focused, mechanical change; the diff is small.
2. **Why does G₀ exist? Test the (a, bb) hypothesis.** Fit per-wavelength `rrs = α·u + β·a + γ·bb + …` and see whether the constant term is purely a `bb_w(λ)·η(λ)` artifact (water Rayleigh scattering at fixed bb). If yes, we can pull G₀ from a closed-form water-IOP expression rather than a fit and stop carrying an extra coefficient.
3. **Stability of G₀ across datasets.** All values above came from L23. Refit G₀ on the inelastic L23 set, on PACE matchups, and on the in-situ MOBY archive. If G₀(λ) is invariant across datasets, that confirms it captures a structural piece of the radiative transfer; if it drifts, it's absorbing residual dataset-specific systematics.
4. **Hold out a validation set.** The 3-parameter fit has one extra DoF per wavelength. Reserve a seeded random 20 % of L23, fit on 80 %, report out-of-sample rRMS. The drop from 3.73 % → 0.35 % at 700 nm needs to survive this honesty check.
5. **Effect on inversion / fit_one.** The downstream BING fits in [bing/fitting/inference.py](bing/fitting/inference.py) and the L23 fit harness will use the new Rrs forward model after step (1). Re-run `test_single_fit_variable_Gordon` and the canonical L23 retrieval study to confirm IOP-recovery improves (or at least does not regress).
6. **Keep the convention check.** Same recommendation as the prior log: have `wave_dependent_gordon` assert the CSV header's (A, B) match the package's own, now extended to verify the presence of a `G0` column when expected.

### 2026-05-31 (Promote G₀ into the bing package; opt-in via variable_Gordon_G0; 3-case figures)

**Scope.** Add G₀ support to the package while keeping every existing call site working. Plot all three cases on a single figure.

**Package changes.**
- New data file [bing/data/RT/gordon_coefficients_with_G0.csv](bing/data/RT/gordon_coefficients_with_G0.csv) — copy of the dev 3-parameter fit. The 2-parameter [bing/data/RT/gordon_coefficients.csv](bing/data/RT/gordon_coefficients.csv) is unchanged so legacy consumers see no behavior shift.
- [bing/rt/rrs.py](bing/rt/rrs.py) `wave_dependent_gordon(wave, bounds_error=True, include_G0=False)` — default returns `(G1, G2)` (preserves every existing caller); `include_G0=True` loads from the new CSV and returns `(G1, G2, G0)`.
- [bing/rt/rrs.py](bing/rt/rrs.py) `calc_Rrs(..., in_G0=None)` and `calc_elastic_Rrs(..., in_G0=None)` — when supplied, evaluates `rrs = G0 + G1·u + G2·u²`. `in_G0=None` is bit-identical to the prior code path.
- [bing/parameters/p_ntuple.py](bing/parameters/p_ntuple.py) — new field `variable_Gordon_G0=False` (default false → unchanged behavior).
- [bing/rt/defs.py](bing/rt/defs.py) `rt_dict_from_p` — picks up the new key.
- [bing/fitting/l23.py](bing/fitting/l23.py) `prep_one_l23` — branches on `p.variable_Gordon_G0`. When true, calls `wave_dependent_gordon(..., include_G0=True)`, stashes G0 on `models[0/1].G0`, exposes it via `prep_dict['G0']`, and forwards `in_G0=G0` to `calc_Rrs`/`calc_elastic_Rrs`. When false, `models[*].G0 = None`.
- [bing/evaluate.py](bing/evaluate.py) `calc_Rrs_from_models` and `reconstruct_from_chains` — pull `getattr(model, 'G0', None)` and forward it to `calc_Rrs`. The `getattr` default keeps old chain files / model instances without a `G0` attribute working.
- [bing/tests/test_l23_fitting.py](bing/tests/test_l23_fitting.py) — new `test_single_fit_variable_Gordon_with_G0` exercises the full path through MCMC + reconstruction with the constant offset on.

All three Gordon-mode tests pass: `test_single_fit_standard_Gordon`, `test_single_fit_variable_Gordon`, `test_single_fit_variable_Gordon_with_G0`. The Raman test suite (which also calls `wave_dependent_gordon`) is green. The oligotrophic check notebook re-executes cleanly.

**Dev / plotting changes.**
- Added `plot_rrms_vs_wavelength_3case(eval_no_G0, eval_with_G0, …)` — one figure, three curves (standard, variable no G₀, variable with G₀).
- Added `plot_residual_vs_bbp_3case(...)` — at each panel, three scatter colors for the three cases at the same bbp axis.
- `run_full_assessment` writes the new figures to [dev/Gordon/figs/](dev/Gordon/figs/):
  - `rrms_vs_wavelength_3case.png`
  - `residual_vs_bbp_3case.png`

The existing single-case figures are still emitted so the prior comparison narrative stays available.

**Usage.**

```python
# Existing call sites — no change required:
from bing.parameters import standard
p = standard.expb_pow(satellite='PACE')          # variable_Gordon_G0 defaults False
# ... fit_one(p, idx) runs the 2-parameter Gordon model exactly as before.

# Enabling G0:
p = standard.expb_pow(satellite='PACE', variable_Gordon_G0=True)
chains, models, prep_dict, idx, extras = fit_l23.fit_one(p, idx)
# models[0].G0 is now a length-nwave array; the MCMC forward model uses it.
```

**Result.** rRMS vs Hydrolight (all 3320 L23 scenes), measured end-to-end through the package code path:

| λ (nm) | standard | variable no G₀ (smooth) | variable with G₀ |
|--------|---------:|------------------------:|-----------------:|
| 400 | 2.54 | 2.40 | **2.27** |
| 500 | 3.71 | 3.66 | **3.47** |
| 550 | 4.91 | 2.51 | **1.11** |
| 600 | 6.46 | 3.38 | **0.29** |
| 650 | 7.66 | 3.53 | **0.31** |
| 700 | 9.05 | 3.75 | **0.35** |

**Files touched.**
- [bing/rt/rrs.py](bing/rt/rrs.py) — `wave_dependent_gordon`, `calc_Rrs`, `calc_elastic_Rrs`.
- [bing/parameters/p_ntuple.py](bing/parameters/p_ntuple.py) — `variable_Gordon_G0` default.
- [bing/rt/defs.py](bing/rt/defs.py) — new rt_dict key.
- [bing/fitting/l23.py](bing/fitting/l23.py) — `prep_one_l23` G0 branch + `prep_dict['G0']`.
- [bing/evaluate.py](bing/evaluate.py) — `calc_Rrs_from_models`, `reconstruct_from_chains` forward `in_G0`.
- [bing/data/RT/gordon_coefficients_with_G0.csv](bing/data/RT/gordon_coefficients_with_G0.csv) — new.
- [bing/tests/test_l23_fitting.py](bing/tests/test_l23_fitting.py) — new `test_single_fit_variable_Gordon_with_G0`.
- [dev/Gordon/calc_gordon.py](dev/Gordon/calc_gordon.py) — `plot_rrms_vs_wavelength_3case`, `plot_residual_vs_bbp_3case`; wired into `run_full_assessment`.
- [dev/Gordon/figs/](dev/Gordon/figs/) — `rrms_vs_wavelength_3case.png`, `residual_vs_bbp_3case.png`.

### 2026-05-31 (Diagnose the 500 nm residual; G₀ sign change near 510 nm; rrs-vs-u figure with 500 nm and all three cases)

**The 500 nm rRMS only drops from 3.66 % → 3.47 % when G₀ is enabled (every other red wavelength drops ~10×). The user noted G₀ also changes sign here.**

**Diagnosis.**

1. **G₀(λ) zero-crossing is at 510 → 515 nm.** Reading [bing/data/RT/gordon_coefficients_with_G0.csv](bing/data/RT/gordon_coefficients_with_G0.csv) around the transition:

   | λ (nm) | G₀ | G₁ | G₂ |
   |--------|-------:|------:|------:|
   | 480 | −2.3×10⁻⁴ | 0.0999 | +0.051 |
   | 490 | −3.2×10⁻⁴ | 0.1040 | +0.014 |
   | 500 | −5.3×10⁻⁴ | 0.1163 | −0.121 |
   | 505 | **−6.2×10⁻⁴** | **0.1257** | **−0.263** |
   | 510 | −4.0×10⁻⁴ | 0.1239 | −0.339 |
   | 515 | +1.2×10⁻⁴ | 0.1043 | −0.209 |
   | 520 | +3.8×10⁻⁴ | 0.0919 | −0.082 |

   At 505 nm all three parameters take extreme values simultaneously — |G₀| at its maximum, G₁ spiked to 0.126 (its global max), G₂ dived to −0.34. All three coefficients are over-deforming to fight the same residual signal, then snap back once λ ≥ 515 nm where the offset can carry its share.

2. **The residual at 500 nm has a structural ~10 % swing with trophic state.** Binning the 3-parameter residual into 8 equal-population quantiles of bbnw(500) (415 scenes per bin):

   | bbnw(500) bin (m⁻¹) | mean (HL−Gordon)/HL [%] |
   |-----------------------|------------------------:|
   | 1.2 – 5.1 ×10⁻⁴ | +5.34 |
   | 5.1 – 7.0 ×10⁻⁴ | +2.76 |
   | 7.0 – 8.6 ×10⁻⁴ | +1.40 |
   | 8.6 – 10.2 ×10⁻⁴ | +0.44 |
   | 10.2 – 12.0 ×10⁻⁴ | −0.33 |
   | 12.0 – 14.7 ×10⁻⁴ | −0.82 |
   | 14.7 – 21.7 ×10⁻⁴ | −2.27 |
   | 21.7 – 136 ×10⁻⁴ | −5.58 |

   Same monotone pattern shows up against anw(500), aph(440), and bbnw/bb_w(500) — they are all surrogates for the same underlying axis (trophic state). The residual is essentially a one-axis function of *how much non-water there is*.

3. **Why a single G₀ can't fix this here.** A wavelength-only constant offset is the best single-axis correction; but at 500 nm the truth needs *opposite* corrections for oligotrophic (+) and eutrophic (−) scenes. The least-squares fit splits the difference and lands near zero, which is why we see G₀ pass through zero in this band. The 5 % residual swing is exactly what's left after subtracting that zero-mean signal.

**Plot change (requested explicitly).** `plot_rrs_vs_u` in [dev/Gordon/calc_gordon.py](dev/Gordon/calc_gordon.py) now accepts an optional `result_with_G0` and, when given, overlays all three cases per panel:

- C0 dashed — standard Gordon (`G₁=0.0949, G₂=0.0794`)
- C3 solid — variable, no G₀ (smooth 2-parameter fit)
- C2 solid — variable, with G₀ (3-parameter fit)

`run_full_assessment` now generates [dev/Gordon/figs/rrs_vs_u_3case.png](dev/Gordon/figs/rrs_vs_u_3case.png) with panels at 370 / 440 / **500** / 550 / 600 / 670 nm. The 500 nm panel makes the structural problem obvious: the gray data scatter has a vertical width that no curve through u alone can collapse — the truth is genuinely a function of (a, bb) at this wavelength, not just u.

**Files touched.**
- [dev/Gordon/calc_gordon.py](dev/Gordon/calc_gordon.py) — extended `plot_rrs_vs_u(..., result_with_G0=None)`; updated `run_full_assessment` to emit the new figure with a 500 nm panel.
- [dev/Gordon/figs/rrs_vs_u_3case.png](dev/Gordon/figs/rrs_vs_u_3case.png) — new.

**Recommendations for next stage.**

1. **Add an explicit second axis to the fit at 500 nm (and only where needed).** The simplest extension that matches what the residual binning shows is a bbp-linear term:
   `rrs = G₀(λ) + G₁(λ)·u + G₂(λ)·u² + G₃(λ)·bbp`.
   `G₃(λ)` would be near zero outside the 480–515 nm trouble band, large there. One DoF added, but only meaningful at a few wavelengths — try a smoothness-regularized fit on G₃(λ) so it stays at zero outside the band without manual gating. Quick check: this term should reproduce the −5 → +5 % bbp-linear residual we just measured.
2. **Or: stratify by trophic regime.** Fit two pairs `(G₀, G₁, G₂)(λ)` on oligotrophic vs eutrophic L23 splits (`aph(440)` threshold), and at runtime pick the pair from a one-shot Chl estimate. No new functional form; the package code path already supports per-fit coefficient swap.
3. **Validate the 500 nm spike in G₁, G₂ is real, not numerical.** The 5-nm-wide bump in G₁ at 505 nm (0.126 vs 0.10 elsewhere) is suspiciously sharp. Re-fit with a stronger Tikhonov smoothness penalty (`α_G1=1e7`, `α_G2=1e5`) and confirm the rRMS table doesn't move; if it does, leave it alone and adopt option (1) above instead.
4. **Document.** The "## Docs" section in this prompt already asks for the G₀ doc update. Add a paragraph on the 510-nm sign change and the residual bbp dependence; cite the 3-case figure.
5. **Operational note.** Because of the 500 nm gap, current `variable_Gordon_G0=True` is *not* uniformly better than `variable_Gordon_G0=False` — it is dramatically better at 550–700 nm and comparable in the blue / mid-blue. If a downstream retrieval consumer is dominated by green bands (e.g. Chl algorithms band-ratioing 490/555), confirm it isn't being hurt by the residual structure here before flipping the default.
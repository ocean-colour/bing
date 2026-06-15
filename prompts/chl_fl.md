 # Chl-a Fluorescence 

## Goals

Adds Chl Fl to the bing.rt module.

## Code

Here are guidelines for the code: 

- Use Python
- When possible use existing methods from the repository
- Add inline comments to explain the effort
- Use methods, not classes
- Place import statements at the top of the file.
- Include a description of inputs/outputs in the doc string of all methods

## Testing

If you need to run python, use the "ocean14" environment in conda.

## Docs

Examine the files in the docs/ directory and update the docs to reflect the new changes.  In particular:

- Make sure the docs/chlorophyll_fluorescence.rst file is up to date.
- Add a new section to the docs/index.rst file to include that file
- Update the docs/radiative_transfer.rst file, as needed
- Note the new dependency on the correct_atmosphere repository, which is located at https://github.com/ocean-colour/correct-atmosphere
- Update the docs/models.rst file, as needed
- Update the docs/parameters.rst file, as needed
- Add docs on how to instantiate and use the rt_dict_from_p() function

## Modifications

1. When examining outputs for the double_gaussian vs. single_gaussian, I note that the 2nd peak for the double gaussian is offset and appears too strong.  Can you:

- Investigate the issue.  Please generate a new module named double_gaussian in dev/ChlFl that makes a calculation and figures.  
- Examine the outputs
- Reconsider how best to normalize the double gaussian in the calc_R_fluorescence_integrated() method.
- Log your work and results below in the Logs section.  
- make recommendations for the next stage of development.  

2. Please update the calc_R_fluorescence_integrated() method to use the new double_gaussian module as per the recommendations in the Logs section.  Modify the tests too.

3. Please make updates to the docs to reflect the new changes.

## Tests

1. Update the tests in the bing/tests/test_chl_fl.py file to include tests for the new functionality.

2. Check that the Chl tests in test_l23_fitting.py are passing.

## Prompts

1.  Generate/update the docs as described in the Docs section above.
2. Re-read this doc.  Execute the first item in the Tests section above.
3. Re-read this doc.  Execute the 2nd item in the Tests section above.
4. Re-read this doc.  Execute the 1st item in the Modifications section above.
5. Re-read this doc.  Execute the 2nd item in the Modifications section above.
5. Re-read this doc.  Execute the 3rd item in the Modifications section above.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-06-02 (double_gaussian vs single_gaussian — root cause is κ_F, not h_C normalization)

**TL;DR.** The double Gaussian emission shape `h_C(λ)` is correctly
normalized — both `emission_line_single_gaussian` and
`emission_line_double_gaussian` integrate to 1.0 over 640–800 nm. The
secondary peak at 730 nm looks too strong because of a separate bug
*downstream* of `h_C` in [bing/rt/rrs.py:calc_Rrs_fluorescence](../bing/rt/rrs.py): the
upwelling attenuation κ_F is evaluated only at λ=685 nm and then re-used
for every emission wavelength. That approximation is fine for the
single-Gaussian case (which is essentially zero away from 685) but
breaks the double Gaussian because pure-water absorption at 730 nm is
~4× larger than at 685 nm.

**Investigation module.** [dev/ChlFl/double_gaussian.py](../dev/ChlFl/double_gaussian.py)
builds a flat-IOP test scene (constant a_nw, bb_nw, aph; pure-water IOPs from
`ocpy.water.absorption.a_water`) and computes Rrs_fl two ways:

- *Current code:* `bing.rt.rrs.calc_Rrs_fluorescence`, which pins
  `kappa_F_em_peak = kappa_F_em[ipeak]` at the bin closest to 685 nm
  ([rrs.py:455–459](../bing/rt/rrs.py)).
- *Reference:* a per-emission-wavelength κ_F implementation written in
  the same module (`calc_Rrs_fluorescence_per_lambda`), which evaluates
  the radiative-transfer denominator `K(λ') + κ_F(λ_em)` separately for
  each λ_em and uses the proper `λ' / λ_em` energy ratio.

**Quantitative result.**

| variant                    | Rrs[685]       | Rrs[730]       | 730/685 |
| -------------------------- | -------------- | -------------- | ------- |
| current,   single Gaussian | 8.07e-04 sr⁻¹  | 9.84e-08 sr⁻¹  | 0.000   |
| current,   double Gaussian | 6.16e-04 sr⁻¹  | 1.01e-04 sr⁻¹  | 0.164   |
| per-λ κ_F, single Gaussian | 8.07e-04 sr⁻¹  | 2.64e-08 sr⁻¹  | 0.000   |
| per-λ κ_F, double Gaussian | 6.16e-04 sr⁻¹  | 2.71e-05 sr⁻¹  | 0.044   |

`a_w(685) = 0.486 m⁻¹`, `a_w(730) = 1.962 m⁻¹` (ratio 4.04). The current
code overestimates the secondary peak by ~3.7×; the per-λ κ_F result
matches the single-Gaussian primary peak exactly (both implementations
collapse to the same answer at 685 nm) and produces a much weaker, more
realistic shoulder at 730 nm.

**Figures.** Saved in `dev/ChlFl/`:

- `double_gaussian_emission_shape.png` — shows both `h_C` curves and the
  separate primary/secondary contributions to the double Gaussian.
  Confirms ∫h_C dλ = 1.0 in both cases.
- `double_gaussian_Rrs_fl.png` — top panel: Rrs_fl for current vs
  per-λ κ_F, single vs double; bottom panel: `a_w(λ_em)` to make the
  cause visible.

**Recommendations.**

1. **Fix `bing.rt.rrs.calc_Rrs_fluorescence` to use κ_F per emission
   wavelength.** This is a correctness fix, not a normalization change.
   Replace the scalar `kappa_F_em_peak` with the full vector `kappa_F_em`
   and broadcast: the integrand becomes 2-D `(n_em, n_ex)` and is
   integrated along the excitation axis. Cost: one extra outer product
   per call — negligible. The MCMC-chains path needs the 3-D analogue
   `(n_samples, n_em, n_ex)`.

2. **Use λ_em (not 685 nm) in the energy ratio** `λ' / λ_em`. Currently
   `lambda_ratio = wavelength_ex / chl_fl.LAMBDA_FL_PRIMARY` is hard-coded
   to 685; the proper factor depends on λ_em. The error from this is
   small (~6% across 685–730) but it's wrong for the same reason as the
   κ_F bug — both are remnants of a "delta-function-at-685" simplification.

3. **Keep the `h_C` normalization as-is.** The double Gaussian already
   integrates to 1.0, with 75 % of the area under the primary and 25 %
   under the wider secondary. The peak ratio is set by the differing σ
   values and that's physically correct.

4. **Document the κ_F fix in `docs/chlorophyll_fluorescence.rst`** —
   spell out the per-λ_em RT formula so future readers don't reintroduce
   the shortcut.

5. **Consider also fixing `calc_R_fluorescence_integrated` in
   `bing/rt/chl_fl.py`.** It already loops per emission wavelength with
   the correct `kappa_F_em[i]` (good), but its trailing
   `R_F = R_F / Ed_total * np.trapz(Ed, wavelength_ex)` line is an
   algebraic no-op and should be removed for clarity.

**Next-stage development.**

- Apply recommendations 1–2 to `bing.rt.rrs.calc_Rrs_fluorescence` and
  re-run [bing/tests/test_chl_fl.py](../bing/tests/test_chl_fl.py); the
  existing `test_calc_Rrs_fluorescence_double_vs_single_gaussian` will
  need its 730-nm threshold adjusted (it'll still pass — the double
  Gaussian still beats the single Gaussian there — but the ratio will
  shrink).
- Add a regression test that pins the 730/685 ratio against the
  per-λ κ_F reference for fixed inputs, so this doesn't silently regress.
- Once fixed, re-run the L23 fluorescence-enabled fits to confirm the
  posterior on `phi_C` doesn't shift dramatically (we expect a modest
  upward shift because each unit of fluorescence now produces less
  730-nm signal).

### 2026-06-02 (apply per-λ κ_F fix to calc_Rrs_fluorescence + update tests)

Applied recommendations 1, 2, and 5 from the previous log entry.

**Changes to [bing/rt/rrs.py](../bing/rt/rrs.py).**
`calc_Rrs_fluorescence` now evaluates κ_F at every emission wavelength
(not just 685 nm) and uses the proper `λ' / λ_em` energy ratio per
(λ_em, λ') pair. The implementation builds the denominator
`K(λ') + κ_F(λ_em)` as a `(n_em, n_ex)` tensor for single spectra and as
`(n_samples, n_em, n_ex)` for chains, and integrates over the excitation
axis. Robustness fixes for shape inconsistencies in the existing callers:

- `evaluate.py:reconstruct_from_chains` tiles `wavelength_ex` and
  `Ed_ex` to `(n_samples, n_ex)` via `np.outer`; the new code squeezes
  them back to 1-D since these are scene properties shared across
  samples.
- `evaluate.py:calc_Rrs_from_models` (the log_prob path) passes
  `aph_ex` as 1-D while `a_ex`/`bb_ex` are 2-D. The new code broadcasts
  `aph_ex` (and `a_em`, `bb_em`) up to `(n_samples, *)` when needed.

**Changes to [bing/rt/chl_fl.py](../bing/rt/chl_fl.py).**
Dropped the trailing `R_F = R_F / Ed_total * np.trapz(Ed, wavelength_ex)`
in `calc_R_fluorescence_integrated` — an algebraic no-op. This function
already integrated correctly per emission wavelength.

**Tests ([bing/tests/test_chl_fl.py](../bing/tests/test_chl_fl.py)).**

- Tightened `test_calc_Rrs_fluorescence_double_vs_single_gaussian` to
  also check that the 685 nm primary peak shrinks to ~75 % of the
  single-Gaussian peak (which is what the weight-0.75 emission area
  enforces under uniform κ_F).
- New `test_calc_Rrs_fluorescence_per_lambda_kappa_F` — uses a stepped
  `a_em(λ_em)` (0.5 at 685 → 2.0 at 730) and asserts the 730/685 ratio
  drops below 0.10 (pre-fix would give ~0.16, locking in the bug).
- New `test_calc_Rrs_fluorescence_matches_reference` — pins the
  implementation against an explicit per-λ_em loop reference at
  `rtol=1e-12`. Any future "freeze κ_F at 685" shortcut will fail here.

**Verification.**

- `dev/ChlFl/double_gaussian.py` re-run: "current code" and
  "per-λ κ_F reference" now agree exactly (730/685 = 0.044 for both,
  down from the 0.164 the old code produced).
- `pytest bing/tests/test_chl_fl.py bing/tests/test_l23_fitting.py`:
  **59 passed, 2 skipped** (the two skips are pre-existing). Notably
  `test_Chl_fitting_MCMC` and `test_Chl_fitting_LM` still pass — the
  L23 fluorescence-enabled fits run cleanly through the new code path.

**Follow-ups not addressed in this commit.**

- Documentation updates (rec. #4) — `docs/chlorophyll_fluorescence.rst`
  still describes the old single-κ_F formula. Worth doing as a separate
  pass once the fix has settled.
- Posterior comparison on real L23 fits — the L23 tests verify the
  pipeline runs; the user may want a longer side-by-side comparison of
  retrieved `phi_C` before/after to quantify the impact on prior science
  results.

### 2026-06-02 (docs: per-λ κ_F formulation in chlorophyll_fluorescence.rst + radiative_transfer.rst)

Closes recommendation #4 from the first Logs entry. The rest of the doc
scaffolding (cross-references, `correct_atmosphere` install notes,
`rt_dict_from_p` reference) was already in place from earlier prompts;
all that remained was to make the per-λ κ_F behavior explicit.

**[docs/chlorophyll_fluorescence.rst](../docs/chlorophyll_fluorescence.rst).**

- Added a `.. note::` block under the ``calc_Rrs_fluorescence`` function
  description spelling out that κ^F(λ) and λ'/λ are evaluated at every
  emission wavelength, with a pointer to ``dev/ChlFl/double_gaussian.py``.
- Rewrote the *Reflectance Formulation* section. The earlier display
  equation showed the single-excitation form; it now shows the full
  integrated formula (``h_C(λ) ∫ … dλ'``) with the proper per-λ_em
  factors, and explicitly identifies the (λ', λ) dependencies of each
  symbol.

**[docs/radiative_transfer.rst](../docs/radiative_transfer.rst).**

- Added a matching short `.. note::` next to the ``calc_Rrs_fluorescence``
  function signature pointing readers to the per-λ_em κ_F behavior and
  why it matters for the double-Gaussian secondary peak.

**Not changed:**

- ``docs/index.rst`` — chlorophyll_fluorescence is already listed in the
  toctree (line 52).
- ``docs/models.rst`` / ``docs/parameters.rst`` — the ``include_Chl_fl``,
  ``phi_C``, ``double_gaussian`` flags are already documented from prior
  prompts; the κ_F fix is below their level of abstraction so no edits
  needed.
- ``docs/installation.rst`` — already documents the
  ``correct_atmosphere`` dependency.

**Caveat.** Could not run ``sphinx-build`` to verify rendering — neither
``sphinx`` nor ``docutils`` is installed in any of the available conda
envs (``base``, ``ocean14``, ``astro14``, ``pypeit14``). The directives
I added (``.. note::``, ``.. math::``) and inline roles (``:math:```,
``:doc:```) match patterns already in the same files, so the syntax
should be fine, but the user should run the read-the-docs build to
confirm before tagging a release.
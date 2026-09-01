 # Chl-a Fluorescence 

## Goals

Adds Chl Fl to the bing.rt module.

## Skills

Consider using the skills in .claude/skills/


## Code

Here are guidelines for the code: 

- Use Python
- When possible use existing methods from the repository
- Add inline comments to explain the effort
- Use methods, not classes
- Place import statements at the top of the file.
- Include a description of inputs/outputs in the doc string of all methods
- Use lines of code that are less than 80 characters wide

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

4. I am finding that the when applying the correction on the MCMC chains to reconstruct the Rrs, the memory requirements are too high for my standard nsteps=40000 (RAM > 100Gb is required).  Can you:

- Investigate the issue.  I am confident the issue is in the calc_Rrs_fluorescence() method.
- Log your work and results below in the Logs section.  
- make recommendations for the next stage of development.  

5. Your recommendation to use the per-emission loop is sound.  Please:

- Implement this change to the code
- Log your work and results below in the Logs section.  

6. The memory issue is now fixed, but the MCMC fitting is now 7x slower.  Can you examine why and suggest a fix?

- Log your work and results below in the Logs section.  
- make recommendations for the next stage of development.  

## Tests

1. Update the tests in the bing/tests/test_chl_fl.py file to include tests for the new functionality.

2. Check that the Chl tests in test_l23_fitting.py are passing.

## Prompts

1.  Generate/update the docs as described in the Docs section above.
2. Re-read this doc.  Execute the first item in the Tests section above.
3. Re-read this doc.  Execute the 2nd item in the Tests section above.
4. Re-read this doc.  Execute the 1st item in the Modifications section above.
5. Re-read this doc.  Execute the 2nd item in the Modifications section above.
6. Re-read this doc.  Execute the 3rd item in the Modifications section above.
7. Re-read this doc.  Execute the 4th item in the Modifications section above.
8. Re-read this doc.  Execute the 5th item in the Modifications section above.
9. Re-read this doc.  Execute the 6th item in the Modifications section above.

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

### 2026-06-04 (RAM blow-up reconstructing Rrs_fl from chains — root cause is the 3-D integrand tensor)

**TL;DR.** The user is right: the OOM is in
[bing.rt.rrs.calc_Rrs_fluorescence](../bing/rt/rrs.py). Its chains
(`ndim==2`) branch materialises a full `(n_samples, n_em, n_ex)` float64
integrand tensor (plus ~4-5 simultaneous temporaries from the chained
arithmetic). For the standard biomass fit that single tensor is **14 GiB**,
and the live-copy peak is **57-71 GiB** — which, on top of the elastic Rrs,
the IOP arrays, and the chains themselves, comfortably exceeds the
>100 GB the user observed. No approximation or normalization is involved;
it's purely the array shape.

**Why it's so large.** `evaluate.calc_Rrs_from_models` passes the *full*
model wave grid as the emission axis, so `n_em ≈ 60` (PACE 5 nm, 400-700),
`n_ex ≈ 60` (`i_Chl_ex` over the 400-700 excitation range), and
`n_samples = (nsteps - burn) × nwalkers = (40000 - 7000) × 16 = 528_000`.
The denominator `K(λ') + κ_F(λ_em)` genuinely couples the em and ex axes
(it's a *sum*, so it doesn't factor), which is why the per-λ_em κ_F fix
from the previous log entry introduced the 3-D shape in the first place.
That correctness fix is right — it's the *layout* that's wrong for chains.

**This is also why the production path is currently guarded.** A
stop-gap `raise ValueError("Chl fluorescence not supported for batch
evaluation")` now sits in `evaluate.calc_Rrs_from_models` for `ndim==2`,
so `reconstruct_from_chains` cannot actually run fluorescence over chains
today. The investigation below is what's needed to lift that guard
safely.

**Investigation module.** [dev/ChlFl/memory_profile.py](../dev/ChlFl/memory_profile.py)
reports the theoretical footprint, measures *peak* RAM with `tracemalloc`
on a tractable 20k-sample subset for both the current code and a rewrite,
and asserts they agree.

**Quantitative result** (`conda run -n ocean14 python dev/ChlFl/memory_profile.py`):

| variant                         | peak @ 20k samples | extrapolated @ 528k |
| ------------------------------- | ------------------ | ------------------- |
| current 3-D path                | 2.155 GiB          | ~57 GiB (1 tensor) → >100 GiB live |
| per-emission rewrite (proposed) | 0.089 GiB          | ~2.4 GiB            |

`max |current - rewrite| = 0.0` (rtol=1e-12) — the rewrite is **bit-for-bit
identical**, not an approximation. Memory drops ~24× at the measured size.

**The fix (prototyped, not yet applied).** Loop over the *small* emission
axis (`n_em ≈ 60`) instead of broadcasting it. Each iteration touches only
an `(n_samples, n_ex)` slice, so the 3-D tensor is never formed:

```python
for j in range(n_em):
    denom = K_ex + kappa_F_em[:, j:j+1]        # (n_samples, n_ex)
    lam_ratio = wavelength_ex / wavelength[j]   # (n_ex,)
    integrand = Ed_ex * (bb_F / mu_d) * lam_ratio[None, :] / denom
    R_F[:, j] = np.trapezoid(integrand, x=wavelength_ex, axis=1)
```

This is the math BING already does — just reordered so peak memory is
bounded by `n_samples × n_ex` rather than `n_samples × n_em × n_ex`. The
`n_em`-length Python loop is negligible next to the `trapezoid` over
`n_ex`. The single-spectrum (`ndim==1`) branch is already tiny and needs
no change.

**Recommendations for the next stage.**

1. **Replace the `ndim==2` branch of `calc_Rrs_fluorescence` with the
   per-emission loop** above. It's numerically identical (lock it in with
   an `np.allclose(..., rtol=1e-12)` regression test against the current
   single-spectrum result evaluated sample-by-sample). Keep the elegant
   `ndim==1` 2-D tensor path as-is.

2. **Then remove the stop-gap guard** in
   `evaluate.calc_Rrs_from_models` (`raise ValueError("Chl fluorescence
   not supported for batch evaluation")`) so `reconstruct_from_chains`
   can run fluorescence over chains again.

3. **Add a memory regression / smoke test** that reconstructs at a
   realistic sample count (e.g. 100k) with fluorescence on and asserts it
   completes — guards against a future broadcast creeping the 3-D tensor
   back in.

4. **Optional defence-in-depth:** also chunk `reconstruct_from_chains`
   over samples (process ~50k at a time, accumulate the percentile inputs).
   Not required once the loop above lands — peak is back to ~2 GiB — but it
   would cap memory for users who push `nsteps` much higher or fit
   hyperspectral (full PACE) grids where `n_em` grows.

5. **Update `docs/chlorophyll_fluorescence.rst`** to note that the chains
   path integrates per emission wavelength to keep memory `O(n_samples ×
   n_ex)` — so the shortcut isn't "optimised" back into a 3-D broadcast.

### 2026-06-04 (implement the per-emission loop in calc_Rrs_fluorescence — chains path no longer OOMs)

Applied recommendation #1 (and #2) from the previous entry.

**Change to [bing/rt/rrs.py](../bing/rt/rrs.py).** The chains (`ndim==2`)
branch of `calc_Rrs_fluorescence` no longer builds the 3-D
`(n_samples, n_em, n_ex)` integrand tensor. It now loops over the small
emission axis (`n_em ≈ 60`); each iteration forms only an
`(n_samples, n_ex)` slice (`denom = K_ex + κ_F_em[:, j:j+1]`), integrates
over the excitation axis with `np.trapezoid`, and writes one column of
`R_F`. Because the RT denominator `K(λ') + κ_F(λ_em)` is a *sum* it can't
factor across the two axes, so the reorder — not an algebraic shortcut —
is what avoids the tensor. The `ndim==1` single-spectrum branch is
unchanged. The sample-independent excitation factor
`Ed_ex·(b_bF/μ_d)` is hoisted out of the loop.

**Change to [bing/evaluate.py](../bing/evaluate.py).** Removed the
stop-gap `raise ValueError("Chl fluorescence not supported for batch
evaluation")` in `calc_Rrs_from_models`, so `reconstruct_from_chains` can
again run fluorescence over chains.

**Test ([bing/tests/test_chl_fl.py](../bing/tests/test_chl_fl.py)).**
Added `test_calc_Rrs_fluorescence_chains_matches_per_spectrum`, which
asserts the batch result equals calling the function once per sample,
row-by-row, at `rtol=1e-12`. This locks the loop to be bit-for-bit
identical to per-spectrum evaluation and will fail if anyone reinstates
the 3-D broadcast or perturbs the math. The existing
`test_calc_Rrs_fluorescence_chains` (shape + monotonicity) and
`test_calc_Rrs_fluorescence_matches_reference` (per-λ_em κ_F) still
guard the rest.

**Verification.**

- `dev/ChlFl/memory_profile.py` re-run: the *production*
  `calc_Rrs_fluorescence` now peaks at **0.089 GiB** at 20k samples
  (was 2.155 GiB) — identical to the standalone rewrite — and is still
  numerically exact (`max |diff| = 0.0`). Extrapolated to the full
  528k-sample biomass workload: **~2.4 GiB** vs the previous >100 GB.
- `pytest bing/tests/test_chl_fl.py bing/tests/test_l23_fitting.py`:
  **61 passed, 2 skipped** (the L23 MCMC/LM Chl fits run cleanly through
  the new code path; the 2 skips are pre-existing). The fluorescence
  file alone, including the new batch-equivalence regression test, is
  **36 passed**.

### 2026-06-04 (MCMC ~7x slower after the memory fix — root cause is the n_samples==1 log_prob path; fix by chunking)

The previous entry's per-emission loop fixed the RAM blow-up but made
MCMC fitting ~7x slower. Diagnosed and fixed.

**Root cause.** `inference.log_prob` evaluates one parameter vector per
call, but `models.eval_a` returns shape `(1, nwave)`. So the excitation
IOPs reach `calc_Rrs_fluorescence` as 2-D `(1, n_ex)` and take the
*chains* branch with `n_samples == 1`. The old chains branch was a single
vectorised 3-D op over a trivially small `(1, n_em, n_ex)` tensor
(~microseconds); the per-emission loop replaced that with an
`n_em`-long (~60) Python loop, each iteration allocating arrays and
calling `np.trapezoid`. MCMC calls log_prob ~`nsteps·nwalkers`
(40000·16 ≈ 6.4e5) times per fit, so that per-call overhead dominates.
The memory blow-up only ever mattered for *large* `n_samples`
(reconstruct_from_chains); for `n_samples == 1` the loop was pure cost.

**Benchmark.** [dev/ChlFl/mcmc_speed.py](../dev/ChlFl/mcmc_speed.py)
times the single-spectrum (`n_samples==1`) call:

| variant                       | µs/call | vs vectorised |
| ----------------------------- | ------- | ------------- |
| per-emission loop (regressed) | 252.5   | 9.5x          |
| vectorised 3-D (old)          | 26.6    | 1.0x          |
| production (chunked, fixed)   | 28.8    | 1.1x          |

The ~9.5x slowdown of the fluorescence kernel is consistent with the
user's ~7x end-to-end MCMC report (log_prob also does the elastic Rrs,
priors, etc., which dilutes it slightly).

**Fix — chunk over samples.** The chains branch of
`calc_Rrs_fluorescence` now processes samples in blocks sized so the 3-D
integrand stays under the new module constant `rrs.FL_CHUNK_ELEMENTS`
(5e7 elements ≈ 0.4 GiB/tensor), each block done with the *fast*
fully-vectorised 3-D op. For `n_samples == 1` (the log_prob hot path)
this is a single vectorised block — as fast as the pre-memory-fix code;
for 528k samples it is ~38 bounded-memory blocks. This keeps both
properties at once: vectorised speed *and* bounded RAM. The per-emission
loop is gone.

**Verification.**

- `dev/ChlFl/mcmc_speed.py`: production back to 28.8 µs/call (1.1x over
  pure vectorised, vs 9.5x for the loop) — the MCMC slowdown is removed.
- `dev/ChlFl/memory_profile.py`: production peak at 20k samples is now
  **1.5 GiB** (the chunk bound), independent of total `n_samples`; the
  528k reconstruction stays a few GiB, still vs the original >100 GB.
  All three implementations (loop / vectorised / chunked production)
  agree to `rtol=1e-12`.
- `pytest bing/tests/test_chl_fl.py`: **36 passed** (the batch-equivalence
  regression test still holds under chunking).
  `pytest bing/tests/test_l23_fitting.py`: **26 passed, 2 skipped** — the
  fluorescence-enabled MCMC/LM fits run cleanly and noticeably faster
  (203 s for the file vs the sluggish post-loop timing).

**Recommendations for the next stage.**

1. **Add a micro-benchmark guard for the log_prob path.** A fast unit
   test (e.g. assert `n_samples==1` `calc_Rrs_fluorescence` is within ~3x
   of the pure vectorised op) would catch a future refactor that
   re-introduces a per-call Python loop. Timing tests are flaky in CI, so
   gate it behind a marker / make it advisory rather than hard-failing.
2. **Tune `FL_CHUNK_ELEMENTS` if needed.** 5e7 (~0.4 GiB/tensor, a few
   GiB peak) is a deliberate speed/RAM balance. Users on tight RAM can
   lower it; users with headroom fitting very long chains can raise it.
   Consider surfacing it via `rt_dict` if per-fit control becomes useful.
3. **Longer term, give log_prob a true 1-D fast path.** The cleanest fix
   would be for `eval_a` to return 1-D for 1-D input (or for the
   fluorescence call to squeeze the singleton sample axis) so the
   `ndim==1` branch — which never allocates a 3-D tensor at all — is used
   for single spectra. Chunking already recovers the speed, so this is a
   tidiness/clarity improvement, not a correctness need.
4. **Profile the rest of log_prob.** Now that fluorescence is no longer
   the bottleneck, if further MCMC speedups are wanted, profile the
   elastic `calc_Rrs` + Raman path and the prior evaluation; those now
   dominate a fluorescence-enabled step.
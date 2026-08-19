# Inelastic RT fixes

Fixes to BING's inelastic radiative transfer, driven by the validation
against the Loisel et al. (2023, "L23") HydroLight database performed in the
retrieve-or-bust repository (`context/RT/rt_inelastic_bing_summary.md` there;
Q&A and decisions in `claude_prompts/RT/rt_inelastic_prompts.md`).

L23 provides paired HydroLight runs over 3320 IOP scenes — X1 (elastic
only), X2 (+ Raman), X4 (+ Raman + Chl fluorescence, phi_C = 0.02) — at
solar zeniths 0/30/60 deg, using HydroLight defaults that match BING's.
The scenario differences isolate each inelastic process exactly.

Decisions (JXP, 2026-08-18): fix the normalization and Ed plumbing in BING;
leave the two-flow formulation limits to retrieve-or-bust. Clean break (no
legacy flags). True-Ed is the default when available; warn on flat-Ed
fallback. Add L23-anchored regression tests.

## Logs

### 2026-08-18 (1/pi fluorescence normalization; true-Ed Raman ratio; per-lambda_em Ed; L23 regression tests)

**The two errors.**

1. `rt.rrs.calc_Rrs_fluorescence` computed the Sathyendranath & Platt
   (1998)-style two-flow *irradiance* reflectance R_F = Eu/Ed and pushed it
   directly through the A·rrs/(1−B·rrs) above-surface conversion — i.e.
   treated an irradiance reflectance as a radiance-based rrs. For isotropic
   emission Lu(0−) = Eu(0−)/π, so the term was ~π ≈ 3.1× too large.
   Validated against L23 X4−X2: median model/truth at 685 nm was
   **3.18/3.00/2.73** (zenith 0/30/60°); with the 1/π conversion it becomes
   **1.01/0.96/0.87**. The residual zenith trend is the separate fixed-μ
   two-flow limitation, deliberately not addressed in BING.
2. The production Raman path (`evaluate.calc_Rrs_from_models` →
   `rrs.calc_Rrs` → `calc_raman_correction_factor`) always used the flat-Ed
   default Ed(λ′)/Ed(λ) = 1. Against L23 X2/X1 this distorts the spectral
   shape of the correction: ~+60 % increment error at 490 nm (zenith
   30–60°), −15 % and worse in the red. Supplying the true solar-spectrum
   ratio removes most of the shape error at zeniths 30–60°.

**Changes.**

- `bing/rt/rrs.py::calc_Rrs_fluorescence` — inserts `rrs_F = R_F / np.pi`
  before the A/(1−B) conversion; docstring carries the derivation and the
  L23 validation numbers. (The production *multiplicative* Raman path never
  had this problem: it uses only the ratio (R_E + R_Raman)/R_E, which
  cancels the normalization.)
- `bing/rt/raman.py::calc_Rrs_with_raman` (additive path, unused in
  production) — same 1/π conversion for consistency (Raman phase function
  is quasi-isotropic).
- `bing/rt/rrs.py::calc_Rrs` — new `Ed_ratio` kwarg, forwarded to
  `calc_raman_correction_factor` (None → flat).
- `bing/models/anw.py` — new `aNWModel.set_raman_Ed(wave_Ed, Ed)`:
  interpolates a user-supplied Ed spectrum onto the model grid and the
  Raman excitation grid and stores `Ed_ratio_raman`. Note wave_ex extends
  ~50 nm blueward of the model grid, so the Ed grid must cover it.
- `bing/evaluate.py::calc_Rrs_from_models` — uses `a_model.Ed_ratio_raman`
  when set; emits a `RuntimeWarning` and falls back to flat Ed when not.
- `bing/models/anw.py::init_Chl_fluorescence` — `Ed_em` now defaults to the
  full Ed vector on the model grid (exact per-λ_em normalization of the
  fluorescence term); legacy scalar still accepted.
- `bing/fitting/l23.py` — sets `set_raman_Ed` from `correct_atmosphere`
  when `include_Raman`, passes the ratio to the synthetic-observation
  `calc_Rrs`, and drops the scalar `Ed_em` (vector default), so mock data
  and MCMC forward model stay consistent.
- Docs: `docs/chlorophyll_fluorescence.rst` (formula rewritten with the
  Eu→Lu step and a validation note), `docs/radiative_transfer.rst`
  (set_raman_Ed note), `docs/changelog.rst` (breaking-change entry).

**Backward compatibility (clean break, per JXP).** Any fit run with
`include_Chl_fl` before this change overestimated Rrs_fl by ~3×; its
effective quantum yield was ~π× smaller than the nominal `phi_C`.
`include_Raman` results change only via the new Ed ratio (and only when
set); pre-fix flat-Ed behavior remains available by not calling
`set_raman_Ed`, though it now warns.

**Tests.**

- New `bing/tests/test_l23_inelastic.py` + committed 69 kB fixture
  `bing/tests/files/l23_inelastic_fixture.npz` (40 scenes, zenith 30°;
  generator `gen_l23_inelastic_fixture.py`). Pins: median fluorescence
  model/truth at 685 nm within ±15 %; true-Ed Raman median increment error
  within ±15 % over 550–700 nm; true-Ed strictly better than flat-Ed at
  490 nm; emission peak at the 685 nm bin.
- `test_chl_fl.py::test_calc_Rrs_fluorescence_matches_reference` reference
  updated with the 1/π step.
- Results: `test_l23_inelastic.py` + `test_chl_fl.py` + `test_raman.py`:
  **53 passed** (the one RuntimeWarning is the intended flat-Ed fallback
  warning exercised by `test_raman_in_models`). `test_evaluate.py`:
  **10 passed**. `test_l23_fitting.py`: **26 passed, 2 skipped** (matches
  the pre-change tally; the Raman/Chl MCMC and LM fits run through the
  new code paths). `test_inference.py` + `test_chisq_fit.py`:
  **13 passed**. Total **102 passed, 2 skipped**.
- Post-fix validation of the *production* functions against the full L23
  database (3320 scenes/zenith): fluorescence median model/truth at
  685 nm = **1.00/0.95/0.86**; true-Ed Raman median increment error over
  550–700 nm = **+1.2 %/−4.3 %** at zenith 30°/60° (−39 % at 0°, the
  known two-flow high-sun limitation, out of scope here).
- Headless note: `test_l23_fitting.py::test_raman_fitting_LM` (and the
  Chl LM test) call `plotting.show_fits` → `plt.show()`, which blocks
  forever in a Tk mainloop when a DISPLAY is reachable but nobody closes
  the window. Run the suite with `MPLBACKEND=Agg` in headless/automated
  contexts. (Pre-existing behavior, discovered while validating; not
  changed here.)

**What was *not* changed (by design).** The fixed-μ two-flow stream
geometry (high-sun red-band Raman underestimate of ~25–45 %; residual
±13 % zenith trend and trophic-state growth of the fluorescence amplitude)
— these are formulation-level and are being addressed in the
retrieve-or-bust forward-model redesign, not in BING.

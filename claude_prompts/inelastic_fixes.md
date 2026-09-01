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

## PR

1. There is an open PR on GitHub.  Please see the comments and address them.  Also, attempt to reconcile the CI failures.  Use Fable if you can. Log your work.  If you have any questions, ask me in the Q&A section below.

## Q&A

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
  Raman/Chl MCMC plot tests) called `plotting.show_fits(..., show=True)`
  → `plt.show()`, which blocks forever in a Tk mainloop when a DISPLAY
  is reachable but nobody closes the window. (Pre-existing behavior,
  discovered while validating.)

### 2026-08-19 (Fix the GUI block in the test suite)

Two-layer fix for the `plt.show()` hang above:

- `bing/tests/conftest.py` now forces `matplotlib.use('Agg', force=True)`
  at import time, before any test module imports pyplot — under Agg,
  `plt.show()` is a no-op, so the whole suite is hang-proof regardless of
  DISPLAY, and future plotting tests are covered automatically.
- The three `show_fits(..., show=True)` calls in `test_l23_fitting.py`
  (`test_raman_fitting_LM`, `test_raman_fitting_MCMC`,
  `test_Chl_fitting_MCMC`) now pass `show=False` — a unit test must not
  request a GUI window; the figure-construction code path is still fully
  exercised. (The `plt.show()` calls in `test_raman.py` sit inside
  `if False:` debug blocks and are inert.)

Verification, with **no** `MPLBACKEND` override in the environment:
`test_raman_fitting_LM` + all of `test_plotting.py`: **20 passed in
3.9 s** (previously the LM test alone hung indefinitely);
`test_raman_fitting_MCMC` + `test_Chl_fitting_MCMC`: **2 passed in
62 s**.

**What was *not* changed (by design).** The fixed-μ two-flow stream
geometry (high-sun red-band Raman underestimate of ~25–45 %; residual
±13 % zenith trend and trophic-state growth of the fluorescence amplitude)
— these are formulation-level and are being addressed in the
retrieve-or-bust forward-model redesign, not in BING.

### 2026-09-01 (CI reconciliation)

**PR comments: nothing actionable.** PR ocean-colour/bing#26
(`inelastic-fixes` → `develop`, head `850000b`) has exactly one issue
comment — JXP's own `@cursor review` trigger — and one review, from
`cursor[bot]`: "Bugbot reviewed your changes and found no new issues!"
Zero inline review comments (`review_comments: 0` per the API). So
"address the comments" reduces to the CI failures.

**CI status.** `Tests (Python 3.11/3.12/3.13)` all **failure** on both
check-suite runs of this head commit (run ids 32378423207 and
33250594483); `Docs build` and `Cursor Bugbot` pass. The jobs API shows
every failing job died at the **Run tests** step (install, ocpy clone,
and import smoke-tests all succeeded), but the actual pytest output is
behind GitHub's auth wall (`gh` not logged in here; raw log download is
403 even for public repos), so the failure was reproduced locally.

**Reproduction.** Built a CI-matching environment: fresh Python 3.13.13
venv (`/Users/xavier/miniforge3/bin/python3.13`), the workflow's exact
curated pip list (resolved to numpy 2.5.2, scipy 1.18.1,
xarray 2026.7.0, emcee 3.1.6), fresh `ocpy` clone installed
`--no-deps -e`, `bing` installed `--no-deps -e`, then
`MPLBACKEND=Agg python -m pytest bing/tests -v -ra` with `$OS_COLOR`
unset. Against the **working tree** this run *passes* (64 passed,
80 skipped) — the failure only appears against what git actually has.
Re-running the identical command against a clean `git archive HEAD`
export (i.e., exactly what CI checks out) reproduces CI:
**2 failed, 62 passed, 80 skipped** —
`test_l23_inelastic.py::test_raman_correction_matches_l23` and
`::test_fluorescence_matches_l23`, both with

    FileNotFoundError: [Errno 2] No such file or directory:
    '.../bing/tests/files/l23_inelastic_fixture.npz'

**Root cause.** `.gitignore` line 16 is a blanket `*.npz`, so the
fixture the 2026-08-18 entry describes as "committed 69 kB fixture" was
never actually tracked — `git status` looked clean because ignored
files are invisible to it, and every local environment (which has the
file on disk) passed. CI's checkout simply doesn't contain the file,
and `FileNotFoundError` on a non-Hydrolight path is (correctly) not
converted to a skip by `conftest.py`, hence hard failures on all three
Python versions. Not a code bug, not an environment/version issue —
the PR's code and tests are fine.

**Changes.**

- `.gitignore` — added `!bing/tests/files/l23_inelastic_fixture.npz`
  (with a comment) under the `*.npz` rule, so the fixture can be
  tracked. It now shows as untracked (`??`) in `git status`.
- `bing/tests/files/gen_l23_inelastic_fixture.py` — docstring corrected
  (~150 kB → ~69 kB, the file is 70,674 bytes) and now notes the
  required .gitignore exception.

**Action required (JXP):** `git add bing/tests/files/l23_inelastic_fixture.npz`
(plus the `.gitignore` / generator-docstring edits), commit, and push —
CI cannot go green until the fixture is actually in the tree. No
`git add -f` needed now that the exception is in `.gitignore`.

**Verification.**

- CI-equivalent env, clean `git archive` tree **with the fixture copied
  in** (simulating the post-commit checkout):
  **64 passed, 80 skipped, 0 failed** in 4.8 s.
- `ocean14`, full suite (`pytest bing/tests/ -q`, `$OS_COLOR` set):
  **180 passed, 2 skipped** in 131 s — matches the pre-change tally, no
  regressions.

**Side observation.** `bing/tests/files/m3_fixed_bp_pin.npz` also sits
ignored-but-untracked in that directory, but nothing in the repo
references it (grep over all `*.py` finds no consumer), so it was left
alone — likely a leftover from other in-flight work.

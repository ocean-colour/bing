# RT using Retreive or Bust (RoB)

## Goals

Interface BING with the radiative transfer model from the retrieve-or-bust repository.

## Prompts

### Design

1. Examine the code in the retrieve-or-bust repository and the BING code.  We wish to interface BING with the radiative transfer model from the retrieve-or-bust repository.  Please ask me a set of questions in Q&A/Design before writing the document.  Use Fable if you can.  Log your work.

2. Read my answers to the Q&A/Design section below. Ask me more questions if needed. Use Fable if you can.  Log your work.

3. Read my answers to the Q&A/Design section below. Ask me more questions if needed. Use Fable if you can.  Log your work.

4. Read my answers to the Q&A/Design section below. Ask me more questions if needed. Use Fable if you can.  Log your work.

5. Based on our discussion, please write a design document.  Name it `docs/design/rob_rt_design.md`.  Use Fable if you can.  Log your work.

### Coding Plan

1. You are going to create a coding plan based on the design document.  Name it `docs/coding_plan/rob_rt_coding_plan.md`.  Before doing so, ask me questions in the Q&A/Coding section below. Use Fable if you can.  Log your work.

2. Read my answers to the Q&A/Coding section below. Ask me more questions if needed. Use Fable if you can.  Log your work.

3. Based on our discussion, please write a coding plan.  Name it `docs/coding_plan/rob_rt_coding_plan.md`.  Use Fable if you can.  Log your work.

4. Generate a series of prompt docs to create the code based on the coding plan.  Name them `claude_prompts/RT/rob_rt_prompt_<number>.md`.  Use Fable if you can.  Log your work.

### Docs

## Q&A

### Coding

Before writing `docs/coding_plan/rob_rt_coding_plan.md`, a few decisions from
`docs/design/rob_rt_design.md` §7 ("Open items for the Coding Plan") need
your input — a Fable investigation grounded each in current code first, so
these are concrete, not open-ended.

1. **Float precision at the JAX/NumPy boundary.** BING's `evaluate.py`/
   `inference.py` are pure NumPy with no explicit dtype — everything is
   implicit float64. `robust` never enables `jax_enable_x64` in its own
   package code (only in tests/design scripts) and documents that its arrays
   are "float32, or float64 when `jax_enable_x64` is on"
   (`robust/rt/conventions.py:135`). Its own bit-for-bit BING cross-check
   only holds at `rtol <= 1e-6` **under** that fixture
   (`test_inelastic_bing_xcheck.py:10-13,64-83`); without it, `jnp.asarray`
   on BING's float64 arrays **silently downcasts to float32** — no error, no
   warning. `jax_enable_x64` is a **global, process-wide** JAX setting, not
   something scoped per-call. Options:
   - (a) Enable `jax_enable_x64` globally at import time of the new
     robust-backend adapter module — every fit in the process (Gordon or
     robust) then runs under it; simplest, matches robust's own tested
     precision, but changes JAX's default dtype process-wide for anything
     else sharing the interpreter (e.g. a notebook mixing this with other
     JAX code).
   - (b) Accept float32 for the robust backend (no global config change) and
     document the precision tradeoff — simpler, no global side effect, but
     never matches the 1e-6 cross-check regime in production use.
   - (c) Something else / want a recommendation.

   **Answer:** (b).  Document this and note that float32 is more than sufficient for our calculations.

2. **Wiring BING's Ed spectrum into `Geometry.Ed`.** Found that BING's
   `aNWModel.set_raman_Ed` (`bing/models/anw.py:480-511`) does **not** store
   the raw Ed spectrum — it immediately collapses `(wave_Ed, Ed)` into a
   dimensionless ratio, `Ed_ratio_raman` (anw.py:254-257, 504-506), which is
   all BING's own Raman path (`rrs.calc_Rrs`'s `Ed_ratio` arg) needs.
   `robust.rt.types.Geometry.Ed`, by contrast, wants the **raw spectrum
   pair** `(wave_Ed, Ed)` on its own grid (`robust/rt/types.py:320-334`) and
   builds the ratio internally (`robust/rt/ed.py:154-183`). The raw pair
   exists transiently at the one production call site,
   `fitting/l23.py:319-322`, generated via
   `correct_atmosphere.downwelling.downwelling_irradiance` at **solar
   zenith 0°** (not the per-pixel `theta_s` — sourcing that is out of scope
   per the design doc's non-goals). Two sub-questions:
   - Should `set_raman_Ed` be modified to *also* stash the raw pair (a small,
     backward-compatible addition serving both backends), or should the
     robust backend capture the pair independently at the `l23.py` call site
     without touching the existing method?
   - For now (since real per-pixel geometry sourcing is out of scope), is it
     correct for the robust backend to just reuse BING's existing zenith-0°
     Ed generation as-is, i.e. no new physics here, only re-routing an
     already-computed spectrum?

   **Answer:** (1) Yes, modify the code to store the raw pair. (2) Yes, the robust backend should just reuse BING's existing zenith-0° Ed generation as-is.

3. **Test strategy for the new test module.** BING's own tests use no
   `importorskip` anywhere; the symmetric pattern already lives on the
   `robust` side (`test_inelastic_bing_xcheck.py` does
   `pytest.importorskip("bing.rt.rrs")` since `bing` is not a runtime
   dependency of `robust`). But per Q10 (resolved), `bing` is taking
   `retrieve-or-bust` as a **real runtime dependency** — so should a new
   `bing/bing/tests/test_evaluate_robust.py` assume `robust` is always
   present (no `importorskip`, fails loudly if broken, consistent with
   `bing`'s existing no-`importorskip` convention), or still skip gracefully
   for contributors without it installed? Separately: since Ed-wiring tests
   likely exercise `bing.fitting.l23`, should this new module follow the
   existing `conftest.py` `collect_ignore` pattern that already drops
   `test_evaluate.py` when `correct_atmosphere` is missing
   (`bing/tests/conftest.py:84-87`)?

   **Answer:** Yes, for tests assume `robust` is always present.

4. **Default solar zenith when no geometry is supplied.** No package-level
   convention exists in BING today, and the two precedents disagree:
   production Ed generation always uses **0°** (`fitting/l23.py:321,356`),
   while the one existing geometry-adjacent test fixture hard-codes **30°**
   (`bing/tests/files/gen_l23_inelastic_fixture.py:32,47`). Which should
   `ObsGeometry`'s / the nadir-fallback default be — `0°` (matches
   production Ed generation), `30°` (matches the test fixture), or should
   the code instead **require** an explicit `theta_s` and raise rather than
   silently default?

   **Answer:** Yes, require an explicit `theta_s` and raise rather than silently default.

#### Coding Q&A: resolved

All four questions are answered with no contradictions between them or with
the resolved Design Q&A — see the Logs entry below for the consistency check
(including one design-doc placeholder that's now stale and worth updating).
Ready for the Coding Plan write-up.

### Design

Findings from examining `robust/rt` (retrieve-or-bust) and `bing/rt` +
`bing/evaluate.py` + `bing/fitting` (BING) are logged below under Logs. Based
on that survey, here are the open design questions — please answer inline
under each.

1. **Scope of the swap.** What's the intended relationship between BING's
   current Gordon/Raman/fluorescence forward model
   (`bing/evaluate.calc_Rrs_from_models` → `bing/rt/rrs.py`,
   `bing/rt/raman.py`, `bing/rt/chl_fl.py`) and `robust.rt`?
   - (a) Full replacement — `calc_Rrs_from_models` calls `robust.rt.forward()`
     for everything (elastic + inelastic); BING's own rrs/raman/chl_fl modules
     become thin wrappers or are retired.
   - (b) Selectable backend — add an `rt_dict["rt_backend"]` flag so
     `robust` is one option alongside the existing Gordon path; both stay
     maintained.
   - (c) Inelastic-only swap — keep BING's Gordon elastic core, replace only
     the Raman/fluorescence terms with `robust.rt.inelastic` (the piece
     already bit-for-bit cross-checked as the "fixed" physics in
     `test_inelastic_bing_xcheck.py`).
   - (d) Other / want a recommendation.

   **Answer:** (b)

2. **Which RT fidelity level for MCMC fitting.** `robust.rt` exposes three
   modes: `ztt` (pure analytic backbone), `hybrid` (ztt + learned emulator
   correction, most accurate vs L23 truth but carries emulator/out-of-domain
   overhead per call), and `baselines` (a Gordon-compatible refit, closest to
   a like-for-like validation swap). Which should the MCMC/least-squares
   fitters use?

   **Answer:** It should be selectable by the user.

3. **Geometry.** `robust.rt` requires explicit BRDF geometry
   (`Geometry(theta_s, theta_v, dphi, wind=...)`); BING currently has no
   explicit viewing/solar geometry — it's baked into Gordon's fixed
   mean-cosine coefficients (`mu_d`, `mu_u`, `mu_R`, `mu_f`). Options:
   - (a) Nadir-viewing default — `Geometry.nadir(theta_s)`, using only solar
     zenith from the scene (closest to BING's current implicit assumption).
   - (b) Extend BING's data model to carry per-scene `theta_s`/`theta_v`/
     `dphi` from satellite metadata through the fitting pipeline.
   - (c) Fixed global default geometry for all fits.
   - (d) Other / want a recommendation.

   **Answer:** (b)

4. **Performance / integration strategy.** MCMC calls the forward model
   ~10^5–10^7 times per fit (nwalkers × nsteps). `robust`'s `hybrid` mode runs
   a JAX/Flax MLP; BING's `emcee`-based loop currently evaluates per-sample.
   - (a) Vectorize + JIT — rework `log_prob` to batch walkers through a
     jit/vmap'd `robust.rt.forward` call.
   - (b) Wire it in as-is, benchmark against current Gordon speed, optimize
     only if it's actually a bottleneck.
   - (c) Use `ztt` inside the sampler for speed, run full `hybrid` only once
     post-hoc on the posterior summary for accuracy checks.

   **Answer:** (a) but have it be a separate function that uses as much of the existing BING code as possible.

5. **Deprecating BING's own inelastic code.** `robust/rt/inelastic.py` is
   described as an explicit JAX port of the BING "fixed" physics (branch
   `inelastic-fixes`), and `bing/claude_prompts/inelastic_fixes.md:126-129`
   says formulation-level fixes are "being addressed in the retrieve-or-bust
   forward-model redesign, not in BING." Does that mean `bing/rt/raman.py`
   and `bing/rt/chl_fl.py` are slated for deprecation/removal once this
   integration lands, with `robust.rt.inelastic` becoming the single source
   of truth?

   **Answer:** Yes, I expect they will not be used anymore but we will keep them around in case someone wants to use them.

6. **Wavelength grid reconciliation.** `robust.rt` uses a fixed canonical
   grid (350–750 nm, 5 nm steps, 81 bands); BING fits on whatever grid the
   data provides (irregular hyperspectral PACE, or discrete MODIS/SeaWiFS
   bands, per the `satellite-band-prep` skill). Should regridding/
   interpolation live inside `robust` (it already has clamped-linear
   `interp_spectrum`), inside BING's `evaluate.py`, or in a new adapter
   module?

   **Answer:** BING and RoB should throw an error if the user tries to use a different wavelength grid than RoB can handle.  We may need to generate a suite of RoB RT models.

7. **Particle phase function parameter.** `robust.rt.PhaseParams.B_p`
   (bb_p/b_p ratio, ~0.005–0.03) has no BING analogue today. Should it be a
   new free MCMC parameter, derived from/tied to an existing bb_nw model
   parameter, or fixed to a constant default?

   **Answer:** Let's allow for it to be a free parameter in the MCMC fitting, but can also be fixed to a constant default.

8. **Out-of-domain policy.** `robust`'s emulator can flag/handle inputs
   outside its L23 training domain (`on_out_of_domain="warn"`/etc). Real BING
   fits (e.g. turbid coastal waters, per `bing/dev/turbid_bbp`) will often
   fall outside that domain. What policy do we want — warn and continue,
   fall back to `ztt`-only, or hard-fail?

   **Answer:** Warn and continue.

9. **Deprecated `RT_correction` multiplier.** `bing/evaluate.py` has a
   multiplicative `RT_correction` knob flagged "THIS SHOULD BE REMOVED"
   (evaluate.py:204). Should this be dropped as part of this integration?

   **Answer:** Yes, this should be dropped.

10. **Dependency direction.** Today `robust`'s tests softly reference `bing`
    conventions via `importorskip("bing...")` (robust depends on bing being
    importable, only for cross-checks). For the actual interfacing, should
    `bing` add `retrieve-or-bust` as a real runtime dependency and call into
    `robust.rt.forward` from `evaluate.calc_Rrs_from_models`? Or is some other
    dependency/packaging arrangement intended (e.g. a shared third package,
    or robust exposing a stable public API that bing pins a version of)?

    **Answer:** Yes, `bing` should add `retrieve-or-bust` as a real runtime dependency and call into `robust.rt.forward` from `evaluate.calc_Rrs_from_models`.

#### Follow-up questions (round 2)

Your answers mostly resolve the design, but three points needed grounding in
the actual code before they can be turned into a coding plan. A follow-up
investigation (Fable) checked each; findings and new questions below.

11. **Re Q3 (geometry) — there is no existing per-pixel container to extend.**
    Checked `bing/fitting/inference.py`, `l23.py`, and `bing/data/`: BING
    currently threads observations through the fitters as bare tuples
    `(Rrs, varRrs, params, idx)` (`inference.py:214-215, 480-486`), and the
    only geometry-adjacent state anywhere is a hard-coded 30° solar zenith in
    a test fixture (`bing/tests/files/gen_l23_inelastic_fixture.py:32,47`).
    Satellite data directories (`bing/data/MODIS`, `PACE`, `SeaWiFS`) hold
    only Rrs/error CSVs — no geometry columns are loaded anywhere today. So
    "extend BING's data model" means building new plumbing, not editing an
    existing field. Specifically:
    - Do solar/viewing zenith and relative azimuth already exist per-pixel in
      BING's upstream input data (e.g. L2 satellite file matchups, Argo
      float matchups) and just aren't wired through yet — or does sourcing
      them require new data-loading work first?
    - Should geometry ride along as a new element in the existing
      `(Rrs, varRrs, params, idx)` tuple (minimal change), or is this the
      moment to introduce a proper per-observation dataclass that could also
      carry other future per-pixel metadata?
    - When per-pixel geometry is unavailable (e.g. older Argo matchups with
      no recorded viewing geometry), should the fit fall back to a
      nadir-only default for that pixel, or fail/skip it?

    **Answer:**  (1) These geometry terms are new and not yet in BING. (2) Add them to params; (3) Always fall back to a nadir viewing when the geometry is not specified.

12. **Re Q6 (wavelength grids) — the "hard error" premise doesn't fully hold.**
    The investigation found `robust.rt`'s two modes are more grid-flexible
    than assumed:
    - `ztt` is purely analytic (no neural net) and is grid-agnostic — its
      only wavelength dependence is a clamped `jnp.interp` lookup valid over
      350–800 nm (`robust/rt/ztt.py:802-811`). It needs no retraining to run
      at arbitrary satellite band centers.
    - The `hybrid` emulator is deliberately pointwise in λ — "one shared
      network maps the features at a wavelength to δ at that wavelength...
      defined on any wavelength grid... with λ as an input it can interpolate
      in" (`robust/rt/emulator.py:75-79`). So evaluating it at satellite band
      centers *inside* 350–750 nm is ordinary interpolation, which is exactly
      what it's designed to do well. Retraining would only be needed for
      bands falling **outside** 350–750 nm (e.g. PACE's UV/NIR edges), or if
      **band-averaged** (not band-center/monochromatic) Rrs is required,
      since the emulator corrects a monochromatic `rrs`.
    - No design doc in `robust/design/` or `docs/` discusses training
      per-instrument emulator suites; the only retraining note found concerns
      richer HydroLight geometry/phase-function runs, unrelated to
      wavelength grids (`design/rt_elastic_model.md:215-217`).

    Given that, three follow-ups:
    - Do you still want a hard grid-match error, or should the policy instead
      be: no restriction for `ztt`; for `hybrid`, error only when a requested
      band falls outside [350, 750] nm (relying on native interpolation
      within range)?
    - Does any BING target instrument (PACE, MODIS, SeaWiFS, ...) actually
      need **band-averaged** Rrs rather than band-center evaluation? That's
      the one case that would genuinely require new emulator training.
    - If per-instrument emulators are still wanted for some other reason
      (not identified above), is training them in scope for this
      integration, or a separate follow-on project?

    **Answer:** (1) Right, only error when the band is outside the range. (2) Don't worry about band-averaged Rrs for now.  But, yes, MODIS and other multi-band instruments do need band-averaged Rrs. (3) Training is not in scope for this integration.

13. **Re Q4 (performance) — confirming the dispatch mechanism.** To keep the
    new JIT'd/vectorized path as "a separate function that uses as much of
    the existing BING code as possible" (your answer to Q4) while also
    making the RT backend "selectable by the user" (Q1/Q2), the natural
    synthesis is: leave `evaluate.calc_Rrs_from_models` untouched for the
    Gordon path, add a new function (e.g. `calc_Rrs_from_models_robust`) that
    both `inference.log_prob` and `chisq_fit.fit_func` dispatch to when
    `rt_dict["rt_backend"]` selects `robust`, with the backend value also
    encoding which `robust.rt` mode (`ztt`/`hybrid`/`baselines`) to use. Is
    that the structure you want, or did you have a different dispatch point
    in mind?

    **Answer:** Your suggestions sound right.

#### Follow-up questions (round 3)

Answers to 11 and 12 resolve most of it, but each contains one point worth
checking/flagging before it goes into the Coding Plan.

14. **Re Q11 — "add them to params" is ambiguous given what `params` actually
    is.** Checked `bing/fitting/inference.py` directly: the `params` element
    of the `(Rrs, varRrs, params, idx)` tuple is literally the MCMC
    **initial-guess parameter vector**, `np.ndarray`, documented as such
    (`inference.py:179-182, 215, 481-485`) — it's a different object from
    `log_prob`'s own `params` argument, which is the live sample vector split
    into `aparams`/`bparams` (`inference.py:93-94`). So "add geometry to
    params" could mean either of two very different things:
    - (a) Geometry becomes new **fittable MCMC parameters** — i.e. `theta_s`/
      `theta_v`/`dphi` get appended into the parameter vector alongside the
      a/bb model params (analogous to how `B_p` can be free per Q7), and
      MCMC could in principle move them.
    - (b) Geometry rides along as **fixed, non-fit per-pixel data** — bundled
      into the same tuple/initial-guess package as a known, measured input
      (like `Rrs`/`varRrs` are), never varied by the sampler.

    Physically these are very different: geometry is normally a *known*
    quantity from satellite/scene metadata, not something you'd want MCMC to
    search over. Which did you mean?

    **Answer:** Ok, you are right.  It needs to be (b).

15. **Re Q12 — apparent tension in the band-averaged Rrs answer.** You wrote
    "don't worry about band-averaged Rrs for now" and then, in the same
    answer, "yes, MODIS and other multi-band instruments do need
    band-averaged Rrs." One resolution that keeps "training not in scope"
    (per your (3)) intact: band-averaging doesn't actually require retraining
    anything — it can be done as SRF-weighted numerical integration over
    several point evaluations of `ztt`/`hybrid` at fine wavelength spacing
    within each instrument band, since both modes already evaluate natively
    at arbitrary wavelengths (per the round-2 finding). Is that the right
    read — i.e. band-averaging via multi-point integration is in scope and
    should be built now (just not new emulator training), or do you want
    band-center approximation for this first integration with proper
    band-averaging deferred to a later phase?

    **Answer:** Band-averaging via multi-point integration is not in scope.  Don't preclude it, but don't include it for now.

#### Design Q&A: resolved

All 15 questions above are now answered with no open contradictions (see
Logs entry below for the consistency check). Ready to move to Coding Plan.


### Docs

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...


## Logs

### 2026-08-29 (Surveyed robust/rt and BING's forward RT model; logged Design questions)

Ran two read-only survey agents (model: Fable) in parallel, one over each
repository, to understand what interfacing BING with retrieve-or-bust's RT
model actually entails before proposing a design.

**retrieve-or-bust (`robust/rt`).** Not a numerical RTE solver — a JAX hybrid
`Rrs = Rrs_ZTT(θ) + ΔRrs_emulator(θ)`: an analytic Twardowski & Tonizzo (2018)
backbone (`ztt.py`) plus a small learned Flax MLP residual trained against L23
(HydroLight) truth (`emulator.py`), composed by `hybrid.py`'s public
`forward()`/`rrs_forward()`. It also ships a Gordon-compatible baseline
(`baselines.py`, same signature as `forward`) and explicit inelastic terms —
Raman (`inelastic.py`) and chlorophyll fluorescence (`inelastic.py`,
corrected by learned "heads" in `inelastic_corr.py`) — that are explicit JAX
ports of BING's own "fixed" physics (branch `inelastic-fixes`). Critically,
`robust/tests/test_inelastic_bing_xcheck.py` already pins `robust`'s Raman
factor and fluorescence kernel against BING's `bing.rt.rrs.calc_raman_correction_factor`
/ `calc_Rrs_fluorescence` at `rtol <= 1e-6` on 150 samples, and
`test_conventions.py` asserts equality of `A_Rrs`/`B_Rrs` and Raman-shift
constants between the two repos. So the inelastic-physics compatibility work
is largely already done and tested; what's undone is the actual call-site
integration. Full BRDF geometry (`theta_s`, `theta_v`, `dphi`) is required as
an explicit input via `Geometry`, and IOPs/wavelengths follow strict
conventions (linear m⁻¹, canonical 350-750 nm/5 nm/81-band grid,
`check_iop`/`check_wave` validators).

**BING (`bing/rt`, `bing/evaluate.py`, `bing/fitting`).** Forward model is the
classic Gordon (1988) `rrs = G1·u + G2·u²`, optionally wavelength-dependent
(G1(λ)/G2(λ), plus optional G0/Gb terms), with no explicit solar/viewing
geometry — it's baked into fixed mean-cosine constants (`mu_d`, `mu_u`,
`mu_R`, `mu_f`). Everything funnels through one choke point,
`evaluate.calc_Rrs_from_models`, called from both the MCMC likelihood
(`fitting/inference.py:log_prob`) and the least-squares initial-guess fitter
(`chisq_fit.py`) — so that's the natural swap point. Configuration is a plain
`rt_dict` (built by `rt_dict_from_p`) with flags for variable Gordon
coefficients, Raman, and fluorescence; a deprecated `RT_correction` multiplier
is flagged in-code as "THIS SHOULD BE REMOVED". Found that
`bing/claude_prompts/inelastic_fixes.md:126-129` explicitly states that
certain formulation-level fixes (fixed-μ two-flow stream geometry) are "being
addressed in the retrieve-or-bust forward-model redesign, not in BING" —
i.e. this integration was already anticipated as the intended next step, not
a new idea.

**Outcome.** Ten open design questions were identified — spanning scope
(replace vs. side-by-side backend vs. inelastic-only swap), which `robust`
fidelity mode to use for MCMC (ztt/hybrid/baselines), how to supply BRDF
geometry BING doesn't currently track, MCMC performance/JIT strategy, whether
BING's own `raman.py`/`chl_fl.py` should be deprecated in favor of
`robust.rt.inelastic`, wavelength-grid reconciliation, the unmodeled
`PhaseParams.B_p` parameter, emulator out-of-domain policy, the deprecated
`RT_correction` knob, and dependency direction between the two packages. Per
the user's request, these were written directly into the `Q&A/Design`
section above (rather than asked interactively) for the user to answer
inline before a design document is written.

### 2026-08-29 (Read Design answers, ran round-2 investigation, logged follow-up questions)

Read all ten answers in `Q&A/Design`. Most resolve cleanly: selectable
`rt_backend` (Q1), user-selectable `robust` fidelity mode (Q2), a JIT'd
vectorized path implemented as a separate function reusing existing BING
code (Q4), BING's own Raman/fluorescence modules kept but unused by default
(Q5), free-or-fixed `B_p` (Q7), warn-and-continue for out-of-domain inputs
(Q8), dropping `RT_correction` (Q9), and `bing` taking `retrieve-or-bust` as
a real runtime dependency (Q10).

Three answers (Q3 geometry, Q6 wavelength grids, Q4 dispatch mechanism)
needed grounding in the actual code before they could become coding-plan
items, so ran a second read-only survey agent (model: Fable) over both
repos. Key findings, now logged as follow-up questions 11-13 in
`Q&A/Design`:

- **Geometry (Q3):** BING has no per-pixel metadata container today —
  observations flow through the fitters as bare `(Rrs, varRrs, params, idx)`
  tuples (`bing/fitting/inference.py:214-215, 480-486`), and no geometry
  column is loaded anywhere in `bing/data/`. "Extending BING's data model"
  is new plumbing, not an edit to an existing field, so asked where the
  per-pixel angles would actually come from and how they should be threaded
  through.
- **Wavelength grids (Q6):** the premise behind "throw an error, may need a
  suite of RoB models" doesn't fully hold. `robust.rt.ztt` is purely
  analytic and grid-agnostic (`robust/rt/ztt.py:802-811`); the `hybrid`
  emulator is explicitly pointwise-in-λ and designed to interpolate at any
  wavelength inside its 350-750 nm training range
  (`robust/rt/emulator.py:75-79`). Retraining would only be needed for bands
  outside that range or for band-averaged (vs. band-center) Rrs — neither of
  which is currently discussed anywhere in `robust/design/` or `docs/`. Asked
  whether the grid policy should be relaxed accordingly, and whether any
  target instrument actually needs band-averaged Rrs.
- **Dispatch mechanism (Q4):** proposed a concrete synthesis (leave
  `evaluate.calc_Rrs_from_models` untouched for Gordon; add
  `calc_Rrs_from_models_robust`, dispatched from `log_prob`/`chisq_fit` via
  `rt_dict["rt_backend"]`) and asked for confirmation before it goes into the
  Coding Plan.

Also incidentally confirmed `PhaseParams.B_p` in `robust/rt/types.py:239-244`
broadcasts against either a scalar or a per-wavelength spectrum, so a scalar
free MCMC parameter (per Q7's answer) is a valid, well-supported choice — no
follow-up question needed there.

### 2026-08-29 (Read round-2 answers, logged round-3 follow-ups)

Read answers 11-13. Q13 (dispatch mechanism) was confirmed as-is, no
follow-up. Two answers needed one more check each:

- **Q11 (geometry → "add them to params"):** grepped
  `bing/fitting/inference.py` directly (no subagent needed for a one-line
  lookup) and found `params` in the `(Rrs, varRrs, params, idx)` tuple is
  literally the MCMC **initial-guess parameter vector**
  (`inference.py:179-182, 215, 481-485`), a different object from
  `log_prob`'s own sample-vector argument (`aparams`/`bparams`,
  `inference.py:93-94`). So "add geometry to params" is genuinely ambiguous
  between "geometry becomes a new fittable MCMC parameter" and "geometry
  rides along as fixed, non-fit per-pixel data" — physically very different,
  since geometry is normally known/measured, not retrieved. Logged as
  follow-up 14.
- **Q12 (band-averaged Rrs):** the answer says both "don't worry about it for
  now" and "yes, MODIS etc. need it," which reads as contradictory as
  written. Logged follow-up 15 proposing a resolution that satisfies both:
  band-averaging via SRF-weighted numerical integration over several `ztt`/
  `hybrid` point evaluations needs no retraining (consistent with "training
  not in scope"), so it may already be compatible with the stated
  constraints — asked for confirmation.

### 2026-08-29 (Read round-3 answers; confirmed Design Q&A is fully resolved)

Read answers 14-15. Both landed cleanly, no further contradiction:

- **Q14 (geometry):** confirmed geometry is fixed, non-fit per-pixel data
  (option b) — never varied by the sampler, bundled alongside `Rrs`/`varRrs`
  as a known input, defaulting to nadir when unspecified (per Q11's answer).
  This also resolves the open sourcing question from Q11: since no
  per-pixel geometry data exists in BING today, the near-term deliverable is
  the plumbing to accept it as optional fixed data (nadir default); wiring
  in real satellite metadata for a given instrument is future work, not
  blocking.
- **Q15 (band-averaging):** deferred — band-center approximation for now,
  explicitly not precluding the SRF-weighted multi-point-integration approach
  later.

Did a full consistency pass over all 15 Q&A entries looking for
contradictions (e.g. Q1 "selectable backend" vs. Q10's phrasing "call
`robust.rt.forward` from `evaluate.calc_Rrs_from_models`" vs. Q13's agreed
dispatch through a new `calc_Rrs_from_models_robust` function) — found
nothing unresolved; Q13's answer already supersedes Q10's literal phrasing
and both point at the same design. No further questions needed this round.
Marked Design Q&A resolved in the doc; ready for the Coding Plan prompt.

### 2026-08-29 (Wrote the design document: `docs/design/rob_rt_design.md`)

Delegated the write to a Fable-model agent, giving it all 15 resolved Q&A
decisions plus the two prior survey findings as source material, and pointed
it at `retrieve-or-bust/design/rt_elastic_model.md` purely as a style
reference (structure/tone, not content). The agent read this file's full
`Q&A`/`Logs` sections itself and wrote
[`bing/docs/design/rob_rt_design.md`](../docs/design/rob_rt_design.md) (415
lines) — reviewed it directly afterward rather than trusting the agent's
summary.

Structure: title + companion-doc preamble citing this Q&A log; 7 sections —
Goals/non-goals, Architecture (ASCII dispatch diagram), Interface and data
model, Wavelength-grid/out-of-domain policy, Dependency/packaging,
Deprecations, Open items for the Coding Plan.

Key concrete decisions it locked in, beyond restating the Q&A:
- A single combined `rt_dict["rt_backend"]` selector (`"gordon"` default,
  `"robust_ztt"`, `"robust_hybrid"`, `"robust_baseline"`) rather than two
  separate keys — keeps illegal states unrepresentable and legacy `rt_dict`s
  working via `.get(..., "gordon")`.
- A new `ObsGeometry` frozen dataclass (`bing/rt/geometry.py`) riding as an
  optional 5th element of the observation tuple
  (`Rrs, varRrs, params, idx, geom`) — fixed, never sampled, nadir fallback.
- Free `B_p` appended as the *last* MCMC vector element so the existing
  `aparams`/`bparams` split (`inference.py:93-94`) survives unchanged.
- New `calc_Rrs_from_models_robust` placed beside (not inside)
  `calc_Rrs_from_models`, with a mapping table from BING objects to
  `robust.rt` inputs, and both fitters gaining a two-line backend branch.
- Grid policy: no restriction for `ztt`/`baseline`; `hybrid` errors only if a
  band falls outside [350, 750] nm, checked once at fit setup.
- A real implementation nuance the earlier survey missed: `robust.rt`'s
  out-of-domain warning is silent under `jax.jit` (traced inputs), so
  warn-and-continue (Q8) requires an explicit un-jitted domain check run once
  before and once after sampling, not inside the hot loop.
- Caught and corrected two stale line-number citations from the earlier
  survey (`forward` at hybrid.py:364, not 353; `PhaseParams.B_p` at
  types.py:250-275, not 239-244) by re-reading current source rather than
  trusting the round-1 report.

The doc explicitly restates non-goals (band-averaging, per-instrument
emulator retraining, real geometry ingestion) and ends with a short "Open
items for the Coding Plan" list rather than trying to fully plan the
implementation — that's deferred to the Coding Plan prompt.

### 2026-08-29 (Investigated design doc's open items; logged Q&A/Coding questions)

Before drafting `docs/coding_plan/rob_rt_coding_plan.md`, ran a Fable
investigation grounding four of the six items in `rob_rt_design.md` §7 in
current code, then logged questions 1-4 in `Q&A/Coding` above.

- **Float precision (§7.1):** confirmed BING is pure NumPy/implicit float64
  throughout `evaluate.py`/`inference.py`, while `robust` never enables
  `jax_enable_x64` in its own package code (only in tests/design scripts) and
  documents itself as float32-unless-x64
  (`robust/rt/conventions.py:135`). Its own cross-check test only hits
  `rtol <= 1e-6` under that fixture. Without it, BING's float64 arrays
  silently downcast to float32 crossing into JAX — no error. Since
  `jax_enable_x64` is a global process setting, this is a real tradeoff
  (global precision vs. a process-wide side effect), not a trivial default —
  logged as question 1.
- **Ed wiring (§7.4):** found BING's `set_raman_Ed`
  (`bing/models/anw.py:480-511`) discards the raw Ed spectrum and keeps only
  a ratio (`Ed_ratio_raman`), but `robust.rt.types.Geometry.Ed` wants the raw
  `(wave_Ed, Ed)` pair — which exists transiently at `fitting/l23.py:319-322`
  generated at solar zenith 0°. Logged question 2: modify `set_raman_Ed` to
  also stash the raw pair, or capture it independently; and confirm reusing
  the existing zenith-0° Ed generation as-is (no new physics) is correct
  given real geometry sourcing is out of scope.
- **Test plan (§7.5):** found BING's own tests use no `importorskip`
  anywhere, while `robust`'s cross-check test does
  (`pytest.importorskip("bing.rt.rrs")`, since `bing` isn't a runtime dep of
  `robust`). Since `bing` *is* taking `robust` as a real runtime dependency
  (Q10), logged question 3 asking whether the new test module should match
  BING's no-`importorskip` convention, and whether it needs the same
  `correct_atmosphere`-missing `collect_ignore` handling as
  `test_evaluate.py` (`bing/tests/conftest.py:84-87`), since Ed-wiring tests
  will likely touch `bing.fitting.l23`.
- **Default `theta_s` (§7.2):** confirmed the two existing precedents
  disagree — production Ed generation uses 0° (`fitting/l23.py:321,356`),
  the one test fixture uses 30°
  (`bing/tests/files/gen_l23_inelastic_fixture.py:32,47`) — logged as
  question 4, including a third option (require explicit `theta_s`, no
  silent default).

§7.3 (`B_p` prior) and §7.6 (docs updates) didn't need further grounding —
both are self-contained implementation choices the coding plan can specify
directly once §7.1-.2/.4-.5 are answered.

### 2026-08-29 (Read Coding Q&A answers; confirmed resolved, no follow-up needed)

Read all four answers. All land cleanly:

- **Float precision:** float32, no global `jax_enable_x64`, documented as
  sufficient precision for BING's use case.
- **Ed wiring:** modify `set_raman_Ed` to also stash the raw `(wave_Ed, Ed)`
  pair; reuse the existing zenith-0° production Ed generation as-is.
- **Test strategy:** new tests assume `robust` is always present, no
  `importorskip` — matching BING's existing convention now that `robust` is
  a real runtime dependency. (The `correct_atmosphere`/`collect_ignore`
  sub-question wasn't explicitly answered, but doesn't need to be: it's an
  orthogonal, already-existing convention for a *different* optional
  dependency that Ed-wiring tests would trigger regardless of the robust
  question — the coding plan can just note the new test module inherits it
  where it touches `bing.fitting.l23`.)
- **Default solar zenith:** require explicit `theta_s`, raise if absent —
  narrower than, and consistent with, the earlier Design-phase nadir
  fallback (Q11/Q14): that fallback covers missing *viewing* geometry
  (`theta_v`/`dphi` default to 0 for nadir), while this answer covers the
  case where `theta_s` itself is unknown, which now errors instead of
  guessing.

**One stale cross-reference found, not yet fixed:** `docs/design/rob_rt_design.md`
§3.2 still documents a *tentative* `theta_s = 0.0` default and frames "0°
vs. 30°" as the open Coding Plan question — superseded by this round's
"raise, don't default" answer. Left as-is for now since only asked to read
answers/log work this round; flagged here so the design doc gets a matching
one-line update before or alongside the coding plan write-up.

No new questions needed. Marked Coding Q&A resolved in the doc; ready for
the coding-plan write-up prompt.

### 2026-08-29 (Wrote the coding plan: `docs/coding_plan/rob_rt_coding_plan.md`)

Delegated the write to a Fable-model agent, same pattern as the design-doc
write. Gave it three authoritative sources to read in full itself: the
design doc, this file's complete `Q&A`/`Logs` sections (all 15 Design + 4
Coding decisions), and `retrieve-or-bust/design/rt_elastic_model_coding_plan.md`
as a structure-only style reference. Also asked it to make one small,
surgical fix while it had the context loaded: the stale `theta_s = 0.0`
tentative-default placeholder in `docs/design/rob_rt_design.md` §3.2,
superseded by the Coding Q&A's "require explicit `theta_s`, raise" answer.
Reviewed both the new file and the design-doc diff directly afterward.

**Coding plan** —
[`bing/docs/coding_plan/rob_rt_coding_plan.md`](../docs/coding_plan/rob_rt_coding_plan.md)
(410 lines): milestone-gated **M0-M5**, each with Tasks/Deliverable/pytest
Gate, plus a "Ground rules" section restating the 4 Coding Q&A decisions as
binding constraints (float32-only, no `jax_enable_x64`; `set_raman_Ed`
stashes the raw Ed pair; new tests assume `robust` is always present, no
`importorskip`; `theta_s` required, never defaulted) and a files-touched
diagram, a milestone table, testing strategy, dependency changes, risks, and
a definition of done.

Sequencing: **M0** dependency + `rt_dict` keys + `ObsGeometry` + a new
`validate_rt_dict` fit-setup checker; **M1** the `calc_Rrs_from_models_robust`
adapter (JIT cache keyed on mode/inelastic-config/wave, the un-jitted
domain-check helper since `robust`'s `DomainWarning` is silent under `jit`)
+ dropping `RT_correction`; **M2** the two-line backend dispatch in both
fitters + geometry threaded as an optional 5th tuple element (a plan choice:
keep the tuple, no dataclass promotion, per design §7.2); **M3** free `B_p`
appended as the last MCMC vector element with a plan-chosen linear-uniform
[0.004, 0.05] prior; **M4** the `set_raman_Ed` raw-pair stash and its route
into `Geometry.Ed`, plus the `correct_atmosphere` `collect_ignore` isolation
so Ed tests don't gate earlier milestones; **M5** docstring-only deprecation
notes, skill/doc updates, and a throughput benchmark (reported, not
thresholded — an accept/optimize call left to the user).

One correction the agent made along the way worth noting: it set parity/
inelastic test tolerances to float32-honest values (`rtol <= 1e-5` /
`5e-4`) rather than reusing `robust`'s own `1e-6` cross-check figure, since
that figure only holds under `jax_enable_x64` — which Coding Q1 explicitly
ruled out. Confirmed this is the correct reading of Q1, not a new
discrepancy.

Design Q&A, Coding Q&A, the design document, and now the coding plan are all
complete and mutually consistent. Next open prompt slots are `### Report`
and `### Docs` (both currently empty) — no Coding Plan follow-up questions
were raised this round since none were needed.

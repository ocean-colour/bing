# RT using Retreive or Bust (RoB)

## Goals

Interaface BING with the radiative transfer model from the retrieve-or-bust repository.

## Prompts

### Design

1. Examine the code in the retrieve-or-bust repository and the BING code.  We wish to interface BING with the radiative transfer model from the retrieve-or-bust repository.  Please ask me a set of questions in Q&A/Design before writing the document.  Use Fable if you can.  Log your work.

2. Read my answers to the Q&A/Design section below. Ask me more questions if needed. Use Fable if you can.  Log your work.

### Coding Plan

### Report

### Docs

## Q&A

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

### Coding Plan

### Report

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

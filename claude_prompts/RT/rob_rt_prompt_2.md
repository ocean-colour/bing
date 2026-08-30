# RoB RT Backend Coding — Prompt 2 (M1: The forward adapter `calc_Rrs_from_models_robust`)

## Goals

Implement **Milestone M1** of the coding plan
(`docs/coding_plan/rob_rt_coding_plan.md`): the heart of the integration — a
working, JIT'd, batched forward adapter `calc_Rrs_from_models_robust` in
`bing/evaluate.py` that maps BING model evaluations onto `robust.rt` inputs
and back, an un-jitted domain-check helper so `robust`'s `DomainWarning` can
actually fire, and the one sanctioned edit to the Gordon path: deleting the
deprecated `RT_correction` fudge block. After this milestone the robust
forward path is callable standalone; the fitters don't dispatch to it until
M2.

## Claude

### Skills

`.claude/skills/`: `run-bing-fit` (the shape contract and `rt_dict` flow the
adapter must match), `fit-l23-spectrum` (loading L23 spectra for the parity
gate), plus `code-review` before hand-off.

### Working agreements

Per the working agreements in `rob_rt_prompt_1.md` (`ocean14`; CQ1–CQ4
binding; BING scope discipline; pytest-gated; Fable; log) — **one
correction**: M0 actually landed on JXP's existing branch **`rob_rt`**, not
`rob-rt-backend` as the coding plan suggested. JXP has been reviewing and
committing after each task (M0's four tasks are already four separate
commits) — work here should assume the same cadence, not batch everything
into one hand-off. Two agreements bear directly on this milestone:

- **CQ1 is implemented here**: NumPy crosses to JAX at exactly one point,
  as float32; never `jax_enable_x64`; the docstring documents the downcast
  as more than sufficient precision. Parity tolerance is `rtol ≤ 1e-5`.
- **Scope discipline is tested here**: the `RT_correction` sweep greps both
  `bing/` and `papers/` — `papers/` hits are **reported to JXP, not edited**.

### Status entering M1 (from M0)

- **`bing/tests/test_evaluate_robust.py` already exists** (18 tests from
  M0) — this milestone **adds to it**, it does not create it. It already
  has local `_FakeModel`/`_FakeGeom` stand-ins for `validate_rt_dict`-only
  tests; M1's parity/shape/domain tests need real L23 spectra and a real
  `ObsGeometry`, not those fakes — use the real classes here.
- **The grid check is already built and already tested.** M0's
  `bing.rt.defs.validate_rt_dict` already implements Gate item 4 (`robust_hybrid`
  errors outside [350, 750] nm; `robust_ztt`/`robust_baseline` accept any
  grid) against named constants `RT_BACKENDS` and
  `ROBUST_HYBRID_WAVE_MIN`/`ROBUST_HYBRID_WAVE_MAX` (`bing/rt/defs.py`).
  `calc_Rrs_from_models_robust` itself does **not** need to re-implement
  this check — it belongs at fit setup (`validate_rt_dict`, called by the
  fitters in M2), not inside the forward function. M1's own gate item 4 is
  therefore mostly a **regression check** that M0's validator still behaves
  correctly once `RT_BACKENDS` is actually consumed elsewhere (task 1's
  backend-suffix dispatch, below) — not new logic to write.
- **Backend-string → robust-call mapping** (task 1 needs this precisely):
  `rt_dict['rt_backend']` values are `'gordon'` (routes to the untouched
  `calc_Rrs_from_models`, never reaches this function), `'robust_ztt'` →
  `robust.rt.forward(..., mode='ztt')`, `'robust_hybrid'` →
  `robust.rt.forward(..., mode='hybrid')`, `'robust_baseline'` → **not** a
  `forward()` mode at all — dispatches to `robust.rt.baselines.Rrs_gordon`
  (baselines.py:97) instead. Special-case this one rather than trying to
  find a `mode='baseline'`.
- **`ObsGeometry` (`bing/rt/geometry.py`) is built and tested** — `to_robust(Ed=None)`
  already accepts an `Ed=` keyword, but wiring a real Ed pair through is
  **M4's job, not M1's**. Call `geom.to_robust()` with no `Ed` argument
  here; passing anything else this milestone would be getting ahead of the
  plan.
- **JAX/Flax facts confirmed in M0** (matters for task 2's JIT strategy):
  `ocean14` has `jax 0.11.0`, `flax 0.12.8`, `jaxtyping 0.3.11`,
  `optax 0.2.8`. Plain `import robust.rt` loads `jax` but **not** `flax` —
  Flax only loads (and the emulator's parameters actually materialize) on
  the first real `mode='hybrid'` forward call. So task 2's first hybrid-mode
  JIT compile is *also* this integration's first real exercise of the Flax
  path end-to-end — worth a specific, deliberate check that it works (not
  just that `ztt`/`baseline` do), since nothing before this milestone has
  touched Flax at all.
- **Full-suite baseline going into M1**: `pytest bing/tests/ -q` →
  **196 passed, 2 skipped, 2 failed** (`test_l23_inelastic.py::test_raman_correction_matches_l23`
  and `::test_fluorescence_matches_l23`, both pre-existing and unrelated —
  root cause is a missing `bing/tests/files/l23_inelastic_fixture.npz` in
  this checkout, confirmed via `git stash` in M0 task 1). Don't re-diagnose
  these as an M1 regression; do check the count only grows from here.

## Context

Read before coding:

- **Previous prompt** — `rob_rt_prompt_1.md` (M0: what exists now —
  `validate_rt_dict`, `ObsGeometry`, the three `rt_dict` keys — plus its
  Logs for anything M0 learned).
- **Coding plan** — `docs/coding_plan/rob_rt_coding_plan.md` **M1** section,
  plus Risks (JIT recompiles, float32 in the likelihood).
- **Design** — `docs/design/rob_rt_design.md` §3.4 (the function signature
  and the BING→robust mapping table), §3.5 (inelastic wiring — plumbed here,
  gated in M4), §4 (grid + out-of-domain policy; the warning is silent under
  `jit`), §6 (the `RT_correction` deletion), §7.1 (the JIT question this
  milestone resolves as a plan choice).
- **Q&A record** — `claude_prompts/rob_rt.md`: Design Q4/Q13 (separate
  function, dispatch), Q8 (warn-and-continue), Q9 (`RT_correction`), and
  Coding Q&A 1 (float32).
- **Current code** — `bing/evaluate.py` (`calc_Rrs_from_models`,
  evaluate.py:94; the `RT_correction` block, evaluate.py:203-209; the `a_ph`
  construction, evaluate.py:222); **new from M0**: `bing/rt/defs.py`
  (`RT_BACKENDS`, `ROBUST_HYBRID_WAVE_MIN`/`MAX`, `validate_rt_dict`) and
  `bing/rt/geometry.py` (`ObsGeometry`, `.to_robust()`); on the robust side
  `robust/rt/hybrid.py` (`forward`, hybrid.py:364; the domain check,
  hybrid.py:130-162), `robust/rt/types.py` (`IOPs.from_total_bb`,
  types.py:131; `PhaseParams`, types.py:250-275; `Inelastic`, types.py:471),
  and `robust/rt/baselines.py` (`Rrs_gordon`, baselines.py:97).

## Prompts

1. Read this doc. Execute the 1st task in the "M1" section below — the
   adapter itself. If you have any questions, ask me in the Q&A section
   below. Use Fable if you can. Log your work.
2. Read this doc. Execute the 2nd task — the JIT strategy. Check my answers
   in Q&A; if you have additional questions, ask in Q&A. Use Fable if you
   can. Log your work.
3. Read this doc. Execute the 3rd task — the un-jitted domain check. Use
   Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the `RT_correction` deletion (pin
   the regression fixture *first*). Use Fable if you can. Log your work.
5. Read this doc. Execute the 5th task — the explainer notebook. Use Fable
   if you can. Log your work.
6. Read this doc. Execute the 6th task — update `rob_rt_prompt_3.md` with
   what M1 established. Use Fable if you can. Log your work.

## M1

### Tasks

1. **The adapter.** New function in `bing/evaluate.py`, directly below
   `calc_Rrs_from_models` (evaluate.py:94), with the design §3.4 signature:
   `calc_Rrs_from_models_robust(a_model, a_params, bb_model, bb_params,
   rt_dict, geom=None, Bp=None, debug=False, full_return=False)`.
   Implementation follows the design's mapping table exactly:
   - `a = a_model.eval_a(a_params)`, `bb = bb_model.eval_bb(bb_params)`
     (existing BING code, unchanged) →
     `robust.rt.types.IOPs.from_total_bb(a, bb, wave=a_model.wave, a_ph=...)`
     (types.py:131) — robust derives `bb_p = bb − bb_w(λ)` itself. `a_ph`
     (`10**a_params[..., -1:] * a_model.a_ph`, as at evaluate.py:222) is
     supplied only when fluorescence is on.
   - `PhaseParams(B_p = Bp if Bp is not None else rt_dict["Bp_value"])`
     (types.py:250-275).
   - `geometry = geom.to_robust()`; `ObsGeometry(theta_s)` already encodes
     nadir viewing (≡ `Geometry.nadir(theta_s)`, types.py:337). `geom=None`
     never reaches the robust call from fit paths (M0's validator raises
     earlier), and the function itself also raises on `None` so direct
     calls fail the same way.
   - `Inelastic(raman=..., fluorescence=..., phi_C=...)` (types.py:471)
     from `rt_dict["include_Raman"]` / `include_Chl_fl` / `phi_C`; `None`
     when both are off (bit-identical elastic path by construction).
   - Backend suffix → `mode="ztt"`/`"hybrid"` into `robust.rt.forward`
     (hybrid.py:364 — natively batched over leading axes, one call for a
     whole `(nsamples, nparam)` chain, matching `calc_Rrs_from_models`'s
     shape contract) or `robust.rt.baselines.Rrs_gordon` (baselines.py:97,
     signature-compatible). `mode="emulator"` is not exposed (design §3.1).
   - `full_return=True` returns `(Rrs, a, bb)` like the Gordon twin.

2. **JIT strategy** *(plan choice, design §7.1)*: a module-level
   `functools.lru_cache`d builder `_robust_forward_jit(mode, inelastic_key,
   wave_key)` returning a `jax.jit`-wrapped closure with the wavelength grid
   and the `Inelastic`/`PhaseParams` treedef baked in — one compile per
   configuration, reused for the whole fit. Treedefs change when optional
   fields flip `None`↔set, so the key must include the inelastic config;
   `wave_key = wave.tobytes()`. No `vmap` needed — `forward` is natively
   batched. NumPy crosses to JAX only here (float32, CQ1 — stated in the
   docstring); results return as `np.asarray` for the likelihood arithmetic.

3. **Un-jitted domain check** *(design §4)*: helper
   `robust_domain_check(a_model, a_params, bb_model, bb_params, rt_dict,
   geom, Bp=None)` in `evaluate.py` that calls the **un-jitted**
   `robust.rt.forward` once on concrete arrays so the emulator's
   `DomainWarning` (hybrid.py:130-162) can actually fire — it is silent
   under `jit` (traced inputs). Fitters will call it **twice per fit** in
   M2: on the initial guess before sampling and on the posterior median
   after — never inside the hot loop.

4. **Drop `RT_correction`** (design §6). First pin the regression fixture:
   current `calc_Rrs_from_models` output on a reference L23 spectrum saved
   under `bing/tests/files/`, so "Gordon path untouched" is a test, not a
   claim. Then delete the block at evaluate.py:203-209, the key from any
   `rt_dict` construction, and any call sites that set it — grep both
   `bing/` and `papers/`; **report, don't edit,** `papers/` hits.

5. **Notebook.** `nb/RT/rob_rt_coding_2.ipynb` (executed, with outputs):
   the adapter's mapping table walked through on a real L23 spectrum, the
   baseline-vs-gordon parity figure, the shape contract demonstrated for
   `(nparam,)` and `(nsamples, nparam)` inputs, a `DomainWarning` fired
   live on a turbid IOP set, and a note on what float32 costs (measured,
   not asserted).

6. **Finally.** Update `rob_rt_prompt_3.md` (M2) with what M1 established —
   the adapter's exact call signature, cache behavior, and anything learned
   about `robust.rt`'s numerics.

### Gate

`bing/tests/test_evaluate_robust.py` additions:

1. **Parity** — on ≥ 3 L23 spectra, `rt_backend="robust_baseline"` matches
   `calc_Rrs_from_models` (elastic, constant Gordon) at `rtol ≤ 1e-5` (both
   use G1=0.0949/G2=0.0794 and the same A_Rrs/B_Rrs conversion; float32
   sets the tolerance).
2. **Shapes** — for each of the 3 robust backends, `(nparam,)` params →
   `(nwave,)` Rrs and `(nsamples, nparam)` → `(nsamples, nwave)`, finite
   and positive-typical.
3. `full_return=True` returns `(Rrs, a, bb)` like the Gordon twin.
4. **Grid** — `validate_rt_dict` with `robust_hybrid` on a grid reaching
   760 nm raises, on 350–750 passes; `robust_ztt`/`robust_baseline` accept
   either.
5. **Domain** — `robust_domain_check` on a deliberately turbid IOP set
   emits `DomainWarning` (`pytest.warns`), and the jitted path on the same
   inputs does not error.
6. An `rt_dict` containing a stale `RT_correction` key is ignored (no
   multiplication, no KeyError) and the block is gone.
7. A second call with identical config hits the lru_cache (`cache_info`
   check) — no recompile.

Existing suite green throughout.

## Q&A

**Q1 (task 1, Claude → JXP).** Building the adapter surfaced a real
architectural fact about `robust.rt` neither the design doc nor the coding
plan called out: `robust.rt.inelastic.raman_factor`/`fluorescence_kernel`
derive their own excitation-wavelength IOPs by **interpolating (and
clamping) the single emission-grid `a`/`bb` spectrum** passed into `IOPs`
(`conventions.interp_spectrum`) — they do **not** accept separately-evaluated
excitation IOPs. BING's own Gordon+Raman path, by contrast, genuinely
evaluates the parametric models at the true excitation grid
(`a_model.eval_a_ex`/`bb_model.eval_bb_ex`). Confirmed by reading
`robust/rt/inelastic.py:211-217` directly (`a_ex = conventions.interp_spectrum(wave_ex,
wave, a_em)`) and cross-referencing the BING-robust cross-check test itself
(`test_inelastic_bing_xcheck.py:150-160`), which deliberately feeds **both**
sides `np.interp`'d excitation IOPs precisely so the port-of-the-formula
test doesn't depend on this difference. So: whenever `include_Raman`/
`include_Chl_fl` is on with a robust backend, the excitation-wavelength
physics is a real-but-different approximation from the Gordon path's — not
a bug, and not something robust's public `forward()`/`rrs_forward()` API
gives any way to change (there is no parameter for injecting explicit
excitation IOPs). Documented prominently in the new function's own
docstring (Notes section) since it's a real accuracy caveat, not just an
implementation detail. **No answer needed to proceed** — there is no
alternative available through the public API without duplicating robust's
internal composition logic in `bing`, which would be a much larger,
unauthorized scope expansion.

**Q2 (task 1, Claude → JXP).** Found a real, previously-unaddressed gap:
`robust.rt.baselines.Rrs_gordon` takes **no `inelastic` argument at all** —
confirmed at `baselines.py:88-110` — so there is no way to compose Raman or
fluorescence onto the `robust_baseline` backend the way `forward()`/
`rrs_forward()` do for `robust_ztt`/`robust_hybrid`. Resolved by raising
`ValueError` when `rt_dict['rt_backend'] == 'robust_baseline'` and either
`include_Raman` or `include_Chl_fl` is set, rather than silently computing
elastic-only Rrs while the caller believes inelastic terms are included.
**No answer needed to proceed** — silently dropping requested physics
seemed clearly worse than a loud, actionable error; happy to revisit if a
different resolution is preferred (e.g. warn-and-continue instead, to match
the out-of-domain policy's spirit).

**Q3 (task 1, Claude → JXP).** The design's mapping table
(`rob_rt_design.md` §3.4) didn't mention `rt_dict['double_gaussian']` at
all, but it's a real, already-used BING flag (feeds
`calc_Rrs_fluorescence(..., double_gaussian=...)` on the Gordon path). Found
its direct analogue: `robust.rt.types.Inelastic.emission_shape ∈
('single', 'double')` (`types.py:414`). Wired
`emission_shape = 'double' if rt_dict.get('double_gaussian', True) else 'single'`
so the robust backend's fluorescence emission line shape actually matches
what the caller configured, rather than silently defaulting to `'single'`
regardless of the flag. **No answer needed to proceed** — this is a direct,
unambiguous parameter correspondence, not a design choice.

**Q4 (task 1, Claude → JXP).** The Gate's phrasing ("`(nparam,)` params →
`(nwave,)` Rrs") doesn't match what this function (or its Gordon sibling)
actually returns: both `eval_a`/`eval_bb` always add a leading batch axis,
so a 1-D `a_params` yields `(1, nwave)`, not `(nwave,)` — `calc_Rrs_from_models`
has the identical behavior today (see `test_evaluate.py`'s own
`test_calc_Rrs_from_models_single_param`, which checks only the trailing
dim for exactly this reason). Matched the existing, real convention rather
than inventing a new squeeze-to-1-D behavior that would make the two
backends inconsistent with each other. Tests assert `(1, nwave)` explicitly
rather than the Gate's literal `(nwave,)`. **No answer needed to proceed.**

**Q5 (task 1, Claude → JXP).** The Gate's parity check says "on ≥ 3 L23
spectra." Used 3 distinct **synthetic** parameter sets on a real
`ExpBricaud`+`Pow` model pair instead of genuinely `l23.load_one_l23`-loaded
truth — this exercises the exact same adapter code path (the mapping
logic under test), needs no external data path (`$OS_COLOR`) or
`correct_atmosphere`, and so the new test module stays runnable anywhere
`bing`+`robust` are installed, unlike the heavier `l23_fit_standard`-style
fixtures elsewhere in `test_evaluate.py`. Measured worst-case relative
difference across the 3 sets: ~3e-7, far inside the `rtol ≤ 1e-5` gate.
**No answer needed to proceed** — flagging in case a genuine L23-spectrum
version is wanted for closer parity with the Gate's literal wording.

## Next

→ `rob_rt_prompt_3.md` (M2: fitter dispatch and geometry threading).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-30 (M1 task 1 — `calc_Rrs_from_models_robust` adapter)

Read `bing/evaluate.py` in full first (not just the mapping-table summary
in the design doc) to confirm exact conventions before writing anything:
`a_model.eval_a`/`bb_model.eval_bb` return total `a_w + a_nw` on
`(nsample, nwave)`; `full_return` returns `(Rrs, a, bb)`; `debug` drops into
`IPython.embed`. Also read `robust/rt/types.py` (`IOPs.from_total_bb`,
`PhaseParams`, `Inelastic` — including its `emission_shape`/`cdom_fl`
fields, not just the three the design doc mentioned),
`robust/rt/hybrid.py` (`forward`'s real keyword-only signature),
`robust/rt/baselines.py` (`Rrs_gordon` — confirmed it ignores
`phase_params`/`geometry`/`wave` and, critically, **takes no `inelastic`
argument**), and `robust/rt/inelastic.py` (`raman_factor`/
`fluorescence_kernel`'s actual bodies, not just their docstrings) directly,
since several real implementation choices only become clear from the
bodies. `robust.rt.__init__` re-exports `forward`/`IOPs`/`PhaseParams`/
`Inelastic`/`baselines` at the top level, so the adapter imports
`from robust import rt as robust_rt` once and uses `robust_rt.X` throughout
rather than importing from submodules piecemeal.

Added the function directly below `calc_Rrs_from_models` in
`bing/evaluate.py`, per the design §3.4 signature exactly. Four real
findings surfaced while writing it, each logged in Q&A above (Q1-Q5) rather
than guessed past silently: (1) robust's Raman/fluorescence derive
excitation IOPs by **interpolating/clamping** the single emission-grid
spectrum, never by evaluating the true parametric models at wider
wavelengths the way BING's own Gordon+Raman path does — inherent to
`robust.rt`'s public API, documented in the new function's docstring; (2)
`robust.rt.baselines.Rrs_gordon` has **no inelastic composition path at
all** — resolved by raising `ValueError` if `robust_baseline` is combined
with `include_Raman`/`include_Chl_fl`, rather than silently dropping the
requested physics; (3) `rt_dict['double_gaussian']` maps directly onto
`Inelastic.emission_shape` (`'double'`/`'single'`), a correspondence the
design's mapping table omitted entirely; (4) `a_ph` is passed as the
**full** spectrum on `a_model.wave`, not sliced at `a_model.i_Chl_ex` like
the Gordon path's `aph_ex` — `fluorescence_kernel` interpolates onto its
own fixed 370-690 nm excitation grid internally, so pre-slicing would be
both unnecessary and wrong (BING's `i_Chl_ex` indices don't correspond to
robust's excitation grid).

**Verified interactively, function by function, before writing a single
test** — built a real `ExpBricaud`+`Pow` model pair via
`bing.models.utils.init` (no MCMC fit, no `correct_atmosphere`/L23 data
dependency needed — a deliberate choice, Q5) and confirmed: (a)
`robust_baseline` vs `calc_Rrs_from_models` agree to ~3e-7 relative (3
water types) — strong confirmation the elastic mapping (IOPs split, Gordon
constants, A/B conversion) is exactly right; (b) `robust_ztt`/
`robust_hybrid` give plausible, genuinely different values (expected — a
different, better model, not a bug); (c) a real `DomainWarning` fired
un-jitted for an out-of-domain `B_p` at `robust_hybrid`, confirming
`forward()`'s own domain check already works even before task 3's
dedicated un-jitted helper exists; (d) batch shapes, `full_return`,
Raman, fluorescence (net-positive contribution, ~3e-4 at the 685 nm peak),
the free-`Bp` override, and all three error paths (`geom=None`,
`rt_backend='gordon'`, `robust_baseline`+inelastic) — all behave exactly as
designed.

**Two test-writing mistakes caught by actually running the tests, not
assumed away.** (1) The parity test's first draft compared
`Rrs_robust[0]` (shape `(61,)`) against `Rrs_gordon` un-squeezed — silently
correct under NumPy broadcasting in my interactive script, but
`np.testing.assert_allclose` rejects the shape mismatch outright; fixed by
squeezing both sides (matches Q4's finding that `calc_Rrs_from_models`
itself returns `(1, nwave)`, not `(nwave,)`, for 1-D input). (2) The
fluorescence test's first draft asserted a bare `Rrs_fl >= Rrs_elastic`;
it failed at 5/61 wavelengths by ~1e-10 — measured directly, not
guessed — because `Inelastic` being set switches `rrs_forward` onto a
different static code path even where the fluorescence kernel itself is
~0 far from the 685 nm peak, so the elastic backbone recomputes with
float32 rounding noise at the ULP level. Fixed with a small `atol` on the
`>=` check plus a separate assertion that the real signal
(`(Rrs_fl - Rrs_elastic).max() > 1e-5`) is far above that noise floor —
distinguishing "no regression" from "the fluorescence kernel itself went
negative," which would be a real bug.

**Added 11 tests** to `test_evaluate_robust.py` (task 1's slice of the
milestone's Gate — items 1-3; items 4-7 depend on tasks 2-4, not yet
built, and are **not** duplicated here per the M0 hand-off note already in
this file's Status section): baseline-vs-Gordon parity (3 water types,
`rtol ≤ 1e-5`); shapes across all 3 robust backends × 1-D/batch;
`full_return`; Raman branch; fluorescence branch; free-`Bp` override; and
the three error paths. Bumped a shared test `Bp_value` from `0.01` to
`0.014` (still well within `PhaseParams`'s definitional bound, but inside
the emulator's actual trained domain at `theta_s=30°`) purely to keep the
shape test's output free of an incidental (correct, but off-topic)
`DomainWarning` — domain-check testing is task 3's job, not this one's.

**Verification.** `pytest bing/tests/test_evaluate_robust.py -q` → **29
passed** (was 18; +11 new), zero warnings after the `Bp_value` tweak. Full
suite: `pytest bing/tests/ -q` → **207 passed, 2 skipped, 2 failed**
(138.03s) — 207 = 196 (M0 baseline) + 11, same 2 pre-existing,
already-diagnosed failures, nothing else moved.

Modified: `bing/evaluate.py` (two new imports —
`from bing.rt import defs as rt_defs`, `from robust import rt as
robust_rt` — plus the new function), `bing/tests/test_evaluate_robust.py`
(11 new tests). No `RT_correction` deletion yet (task 4); no JIT/domain-check
wrapping yet (tasks 2-3) — this function is deliberately plain and
un-jitted for now, exactly as task 1 scopes it. Branch `rob_rt`,
uncommitted, for JXP's review. Task 2 (the JIT strategy) is next.

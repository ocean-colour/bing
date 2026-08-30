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

**Q6 (task 2, Claude → JXP).** Implementing the JIT strategy hit **two**
real, previously-undocumented `robust.rt` gotchas — both are lazy
disk-loads triggered by leaving a `forward()` keyword at its default, and
both raise `jax.errors.UnexpectedTracerError` the *first* time they fire
inside a `jax.jit` trace (a side effect escaping the trace scope), not a
one-off fluke — reproduced deterministically on every fresh
`_robust_forward_jit` cache miss:

1. `corrections=None` (the default) makes `hybrid.py:_resolve_corrections`
   call `inelastic_corr.load_default()` — this fires whenever `Inelastic`
   is set (**any** mode, not just `hybrid`; first hit was actually on
   `robust_ztt` + fluorescence). Fixed with `corrections=False` — the
   *documented* "analytic-only, explicit and silent" option
   (`hybrid.py:102-120`), and also the only inelastic physics this
   integration ever resolved or cross-checked in the first place (M3's
   learned correction heads were never part of any Q&A/design decision).
2. `emulator=None` (the default) makes `hybrid.py:_resolve_emulator` call
   `emulator.load_default()` — `hybrid`-mode only. Fixed by loading it
   once **outside** `jit`, inside the (already-cached) builder itself —
   `robust_rt.emulator.load_default()` is itself memoised process-wide
   (`emulator.py:1093-1108`, "read once per process"), so this adds no
   redundant I/O — and passing the loaded object explicitly as
   `emulator=emulator_obj`, so `forward()` never has a reason to load
   anything at trace time.

Neither is a design decision to revisit — both are the only physically
sensible fix given the constraint (`robust`/retrieve-or-bust source cannot
be modified). **No answer needed to proceed** — flagging because this is
exactly the kind of "JIT recompiles across configs" risk the coding plan's
Risks section named in the abstract (M1 risk 1), now a concrete, reproduced
failure mode with its actual fix on record rather than a hypothetical.

**Q7 (task 3, Claude → JXP).** The task spec says `robust_domain_check`
"calls the un-jitted `robust.rt.forward` once so the emulator's
`DomainWarning` can actually fire" — but only **one** of the three robust
backends has a domain to check at all. Confirmed by reading robust's source
directly: `forward()`'s `_check_domain` call sits *past* the `mode='ztt'`
early return (`hybrid.py:303-309` — ztt returns before `_resolve_emulator`
is ever reached), and `robust.rt.baselines.Rrs_gordon` (the
`robust_baseline` dispatch target) is a closed-form expression with no
emulator, no `check_domain` keyword, and no domain logic anywhere in
`baselines.py`. Resolved by making the helper a **validated no-op** for
`robust_ztt`/`robust_baseline`: it still runs the full argument
construction (so the same `ValueError`s fire as on the hot path — missing
`geom`, non-robust backend, baseline+inelastic), but performs no forward
call — running the un-jitted forward for a mode that cannot warn would be
pure wasted compute with no diagnostic value. Documented in the function's
docstring and pinned by a parametrized test. A secondary decision made in
the same task: rather than duplicating the adapter's BING→robust argument
construction (IOPs split, `PhaseParams`, `geom.to_robust()`, inelastic
flags, the three validity errors), that logic was factored out of
`calc_Rrs_from_models_robust` into a shared private builder
`_build_robust_inputs` returning a `_RobustInputs` namedtuple — both the
jitted hot path and the un-jitted check now consume it, so the checked
configuration can never drift from the fitted one. The adapter's behavior
is unchanged (all 34 pre-existing tests pass untouched, same error
messages). **No answer needed to proceed.**

**Q8 (task 4, Claude → JXP).** The task spec offered "a reference L23
spectrum" for the regression fixture while Q5's precedent for this milestone
was synthetic parameters. Resolved with a hybrid that satisfies both: the
fixture *generator* (`bing/tests/files/gen_l23_gordon_fixture.py`) derives
its inputs from real L23 data — idx=170 (the same reference index
`test_evaluate.py`'s end-to-end fixtures fit), PACE grid, the true L23 `Chl`
via `prep_one_l23`, and parameters from `init_guess` on the spectrum's true
`anw`/`bbnw` — but stores everything needed to re-evaluate (wave, Chl, the
exact parameter arrays, RT flags) in the `.npz`, so the regression *test*
rebuilds the models from the public API alone (`model_utils.init` +
`set_aph`) with zero L23/`$OS_COLOR` dependency at test time. This avoids
exactly the failure mode the two pre-existing `test_l23_inelastic.py`
failures demonstrate (a test needing a data-dependent fixture missing from
a checkout): the `.npz` is committed and self-contained. Three Gordon
configurations are pinned, bracketing the deleted block on both sides of
the function body — elastic constant-Gordon (1-D + batch, `full_return` →
Rrs/a/bb), +Raman (batch), +fluorescence (batch). **No answer needed to
proceed.**

**Q9 (task 4, Claude → JXP).** The `RT_correction` sweep of `papers/` found
**7 hits in 2 files, both in `papers/biomass/Analysis/py/` — reported, not
edited, per the working agreement — and they now need a decision**, because
with the block deleted the key they set is silently ignored:

- `fitting.py:55` — `fit_me(..., RT_correction:np.ndarray=None)` keyword
  argument; `fitting.py:72` — its docstring ("This is a hack to explore RT
  effects"); `fitting.py:128-129` — `if RT_correction is not None:
  rt_dict['RT_correction'] = RT_correction` just before the LM + MCMC fit.
- `lowest_bbp.py:518` — under its `correct_RT` option, computes
  `RT_correction = spec['Rrs_true'] / Rrs_GordonE` (the true-Rrs/Gordon
  ratio on the L23 truth); `lowest_bbp.py:520` — the `else: RT_correction =
  None` branch; `lowest_bbp.py:528` — passes it into `fit_me(...,
  RT_correction=RT_correction)`.

`bing/` itself never constructed the key (`rt_dict_from_p` never emitted
it), so these two scripts were its only real users. Post-deletion,
`fit_me(RT_correction=...)` and `lowest_bbp.py`'s `correct_RT=True` path
run without error but **silently apply no correction** — arguably worse
than crashing for anyone re-running that exploration. **Question for JXP:**
should these be (a) left as-is (dead plumbing, historical scripts), (b)
stripped of the `RT_correction` plumbing, or (c) made to raise loudly if a
correction is requested? (b) or (c) require edits under `papers/`, which I
am not authorized to make.

**Q10 (task 4, Claude → JXP).** Found the likely root cause of the 2
pre-existing `test_l23_inelastic.py` failures while committing-prepping the
new fixture: the repo's `.gitignore` (line 16) has a blanket `*.npz`, so
`git status` does not even show `bing/tests/files/l23_gordon_fixture.npz`
as untracked — and `l23_inelastic_fixture.npz` (whose generator's docstring
says "committed") was presumably silently excluded the same way, which is
exactly why it's missing from this checkout. **Question for JXP:** when
committing task 4, please either `git add -f
bing/tests/files/l23_gordon_fixture.npz` or add a
`!bing/tests/files/*.npz` exception to `.gitignore` (the latter would also
let `l23_inelastic_fixture.npz` be committed once regenerated, fixing the
2 standing failures for every checkout). I did not edit `.gitignore`
myself — changing what the repo commits is your call.

**Q11 (task 5, Claude → JXP).** Building the notebook produced the first
actual *measurement* of what CQ1's float32 downcast costs, plus two smaller
numeric facts worth having on record:

1. **Measured float32 cost.** `robust_baseline` computes the identical
   physics to `calc_Rrs_from_models` (same Gordon constants, same A/B
   surface conversion), so on a real L23 spectrum (idx=170, PACE grid) the
   baseline-vs-Gordon residual *is* the float64-vs-float32 difference,
   isolated: **max 2.31e-7 relative (1.9 float32 ULPs; median 8.2e-8),
   max 7.7e-10 sr⁻¹ absolute** — 26,124× below even an optimistic 2%
   measurement-noise floor (2.0e-5 sr⁻¹ on this spectrum's median Rrs of
   1.0e-3 sr⁻¹). Task 1's independently-observed ~1e-10-scale inelastic
   static-path noise reproduces live in the same notebook: −4.7e-10 sr⁻¹
   worst "negative fluorescence" dip at 15/61 wavelengths vs a +7.1e-5
   sr⁻¹ real 685 nm emission signal (signal/noise ≈ 152,000×).
2. **Where the downcast actually happens.** The task-2 docstring's "NumPy
   crosses to JAX at the jit boundary" is not literally where the bits
   change: `IOPs.from_total_bb` itself calls `jnp.asarray`, so the leaves
   land as float32 there — *before* the jit boundary (confirmed live:
   `iops.a.dtype == float32` straight out of the constructor). Same CQ1
   outcome (one NumPy→JAX crossing, float32, nothing in `bing` casts
   explicitly), just one call earlier than the docstring's mental model —
   noted in the notebook, not worth a code change.
3. **jit-vs-eager rounding.** The adapter's jitted closure and a manual
   un-jitted `robust_rt.forward` call on bit-identical inputs differ by up
   to 9.3e-10 sr⁻¹ (XLA fusion reorders float32 arithmetic) — same
   ULP-level scale as (1), harmless, but worth knowing the two paths are
   not bit-identical. Also for the record: on this L23 spectrum the real
   robust backends sit **above** Gordon by +0.3% to +7.4% (median +4.5%,
   ztt; hybrid similar) — smooth, physical, and exactly the kind of
   difference the integration exists to capture.

**No answer needed to proceed** — all three are measurements now recorded
in the executed notebook (`nb/RT/rob_rt_coding_2.ipynb`), not code changes.

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

### 2026-08-30 (M1 task 2 — JIT strategy: `_robust_forward_jit`)

Implemented the task spec's `functools.lru_cache`d builder
`_robust_forward_jit(mode, inelastic_key, wave_key)` exactly as designed:
`wave` decoded from `wave_key` bytes (`np.frombuffer(..., dtype=np.float64)`,
BING's convention) and baked into the closure as a Python-level constant,
never a traced argument; `inelastic_key` either `None` (elastic-only,
genuinely distinct from an `Inelastic(raman=False, fluorescence=False)`
instance per the design's own "None is bit-identical by construction"
note — kept as separate cache entries, never conflated) or
`(raman, fluorescence, emission_shape)`, the three *static* `Inelastic`
fields; `phi_C` stays a real traced leaf, reconstructed into a fresh
`Inelastic` instance inside the closure body each call. Rewired
`calc_Rrs_from_models_robust` to look up the cached closure and call it,
instead of calling `robust_rt.forward`/`baselines.Rrs_gordon` directly (its
task-1 code). Confirmed `IOPs`/`PhaseParams`/`Geometry` need no manual
`jnp.asarray`/dtype casting anywhere in this module: they're plain
float64 NumPy-backed objects until they cross into a `jax.jit`-wrapped
closure, at which point JAX's own argument-conversion handles the
float64→float32 downcast (CQ1) automatically — "NumPy crosses to JAX only
here" is a property of *where the objects are passed into `jit`*, not
something requiring an explicit cast anywhere in this file.

**Two real bugs, found by running the code, not by reading docs.** First
run of the parametrized shape test crashed with
`jax.errors.UnexpectedTracerError` — not a logic error in my closure, but
`robust.rt.forward`'s own default `corrections=None` lazily calling
`inelastic_corr.load_default()` (a disk read) *inside* the trace, on
whichever mode/config combination happened to hit fluorescence first
(`robust_ztt`, not `hybrid` — this fires for **any** mode when `Inelastic`
is set, not just `hybrid`). Fixed with `corrections=False`
(`hybrid.py:102-120`'s own documented "analytic-only, explicit, silent"
option) — also the *only* inelastic behavior ever authorized by any
resolved Q&A/design decision, so this isn't a scope reduction, just making
explicit what was already true. Re-ran; a **second**, independent
tracer-leak surfaced immediately after, this time only on `robust_hybrid`:
`_resolve_emulator`'s own lazy `emulator.load_default()` disk read. Fixed
by calling `robust_rt.emulator.load_default()` once inside the (already
`lru_cache`d) builder — outside the `jax.jit` boundary entirely — and
passing the loaded `Emulator` object explicitly as `emulator=emulator_obj`
so `forward()` has no default left to fall back on at trace time. Checked
`emulator.load_default()`'s own docstring before assuming this was safe to
call once per builder invocation rather than once per process: it's
memoised process-wide already ("read once per process"), so this adds no
redundant I/O beyond what `robust` itself already does. Logged both as
**Q6** — genuinely new information (the coding plan's Risks section named
"JIT recompiles across configs" only in the abstract; this is the concrete
failure mode and its fix), not a design question needing an answer.

**Verified the caching actually works, at both levels, not just that
tests pass.** Interactively (before writing tests): `cache_info()` showed
`misses=1, hits=2` after 3 identical-config calls, and a 4th call under a
different `rt_backend` bumped `misses` to 2 — the Python-level cache
behaves correctly. Separately checked the *underlying* `jax.jit` object's
own compiled-trace count (`jit_fn._cache_size()`) stays at 1 across
repeat calls with the same shapes — confirming XLA itself isn't
recompiling either, which the `lru_cache` layer alone wouldn't prove (a
cache hit returning the same jitted closure says nothing about whether
*that* closure would still retrace on each call).

**Added 5 tests**: cache hit/miss counts across a repeat call, parametrized
over all 3 backends (also checks the underlying `jit_fn._cache_size()`);
distinct `rt_backend` values produce distinct cache entries (`misses == 3`
across 3 different configs); `inelastic_key=None` vs a real key are
separate entries (`misses == 2`, not collapsed to 1) — directly exercising
the None-vs-instance distinction Q6/the design's own note calls out.

**Verification.** `pytest bing/tests/test_evaluate_robust.py -q` → **34
passed** (was 29; +5 new). Full suite: `pytest bing/tests/ -q` → **212
passed, 2 skipped, 2 failed** (135.93s) — 212 = 207 + 5, same 2
pre-existing, already-diagnosed failures, nothing else moved.

Modified: `bing/evaluate.py` (`_robust_forward_jit` added; `import
functools`, `import jax`, `import jax.numpy as jnp` added;
`calc_Rrs_from_models_robust`'s body rewired to dispatch through it),
`bing/tests/test_evaluate_robust.py` (5 new tests). Branch `rob_rt`,
uncommitted, for JXP's review. Task 3 (the un-jitted domain check) is
next — building on a codebase where `forward()` now always runs with
`corrections=False` and an explicitly-loaded `emulator_obj`, both facts
task 3's un-jitted call should reuse rather than re-derive.

### 2026-08-30 (M1 task 3 — un-jitted domain check: `robust_domain_check`)

Read before coding: the current actual state of `bing/evaluate.py` (task 2
rewired the adapter through `_robust_forward_jit` since the task-1 log was
written — the file, not the log prose, is ground truth for what the helper
had to reuse), `robust/rt/hybrid.py`'s domain machinery (`DomainWarning` at
hybrid.py:84, `_is_traced` at 123-136, `_check_domain` at 139-163 — silent
whenever *any* pytree leaf is a tracer, which is exactly why the jitted hot
path can never warn), and `robust/rt/baselines.py` end to end. That reading
produced the task's one real architectural finding, logged as **Q7**: the
domain check only exists for `mode='hybrid'` — `forward()` returns for
`mode='ztt'` *before* `_resolve_emulator`/`_check_domain` are reached
(hybrid.py:303-309), and `baselines.Rrs_gordon` has no domain logic at
all — so `robust_domain_check` is a **validated no-op** for
`robust_ztt`/`robust_baseline` (argument errors still fire; no wasted
un-jitted forward call for a mode that cannot warn).

Built two things in `bing/evaluate.py`. (1) A shared private builder
`_build_robust_inputs(a_model, a_params, bb_model, bb_params, rt_dict,
geom, Bp)` returning a `_RobustInputs` namedtuple — the adapter's entire
BING→robust argument construction (geom/backend/baseline+inelastic
validation, `eval_a`/`eval_bb`, the `a_ph` fluorescence source term,
`IOPs.from_total_bb`, `PhaseParams` with the free-`Bp` override,
`geom.to_robust()` with no `Ed` per the M4 boundary, `emission_shape`,
`phi_C`) factored out of `calc_Rrs_from_models_robust` verbatim, which now
consumes it; behavior and error messages unchanged (all 34 pre-existing
tests pass untouched). (2) `robust_domain_check(a_model, a_params,
bb_model, bb_params, rt_dict, geom, Bp=None)` — the exact spec signature,
no `debug`/`full_return` — directly below the adapter: builds inputs
through the same shared builder (the checked configuration can never drift
from the fitted one), returns immediately for non-hybrid backends (Q7),
and otherwise calls the **un-jitted** `robust_rt.forward` once on the
concrete NumPy-backed inputs with `mode='hybrid'`, `corrections=False`,
and an explicitly-loaded `emulator=robust_rt.emulator.load_default()` —
task 2's two Q6 fixes reused deliberately (not for jit-safety here, since
nothing is traced, but so the domain check evaluates the identical forward
configuration the fit uses; `load_default()` is memoised process-wide, so
no added I/O). Returns `None` — a diagnostic side-effect function, and the
docstring says so, along with *why* it must stay un-jitted.

**Verified interactively before writing tests** (ocean14, real
`ExpBricaud`+`Pow` pair, `theta_s=30°`): `Bp_value=0.01` — the task-1
log's incidental trigger — fires exactly one `DomainWarning` un-jitted
("B_p 100.0% of values outside [0.01026, …]"), and the emulator's actual
trained lower bound is ~0.01026, i.e. 0.01 sits only ~2.5% of the span
outside it. The tests therefore use `Bp_value=0.005` instead — well
outside, so the gate test isn't riding a knife edge of the trained domain.
Also confirmed live: the jitted path (`calc_Rrs_from_models_robust`) on
the *same* out-of-domain inputs raises nothing, warns nothing, and returns
finite Rrs (the silence being the entire point, design §4/Q8);
`Bp_value=0.014` (the in-domain value the rest of the test module uses) is
silent un-jitted; ztt/baseline are silent no-ops even on the turbid B_p;
and both shared error paths (`geom=None`, `rt_backend='gordon'`) raise the
same `ValueError`s from the helper as from the adapter. No new bugs — the
one surprise was Q7's ztt-has-no-domain-check fact, caught by reading
`forward()`'s body rather than assuming every mode could warn.

**Added 5 test functions (6 test items)** to `test_evaluate_robust.py`:
Gate item 5's two halves as separate tests (`pytest.warns(DomainWarning,
match='outside its training range')` un-jitted on the turbid set; the
jitted path on identical inputs finite with zero `DomainWarning`s
recorded), in-domain silence + `None` return, the ztt/baseline no-op
(parametrized, 2 items), and the shared adapter error paths. Also updated
the module docstring's coverage note (it still said tasks 2-3 "land in
later additions").

**Verification.** `pytest bing/tests/test_evaluate_robust.py -q` → **40
passed** (was 34; +6 items). Full suite: `pytest bing/tests/ -q` → **218
passed, 2 skipped, 2 failed** (139.92s) — 218 = 212 + 6, same 2
pre-existing, already-diagnosed `test_l23_inelastic.py` failures (missing
fixture file), nothing else moved.

Modified: `bing/evaluate.py` (`import collections`; `_RobustInputs` +
`_build_robust_inputs` added; `calc_Rrs_from_models_robust` body rewired
through the builder, behavior unchanged; `robust_domain_check` added),
`bing/tests/test_evaluate_robust.py` (5 new tests + docstring update).
Branch `rob_rt`, uncommitted, for JXP's review. Task 4 (dropping
`RT_correction`) is next — pin the Gordon-path regression fixture on a
reference L23 spectrum under `bing/tests/files/` *first*, then delete the
block at evaluate.py's `RT_correction` stanza and sweep call sites
(`papers/` hits reported, not edited).

### 2026-08-30 (M1 task 4 — pin regression fixture, drop `RT_correction`)

Read before coding: the *current* `calc_Rrs_from_models` body end to end
(the doc's `evaluate.py:203-209` reference is stale after tasks 1-3 grew
the file — the block actually sat at evaluate.py:211-217, found by
searching the string, not the line number), `rt_dict_from_p`
(`bing/rt/defs.py:21-55` — confirmed it never emitted an `RT_correction`
key, so no rt_dict construction in `bing/` needed editing), the full
`RT_correction`/`RT_corr` grep across `bing/` and `papers/` (no
differently-cased or partial-name aliases exist — the block used the
rt_dict key directly, no local variable), and the two existing fixture
generators in `bing/tests/files/` for the naming convention
(`gen_<name>.py` → `<name>.npz`, generator committed next to the fixture
with a regenerate-once docstring).

**What the block actually did** (deletion rationale on record, design §6):
flagged `# THIS SHOULD BE REMOVED` in the source itself, it multiplied the
already-computed Gordon `Rrs` by a caller-supplied per-wavelength array
whenever `rt_dict.get('RT_correction')` was not None — `Rrs *
rt_dict['RT_correction']` for 1-D params, `Rrs * np.outer(ones(nsample),
...)` for batches. A fudge for forcing Gordon Rrs toward HydroLight truth,
superseded outright by this integration (the robust backends *are* the
principled version of that correction). Verified live on the pre-deletion
code that the block genuinely fired when the key was set (a uniform
factor-2 key exactly doubled Rrs on both the 1-D and batch branches) — so
the Gate-item-6 test added below is decisive, not vacuous — and that no
real config ever set the key (`rt_dict_from_p` never built it), so the
fixture snapshot is a true "what the function outputs today, dead code and
all" capture.

**Step A first, exactly as specced.** Fixture:
`bing/tests/files/l23_gordon_fixture.npz` (17 kB, committed), generated by
the new `gen_l23_gordon_fixture.py` on the *unmodified* code. L23-derived
inputs, data-free test — Q8 has the full rationale: parameters from
`prep_one_l23(p, idx=170)`'s `init_guess` on the true L23 anw/bbnw (PACE
grid, true L23 Chl), plus a deterministic ±2% 5-sample batch, with the
realized arrays stored in the `.npz` so the test rebuilds everything from
the public API alone (no `$OS_COLOR` at test time). Pinned three Gordon
configs bracketing the block: elastic constant-Gordon (1-D + batch,
`full_return` → Rrs/a/bb), +Raman (batch; `set_raman_Ed` with the real
downwelling Ed, same recipe as `prep_one_l23`), +fluorescence (batch;
`init_Chl_fluorescence`, `phi_C=0.02`, double Gaussian). Added 3 tests to
`test_evaluate.py` (where the Gordon-path `calc_Rrs_from_models` tests
live) at `rtol=1e-10` — same code path, same float64 inputs, only
cross-platform BLAS noise allowed; a live `RT_correction` would miss by
O(1). **Ran them on the still-unmodified code first: 3 passed.**

**Step B.** Deleted the 7-line stanza (comment + `if`/`else`) at
evaluate.py:211-217 — the function's docstring never mentioned the key, so
no docstring edit needed. That was the *only* functional `RT_correction`
code anywhere in `bing/`: the sweep found no key in any rt_dict
construction and no `bing/` call sites (the remaining `bing/` grep hits are
the new test/generator docstrings describing the deletion itself, plus
docs/prompts history). `papers/` hits: **7, in 2 files**
(`papers/biomass/Analysis/py/fitting.py` and `lowest_bbp.py`) — reported
in Q9 with a real question attached, since those scripts' `correct_RT`
machinery now silently no-ops rather than erroring; not edited, per the
working agreement. Added the Gate-item-6 test to `test_evaluate_robust.py`
(`test_calc_Rrs_from_models_ignores_stale_RT_correction_key`): a stale
factor-2 `RT_correction` key in an otherwise-identical rt_dict changes
nothing, bit-for-bit, on both the 1-D and batch paths, and raises nothing.

**Verification.** Before: `test_evaluate_robust.py` **40 passed**, full
suite **218 passed, 2 skipped, 2 failed** (task-3 baseline). After:
`test_evaluate.py -k gordon_fixture` → **3 passed** (both before and after
the deletion — the pin held, proving the block was inert in every real
config); `test_evaluate_robust.py -q` → **41 passed** (+1, Gate item 6);
full suite `pytest bing/tests/ -q` → **222 passed, 2 skipped, 2 failed**
(140.29s) — 222 = 218 + 4 (3 fixture-regression + 1 stale-key), same 2
pre-existing `test_l23_inelastic.py` failures (missing
`l23_inelastic_fixture.npz`), nothing else moved. Gate items 1-7 are now
all covered: 6 landed here; 1-3, 5, 7 in tasks 1-3; 4 was M0's validator.

Modified: `bing/evaluate.py` (block deleted, nothing else),
`bing/tests/test_evaluate.py` (3 regression tests + `os`/`model_utils`
imports), `bing/tests/test_evaluate_robust.py` (1 test + module-docstring
coverage note). Added: `bing/tests/files/gen_l23_gordon_fixture.py`,
`bing/tests/files/l23_gordon_fixture.npz` — **note the `.npz` is invisible
to git under the repo's blanket `*.npz` ignore rule; Q10 explains, and asks
for `git add -f` or a `.gitignore` exception when committing** (the same
rule is the likely root cause of the 2 standing `l23_inelastic` failures).
Branch `rob_rt`, uncommitted, for JXP's review — plus Q9 awaiting an answer
on the `papers/biomass` plumbing. Task 5 (the explainer notebook
`nb/RT/rob_rt_coding_2.ipynb`) is next.

### 2026-08-30 (M1 task 5 — explainer notebook)

Built and **executed** `nb/RT/rob_rt_coding_2.ipynb` (25 cells: 13 markdown,
12 code, every code cell with real output), next to M0's
`rob_rt_coding_1.ipynb` and matching its style (markdown-explained sections,
same kernel metadata). Every number below is quoted *from the executed
outputs*, written after the cells ran — not before.

**Setup + Section 1 (mapping table, walked live).** Loads a genuinely real
L23 spectrum — idx=170 on the PACE grid via `prep_one_l23`/`init_guess`,
the exact recipe of task 4's fixture generator (needs `$OS_COLOR` at run
time; Chl=0.1306, 61 bands 400–700 nm) — then walks design §3.4's mapping
table step by step with printed intermediates: `eval_a`/`eval_bb`
(a(440)=0.02622, bb(440)=0.00263 m⁻¹) → `IOPs.from_total_bb` (robust's own
water split: bb_w(440)=0.002196, bb_p(440)=0.000439; `bb_p == bb − bb_w`
verified True; u(440)=0.09130) → `PhaseParams(B_p=0.014)` →
`geom.to_robust()` (nadir `Geometry`, Ed=None per the M4 boundary) →
`Inelastic=None` → backend dispatch. Closes the loop by composing the
robust call *by hand* (`robust_rt.forward(..., mode='ztt',
corrections=False)`) and showing the adapter reproduces it to 9.3e-10 sr⁻¹
(jit-vs-eager float32 rounding — Q11 item 3). Also shown live: the
float64→float32 crossing actually happens inside `IOPs.from_total_bb`
(`iops.a.dtype == float32` straight out of the constructor), one call
before the jit boundary the task-2 docstring describes (Q11 item 2).

**Section 2 (parity figure).** Two-panel matplotlib figure on the same L23
spectrum: Gordon vs `robust_baseline` overlaid (indistinguishable), with
`robust_ztt`/`robust_hybrid` as context curves (genuinely different — +0.3%
to +7.4%, median +4.5%, the point of the integration), and a log-scale
|baseline/Gordon − 1| panel with the 1e-5 gate line and the float32-ε line
drawn in. Printed: **max rel diff 2.31e-7, median 8.17e-8** — 43× inside
the gate, consistent with Q5's task-1 "~3e-7" on synthetic params, now
demonstrated on real L23.

**Section 3 (shape contract).** A genuine 1-D `(3,)`/`(2,)` param pair and
a genuine `(5, nparam)` batch through `robust_ztt`: printed shapes
`(1, 61)` and `(5, 61)` — Q4's real convention, and the Gordon twin run on
the same 1-D input prints `(1, 61)` too, showing it is a shared convention,
not a robust quirk.

**Section 4 (`DomainWarning` live).** `Bp_value=0.005` (task 3's verified
out-of-domain value, vs the emulator's trained lower bound ~0.01026) into
`robust_domain_check` with `rt_backend='robust_hybrid'` under
`warnings.catch_warnings(record=True)`: the full real warning text is in
the executed output ("B_p 100.0% of values outside [0.01026, 0.018], worst
0.005 — 68% of the trained span beyond it. Consider mode='ztt'."). The
immediately following cell runs the *jitted* adapter on the identical
turbid inputs: 0 DomainWarnings, finite Rrs — the silent-under-jit /
warn-and-continue pairing (design §4/Q8) demonstrated as two live cells.

**Section 5 (float32 cost, measured).** The key observation: since
`robust_baseline` computes the identical physics to `calc_Rrs_from_models`,
section 2's residual *is* the float64-vs-float32 difference isolated —
**max 2.31e-7 relative = 1.9 float32 ULPs (median 8.2e-8), max 7.7e-10 sr⁻¹
absolute, 26,124× smaller than a 2% noise floor** (2.0e-5 sr⁻¹ on this
spectrum's median Rrs of 1.0e-3 sr⁻¹). Second, independent measurement:
task 1's inelastic static-path ULP noise reproduced live — fluor-minus-
elastic dips to −4.7e-10 sr⁻¹ at 15/61 wavelengths vs the real +7.1e-5 sr⁻¹
685 nm emission peak (signal/noise 151,980×). All logged as Q11.

**How the outputs were verified as real** (not just exit-code-0 from
`jupyter nbconvert --to notebook --execute --inplace` under ocean14): every
cell was smoke-tested first in a standalone script (same env, same
numbers); after execution the saved `.ipynb` was read back
programmatically — execution_counts are exactly `[1..12]` sequential with
no nulls, every code cell has ≥1 real output, the parity figure's PNG is
embedded (and was extracted and visually inspected); and every numeric
claim in the markdown cells was re-diffed against the corresponding cell's
actual printed output. That re-diff caught one real prose error — a
markdown cell said the parity residual sits "two orders of magnitude" under
the gate when the measured headroom is 43× (~1.6 orders) — fixed to "a
factor of ~40" (markdown-only edit; outputs untouched).

Added: `nb/RT/rob_rt_coding_2.ipynb` (executed, outputs embedded). No
source or test changes — `bing/evaluate.py` and the test suite are exactly
as task 4 left them. Branch `rob_rt`, uncommitted, for JXP's review. Task 6
(updating `rob_rt_prompt_3.md` with what M1 established — the adapter's
exact signature, the lru_cache/jit behavior, and Q6/Q7/Q11's robust.rt
numerics facts) is next.

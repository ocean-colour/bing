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

_(none yet — this prompt has not been executed)_

## Next

→ `rob_rt_prompt_3.md` (M2: fitter dispatch and geometry threading).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

_(none yet)_

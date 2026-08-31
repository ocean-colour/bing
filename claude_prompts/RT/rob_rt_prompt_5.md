# RoB RT Backend Coding — Prompt 5 (M4: Ed wiring and inelastic terms through `robust`)

## Goals

Implement **Milestone M4** of the coding plan
(`docs/coding_plan/rob_rt_coding_plan.md`): feed the robust backend's
inelastic physics with BING's real solar spectrum. `aNWModel.set_raman_Ed`
stashes the raw `(wave_Ed, Ed)` pair (CQ2), the adapter routes it into
`robust`'s `Geometry.Ed`, and the robust-backend Raman/fluorescence terms are
cross-checked against BING's own path at float32 tolerance. After this
milestone `bing.rt.raman`/`chl_fl` are no longer on the recommended path
(their code stays untouched until M5's docstring pointers).

## Claude

### Skills

`.claude/skills/`: `inelastic-rrs` (BING's own Raman/fluorescence wiring —
the comparison side of every gate here; note this skill's own doc gets
updated in M5), `fit-l23-spectrum`, `run-bing-fit`, `code-review`.

### Working agreements

Per the working agreements in `rob_rt_prompt_1.md` (git by JXP; `ocean14`;
CQ1–CQ4 binding; scope discipline; pytest-gated; Fable; log) — **one
correction, carried over from `rob_rt_prompt_2.md`–`rob_rt_prompt_4.md`**:
the branch is JXP's existing **`rob_rt`**, not `rob-rt-backend` as the
coding plan suggested; all of M0–M3 landed there, with JXP reviewing and
committing after each task — assume the same cadence. Milestone-specific
emphasis:

- **CQ2 lands here**: `set_raman_Ed` stashes the raw pair verbatim,
  backward-compatibly — no signature change, `Ed_ratio_raman` bit-identical,
  BING's own Raman path (evaluate.py:167) untouched. Production Ed sourcing
  stays the existing zenith-0° generation; per-pixel-`theta_s` irradiance is
  a design non-goal.
- **`correct_atmosphere` optional-dependency handling (CQ3 sub-point)**: the
  Ed tests import `bing.fitting.l23`, which needs `correct_atmosphere` —
  they live in a **separate `test_evaluate_robust_ed.py`** *(plan choice)*
  added to the `collect_ignore` list (bing/tests/conftest.py:84-87), so
  only that file is dropped in an env without `correct_atmosphere` and
  everything gated in M0–M3 stays runnable. Still no `importorskip` for
  `robust` anywhere.
- **Float32-honest tolerance (CQ1)**: the inelastic cross-checks gate at
  `rtol ≤ 5e-4` — robust's own `1e-6` cross-check holds only under x64,
  which is ruled out.

## Context

Read before coding:

- **Previous prompt** — `rob_rt_prompt_4.md` (M3, **complete**: all 5
  tasks done; full-suite baseline entering M4 is **260 passed, 2 skipped,
  2 failed** — the same 2 pre-existing `test_l23_inelastic.py`
  missing-fixture failures diagnosed in M0/Q10, not a regression;
  re-measured live 2026-08-31 in `ocean14`). Its Q&A/Logs are the record;
  the load-bearing facts for this milestone:
  - **`fit_Bp=False` is the recommended default for M4's inelastic fits
    (M3 Q7 — measured in `nb/RT/rob_rt_coding_4.ipynb`)**: the free-`B_p`
    machinery works end-to-end, but noise, not the likelihood surface,
    dominates the posterior — on bb-bright synthetic data truth is
    recovered decisively (median 0.0202 vs truth 0.02; noiseless CI
    [0.0181, 0.0263]) yet a *single* 0.5%-noise draw more than doubles the
    interval ([0.0058, 0.0245]), and on a **real clear-water L23 spectrum**
    (idx=170, 0.5% noise) the CI is [0.0160, 0.0437] — **60% of the prior
    range**, samples brushing the 0.05 edge. Task 4's robust-vs-BING
    Raman/fluorescence comparisons on L23 should keep `B_p` fixed (the
    default) so the inelastic-term comparison is not smeared by `B_p`
    posterior noise; reserve `fit_Bp=True` for bb-bright water or
    multi-spectrum constraints. (Also from Q7: at long chain lengths one
    stuck emcee walker can own the 5th-percentile edge — exactly 1/16 of
    the samples with 16 walkers — a documented artifact, not a dispatch
    regression.)
  - **Chain-shape/bookkeeping contracts M4's tests can reuse** (line
    numbers re-verified against live source 2026-08-31): the tailed layout
    is `[a_params..., bb_params..., B_p]` whenever `rt_dict['fit_Bp']` is
    True; `init_mcmc(..., rt_dict=None)` (inference.py:202) adds 1 to ndim
    and records `pdict['ndim']`; `prior_bounds`/`init_walkers`
    (inference.py:467/516) extend the clip window from
    `rt_defs.BP_PRIOR_PMIN`/`PMAX` (defs.py:31-32); the module-level
    `UniformPrior` `BP_PRIOR` (inference.py:64) gates `log_prob`;
    `append_Bp_seed` (inference.py:427) tails p0 at `Bp_value` (0.01);
    `chain_param_names(models, rt_dict)` (evaluate.py:102) and
    `plotting.log_param_mask`/`corner_plot` (plotting.py:57/465) append a
    **linear** `'B_p'` (`log_params` False).
  - **The batched-`B_p` shape trap (M3 Q6 — watch for it in any
    per-sample/batched robust forwarding M4 adds)**: the *only* batched
    shape all three robust backends accept is the `(nsamples, nwave)`
    broadcast view — `(nsamples,)` breaks `ztt.bb_tilde`'s broadcast,
    `(nsamples, 1)` breaks the hybrid emulator's `features()`. The correct
    reference implementation is `reconstruct_from_chains`
    (evaluate.py:717): `np.broadcast_to(chains[:, -1:], (nsamples, nwave))`
    — the tail column broadcast read-only across wavelength
    (evaluate.py:806-811). Note that function gained its
    `rt_backend` dispatch + trailing `geom=None` only in M3 (Q6i — it was
    Gordon-only before, silently wrong for robust chains); `io.save_fit`
    and `plotting.show_fits` are the geom-passthrough callers.
  - **Two threads remain open, neither M4's to solve**: M3 Q2 (should
    `validate_rt_dict` reject `robust_baseline`+`fit_Bp=True`? — JXP's
    call) and M2 Q5 (no true M2-level fitter pin exists;
    `bing/tests/files/m3_fixed_bp_pin.npz` is an explicitly **provisional**
    stand-in — portable but fragile across BLAS/emcee versions: if the
    byte-identity tests stay green but its exact values drift, regenerate
    via `gen_m3_fixed_bp_pin.py` rather than suspecting a regression).
- **Coding plan** — `docs/coding_plan/rob_rt_coding_plan.md` **M4** section.
- **Design** — `docs/design/rob_rt_design.md` §3.5 (inelastic terms; the
  `Geometry.Ed` seam), §7.4 (the Ed-wiring open item this milestone closes).
- **Q&A record** — `claude_prompts/rob_rt.md`: Coding Q&A 2 (both
  sub-answers) and Design Q5 (BING's inelastic modules kept, not
  recommended).
- **Current code** — `bing/models/anw.py` (`set_raman_Ed`, anw.py:480-511;
  the `Ed_ratio_raman` field block, anw.py:254-260), `bing/fitting/l23.py`
  (the zenith-0° Ed generation, l23.py:319-322 and :356),
  `bing/tests/conftest.py` (the `collect_ignore` pattern,
  conftest.py:84-87), `bing/evaluate.py` (BING's own Raman call,
  evaluate.py:167); on the robust side `robust/rt/types.py`
  (`Geometry.Ed`, types.py:320-327) and `robust/rt/ed.py` (the internal
  ratio build, ed.py:154-183).

## Prompts

1. Read this doc. Execute the 1st task in the "M4" section below — the
   `set_raman_Ed` stash. If you have any questions, ask me in the Q&A
   section below. Use Fable if you can. Log your work.
2. Read this doc. Execute the 2nd task — routing into `Geometry.Ed`. Check
   my answers in Q&A; if you have additional questions, ask in Q&A. Use
   Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task — the fluorescence guard and the
   conftest isolation. Use Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the explainer notebook. Use Fable
   if you can. Log your work.
5. Read this doc. Execute the 5th task — update `rob_rt_prompt_6.md` with
   what M4 established. Use Fable if you can. Log your work.

## M4

### Tasks

1. **`set_raman_Ed` stash (CQ2).** Add two fields to `aNWModel` beside
   `Ed_ratio_raman` (anw.py:254-260) — e.g. `wave_Ed_raw` / `Ed_raw` — and
   have `set_raman_Ed` (anw.py:480-511) store the incoming pair verbatim
   before computing the ratio exactly as today. Backward compatible: no
   signature change, `Ed_ratio_raman` unchanged, BING's own Raman path
   (evaluate.py:167) untouched. The zenith-0° production generation at
   `fitting/l23.py:319-322` (and :356) keeps feeding `set_raman_Ed` as-is —
   the robust backend just re-uses what lands on the model.

2. **Route into `Geometry.Ed`.** In `calc_Rrs_from_models_robust`, when
   `include_Raman` and the a-model carries the stashed pair, build the
   robust geometry with `Ed=(wave_Ed_raw, Ed_raw)` (types.py:320-327;
   robust builds the ratio internally, robust/rt/ed.py:154-183). When no
   pair is stashed, pass `Ed=None` — robust falls back to its packaged L23
   spectra interpolated in `theta_s`, its documented default (no BING-side
   flat-Ed fallback replicated).

3. **Fluorescence guard + conftest isolation.** `include_Chl_fl` + `phi_C`
   already flow through `Inelastic` (plumbed in M1); assert the required
   `IOPs.a_ph` with a clear error when fluorescence is on but the a-model
   has no `a_ph` set (mirrors `set_aph` conventions). Put the Ed tests in
   the new `bing/tests/test_evaluate_robust_ed.py` and add that file to the
   `correct_atmosphere` `collect_ignore` list (conftest.py:84-87) — nothing
   gated in M0–M3 may require `correct_atmosphere`.

4. **Notebook.** `nb/RT/rob_rt_coding_5.ipynb` (executed, with outputs):
   the Ed seam end-to-end — the raw pair stashed on the model, the robust
   Raman term with the pair vs `Ed=None` (show the difference is real), the
   robust-vs-BING Raman and fluorescence comparisons on L23 at the float32
   tolerance, and a short note on why `5e-4` and not `1e-6`.

5. **Finally.** Update `rob_rt_prompt_6.md` (M5) with what M4 established —
   the final state of the integration entering the docs/benchmark
   milestone, measured inelastic agreement numbers, and anything the M5
   skill-doc updates should say.

### Gate

`bing/tests/test_evaluate_robust_ed.py` (dropped, whole-file, in an env
without `correct_atmosphere`) + additions to `test_evaluate_robust.py` where
independent of it:

1. After `set_raman_Ed`, the raw pair is stored and `Ed_ratio_raman` is
   bit-identical to the pre-change value.
2. With the same Ed spectrum, `robust_baseline`+Raman vs `gordon`+Raman on
   L23 agree at float32 tolerance (`rtol ≤ 5e-4` — the physics is a pinned
   port; robust/tests/test_inelastic_bing_xcheck.py holds `1e-6` only under
   x64); same comparison for fluorescence with matched `phi_C`.
3. Passing an Ed pair vs `Ed=None` measurably changes the robust Raman term
   (the seam is live).
4. Fluorescence-on without `a_ph` raises the clear error.
5. The suite still collects cleanly in an env without `correct_atmosphere`
   (the Ed test file is dropped, nothing else).

Existing suite green throughout.

## Q&A

**Q1 (task 1, Claude → JXP). REAL OPEN QUESTION — needs your answer before
the Gate-item-2 tests are written (task 2/3) and before task 4's notebook.**
This milestone's **Gate item 2 as literally written is currently impossible
to satisfy**: it asks that "`robust_baseline`+Raman vs `gordon`+Raman on L23
agree at float32 tolerance", but M1's Q2 (`rob_rt_prompt_2.md`, Q2) found
`robust.rt.baselines.Rrs_gordon` takes **no `inelastic` argument at all**
and resolved it by making the adapter raise. Verified live against current
source: `_build_robust_inputs` (evaluate.py:438-444) still raises
`ValueError` ("...'robust_baseline' has no inelastic composition path...")
whenever `rt_backend == 'robust_baseline'` is combined with `include_Raman`
or `include_Chl_fl` — nothing in M2/M3 touched that guard. So
`robust_baseline`+Raman produces an immediate `ValueError`, not a
comparable Rrs. **What did Gate item 2 intend?** Most likely candidates:
(a) it should read `robust_ztt` (and/or `robust_hybrid`) + Raman vs
`gordon`+Raman — the backends that *do* have the `Inelastic` composition
path; or (b) the M1 guard itself is to be revisited in M4 (e.g. compose
inelastic terms around `Rrs_gordon` BING-side). I did **not** guess: the
guard is untouched and no Gate-item-2 test exists yet. This does not block
task 1 (the stash is independent of backend dispatch, done below) and
likely not task 2's routing either, but it gates how the milestone's
cross-check tests and the task-4 notebook comparisons are framed.

**Q2 (task 1, Claude → JXP).** Minor citation drift, resolved, no answer
needed: the doc's "BING's own Raman path (evaluate.py:167)" points into
`calc_Rrs_from_models`' docstring in current source; the actual
`Ed_ratio_raman` consumption is evaluate.py:209 (the `getattr` + flat-Ed
fallback warning) feeding the `calc_Rrs(..., Ed_ratio=...)` call at
evaluate.py:239. That path is untouched by task 1 (git diff confirms
`evaluate.py` unmodified). The anw.py citations (254-260, 480-511) and
l23.py citations (319-322, :356 — the `calc_Rrs` call consuming
`Ed_ratio_raman` is actually l23.py:358) were accurate at the pre-edit
state.

**Q3 (task 1, Claude → JXP).** "Verbatim" implemented as **uncopied
references**: `set_raman_Ed` stores the caller's own array objects on
`wave_Ed_raw`/`Ed_raw` (`is`-identity, no `np.asarray`, no copy), matching
the spec's wording and the cheapest contract for task 2's pass-through into
`Geometry.Ed`. Consequence: a caller mutating its Ed array in place after
the call mutates the stash (but *not* `Ed_ratio_raman`, computed eagerly) —
same aliasing behavior as elsewhere in BING (e.g. `set_aph`), so no answer
needed; flagging only so task 2 knows the stash is a live reference.

**Q4 (task 2, Claude → JXP). Resolved finding, no answer needed — does
fluorescence need the Ed routing too?** The task-2 spec conditions the
routing on `include_Raman` only, and that is exactly what was implemented.
But checked, per the spec's implicit question: **robust's fluorescence
kernel does consume `Geometry.Ed`** — `fluorescence_kernel` builds
`ed_ex`/`ed_em` with `override=geometry.Ed` (robust/rt/inelastic.py:399-400;
`cdom_fluorescence_kernel` likewise, cdom_fl.py:377-378), weighting the
excitation integral by the true `Ed(λ')` and normalizing by `Ed(λ)` — so
the Ed *shape* changes `K_fl`, not just the Raman term. Consequences of the
as-specced `include_Raman`-only gate: (a) when **both** inelastic flags are
on, the one shared `Geometry` means the stashed sky feeds fluorescence too
— desirable, and robust's `ed` module guarantees numerator and denominator
come from the same sky by construction; (b) a **fluorescence-only** call
(`include_Chl_fl=True`, `include_Raman=False`) keeps `Ed=None` → robust's
packaged-L23 default, even with a pair stashed. Deliberately **not**
extended to case (b): the spec is explicit about `include_Raman`; the stash
setter is Raman-named (`set_raman_Ed`); and BING's own fluorescence path
sources its Ed independently via `init_Chl_fluorescence(Ed=...)` (a
model-grid Ed, a different contract), so silently rerouting the Raman stash
into fluorescence-only calls would invent untested behavior with no BING
counterpart. If wanted later it is a one-line condition change
(`include_raman or include_fl` at the routing site) plus tests. Does not
block task 3 (the `a_ph` guard and conftest isolation are independent of
this); Q1 remains the only real open question.

**Q5 (task 3, Claude → JXP). Resolved finding, no answer needed — was
task 1's test misplaced?** Re-audited whether
`test_set_raman_ed_stashes_raw_pair` (and task 2's two tests) actually
depend on `correct_atmosphere` and therefore belonged in the new
`test_evaluate_robust_ed.py` rather than `test_evaluate_robust.py`. **No —
all three are correctly placed.** Read each test body, not the logs: task
1's test uses only a synthetic 152-point pair plus `scipy.interpolate.
interp1d` for the independent ratio recomputation; the task-1 log's
mention of "the production-style zenith-0° path" describes a *one-off
pre-/post-edit bit-identity verification run while the change landed*,
referenced in the test's docstring but never executed by the test itself
(no `correct_atmosphere`/`bing.fitting.l23` import anywhere in
`test_evaluate_robust.py` — verified against the module's full import
list). Task 2's routing/seam tests likewise use purely synthetic
exponential skies. Nothing moved; the Gate preamble's independence rule
was already being followed.

**Q6 (task 3, Claude → JXP). Resolved finding, no answer needed — where
the `a_ph` guard can and cannot fire.** The guard sits *after*
`eval_a` in `_build_robust_inputs`, deliberately: free-Chl Bricaud models
(`ExpBricaud`, `Bricaud`, `ExpBricaudFree`; `fix_Chl=False`) set `a_ph`
implicitly inside `eval_anw` (anw.py:389), so a fresh such model +
fluorescence works today without an explicit `set_aph` call and a
pre-`eval_a` guard would falsely reject it. Consequently the guard fires
exactly for a-models whose `eval_anw` never touches `a_ph` (`Exp`,
`ExpFix`, `Cst`, `Every`, `ExpNMF`) — previously a bare
`TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`
(measured pre-guard). One adjacent case is *out of this guard's reach by
construction*: `ExpBricaudFix` (`fix_Chl=True`) with `a_ph` unset dies
inside `eval_anw` itself (anw.py:391, the same bare TypeError) *before*
the adapter reads `a_ph` — on the Gordon path too, a pre-existing
model-level behavior shared by every backend (the `debug-priors` skill
already documents "missing set_aph" as a known trap), so hardening
`eval_anw` itself would be a separate, backend-agnostic change outside
task 3's adapter scope. Not blocking anything; flagged for completeness.

## Next

→ `rob_rt_prompt_6.md` (M5: deprecation notes, docs, throughput benchmark).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-31 (M4 task 1 — set_raman_Ed stash)

**Read/verified first.** The doc's anw.py citations were still accurate at
the pre-edit state (`Ed_ratio_raman` field block anw.py:254-260;
`set_raman_Ed` anw.py:480-511 — first M4 file, untouched by M1-M3, so no
drift), as were l23.py:319-322/:356 (zenith-0° generation; the
`Ed_ratio=models[0].Ed_ratio_raman` consumption is l23.py:358) and
conftest.py:84-87 (`collect_ignore`). The evaluate.py:167 citation drifted
into a docstring — actual Raman-path consumption is evaluate.py:209/239
(Q2). Cross-read M1's Q2 in `rob_rt_prompt_2.md` and current
`_build_robust_inputs`: **the `robust_baseline`+inelastic `ValueError`
guard is still live at evaluate.py:438-444, which makes this milestone's
Gate item 2 as literally written impossible — flagged as Q1, needs JXP
before the Gate-item-2 tests/notebook comparisons are built.**

**Implemented** (bing repo, `rob_rt` branch, 93 insertions, 2 files, both
purely additive; `evaluate.py`, `l23.py`, robust untouched):

- `bing/models/anw.py` — two new `aNWModel` fields directly beside
  `Ed_ratio_raman`: `wave_Ed_raw` (anw.py:262) and `Ed_raw` (anw.py:270),
  both `None`-defaulted with docstrings matching the existing field style.
  `set_raman_Ed` (now anw.py:496) stashes the incoming pair verbatim —
  uncopied references, Q3 — at anw.py:539-540, *before* the ratio
  computation, which is byte-identical to before (anw.py:541-545; the
  spec's "store before computing" order means a `bounds_error` raise from
  a too-narrow Ed grid leaves the stash set but `Ed_ratio_raman` not —
  same partial-state behavior class as any mid-method raise, accepted as
  the spec's literal instruction). No signature change. Docstring gained a
  stash paragraph and a Sets section.

- `bing/tests/test_evaluate_robust.py` — new section "M4 task 1" with
  `test_set_raman_ed_stashes_raw_pair` (test file line 1754) covering Gate
  item 1: fields `None` on a fresh model; after `set_raman_Ed` the exact
  input arrays are stored (`is`-identity, stronger than `array_equal`);
  `Ed_ratio_raman` bit-identical (`np.array_equal`, not `allclose`) to an
  independent inline recomputation of the unchanged formula (linear
  `interp1d`, `bounds_error=True`); a second call replaces the stash.
  Placed here, not the future `test_evaluate_robust_ed.py`, because it
  needs neither `correct_atmosphere` nor even `robust` — the Gate
  preamble's "additions to `test_evaluate_robust.py` where independent"
  rule. Module docstring's coverage list extended accordingly.

**Bit-identity verified empirically, not just by inspection**: captured
`Ed_ratio_raman` from the *pre-edit* code via (a) the production-style
zenith-0° path (`correct_atmosphere.downwelling.downwelling_irradiance`
on the l23.py:325-327 grid recipe, `ExpBricaud`+`Pow` on a 61-point
400-700 nm grid) and (b) a synthetic 3-point pair; re-ran post-edit:
`np.array_equal` **True for both**, and the stashed arrays are the very
objects passed in (`is` → True twice).

**Tests** (`ocean14`): `test_evaluate_robust.py` alone 80 passed. Full
suite **261 passed, 2 skipped, 2 failed** vs the M3-exit baseline of 260
passed, 2 skipped, 2 failed — exactly +1 (the new test), and the 2
failures are the same pre-existing `test_l23_inelastic.py` missing-fixture
failures (M0/Q10), not a regression.

**Next**: task 2 — route the stashed pair into `Geometry.Ed` in
`calc_Rrs_from_models_robust`/`_build_robust_inputs` (`Ed=(wave_Ed_raw,
Ed_raw)` when stashed and `include_Raman`; `Ed=None` fallback to robust's
packaged spectra). Nothing consumes the new fields yet. JXP's Q1 answer
shapes the Gate-item-2 cross-check tests that follow.

### 2026-08-31 (M4 task 2 — routing into Geometry.Ed)

**Read/verified first.** The routing site is `_build_robust_inputs`
(evaluate.py:396 pre-edit; the doc's `calc_Rrs_from_models_robust` wording
resolves there — M1/M3's factoring puts all argument construction,
including `geom.to_robust()`, in the helper, previously the bare
no-`Ed` call at evaluate.py:460). Robust-side citations both still
accurate: `Geometry.Ed` is types.py:301-334 (the doc's 320-327 covers the
attribute block; the field is a `(wave_Ed, Ed)` tuple of 1-D arrays,
`None`-defaulted, and `validate()` at types.py:392-408 enforces the
2-tuple/1-D/strictly-increasing contract), and the internal ratio build is
`ed.ratio` (ed.py:155-183, calling `ed.Ed` with `override=geometry.Ed`;
consumed by `raman_correction_factor` at inelastic.py:219). Verified the
`Ed=None` fallback is genuinely robust's default, nothing to implement:
`ed.Ed` (ed.py:96-153) loads the packaged L23 anchor table and
interpolates linearly in `theta_s` between 0/30/60 deg (clamped outside)
whenever `override is None`. BING-side, `ObsGeometry.to_robust(Ed=None)`
(bing/rt/geometry.py:38) already had the pass-through keyword (built in
M1 for exactly this task), so the adapter change is the condition alone.

**Implemented** (bing repo, `rob_rt` branch; evaluate.py +42/-6 lines,
test file +115; robust untouched):

- `bing/evaluate.py` — the routing in `_build_robust_inputs`, now
  evaluate.py:463-478: when `include_raman` and
  `a_model.wave_Ed_raw is not None` (the stash-presence convention from
  task 1 — fields are `None`-defaulted dataclass-style attributes, and
  `set_raman_Ed` always sets both together, so one check suffices),
  `geometry = geom.to_robust(Ed=(a_model.wave_Ed_raw, a_model.Ed_raw))`
  — the uncopied Q3 references flow straight through to `Geometry.Ed`.
  Otherwise the exact pre-M4 call `geom.to_robust()` (Ed=None), so
  elastic-only and no-stash callers are byte-for-byte unaffected (the
  `Geometry` pytree structure is also unchanged for them — no jit
  retrace). Docstrings updated: `_build_robust_inputs`'s "(no ``Ed`` --
  M4's job)" phrase replaced (evaluate.py:404-408), and
  `calc_Rrs_from_models_robust` gained an "Ed routing (M4 task 2)" Notes
  paragraph (evaluate.py:585-597) plus a pointer in the rt_dict key list
  (evaluate.py:534-537).

- **The fluorescence question (Q4)**: robust's `fluorescence_kernel`
  *does* consume `Geometry.Ed` (inelastic.py:399-400) — so with both
  flags on the stashed sky feeds both terms through the one shared
  geometry (desirable; one sky by construction), but a
  fluorescence-only call keeps the packaged default. Deliberately not
  extended beyond the spec's `include_Raman` gate — see Q4 (resolved
  finding, not blocking task 3).

- `bing/tests/test_evaluate_robust.py` — new section "M4 task 2" (line
  1811): `test_build_robust_inputs_routes_stashed_ed_pair` (line 1822;
  the condition truth table — Raman-on/no-stash → `Ed is None`,
  Raman-on/stash → the very array objects on `Geometry.Ed`
  (`is`-identity, extending task 1's verbatim contract through the
  adapter), Raman-off/stash → `Ed is None`) and
  `test_robust_raman_ed_pair_vs_none_changes_rrs` (line 1859; **Gate
  item 3**). Placed here, not the future `test_evaluate_robust_ed.py`,
  per the Gate preamble's independence rule: both use synthetic Ed pairs
  and need no `correct_atmosphere` (the module docstring's task-1
  pointer saying the routing tests would live in the `_ed` file was
  amended accordingly — that file remains task 3's, for the
  `correct_atmosphere`-dependent generation and Gate-item-2
  cross-checks). Also re-checked the one pre-existing robust+Raman+stash
  test (`test_calc_Rrs_from_models_robust_raman_branch`, line ~404): it
  now routes its flat-Ed stash through the seam — its finite/positive
  assertions still pass (numbers moved from packaged-L23 to
  flat-override, plausibility unchanged).

**Gate item 3 measured (the seam is live).** `robust_ztt` + Raman,
ExpBricaud+Pow on the 61-point 400-700 grid, `_PARAM_SETS[0]`
(Chl=1.0), theta_s=30 deg, B_p=0.014: packaged-L23 default (no stash) vs
a stashed steep exponential sky `Ed = exp((wave-550)/150)` on
340-760 nm — **max relative Rrs difference 6.34e-2 (6.3%, at 690 nm),
mean 3.73e-2, max absolute 1.05e-4 sr^-1** — five orders of magnitude
above float32 ULP noise (~1e-7); the test gates at >1e-4. The steep
slope suppresses Ed(lambda')/Ed(lambda) (lambda' blueward) well below
the solar-shape ratio, hence the Raman term drops and the red end moves
most. The same test also pins the flag gating bitwise: with
`include_Raman=False` a stashed pair changes nothing
(`np.array_equal` against the pre-stash elastic call).

**Tests** (`ocean14`): `test_evaluate_robust.py` alone 82 passed (was
80). Full suite **263 passed, 2 skipped, 2 failed** vs task 1's exit
baseline of 261/2/2 — exactly +2 (the two new tests), and the 2
failures are the same pre-existing `test_l23_inelastic.py`
missing-fixture failures (M0/Q10), not a regression.

**Next**: task 3 — the fluorescence `a_ph` guard (assert `IOPs.a_ph`
with a clear error when `include_Chl_fl` is on but the a-model has no
`a_ph`; note robust's own `fluorescence_kernel` already raises a good
`ValueError` at inelastic.py:385-392, but only at trace time — the
BING-side guard should fire earlier, in `_build_robust_inputs`, where
`a_model.a_ph` is read at evaluate.py:453) + `test_evaluate_robust_ed.py`
and its `collect_ignore` entry (conftest.py:84-87). Q1 (Gate item 2's
`robust_baseline` framing) is still the only real open question.

### 2026-08-31 (M4 task 3 — fluorescence guard + conftest isolation)

**Read/verified first.** The `include_Chl_fl`/`phi_C` plumbing through
`Inelastic` is untouched since M1, as expected — `_build_robust_inputs`
reads the flags (pre-edit evaluate.py:438-439), builds
`a_ph = (10**a_params[..., -1:]) * a_model.a_ph if include_fl else None`
(pre-edit evaluate.py:456) into `IOPs.from_total_bb(..., a_ph=a_ph)`, and
`phi_C` rides out in the `_RobustInputs` tuple. `aNWModel.a_ph` is a
`None`-defaulted field (anw.py:283), matching the `set_aph` convention
(anw.py:771, on `aNWBricaud`: sets `self.a_ph`; unset means `None`).
**Measured the pre-guard failure mode live** (`ocean14`): `Exp`+`Pow`
with `include_Chl_fl=True` dies at the a_ph multiplication with the bare
`TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'` —
no BING-side guard existed anywhere (BING's own Gordon fluorescence path,
evaluate.py:256, has the identical unguarded multiplication; robust's
`fluorescence_kernel` ValueError at inelastic.py:385-392 fires only at
jit trace time). Key placement fact: free-Chl Bricaud models set `a_ph`
*implicitly inside `eval_anw`* (anw.py:389, `fix_Chl=False`), so the
guard must sit after `eval_a`, not before — see Q6 for the full
can/cannot-fire map (including `ExpBricaudFix`'s pre-existing
model-level TypeError, out of adapter scope). Also re-audited the
placement of task 1/2's tests against their actual imports and bodies —
**nothing was misplaced, nothing moved** (Q5): task 1's production-path
mention is a one-off dev verification described in a docstring, not
executed test code; `test_evaluate_robust.py` imports neither
`correct_atmosphere` nor `bing.fitting.l23`.

**Implemented — piece 1, the guard** (bing repo, `rob_rt` branch;
evaluate.py):

- `bing/evaluate.py` — in `_build_robust_inputs`, immediately before the
  a_ph multiplication (now evaluate.py:464-472, comment block from :456):
  `include_fl` with
  `a_model.a_ph is None` raises a `ValueError` naming the flag
  (`include_Chl_fl`), the missing piece (`a_model.a_ph is None`), the
  physics (the source term is `phi_C * a_ph`; wording borrowed from
  robust's own `fluorescence_kernel` error, "bulk absorption cannot stand
  in for the phytoplankton component", for cross-backend consistency),
  and the fix (`call a_model.set_aph(Chl) first`, or disable
  `include_Chl_fl`). Docstrings updated: `_build_robust_inputs` now owns
  "four argument-validity errors" (evaluate.py:409-412), and
  `calc_Rrs_from_models_robust`'s Raises section gained the new case
  (evaluate.py:585-588).

- `bing/tests/test_evaluate_robust.py` — new section "M4 task 3" (line
  1919): `test_robust_fluorescence_without_aph_raises_clear_error` (line
  1930; **Gate item 4**: `Exp`+`Pow` — a model whose `eval_anw` never
  touches `a_ph` — raises the specific ValueError matching
  `a_ph.*set_aph` and naming `include_Chl_fl`, from
  `calc_Rrs_from_models_robust` *and* from a direct
  `_build_robust_inputs` call, i.e. eagerly, pre-jit; plus the same
  a_ph-less model stays fully usable elastic-only) and
  `test_robust_fluorescence_with_aph_set_does_not_raise` (line 1972; the
  negative/regression case: ExpBricaud + `set_aph` builds a non-None
  `IOPs.a_ph` and returns finite positive Rrs). Placed here per the
  independence rule — no Ed, no `correct_atmosphere` involved. Module
  docstring coverage list extended.

**Implemented — piece 2, conftest isolation**:

- `bing/tests/test_evaluate_robust_ed.py` — **new file**. Since nothing
  existing needed to move (Q5) and Gate item 2's cross-checks stay
  blocked on Q1, it starts with the one genuinely
  `correct_atmosphere`-dependent test this task adds:
  `test_production_ed_routes_and_changes_robust_raman` — the *exact*
  `fitting/l23.py:325-328` zenith-0° recipe
  (`downwelling.downwelling_irradiance` on the wave_ex-covering 1-nm
  grid) stashed via `set_raman_Ed`, routed verbatim (`is`-identity on
  `Geometry.Ed`) and producing finite positive robust_ztt+Raman Rrs that
  differs from the packaged-L23 default — **measured max relative
  difference 1.5e-2 at 505 nm, mean 4.2e-3** (the real zenith-0° sky vs
  the packaged anchor table at theta_s=30°); gated at >1e-3. The module
  docstring records that Gate item 2's cross-checks belong here when Q1
  is answered. `from correct_atmosphere import downwelling` sits at
  module top level — the import that makes the whole-file drop real.

- `bing/tests/conftest.py` — `'test_evaluate_robust_ed.py'` appended to
  the existing `collect_ignore` conditional (now conftest.py:84-88),
  following the exact `_module_importable('correct_atmosphere')` pattern.

**Gate item 5 verified without breaking the env** (no uninstall):
`_module_importable` does a bare `__import__('correct_atmosphere')`
catching `Exception`, so absence was simulated by a throwaway shadowing
stub — a scratchpad `correct_atmosphere.py` whose body is a single
`raise ImportError(...)`, prepended via `PYTHONPATH` (shadows
site-packages; the import then fails exactly as in an env lacking the
package, driving the same conftest code path). With the stub:
`pytest bing/tests --collect-only -q` → **228 tests collected, zero
collection errors**, and all four ignored files
(`test_evaluate.py`/`test_io.py`/`test_l23_fitting.py`/
`test_evaluate_robust_ed.py`) absent from the listing. Without the stub:
270 collected, `test_evaluate_robust_ed.py` present. The stub lives only
in the session scratchpad — nothing permanent, `ocean14` untouched.

**Tests** (`ocean14`): the two robust files together 85 passed (was 82).
Full suite **266 passed, 2 skipped, 2 failed** vs task 2's exit baseline
of 263/2/2 — exactly +3 (the two guard tests + the production-Ed test),
and the 2 failures are the same pre-existing `test_l23_inelastic.py`
missing-fixture failures (M0/Q10), not a regression.

**Next**: task 4 — the explainer notebook `nb/RT/rob_rt_coding_5.ipynb`
(the Ed seam end-to-end, pair-vs-None difference, and the robust-vs-BING
inelastic comparisons at float32 tolerance). **Reminder: Q1 still needs
JXP's answer first** — the notebook's `robust_baseline`+Raman comparison
as originally worded is impossible (M1's ValueError guard, still live),
so JXP must pick the reframing (likely `robust_ztt`/`robust_hybrid` vs
`gordon`) before that comparison and the Gate-item-2 tests are built.

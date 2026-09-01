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

**Q7 (task 4, Claude → JXP). Resolved finding, no answer needed — does
Q4's fluorescence Ed-routing gap (the `include_Raman`-only condition)
actually explain the robust-vs-BING fluorescence disagreement?** Measured
in `nb/RT/rob_rt_coding_5.ipynb` section 4: **no, not meaningfully.** On
L23 idx=170 (PACE grid), the "as-specced" fluorescence-only comparison
(`include_Raman=False`, so per Q4 robust's `fluorescence_kernel` reads
`Geometry.Ed=None` — the packaged-L23 default — while BING's own
`init_Chl_fluorescence` uses the real zenith-0 production Ed) measured max
relative disagreement 9.18% (mean 5.77%). Re-running with
`include_Raman=True` too (so the Q4 routing condition fires and the
*same* stashed sky reaches both backends' fluorescence kernels — an
Ed-matched combined Raman+fluorescence check) measured max 18.49% (mean
7.21%) — **larger, not smaller**. So the Ed-source mismatch Q4 flagged is
not the dominant contributor to the fluorescence-path disagreement; the
excitation-grid approximation gap M1's Q1 identified for Raman
(interpolating/clamping the emission-grid spectrum, vs BING's true
parametric evaluation at the wider excitation grid) evidently dominates
for fluorescence too — unsurprising since `inelastic.py`'s
`fluorescence_kernel` builds its excitation grid the same
interpolate/clamp way. Consequence: extending Q4's routing condition to
fluorescence-only calls (the one-line change Q4 flagged as available)
would not, on its own, bring Gate item 2's fluorescence cross-check
anywhere near `rtol <= 5e-4` — the real gap is the physics composition,
not the sky source. Not blocking anything; sharpens what Q1's eventual
answer needs to account for (a `robust_baseline`-based Gate item 2 would
have the *same* problem, since M1's Q2 already found `Rrs_gordon` has no
inelastic path at all — this is orthogonal to that, a fact about
`robust_ztt`/`robust_hybrid`'s inelastic kernels specifically).

**Q8 (task 4, Claude → JXP). Resolved finding, no answer needed — the
robust-vs-BING inelastic agreement, measured on real data instead of
synthetic fixtures.** `nb/RT/rob_rt_coding_5.ipynb` sections 3-4 are the
first time this milestone's Raman/fluorescence cross-check has been run
on a real L23 spectrum with a real production Ed spectrum on both sides
(task 1-3's tests use either synthetic Ed pairs or synthetic IOP fixtures,
never both real together). Measured, honestly, at rtol: Raman max 11.37%
/ mean 5.64% (worst at 400 nm); fluorescence max 9.18% / mean 5.77%
(mismatched-Ed case) or max 18.49% / mean 7.21% (Ed-matched case) — all
many orders of magnitude outside the milestone's `rtol <= 5e-4` working
tolerance, consistent with (and roughly the same order of magnitude as)
M1's Q1 finding on synthetic fixtures. Nothing here changes Q1's
open status — if anything it strengthens the case that Gate item 2 as
literally written (whichever backend it ends up naming) cannot pass at
`5e-4` for the inelastic terms specifically, only for the elastic path
(which M1 already measured at ~2.3e-7, comfortably inside). Flagging so
JXP's eventual Q1 answer can be made with this number in hand: **if the
intent was ever for Gate item 2 to gate the inelastic comparison at
`5e-4` too, that bar is not reachable with robust's current
interpolated/clamped excitation-grid kernels** — only a tolerance
specific to the physics-approximation gap (or a scope change to what
Gate item 2 actually checks) would close it.

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

### 2026-08-31 (M4 task 4 — Ed seam explainer notebook)

**Built** `bing/nb/RT/rob_rt_coding_5.ipynb` (5 sections, 6 code cells, 9
markdown cells; no source files touched — this task is notebook-only).
Per Q1's still-open status, every robust-vs-BING comparison uses
`rt_backend='robust_ztt'` rather than `'robust_baseline'`, called out
explicitly in the intro markdown cell rather than substituted silently.

- **Section 1 (the stash).** A fresh `ExpBricaud`+`Pow` model
  (61-band, 400-700 nm, matching the `test_evaluate_robust*.py`
  fixtures), Chl=1 via `set_aph`. The Ed pair is BING's real zenith-0
  production recipe (`correct_atmosphere.downwelling.downwelling_irradiance`
  on the exact `fitting/l23.py:319-328` grid, not a synthetic
  stand-in — `correct_atmosphere` is a fine dependency for a notebook,
  per the doc's own suggestion). Measured: `wave_Ed_raw`/`Ed_raw` are
  `None` before the call; after `set_raman_Ed`, both hold the *exact*
  input array objects (`is`-identity True for both), `array_equal` also
  True, `Ed_ratio_raman` computed to the unchanged (61,)-shape formula.
  wv_Ed grid measured 347-705 nm (359 pts); Ed 52.65-189.55 mW cm^-2
  um^-1 (TSIS-1 HSRS spectrum, the package's default source).

- **Section 2 (pair vs `Ed=None`, seam is live).** Same IOPs
  (`_PARAM_SETS[0]`-style: Chl=1, Adg/Sdg/Bnw/beta fixed), `robust_ztt`
  + Raman, $\theta_s=30\degree$, computed once with a fresh unstashed
  model (`Ed=None`, robust's packaged-L23 default) and once with
  section 1's stashed real production Ed. **Measured max relative Rrs
  difference 1.538e-2 (1.54%) at 505 nm, mean 4.209e-3 (0.42%), max
  absolute 4.47e-5 sr^-1** — ~150,000x the ~1e-7 float32 ULP noise
  floor, a fresh, independent measurement (not task 2's 6.3%
  deliberately-steep-sky number) that happens to land close to task
  3's own production-Ed test (1.5e-2 at 505 nm, mean 4.2e-3) — expected,
  since both use a real zenith-0 sky at similar theta_s and IOPs; this
  notebook's number was measured fresh, not copied.

- **Section 3 (robust-vs-BING Raman on real L23, float32 tolerance).**
  L23 idx=170 (Chl=0.1306 mg/m^3), PACE grid, via
  `fit_l23.prep_one_l23(..., include_Raman=True)` — which itself runs
  the production Ed recipe and stashes it on `models_l23[0]`, so BING's
  Gordon+Raman path (`Ed_ratio_raman`) and `robust_ztt`+Raman
  (`Geometry.Ed` routed from the same stash) consume the identical
  sky. **Measured max relative disagreement 1.1368e-01 (11.37%) at
  400 nm, mean 5.641e-02 (5.64%) — does NOT satisfy `rtol <= 5e-4`**,
  reported plainly rather than reframed. Consistent with M1's Q1
  finding: robust's excitation-grid IOPs are an interpolated/clamped
  emission-grid spectrum, not the true parametric evaluation at
  `wave_ex` that BING's own path does.

- **Section 4 (fluorescence, matched `phi_C`).** Same L23 model/params;
  `a_ph` already set (implicit `eval_anw` side effect for free-Chl
  `ExpBricaud`, confirmed live — no explicit `set_aph` call needed,
  task 3's guard never fires here); BING's `init_Chl_fluorescence`
  fed the same zenith-0 recipe evaluated on the model grid. `phi_C=0.02`
  matched both sides. **Mismatched-Ed case** (`include_Raman=False`,
  so per Q4 robust's fluorescence kernel falls back to the packaged-L23
  default while BING uses the real production Ed): max relative
  disagreement **9.178e-02 (9.18%) at 450 nm, mean 5.770e-02 (5.77%)**
  — does NOT satisfy `rtol <= 5e-4`. Added a second, Ed-matched check
  (`include_Raman=True` too, so the stash reaches both kernels, per Q4):
  max **1.849e-01 (18.49%)**, mean **7.207e-02 (7.21%)** — *larger*,
  not smaller, showing the Q4 Ed-source mismatch is not the dominant
  driver (new finding, logged as **Q7**); the excitation-grid
  interpolation gap (Q1-M1) evidently dominates for fluorescence too.
  Logged the combined real-data measurement as **Q8** for JXP's
  eventual Q1 answer to account for.

- **Section 5 (why `5e-4`, not `1e-6`).** Markdown-only, tied explicitly
  to sections 3-4's own numbers: float32-vs-float64 alone costs only
  ~2.3e-7 relative on the elastic path (M1's own measurement, cited from
  `rob_rt_prompt_3.md`), four orders of magnitude below `5e-4`; the
  actual inelastic disagreement measured above (several *percent*) is
  overwhelmingly a physics-composition gap (M1 Q1), not numerical noise
  — so `5e-4` is the honest float32 floor for the parts of this
  milestone that really are float32 JAX-vs-JAX agreement (the elastic
  path; the pair-vs-`Ed=None` seam check, whose real differences sit
  far above `5e-4` in the other direction), not a bar that was ever
  going to make the Raman/fluorescence robust-vs-BING cross-check pass.

**Execution verified for real.** Ran
`jupyter nbconvert --to notebook --execute --inplace` (`ocean14`); all 6
code cells have sequential `execution_count` 1-6 and non-empty outputs,
zero error outputs. Re-diffed every numeric claim in the 9 markdown
cells against the executed cells' actual printed output before writing
this log entry — sections 2-4's markdown deliberately refers to
"the max/mean relative error printed above" rather than restating
digits inline (avoiding the recurring prose-drift failure mode
entirely for those cells); the one markdown cell with an explicit
qualitative comparison ("larger magnitude, not smaller") was checked
against the actual printed numbers (9.18%/5.77% vs 18.49%/7.21% —
holds). Also smoke-tested every cell's logic standalone as a plain
script before building the `.ipynb` (a fragile-import class, given
`correct_atmosphere` and `robust_ztt` jit compilation) — full notebook
execution then took ~3.4s wall-clock (all `robust_ztt`, no emulator
load).

**Suite unaffected** (no source files touched by this task): re-ran the
full suite anyway as a sanity check — **266 passed, 2 skipped, 2
failed**, identical to task 3's exit baseline, same 2 pre-existing
`test_l23_inelastic.py` failures (M0/Q10).

**Next**: task 5 — update `rob_rt_prompt_6.md` (M5) with M4's final
state: the Ed-seam integration is complete and gated on Q1's answer for
how Gate item 2 should be framed; the measured robust-vs-BING inelastic
agreement (Raman ~6-11%, fluorescence ~6-18% depending on Ed matching)
is a real physics-composition gap, not a numerical one, and M5's
skill-doc updates (the `inelastic-rrs` skill in particular) should say
plainly that `bing.rt.raman`/`chl_fl` remain the higher-fidelity choice
when the excitation-grid approximation matters, with `robust_ztt`/
`robust_hybrid` as the not-yet-precisely-matching alternative.

### 2026-08-31 (M4 task 5 — handing off to `rob_rt_prompt_6.md`)

**M4 is now fully complete (all 5 tasks done).** Updated
`rob_rt_prompt_6.md`'s **Working agreements** and **Context** sections
only (no Q&A/Logs/Tasks/Gate/Definition-of-done touched — those stay
JXP's plan; no source files touched).

- **Working agreements**: fixed the stale `rob-rt-backend` branch
  reference to `rob_rt`, one line, the same correction every prior prompt
  in this series (2–5) has carried forward.
- **Context**, added as a new lead bullet, ahead of everything else,
  titled "READ THIS FIRST": the measured robust-vs-BING inelastic
  agreement numbers from Q7/Q8 (Raman 11.37% max / 5.64% mean; fluorescence
  9.18%/5.77% mismatched-Ed or 18.49%/7.21% Ed-matched), cited to
  `nb/RT/rob_rt_coding_5.ipynb` §§3-4, stated as several orders of
  magnitude outside `rtol <= 5e-4` and rooted in a genuine physics-
  composition difference (excitation-grid interpolation/clamping vs true
  parametric evaluation), not numerical noise. Made explicit that M5
  task 2's current verbatim wording — "the robust inelastic path is now
  the recommended one" — cannot be written as-is without accounting for
  this gap; did **not** decide what the "recommended path" note should
  actually say, only surfaced the number so it can't be missed.
- **Context**, second bullet: Q1 is still open, and Q8's finding gives it
  extra weight — even once JXP names the intended backend for Gate item 2,
  the `5e-4` inelastic comparison is unreachable regardless of backend
  choice, so Q1's answer alone can't close M4's Gate item 2; framed this
  as a milestone-completion decision (relax tolerance vs. treat as
  informational) that predates M5 and isn't M5's to silently resolve.
- **Context**, third bullet: the elastic robust-vs-Gordon path (M1,
  `robust_baseline`) remains solid at ~2.3e-7 relative, for contrast —
  the problem is specifically and only the inelastic terms.
- **Context**, fourth bullet (task-by-task): summarized M4 tasks 1-3 as
  mechanically correct and tested — the `set_raman_Ed` stash, the
  `Geometry.Ed` routing (gated on `include_Raman`, also feeding
  `fluorescence_kernel` when both flags are on, per Q4), and the `a_ph`
  guard — plus task 4's notebook as the source of every number cited
  above.
- **Context**: replaced the stale M3-era "260 passed, 2 skipped, 2
  failed" full-suite baseline with a freshly re-verified **266 passed, 2
  skipped, 2 failed** (`pytest bing/tests/ -q`, `ocean14`, run live
  2026-08-31), matching M4 task 4's exit baseline exactly; the 2 failures
  remain the same pre-existing `test_l23_inelastic.py` fixture gaps
  (M0/Q10).
- Also updated the "Previous prompt" bullet to note M4 is now fully
  complete (all 5 tasks), keeping the rest of that bullet's wording.

**Overall state handed to M5**: the Ed-wiring mechanics from M4 all work
and are fully tested (stash, routing, guard, conftest isolation) — nothing
here is a plumbing bug. What remains unresolved is a real accuracy gap:
robust's Raman/fluorescence terms diverge from BING's own by 5-18% on real
L23 data because of a structural excitation-grid approximation, not
numerical precision, and this bears directly on how M5's documentation
(especially the `inelastic-rrs/SKILL.md` "recommended path" note) should
describe the robust backend's inelastic physics — plus Q1 remains open,
now with the added weight that answering it will not, by itself, make
M4's Gate item 2 pass at `5e-4` for the inelastic comparison.

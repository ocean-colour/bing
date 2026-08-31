# RoB RT Backend Coding — Prompt 3 (M2: Fitter dispatch and geometry threading)

## Goals

Implement **Milestone M2** of the coding plan
(`docs/coding_plan/rob_rt_coding_plan.md`): make all four `rt_backend` values
selectable end-to-end from `fit_one`/`fit_batch`/`chisq_fit.fit`. Both fitters
gain the two-line backend dispatch, the observation tuple grows one optional
trailing `geom` element, `validate_rt_dict` and `robust_domain_check` are
wired into fit setup, and legacy calls remain byte-for-byte untouched
(regression-pinned, not just claimed).

## Claude

### Skills

`.claude/skills/`: `run-bing-fit` (the canonical fit workflow the dispatch
must not disturb), `fit-l23-spectrum` (the smoke-fit spectra),
`debug-priors`/`diagnose-mcmc` (if the smoke fits misbehave), `code-review`.

## Working agreements

Per the working agreements in `rob_rt_prompt_1.md` (git by JXP; `ocean14`;
CQ1–CQ4 binding; scope discipline; pytest-gated; Fable; log) — **one
correction, carried over from `rob_rt_prompt_2.md`**: the branch is JXP's
existing **`rob_rt`**, not `rob-rt-backend` as the coding plan suggested;
all of M0 and M1 landed there, with JXP reviewing and committing after
each task — assume the same cadence. Milestone-specific emphasis:

- **CQ4 lands here as behavior**: a robust-backend fit with a legacy
  4-tuple (no geom) must raise at setup with a message naming `theta_s`.
- **Plan choice held (design §7.2)**: the observation package stays a tuple
  — the optional 5th element, no dataclass promotion. Don't re-open it.

## Context

Read before coding:

- **Previous prompt** — `rob_rt_prompt_2.md` (M1, **complete**: all 6
  tasks done, Gate items 1–7 covered; full-suite baseline entering M2 is
  **222 passed, 2 skipped, 2 failed** — the same 2 pre-existing
  `test_l23_inelastic.py` failures diagnosed in M0/Q10, not a regression).
  Its Q&A/Logs are the record; the load-bearing facts for this milestone:
  - **Adapter signature** (evaluate.py:438):
    `calc_Rrs_from_models_robust(a_model, a_params, bb_model, bb_params,
    rt_dict, geom=None, Bp=None, debug=False, full_return=False)` — task
    1's dispatch snippet calls it exactly so. Shape convention (Q4): 1-D
    `(nparam,)` params → `(1, nwave)` Rrs, **not** `(nwave,)` — identical
    to the Gordon path's own convention; Gate item 1's "correct shape"
    means this.
  - **Domain check** (evaluate.py:576):
    `robust_domain_check(a_model, a_params, bb_model, bb_params, rt_dict,
    geom=None, Bp=None)` — no `debug`/`full_return`; `geom=None` raises
    the same `ValueError` as the adapter (both consume the shared
    `_build_robust_inputs`, evaluate.py:362, so the checked configuration
    can never drift from the fitted one). Per Q7 it is a **validated
    no-op** for `robust_ztt`/`robust_baseline` — only `mode='hybrid'` has
    a domain to check in `robust.rt` at all (ztt returns before
    `_check_domain`; `baselines.Rrs_gordon` has no domain logic). Task 3's
    "robust backends only" wiring is therefore harmless and cheap for
    ztt/baseline (argument validation still fires) but can only ever emit
    `DomainWarning` for `robust_hybrid`.
  - **Cache** (evaluate.py:245): `_robust_forward_jit(mode, inelastic_key,
    wave_key)`, a module-level `functools.lru_cache`d builder of jitted
    closures — one compile per (backend, inelastic config, grid), verified
    at both the `lru_cache` and underlying XLA compiled-trace levels
    (M1 Gate item 7). `forward()` always runs with `corrections=False` and
    an explicitly pre-loaded `emulator` object (Q6 — both defaults
    lazy-load from disk and leak tracers under `jit`); settled, load-bearing
    behavior, not a TODO. The Risks note below (per-worker caches in
    `fit_batch`) stands as written.
  - **`robust_baseline` + inelastic raises** (Q2): the adapter and domain
    check raise `ValueError` when `rt_backend='robust_baseline'` is
    combined with `include_Raman`/`include_Chl_fl`
    (`baselines.Rrs_gordon` has no inelastic composition path).
    **`validate_rt_dict` does *not* check this combination** — its checks
    are backend name, `fit_Bp`+gordon, robust-needs-geom, and the hybrid
    grid (defs.py:87-111) — so at fit setup this error surfaces only when
    the forward path (or task 3's domain-check call on `p0`) first runs.
  - **Numerics for the smoke fits** (Q11 / `nb/RT/rob_rt_coding_2.ipynb`):
    on a real L23 spectrum the real robust backends sit **above** Gordon
    by +0.3% to +7.4% (median +4.5%) — smooth and physical, not a bug;
    expect robust-backend chains centered on genuinely different values
    than a Gordon fit of the same spectrum. Measured float32 cost:
    ~2.3e-7 relative / ~7.7e-10 sr⁻¹ absolute worst case, ~26,000× below
    a 2% noise floor (CQ1 confirmed empirically). Raman/fluorescence on
    robust backends derive excitation IOPs by interpolating/clamping the
    emission-grid spectrum (Q1) — a real-but-different approximation from
    BING's Gordon+Raman path, so inelastic robust fits will differ
    slightly from Gordon+Raman fits of the same data by design.
  - **Fixture caveat** (Q10, resolved): the repo's blanket `*.npz`
    gitignore hid `bing/tests/files/l23_gordon_fixture.npz` (Gate item 2's
    pinned pre-change result) from `git status` — JXP force-added it
    (`git add -f`, confirmed staged), so it's safe on a fresh checkout;
    `l23_inelastic_fixture.npz` (the two pre-existing, unrelated failures'
    missing fixture) is pending the same treatment once regenerated.
  - **`papers/` scope** (Q9, resolved): JXP's answer is to ignore any code
    under `papers/` — the 7 now-inert `RT_correction` hits in
    `papers/biomass/Analysis/py/` stay as-is, no follow-up needed.
- **Coding plan** — `docs/coding_plan/rob_rt_coding_plan.md` **M2** section,
  plus Risks (`fit_batch` + ProcessPoolExecutor: each worker compiles its
  own JIT cache — acceptable, measured at M5, not silently redesigned).
- **Design** — `docs/design/rob_rt_design.md` §2 (architecture diagram),
  §3.2 (geometry threading and the nadir fallback), §3.4 (the dispatch
  snippet).
- **Q&A record** — `claude_prompts/rob_rt.md`: Design Q3/Q11/Q14 (geometry
  is fixed, non-fit, per-pixel data; nadir fallback for viewing geometry
  only) and Coding Q&A 4 (`theta_s` required).
- **Current code** — `bing/fitting/inference.py` (`log_prob` and the single
  forward call it replaces, inference.py:105; the `aparams`/`bparams` split,
  inference.py:93-94; `fit_one` unpack, inference.py:215; `run_emcee` args,
  inference.py:443; `fit_batch` docs, inference.py:480-486) and
  `bing/fitting/chisq_fit.py` (`fit_func` call site, chisq_fit.py:174-175;
  `fit` unpack, chisq_fit.py:111).

## Prompts

1. Read this doc. Execute the 1st task in the "M2" section below — the
   dispatch. If you have any questions, ask me in the Q&A section below.
   Use Fable if you can. Log your work.
2. Read this doc. Execute the 2nd task — the observation tuple. Check my
   answers in Q&A; if you have additional questions, ask in Q&A. Use Fable
   if you can. Log your work.
3. Read this doc. Execute the 3rd task — setup validation + domain-check
   wiring, and the docstrings. Use Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the explainer notebook. Use Fable
   if you can. Log your work.
5. Read this doc. Execute the 5th task — update `rob_rt_prompt_4.md` with
   what M2 established. Use Fable if you can. Log your work.

## M2

### Tasks

1. **Two-line dispatch** at both call sites (design §3.4): in
   `inference.log_prob` (branch replacing the single call at
   inference.py:105) and `chisq_fit.fit_func` (chisq_fit.py:174-175):
   `rt_dict.get("rt_backend", "gordon") == "gordon"` → existing call,
   unchanged; else → `calc_Rrs_from_models_robust(..., geom=geom, Bp=Bp)`.
   `log_prob` gains optional trailing `geom=None` (`Bp` handling comes in
   M3); same for `fit_func`'s keyword set.

2. **Observation tuple grows one optional trailing element** (design §3.2):
   `(Rrs, varRrs, params, idx[, geom])`. `fit_one` (unpack at
   inference.py:215) and `chisq_fit.fit` (unpack at chisq_fit.py:111)
   accept either length (`geom = items[4] if len(items) > 4 else None`);
   `fit_batch` docs (inference.py:480-486) gain the fifth element. *(Plan
   choice, design §7.2: keep the tuple — minimal, matches every existing
   call site.)* `geom` rides by value into `emcee`'s `args` list
   (`run_emcee`, inference.py:443) exactly like `Rrs`/`varRrs` — the
   sampler never sees it as a dimension.

3. **Setup validation and domain-check wiring.** `fit_one` and
   `chisq_fit.fit` call `validate_rt_dict(rt_dict, models=models,
   geom=geom)` (M0) before running — this is where the hybrid grid error
   and the missing-`theta_s` error surface, once per fit. `fit_one` runs
   `robust_domain_check` on `p0` before `run_emcee` and on the posterior
   median after (robust backends only); `chisq_fit.fit` runs it on `p0`
   only. Update docstrings for `fit_one`, `fit_batch`, `chisq_fit.fit`,
   `log_prob` for the new element/keys.

4. **Notebook.** `nb/RT/rob_rt_coding_3.ipynb` (executed, with outputs):
   one L23 spectrum fit end-to-end under each of the four backends (small
   nsteps), the nadir-fallback demonstration (implicit vs explicit nadir
   `log_prob` identical), and a non-nadir geometry visibly changing Rrs
   under `robust_ztt`. Show the raised errors (missing geom; hybrid grid
   out of range) as they'd appear to a user.

5. **Finally.** Update `rob_rt_prompt_4.md` (M3) with what M2 established —
   the fitter signatures as they now stand and any smoke-fit surprises
   (walker counts, runtimes, JIT compile times per backend).

### Gate

`bing/tests/test_evaluate_robust.py` additions (deterministic seeds, tiny
nsteps/nwalkers):

1. **End-to-end smoke fit**: one L23 spectrum through `fit_one` (small
   nsteps) and `chisq_fit.fit` under **each** of the four backend values —
   chains/params finite, correct shape.
2. Legacy 4-tuple + default (gordon) rt_dict reproduces today's behavior —
   regression against the pinned pre-change result from M1.
3. Robust backend + 4-tuple (no geom) raises at setup with a message naming
   `theta_s`.
4. `ObsGeometry(theta_s=30.)` (implicit nadir) and `ObsGeometry(30., 0., 0.)`
   give identical `log_prob`.
5. `ObsGeometry(30., theta_v=40., dphi=90.)` changes Rrs under `robust_ztt`
   (non-nadir sensitivity).
6. `fit_batch` with a mixed list of 4- and 5-tuples runs (gordon) / raises
   correctly (robust).

Existing suite green throughout.

## Q&A

**Q1 (task 1, Claude → JXP).** The task-1 snippet says the robust branch
calls `calc_Rrs_from_models_robust(..., geom=geom, Bp=Bp)`, but no `Bp`
variable exists in `log_prob`/`fit_func` at this point in the milestone —
the task's own parenthetical defers `Bp` handling to M3. Resolution, by
reading the adapter: `Bp=None` is the adapter's own default and means
"fall back to `rt_dict['Bp_value']`" (`_build_robust_inputs`,
evaluate.py:423; documented on `calc_Rrs_from_models_robust` as "the
fixed-B_p, default case"). So the dispatch passes `Bp=None` explicitly,
with a comment noting the free/sampled `Bp` arrives in M3 — the design
snippet's `Bp=Bp` describes the *final* (post-M3) form of the call, not
something to fabricate now. No answer needed to proceed.

**Q2 (task 2, Claude → JXP, resolved — no answer needed).** The task-2 text
says `geom` "rides by value into `emcee`'s `args` list" but names only
`fit_one`/`chisq_fit.fit` as gaining code. Reading the actual mechanism:
`emcee`'s `_FunctionWrapper` calls `f(x, *args)` — the `args` sequence is
delivered **purely positionally** after the walker's parameter vector, so
appending `geom` to `run_emcee`'s `args=[models, Rrs, varRrs, rt_dict]`
lands it exactly in `log_prob`'s trailing `geom` slot (its signature order
is `params, models, Rrs, varRrs, rt_dict, geom`) — but only because that
list mirrors the signature; the ordering constraint is now stated in a
comment at the `EnsembleSampler` construction. Consequence: `run_emcee`
itself had to gain a trailing `geom=None` keyword (not mentioned in the
task text) to carry the unpacked value from `fit_one` into that list —
the minimal faithful implementation, since `run_emcee` is the only thing
that builds `args`. `fit_batch` needed **no code change** (it forwards
each tuple to `fit_one` unchanged via `executor.map`, with no length
assertion anywhere), so its change is docstring-only as the task
predicted. Resolved by reading the code; no answer needed to proceed.

**Q3 (task 3, Claude → JXP, resolved — no answer needed).** The task-3
text says `fit_one` runs `robust_domain_check` on `p0`/posterior-median
"(robust backends only)". Verified empirically (live call in `ocean14`,
plus the existing pin `test_robust_domain_check_shares_adapter_error_paths`)
that this parenthetical is **mandatory, not stylistic**: calling
`robust_domain_check` with `rt_backend='gordon'` raises `ValueError`
("not a robust backend", from `_build_robust_inputs`), and with the
typical Gordon-fit `geom=None` it raises the missing-geom error even
earlier — it is *not* a no-op for gordon the way it is for
ztt/baseline. Both fitters therefore gate the call on
`rt_dict.get('rt_backend', 'gordon') != 'gordon'`, and a new bomb-
monkeypatch test (`test_gordon_fit_never_calls_domain_check`) pins that
a Gordon fit never invokes it at all. Also: `fit_one` computes no
posterior median anywhere today (it returns the raw sampler/chains), so
the post-run check computes it directly —
`np.median(chain.reshape(-1, ndim), axis=0)` over all steps × walkers,
split into `a_params`/`bb_params` by the same `models[0].nparam`
convention as `log_prob`. Nothing existed to reuse. Resolved by reading
and running the code; no answer needed to proceed.

**Q4 (task 3, Claude → JXP, resolved — no answer needed; one FYI).**
`fit_one`'s signature has `rt_dict:dict=None`, and one in-repo caller
actually exercises that default: `bing/scripts/fit_Rrs.py:121` calls
`fit_one(items[0], models=models, pdict=pdict, chains_only=True)` with
no `rt_dict`. A bare `validate_rt_dict(None, ...)` would turn that into
a *new* setup-time `AttributeError`, so both fitters validate
`rt_dict if rt_dict is not None else {}` — an empty dict validates as
the default Gordon configuration, preserving that caller's behavior
exactly (it fails/succeeds wherever it did before, never in the new
validation). FYI, not an open question: the *next* line of that script
(`fit_Rrs.py:124`, `chisq_fit.fit(items[0], models)`) omits the
**required positional** `rt_dict` and has therefore been a `TypeError`
since long before this milestone — pre-existing, untouched, out of
scope (per the M1 Q9 precedent of not chasing inert callers).

**Q5 (task 4, Claude → JXP — OPEN: Gate-coverage gap, needs a decision
before M2 is called fully gated).** Task 4 is the notebook, not "write the
remaining Gate tests", so this is reported rather than silently fixed —
but after tasks 1–4, cross-checked against the actual test bodies in
`bing/tests/test_evaluate_robust.py` (not the Logs prose), **Gate items 1,
4, and 5 have no automated pytest coverage, and item 2 is only partially
covered**:

- **Item 1 (end-to-end smoke fit, L23 spectrum, both fitters × all four
  backends): NOT covered.** Task 2's real fits are Gordon-only on a
  synthetic spectrum; task 3's only real robust-backend MCMC is the turbid
  `robust_hybrid` DomainWarning test (synthetic spectrum, `fit_one` only).
  No test anywhere runs a successful fit under `robust_ztt` or
  `robust_baseline`, none fits an L23 spectrum through the new dispatch,
  and `chisq_fit.fit` has no successful robust-backend fit test at all.
  Note the task-2 section comment (test_evaluate_robust.py:706-708) says
  these smoke fits were "the milestone Gate's item 1 (task 3)" — but task
  3 did not add them.
- **Item 2 (legacy 4-tuple + default gordon vs the pinned M1 result):
  PARTIAL.** The M1 pin (`files/l23_gordon_fixture.npz`, exercised in
  `test_evaluate.py`) covers the *forward model* only; task 2 pins
  identical positional args into `log_prob` (identity, every walker
  evaluation) and bit-identical 4- vs 5-tuple `chisq_fit` answers. But no
  test regresses a *fitter-level* result (seeded chains or chisq params)
  against a pinned pre-change value.
- **Item 4 (implicit vs explicit nadir → identical `log_prob`): NOT
  covered.** Only the `ObsGeometry` field-default/`to_robust` roundtrip
  tests exist (M0) — nothing compares `log_prob` or Rrs between
  `ObsGeometry(theta_s=30.)` and `ObsGeometry(30., 0., 0.)`.
- **Item 5 (non-nadir geometry changes Rrs under `robust_ztt`): NOT
  covered.** No test anywhere evaluates a non-nadir geometry's effect on
  Rrs (`grep theta_v\|dphi` over `bing/tests/` hits only
  test_evaluate_robust.py's roundtrip tests).
- Items 3 and 6 are genuinely covered (task 3; task 2 + task 3 for the
  two halves of 6).

The task-4 notebook (`nb/RT/rob_rt_coding_3.ipynb`) *demonstrates* items
1, 4, and 5 with real executed outputs — but an executed notebook is not
an automated regression test, and "Existing suite green throughout" plus
the Gate list is a milestone-level claim. Options: a small follow-up test
task before task 5, folding them into task 5, or explicitly accepting the
notebook demonstrations for this milestone. JXP's call.

## Next

→ `rob_rt_prompt_4.md` (M3: `B_p` as an optional free MCMC parameter).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-30 (M2 task 1 — two-line fitter dispatch)

**Read before coding.** `bing/fitting/inference.py` and
`bing/fitting/chisq_fit.py` in full, plus `bing/evaluate.py:360-540`
(`_build_robust_inputs` and the `calc_Rrs_from_models_robust` docstring)
to confirm the adapter's exact signature and its `Bp=None` semantics.
Unlike M1's `evaluate.py` drift, this doc's line citations were still
accurate: the single forward call in `log_prob` sat exactly at
inference.py:105, and `fit_func`'s at chisq_fit.py:174-175. Also
confirmed no current caller anywhere sets `rt_backend` in an `rt_dict`
passed to these two functions, so `rt_dict.get('rt_backend', 'gordon')`
degrades to the old behavior for every existing call site.

**Implemented.**

- `inference.log_prob` (signature now inference.py:52-53): gained
  trailing `geom=None`; the forward call became the two-line dispatch at
  inference.py:116-123 — `'gordon'` (or absent key) keeps the legacy
  `calc_Rrs_from_models` call byte-for-byte, else
  `calc_Rrs_from_models_robust(models[0], aparams, models[1], bparams,
  rt_dict, geom=geom, Bp=None)`. Docstring documents `rt_backend` and
  `geom`.
- `chisq_fit.fit_func` (signature now chisq_fit.py:123-124): gained
  trailing `geom=None`; same dispatch at chisq_fit.py:184-191. Docstring
  updated likewise.
- **`Bp` resolution (Q1)**: the snippet's `Bp=Bp` is the post-M3 form;
  at this stage the dispatch passes `Bp=None` explicitly, which the
  adapter documents as "fall back to `rt_dict['Bp_value']`"
  (evaluate.py:423) — the fixed-B_p case. In-code comments at both
  branches say so.

Nothing else was touched: no observation-tuple threading (task 2), no
`validate_rt_dict`/`robust_domain_check` wiring (task 3) — `run_emcee`'s
`args` list and both `fit`/`fit_one` unpacks are unchanged, so nothing
yet passes a real `geom` into either function.

**Tested.** Full suite in `ocean14`, before-vs-after identical:
**222 passed, 2 skipped, 2 failed** (the same two pre-existing
`test_l23_inelastic.py` missing-fixture failures — not a regression).
No test file changed. Plus a throwaway interactive smoke check (not
committed, per the task's guidance not to pre-build task-2/3 test
infrastructure): direct `log_prob`/`fit_func` calls with an
`expb_pow2` setup — gordon branch gives `log_prob = 0.0` against its
own forward Rrs (exact legacy behavior); `rt_backend='robust_ztt'` +
`ObsGeometry(theta_s=30.)` gives finite log_prob and a finite `(61,)`
flattened `fit_func` Rrs (the adapter's `(1, nwave)` flattened, matching
the Gordon path's convention); robust backend without `geom` raises the
adapter's `ValueError` naming geom/theta_s.

**Next.** Task 2: grow the observation tuple to
`(Rrs, varRrs, params, idx[, geom])` — unpacks at inference.py `fit_one`
and chisq_fit.py `fit`, `geom` appended to `run_emcee`'s emcee `args`
list, `fit_batch` docs updated.

### 2026-08-30 (M2 task 2 — observation tuple threading)

**Read before coding.** `bing/fitting/inference.py` (`fit_one`,
`run_emcee`, `fit_batch`) and `bing/fitting/chisq_fit.py` (`fit`) as they
stand after task 1, plus emcee's call convention as used here. Task 1's
edits **shifted this doc's inference.py citations**: `fit_one`'s unpack
sat at inference.py:232 (doc said 215), `run_emcee`'s
`EnsembleSampler(..., args=...)` at inference.py:458-461 (doc said 443),
`fit_batch`'s items docs at inference.py:497-503 (doc said 480-486).
`chisq_fit.fit`'s unpack was still exactly at chisq_fit.py:111 (`fit` sits
above `fit_func`, so task 1's edits didn't move it). Load-bearing
mechanism fact (Q2): emcee's `_FunctionWrapper` calls `f(x, *args)` —
`args` is delivered **purely positionally** after the parameter vector,
so the list must mirror `log_prob`'s signature order
`(params, models, Rrs, varRrs, rt_dict, geom)`; appending `geom` as the
5th element is correct only because task 1 made `geom` the 5th
post-`params` parameter. Also verified `fit_batch` has no tuple-length
assertion — it forwards each tuple to `fit_one` unchanged through
`executor.map` — so it needed no code change.

**Implemented** (current line numbers, post-edit):

- `inference.fit_one`: length-tolerant unpack at inference.py:238-239
  (`Rrs, varRrs, params, idx = items[:4]`;
  `geom = items[4] if len(items) > 4 else None`), passes `geom=geom` into
  `run_emcee` (inference.py:258). Docstring documents the optional 5th
  element (inference.py:201-205).
- `inference.run_emcee`: gained trailing `geom=None`
  (inference.py:362-368, signature) with a docstring entry
  (inference.py:409-415); the `args` list became
  `args=[models, Rrs, varRrs, rt_dict, geom]` (inference.py:478) under a
  comment stating the positional-order constraint. This new keyword is
  the one piece the task text didn't name — see Q2 (resolved).
- `inference.fit_batch`: **docstring only** (inference.py:515-527) — the
  optional 5th element, forwarded unchanged to `fit_one`, 4-/5-tuples
  mixable in one list. No code change needed (verified, not assumed).
- `chisq_fit.fit`: same length-tolerant unpack at chisq_fit.py:118-119;
  `geom` threads into the optimizer via
  `partial(fit_func, models=..., rt_dict=..., geom=geom)`
  (chisq_fit.py:124-125), so every curve_fit evaluation of `fit_func`
  receives it. Docstring documents the 5th element.

Deliberately **not** done (task 3): no `validate_rt_dict` /
`robust_domain_check` wiring — a robust backend with a 4-tuple still
raises only the adapter's own `ValueError` when `log_prob`/`fit_func`
first runs, not a setup-time `theta_s` message (CQ4 lands in task 3).

**Tested.** Five new tests in `bing/tests/test_evaluate_robust.py`
("M2 task 2" section at the end), all Gordon-backend, cheap synthetic
ExpBricaud+Pow spectrum (same recipe as test_chisq_fit.py), tiny
nsteps=30/nburn=10/nwalkers=16:

- `test_fit_one_legacy_4tuple_args_unchanged`: a recording `log_prob`
  wrapper (monkeypatched; emcee resolves the module global at call time)
  pins that a 4-tuple delivers the **identical objects** in the identical
  positional slots with `geom=None`, on every walker evaluation; chains
  finite, shape `(30, 16, 5)`, idx echoed.
- `test_fit_one_5tuple_geom_reaches_log_prob`: the very
  `ObsGeometry(theta_s=30.)` instance from the tuple arrives in
  `log_prob`'s geom slot (`is`-identity) on every evaluation of a real
  emcee run, and no other slot shifts; chain width stays nparam (geom is
  never a sampled dimension).
- `test_chisq_fit_4tuple_and_5tuple_none_identical`: trailing `None` is
  bit-identical to the legacy 4-tuple (curve_fit is deterministic), and
  the fit recovers the noiseless truth.
- `test_chisq_fit_5tuple_geom_reaches_fit_func`: the same instance
  reaches `fit_func`'s `geom` kwarg on every optimizer evaluation via the
  partial; answer unperturbed (Gordon ignores geom).
- `test_fit_batch_mixed_4_and_5_tuples_gordon`: a mixed 4-/5-tuple list
  through `fit_batch` (n_cores=1, real ProcessPoolExecutor) — shapes and
  idx set correct. (The robust-raises half of Gate item 6 lands with
  task 3.)

Full suite in `ocean14`: **227 passed, 2 skipped, 2 failed** — exactly
the 222-pass baseline plus the 5 new tests, and the same two pre-existing
`test_l23_inelastic.py` missing-fixture failures (not a regression).

**Next.** Task 3: wire `validate_rt_dict(rt_dict, models=models,
geom=geom)` into `fit_one`/`chisq_fit.fit` setup (the CQ4 `theta_s`
message), `robust_domain_check` on `p0` (both fitters) and on the
posterior median (`fit_one`, robust backends only), and the remaining
docstring updates. Note for task 3: the doc's inference.py line citations
have drifted again after this task's edits — use the line numbers above.

### 2026-08-30 (M2 task 3 — setup validation + domain-check wiring)

**Read before coding.** `bing/rt/defs.py` in full: `validate_rt_dict`'s
real signature is exactly the task's `validate_rt_dict(rt_dict,
models=None, geom=None)` (`models=` is the correct keyword — the
`[a_model, bb_model]` list, consulted only for the hybrid grid check via
`models[0].wave`; defs.py:58). Its missing-geom error text (defs.py:98-102),
confirmed live in `ocean14`, is:
`"rt_dict['rt_backend']='robust_ztt' requires geometry -- pass an
ObsGeometry via geom= (theta_s is never silently defaulted;
claude_prompts/rob_rt.md Q&A/Coding item 4)"` — it names `theta_s`, so
CQ4's gate test asserts `pytest.raises(ValueError, match='theta_s')`
against the real text. The hybrid grid error text ends "...the emulator's
training range); got range [...]" (`match='training range'`). Also read
`evaluate.py`'s `robust_domain_check`/`_build_robust_inputs` and both
fitters as they stand after task 2 (task 2's corrected line numbers were
still accurate — no further drift before this task). **Verified
empirically** (live calls, not assumed): `robust_domain_check` with
`rt_backend='gordon'` raises `ValueError` "not a robust backend"
(and with `geom=None` raises the missing-geom error first), so the
task's "(robust backends only)" is mandatory — see Q3. One caller found
by grep exercises `fit_one`'s `rt_dict=None` default
(`bing/scripts/fit_Rrs.py:121`) — see Q4 for how validation tolerates it.

**Implemented** (current line numbers, post-edit — this doc's earlier
citations have drifted again; task 4 should use these):

- `inference.py`: new import `from bing.rt import defs as rt_defs`
  (inference.py:47). `fit_one`: `validate_rt_dict(rt_dict if rt_dict is
  not None else {}, models=models, geom=geom)` immediately after the
  tuple unpack (inference.py:285-286; unpack now at 276-277), with
  `rt_backend` resolved once at inference.py:287-288. Domain check on
  the initial guess — gated `rt_backend != 'gordon'`, after
  `init_other_bits` (so checked models carry the fit's Chl/Y) and
  before `run_emcee` — at inference.py:304-309, splitting `params` by
  `models[0].nparam` exactly like `log_prob`'s `aparams`/`bparams`.
  Domain check on the posterior median after `run_emcee` at
  inference.py:326-332: no median existed anywhere in `fit_one` to
  reuse (Q3), so it computes
  `np.median(sampler.get_chain().reshape(-1, ndim), axis=0)`.
  Docstrings: `fit_one` gained Raises/Warns sections and a setup-
  validation Note (inference.py:237-267); `log_prob` a "performs no
  validation itself — the fitters validate at setup" Note
  (inference.py:101-110); `fit_batch` a per-worker-validation note
  incl. the warnings-don't-cross-processes caveat (inference.py:622-629).
  `run_emcee` itself unchanged (validation lives in `fit_one`, once).
- `chisq_fit.py`: same import (chisq_fit.py:43). `fit`: validation at
  chisq_fit.py:142-148 (unpack now at 139-140), p0-only domain check
  (no posterior in a χ² fit) gated on non-gordon at chisq_fit.py:150-162,
  both before the `curve_fit` call. Docstring gained Raises/Warns
  (chisq_fit.py:100-121). `fit_func` unchanged.
- **Gordon decision (Q3)**: skip the domain check entirely for
  `'gordon'` — verified empirically that calling it would raise, so the
  gate is correctness, not optimization; a bomb-monkeypatch test pins
  that no Gordon fit ever invokes it.

**Tested.** Eight new tests (12 test items — two are parametrized over
the three robust backends) in `bing/tests/test_evaluate_robust.py`,
"M2 task 3" section (test_evaluate_robust.py:863-1043), all with the
task-2 `threading_setup` fixture (tiny nsteps=30/nburn=10/nwalkers=16,
seeded):

- `test_fit_one_robust_4tuple_raises_theta_s_at_setup` /
  `test_chisq_fit_robust_4tuple_raises_theta_s_at_setup` (×3 backends
  each): CQ4 / Gate item 3 — legacy 4-tuple + robust backend raises
  `ValueError` matching `theta_s`, with `run_emcee`/`curve_fit`
  monkeypatched to sentinels proving zero sampling/optimizer work.
- `test_fit_one_hybrid_grid_error_at_setup` /
  `test_chisq_fit_hybrid_grid_error_at_setup`: M0's grid check now
  surfaces through the fitters (400–760 nm grid, `match='training
  range'`), again before the sentinel.
- `test_fit_one_turbid_hybrid_fires_domain_warning`: a real tiny
  robust_hybrid MCMC with M1's out-of-domain `Bp_value=0.005` —
  `pytest.warns(DomainWarning, match='outside its training range')`,
  chains still finite (warn-and-continue).
- `test_chisq_fit_turbid_hybrid_fires_domain_warning_before_optimizer`:
  same turbid config through `chisq_fit.fit` with `curve_fit` stubbed
  (echoes p0), so the warning can only be the setup p0 check.
- `test_gordon_fit_never_calls_domain_check`: bomb monkeypatch on
  `evaluate.robust_domain_check`; both fitters complete normal Gordon
  fits untouched.
- `test_fit_batch_robust_4tuple_raises_theta_s`: Gate item 6's robust
  half — a 4-tuple in the items list under `robust_ztt` raises through
  the worker with the `theta_s` message (offender listed first so the
  chunk fails before the 5-tuple's real fit runs).

Full suite in `ocean14`: **239 passed, 2 skipped, 2 failed** — exactly
the task-2 baseline of 227 plus the 12 new test items, and the same two
pre-existing `test_l23_inelastic.py` missing-fixture failures (not a
regression). Task 1/2's tests re-ran unchanged and green, pinning that
the Gordon path is unaffected by the new setup validation (which *does*
run for gordon rt_dicts, as a genuine validated pass-through — tested,
not assumed).

**Next.** Task 4: the explainer notebook `nb/RT/rob_rt_coding_3.ipynb` —
all four backends end-to-end (small nsteps), the nadir-fallback
demonstration, non-nadir geometry changing Rrs under robust_ztt, and the
raised errors (missing geom → the `theta_s` message; hybrid grid out of
range) as a user would see them. All Q&A entries through Q4 are resolved
findings — nothing blocks task 4.

### 2026-08-30 (M2 task 4 — explainer notebook)

**Read before coding.** This doc (all Logs and Q1–Q4), task 3's current
line numbers re-verified against the live files (no further drift — no
source file was touched in this task), `nb/RT/rob_rt_coding_2.ipynb` for
house style and the L23 setup recipe, the `fit-l23-spectrum` /
`run-bing-fit` skills, and `bing/tests/test_evaluate_robust.py` in full
for the scope-boundary Gate-coverage cross-check (see Q5). Both expensive
steps (the four seeded smoke fits incl. the Flax-emulator-loading
`robust_hybrid` one; the error triggers) were smoke-tested standalone in
`ocean14` before the notebook run — total ~3.4 s, so no runtime surprises.

**Built and executed** `nb/RT/rob_rt_coding_3.ipynb` (18 cells, 7 code;
executed top-to-bottom with `jupyter nbconvert --execute` in `ocean14`).
Sections, with the real measured numbers now in its outputs:

1. **One L23 spectrum (idx=170, PACE grid, 61 bands), four backends,
   end-to-end through `fit_one`** — nsteps=200/nburn=50/nwalkers=16,
   re-seeded identically per backend (`np.random.seed(2026)`). The Gordon
   fit deliberately uses the legacy 4-tuple; the robust fits the 5-tuple
   with `ObsGeometry(theta_s=30.)`. All four: chains `(200, 16, 5)`,
   finite throughout, posterior medians side by side. The medians tell
   M1's physics story: robust_ztt/hybrid center on genuinely different
   values (e.g. Bnw −3.68 vs Gordon's −3.55; beta 0.90/1.21 vs 1.13),
   while `robust_baseline`'s medians match Gordon's **exactly** (printed:
   max |Δmedian| = 0.00e+00 — same seed + float32-level forward parity ⇒
   same accept/reject sequence).
2. **Nadir fallback** — `ObsGeometry(theta_s=30.)` vs
   `ObsGeometry(30., 0., 0.)` under robust_ztt at p0: log_prob
   −27.05867534787494 both ways, difference 0.0, `==` True; full Rrs
   max |Δ| = 0.0 over all 61 bands. Bitwise identical, printed not
   asserted.
3. **Non-nadir sensitivity** — `ObsGeometry(30., theta_v=40., dphi=90.)`
   vs nadir, identical IOPs, robust_ztt: median ΔRrs/Rrs = −10.28%,
   range [−11.54%, −8.19%], max |ΔRrs| = 8.986e-04 sr⁻¹; two-panel
   figure (spectra + relative difference).
4. **Raised errors, verbatim** — (a) `fit_one` + robust_ztt + legacy
   4-tuple → the CQ4 ValueError naming theta_s ("requires geometry --
   pass an ObsGeometry via geom= (theta_s is never silently
   defaulted...)"); (b) `chisq_fit.fit` + robust_hybrid on a 400–760 nm
   grid → "...within [350.0, 750.0] nm (the emulator's training range);
   got range [400.0, 760.0]". Both through real fitter calls in
   try/except, so both fitters' setup validation appears.
5. Closing markdown: what M2 established + pointer to task 5.

**Output verification** (the M1-era lesson, applied): after execution the
saved .ipynb was re-read programmatically — execution_counts sequential
1–7, no code cell without outputs, no error outputs, figure present; the
notebook was then executed a *second* time after a markdown-only fix and
every quoted number above confirmed byte-identical in the saved outputs
(seeded fits are deterministic). One prose error was caught by re-diffing
markdown against outputs before finishing: an initial draft called the
~10% non-nadir change "an order of magnitude above PACE-class noise" —
it is ~5× the 2% floor, and the cell was corrected to "several times".

**Gate-coverage finding (Q5, open).** Cross-checking the Gate list
against the actual test suite: items 3 and 6 are covered; **items 1, 4,
5 have no automated pytest coverage and item 2 is only partial** (details
in Q5). The notebook demonstrates 1/4/5 with executed outputs but is not
a substitute for regression tests — flagged for JXP rather than fixed,
since adding tests is outside task 4's scope.

**Tested.** No package source changed in this task (new notebook + this
doc only), so the suite baseline stands at task 3's
**239 passed, 2 skipped, 2 failed** (the same two pre-existing
`test_l23_inelastic.py` missing-fixture failures).

**Next.** Task 5: update `rob_rt_prompt_4.md` (M3) with what M2
established — the fitter signatures as they now stand, the smoke-fit
numbers (16 walkers, 0.1–1.7 s per 250-step fit on this machine,
one-time JIT compile per robust backend/grid, hybrid's emulator load
dominating its first call), and Q5's Gate-coverage gap so M3 doesn't
inherit it silently.

### 2026-08-31 (M2 task 5 — handing off to `rob_rt_prompt_4.md`)

**Read before writing.** This doc in full (Q1–Q5, all four Logs entries),
`rob_rt_prompt_4.md` in full, and the live source — every signature and
line number written into the M3 doc was re-verified against
`bing/fitting/inference.py`, `bing/fitting/chisq_fit.py`,
`bing/evaluate.py`, and `robust/rt/types.py` on this date (the Logs'
recurring lesson: citations drift; the task-3/4 numbers were still
accurate, the M3 doc's pre-existing ones were not). One doc-internal
wrinkle resolved by the doc itself: the Prompts section's item 5 says
"update `rob_rt_prompt_5.md`", but the M2 → Tasks section's own task-5
text, the Next pointer, and M3's Context all name **`rob_rt_prompt_4.md`**
— the Tasks text governs, and `rob_rt_prompt_4.md` is what was updated.

**Updated `rob_rt_prompt_4.md`** (Context + Working agreements only —
its M3 Tasks/Gate sections are JXP's plan and were not touched; Q&A/Logs
correctly remain "none yet"):

- **Working agreements**: carried over the branch correction (**`rob_rt`**,
  not `rob-rt-backend`) exactly as prompt 2/3 did, and refreshed the
  load-bearing split citation to post-M2 lines (inference.py:113-114,
  chisq_fit.py:230-231).
- **Context, "Previous prompt" bullet**: expanded from one line into the
  M2 fact sheet — the verified fitter signatures and dispatch/peel sites,
  the `Bp=None` → `rt_dict['Bp_value']` fallback (Q1) that M3 task 1
  replaces, the mandatory `!= 'gordon'` domain-check gate (Q3, incl. that
  the check accepts `Bp=` so the peeled value should flow into it), the
  measured smoke-fit numbers below, and the Q5 gap with its M3
  consequence.
- **Context, "Current code" bullet**: stale line numbers corrected
  (`init_mcmc` ndim now inference.py:187-188, not 159-160;
  `reconstruct_from_chains` now evaluate.py:683, not 245 — 245 is
  `_robust_forward_jit` these days; `PhaseParams` types.py:250/277).

**Smoke-fit measurements** (new, standalone in `ocean14` — the task-4
notebook demonstrated the fits but never timed them; same recipe: L23
idx=170, 61-band PACE grid, seeded `fit_one`, nsteps=200/nburn=50/
nwalkers=16): gordon 0.12 s cold and warm (no JIT, ~2200 emcee it/s).
Warm robust fits 0.51–0.61 s (ztt 0.59 / hybrid 0.61 / baseline 0.51;
~420–520 it/s — ~5× Gordon per step at this size). One-time first-call
compile on top: ztt ~0.35 s, hybrid ~1.1 s (Flax emulator load
dominating; 1.72 s cold total, identical whether run first in a fresh
process or after other backends — verified both ways), baseline ~0.03 s
(negligible; `Rrs_gordon` is a trivial trace). **No genuine surprises**:
repeated identical-config fits never recompile (M1's `lru_cache`
behavior paying off in a real fit context, observed not assumed), no
memory growth at these sizes — a confirmed "nothing unusual", stated for
M3's benefit as the pre-`fit_Bp` baseline (ndim 5→6 keeps nwalkers at
max(16, 2×ndim)=16).

**The Q5 → M3 Gate item 6 dependency, flagged.** M3's Gate item 6 reads
"`fit_Bp=False` results identical to an M2-pinned value" — but Q5's
still-open finding is precisely that **no fitter-level regression pin
exists** (Gate item 2 partial: only M1's forward-model fixture and task
2's tuple-arg-identity tests). There is no "M2-pinned value" for M3 to
compare against; M3 task 3 will hit this directly. Stated in the M3 doc
with the options (answer Q5 first, or let M3 create its own fixed-`B_p`
pin when `fit_Bp=False` is first tested) and left as JXP's decision —
not resolved unilaterally, same as Q5 itself.

**Tested.** No package source changed (doc edits + a scratchpad timing
script only), so the suite baseline stands at task 3's **239 passed,
2 skipped, 2 failed** (the same two pre-existing `test_l23_inelastic.py`
missing-fixture failures).

**M2 is now fully complete — all 5 tasks done.** Handed to M3: all four
`rt_backend` values run end-to-end through both fitters with setup
validation, geometry threading by value, and legacy calls regression-
pinned at the argument level; measured per-backend cost baselines; and
one genuinely open item — Q5's Gate-coverage gap (items 1/4/5 untested,
item 2 partial), which now has a concrete downstream consequence in M3
Gate item 6 and should be resolved before or during M3 task 3 rather
than deferred indefinitely.

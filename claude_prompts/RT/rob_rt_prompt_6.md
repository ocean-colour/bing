# RoB RT Backend Coding — Prompt 6 (M5: Deprecation notes, docs, throughput benchmark)

## Goals

Implement **Milestone M5** — the last milestone — of the coding plan
(`docs/coding_plan/rob_rt_coding_plan.md`): close out the integration with
docstring-only deprecation pointers on BING's own inelastic modules, complete
the documentation surface (`rt_dict_from_p` docstring, the two skill docs,
`CLAUDE.md`), and produce the throughput benchmark that informs JXP's
accept/optimize call on the hybrid backend. The evidence is **reported, not
thresholded** — if `robust_hybrid` is grossly slower than `gordon`, that is a
decision for JXP, not a silent fix.

## Claude

### Skills

`.claude/skills/`: `run-bing-fit` and `inelastic-rrs` — both are **edit
targets** this milestone, so read them as they stand before changing them;
`code-review` for the final review pass before JXP's merge review.

### Working agreements

Per the working agreements in `rob_rt_prompt_1.md` (git by JXP; `ocean14`;
CQ1–CQ4 binding; scope discipline; pytest-gated; Fable; log) — **one
correction, carried over from `rob_rt_prompt_2.md`–`rob_rt_prompt_5.md`**:
the branch is JXP's existing **`rob_rt`**, not `rob-rt-backend` as the
coding plan suggested; all of M0–M4 landed there, with JXP reviewing and
committing after each task — assume the same cadence. Milestone-specific
emphasis:

- **No code is deleted from `bing/rt/raman.py` / `chl_fl.py`** — docstring
  pointers only (design §6). They still serve the `gordon` backend.
- **Benchmark numbers are evidence, not a gate**: record them; do not
  optimize the hybrid path on your own judgment.

## Context

Read before coding:

- **READ THIS FIRST — the measured robust-vs-BING inelastic agreement is
  far outside tolerance; it directly bears on task 2's `inelastic-rrs`
  "recommended path" note.** M4 (`rob_rt_prompt_5.md` Q7/Q8, measured live
  in `nb/RT/rob_rt_coding_5.ipynb` §§3-4 on a real L23 spectrum, idx=170,
  with a real production Ed on both sides) found robust-vs-BING agreement
  on the inelastic terms of **Raman: max 11.37% / mean 5.64% (worst at
  400 nm)**; **fluorescence: max 9.18% / mean 5.77% (mismatched-Ed case,
  as the M4 code ships) or max 18.49% / mean 7.21% (Ed-matched case,
  larger not smaller)** — all several **orders of magnitude** outside the
  milestone's own `rtol <= 5e-4` working tolerance, and not shrinkable by
  going to float64 (float32-vs-float64 alone costs ~2.3e-7 relative, four
  orders of magnitude smaller than the gap). Root cause (M1 Q1,
  reconfirmed by M4 Q7): a genuine **physics-composition difference**, not
  numerical noise — `robust`'s Raman/fluorescence kernels
  interpolate/clamp the emission-grid IOP spectrum to stand in for the
  excitation-wavelength IOPs, rather than re-evaluating the parametric a/bb
  models at the true excitation grid the way BING's own Gordon+Raman/
  fluorescence path does. **Task 2's current wording — "the robust
  inelastic path is now the recommended one" — must not be written
  verbatim without accounting for this number; whoever executes task 2
  needs to decide how the `inelastic-rrs/SKILL.md` "recommended path" note
  should actually read given a 5-18% disagreement on real data, not assume
  unqualified equivalence to BING's own Raman/fluorescence terms.** This
  hand-off is not deciding that wording — only making sure the numbers are
  impossible to miss going into task 2.
- **Q1 (`rob_rt_prompt_5.md`) is still open, and now carries extra
  weight.** M4's own Gate item 2 ("`robust_baseline`+Raman vs
  `gordon`+Raman... agree at float32 tolerance") is impossible as literally
  written — `_build_robust_inputs` raises `ValueError` for
  `robust_baseline`+inelastic (an M1 decision: `Rrs_gordon` has no
  inelastic composition path at all) — and JXP has never answered which
  backend Gate item 2 actually meant. M4's Q8 finding above means **Q1's
  answer alone will not make Gate item 2 pass**: even substituting the
  intended-seeming backends (`robust_ztt`, matched Ed), the inelastic
  comparison at `5e-4` is unreachable regardless of which backend is
  named — the gap is structural, not a backend-choice artifact. This is an
  unresolved **milestone-completion** question that predates M5 (it's
  M4's Gate, not M5's scope) and needs a real decision from JXP — either
  relax the tolerance for the inelastic comparison specifically, or treat
  it as informational/reported rather than gating (design intent already
  leans this way per this prompt's own "reported, not thresholded"
  framing for the throughput benchmark, task 3) — **not something M5
  should silently paper over by dropping the comparison or reframing it
  as passing.**
- **For contrast, the elastic path remains solid.** The *elastic*
  robust-vs-Gordon agreement (via `robust_baseline`, established in M1)
  measures ~2.3e-7 relative — comfortably inside any reasonable tolerance,
  many orders of magnitude tighter than the inelastic gap above. The
  problem is specifically and only the inelastic (Raman/fluorescence)
  terms; nothing about the elastic integration is in question.
- **What M4 built and verified — all mechanically correct, tested,
  working; the disagreement above is about physics fidelity, not
  plumbing bugs.** `rob_rt_prompt_5.md`'s Logs (tasks 1-4, M4 now fully
  complete) are the record; load-bearing for M5:
  - **Task 1 — `set_raman_Ed` stash (CQ2).** `aNWModel` gained
    `wave_Ed_raw`/`Ed_raw` (anw.py, beside `Ed_ratio_raman`),
    `set_raman_Ed` stashes the incoming pair verbatim (uncopied
    references — a live alias, not a copy) before computing the ratio
    exactly as before; `Ed_ratio_raman` verified bit-identical
    (`np.array_equal`) pre-/post-change. Backward compatible, no
    signature change, BING's own Raman path untouched.
  - **Task 2 — `Geometry.Ed` routing.** `_build_robust_inputs` now
    passes `Ed=(wave_Ed_raw, Ed_raw)` into `geom.to_robust(...)` when
    `include_Raman` and a pair is stashed; `Ed=None` (robust's packaged
    L23 default) otherwise. Measured live: passing a real pair vs
    `Ed=None` changes the robust Raman term by up to ~1.5-6.3% depending
    on sky shape (the seam is unambiguously live, five-plus orders of
    magnitude above float32 ULP noise). **Resolved finding (Q4)**:
    robust's `fluorescence_kernel` also consumes `Geometry.Ed`, so with
    both `include_Raman` and `include_Chl_fl` on, the one shared
    geometry feeds the stashed sky to fluorescence too; a
    fluorescence-only call (`include_Raman=False`) keeps `Ed=None` by
    design — not extended further, per Q4's reasoning.
  - **Task 3 — fluorescence `a_ph` guard + conftest isolation.**
    `_build_robust_inputs` now raises a clear `ValueError` (naming
    `include_Chl_fl`, the missing `a_ph`, and the fix) when fluorescence
    is requested but the a-model has no `a_ph` set — fires for `Exp`,
    `ExpFix`, `Cst`, `Every`, `ExpNMF`; free-Chl Bricaud models set
    `a_ph` implicitly and never trip it (Q6 maps the full can/cannot-fire
    surface). `test_evaluate_robust_ed.py` (the new
    `correct_atmosphere`-gated file) added to the `collect_ignore` list
    (conftest.py:84-88) — verified 228 tests still collect cleanly with
    `correct_atmosphere` shadowed absent, 270 with it present.
  - **Task 4 — notebook.** `nb/RT/rob_rt_coding_5.ipynb` (5 sections,
    executed with outputs) is the source of every number in this
    hand-off's dominant finding above, plus §5's explicit "why `5e-4` and
    not `1e-6`" note — the honest float32 floor applies to the elastic
    path and the seam-liveness check, not to the inelastic
    robust-vs-BING comparison.
- **Full-suite baseline entering M5**: **266 passed, 2 skipped, 2 failed**
  — re-verified live 2026-08-31 in `ocean14` (`pytest bing/tests/ -q`),
  matching M4 task 4's exit baseline exactly. The 2 failures are the same
  pre-existing `test_l23_inelastic.py` missing-fixture failures diagnosed
  in M0/Q10, not a regression.
- **Previous prompt** — `rob_rt_prompt_5.md` (M4, **now fully complete**:
  all 5 tasks done) and its Logs — and skim the Logs of prompts 1–4 for
  anything flagged "for M5".
- **Coding plan** — `docs/coding_plan/rob_rt_coding_plan.md` **M5** section
  and the **Definition of done**.
- **Design** — `docs/design/rob_rt_design.md` §6 (deprecations), §7.1 (the
  throughput accept/optimize decision), §7.6 (docs).
- **Q&A record** — `claude_prompts/rob_rt.md`: Design Q5 (kept, not
  recommended) and Q9 (`RT_correction` gone).
- **Docs to edit** — `bing/rt/raman.py` and `bing/rt/chl_fl.py` module
  docstrings; `bing/rt/defs.py` (`rt_dict_from_p` docstring);
  `.claude/skills/run-bing-fit/SKILL.md` (it documents `rt_dict` at lines
  38-41/110/157 today); `.claude/skills/inelastic-rrs/SKILL.md`;
  `bing/CLAUDE.md`'s rt-subpackage bullet.

## Prompts

1. Read this doc. Execute the 1st task in the "M5" section below — the
   docstring pointers. If you have any questions, ask me in the Q&A section
   below. Use Fable if you can. Log your work.
2. Read this doc. Execute the 2nd task — the docs and skill updates. Check
   my answers in Q&A; if you have additional questions, ask in Q&A. Use
   Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task — the throughput benchmark. Use
   Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the sweep and the full-suite gate.
   Use Fable if you can. Log your work.
5. Read this doc. Execute the 5th task — the explainer notebook. Use Fable
   if you can. Log your work.
6. Read this doc. Execute the closing task — confirm the coding plan's
   **Definition of done** item by item against what is actually on the
   `rob-rt-backend` branch, record the confirmation (with the benchmark
   numbers) in the Logs below, and note that the integration is ready for
   JXP's final review and merge. Use Fable if you can. Log your work.

## M5

### Tasks

1. **Docstring pointers only** (design §6): `bing/rt/raman.py` and
   `bing/rt/chl_fl.py` module docstrings note they serve the `gordon`
   backend and are no longer the recommended inelastic path — point to
   `robust.rt.inelastic` via `rt_backend`. No code deleted.

2. **Docs.** `rt_dict_from_p` docstring documents `rt_backend` (four
   values), `fit_Bp`, `Bp_value`. `.claude/skills/run-bing-fit/SKILL.md`
   gains the backend-selection knob in its config/pitfall sections (it
   documents `rt_dict` at lines 38-41/110/157 today).
   `inelastic-rrs/SKILL.md` gains a "recommended path" note — its current
   text describes the Gordon-only inelastic wiring; the robust inelastic
   path is now the recommended one. `CLAUDE.md`'s rt-subpackage bullet gets
   one line on the new backend.

3. **Throughput benchmark** (design §7.1 accept/optimize decision): a small
   script (suggest `dev/rob_rt/benchmark_backends.py`) timing `log_prob`
   through each backend at MCMC-realistic batch shapes; record calls/s vs
   `gordon` in the script's header. Report, not threshold — if
   `robust_hybrid` is grossly slower, that's an optimize decision for JXP,
   not a silent fix. Include the per-worker JIT-compile cost note for
   `fit_batch` (each ProcessPoolExecutor worker compiles its own cache).

4. **Sweep + full gate.** `grep -rn "RT_correction"` must be empty in
   `bing/` (any `papers/` hits were reported in M1, not edited);
   `grep -rn "rt_backend"` confirms docs coverage; stale line-number check
   on the design doc's citations touched by this work. Then the full
   `pytest bing/tests/` in `ocean14`.

5. **Notebook.** `nb/RT/rob_rt_coding_6.ipynb` (executed, with outputs):
   the capstone — the four-backend comparison on a real L23 fit (Rrs,
   retrieved IOPs, and where they differ), the benchmark numbers as a
   figure/table, and a short "how to choose a backend" section a future
   user can read cold.

### Gate

- Full `pytest bing/tests/` green in `ocean14` — with and without the
  `correct_atmosphere`-gated files.
- The benchmark script runs and its numbers are recorded (for the PR
  description).
- `RT_correction` grep empty under `bing/`.

**Definition of done** (from the coding plan, confirmed in the closing
prompt): all four `rt_backend` values fit end-to-end through
`fit_one`/`fit_batch`/`chisq_fit.fit` with geometry threaded, `theta_s`
required, free-or-fixed `B_p`, robust-side inelastic fed by BING's solar
spectrum, `RT_correction` gone, the Gordon path otherwise regression-pinned
untouched — pytest-green in `ocean14`, benchmarked, documented, on the
`rob-rt-backend` branch for JXP to review and merge.

## Q&A

**Q1 (task 1, Claude → JXP). REAL OPEN QUESTION — needs your confirmation
before task 2 writes the `inelastic-rrs/SKILL.md` "recommended path" note.**
Task 1's literal wording says `raman.py`/`chl_fl.py` "are no longer the
recommended inelastic path" and should "point to `robust.rt.inelastic`."
Read at face value and written unqualified, that phrasing endorses
`robust.rt.inelastic` as the *preferred* inelastic path over BING's own —
but M4's measured evidence (Q7/Q8, `rob_rt_prompt_5.md`, this prompt's own
"READ THIS FIRST" Context bullet) shows the opposite on accuracy: robust's
Raman/fluorescence terms diverge from BING's own Gordon+Raman/fluorescence
physics by ~11% max/~6% mean (Raman) and ~9-18% max/~6-7% mean
(fluorescence) on real L23 data — many orders of magnitude outside the
project's `rtol <= 5e-4` tolerance, and root-caused (not a numerics
artifact) to robust interpolating/clamping the emission-grid IOP spectrum
for excitation wavelengths rather than re-evaluating the true parametric
a/bb models at the real excitation grid. Writing an unqualified "no longer
recommended, use robust instead" would misstate the actual state of
validation. **What I did**: implemented the letter of the task — both
module docstrings now carry a factual "Backend note" pointing to
`robust.rt.inelastic` via `rt_dict['rt_backend'] = 'robust_ztt'` /
`'robust_hybrid'` (not `'robust_baseline'`, which has no inelastic
composition path) — but worded it as "a wired, selectable alternative,"
not a validated or endorsed replacement, and included the measured
disagreement numbers so a future reader isn't steered toward the
less-validated path without the caveat attached. **What I did not do**: I
did not invent a project decision either way — I did not write robust's
inelastic path as discouraged, and I did not leave the docstrings
unqualified/unchanged contrary to the task spec. **Needs your
confirmation**: is this conservative framing the right call, or did you
intend "no longer recommended" to be read as literally as written (i.e.
robust genuinely preferred despite the accuracy gap, perhaps because the
gap is expected to close later, or because some other consideration
outweighs the measured divergence)? This same tension applies with higher
stakes to task 2's `inelastic-rrs/SKILL.md` update, which is user-facing
guidance rather than an internal module docstring — please confirm before
that task is executed.

**Q2 (task 2, Claude → JXP). Carries Q1 forward — Q1 is still unanswered,
this is not a resolution of it.** JXP gave direct instruction to proceed
with task 2 despite Q1 (above) having no `>A.` response anywhere in this
doc. Per that instruction, task 2 was executed — but its wording tension is
the same as Q1's, at higher stakes: the task spec's literal text says
`inelastic-rrs/SKILL.md` should state "the robust inelastic path is now the
recommended one," and that skill doc is user-facing guidance a person reads
to choose a backend, not an internal module docstring. Writing that
sentence unqualified would tell a reader to prefer the less-accurate path
on the strength of a claim the M4 numbers don't support. **What I did**:
applied task 1's already-established conservative framing to
`inelastic-rrs/SKILL.md` consistently — not a new, independent judgment
call, and not the literal spec wording. The new section states the robust
alternative exists and is wired/selectable (`rt_backend='robust_ztt'`/
`'robust_hybrid'`), gives the measured disagreement numbers (Raman ~11%
max/~6% mean; fluorescence ~9-18% max/~6-7% mean, both far outside
`rtol<=5e-4`, root-caused to the excitation-grid interpolation issue), and
explicitly declines to write an unqualified recommendation — offering
instead the two factual, situational considerations (accuracy favors
`gordon`; internal consistency with an already-robust elastic backend is a
different, non-accuracy argument for the robust inelastic path) without
manufacturing a verdict between them. A provenance note in the skill doc
itself says why this wording departs from the task spec's literal text and
points back to Q1/Q2 here. **What I did not do**: I did not silently adopt
the literal "now the recommended one" wording just because Q1 had gone
unanswered for a full task cycle, and I did not invent a new resolution
independent of task 1's — I treated task 1's framing as the controlling
precedent for this milestone. **Needs your confirmation**: same question as
Q1 — is this the framing you intended, or did you mean "recommended"
literally (e.g. because the accuracy gap is expected to close, or some
other consideration should dominate)? Q1 itself remains open and
unanswered; this entry does not close it, only carries it forward through
task 2's execution.

**Q3 (task 3, Claude → JXP). Informational finding, not a decision
needed — flagged per this doc's own instruction to log genuine findings
in Q&A.** `dev/rob_rt/benchmark_backends.py`'s freshly-measured
first-call/JIT-compile costs for `robust_ztt`/`robust_hybrid` (~0.18-0.19 s
each, three repeat runs) come in well under M2's smoke-fit figure
(`rob_rt_prompt_3.md`: ztt ~0.35 s, hybrid ~1.1 s, the latter "Flax
emulator load dominating"). Root cause, as best determined without
further instrumentation: the two measurements are of genuinely different
things under genuinely different process states, not a contradiction —
M2 timed a full tiny end-to-end `fit_one` MCMC run's first step in
whatever process state that smoke-fit script started in (plausibly a
colder one: first-ever JAX call in that process, cold OS disk cache for
the packaged Flax emulator weights file), while this benchmark isolates
just the first `log_prob` call in a process that has already imported
`jax`/`robust` for other work in the same script (the `gordon` backend is
benchmarked first) and, on this machine across repeat runs, benefits from
a warm OS file-cache for the emulator weights. Manually confirmed the
order effect is small on this machine (a variant benchmarking
`robust_hybrid` as the very first robust call in a fresh process still
measured ~0.32 s, not ~1.1 s) — so process-freshness alone doesn't fully
explain the gap; OS disk-cache state for the emulator weights file is the
remaining, most likely explanation, but wasn't independently isolated
(e.g. by dropping the OS page cache between runs, which needs
privileges this session doesn't have). **Not something this task should
resolve or paper over**: both numbers are real, reproducible
measurements under their own stated conditions, and the design-doc
question this benchmark exists to inform (§7.1's accept/optimize call on
`robust_hybrid`) turns on warm-path throughput, not this first-call
figure, so the discrepancy doesn't change this task's headline finding.
Recorded here in case a future cold-process measurement (e.g. the very
first `fit_batch` worker on a machine with a cold disk cache) is needed
and someone wants to reconcile the two numbers rather than rediscover the
gap.

## Next

This is the **last milestone** — there is no `rob_rt_prompt_7.md`.
Integration completion is gated by the M5 Definition of Done in
`docs/coding_plan/rob_rt_coding_plan.md`; after the closing prompt confirms
it, the work is in JXP's hands for final review and merge of
`rob-rt-backend`.

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-31 (M5 task 1 — raman/chl_fl deprecation docstrings)

**Read/verified first.** `bing/rt/raman.py`'s module docstring (lines 1-19
pre-edit) documents the Bartlett/Walrafen/Desiderio/Mobley references and
the Ocean Optics Web Book link; `bing/rt/chl_fl.py`'s (lines 1-37 pre-edit)
documents the Gordon/Maritorena/Behrenfeld references plus a "Theory"
section on emission peaks/quantum yield. Neither had any existing note
about backend selection. Confirmed `bing.rt.defs.RT_BACKENDS = ('gordon',
'robust_ztt', 'robust_hybrid', 'robust_baseline')` (defs.py:12) and, in
`evaluate.py`'s `_build_robust_inputs`, that `rt_backend == 'robust_baseline'`
combined with `include_Raman`/`include_Chl_fl` raises `ValueError`
(evaluate.py:442-447) — i.e. only `robust_ztt`/`robust_hybrid` actually
carry an inelastic composition path, matching M1's Q2 and this milestone's
Q1 framing. Confirmed `robust.rt.inelastic` is a real, importable module at
`/Users/xavier/Oceanography/python/retrieve-or-bust/robust/rt/inelastic.py`
(read-only reference repo, not edited).

**Read the Context "READ THIS FIRST" bullet and the M4 numbers before
writing anything**, per this prompt's own instruction — the elastic
robust-vs-Gordon agreement (~2.3e-7 relative) is essentially exact, but the
inelastic agreement (Raman: 11.4% max/5.6% mean; fluorescence: 9.2-18.5%
max/5.8-7.2% mean, both on real L23 data) is several orders of magnitude
outside the project's `rtol <= 5e-4` tolerance and is a genuine physics-
composition difference (robust interpolates/clamps the emission-grid IOP
spectrum for excitation wavelengths instead of re-evaluating the true
parametric a/bb models at the real excitation grid), not a numerical
artifact.

**Implemented** (bing repo, `rob_rt` branch; 2 files, purely additive,
docstring-only — confirmed via `git diff --stat`: `bing/rt/chl_fl.py` +19,
`bing/rt/raman.py` +18, no other lines touched, no code/signature/behavior
changes):

- `bing/rt/raman.py` — appended a "Backend note" section to the module
  docstring, after the existing "References" section (new lines ~19-36).
  States: (a) this module implements BING's own Raman physics and serves
  the `gordon` `rt_backend`; (b) an alternative exists via
  `robust.rt.inelastic`, selectable via `rt_dict['rt_backend'] =
  'robust_ztt'`/`'robust_hybrid'` (explicitly not `'robust_baseline'`,
  which has no inelastic path); (c) that alternative's excitation-grid
  approximation was measured on real L23 data to disagree with this
  module's Raman term by up to ~11% max/~6% mean, well outside `rtol <=
  5e-4`; (d) this module remains the physics implementation in active use
  for `gordon` — the robust alternative is "a wired, selectable option,
  not a validated equivalent replacement."
- `bing/rt/chl_fl.py` — same structure and placement (new lines ~35-53),
  substituting the fluorescence-specific numbers (~9-18% max/~6-7% mean
  depending on sky-irradiance case) and "fluorescence" for "Raman"
  throughout.

**Why the conservative framing, not the task's literal "no longer
recommended" wording**: written unqualified, "no longer the recommended
inelastic path, point to `robust.rt.inelastic`" reads as an endorsement of
robust's inelastic path as more correct/preferred — but the measured
evidence shows the opposite on accuracy (elastic: essentially exact;
inelastic: real, large, structural disagreement, per this prompt's own
"READ THIS FIRST" framing). I was explicitly told not to resolve this
tension by inventing new project direction in either direction (not
silently downgrading robust's status beyond what's measured, and not
leaving the docstrings as the unqualified spec literally states). So the
docstring note states the alternative exists and is wired/selectable (a
true, verified fact — M4 landed it) without characterizing it as
"recommended" or superior, and attaches the actual measured numbers so a
reader can judge for themselves. Flagged as Q1 above — this is a real
open question for JXP, not a decision I made unilaterally.

**Tested** (`ocean14`, `pytest bing/tests/ -q`): **266 passed, 2 skipped, 2
failed** — identical to the M5 entering baseline recorded in this
prompt's Context section (266/2/2), exit code non-zero only because of the
2 pre-existing `test_l23_inelastic.py` failures (missing
`l23_inelastic_fixture.npz`, diagnosed in M0/Q10, not a regression). Zero
tests added, zero outcomes changed — consistent with a pure docstring
edit.

**Next**: task 2 — `rt_dict_from_p` docstring, `run-bing-fit/SKILL.md`,
`inelastic-rrs/SKILL.md`, and `bing/CLAUDE.md`'s rt-subpackage bullet. The
same "recommended path" wording tension applies there, at higher stakes:
`inelastic-rrs/SKILL.md` is user-facing guidance a person reads to decide
which path to use, not an internal module docstring, so the task-2 note
needs the same accuracy caveat attached at least as prominently — pending
JXP's confirmation on Q1 above before that wording is finalized.

### 2026-08-31 (M5 task 2 — docs and skill updates)

**Proceeded under direct instruction despite Q1 still open.** No `>A.`
response to Q1 appears anywhere in this doc. JXP instructed proceeding with
task 2 anyway; per that instruction this task applied task 1's already-
established conservative framing to every piece touching the "recommended
path" wording, rather than treating the passage of time as implicit
sanction of the task spec's literal wording. Logged as Q2 above — Q1 itself
is left untouched/unresolved.

**Piece 1 — `rt_dict_from_p` docstring (`bing/rt/defs.py`).** Read the
current function (lines 35-69 pre-edit) and the surrounding module: exact
current keys/defaults confirmed by reading the code, not recalled from
memory. `RT_BACKENDS = ('gordon', 'robust_ztt', 'robust_hybrid',
'robust_baseline')` (defs.py:12). `rt_dict_from_p` builds `rt_dict['rt_backend']
= getattr(p, 'rt_backend', 'gordon')`, `rt_dict['fit_Bp'] = getattr(p,
'fit_Bp', False)`, `rt_dict['Bp_value'] = getattr(p, 'Bp_value', 0.01)`
(defs.py:64-66) — confirmed exact key names, types (str/bool/float), and
defaults directly from source, not guessed. Cross-checked
`calc_Rrs_from_models_robust` in `bing/evaluate.py` (lines 420-448) for the
exact `robust_baseline`+inelastic raise (`ValueError`, message quoting
`robust.rt.baselines.Rrs_gordon` takes no `inelastic` argument) and
`validate_rt_dict` (defs.py:72-125) for the `fit_Bp`+`'gordon'`,
missing-`geom`, and `robust_hybrid` wavelength-range raises. **Wrote**: a
full `Args:`/`Returns:` docstring rewrite documenting all four
`rt_backend` values (what each does, which support
`include_Raman`/`include_Chl_fl`, `robust_baseline`'s elastic-only
restriction and its exact raise condition), `fit_Bp` (bool, requires a
non-`'gordon'` backend, linear-space sampling range via
`BP_PRIOR_PMIN`/`BP_PRIOR_PMAX` = [0.004, 0.05]), and `Bp_value` (float,
default 0.01, fixed value or walker-ball seed). Also folded in one
paragraph under the `rt_backend` docs carrying the same M4 accuracy numbers
(Raman ~11% max/~6% mean; fluorescence ~9-18% max/~6-7% mean; elastic
~2.3e-7) with a pointer to `inelastic-rrs/SKILL.md`, so a reader who only
opens this docstring still sees the caveat rather than a bare list of
values. Purely additive — no code/behavior/signature changes.

**Piece 2 — `.claude/skills/run-bing-fit/SKILL.md`.** Verified the doc's
own line citations (38-41/110/157) against the file as it stands: line 41
is `rt_dict = rt_defs.rt_dict_from_p(p)` in the canonical script, line 110
is the `**kwargs` common-keys list under "Picking a standard combination,"
line 157 is the `rt_dict missing` pitfall — all three still accurate, no
drift since M0-M4 (this file was untouched by them, matching the spec's
"may still be accurate" hedge). **Wrote**: a new "### RT backend
selection" subsection immediately after the common-keys line, with a
table (`rt_backend`/`fit_Bp`/`Bp_value` — type, default, meaning), a short
code example showing them passed into `standard.expb_pow(...)`, a note
that robust backends require `geom=ObsGeometry(...)` (citing
`validate_rt_dict`'s raise), and a pointer to `inelastic-rrs/SKILL.md` for
the accuracy caveat rather than duplicating it (per this task's own
instruction to keep this file operational/mechanical, not a second copy
of the accuracy framing). Added five new pitfalls: `robust_baseline` +
inelastic raises; missing `geom` raises; `fit_Bp=True` with `'gordon'`
raises; `robust_hybrid`'s first-call JIT-compile cost (~1s) and the
per-`fit_batch`-worker compile-cache note from the Context section;
`robust_hybrid`'s wavelength-range restriction (350-750 nm). Kept this
file free of "recommended"/accuracy-comparison language, per the task's
explicit instruction that this tension belongs in `inelastic-rrs/SKILL.md`
instead.

**Piece 3 — `.claude/skills/inelastic-rrs/SKILL.md` (highest stakes).**
Read the file in full (152 lines pre-edit) — confirmed it describes only
the `gordon`-backend wiring (`bing.rt.raman`/`bing.rt.chl_fl`,
`calc_Rrs`/`calc_Rrs_with_raman`/`calc_Rrs_with_fluorescence`), with no
existing mention of `rt_backend` or the robust alternative anywhere in the
file. **Wrote**, immediately after the three-forward-model table (before
"Turning Raman on for a fit"): one sentence framing the existing content
as the `gordon`-backend path specifically, then a new "## Alternative
inelastic path: `robust.rt.inelastic`" section stating (a) the alternative
exists, is reachable via `rt_dict['rt_backend'] = 'robust_ztt'`/
`'robust_hybrid'` (not `'robust_baseline'`, which raises), and is wired/
tested by M0-M4; (b) the measured numbers verbatim from this prompt's
Context section (Raman 11.4% max/5.6% mean; fluorescence 9.2%/5.8% mean to
18.5%/7.2% mean depending on Ed-matching; elastic ~2.3e-7 for contrast);
(c) the root cause (excitation-grid interpolation/clamping vs. true
parametric re-evaluation); (d) an explicit statement that this note does
**not** recommend one path over the other, followed by two factual,
situational bullets (accuracy favors `gordon`; internal consistency with
an already-robust elastic backend is a different, non-accuracy argument
for the robust inelastic path) with no manufactured verdict between them;
(e) an explicit line telling the reader not to treat robust's inelastic
path as a drop-in more-accurate substitute. Closed with a provenance note
(italicized) stating this wording departs from the task spec's literal
"now the recommended one" text, why, and pointing to Q1/Q2 in this doc.
This is the section that most directly answers the "how does the
`inelastic-rrs/SKILL.md` note actually read" question this prompt's Context
section raised — logged in full above rather than paraphrased, since the
exact wording is the consequential artifact of this task.

**Piece 4 — `bing/CLAUDE.md`'s rt-subpackage bullet.** Found the bullet
under "Module Organization" → `bing/rt/` (the `defs.py` line item, "Shared
definitions/constants for the subpackage"). **Wrote**: extended that single
line to also name the `rt_dict['rt_backend']` knob and its four values,
staying to one line/one bullet as the task spec's own sizing calls for —
no accuracy claims here, per instruction (that belongs in the skill docs).

**Tested** (`ocean14`, `pytest bing/tests/ -q`): **266 passed, 2 skipped, 2
failed** — identical to the M5 entering baseline and to task 1's exit
state. The 2 failures are the same pre-existing `test_l23_inelastic.py`
missing-fixture failures (M0/Q10), not a regression. Zero tests
added/changed, consistent with a doc-only task across four files (one
Python docstring, two Markdown skill files, one Markdown CLAUDE.md).

**Why the conservative framing again, not the task's literal wording**:
same reasoning as task 1's Q1 entry, now applied to user-facing skill
documentation rather than an internal docstring — the measured M4 evidence
does not support an unqualified "robust is now recommended" claim for the
inelastic terms, and task 2 was explicitly instructed to apply task 1's
resolution consistently rather than make an independent call. Logged as Q2
above; Q1 remains open.

**Next**: task 3 — the throughput benchmark script (`dev/rob_rt/
benchmark_backends.py` suggested), timing `log_prob` through each backend
at MCMC-realistic batch shapes, reporting calls/s vs. `gordon` and the
per-`fit_batch`-worker JIT-compile cost — evidence reported, not
thresholded, per this prompt's explicit instruction.

### 2026-08-31 (M5 task 3 — throughput benchmark)

**Read/verified first.** `bing/fitting/inference.py`'s `log_prob`
signature (`params, models, Rrs, varRrs, rt_dict, geom=None`) and its full
body (B_p tail peel, model + B_p prior evaluation, dispatch to
`calc_Rrs_from_models`/`calc_Rrs_from_models_robust`, Gaussian likelihood
reduction) — confirmed the prior/likelihood overhead around the forward
call is real and small, so benchmarking `log_prob` itself (not the bare
forward call) is the faithful per-step MCMC cost. `run_emcee`
(inference.py) builds `emcee.EnsembleSampler(nwalkers, ndim, log_prob,
args=[models, Rrs, varRrs, rt_dict, geom])` with **no** `pool=` (commented
out) — confirmed against emcee's own default serial-map behavior that this
calls `log_prob` once per walker per step with a single 1-D `params`
vector, never a batch of walkers in one call. This settles "MCMC-realistic
batch shape" concretely: unbatched, single-walker calls, exactly what the
prompt's Context section anticipated but did not itself resolve. Also read
`fit_one`/`fit_batch` to confirm nothing upstream ever batches walkers
before calling `log_prob`.

**Script**: `dev/rob_rt/benchmark_backends.py` (new file, `dev/rob_rt/`
directory created). Builds one ExpBricaud+Pow model pair on a 61-band
400-700 nm grid (robust_hybrid-legal), a noiseless synthetic
`(Rrs, varRrs)` pair from the Gordon forward model, and
`geom=ObsGeometry(theta_s=30.)` — the same recipe
`test_evaluate_robust.py`'s `robust_models`/`threading_setup` fixtures
use, not a new ad-hoc setup. For each of the four `rt_defs.RT_BACKENDS`
values it: (1) clears `evaluate._robust_forward_jit`'s `lru_cache` (robust
backends only) and times one call to `log_prob` as the first-call/
JIT-compile cost; (2) times `N_WARM=2000` further calls, each with a tiny
(0.1% of the parameter value) random jitter on the length-5 `params`
vector so the loop resembles a sampler walking nearby posterior values
without ever changing shape/dtype (no JAX retrace risk), and reports
calls/s. No `assert`, no threshold, no exit-code-based pass/fail anywhere
in the script — printed evidence only, per the working agreement quoted
in this prompt's Goals/Working-agreements sections.

**Measured numbers** (`ocean14`, this script, three repeat runs agreeing
to within a few percent; canonical run transcribed in the script's own
header docstring):

| backend | warm calls/s | vs gordon | first-call cost (s) |
|---|---|---|---|
| gordon | 38936 | 1.000x | 0.0001 (no JIT) |
| robust_ztt | 5991 | 0.154x | 0.182 |
| robust_hybrid | 5667 | 0.146x | 0.181 |
| robust_baseline | 7274 | 0.187x | 0.019 |

All three robust backends land at roughly **15-19% of gordon's per-call
throughput** (~5.3-6.9x slower per `log_prob` call) at this batch shape.
Notably, `robust_hybrid` (0.146x) is not dramatically worse than
`robust_ztt`/`robust_baseline` (0.154x/0.187x) — the learned-emulator
correction is not the dominant extra cost among the robust backends; the
~5-7x gap vs. `gordon` looks like a property of dispatching through
JAX/robust.rt at all (device dispatch + per-call trace overhead, even
post-compilation) rather than something specific to the hybrid emulator.
Per the working agreement, this is reported as evidence for JXP's
accept/optimize call, not treated as a problem this task should fix.

**First-call/JIT-compile cost, measured fresh here** (not just cited from
M2): ztt/hybrid ~0.18-0.19 s, baseline ~0.019 s — all comfortably small in
absolute terms. This is noticeably lower than M2's smoke-fit figure for
hybrid (`rob_rt_prompt_3.md`: ~1.1 s, "Flax emulator load dominating").
Investigated (not left unexplained): a variant run measuring
`robust_hybrid` as the very first robust-backend call in a fresh process
(rather than after `gordon`/`robust_ztt` in the same script) still only
measured ~0.32 s — so process order alone doesn't explain the gap. Most
likely explanation, not independently isolated (would need to drop the OS
page cache, which this session can't do): M2's number was measured in a
colder process state (first-ever JAX call, cold disk cache for the
packaged Flax emulator weights file) than this benchmark's repeat-run
environment. Both numbers are legitimate under their own conditions;
recorded as **Q3** in the Q&A section above as an informational finding,
not a contradiction needing resolution — the design question this
benchmark exists to inform (§7.1's hybrid accept/optimize call) turns on
warm-path throughput, which both this script and M2 agree is the number
that matters, not the first-call figure.

**Per-worker `fit_batch` JIT-compile-cost note** (included in the script's
docstring and as a printed message, per the task spec): `fit_batch` uses a
`ProcessPoolExecutor`, and each worker process gets its own fresh,
initially-empty `_robust_forward_jit` module-level `lru_cache` — caches
are process-local and never shared via `ProcessPoolExecutor`'s pickling.
So a `fit_batch(n_cores=N)` robust-backend run pays the first-call
JIT-compile cost measured above **N times over, once per worker process**,
not once for the whole batch. Stated as a fact for JXP's awareness only —
the script does not attempt to pre-warm workers, share a compilation cache
across processes, or otherwise work around this, per the task's explicit
instruction that this is not something to fix here.

**Tested** (`ocean14`, `pytest bing/tests/ -q`): **266 passed, 2 skipped, 2
failed** — identical to the M5 entering baseline and to tasks 1-2's exit
state. The 2 failures are the same pre-existing `test_l23_inelastic.py`
missing-fixture failures (M0/Q10), not a regression. This is a `dev/`
script, not test code, and adds no pytest tests; confirmed importing/
running it has no import-time side effects on the suite (ran the full
suite once after adding the script, from a clean process).

**Gate item satisfied**: "The benchmark script runs and its numbers are
recorded (for the PR description)" — `dev/rob_rt/benchmark_backends.py`
runs cleanly end-to-end, prints the table above, and the same numbers are
written into the script's own header docstring (not placeholder text) so
they're readable without re-running it.

**Next**: task 4 — the `RT_correction`/`rt_backend` grep sweep, the stale
line-number check on the design doc's citations, and the full-suite gate
run (already reconfirmed 266/2/2 as part of this task, but task 4 owns the
formal gate check).

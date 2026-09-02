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

7. I have issued a PR on GitHub.  Please see the comments and address them.  If you have any questions, ask me in the Q&A section below.  Use Fable if you can. Log your work.

8. I have answered the questions in the Q&A section below.  Please see the comments and address them.  Use Fable if you can. Log your work.

9. Please modify the CI tests to only include Python 3.12 and greater.  And add a new 3.14 test.  Use Fable if you can. Log your work.

10.  I just pushed this branch into `develop` and I see the docs are failing.  Please make a pass on the docs, making sure they are up to date for the new integration with RoB.  Use Fable if you can. Log your work.

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
>A. Leave that original code as is and we will discourage the community 
from using it

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
>A. Don't worry about the old code.

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

**Q4 (task 4, Claude → JXP). Informational finding, not a decision
needed — the design-doc citation check this task's spec calls for.**
`docs/design/rob_rt_design.md` cites specific `bing`/`robust` file:line
locations throughout §§2-7 (e.g. `inference.py:52` for `log_prob`,
`chisq_fit.py:123` for `fit_func`, `evaluate.py:94` for
`calc_Rrs_from_models`, `inference.py:93-94` for the `aparams`/`bparams`
split, `inference.py:179-183, 215` and `inference.py:480-486` for the
observation-tuple unpacking, `inference.py:214-215` for `fit_one`).
**Every one of these `bing`-side line-number citations has drifted**,
by anywhere from ~1 to ~120+ lines, purely because the cited files grew
across M0-M5 (new docstrings, new backend-dispatch branches, new
`geom`/`Bp` plumbing) — confirmed by grepping each cited function/def
directly rather than trusting the old numbers: `log_prob` is now at
`inference.py:68` (not 52); `fit_func` is now at `chisq_fit.py:192` (not
123); `calc_Rrs_from_models` is now at `evaluate.py:136` (not 94); the
`aparams`/`bparams` split is now at `inference.py:161-162` (not 93-94);
`fit_one`'s def is now at `inference.py:269` (not 214-215), and its
tuple-unpack line the design doc meant is now at `inference.py:350` (not
215); `fit_batch`'s items docstring is now at `inference.py:748-759`
(not 480-486). By contrast, **every function/class *name* and module
*path* the design doc cites is still correct** — nothing it describes
has moved to a different file or been renamed, and the **`robust`-side
citations are almost all still exact** (`Geometry` class `types.py:302`,
`Geometry.nadir` `types.py:337`, `PhaseParams` `types.py:250`,
`Inelastic` `types.py:471`, `IOPs.from_total_bb` `types.py:131`,
`bing/setup.py`'s `install_requires` at line 23 — all confirmed exact by
direct grep), with only `robust.rt.forward` off by a small amount
(`hybrid.py:364` cited vs. `hybrid.py:383` actual — `retrieve-or-bust` is
a separate repo with its own independent history, not touched by this
integration, so this is not something M0-M5 caused). §6's `RT_correction`
citation (`bing/evaluate.py:204-209`) is correctly describing now-deleted
code and needs no update. **Not fixed here**: per this task's own
instruction, line-number drift this pervasive (specific citations
scattered across most of §§2-4) is substantive enough that silently
patching a couple of numbers would leave the rest stale and give a false
impression of freshness — reported as a finding for JXP rather than
edited. The underlying architecture/interface the design doc describes
remains accurate; only its line-number pointers need a pass if JXP wants
them kept current.

**Q5 (task 5, Claude → JXP). Informational note, not a decision needed —
flagging that this is the third surface of the still-open Q1/Q2 tension,
per this task's own explicit instruction to note that fact rather than let
it pass silently.** Task 5's spec calls for a "how to choose a backend"
section in `nb/RT/rob_rt_coding_6.ipynb` and says to apply Q1/Q2's
established conservative framing rather than resolve it. That section (the
notebook's §3) does exactly that: it states the measured facts (elastic
Rrs agreement is tight for `robust_baseline`, a real few-percent difference
for `robust_ztt`/`robust_hybrid`; the inelastic terms on the latter two
disagree with BING's own Gordon+Raman/fluorescence physics by ~11%
max/~6% mean (Raman) and ~9-18% max/~6-7% mean (fluorescence), several
orders of magnitude outside `rtol <= 5e-4`), and explicitly declines to
recommend one backend over another — closing with a "no single verdict"
subsection that hands the choice to the reader based on their own
priorities (speed vs. elastic-physics fidelity vs. inelastic-accuracy
risk), the same three-way framing task 2's `inelastic-rrs/SKILL.md` note
already uses. **This is the third place in this milestone where the same
Q1/Q2 tension surfaces** — after task 1's `raman.py`/`chl_fl.py` Backend
notes and task 2's `inelastic-rrs/SKILL.md` section, both logged under
Q1/Q2 above. **Q1 and Q2 remain open and unanswered** — this entry does
not resolve them, only records that task 5 carried the same established
framing forward for a third time, consistently, rather than treating the
notebook's own fresh measurements as license to make an independent call.
No new project decision was made here; if JXP's eventual answer to Q1/Q2
changes the framing, all three surfaces (task 1's docstrings, task 2's
skill doc, and this notebook's §3) would need the same update together.
>A. We will trust and adopt the `robust` code going forward.  But let's be 
clear about the differences in the BING docs.

**Q6 (PR #27 CI, Claude → JXP). Needs your confirmation — which
`retrieve-or-bust` branch should CI track?** Fixing the CI failures
required installing `robust` into the workflow's environment, but
`robust/rt` **does not exist on `retrieve-or-bust`'s `main`** — `main`
holds only `robust/__init__.py`; the rt subpackage lives on the branches
`cdom-rt` (18 files, current), `inelastic-rt`/`inelastic-rt-staging` (17),
`RT`/`rt-elastic-prototype` (13), `vicc-proposal` (11). **What I did**:
pinned the workflow's clone to `--branch cdom-rt`, because it is exactly
the state your local `retrieve-or-bust` checkout is on (local HEAD
`0002b3e` == `origin/cdom-rt`) — i.e. the state every M0-M5 validation run
actually imported — and documented in the workflow comment that the pin
should be re-pointed (or dropped) once `robust/rt` merges to `main`. **The
caveat**: `cdom-rt` is a moving feature branch (active CDOM work), so a
future push there could break bing's CI through no fault of bing's own.
If you'd rather CI track a quieter branch (e.g. merge the rt work to
`main` first, or keep a dedicated `rt-stable` branch), say so and the
`--branch` value is a one-line change in both jobs of
`.github/workflows/tests.yml`.

>A. Pinning to `cdom-rt` is fine for now.  I'll redirect to `main` eventually.

**Q7 (PR #27 CI, Claude → JXP). Action needed from you (git):
`bing/tests/files/m3_fixed_bp_pin.npz` is gitignored and was never
committed.** The repo's `.gitignore` line 16 (`*.npz`) catches it; the
other two test fixtures (`l23_gordon_fixture.npz`,
`l23_inelastic_fixture.npz`) are tracked (evidently force-added at some
point), but the M3 pin never was — its generator's docstring says
"(committed)", which was aspirational. Found by reproducing CI against a
clean `git ls-files` export: `test_fit_Bp_false_matches_provisional_pin`
dies with `FileNotFoundError: ... m3_fixed_bp_pin.npz`. It does **not**
affect CI today (that test needs Hydrolight data and skips there), but it
fails for anyone running the suite on a fresh clone *with* data. Since
git actions are yours: please run
`git add -f bing/tests/files/m3_fixed_bp_pin.npz` when you commit this
work. Note the fixture was also **regenerated** this session (see the
2026-09-01 Logs entry: the old pin had frozen the robust chi-squared
optimizer's stalled-at-p0 output; only `chisq_robust_ztt` changed,
`chisq_gordon` and both MCMC chains verified byte-identical), so the file
on disk is the current, correct pin.

>A. I have added it to the Repo

**Q8 (PR #27 CI, Claude → JXP). Decision when convenient — CI is green
in the meantime, but the Python-floor mismatch between the two repos
should eventually be reconciled properly, and both clean resolutions are
yours to make.** After the Q6/Q7 round, PR #27's head (`eb4792c`) had
three of four checks green but **Tests (Python 3.11) failing in the
"Install dependencies" step**. Reproduced in a fresh Python 3.11.16
environment against a clean `git archive HEAD` export, running the
workflow's install commands verbatim: every step passes until
`pip install --no-deps -e ../retrieve-or-bust`, which dies with
`ERROR: Package 'retrieve-or-bust' requires a different Python: 3.11.16
not in '>=3.12'`. Root cause: `retrieve-or-bust/setup.py:46` declares
`python_requires = '>=3.12'` (present since the repo's initial commit
`06f6a38`), while `bing`'s stated minimum is 3.11 (`setup.py:22`) and
the CI matrix tests 3.11 — and since PR #27 bing imports robust
unconditionally, so a 3.11 bing cannot get robust installed at all. pip
enforces Requires-Python even under `--no-deps`, and a direct editable
install has no older version to backtrack to, hence the hard failure
(3.12/3.13 satisfy the floor and never see it). **What I did**: measured
that the floor is metadata-only for what bing exercises — robust's
source has no 3.12-only syntax, and on 3.11 (which resolves jax 0.10.2,
since jax >=0.11 itself requires Python >=3.12) every workflow import
check passes and the CI-equivalent suite gives **84 passed /
151 skipped**, identical to 3.12/3.13 — then added
`--ignore-requires-python` to that one install line in the tests job,
with a comment explaining why and when to drop it. **The decision that
is yours**: the override is a bridge, not the proper resolution. Either
(a) lower `retrieve-or-bust`'s floor to `'>=3.11'` (one line in its
setup.py; the 3.11 measurements above are the supporting evidence), or
(b) decide bing itself now effectively requires >=3.12 (bump
`bing/setup.py`, drop 3.11 from the matrix). Whichever you choose, the
`--ignore-requires-python` flag should be removed at that point — the
workflow comment points back to this entry.

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

### 2026-08-31 (M5 task 4 — sweep and full-suite gate)

**Piece 1 — `RT_correction` sweep.** `grep -rn "RT_correction" bing/` is
**not** literally empty (6 hits: `test_evaluate.py:305,335`,
`test_evaluate_robust.py:38,722,724-725,743` region,
`gen_l23_gordon_fixture.py:10,13`) — but every hit is a comment, test
name, or generator docstring *describing* the M1 deletion, not live code;
this is the exact same residual state M1's own task-4 log already
identified and accepted ("the remaining `bing/` grep hits are the new
test/generator docstrings describing the deletion itself, plus
docs/prompts history") — confirmed unchanged since M1, not a regression.
`grep -rn "RT_correction" papers/`: **7 hits in 2 files**
(`papers/biomass/Analysis/py/fitting.py`, `lowest_bbp.py`) — identical
count and files to M1's Q9 finding; not edited, per the working
agreement.

**Piece 2 — `rt_backend` docs-coverage sweep.**
`grep -rn "rt_backend" bing/ .claude/skills/ CLAUDE.md` confirms every
surface named in this milestone's Context section documents the knob:
`bing/rt/defs.py` (comment + `rt_dict_from_p`/`validate_rt_dict`),
`bing/rt/raman.py` and `chl_fl.py` (task 1's Backend-note docstrings),
`bing/evaluate.py`/`bing/fitting/inference.py`/`chisq_fit.py` (dispatch +
docstrings), `.claude/skills/run-bing-fit/SKILL.md` (the new "RT backend
selection" table + pitfalls from task 2), `.claude/skills/inelastic-rrs/
SKILL.md` (the "Alternative inelastic path" section from task 2), and
root `CLAUDE.md` (the `defs.py` bullet extended in task 2; note: there is
no separate `bing/CLAUDE.md` — the file lives at the repo root and was
matched there). No gaps found; this is a confirmation of tasks 1-2's own
Logs, not new work.

**Piece 3 — design-doc citation check (`docs/design/rob_rt_design.md`).**
Read the full 420-line doc and spot-checked its `bing`/`robust` file:line
citations against current source by grepping each cited
function/class directly (not trusting the old numbers). Finding, filed as
**Q4** above: every `bing`-side line-number citation in §§2-4 has drifted
(by ~1 to 120+ lines) because the cited files grew across M0-M5 —
`log_prob` (cited `inference.py:52`) is now at line 68; `fit_func` (cited
`chisq_fit.py:123`) is now at line 192; `calc_Rrs_from_models` (cited
`evaluate.py:94`) is now at line 136; the `aparams`/`bparams` split
(cited `inference.py:93-94`) is now at lines 161-162; `fit_one`'s def
(cited `inference.py:214-215`) is now at line 269, with its tuple-unpack
now at line 350; `fit_batch`'s items docstring (cited
`inference.py:480-486`) is now at lines 748-759. Every cited function
*name* and module *path* is still correct — nothing moved files or was
renamed, so this is pure line-drift from the files growing, not
architectural staleness. The `robust`-side citations are almost all
still exact (`Geometry` `types.py:302`, `Geometry.nadir` `types.py:337`,
`PhaseParams` `types.py:250`, `Inelastic` `types.py:471`,
`IOPs.from_total_bb` `types.py:131`, `bing/setup.py:23` — all confirmed
exact), with `robust.rt.forward` (cited `hybrid.py:364`) off by ~19
lines (actual 383) — `retrieve-or-bust` is a separate, untouched-by-us
repo with its own independent history, so this drift did not come from
M0-M5. §6's `RT_correction` citation (`bing/evaluate.py:204-209`)
correctly describes now-deleted code. Per this task's explicit
instruction, **not rewritten** — the drift is pervasive enough (scattered
across most of §§2-4) that it is a real finding for JXP, not a trivial,
unambiguous fix; filed as Q4.

**Piece 4 — full-suite gate (the substantive piece).** Investigated the 2
pre-existing `test_l23_inelastic.py` failures rather than accepting them.
Read the test file and the fixture generator
(`bing/tests/files/gen_l23_inelastic_fixture.py`, already present in the
repo, docstring-documented, extracting 40 L23 HydroLight scenes at solar
zenith 30 deg from `Hydrolight{1,2,4}30.nc`). The generator needs the L23
store at `$OS_COLOR_DATA` (default
`/mnt/tank/Oceanography/data/Color/Loisel2023`, not present on this
machine) — but this session's `$OS_COLOR` env var
(`/Users/xavier/Projects/Oceanography/data/Color/`) already holds a local
`Loisel2023/` directory with the exact three files needed
(`Hydrolight130.nc`, `Hydrolight230.nc`, `Hydrolight430.nc`). Ran the
generator unmodified with `OS_COLOR_DATA="$OS_COLOR"
python gen_l23_inelastic_fixture.py` (`ocean14`) — **succeeded**, wrote
`bing/tests/files/l23_inelastic_fixture.npz` (69 kB, 40 scenes). Re-ran
the two previously-failing tests alone: **2 passed**
(`test_raman_correction_matches_l23`, `test_fluorescence_matches_l23`).
The fixture is `*.npz`-gitignored (confirmed via `git check-ignore -v`)
and left untracked, per the established M1/Q10 pattern — no `git add -f`
run, that is JXP's call.

**Full suite, twice, per the Gate's "with and without `correct_atmosphere`"
requirement** (`ocean14`, `pytest bing/tests/ -q`):
- **Without shadowing `correct_atmosphere`** (normal env, `correct_atmosphere`
  present and importable): **268 passed, 2 skipped, 0 failed** (147.94s;
  270 collected). This is a **genuinely green suite** — the two
  pre-existing failures are gone, not papered over; 268 = 266 + 2 (the
  two now-passing `test_l23_inelastic.py` tests), matching the doc's own
  predicted outcome exactly.
- **With `correct_atmosphere` shadowed absent** (scratchpad
  `correct_atmosphere.py` containing only `raise ImportError(...)`,
  prepended via `PYTHONPATH`, following M4 task 3's exact recipe;
  `ocean14` itself untouched): `--collect-only` → **228 tests collected,
  zero collection errors**, the same four files
  (`test_evaluate.py`/`test_io.py`/`test_l23_fitting.py`/
  `test_evaluate_robust_ed.py`) dropped via `conftest.py`'s
  `collect_ignore`, matching M4's own 228/270 baseline exactly. Full run:
  **225 passed, 3 skipped, 0 failed** (13.33s; 225+3=228, consistent with
  collection). The skip count is 3 here vs. 2 unshadowed because two
  `test_bbnw.py` tests skip via a runtime `bing.fitting.l23 unavailable`
  path when `correct_atmosphere` is absent (verified with `-rs`) —
  expected, not a new failure; 268 passed (unshadowed) − 225 passed
  (shadowed) = 43, and 270 − 228 = 42 collected difference, consistent
  with the 42 tests living in the 4 dropped files (one of which apparently
  double-counts against a skip in the reconciliation above; not
  investigated further since both runs are independently green and the
  Gate only requires clean collection + a green run in both conditions,
  both of which hold).

**Gate status — reported exactly as measured, not rounded up**: **Full
`pytest bing/tests/` is genuinely green in `ocean14`, both with and
without `correct_atmosphere`** (268/2/0 and 225/3/0 respectively, zero
failures either way). `RT_correction` grep is empty of live code under
`bing/` (6 residual doc/comment hits, identical to M1's own accepted
baseline). This is the first task in the whole M0-M5 series where the
suite is actually green rather than carrying the 2 known pre-existing
failures forward — achieved by generating the missing fixture from its
already-existing, already-reviewed generator script (a data-artifact
generation, not new test logic), per this task's explicit instruction and
consistent with JXP's own Q10 (M1) intent to have this fixture generated
and then committed.

**Next**: task 5 — the capstone notebook `nb/RT/rob_rt_coding_6.ipynb`:
the four-backend comparison on a real L23 fit (Rrs, retrieved IOPs, and
where they differ), the benchmark numbers as a figure/table, and a
"how to choose a backend" section. Task 4 leaves two open items for
JXP: **Q4** (design-doc line-number citations, informational) and the
still-open **Q1/Q2** (inelastic-path "recommended" wording) — neither
blocks task 5 or task 6.

### 2026-08-31 (M5 task 5 — capstone notebook)

**Built and executed** `nb/RT/rob_rt_coding_6.ipynb` (22 cells: 10 code, 12
markdown) via a scripted `nbformat` builder followed by
`jupyter nbconvert --to notebook --execute --inplace` (`ocean14`) — the same
build pattern prior M1-M4 notebooks in this series used. Smoke-tested the
four-backend `fit_one` comparison and the live benchmark re-run as
standalone scripts first, confirmed both finished quickly and produced
sane numbers, before committing to the full notebook build. After
execution, read the saved `.ipynb` back and confirmed: 10/10 code cells
carry sequential `execution_count`s 1-10 (the one code cell with no printed
output is the imports-only setup cell, which has no `print` calls by
design); every other code cell has real stream and/or `image/png` output
(three figures, all with genuine `image/png` data, not placeholders); every
markdown-cell numeric claim was re-diffed against its own preceding code
cell's actual printed output (not written before the cell was run, not
carried over from memory of the smoke test) and corrected in two places
where an initial draft's numbers didn't match the executed output exactly
(see below) — the recurring failure mode this project's own conventions
flag, checked deliberately rather than assumed clean.

**Section-by-section, with this notebook's own real measured numbers**
(L23 idx=170, PACE grid, 61 bands 400-700 nm, elastic-only,
`nsteps=200`/`nburn=50`/16 walkers, identically reseeded
(`np.random.seed(2026)`) before every backend's `fit_one` call — a
smoke-scale fit, matching M2's own notebook, not a converged posterior):

- **Setup**: `prep_one_l23(p, idx=170)` — Chl = 0.1306 mg/m3, Y = 2.1521,
  `p0 = [Adg, Sdg, Aph, Bnw, beta] = [-1.8799, 0.017, -1.8799, -3.4926,
  1.0]`, `geom=ObsGeometry(theta_s=30.)`, `Bp_value=0.014`.
- **§1a-1b**: all four backends (`gordon`, `robust_ztt`, `robust_hybrid`,
  `robust_baseline`) ran end-to-end through `fit_one`, all four chains
  `(200, 16, 5)` and fully finite. Posterior-median parameters:
  `robust_baseline` reproduces `gordon`'s median exactly (max
  |difference| = 0.00e+00, identical seed + near-identical forward model);
  `robust_ztt`/`robust_hybrid` diverge most visibly in `Bnw`/`beta`
  (`beta`: 1.1287 gordon vs. 0.9039 ztt vs. 1.2125 hybrid).
- **§1c (Rrs at the posterior median, each backend's own forward model)**:
  `robust_baseline` vs. `gordon` — max 2.31e-07 / mean 7.93e-08 relative
  (matches M1's original ~2.3e-7 finding); `robust_ztt` vs. `gordon` — max
  5.58e-02 (5.6%) / mean 1.69e-02 (1.7%); `robust_hybrid` vs. `gordon` —
  max 7.03e-02 (7.0%) / mean 1.76e-02 (1.8%). Consistent with M1-M4's
  repeated finding that the real robust backends predict Rrs a few percent
  above Gordon on real L23 data.
- **§1d (retrieved IOPs, where they differ)**: `robust_baseline` matches
  `gordon` exactly on both `anw` and `bbnw` (0.00e+00 relative, both
  quantities). `robust_ztt` vs. `gordon`: anw max 6.42e-02 (6.4%) / mean
  3.88e-02 (3.9%), bbnw max 3.22e-01 (32.2%) / mean 2.73e-01 (27.3%).
  `robust_hybrid` vs. `gordon`: anw max 1.55e-01 (15.5%) / mean 6.52e-02
  (6.5%), bbnw max 2.58e-01 (25.8%) / mean 2.42e-01 (24.2%). The concrete
  "where they differ" answer: a several-percent elastic Rrs difference
  resolves mostly into a substantially different retrieved $b_{b,nw}$
  (tens of percent) and only a modest difference in $a_{nw}$ (single-digit
  to teens of percent) — flagged in the notebook's own prose as partly
  attributable to the smoke-fit's short chain length (200 steps), not
  purely a backend-physics effect, since a genuinely converged posterior
  was out of this task's scope.
- **§2 (benchmark)**: re-ran `dev/rob_rt/benchmark_backends.py`'s own
  functions live inside the notebook (not a subprocess, not a
  transcription) — **this notebook's own fresh numbers**: gordon 38263.6
  calls/s (reference); robust_ztt 5705.5 (0.149x), first-call 0.0825 s;
  robust_hybrid 5425.9 (0.142x), first-call 0.1179 s; robust_baseline
  6884.4 (0.180x), first-call 0.0197 s. Warm-path calls/s agree with task
  3's recorded canonical run (38936/5991/5667/7274, ratios
  1.000x/0.154x/0.146x/0.187x) to within a few percent, consistent with
  that script's own "repeat runs agree to within a few percent" note.
  First-call/JIT-compile costs measured here for ztt/hybrid (0.08-0.12 s)
  came in lower than task 3's recorded ~0.18-0.19 s — not a new finding,
  attributed in the notebook's prose to the same process-warmth/OS-disk-
  cache effect already flagged as **Q3**: this notebook's benchmark cell
  runs after four MCMC fits have already warmed `jax`/`robust` in the same
  process, a warmer state than task 3's own script. Both figures cited
  accurately with their own provenance (this notebook's live run vs. task
  3's recorded run) rather than conflated. The per-`fit_batch`-worker
  JIT-compile-cost note (each `ProcessPoolExecutor` worker pays its own
  first-call cost, caches never shared across processes) is carried as
  prose, cited to task 3, not re-measured (re-measuring it would need an
  actual multi-worker `fit_batch` run, out of this task's scope).
- **§3 ("how to choose a backend")**: applies Q1/Q2's established
  conservative framing (gordon: fastest, fully self-consistent own
  physics, good default; robust_baseline: elastic parity check, no
  inelastic path, not obviously useful on its own; robust_ztt/hybrid: a
  real, deliberate elastic-physics difference at a real, measured speed
  cost, with the inelastic terms' measured accuracy gap stated plainly
  and cited (not re-measured here) rather than recommended either way) and
  ends with an explicit "no single verdict" subsection. Filed as **Q5**
  above: this is the third place in the milestone (after task 1's
  docstrings, task 2's skill doc) where the same Q1/Q2 tension surfaces,
  applied identically for consistency, not independently re-litigated.

**Numbers corrected during the re-diff pass** (the known recurring failure
mode this project's conventions call out explicitly): an initial draft of
§3's `robust_ztt`/`robust_hybrid` paragraph stated the JIT-compile cost as
"~0.18-0.21 s" (copied from memory of task 3's number before this
notebook's own benchmark cell had actually run) — corrected after
execution to cite both this notebook's own live figure (~0.08-0.12 s) and
task 3's recorded figure (~0.18-0.19 s) separately and accurately, per the
methodology note in §2. The throughput ratio in the same paragraph was
similarly corrected from a rounded "15-20%" to the exact measured range
(~14-18%, both runs) with each run's own number attributed to its own
source.

**What's next**: task 6 — the closing Definition-of-done confirmation
against the coding plan (`docs/coding_plan/rob_rt_coding_plan.md`), item by
item against what's actually on the `rob_rt` branch, recording the
confirmation (with the benchmark numbers) and noting the integration is
ready for JXP's final review and merge. Task 6 inherits three open,
unresolved items from tasks 1-5: **Q1/Q2** (inelastic-path "recommended"
wording, now surfaced three times — tasks 1, 2, and this notebook's §3),
**Q4** (design-doc line-number citation drift, informational), and this
task's own **Q5** (informational, notes the third surfacing of Q1/Q2) —
none of which block task 6's confirmation itself.

### 2026-09-01 (M5 closing — Definition of Done confirmation)

**Q1/Q2/Q5 answered by JXP.** JXP's `>A.` answers now appear above: Q1 —
"Leave that original code as is and we will discourage the community from
using it"; Q2 — "Don't worry about the old code."; Q5 — "We will trust and
adopt the `robust` code going forward. But let's be clear about the
differences in the BING docs." These supersede the conservative
"no-verdict" framing tasks 1, 2, and 5 applied to the three doc surfaces
below — the framing is now an actual recommendation, not a declined one,
while the measured accuracy-gap numbers stay exactly as they were measured
(per Q5, not softened, shortened, or removed).

**Piece (a) — doc-wording revision, three surfaces.** No code was changed
in any of the three; no numbers were changed, added, or removed — only the
verdict language around them.

- **`bing/rt/raman.py` / `bing/rt/chl_fl.py`** module docstrings: the
  "Backend note" section was rewritten from "a wired, selectable
  alternative, not a validated or endorsed replacement" to a "legacy
  module, kept for the `gordon` backend only" framing — states plainly
  that this module is BING's own **legacy** Raman/fluorescence physics,
  that the project has adopted `robust.rt.inelastic` (via
  `rt_dict['rt_backend'] = 'robust_ztt'`/`'robust_hybrid'`) as the
  recommended path going forward, that new work should use `robust`
  instead, and that this module is retained unmodified only for backward
  compatibility with existing `gordon`-backend code (per Q1/Q5, cited by
  section reference in the docstring itself). The measured disagreement
  numbers (Raman ~11% max/~6% mean; fluorescence ~9-18% max/~6-7% mean)
  are carried over verbatim, now framed as "documented for transparency,
  not as a reason to prefer this module." No code deleted, per Q1's
  explicit instruction — confirmed via `git diff --stat`: docstring-only
  changes in both files, no other lines touched.
- **`.claude/skills/inelastic-rrs/SKILL.md`**: the "Alternative inelastic
  path" section (task 2) is now "Recommended inelastic path" — opens by
  stating the project has adopted `robust.rt.inelastic` as recommended
  going forward and that new work should prefer `robust_ztt`/
  `robust_hybrid` over the legacy `gordon`-backend modules. The full
  measured-numbers paragraph (Raman 11.4%/5.6%; fluorescence
  9.2%/5.8%-18.5%/7.2%; elastic ~2.3e-7 contrast; root-cause explanation)
  is unchanged, word-for-word, from the pre-existing text — only the
  surrounding "does not recommend one path over the other" / situational
  bullets were replaced with "what this means in practice" guidance that
  actually recommends `robust_ztt`/`robust_hybrid`, while still telling
  the reader plainly not to expect the inelastic terms to match BING's
  legacy physics to the project's usual tolerance. The obsolete provenance
  note (explaining why the wording departed from the task spec) was
  removed — it no longer applies, since the wording now matches an actual
  recommendation.
- **`nb/RT/rob_rt_coding_6.ipynb` §3 ("How to choose a backend")**: read
  the cell's markdown source directly via `json.load` on the `.ipynb`
  (cell index 20, a plain-string `source` field, no list-of-lines
  wrapping) — confirmed every specific number the section cites (Rrs/IOP
  agreement percentages, throughput ratios, JIT-compile costs) is quoted
  from sections 1c/1d/2 earlier in the *same* notebook, computed and
  printed in code cells that execute *before* this markdown cell, not
  computed inside it — so no re-execution was needed, only a markdown-text
  rewrite. Replaced the "No single verdict" closing subsection and the
  "still-open Q1/Q2 tension" framing throughout with a "The recommendation"
  subsection reflecting JXP's actual decision: `robust_ztt`/`robust_hybrid`
  as the default recommendation for new work, `gordon` as a legacy,
  actively-discouraged-but-not-removed default, `robust_baseline` as a
  parity-check tool. Every measured number (Rrs agreement, IOP-retrieval
  differences, throughput ratios, JIT-compile costs, the inelastic
  accuracy-gap percentages) was left byte-for-byte identical to the
  original text — only rewrote the surrounding verdict language.
  **Verification the edit is valid**: after writing the new cell 20
  `source` via a `json.load`/`json.dump` round-trip, re-loaded the file
  with `json.load` (parses cleanly, `nbformat=4`, `nbformat_minor=5`, 22
  cells) and confirmed all 10 code cells retain their original sequential
  `execution_count`s (1-10) and original `outputs` (`n_outputs` unchanged
  per cell, e.g. cell 5 still carries 47 outputs) — nothing about the
  executed state was disturbed. Also ran
  `jupyter nbconvert --to notebook --stdout` on the file (`ocean14`) as an
  independent parse/structure check — completed without error.

**Piece (b) — Definition of Done confirmation, item by item, verified live
against the current repo state (not against the Logs' claims):**

1. **All four `rt_backend` values fit end-to-end through
   `fit_one`/`fit_batch`/`chisq_fit.fit`** — confirmed: `robust_baseline`,
   `robust_ztt`, and `robust_hybrid` all appear in
   `bing/tests/test_evaluate_robust.py` and
   `bing/tests/files/gen_m3_fixed_bp_pin.py`; task 4's fixture-generation
   fix (`l23_inelastic_fixture.npz`, regenerated from
   `gen_l23_inelastic_fixture.py`) is present on disk
   (`bing/tests/files/l23_inelastic_fixture.npz`) and its two dependent
   tests are part of the green suite below. **Satisfied.**
2. **Geometry threaded** — confirmed: `bing/evaluate.py:493-495` passes
   `geom.to_robust(Ed=(a_model.wave_Ed_raw, a_model.Ed_raw))` /
   `geom.to_robust()` into the robust dispatch, matching M2/M4's
   observation-tuple/`geom` threading. **Satisfied.**
3. **`theta_s` required** — confirmed: `bing/rt/defs.py:153,179` raise a
   setup-validation error naming `theta_s`/`geom=` explicitly when a robust
   backend is selected without it ("theta_s is never silently defaulted").
   **Satisfied.**
4. **Free-or-fixed `B_p`** — confirmed: `bing/rt/defs.py` defines
   `BP_PRIOR_PMIN`/`BP_PRIOR_PMAX` (lines 31-32), `rt_dict_from_p` sets
   `rt_dict['fit_Bp']`/`rt_dict['Bp_value']` (lines 129-130), and
   `validate_rt_dict` raises on `fit_Bp=True` with `rt_backend='gordon'`
   (lines 170-172) — the free/fixed `B_p` bookkeeping machinery from M3 is
   live. **Satisfied.**
5. **Robust-side inelastic fed by BING's solar spectrum** — confirmed:
   `bing/models/anw.py:496` defines `set_raman_Ed`, storing
   `wave_Ed_raw`/`Ed_raw` verbatim; `bing/evaluate.py:481-495` routes that
   stashed pair into `geom.to_robust(Ed=...)`, which is `Geometry.Ed` on
   the robust side. This is a mechanical routing fact, confirmed true
   independent of the accuracy gap (which is a separate, already-documented
   caveat, not conflated here). **Satisfied.**
6. **`RT_correction` gone** — confirmed live:
   `grep -rn "RT_correction" bing/` returns exactly the same 6 hits M1/M5
   task 4 already identified and accepted
   (`test_evaluate_robust.py:38,722,724-725,743`,
   `test_evaluate.py:305,335`, `gen_l23_gordon_fixture.py:10,13`) — all
   comments, test names, or generator docstrings describing the deletion,
   zero live code. **Satisfied.**
7. **Gordon path otherwise regression-pinned untouched** — confirmed:
   `bing/tests/files/l23_gordon_fixture.npz` exists on disk and is
   referenced by both `test_evaluate.py` and `test_evaluate_robust.py`.
   **Satisfied.**
8. **pytest-green in `ocean14`** — ran `pytest bing/tests/ -q` live, just
   now, in `ocean14`: **268 passed, 2 skipped, 0 failed, exit code 0**
   (148.93s) — identical to M5 task 4's recorded exit state; piece (a)'s
   doc-only edits changed no test outcome, confirmed directly rather than
   assumed. **Satisfied.**
9. **Benchmarked** — `dev/rob_rt/benchmark_backends.py` exists, runs, and
   its header docstring records the canonical numbers (verified present in
   the file, lines ~91-95): gordon 38936 calls/s (1.000x, first-call
   0.0001s); robust_ztt 5991 (0.154x, first-call 0.182s); robust_hybrid
   5667 (0.146x, first-call 0.181s); robust_baseline 7274 (0.187x,
   first-call 0.019s) — all three robust backends at roughly 15-19% of
   `gordon`'s warm-path throughput, reported as evidence per the working
   agreement, not thresholded. **Satisfied.**
10. **Documented** — piece (a) above confirms all three doc surfaces
    (`raman.py`/`chl_fl.py` docstrings, `inelastic-rrs/SKILL.md`, and the
    notebook's §3) now accurately reflect JXP's actual Q1/Q2/Q5 decision;
    tasks 1-2's other doc surfaces (`rt_dict_from_p` docstring,
    `run-bing-fit/SKILL.md`, `CLAUDE.md`'s rt bullet) were untouched by
    this closing task because they carry no "recommended path" language
    needing revision (confirmed by their own task-2 Log entries, which
    explicitly kept those files free of accuracy/recommendation framing).
    **Satisfied.**
11. **On the correct branch, ready for review/merge** —
    `git branch --show-current` returns **`rob_rt`** (not
    `rob-rt-backend`, the coding plan's stale original name — this
    discrepancy is cosmetic and already-known per every milestone's
    Working-agreements correction, not a gap). Checked twice during this
    closing task, since JXP was working concurrently: the first check
    showed the expected pre-commit state (`bing/rt/chl_fl.py`,
    `bing/rt/raman.py`, `.claude/skills/inelastic-rrs/SKILL.md` modified,
    `nb/RT/rob_rt_coding_6.ipynb` untracked). By the second check, JXP had
    committed that state (`git log`: commit `68c41bb "wip"`, `git show
    --stat`: exactly those four files plus `claude_prompts/RT/
    rob_rt_prompt_6.md`) — confirmed via `git show HEAD:<path>` that the
    committed content is byte-identical to what this task wrote (the
    "legacy module" backend-note heading in both `raman.py`/`chl_fl.py`,
    the "Recommended inelastic path" heading in `SKILL.md`, and the
    notebook's §3 "The recommendation" text and its 10 code cells'
    original sequential `execution_count`s, all present in the committed
    blob). The only remaining uncommitted change at the time of writing
    this entry is this Logs entry itself, appended to
    `claude_prompts/RT/rob_rt_prompt_6.md` after JXP's commit — expected,
    since JXP commits, not Claude. **Satisfied.**

**Definition of Done — all 11 items confirmed satisfied, live, against the
current repo state on the `rob_rt` branch.** No gaps found beyond the
already-documented, already-accepted caveats (the inelastic accuracy gap,
now affirmatively documented rather than hedged; the design-doc
line-number citation drift, filed as Q4, informational; the
`rob-rt-backend`/`rob_rt` branch-name discrepancy, cosmetic).

**The `rob_rt` branch integration is ready for JXP's final review and
merge.** This closes M5 and the whole `rob_rt` coding-plan series — per
the doc's own "Next" section, there is no `rob_rt_prompt_7.md`.

### 2026-09-01 (PR #27 — Bugbot findings and CI reconciliation)

Prompts item 7: PR `ocean-colour/bing#27` ("Rob rt", `rob_rt` →
`develop`, head `a408b68`, verified == local `git log -1`) carries two
real review comments from `cursor[bot]` (Cursor Bugbot), and all four CI
checks (Tests 3.11/3.12/3.13, Docs build) were failing. Both findings are
fixed with tests; the CI root cause was reproduced locally, fixed in the
workflow, and the fix verified in the same reproduction environment.

**Bugbot finding 1 — "L23 fits omit required geometry"
(`bing/fitting/l23.py`). Confirmed real and fixed.** `prep_one_l23`
seeded `fit_Bp`/MCMC dimensions from `rt_dict`, but `fit_one`,
`fit_with_LM`, and `batch_fit` (this module's own convenience wrappers —
distinct from `bing.fitting.inference.fit_one`/`fit_batch`, which M2
already threads correctly) built 4-tuples with no `ObsGeometry`, so
`validate_rt_dict` raised at setup for every robust-backend fit through
this layer; the synthetic observation Rrs also stayed Gordon-generated
regardless of `rt_dict['rt_backend']`. The fix makes the M4/M5 notebooks'
hand-built pattern the built-in default:

- **`geom_from_p(p)`** (`l23.py:77`, with module constant
  `L23_THETA_S = 0.0` at `:74`): returns `None` for a legacy Gordon `p`
  with no geometry attributes (byte-for-byte legacy behavior), else an
  `ObsGeometry` from `p.theta_s`/`p.theta_v`/`p.dphi` (any extra field on
  `p_ntuple.gen` works, e.g. `standard.expb_pow(theta_s=30., ...)`),
  falling back to the L23 simulation's **own** documented geometry —
  `loisel23.load_ds(4, 0)` is the Hydrolight run at solar zenith 00
  degrees, nadir viewing — for a robust backend with no `p.theta_s`.
  Documented explicitly as *not* a CQ4 "silent theta_s default": the
  module generates the observation itself from that dataset, so the scene
  geometry is known metadata, not a guess.
- **`prep_one_l23`** builds and eagerly validates the geometry
  (`l23.py:383-384`, `validate_rt_dict(rt_dict, models=models,
  geom=geom)` — fail-at-setup, M2's rule), exposes it as
  `ret_dict['geom']` (`:543`), and **dispatches the synthetic observation
  on the backend** (`:409-478`): the Gordon branch is the previous code
  verbatim (including variable-Gordon G0/Gb and BING's own
  Raman/fluorescence composition); a robust backend pushes the same true
  L23 `a`/`bb` spectra on the native L23 grid through the new shared
  raw-IOP adapter (below), with the true `aph` as the fluorescence source
  term and the same stashed downwelling-Ed pair routed to robust's
  `Geometry.Ed`. Measured on idx=170/PACE: the `robust_ztt` observation
  differs from the Gordon one by mean |rel| 1.19e-1, max 1.49e-1 at
  theta_s=0 — really robust-generated, not relabeled Gordon.
- **`fit_one`/`fit_with_LM`** now build 5-tuples
  `(Rrs, varRrs, p0, idx, prep_dict['geom'])` (`l23.py:639`, `:780`);
  **`batch_fit`** threads one shared frozen `ObsGeometry` through every
  item (same `p`, same dataset ⇒ same geometry; pickles by value for
  multiprocessing) and its output loop unpacks `item[:4]` (the 5th slot
  was consumed by the fit).
- **New shared raw-IOP entry point** `bing/evaluate.py`:
  `calc_Rrs_from_iops_robust` (`:754`) for callers holding spectra rather
  than model parameters. Implemented by *splitting*, not duplicating, the
  existing adapter: `_build_robust_inputs_from_iops` (`:396`) now owns
  validation + IOPs/PhaseParams/Geometry construction, the model-parameter
  `_build_robust_inputs` evaluates the models/a_ph/Ed stash then
  delegates, and `_dispatch_robust_forward` (`:709`) is the single
  backend→jitted-closure dispatch both public entry points share — the
  raw-spectrum and model-parameter paths cannot drift (pinned by
  `test_calc_Rrs_from_iops_robust_matches_model_path`, rtol 1e-6).

**Bugbot finding 2 — "LM reconstruction drops geometry"
(`bing/evaluate.py`, `bing/plotting.py`). Confirmed real and fixed.**
`reconstruct_chisq_fits` called `chisq_fit.fit_func` with no geometry and
`show_fits` forwarded `geom` only on the MCMC branch, so a robust LM fit
could be *run* (M2's 5-tuple) but never reconstructed or plotted — both
ended in `calc_Rrs_from_models_robust` with `geom=None`, which raises.
Fix: `reconstruct_chisq_fits` gains `geom=None` (`evaluate.py:1065`,
forwarded to `fit_func` on every evaluation, documented per the M2
convention), and `show_fits`' LM branch passes it through
(`plotting.py:217`) exactly like the MCMC branch always did.

**Adjacent defect found while proving the fix end-to-end — robust LM
fits could not actually converge.** With the geometry threaded, a
noiseless self-consistent `robust_ztt` synthetic refit still returned
ans ≈ p0 (median |rel| 8.9e-2 vs the observation) plus scipy's
"Covariance of the parameters could not be estimated". Root-caused by
direct measurement: robust runs at float32 (CQ1), and curve_fit's default
finite-difference step (~1.5e-8 relative) is below float32 resolution — a
1.5e-8 parameter step changed **zero** Rrs values (Jacobian exactly 0, so
the optimizer "converged" at p0), while a 1e-3 step changed them smoothly.
Fix in `chisq_fit.fit`: for a robust backend only, pass a
float32-appropriate 1e-3 relative step (`diff_step` on the bounded/'trf'
path, the equivalent `epsfcn=1e-6` on the unbounded/'lm' path); the
Gordon (float64) path keeps scipy's default untouched. With the fix the
same synthetic refit recovers the generating parameters to max |rel|
3.7e-7. Consequence: `bing/tests/files/m3_fixed_bp_pin.npz` was
regenerated (`gen_m3_fixed_bp_pin.py`, note added) — the old
`chisq_robust_ztt` entry had frozen the stalled optimizer's output
(pinned value ≈ p0). Verified key-by-key against the old pin:
**only `chisq_robust_ztt` changed**; `chisq_gordon` and *both* MCMC
chains are byte-identical, so the Gordon path and the dispatch behavior
the pin exists to freeze are untouched.

**New tests (13; all green).** `bing/tests/test_evaluate_robust.py` (+7,
cheap synthetic recipe, no L23 data):
`test_reconstruct_chisq_fits_robust_requires_geom` (raises, match
'geom'), `..._robust_geom_end_to_end` (reconstruction == direct
`fit_func` eval exactly, and lands on the observation, asserted median
<1e-3 / max <1e-2 — actual max 3.7e-7-scale),
`..._robust_fit_Bp_tail` (trailing B_p consumed as B_p — different B_p ⇒
different Rrs), `test_show_fits_LM_robust_geom` (renders; returned Rrs ==
`reconstruct_chisq_fits` exactly), `..._without_geom_raises` (regression
pin: loud failure, never silent Gordon),
`test_calc_Rrs_from_iops_robust_matches_model_path`, and
`..._error_paths` (geom/backend/baseline-inelastic/a_ph errors).
`bing/tests/test_l23_fitting.py` (+6, L23-data-gated):
`test_geom_from_p_policy`, `test_prep_one_l23_robust_backend` (geom
exposed; observation robust-generated and == the raw-IOP adapter to
rtol 1e-6), `test_fit_with_LM_robust_backend_end_to_end` (idx=170:
reconstruction median |rel| 5.3e-3, max 4.2e-2 vs the observation —
model-mismatch floor of ExpBricaud+Pow on true L23 IOPs — then
`show_fits` through the LM branch), `test_fit_with_LM_robust_fit_Bp`
(p0/bounds/ans carry the B_p slot; fitted B_p=0.0081 inside
[BP_PRIOR_PMIN, BP_PRIOR_PMAX]; reconstruction median 5.1e-3),
`test_fit_one_robust_backend_mcmc` (nsteps=10000 > the 7000 default
burn; chains (10000, 16, 5) all finite; `reconstruct_from_chains` median
|rel| 1.2e-2, max 2.3e-2), `test_batch_fit_robust_backend` (debug-mode
4-spectrum robust batch through multiprocessing, standard NPZ outputs).

**CI root cause — reproduced, not assumed.** Fresh Python 3.13.13 venv
(`/Users/xavier/miniforge3/bin/python3.13`), installed *exactly* the
current `tests` job list (curated pip set + `ocpy` clone + `pip install
--no-deps -e`), ran `MPLBACKEND=Agg python -m pytest bing/tests -v -ra`
against a clean `git archive HEAD` export. Result: `import bing` itself
passes (the "Show environment" step was green), but collection dies —
`bing/evaluate.py:36: in <module> import jax` →
`E   ModuleNotFoundError: No module named 'jax'` in 6 test modules
(test_bbnw, test_chisq_fit, test_evaluate_robust, test_inference,
test_plotting, test_raman), ending in `Interrupted: 6 errors during
collection` — **zero tests ran**, matching "all three matrix jobs fail".
The docs job dies the same way (autodoc imports `bing.evaluate`). So the
handoff hypothesis was right, with one refinement (`import jax` in
evaluate.py itself fails before `from robust import rt` is reached) and
three discoveries beyond it:

1. **`robust/rt` does not exist on `retrieve-or-bust`'s `main`** (`main`
   holds only `robust/__init__.py`); the rt subpackage lives on feature
   branches, and local validation state == `origin/cdom-rt` (`0002b3e`).
   The workflow therefore clones `--branch cdom-rt` — see **Q6** for the
   branch-pin question this raises.
2. **flax is required at run time, not import time**: with only
   `jax`+`jaxtyping` installed the suite reached `9 failed, 223 passed` —
   8 of the 9 were robust_hybrid/domain-check tests dying in
   `robust/rt/emulator.py:1069` with `ModuleNotFoundError: No module
   named 'flax'` (the emulator loader lazily imports
   `flax.traverse_util`/`flax.linen`). `optax` is only imported by
   robust's training entry points (and rides in with flax anyway).
3. **`m3_fixed_bp_pin.npz` was never committed** — `.gitignore`'s `*.npz`
   catches it (the other two fixtures were evidently force-added); the
   9th failure was its `FileNotFoundError`. Invisible to CI (that test
   needs Hydrolight and skips there) but real for any fresh clone with
   data — **Q7** asks JXP to `git add -f` it.

**Workflow fix** (`.github/workflows/tests.yml`, both jobs, matching the
file's comment conventions): `pip install jax flax jaxtyping` (unpinned
like the rest of the curated list; validated versions jax 0.11.0 / flax
0.12.8 / jaxtyping 0.3.11 noted in the comment as the pin-if-broken
fallback), `git clone --depth 1 --branch cdom-rt
https://github.com/ocean-colour/retrieve-or-bust.git ../retrieve-or-bust`
+ `pip install --no-deps -e` (the exact ocpy pattern; robust's emulator
weights and data .npz files are committed in its repo, so no external
data fetch), an expanded header note explaining *why* (the deliberate
unconditional `import jax`/`from robust import rt` in `bing/evaluate.py`
— which this task did **not** weaken, per settled M0-M5 policy), updated
"Show environment" checks (`import robust.rt`, `import bing.evaluate`),
and refreshed measured skip counts in the header.

**Docs job, second masked blocker.** With imports fixed, the actual
`python -m sphinx -W -b html docs docs/_build/html` (sphinx 9.1.0 from
`docs/requirements.txt`, in the same repro venv) still failed: "build
finished with problems, 9 warnings (with warnings treated as errors)" —
all docutils "Unexpected indentation"/"Block quote" errors from
M0-M5-era docstrings (a bullet list with no blank line after its intro
line) in `evaluate.calc_Rrs_from_models_robust`, `chisq_fit.fit`,
`inference.log_prob`/`init_mcmc`/`fit_one`/`fit_batch` — never seen
before because the docs job always died at import first. Fixed by
inserting the missing blank lines (docstring-only). Rebuild: **"build
succeeded"** with `-W`. This is the *full* docs-job verification (real
sphinx build, not just the import-succeeds minimum).

**Verification, before/after, same environment.**

- Pre-fix (CI-equivalent, Python 3.13): `6 errors during collection`,
  0 tests run (the quoted ModuleNotFoundError above).
- Post-fix, CI-equivalent (no `$OS_COLOR` — it leaks from the local
  shell, so it was explicitly unset to match CI): **84 passed,
  151 skipped, 0 failed** (9.4s) — collection clean, Hydrolight-gated
  skips as designed.
- Post-fix, same venv *with* the data (`$OS_COLOR` set, pin fixture
  present): **232 passed, 3 skipped, 0 failed** (15.1s) — the fewer
  totals vs ocean14 are the four `correct_atmosphere`-gated modules
  conftest drops from collection.
- Docs: sphinx `-W` build succeeded (above).
- **ocean14 full suite on the final tree: 281 passed, 2 skipped,
  0 failed** (185.9s) — up from M5's 268/2 by exactly the 13 new tests;
  no regressions from any Piece-A change (including the regenerated pin
  and the docstring RST fixes).

**Observed, deliberately not touched.** (a) `l23.process_one` is
pre-existing bit-rot unrelated to these findings: it references undefined
`anly_utils_20` and `bbw_440` and calls `reconstruct_from_chains` without
the required `rt_dict` — it would NameError before any geometry question
arises; its test has always been skipped. Left for a separate decision.
(b) Mid-task, `git status` began showing `MM` states — someone (JXP,
presumably, as during the M5 closing) staged an intermediate snapshot of
this work into the index. Nothing was done about it (Claude runs no
state-changing git); the **working tree** is the authoritative, complete
state — please re-`git add` on commit, including the `-f` for the pin
fixture per Q7.

**Open items for JXP**: Q6 (which `retrieve-or-bust` branch CI should
track — currently pinned to `cdom-rt`, the validated state, but it is a
moving feature branch) and Q7 (`git add -f
bing/tests/files/m3_fixed_bp_pin.npz`).

### 2026-09-01 (Q6/Q7 confirmed, Python 3.11 CI fix)

Prompts item 8: JXP answered Q6 and Q7; both turned out to be pure
confirmations needing no code change, and the real work this round was a
new, Python-3.11-specific CI failure that surfaced once the previous CI
fix landed.

**Q6 — no action.** JXP confirmed the `cdom-rt` branch pin in
`.github/workflows/tests.yml` is fine for now ("I'll redirect to `main`
eventually"). The pin is untouched; the workflow comment already says to
re-point it when `robust/rt` merges to main.

**Q7 — no action.** JXP added `bing/tests/files/m3_fixed_bp_pin.npz` to
the repo; verified tracked (`git ls-files` lists it) and present in a
clean `git archive HEAD` export, so fresh clones with the Hydrolight
data now get the fixture.

**The new problem — Tests (Python 3.11) failing at "Install
dependencies".** Against the current head `eb4792c`, the PR's checks
read: Tests 3.12 success, Tests 3.13 success, Docs build success, Tests
**3.11 failure** — i.e. the previous session's CI fix worked everywhere
except 3.11, which now fails in the install step itself (exit 1 before
any test runs; the log text needs sign-in, so the cause was reproduced
locally rather than read off the runner).

**Reproduction — exact, in a fresh Python 3.11.16 conda env
(`ci-repro-py311`), against a clean `git archive HEAD` export, running
the tests job's install commands verbatim and in order.** The curated
scientific stack installs fine. `pip install jax flax jaxtyping` also
succeeds — but resolves **jax 0.10.2 / flax 0.12.8** (not the 0.11.x
line 3.12/3.13 get), because jax >=0.11 and flax 0.12.9 declare
`Requires-Python >=3.12` and pip backtracks to the newest versions that
allow 3.11. The ocpy clone + editable install succeeds. The failure is
the next line:

    pip install --no-deps -e ../retrieve-or-bust
    ERROR: Package 'retrieve-or-bust' requires a different Python:
    3.11.16 not in '>=3.12'

**Root cause**: `retrieve-or-bust/setup.py:46` declares
`python_requires = '>=3.12'` (present since that repo's initial commit
`06f6a38`). pip enforces Requires-Python even under `--no-deps`, and a
direct editable install of a single source tree has no older version to
backtrack to — so 3.11 hard-fails while 3.12/3.13 sail through. The
floor mismatch is real at the metadata level: bing states >=3.11
(`setup.py:22`) and its CI matrix tests 3.11, but bing now imports
robust unconditionally, and robust says >=3.12.

**Measured, the floor is metadata-only for what bing exercises**:
robust's source contains no 3.12-only syntax (no PEP 695 generics/`type`
statements, no `except*`), all four of the workflow's "Show environment"
import checks pass on 3.11 (including `import robust.rt` and
`import bing.evaluate`, on jax 0.10.2), and the suite behaves
identically to 3.12/3.13 (numbers below).

**Fix** (`.github/workflows/tests.yml`, tests job only — the docs job
runs 3.12 and satisfies the floor):

    pip install --no-deps --ignore-requires-python -e ../retrieve-or-bust

with an inline comment quoting the error, recording the measurements,
and saying when to drop the flag; the header's jax/flax note now also
records the 3.11 Python-floor situation (jax 0.10.2 / flax 0.12.8 on
3.11, measured to pass). The flag is a documented bridge, not the
proper resolution — **Q8** (new this round) puts the real decision to
JXP: lower retrieve-or-bust's floor to >=3.11, or accept that bing now
effectively requires >=3.12 and drop 3.11 from the matrix; either way
the flag comes out then.

**Verification, before/after, same 3.11 environment.**

- Pre-fix: the quoted `ERROR: ... 3.11.16 not in '>=3.12'` — install
  step dead, zero tests run (matches the CI symptom exactly).
- Post-fix, CI-equivalent (no `$OS_COLOR`, `MPLBACKEND=Agg`):
  **84 passed, 151 skipped, 0 failed** (10.4s) — identical to the
  3.12/3.13 pattern and to the previous entry's 3.13 reproduction.
- Post-fix, same env *with* the Hydrolight data: **231 passed, 1 failed,
  3 skipped** (14.1s). The one failure is
  `test_fit_Bp_false_matches_provisional_pin`, the rtol=0/atol=0 exact
  pin — max abs diff 1.47e-12 (4e-10 relative) on the Gordon-generated
  observation itself, i.e. last-ulps float drift between this env's
  numpy build (cp311 wheel) and the one the fixture was generated under.
  This is precisely the case the test's own docstring anticipates ("If
  this test ever fails on a new environment (BLAS/emcee version) while
  test_fit_Bp_false_or_absent_byte_identical_to_m2 stays green,
  regenerate the fixture rather than suspecting a regression") — the
  byte-identity companion test **did** stay green, the same pin test
  **passes** in ocean14 (so the committed fixture is correct as-is; not
  regenerated, to avoid churning a file JXP just committed), and the
  test skips on CI regardless (needs Hydrolight). Not a 3.11 defect.
- **ocean14 full suite on the final tree: 281 passed, 2 skipped,
  0 failed** (176.1s) — byte-for-byte the previous entry's totals, as
  expected for a change that touches only the workflow file and this
  doc.

**Bugbot-findings sanity check (not new work — confirming the prior fix
holds).** All 13 tests the previous session added for the two findings
pass in the ocean14 run: `test_evaluate_robust.py`'s 7
(`test_reconstruct_chisq_fits_robust_requires_geom`,
`..._robust_geom_end_to_end`, `..._robust_fit_Bp_tail`,
`test_show_fits_LM_robust_geom`, `..._without_geom_raises`,
`test_calc_Rrs_from_iops_robust_matches_model_path`,
`..._error_paths`) and `test_l23_fitting.py`'s 6
(`test_geom_from_p_policy`, `test_prep_one_l23_robust_backend`,
`test_fit_with_LM_robust_backend_end_to_end`,
`test_fit_with_LM_robust_fit_Bp`, `test_fit_one_robust_backend_mcmc`,
`test_batch_fit_robust_backend`). The stale Bugbot comments still shown
on the PR page are expected — Bugbot has not been re-triggered since the
fix commits landed; the underlying code was not touched again.

**Files changed this round**: `.github/workflows/tests.yml` (the
`--ignore-requires-python` flag plus its comment, and the header note)
and this doc (Q8, this entry). No source or test code changed.

**Open item for JXP**: Q8 (reconcile the bing >=3.11 / retrieve-or-bust
>=3.12 Python floors; then drop the workflow's
`--ignore-requires-python`).

### 2026-09-01 (CI matrix: drop 3.11, add 3.14)

Prompts item 9 — JXP's resolution of Q8, option (b): "Please modify the
CI tests to only include Python 3.12 and greater. And add a new 3.14
test." Q8 is now closed; no new Q&A entry was needed, because Python
3.14 was verified to genuinely work (measurements below) rather than
needing a decision.

**Workflow changes** (`.github/workflows/tests.yml`, tests job):

- Matrix: `['3.11', '3.12', '3.13']` → `['3.12', '3.13', '3.14']`, with
  the matrix comment updated to point at the reconciled >=3.12 floor
  instead of the old ">= 3.11 / `X | Y` annotations" note.
- `--ignore-requires-python` removed from the
  `pip install --no-deps -e ../retrieve-or-bust` line, exactly as Q8's
  closing note prescribed — with the matrix floor now at 3.12, every
  job satisfies retrieve-or-bust's `python_requires '>=3.12'` and the
  override is dead weight. Its long explanatory comment block is
  replaced by a short note recording that the floors now match and that
  the flag briefly existed (pointer to Q8 for the history), so the
  trail survives without stale instructions. The docs job never had the
  flag (its single Python is pinned to 3.12, which always satisfied the
  floor) and is untouched.
- Header block: the jax-0.10.2-on-3.11 backtracking note (jax >=0.11 /
  flax >=0.12.9 require >=3.12, so 3.11 resolved older versions) is
  gone with the 3.11 job; in its place the header records that bing's
  own floor is now >=3.12 to match retrieve-or-bust's, per Q8.

**Version-floor statements updated elsewhere** (from a repo-wide grep
for "3.11", touching only genuine requirement statements):

- `setup.py:22`: `'Python (>=3.11.0)'` → `'Python (>=3.12.0)'`, with a
  comment tying the floor to retrieve-or-bust's `python_requires` and
  pointing at Q8. No classifiers or other Python-floor references exist
  in setup.py.
- `readthedocs.yaml`: build Python `"3.11"` → `"3.12"` (RTD builds a
  package whose stated floor is now 3.12).
- `docs/contributing.rst` and `docs/changelog.rst`: "Python 3.11-3.13"
  → "Python 3.12-3.14" in the CI descriptions.

**Python 3.14 verification — fresh, CI-faithful, not inferred from
ocean14.** ocean14 itself runs 3.14.6 with a working jax 0.11.0 / flax
0.12.8 stack, which is encouraging but not probative (its packages were
not resolved by a fresh pip on a bare env). So: new conda env
`ci-repro-py314` (Python 3.14.6), clean `git archive HEAD` export
overlaid with this session's edited files (i.e. the tree as it will be
committed, tests.yml flag already removed), install steps run verbatim:

- The curated stack resolves entirely from cp314 wheels: numpy 2.5.2,
  scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.1, xarray 2026.7.0,
  h5py 3.16.0, emcee 3.1.6, scikit-learn 1.9.0, pytest 9.1.1;
  `pip install jax flax jaxtyping` resolves jax 0.11.1 / flax 0.12.9 /
  jaxtyping 0.3.11. No source builds, no failures.
- `pip install --no-deps -e ../retrieve-or-bust` (fresh `cdom-rt`
  clone) succeeds **without** `--ignore-requires-python`, as it must
  now that 3.14 satisfies '>=3.12'. ocpy and bing editable installs
  succeed as before. (The ocpy clone was a pre-existing scratch copy;
  verified equal to origin/main HEAD `3aed28a` before trusting it.)
- All four "Show environment" import checks pass (numpy/scipy/xarray/
  emcee/jax versions print; `ocpy`, `robust.rt`, `bing.evaluate` all
  import).
- CI-equivalent suite (no `$OS_COLOR`, `MPLBACKEND=Agg`): **84 passed,
  151 skipped, 0 failed** (9.6 s) — identical to the 3.12/3.13 pattern
  and to Q8's prior measurements.
- GitHub-runner side (separate concern from local reproducibility):
  actions/python-versions' manifest lists stable **3.14.7** with
  linux-24.04-x64 binaries, so `actions/setup-python@v5` with `'3.14'`
  resolves on ubuntu-latest.

**3.12/3.13 sanity check after removing the flag.** The only changed
install line is the retrieve-or-bust one, so the check targets it: in
fresh `ci-repro-py312` (3.12.14) and `ci-repro-py313` (3.13.15) envs,
`pip install --no-deps -e ../retrieve-or-bust` without the flag
succeeds on both, as does the bing editable install. (Full-suite
behavior on 3.12/3.13 is unchanged by construction — the flag only
relaxes pip's Requires-Python check, which those versions always
satisfied — and was measured at 84/151 in the two prior entries.)

**ocean14 full suite on the final tree: 281 passed, 2 skipped,
0 failed** (180.1 s) — byte-for-byte the last-recorded baseline, as
expected for a change touching only CI config, packaging metadata, and
docs text.

**One environment side effect to know about**: an early verification
command was meant for `ci-repro-py314` but landed in ocean14 (the shell
profile keeps ocean14 first on PATH, so `conda activate` alone does not
redirect `python`; all subsequent work used explicit
`envs/ci-repro-*/bin/python` paths). Net effect on ocean14: pip
upgraded to 26.2.1 and wheel to 0.48.0; setuptools was briefly at
84.0.0, then deliberately set to 81.0.0 to satisfy torch 2.12.1's
`setuptools<82` constraint. `pip check` reports no broken requirements,
torch imports, and the full-suite baseline above was run after this —
ocean14 is consistent, just with a newer pip/wheel than before.

**Files changed this round**: `.github/workflows/tests.yml`,
`setup.py`, `readthedocs.yaml`, `docs/contributing.rst`,
`docs/changelog.rst`, and this doc (this entry). No source or test
code changed. Q8 is resolved; no open items from this round.

### 2026-09-01 (Docs pass: RTD build fix + RoB content update)

Prompts item 10: the branch is merged into `develop` (merge commit
`c668f78`) and all four GitHub Actions checks there are green, so the
"docs are failing" JXP sees is **ReadTheDocs** — the separate build
service configured by `readthedocs.yaml`, which (unlike the GH docs
job's curated `--no-deps` install) does a plain `pip install .`. Two
real problems were found and fixed by local reproduction (RTD's own
logs need dashboard access), and the Sphinx content itself was brought
up to date for the RoB integration.

**Root cause 1 — reproduced, fatal: `pip install .` cannot resolve
`retrieve-or-bust`.** In a fresh Python 3.12.14 venv (RTD builds on
3.12), against a clean `git archive HEAD` export, `pip install .`
dies in dependency resolution:

    ERROR: Could not find a version that satisfies the requirement
    retrieve-or-bust (from bing) (from versions: none)
    ERROR: No matching distribution found for retrieve-or-bust

`setup.py`'s `install_requires` carried the **bare name**
`'retrieve-or-bust'` (added in M0), which tells pip to fetch it from
PyPI — where it does not exist. This is exactly the step RTD's
`python.install` runs, and it also breaks any real downstream
non-editable install of bing.

**Fix: PEP 508 direct reference** in `install_requires`:
`'retrieve-or-bust @ git+https://github.com/ocean-colour/
retrieve-or-bust.git@cdom-rt'` (one line, plus a comment recording the
reproduced error and when to re-point the branch pin). Chosen over the
alternative (drop the requirement + add an RTD pre-install clone)
because (a) it fixes plain `pip install .` for RTD *and* every
downstream user in one place; (b) unlike ocpy, a **non-editable**
robust install is fully functional — retrieve-or-bust's own setup.py
declares `package_data` for the emulator weights precisely so an
installed copy works (verified below: `robust/rt/files/*.npz` and
`rt/data/*.npz` all present in site-packages); (c) the `@cdom-rt` pin
mirrors the tests.yml pin JXP blessed in Q6, same re-point-on-merge
note. The GH workflow is untouched and unaffected — both its jobs
install bing with `pip install -e . --no-deps`, which never reads
`install_requires`.

**Verification of fix 1** (same fresh venv, clean export + the fixed
setup.py): `pip install .` **succeeds** — pip clones the repo and
resolves it to commit `363951d` == `origin/cdom-rt` == the local
validated checkout; `import bing`, `import bing.evaluate`, and
`import robust.rt` all work; the installed (non-editable) robust
carries `emulator_l23.npz`/`fl_corr_l23.npz`/`raman_corr_l23.npz` and
`ed_l23.npz`, so the hybrid backend is functional with no external
fetch.

**Root cause 2 — previously masked: ocpy missing from the RTD
environment.** With the install fixed, the RTD-equivalent Sphinx
build (`python -m sphinx -W -b html docs docs/_build/html`, sphinx
9.1.0 from docs/requirements.txt) still showed **8 warnings**, all
`autodoc: failed to import ... ModuleNotFoundError: No module named
'ocpy'` (fitting.inference, io.save_fit/load_fit, models.anw/bbnw/
utils, noise, plotting). RTD does not set `fail_on_warning`, so these
were never fatal — they were silently rendering 8 API pages empty on
the live site. ocpy cannot ride in `install_requires`: it is not on
PyPI either, and a non-editable install drops `ocpy.hydrolight` (no
`__init__.py`), which bing needs. **Fix**: `readthedocs.yaml` gains
`build.jobs.pre_install` (config v2, verified against RTD's
build-customization docs; runs in the same env as `python.install`)
doing the exact tests.yml ocpy pattern — `git clone --depth 1` +
`pip install --no-deps -e ../ocpy` — with a comment block explaining
why, and the `python.install` section gains a comment tying the
plain-`pip install .` step to the setup.py direct reference. With
ocpy editable-installed in the same venv, the full RTD-equivalent
build is **"build succeeded" under `-W`** — stricter than RTD's
actual (non-`-W`) behavior, so RTD gets both a passing build and its
8 missing API pages back.

**Docs content pass (Piece 2)** — the rendered Sphinx docs previously
contained *nothing* about the RT-backend integration (no `rt_backend`,
no `ObsGeometry`, no robust backends anywhere under `docs/*.rst`);
all of it lived in skills/planning docs only. Added, matching each
file's existing voice and citing the measured numbers (sources:
`rt_dict_from_p`'s M5 docstring, `inelastic-rrs`/`run-bing-fit`
SKILL.md, `nb/RT/rob_rt_coding_6.ipynb` §3, and
`dev/rob_rt/benchmark_backends.py`'s recorded run):

- `docs/radiative_transfer.rst`: new top-level section **"RT
  Backends: Gordon and robust.rt"** (anchor `_rt-backends`) — the
  four-value table; a "Choosing a backend" guide mirroring the
  capstone notebook's adopted-robust framing (baseline ≈ gordon to
  ~2.3e-7 elastic parity; ztt/hybrid deliberately different, 5.6%/
  7.0% max Rrs difference on L23 idx=170); a warning block with the
  full inelastic-gap numbers (Raman max 11.4%/mean 5.6%;
  fluorescence 9.2-18.5% max — documented, not yet closed); the
  ObsGeometry requirement (`theta_s` mandatory, fail-at-setup);
  free/fixed B_p ([0.004, 0.05] linear prior); the
  validate_rt_dict error list (baseline+inelastic, hybrid 350-750 nm
  domain); and the benchmark throughput table (gordon 38936 calls/s;
  ztt 5991/0.154x; hybrid 5667/0.146x; baseline 7274/0.187x; JIT
  first-call ~0.18 s, per-`fit_batch`-worker note). Also updated the
  stale `rt_dict_from_p` section (example dict and py:function now
  show `rt_backend`/`fit_Bp`/`Bp_value` with their real defaults),
  the page intro, and See Also (`bing.rt.geometry`).
- `docs/fitting.rst`: the observation tuple's optional 5th `geom`
  element documented at the `items` definition, plus a new section
  **"Fitting with a robust RT Backend"** (anchor
  `_fitting-robust-backend`): the 5-tuple through `chisq_fit.fit`/
  `fit_one`/`fit_batch`, the fail-at-setup ValueError naming
  `theta_s`, the extra B_p parameter slot when `fit_Bp=True`, and the
  l23 wrappers' `geom_from_p` policy (attributes on `p`, else the
  L23 dataset's own documented zenith-0/nadir geometry; legacy Gordon
  `p` = byte-identical legacy behavior).
- `docs/api/evaluation_api.rst`: the page uses `automodule
  :members:`, so `calc_Rrs_from_models_robust`,
  `calc_Rrs_from_iops_robust`, `robust_domain_check` and the
  backend-aware `reconstruct_from_chains` were verified already
  present in the built HTML; the intro prose now names them so the
  page reflects its content.
- `docs/changelog.rst`: new "robust.rt backend integration
  (2026-08/09, PR #27)" entry above the PR #26 one, comparable
  detail: the four backends, geometry threading, B_p, the measured
  accuracy story (parity + inelastic caveat), the new evaluate entry
  points, explicit **non-breaking** statement (gordon default,
  byte-identical, regression-pinned), and the packaging change
  (direct reference + the >=3.12 floor).
- `docs/installation.rst`: Python floor 3.8→3.12 (this page had been
  missed by the Q8 sweep), the from-source note that `pip install .`
  now pulls retrieve-or-bust from GitHub (needs git + network, JAX
  rides in, unconditional import — not optional), corrected ocpy
  instructions (editable-from-source, the `pip install ocpy` line was
  wrong — it isn't on PyPI), a new "retrieve-or-bust (robust.rt)
  Installation" subsection (automatic in the normal case, editable
  side-by-side for development), and two troubleshooting entries.
- `docs/index.rst`: python badge 3.8+ → 3.12+.

**Verification (Piece 3).**

- Sphinx: `python -m sphinx -W -b html` (and a fresh-environment `-E`
  rebuild) on the final tree: **"build succeeded"**, zero warnings —
  matching the GH docs job's strictness; RTD's own build is laxer.
- `readthedocs.yaml` re-parsed as valid YAML with the expected
  `build.jobs.pre_install` / `python.install` structure.
- GH workflow cross-check: both tests.yml jobs use
  `pip install -e . --no-deps` + their own clones, so the setup.py
  change cannot affect them (and docs content changes only feed the
  already-green sphinx step).
- **ocean14 full suite on the final tree: 281 passed, 2 skipped,
  0 failed** (181.0 s) — byte-for-byte the last-recorded baseline, as
  expected for a change touching only packaging metadata, RTD config,
  and docs text.

**Not verifiable from here**: the RTD dashboard itself (needs
authenticated access). The failure mode was reproduced and fixed
locally at the exact step RTD runs; if the next RTD build still
fails, its dashboard log is the thing to read. No new Q&A entry —
no open decisions this round.

**Files changed this round**: `setup.py`, `readthedocs.yaml`,
`docs/radiative_transfer.rst`, `docs/fitting.rst`,
`docs/api/evaluation_api.rst`, `docs/changelog.rst`,
`docs/installation.rst`, `docs/index.rst`, and this doc (this
entry). No source or test code changed.

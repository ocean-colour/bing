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

_(none yet — this prompt has not been executed)_

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

_(none yet)_

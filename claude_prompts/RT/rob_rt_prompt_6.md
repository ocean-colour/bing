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

Per the working agreements in `rob_rt_prompt_1.md` (git by JXP on
`rob-rt-backend`; `ocean14`; CQ1–CQ4 binding; scope discipline;
pytest-gated; Fable; log). Milestone-specific emphasis:

- **No code is deleted from `bing/rt/raman.py` / `chl_fl.py`** — docstring
  pointers only (design §6). They still serve the `gordon` backend.
- **Benchmark numbers are evidence, not a gate**: record them; do not
  optimize the hybrid path on your own judgment.

## Context

Read before coding:

- **Previous prompt** — `rob_rt_prompt_5.md` (M4: final integration state,
  measured inelastic agreement) and its Logs — and skim the Logs of prompts
  1–4 for anything flagged "for M5".
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

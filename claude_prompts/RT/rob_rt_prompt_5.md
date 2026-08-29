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

Per the working agreements in `rob_rt_prompt_1.md` (git by JXP on
`rob-rt-backend`; `ocean14`; CQ1–CQ4 binding; scope discipline;
pytest-gated; Fable; log). Milestone-specific emphasis:

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

- **Previous prompt** — `rob_rt_prompt_4.md` (M3: chain bookkeeping, how
  well `B_p` was constrained) and its Logs.
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

_(none yet — this prompt has not been executed)_

## Next

→ `rob_rt_prompt_6.md` (M5: deprecation notes, docs, throughput benchmark).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

_(none yet)_

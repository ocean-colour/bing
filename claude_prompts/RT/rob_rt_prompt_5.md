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

_(none yet — this prompt has not been executed)_

## Next

→ `rob_rt_prompt_6.md` (M5: deprecation notes, docs, throughput benchmark).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

_(none yet)_

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

Per the working agreements in `rob_rt_prompt_1.md` (git by JXP on
`rob-rt-backend`; `ocean14`; CQ1–CQ4 binding; scope discipline;
pytest-gated; Fable; log). Milestone-specific emphasis:

- **CQ4 lands here as behavior**: a robust-backend fit with a legacy
  4-tuple (no geom) must raise at setup with a message naming `theta_s`.
- **Plan choice held (design §7.2)**: the observation package stays a tuple
  — the optional 5th element, no dataclass promotion. Don't re-open it.

## Context

Read before coding:

- **Previous prompt** — `rob_rt_prompt_2.md` (M1: the adapter's final
  signature, the JIT cache, `robust_domain_check`) and its Logs.
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

_(none yet — this prompt has not been executed)_

## Next

→ `rob_rt_prompt_4.md` (M3: `B_p` as an optional free MCMC parameter).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

_(none yet)_

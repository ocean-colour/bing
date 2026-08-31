# RoB RT Backend Coding — Prompt 4 (M3: `B_p` as an optional free MCMC parameter)

## Goals

Implement **Milestone M3** of the coding plan
(`docs/coding_plan/rob_rt_coding_plan.md`): when `rt_dict["fit_Bp"]` is True,
the particulate backscattering ratio `B_p` becomes the last element of the
MCMC parameter vector, with its own prior and full chain bookkeeping (ndim,
walkers, `calc_stats` names, `reconstruct_from_chains`, corner plots). The
fixed-`B_p` default path must remain byte-identical to M2.

## Claude

### Skills

`.claude/skills/`: `diagnose-mcmc` (convergence of the round-trip gate),
`debug-priors` (the new `B_p` prior — a linear-space uniform among log10
amplitudes is exactly the log10-vs-linear trap that skill covers),
`plot-bing-fit` (corner plots with the extra column), `run-bing-fit`,
`code-review`.

### Working agreements

Per the working agreements in `rob_rt_prompt_1.md` (git by JXP; `ocean14`;
CQ1–CQ4 binding; scope discipline; pytest-gated; Fable; log) — **one
correction, carried over from `rob_rt_prompt_2.md`/`rob_rt_prompt_3.md`**:
the branch is JXP's existing **`rob_rt`**, not `rob-rt-backend` as the
coding plan suggested; all of M0–M2 landed there, with JXP reviewing and
committing after each task — assume the same cadence. Milestone-specific
emphasis:

- **The prior framing is a plan choice, already made (design §7.3 →
  coding-plan M3)**: `B_p` is a **linear-space** parameter (a ratio, like
  slopes — not a log10 amplitude), `uniform` flavor over **[0.004, 0.05]**
  by default, inside `PhaseParams.validate`'s definitional (0, 1] bound.
  Don't re-open it; if the round-trip gate argues for a different range,
  raise it in Q&A rather than changing it silently.
- **The `aparams`/`bparams` split is load-bearing** (post-M2 lines:
  inference.py:113-114, chisq_fit.py:230-231): peel the `B_p` tail
  *first*, leave the split untouched.

## Context

Read before coding:

- **Previous prompt** — `rob_rt_prompt_3.md` (M2, **complete**: all 5
  tasks done; full-suite baseline entering M3 is **239 passed, 2 skipped,
  2 failed** — the same 2 pre-existing `test_l23_inelastic.py`
  missing-fixture failures diagnosed in M0/Q10, not a regression). Its
  Q&A/Logs are the record; the load-bearing facts for this milestone:
  - **Fitter signatures as they now stand** (re-verified against source
    2026-08-31 — line numbers drift with every edit; trust these over any
    older citation):
    `log_prob(params, models, Rrs, varRrs, rt_dict, geom=None)`
    (inference.py:53-54), dispatching on
    `rt_dict.get('rt_backend', 'gordon')` at inference.py:127-134;
    `fit_func(wave, *params, models=None, return_full=False,
    rt_dict=None, geom=None)` (chisq_fit.py:175-176), same dispatch at
    chisq_fit.py:236-243 — task 1's tail peel goes in both, *before* the
    `aparams`/`bparams` split (inference.py:113-114, chisq_fit.py:230-231).
    `fit_one` (inference.py:196-197) and `chisq_fit.fit` (chisq_fit.py:47)
    accept the 4- or 5-tuple `(Rrs, varRrs, params, idx[, geom])` and run
    `validate_rt_dict` at setup plus `robust_domain_check` (non-gordon
    only; `fit_one` on p0 and the posterior median, `fit` on p0 only —
    there is no posterior in a χ² fit). `run_emcee` gained trailing
    `geom=None` (inference.py:436-442) and threads it into emcee's
    **positionally-ordered** `args` list (inference.py:550-553 — the list
    must mirror `log_prob`'s signature order; M2 Q2): a `B_p` tail is a
    *sampled dimension*, not another `args` element. `fit_batch`
    (inference.py:574) is code-unchanged; its docstring documents the
    optional 5th tuple element.
  - **`Bp=None` semantics (M2 Q1)** — task 1's exact target: both
    dispatch sites currently pass `Bp=None` explicitly, which the adapter
    documents as "fall back to `rt_dict['Bp_value']`"
    (`_build_robust_inputs`, evaluate.py:362) — the fixed-`B_p` case.
    When `fit_Bp` is True, the peeled tail value replaces that `Bp=None`;
    the in-code comments at both branches ("a free/sampled Bp arrives in
    M3") mark the spots.
  - **Domain-check gating (M2 Q3)**: `robust_domain_check` is gated
    `!= 'gordon'` in both fitters because calling it on a Gordon config
    **raises** `ValueError` ("not a robust backend") — it is not a no-op;
    `test_gordon_fit_never_calls_domain_check` pins the gate with a bomb
    monkeypatch. The gate stands once `B_p` is free — the check accepts
    `Bp=`, so task 1's peeled value should flow into the fitters'
    existing check calls too (they currently pass `Bp=None`).
  - **Smoke-fit numbers** (measured 2026-08-31, `ocean14`, same recipe as
    `nb/RT/rob_rt_coding_3.ipynb`: L23 idx=170, 61-band PACE grid,
    seeded `fit_one`, nsteps=200/nburn=50/**nwalkers=16**): gordon
    0.12 s; warm robust fits (cached jitted closure) 0.5–0.6 s —
    ztt/hybrid/baseline 0.59/0.61/0.51 s, i.e. ~420–520 emcee it/s vs
    Gordon's ~2200, ~5× per step at this size. One-time first-call
    compile cost on top: ztt ~0.35 s, hybrid ~1.1 s (Flax emulator load
    dominating; 1.72 s cold total, identical whether run first in a
    fresh process or after other backends), baseline ~0.03 s
    (negligible — `Rrs_gordon` is a trivial trace). No surprises beyond
    that: repeated identical-config fits never recompile (M1's
    `lru_cache` paying off in a real fit context), no memory growth
    observed. This is M3's baseline once `fit_Bp` adds a dimension
    (ndim 5→6; `nwalkers = max(16, 2×ndim)` stays 16).
  - **Gate-coverage gap (M2 Q5 — OPEN, and a direct dependency of this
    milestone's Gate item 6)**: after M2, Gate items 1/4/5 have no
    automated pytest coverage and item 2 is only partial — in particular
    **no fitter-level regression pin (seeded chains or chisq params)
    against a pre-change value exists anywhere**; the only pins are M1's
    *forward-model* fixture (`files/l23_gordon_fixture.npz`) and task 2's
    tuple-arg-identity tests. Consequence: M3 Gate item 6
    ("`fit_Bp=False` results identical to an M2-pinned value") **has no
    pinned value to compare against yet** — task 3 will hit this
    directly. JXP's call how to resolve: answer Q5 (a follow-up test
    task closing M2's gap first), or have M3 create its own fixed-`B_p`
    fitter-level pin at the point `fit_Bp=False` is first tested
    (treating "identical to M2" as "identical to the current, un-pinned
    dispatch behavior, verified live" until Q5 is resolved). Don't pick
    silently — raise it in Q&A before task 3 if still unresolved.
- **Coding plan** — `docs/coding_plan/rob_rt_coding_plan.md` **M3** section.
- **Design** — `docs/design/rob_rt_design.md` §3.3 (`B_p` free/fixed, the
  tail-append layout, the gordon+`fit_Bp` configuration error).
- **Q&A record** — `claude_prompts/rob_rt.md`: Design Q7 (free-or-fixed
  `B_p`) and the round-2 finding that `PhaseParams.B_p` broadcasts against a
  scalar, so a scalar MCMC parameter is well-supported.
- **Current code** (post-M2 line numbers, re-verified 2026-08-31) —
  `bing/fitting/inference.py` (the split, inference.py:113-114;
  `init_mcmc` ndim/nwalkers, inference.py:187-188),
  `bing/fitting/chisq_fit.py` (the split, chisq_fit.py:230-231),
  `bing/evaluate.py` (`reconstruct_from_chains`, evaluate.py:683;
  the adapter `calc_Rrs_from_models_robust`, evaluate.py:438), and
  `robust/rt/types.py` (`PhaseParams`, types.py:250; its `validate`,
  types.py:277).
- **BING conventions** — `bing/CLAUDE.md` "Which parameters are log10":
  `log_params` semantics for the new linear `'B_p'` name (False), and the
  fitters keying p0 conversion on prior flavor — keep the two consistent.

## Prompts

1. Read this doc. Execute the 1st task in the "M3" section below — the tail
   peel and forward flow. If you have any questions, ask me in the Q&A
   section below. Use Fable if you can. Log your work.
2. Read this doc. Execute the 2nd task — the prior and p0 seeding. Check my
   answers in Q&A; if you have additional questions, ask in Q&A. Use Fable
   if you can. Log your work.
3. Read this doc. Execute the 3rd task — the bookkeeping. Use Fable if you
   can. Log your work.
4. Read this doc. Execute the 4th task — the explainer notebook. Use Fable
   if you can. Log your work.
5. Read this doc. Execute the 5th task — update `rob_rt_prompt_5.md` with
   what M3 established. Use Fable if you can. Log your work.

## M3

### Tasks

1. **Tail peel + forward flow.** When `rt_dict["fit_Bp"]` is True, `B_p` is
   appended as the **last** element of the combined vector —
   `[a_params..., bb_params..., B_p]` (design §3.3). `log_prob` and
   `chisq_fit.fit_func` **peel the tail first** (`Bp = params[-1]; params =
   params[:-1]`) so the existing `aparams`/`bparams` split
   (inference.py:93-94, chisq_fit.py:170-171) is untouched; `Bp` flows to
   `calc_Rrs_from_models_robust(..., Bp=Bp)`.

2. **Prior + seeding** *(plan choice, design §7.3)*: linear-space `uniform`
   over **[0.004, 0.05]** by default, evaluated in `log_prob` alongside the
   model priors (an out-of-range tail → `-np.inf` before any forward call).
   p0 seeds at `rt_dict["Bp_value"]` (0.01) — nonzero, so `init_walkers`'s
   floor gives it spread.

3. **Bookkeeping.** `init_mcmc` gains optional `rt_dict` and adds 1 to ndim
   (inference.py:159-160) when `fit_Bp`; `prior_bounds`/`init_walkers`
   clipping extended by the `B_p` bounds; `reconstruct_from_chains`
   (evaluate.py:245) strips the tail column before model evaluation and
   forwards it as `Bp`; parameter name `'B_p'` appended for `calc_stats`
   names and corner plots (linear label — `log_params` semantics: False).
   `validate_rt_dict` already rejects `fit_Bp` + `gordon` (M0); pin it with
   a test here, where it matters.

4. **Notebook.** `nb/RT/rob_rt_coding_4.ipynb` (executed, with outputs):
   the synthetic round-trip — generate Rrs with `robust_ztt` at a known
   `B_p=0.02`, fit with `fit_Bp=True`, show the posterior against truth and
   the prior edges (corner plot with the `B_p` column via `plot-bing-fit`
   conventions), plus a short section on why `B_p` is linear-space among
   log10 amplitudes.

5. **Finally.** Update `rob_rt_prompt_5.md` (M4) with what M3 established —
   chain-shape/bookkeeping details M4's tests will reuse, and how well the
   round-trip constrained `B_p` (it informs whether M4's inelastic fits
   should fix or free it).

### Gate

`bing/tests/test_evaluate_robust.py` additions (deterministic seeds):

1. **Synthetic round-trip**: generate Rrs with `robust_ztt` at a known
   `B_p=0.02`, fit with `fit_Bp=True` — posterior median of `B_p` within
   its 5–95% credible interval of truth, and the interval excludes the
   0.004/0.05 prior edges.
2. ndim/walker count = `sum(nparam) + 1` and the chain array has the extra
   column.
3. `calc_stats` names end in `'B_p'`.
4. `reconstruct_from_chains` on a `fit_Bp` chain returns correctly-shaped
   IOPs/Rrs.
5. `fit_Bp=True` + `rt_backend="gordon"` raises at setup.
6. `fit_Bp=False` results identical to an M2-pinned value.

Existing suite green throughout.

## Q&A

_(none yet — this prompt has not been executed)_

## Next

→ `rob_rt_prompt_5.md` (M4: Ed wiring and inelastic terms through `robust`).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

_(none yet)_

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

Per the working agreements in `rob_rt_prompt_1.md` (git by JXP on
`rob-rt-backend`; `ocean14`; CQ1–CQ4 binding; scope discipline;
pytest-gated; Fable; log). Milestone-specific emphasis:

- **The prior framing is a plan choice, already made (design §7.3 →
  coding-plan M3)**: `B_p` is a **linear-space** parameter (a ratio, like
  slopes — not a log10 amplitude), `uniform` flavor over **[0.004, 0.05]**
  by default, inside `PhaseParams.validate`'s definitional (0, 1] bound.
  Don't re-open it; if the round-trip gate argues for a different range,
  raise it in Q&A rather than changing it silently.
- **The `aparams`/`bparams` split is load-bearing** (inference.py:93-94,
  chisq_fit.py:170-171): peel the `B_p` tail *first*, leave the split
  untouched.

## Context

Read before coding:

- **Previous prompt** — `rob_rt_prompt_3.md` (M2: fitter signatures as they
  now stand) and its Logs.
- **Coding plan** — `docs/coding_plan/rob_rt_coding_plan.md` **M3** section.
- **Design** — `docs/design/rob_rt_design.md` §3.3 (`B_p` free/fixed, the
  tail-append layout, the gordon+`fit_Bp` configuration error).
- **Q&A record** — `claude_prompts/rob_rt.md`: Design Q7 (free-or-fixed
  `B_p`) and the round-2 finding that `PhaseParams.B_p` broadcasts against a
  scalar, so a scalar MCMC parameter is well-supported.
- **Current code** — `bing/fitting/inference.py` (the split,
  inference.py:93-94; `init_mcmc` ndim, inference.py:159-160),
  `bing/fitting/chisq_fit.py` (the split, chisq_fit.py:170-171),
  `bing/evaluate.py` (`reconstruct_from_chains`, evaluate.py:245), and
  `robust/rt/types.py` (`PhaseParams` + `validate`, types.py:250-275).
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

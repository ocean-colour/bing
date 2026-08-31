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

**Q1 (task 1, Claude → JXP, resolved — no answer needed).** The Context
note says task 1's peeled `Bp` "should flow into the fitters' existing
`robust_domain_check` calls too" — but those are *setup-time* calls on
`p0`/posterior-median (full parameter vectors), where no peeled `Bp`
variable exists; they need their own tail peel, not just a threaded name.
The sequencing question was whether that peel belongs in task 1 or must
wait for task 3's ndim bookkeeping. Resolution, by reading the code: it
belongs in task 1, because the peel there is well-defined *now* — design
§3.3's layout contract (`[a_params..., bb_params..., B_p]` whenever
`fit_Bp` is True) binds regardless of who built `p0`, and nothing at
those call sites depends on `init_mcmc`'s not-yet-built ndim slot
(`run_emcee` already derives `ndim = len(p0)` itself, inference.py:534
pre-change). So all three call sites — `fit_one`'s p0 check and
posterior-median check, `chisq_fit.fit`'s p0 check — gained the same
`fit_Bp`-gated peel, byte-identical no-ops when `fit_Bp` is False/absent.
What genuinely *cannot* work before task 3 is a real `fit_one` MCMC run
with `fit_Bp=True`: `init_walkers` clips against `prior_bounds(models)`,
whose length has no `B_p` slot yet (shape mismatch against a tailed p0)
— hence the direct-call/stubbed-sampler tests, per M2 task 1's
precedent. (`chisq_fit.fit` with a tailed p0, by contrast, already works
end-to-end mechanically — curve_fit just optimizes the tail as one more
free parameter — but pinning that is task 3's gate, not task 1's.) No
answer needed to proceed.

**Q2 (task 1, Claude → JXP — finding resolved for tasks 1–2; one call
wanted from JXP before task 3's `validate_rt_dict` pin).**
`robust_baseline` + `fit_Bp=True` is mechanically valid but physically
inert. Verified by reading `robust/rt/baselines.py`: `Rrs_gordon`
"accepts and discards" `phase_params` **by construction** — its module
docstring calls this the model's defining limitation (standard Gordon
has no phase-function input), not an implementation shortcut. The
adapter builds `PhaseParams(B_p=...)` for every backend (evaluate.py:424),
so the peeled tail flows in and is then ignored: two different tail
values give bit-identical baseline Rrs (pinned:
`test_fit_Bp_tail_inert_for_robust_baseline`). Consequence for a real
fit: the `B_p` posterior under `robust_baseline` would just reproduce
the prior — harmless but misleading. **Question for JXP (non-blocking
for task 2, wanted before task 3):** should `validate_rt_dict` reject
`fit_Bp=True` + `rt_backend='robust_baseline'` at setup, on the same
rationale as the existing gordon+`fit_Bp` error (design §3.3: "Gordon
has no phase-function input" — the baseline *is* Gordon)? Task 3 owns
the `validate_rt_dict` pin, so that's the natural landing spot either
way; task 2's prior/seeding work is unaffected.

**Q3 (task 2, Claude → JXP, resolved — no answer needed).** The
p0-seeding sequencing question: the spec says "p0 seeds at
`rt_dict['Bp_value']` (0.01)", but where p0 is *built* looked like it
might depend on task 3's bookkeeping. Resolution, by reading the code:
it does not — the seed is fully buildable now. The package's canonical
p0 builder is `l23.prep_one_l23` (model `init_guess` calls +
log10-by-prior-flavor loop, l23.py:388-399 pre-change), whose p0 feeds
all three l23 entry points (`l23.fit_one`, `fit_with_LM`, `batch_fit`);
notebooks/tests assemble p0 by hand. So task 2 added a small helper,
`inference.append_Bp_seed(p0, rt_dict)` (inference.py:411), which
appends `rt_dict.get('Bp_value', 0.01)` only when `fit_Bp` is truthy
(no-op otherwise, `rt_dict=None` included), and wired it into
`prep_one_l23` *after* the log10 loop and the `chk` block (both see
model parameters only; l23.py:407-414 post-change). Nothing here gets
redone in task 3, which extends the orthogonal machinery (`init_mcmc`
ndim, `prior_bounds`, `init_walkers` clipping). One correction to Q1's
parenthetical while verifying this: `init_walkers` clips with
`nclip = min(walkers.shape[1], low.size)` (inference.py:541-543
post-change), so a tailed p0 against an un-extended `prior_bounds` is
**not** a shape mismatch — the tail column is simply left unclipped.
Consequently a real `fit_one` MCMC run with `fit_Bp=True` is already
mechanically possible after task 2 (the 0.01 seed ± the 1e-3 walker
floor stays comfortably inside [0.004, 0.05], and `log_prob`'s new
prior guards any stray walker with -inf); task 3's
`prior_bounds`/`init_walkers` extension makes the clipping real rather
than fixing a crash. Pinning such a run remains task 3's gate.

**Q4 (task 2, Claude → JXP, resolved — no answer needed).** Whether
`chisq_fit.fit_func` needs a `-np.inf`-style B_p range check like
`log_prob`'s: no. Verified by reading the chi-square path end to end —
the *model* parameters get no in-`fit_func` range enforcement either;
bounds are entirely the optimizer's, via `curve_fit`'s `bounds=`
argument, which `chisq_fit.fit` threads through (default
`(-np.inf, np.inf)`, chisq_fit.py:135-136) and which `l23.fit_with_LM`
builds at the caller level from `p.apriors`/`p.bpriors` pmin/pmax. A
`-inf` branch inside `fit_func` would in fact be wrong: a curve_fit
model function must return a prediction vector for the residual, not a
log-density sentinel. So B_p got the identical treatment: when
`fit_Bp`, `fit_with_LM` appends `rt_defs.BP_PRIOR_PMIN`/`PMAX` as the
trailing bounds slot (l23.py:637-655 — `rt_dict` construction moved
above the bounds block), matching the now-tailed p0 from
`prep_one_l23`; `fit_func` itself is code-unchanged (docstring
documents the asymmetry against `log_prob`), and `fit`'s `bounds`
docstring tells hand-rolling callers to add the trailing slot from the
same constants.

**Q5 (task 3, Claude → JXP — decision made and clearly labeled;
M2 Q5 itself remains OPEN).** Gate item 6 ("`fit_Bp=False` results
identical to an M2-pinned value") had nothing to compare against: M2
never produced a fitter-level pin (M2 Q5, still unanswered), and the
Context section said not to pick silently but task 3 could not skip the
item. Resolution — the Context section's own fallback, implemented
deliberately, not silently: task 3 created a **provisional** fixed-`B_p`
fitter-level pin, `bing/tests/files/m3_fixed_bp_pin.npz` (generator:
`gen_m3_fixed_bp_pin.py`, run 2026-08-31 in `ocean14`), captured live
under task-3-complete code at the point `fit_Bp=False` was first tested
here. What it freezes: `chisq_fit.fit` best-fit params and seeded
`fit_one` chains (nsteps=30/nburn=10/nwalkers=16,
`np.random.seed(20260831)` — verified bit-reproducible in a fresh
process, so emcee really draws from the global legacy `np.random`
state) for both `gordon` and `robust_ztt` on the test module's own
synthetic recipe. Why "identical to M2" is a fair reading: at capture
time the M3 task-1 byte-identity tests
(`test_fit_Bp_false_or_absent_byte_identical_to_m2`) were green, so
these values *are* the M2 dispatch values — they had just never been
frozen to disk. The consuming test
(`test_fit_Bp_false_matches_provisional_pin`,
test_evaluate_robust.py:1696) and the generator's docstring both label
it provisional and stand-in explicitly, with a portability caveat
(chains can drift across BLAS/emcee versions — regenerate, don't
suspect a dispatch regression, if the byte-identity tests stay green).
**M2 Q5 is still genuinely unresolved at the M2 level**: if JXP wants a
true *pre-M3* pin, it has to come from a checkout of the M2 branch
state; this fixture supersedes nothing and is superseded by that if it
ever exists. **Q2 likewise remains open and untouched**: task 3 pinned
only the *existing* gordon+`fit_Bp` rejection at the fitter level (Gate
item 5) and did not extend `validate_rt_dict` to reject
`robust_baseline`+`fit_Bp` — that is still JXP's call (nothing in task
4 blocks on it; a notebook fit under `robust_baseline`+`fit_Bp` would
just reproduce the prior, per Q2).

**Q6 (task 3, Claude → JXP, resolved — no answer needed; two findings
worth knowing).** (i) The spec's "`reconstruct_from_chains` strips the
tail column ... and forwards it as `Bp`" quietly implied more than a
strip: the function had **no robust dispatch at all** — it called
`calc_Rrs_from_models` (Gordon) unconditionally and took no `geom`, so
a robust-backend chain was being silently reconstructed through the
wrong forward model (pre-existing, not an M3 regression; forwarding
`Bp` is only meaningful in a robust call). Resolution by reading
`log_prob`'s precedent: `reconstruct_from_chains` gained the same
`rt_dict.get('rt_backend', 'gordon')` dispatch plus a trailing
`geom=None` parameter (the adapter's own loud ValueError enforces geom
for robust backends; the Gordon path is byte-identical to before, and
the whole pre-existing suite stayed green). The `geom` threading was
extended minimally to its two live callers that could meet a robust
rt_dict — `io.save_fit` and `plotting.show_fits` (both `geom=None`
passthroughs) — nothing else. (ii) The **batched per-sample `B_p`
shape** is a real trap, verified live against robust: a `(nsamples,)`
tail column breaks `ztt.bb_tilde`'s broadcast against
`(nsamples, nwave)` IOPs, while `(nsamples, 1)` breaks the hybrid
emulator's `features()` (its `B_p[..., None]` promotion makes a 3-D
array that `broadcast_to` cannot squeeze back — robust's own
"degenerate case" docstring note, emulator.py:436-439). The one shape
all three backends accept is the full `(nsamples, nwave)` spectrum —
robust's own batched-`B_p` convention (`robust/rt/validation.py:271`)
— so `reconstruct_from_chains` forwards
`np.broadcast_to(tail[:, None], (nsamples, nwave))`, a read-only view,
no copy. Consequence pinned in the tests: the scalar-`B_p` and
per-sample-`B_p` jit traces are *different XLA programs*, so the
fixed-vs-free equivalence holds exactly for the IOPs but only to
float32 ulp for Rrs (rtol 1e-5 in the Gate-4 test).

**Q7 (task 4, Claude → JXP, resolved — no answer needed; three measured
findings that matter for M4's fix-or-free call and for anyone rerunning
the Gate test at higher nsteps).** The notebook's own long-chain fits
(nsteps=8000/nburn=1000/16 walkers — nsteps forced above `corner_plot`'s
hard-coded 7000-step burn, as the task-3 log predicted) exposed three
things the 800-step Gate test could not see. (i) **Noise, not the
likelihood surface, dominates the `B_p` interval**: on the Gate recipe,
noiseless data (0.5% assumed error) gives CI [0.0181, 0.0263] — task 3's
800-step measurement ([0.0177, 0.0240]) confirmed at 10x the chain
length — while a *single* seeded 0.5% noise draw more than doubles the
width to [0.0058, 0.0245] (median still 0.0202). (ii) **The noisy CI's
low edge is one stuck emcee walker, not posterior structure**: walker 6
sits pinned at `B_p`≈0.006 with compensating shifts in every other
parameter at log-prob −1288 vs the main mode's −37 (so not a genuine
second mode), holding exactly 1/16 = 6.25% of the samples — which is why
the 5th percentile lands inside it; the main 15-walker mode gives
[0.0175, 0.0246], and rerunning at nsteps=12,000 moves the full-sample
CI by <0.001 ([0.0058, 0.0254] — the walker stays stuck). Consequence
worth remembering: with 16 walkers, any 5th-percentile assertion is one
stuck walker away from moving — the Gate test passes at nsteps=800
(walkers haven't wandered yet), but if it is ever rerun at long nsteps
its `p5 > BP_PRIOR_PMIN` margin will narrow for exactly this reason.
(iii) **Clear water constrains `B_p` weakly even at optimistic noise**:
the identical round trip built on L23 idx=170's own initial-guess vector
(truth `B_p`=0.02, 0.5% noise, `Bnw`≈−3.5 — ~30x lower bb than the Gate
recipe) gives CI [0.0160, 0.0437] — 60% of the prior range, with samples
brushing the 0.05 prior edge (max sample 0.0500, the clip boundary).
Together with task 3's 2%-noise finding (prior nearly filled), the
notebook's recommendation for M4 is `fit_Bp=False` as the default for
single-spectrum inelastic fits, reserving the free tail for bb-bright
water or multi-spectrum constraints. No answer needed; recorded here
because these numbers are the empirical basis for that M4 default.

## Next

→ `rob_rt_prompt_5.md` (M4: Ed wiring and inelastic terms through `robust`).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-31 (M3 task 1 — B_p tail peel and forward flow)

**Read before coding.** This doc in full; `bing/fitting/inference.py` and
`bing/fitting/chisq_fit.py` (the fitters, in full); `bing/evaluate.py`
`_build_robust_inputs`/`calc_Rrs_from_models_robust`/`robust_domain_check`
(the `Bp=None` → `rt_dict['Bp_value']` fallback at evaluate.py:423, the
`PhaseParams(B_p=...)` construction at evaluate.py:424); design §3.3; and
`robust/rt/baselines.py`'s module docstring + `Rrs_gordon` (for Q2). The
Context section's re-verified citations were all still accurate:
`log_prob` at inference.py:53-54 with the split at 113-114 and the robust
dispatch's `Bp=None` at 127-134; `fit_func` at chisq_fit.py:175-176 with
the split at 230-231 and dispatch at 236-243; the three setup-time
`robust_domain_check` calls (fit_one p0 / fit_one posterior median /
chisq_fit.fit p0), all passing `Bp=None`. Also confirmed `run_emcee`
derives `ndim = len(p0)` itself, and that `init_walkers` clips against
`prior_bounds(models)` — the reason a real `fit_Bp=True` MCMC run must
wait for task 3 (Q1).

**Implemented** (post-change line numbers):

- `inference.log_prob` (inference.py:53): the `fit_Bp`-gated tail peel at
  inference.py:124-135 — `if rt_dict.get('fit_Bp', False): Bp =
  params[-1]; params = params[:-1]`, else `Bp = None` — placed *before*
  the untouched `aparams`/`bparams` split (now inference.py:137-138); the
  robust dispatch's hardcoded `Bp=None` became `Bp=Bp`
  (inference.py:159). `params` is an ndarray on the emcee path (slicing
  and scalar tail indexing are native); a plain sequence from a direct
  caller peels identically. Docstring documents the tailed layout and the
  `fit_Bp` key's two cases. No range check on the peeled value yet — the
  `B_p` prior is task 2, per the task split (an in-code comment says so).
- `chisq_fit.fit_func` (chisq_fit.py:184): same peel at
  chisq_fit.py:251-260, on the raw `*params` *tuple* (tuple slicing,
  before the existing `np.array(...)` split, now chisq_fit.py:262-264);
  guarded `rt_dict is not None and ...` so a `rt_dict=None` direct call
  fails exactly where it did before (at the dispatch). Dispatch's
  `Bp=None` → `Bp=Bp` (chisq_fit.py:277). Docstring updated likewise.
- **Setup-time domain checks (Q1's resolution)**: the same gated peel at
  all three `robust_domain_check` call sites — `fit_one`'s p0 check
  (inference.py:329-342, `Bp_check`), `fit_one`'s posterior-median check
  (inference.py:360-371, `Bp_med` — with `fit_Bp` the chain's flattened
  median carries the extra trailing column, so the median of the tail
  column is peeled and forwarded), and `chisq_fit.fit`'s p0 check
  (chisq_fit.py:160-171, `Bp_check`). Byte-identical no-ops when
  `fit_Bp` is False/absent.

Nothing else touched: no prior/p0 seeding (task 2), no
`init_mcmc`/`prior_bounds`/`init_walkers`/`reconstruct_from_chains`/
`calc_stats` changes (task 3), and M2 Q5's missing fitter-level pin left
untouched for task 3 as instructed.

**Tested.** Six new tests in `bing/tests/test_evaluate_robust.py`
(new "M3 task 1" section; module docstring's coverage list extended),
all direct-call or stubbed per M2 task 1's precedent:

- `test_log_prob_fit_Bp_peels_tail_and_forwards` /
  `test_fit_func_fit_Bp_peels_tail_and_forwards`: a tailed vector under
  `fit_Bp=True` is *exactly* equal (==/`array_equal`, not allclose) to
  the untailed call with `Bp_value` set to the tail value — the
  adapter-contract equivalence — and differs from the fixture-default
  `Bp_value`, proving the tail really flowed (robust_ztt).
- `test_fit_Bp_false_or_absent_byte_identical_to_m2`: the Goals
  section's hard constraint, pinned three ways on both gordon and
  robust_ztt — `'fit_Bp': False` vs the key deleted vs the M2 branches'
  own direct forward calls (`calc_Rrs_from_models` /
  `..._robust(Bp=None)`), all exactly equal, through both `log_prob` and
  `fit_func`.
- `test_fit_Bp_tail_inert_for_robust_baseline`: Q2's pin — two different
  peeled tails give bit-identical baseline Rrs.
- `test_chisq_fit_fit_Bp_p0_tail_reaches_domain_check` (curve_fit
  stubbed) and `test_fit_one_fit_Bp_p0_and_median_tails_reach_domain_check`
  (run_emcee stubbed with a fake sampler carrying a distinguishable
  constant `B_p` column): the recorded `robust_domain_check` calls see
  the un-tailed aparams/bparams lengths plus `Bp` = p0's tail and the
  chain median's tail respectively — never `Bp=None`.

Full suite in `ocean14`: **245 passed, 2 skipped, 2 failed** vs the M2
baseline's 239/2/2 — +6 for exactly the six new tests, and the 2
failures are the same pre-existing `test_l23_inelastic.py`
missing-fixture failures (M0/Q10), not a regression. The `fit_Bp=False`
byte-identity requirement is demonstrated by the new
`test_fit_Bp_false_or_absent_byte_identical_to_m2` plus the untouched,
green existing fitter tests (all of which run with `fit_Bp`
False/absent).

**Next.** Task 2: the `B_p` prior (linear-space `uniform` over
[0.004, 0.05], evaluated in `log_prob` alongside the model priors —
out-of-range tail → `-np.inf` before any forward call) and p0 seeding at
`rt_dict['Bp_value']`. Check Q2 for JXP's call before task 3's
`validate_rt_dict` pin.

### 2026-08-31 (M3 task 2 — B_p prior and p0 seeding)

**Read before coding.** This doc (Q1/Q2 answers, working agreements —
the prior framing is a made plan choice); the existing prior mechanism
in `bing/priors/priors.py`: each model carries a `Priors` whose
per-parameter objects return **0 in range / `-np.inf` out of range**
for the uniform flavors (`LogUniformPrior.calc`, priors.py:139-152,
strict `<`/`>` comparisons — i.e. **inclusive** bounds — and
`UniformPrior` subclasses it unchanged, priors.py:158); `log_prob` sums
`a_prior + b_prior` and short-circuits on
`np.any(np.isneginf([...]))` *before* the backend dispatch, so priors
already gate the forward call. Crucially, the in-range uniform
contribution is **exactly 0** — this codebase tracks posterior shape
only, no `-log(width)` normalization anywhere — so the `B_p` prior had
to match that convention. Also read the p0 construction chain
(`l23.prep_one_l23`'s `init_guess` + log10-by-prior-flavor loop, whose
p0 feeds `l23.fit_one`/`fit_with_LM`/`batch_fit`), `fit_with_LM`'s
caller-level `curve_fit` bounds built from `p.apriors`/`p.bpriors`, and
`init_walkers`' floored perturbation + `nclip = min(...)` clipping (the
Q3 correction to Q1's "shape mismatch" parenthetical). The
`debug-priors` skill's log10-vs-linear trap is the reason `B_p` is
deliberately `uniform` (linear) among `log_uniform` amplitudes; the
prep log10 loop keys on `flavor.startswith('log')`, so a
'uniform'-flavored `B_p` is never log10'd — the two stay consistent by
construction.

**Implemented** (post-change line numbers):

- `bing/rt/defs.py`: module constants **`BP_PRIOR_PMIN = 0.004` /
  `BP_PRIOR_PMAX = 0.05`** (defs.py:20-32, next to the hybrid wave
  constants; `validate_rt_dict` untouched) — the single source for the
  default free-`B_p` range, consumed by the MCMC prior, the chisq
  bounds, and (task 3) `prior_bounds`/`init_walkers`.
- `inference.py`: module-level **`BP_PRIOR`** (inference.py:53-66) — a
  `bing_priors.UniformPrior` built from those constants, so the
  convention (inclusive bounds, 0-in-range) matches the model priors by
  construction, and task 3 gets a pmin/pmax-bearing object for
  `prior_bounds`. In `log_prob`, `Bp_prior = 0. if Bp is None else
  BP_PRIOR.calc(Bp)` sits in the Priors block (inference.py:168-175),
  joins the existing `isneginf` short-circuit (inference.py:177 — so an
  out-of-range tail returns `-np.inf` *before* the dispatch at
  inference.py:180-191, mirroring how an out-of-range model parameter
  already did) and the final sum (`+ Bp_prior`, inference.py:200 —
  identically 0 whenever finite, so `fit_Bp=False` values are
  unchanged). Docstring: 'fit_Bp' key, Returns, and the log(P) formula
  all document the prior. Task 1's "no range check yet" comment removed.
- `inference.append_Bp_seed(p0, rt_dict)` (inference.py:411-448, Q3's
  resolution): appends `rt_dict.get('Bp_value', 0.01)` — linear-space,
  inside the prior, nonzero so `init_walkers`' 1e-3 floor gives the
  dimension real spread — only when `fit_Bp` is truthy; returns
  `np.asarray(p0)` unchanged otherwise (`rt_dict=None` included).
- `l23.prep_one_l23`: seeds via `append_Bp_seed(p0,
  rt_defs.rt_dict_from_p(p))` (l23.py:407-414), *after* the log10 loop
  and the `chk` block (both see model parameters only) — all three l23
  entry points now emit a correctly tailed p0 under `fit_Bp`.
- `l23.fit_with_LM` (Q4's resolution): `rt_dict` construction moved
  above the bounds block; when `fit_Bp`, the `curve_fit` bounds gain
  the trailing `BP_PRIOR_PMIN`/`PMAX` slot (l23.py:637-655). **No
  range check inside `fit_func`** — the model parameters have none
  either (bounds are the optimizer's job, and a `-inf` return would be
  wrong for a curve_fit model function); `fit_func`'s and `fit`'s
  docstrings document the asymmetry (chisq_fit.py:77-87, 227-237).

Nothing else touched: no `init_mcmc`/`prior_bounds`/`init_walkers`/
`reconstruct_from_chains`/`calc_stats` changes and no fitter-level pin
(task 3, with M2 Q5 and Q2 still pending JXP), no notebook (task 4).

**Tested.** Five new tests in `bing/tests/test_evaluate_robust.py`
(new "M3 task 2" section at test_evaluate_robust.py:1276; module
docstring extended; the task-1 section's stale "no prior yet" comment
updated — its tails all sit in-range, so those exact-equivalence pins
hold verbatim, which itself demonstrates the in-range contribution is
exactly 0):

- `test_log_prob_Bp_prior_in_range_reaches_forward`: tails 0.004 /
  0.02 / 0.05 (the inclusive bounds themselves included) give finite
  log-probability and demonstrably reach the adapter (pass-through
  recorder on `calc_Rrs_from_models_robust` sees exactly those Bp's).
- `test_log_prob_Bp_prior_out_of_range_short_circuits`: 0.0039 /
  0.0501 / 0. / -0.01 / 0.2 return `-np.inf` with **both** forward
  adapters replaced by bombs — the short-circuit is real, not a
  coincidental value.
- `test_log_prob_fit_Bp_false_never_touches_Bp_prior`: with
  `BP_PRIOR.calc` itself a bomb, fit_Bp False/absent stays finite on
  gordon and robust_ztt — and the untailed vector's last *model*
  parameter (beta=0.9, far outside [0.004, 0.05]) shows the check
  cannot misfire on the model tail.
- `test_append_Bp_seed_appends_under_fit_Bp` /
  `test_append_Bp_seed_noop_without_fit_Bp`: tail = `Bp_value`
  verbatim (0.01 default when the key is missing), head untouched,
  seed legal under `BP_PRIOR` and nonzero; False/absent/None rt_dicts
  return p0 unchanged.

Full suite in `ocean14`: **250 passed, 2 skipped, 2 failed** vs task
1's 245/2/2 — +5 for exactly the five new tests, and the 2 failures
are the same pre-existing `test_l23_inelastic.py` missing-fixture
failures (M0/Q10), not a regression. Also smoke-checked
`from bing.fitting import l23` imports and `append_Bp_seed` round-trips
live in `ocean14` (the suite does not import l23's heavy deps).

**Next.** Task 3, the bookkeeping: `init_mcmc` gains optional `rt_dict`
(+1 ndim under `fit_Bp`); `prior_bounds`/`init_walkers` extended by the
`B_p` bounds (per Q3 this upgrades an *unclipped* tail column to a
clipped one — no crash today — and `BP_PRIOR`/the defs constants are
ready for it); `reconstruct_from_chains` strips/forwards the tail;
`'B_p'` appended to `calc_stats` names and corner plots (linear label,
`log_params` False); the `validate_rt_dict` gordon+`fit_Bp` pin. Two
JXP calls wanted first: **Q2** (reject `robust_baseline`+`fit_Bp`?) and
M2 **Q5** (where the `fit_Bp=False` fitter-level pin for Gate item 6
comes from).

### 2026-08-31 (M3 task 3 — bookkeeping)

**Read before coding.** This doc in full (Q1–Q4; the Context section's
M2-Q5/Gate-6 fallback instructions); `bing/fitting/inference.py` in full
(post-task-2 state); `bing/evaluate.py` (`calc_stats`,
`_build_robust_inputs`/`calc_Rrs_from_models_robust`,
`reconstruct_from_chains`); `bing/rt/defs.py` (`validate_rt_dict`'s
existing gordon+`fit_Bp` error, defs.py:106-110, and the task-2
constants); `bing/io.py` `save_fit` (the `calc_stats` names caller);
`bing/plotting.py` (`log_param_mask`, `corner_plot`);
`bing/fitting/l23.py` `prep_one_l23`; the coding plan's M3 section and
design §3.3; robust's `PhaseParams` (types.py:250-297), `ztt.bb_tilde`
(ztt.py:471-495), `emulator.features` (emulator.py:403-463), and
`validation.py:271` (the batched-`B_p` convention — see Q6). **Citation
drift, as expected**: the task text's `inference.py:159-160` for
`init_mcmc` was stale (found at inference.py:202 pre-change) and
`evaluate.py:245` for `reconstruct_from_chains` was very stale (found at
evaluate.py:683 pre-change — the Context section's re-verified citation
was the accurate one). Trust the post-change numbers below, not either
of those.

**Implemented** (post-change line numbers):

- `inference.init_mcmc` (inference.py:202): optional trailing
  `rt_dict=None`; ndim = `sum(model.nparam)` **+1** when
  `rt_dict.get('fit_Bp', False)` (inference.py:256-259), and the dict
  now also carries the computed `'ndim'` key (inference.py:260) so the
  Gate-2 bookkeeping is directly testable — `nwalkers = max(16, 2×ndim)`
  is unchanged in form and stays 16 for the standard 5→6 case (the
  Context section's smoke-fit expectation, confirmed).
- `inference.prior_bounds` (inference.py:467): optional `rt_dict=None`;
  under `fit_Bp` the (low, high) arrays gain the trailing slot from
  `rt_defs.BP_PRIOR_PMIN`/`PMAX` (inference.py:509-513 — the constants,
  never a re-typed literal).
- `inference.init_walkers` (inference.py:516): optional `rt_dict=None`
  forwarded into `prior_bounds` (inference.py:585); `run_emcee` passes
  its own `rt_dict` through (inference.py:693-695). Verified live per
  Q3's prediction: the existing `nclip = min(...)` logic now really
  clips the tail column — a 0.0045 seed with the 1e-3 floor throws
  walkers below 0.004 without the extension and clips them into
  [0.004, 0.05] with it, model columns bit-identical either way.
- `evaluate.reconstruct_from_chains` (evaluate.py:717): under `fit_Bp`
  the flattened chain's trailing column is stripped *before* the
  aparams/bparams split and forwarded as `Bp` (evaluate.py:800-812) —
  as a `(nsamples, nwave)` broadcast view, the one batched shape all
  three robust backends accept (Q6ii); plus the backend dispatch and
  trailing `geom=None` the spec's "forwards it as Bp" turned out to
  require (evaluate.py:817-828; Q6i — the function was Gordon-only
  before). Gordon path byte-identical to before.
- Naming: new helper `evaluate.chain_param_names(models, rt_dict=None)`
  (evaluate.py:102) — concatenated model `pnames` + trailing `'B_p'`
  under `fit_Bp`; consumed by `io.save_fit` (io.py:129, which also
  gained the `geom=None` passthrough, io.py:76/134) and
  `plotting.corner_plot` (plotting.py:539). `plotting.log_param_mask`
  (plotting.py:57) appends the `B_p` entry as **False** (linear —
  the debug-priors trap, dodged on purpose); `corner_plot`
  (plotting.py:465) gained `rt_dict=None` and labels the extra column
  `'B_p'`, never log10-wrapped/exponentiated; `show_fits`
  (plotting.py:105) gained the `geom=None` passthrough (plotting.py:215).
- `l23.prep_one_l23`: builds `rt_dict` once (l23.py:302), feeds it to
  `init_mcmc` (l23.py:305-306) and reuses it for task 2's
  `append_Bp_seed` (l23.py:420).
- `validate_rt_dict` itself **untouched** (Q2 honored — no
  `robust_baseline` rejection added); its existing gordon+`fit_Bp`
  error is now pinned at the fitter level (Gate 5).
- Gate 6: the provisional pin `bing/tests/files/m3_fixed_bp_pin.npz` +
  generator `gen_m3_fixed_bp_pin.py` (see Q5 for the full reasoning and
  labeling; verified bit-reproducible in a fresh process before
  committing to the approach). **Commit note for JXP**: the repo's
  blanket `.gitignore` `*.npz` rule catches the new fixture — it needs
  `git add -f bing/tests/files/m3_fixed_bp_pin.npz`, exactly as
  `l23_gordon_fixture.npz` evidently was (it is tracked despite the
  same rule).

**Round-trip result (Gate 1, real numbers).** Synthetic `robust_ztt`
Rrs at `B_p=0.02` (truth `_THREAD_TRUTH`, 61-band 400–700 nm grid,
0.5% assumed error, θs=30°), fit with `fit_Bp=True` seeded at the 0.01
default, `np.random.seed(1234)`, nsteps=800/nburn=200/nwalkers=16
(~3 s): **posterior median B_p = 0.0205, 5–95% CI [0.0177, 0.0240]** —
truth inside, median inside its own CI, both prior edges (0.004/0.05)
comfortably excluded. (At 2% assumed error the posterior is much
broader — CI [0.0053, 0.0429], still edge-free — which is why the test
uses 0.5%; worth remembering for M4's fix-or-free call: with realistic
noise B_p is only weakly constrained by a single spectrum.)

**Tested.** Ten new tests in `bing/tests/test_evaluate_robust.py`
(new "M3 task 3" section at test_evaluate_robust.py:1409; module
docstring extended): `test_init_mcmc_fit_Bp_adds_dimension` (:1426),
`test_prior_bounds_fit_Bp_appends_slot` (:1452),
`test_init_walkers_clips_Bp_tail_into_prior` (:1472),
`test_fit_Bp_roundtrip_recovers_Bp` (:1498, Gates 1+2),
`test_chain_param_names_and_calc_stats_end_in_B_p` (:1551, Gate 3),
`test_corner_plot_B_p_column_linear_label` (:1575, labels read back off
rendered axes per test_plotting's recipe),
`test_reconstruct_from_chains_fit_Bp_shapes_and_tail_flows` (:1609,
Gate 4 — IOPs exactly equal fixed-vs-free, Rrs to float32 ulp, see
Q6ii), `test_fit_one_gordon_fit_Bp_raises_at_setup` (:1660) and
`test_chisq_fit_gordon_fit_Bp_raises_at_setup` (:1680) (Gate 5, with
run_emcee/curve_fit bombed to prove "at setup"), and
`test_fit_Bp_false_matches_provisional_pin` (:1696, Gate 6 via Q5's
provisional pin). Full suite in `ocean14`: **260 passed, 2 skipped,
2 failed** vs task 2's 250/2/2 — +10 for exactly the ten new tests,
and the 2 failures are the same pre-existing `test_l23_inelastic.py`
missing-fixture failures (M0/Q10), not a regression. Also smoke-checked
the `l23`/`io`/`plotting` imports and new signatures live (the suite
does not import l23's heavy deps).

**Next.** Task 4, the explainer notebook `nb/RT/rob_rt_coding_4.ipynb`:
the synthetic round-trip with the corner plot's `B_p` column (pass
`rt_dict` to `plotting.corner_plot`, and note `thin_burn_chains`/
`corner_plot` both hard-code a 7000-step burn — the notebook needs
nsteps comfortably above that, or its own flattening as the round-trip
test does), plus the why-linear-space section. Q2 remains the one open
JXP call (nothing in task 4 blocks on it); M2 Q5 remains open at the
M2 level, with Q5's provisional pin covering M3's Gate 6 meanwhile.

### 2026-08-31 (M3 task 4 — round-trip notebook)

**Read before coding.** This doc in full (Q1–Q6, the task-3 log's two
notebook-specific warnings: pass `rt_dict` to `corner_plot`, and clear
the hard-coded 7000-step burn); the `plot-bing-fit` and `debug-priors`
skills; the Gate test `test_fit_Bp_roundtrip_recovers_Bp` and the
`threading_setup` fixture (test_evaluate_robust.py:752-778, 1498-1548 —
the recipe the notebook reuses); `plotting.corner_plot`/`show_fits`/
`log_param_mask`, `evaluate.reconstruct_from_chains`/`thin_burn_chains`/
`chain_param_names`, `inference.init_mcmc`/`append_Bp_seed`/`BP_PRIOR`,
`l23.prep_one_l23`/`load_one_l23`, `rt_defs.rt_dict_from_p`; and
M1/M2's notebooks (`rob_rt_coding_2/3.ipynb`) for house style. One
style decision, explained in the notebook's own Setup cell: the
*primary* round trip uses the Gate test's synthetic recipe (truth
`[-1.0, 0.015, -0.7, -2.0, 1.0]` + `B_p=0.02`, 61-band 400–700 nm
grid, θs=30°) rather than M1/M2's L23 idx=170 anchor, because a round
trip needs a parameter-space truth and bb bright enough to constrain
`B_p`; the L23 spectrum returns in section 6 where the clear-water
contrast is itself the point.

**Built and executed** `nb/RT/rob_rt_coding_4.ipynb` (27 cells, 11 code;
executed top-to-bottom in `ocean14` via
`jupyter nbconvert --to notebook --execute --inplace`, ~80 s):

- **Setup + §1–2**: models/`rt_dict` construction with `fit_Bp=True`
  (`rt_backend='robust_ztt'`); truth Rrs generated through
  `chisq_fit.fit_func` with the *tailed* truth vector (the same forward
  call the sampler uses), one seeded 0.5% multiplicative Gaussian noise
  draw added (`default_rng(1234)`), `varRrs=(0.005·Rrs_true)²`; the
  fit: `init_mcmc(..., rt_dict=...)` books ndim 5→6 / 16 walkers,
  `append_Bp_seed` tails p0 at the 0.01 default, seeded `fit_one`
  (`np.random.seed(1234)`, nsteps=8000/nburn=1000 — nsteps chosen to
  clear the 7000-step corner-plot burn) returns finite (8000, 16, 6)
  chains in ~25 s.
- **§3 (the round-trip numbers, all freshly measured by the notebook's
  own run)**: flattened with the same burn=7000 the corner plot uses
  (16,000 samples) — **B_p posterior median 0.0202 vs truth 0.02, 5–95%
  CI [0.0058, 0.0245]**; truth inside, both prior edges (0.004/0.05)
  excluded, every sample inside the prior (range [0.0057, 0.0312]),
  χ²ν(median model)=1.40, rough τ(B_p)≈167 steps (~42τ discarded). A
  `B_p`-marginal figure shows posterior vs truth vs prior edges.
  Two companion measurements: the **noiseless** rerun (identical
  otherwise) gives median 0.0211, CI [0.0181, 0.0263] — task 3's
  800-step numbers confirmed at 10x chain length — and a
  `diagnose-mcmc` per-walker check shows the noisy CI's odd low edge is
  **one stuck walker** (walker 6 at B_p≈0.006, log-prob −1288 vs the
  main mode's −37, exactly 1/16=6.25% of samples; main mode
  [0.0175, 0.0246]). See Q7 for all three findings and the M4
  implication.
- **§4**: `plotting.corner_plot(chains, models=models, rt_dict=rt_free)`
  — the full 6-parameter posterior with the `B_p` column labeled plain
  `B_p` (linear) among `log10(...)` amplitude labels and linear slopes;
  blue 5/95 lines match §3's printed CI.
- **§5**: `show_fits` three-panel on the free-`B_p` robust chain
  (`geom` passed through; truth overlays evaluated from the models at
  truth) — exercising task 3's `reconstruct_from_chains` robust
  dispatch + tail strip + `(nsamples, nwave)` `Bp` broadcast live
  (χ²ν=1.32 shown on the Rrs panel).
- **§6**: the identical round trip on L23 idx=170's own
  `prep_one_l23` init-guess vector as truth (+`B_p=0.02`, same noise
  and sampler settings): median 0.0257, **CI [0.0160, 0.0437] — 60% of
  the prior range**, samples brushing the 0.05 edge (max 0.0500). The
  clear-water weak-constraint finding that feeds M4's fix-or-free call.
- **§7 + bottom line**: the why-linear-space section — the live-printed
  prior table (log_uniform [-6,5] amplitudes vs uniform slopes vs
  uniform [0.004, 0.05] `B_p`; 10^11 vs 12.5x dynamic range), the
  `debug-priors` log10-vs-linear trap and why the
  `flavor.startswith('log')` p0 loop makes the linear `B_p` consistent
  by construction, and a "Measured bottom line" summary of M3 in the
  notebook's own numbers. Q2's `robust_baseline`+`fit_Bp` combination
  is not touched by the notebook (still JXP's open call).

**Output verification (the recurring-defect check, done deliberately).**
Smoke-tested the exact fit recipe standalone in `ocean14` *before* the
notebook run (three variants: noisy, noiseless, L23 — plus a
12,000-step convergence check and the per-walker stuck-walker
diagnosis, all in scratch scripts); then, after `nbconvert --execute`,
re-read the saved .ipynb and confirmed execution_counts 1–11 sequential
with no nulls, every code cell carrying real stream/image outputs
(three image/png figures present), `nbformat.validate` clean; then
re-diffed **every** numeric claim in the markdown against the printed
outputs. That re-diff caught three prose errors, fixed by
markdown-only patches (no outputs invalidated): "Bnw pinned to a
fraction of a percent" (the table shows ~4% full-sample), "median 30%
above truth" (0.0257/0.02 = ~29%), "~26 s" (cell prints 25.3 s) — plus
one physics-claim fix, `b_bnw=10^-2` quoted at 400 nm when the `Pow`
pivot is 600 nm (verified in bbnw.py:608).

**Next.** Task 5: update `rob_rt_prompt_5.md` (M4) with what M3
established — the chain-shape/bookkeeping contracts M4's tests will
reuse (tailed `[a..., bb..., B_p]` layout; `init_mcmc`'s
`pdict['ndim']`; `chain_param_names`/`log_param_mask` trailing entries;
`reconstruct_from_chains`' tail strip + `(nsamples, nwave)` broadcast;
the `BP_PRIOR` constants), and the round-trip's constraint verdict for
the fix-or-free decision: `B_p` is recovered decisively on bb-bright
synthetic data at 0.5% noise (median 0.0202 vs truth 0.02) but is
weakly identified on a real clear-water spectrum (CI spanning 60% of
the prior at 0.5% noise; prior nearly filled at 2%) — i.e. M4's
inelastic fits should default to `fit_Bp=False` (Q7). Q2 and M2 Q5
remain the open JXP calls.

### 2026-08-31 (M3 task 5 — handing off to `rob_rt_prompt_5.md`)

**M3 is now fully complete (all 5 tasks done).** This task updated
`rob_rt_prompt_5.md` (M4) only — no source, test, or notebook changes.

**Read before writing.** This doc in full (Q1–Q7, all four task logs);
`rob_rt_prompt_5.md` in full (M4's Tasks/Gate are JXP's plan — its scope
is the Ed/Raman/fluorescence seam and nowhere mentions `fit_Bp`, which
is exactly why the hand-off facts go in its Context, not its Tasks);
and, per this series' recurring citation-drift lesson, every line number
written into the new Context bullets was re-verified against live source
first (`BP_PRIOR` inference.py:64, `init_mcmc` :202, `append_Bp_seed`
:427, `prior_bounds` :467, `init_walkers` :516; `BP_PRIOR_PMIN`/`PMAX`
defs.py:31-32; `chain_param_names` evaluate.py:102,
`reconstruct_from_chains` :717 with the tail strip/broadcast at
:806-811; `log_param_mask`/`show_fits`/`corner_plot`
plotting.py:57/105/465 — all matched the task-3 log's post-change
numbers, no drift since). The full-suite baseline was re-measured live
rather than assumed: `pytest bing/tests/ -q` in `ocean14` gives
**260 passed, 2 skipped, 2 failed** in ~2.5 min — identical to the
task-3 count (tasks 4–5 added no tests), the 2 failures the same
pre-existing `test_l23_inelastic.py` missing-fixture pair (M0/Q10).

**What `rob_rt_prompt_5.md` got** (Context section's "Previous prompt"
bullet expanded into four sub-bullets, matching the terse citation style
the M2→M3 hand-off used in this doc's own Context; plus the same
one-line Working-agreements branch correction `_2`/`_3`/`_4` carry —
`rob_rt`, not `rob-rt-backend`; Tasks/Gate/Q&A/Logs untouched):

1. **The fix-or-free verdict (Q7)**, with the measured numbers
   (synthetic median 0.0202 vs truth 0.02; noiseless [0.0181, 0.0263]
   vs one 0.5%-noise draw's [0.0058, 0.0245]; L23 idx=170's
   [0.0160, 0.0437] = 60% of the prior) and the explicit
   recommendation that M4 task 4's robust-vs-BING Raman/fluorescence
   comparisons keep `fit_Bp=False` so the inelastic comparison is not
   smeared by `B_p` posterior noise — plus Q7's stuck-walker
   5th-percentile artifact so it is not mistaken for a regression.
2. **The chain-shape/bookkeeping contracts** M4's tests can reuse
   (tailed layout, `pdict['ndim']`, the `BP_PRIOR` constants,
   `chain_param_names`, linear `'B_p'` labeling), with the re-verified
   citations above.
3. **The batched-`B_p` shape trap (Q6)** — `(nsamples, nwave)` is the
   only shape all three backends accept — with
   `reconstruct_from_chains`'s `np.broadcast_to(chains[:, -1:], ...)`
   named as the reference implementation, and the note that the
   function gained its robust dispatch only in M3 (Gordon-only before).
4. **The open threads**, one line: Q2 (`robust_baseline`+`fit_Bp`
   rejection) and M2 Q5 (the provisional
   `m3_fixed_bp_pin.npz` stand-in, regenerate-don't-suspect caveat) —
   JXP's calls, not M4's problem.

**State handed to M4.** The `B_p` machinery works end-to-end — tail
peel in both fitters and all three setup-time domain checks, the linear
[0.004, 0.05] prior gating `log_prob`, p0 seeding, ndim/walker/naming
bookkeeping, `reconstruct_from_chains` tail strip with the robust
dispatch it had been missing, and corner-plot labeling — with the
`fit_Bp=False` path pinned byte-identical to M2 throughout. Empirically,
`B_p` is only weakly constrained by a single real spectrum at realistic
noise, so M4's inelastic fits should default to `fit_Bp=False` and
treat the free tail as a bb-bright/multi-spectrum tool. Q2 and M2 Q5
remain open JXP calls; neither blocks M4.

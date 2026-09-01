# Coding Plan — `robust.rt` as a Selectable RT Backend for BING

*Staged, milestone-gated implementation plan for the design in
[`docs/design/rob_rt_design.md`](../design/rob_rt_design.md), which locks 15
Design + 4 Coding Q&A decisions recorded in
[`claude_prompts/rob_rt.md`](../../claude_prompts/rob_rt.md).*

This plan turns the design's 7 sections into concrete milestones **M0–M5**,
each with tasks, a deliverable, and a **pytest acceptance gate**. Nothing here
re-opens a resolved Q&A item; where the design left an implementation choice
open (§7), the choice is made explicitly below and marked *(plan choice)*.

## Ground rules (from Q&A/Coding and repo conventions)

- **Execution.** Claude implements on a **branch** (suggest `rob-rt-backend`);
  each milestone is a reviewable commit/PR-sized unit. **JXP runs all git**
  and reviews (per `CLAUDE.md`); nothing self-merges. Both repos run in the
  **`ocean14`** conda environment.
- **Float precision (CQ1).** The robust backend runs **float32** at the JAX
  boundary — `jax_enable_x64` is **never** enabled, globally or locally.
  BING's float64 NumPy arrays downcast to float32 crossing into
  `robust.rt`; this is documented in the new function's docstring as more
  than sufficient precision for these calculations. Test tolerances are set
  for float32 (see gates), not the x64-only `rtol ≤ 1e-6` cross-check regime.
- **Ed wiring (CQ2).** `aNWModel.set_raman_Ed` (bing/models/anw.py:480-511)
  is modified to **also stash the raw `(wave_Ed, Ed)` pair** (small,
  backward-compatible; `Ed_ratio_raman` is still computed exactly as today).
  The robust backend **reuses BING's existing zenith-0° production Ed
  generation as-is** (`fitting/l23.py:319-322`, via
  `correct_atmosphere.downwelling`) — re-routing an already-computed
  spectrum, no new physics.
- **Tests (CQ3).** New tests live in `bing/tests/test_evaluate_robust.py`
  and **assume `robust` is always present — no `importorskip`** (it is a real
  runtime dependency after M0; a broken install fails loudly, matching
  BING's existing convention). Where tests touch `bing.fitting.l23`, the
  module is added to the existing `correct_atmosphere` `collect_ignore` list
  (bing/tests/conftest.py:84-87).
- **Geometry (CQ4).** `theta_s` is **required, never defaulted**: a
  robust-backend fit with no geometry supplied raises at fit setup. The nadir
  fallback covers only missing *viewing* geometry (`theta_v = dphi = 0`).
- **BING conventions.** Amplitude parameters log10, slopes/ratios linear;
  `eval_*` returns `(nsample, nwave)`; the parameter vector splits as
  `aparams = params[:models[0].nparam]` / `bparams = params[models[0].nparam:]`
  (inference.py:93-94) and that split is preserved untouched; the Gordon
  path and its numerics are untouched except the one sanctioned deletion
  (`RT_correction`, design §6).

## Files touched (by milestone)

```
bing/
  setup.py                     # + 'retrieve-or-bust' in install_requires   (M0)
  bing/rt/
    defs.py                    # rt_dict_from_p: + rt_backend, fit_Bp,
                               #   Bp_value; validate_rt_dict() (NEW)       (M0)
    geometry.py                # NEW — ObsGeometry frozen dataclass         (M0)
    raman.py, chl_fl.py        # code unchanged; docstring pointers only    (M5)
  bing/evaluate.py             # NEW calc_Rrs_from_models_robust (+ JIT
                               #   cache + un-jitted domain-check helper);
                               #   RT_correction block deleted              (M1)
  bing/fitting/
    inference.py               # log_prob dispatch; fit_one/fit_batch
                               #   optional 5th tuple element; run_emcee
                               #   args; init_mcmc ndim; setup validation   (M2, M3)
    chisq_fit.py               # fit()/fit_func dispatch + 5th element      (M2, M3)
  bing/models/anw.py           # set_raman_Ed stashes raw (wave_Ed, Ed)     (M4)
  bing/tests/
    test_evaluate_robust.py    # NEW — gates for M1–M4                      (M1–M4)
    conftest.py                # collect_ignore addition (CQ3)              (M4)
  .claude/skills/
    run-bing-fit/SKILL.md      # new rt_dict keys, backend selection        (M5)
    inelastic-rrs/SKILL.md     # robust inelastic path is the recommended
                               #   one; bing.rt raman/chl_fl not recommended (M5)
```

No `robust`/retrieve-or-bust source is modified by this plan —
`robust.rt.forward` (robust/rt/hybrid.py:364) and its siblings are consumed
as published.

---

## Milestones at a glance

| M | Goal | pytest acceptance gate |
|---|------|------------------------|
| **M0** | Dependency, config keys, `ObsGeometry` | `import robust` works from bing's env; `rt_dict_from_p` emits the 3 new keys with legacy-safe defaults; `ObsGeometry` requires `theta_s`; illegal configs raise |
| **M1** | `calc_Rrs_from_models_robust` + drop `RT_correction` | `robust_baseline` ≡ `gordon` parity on L23 (float32 tol); shapes `(nwave,)`/`(nsamples, nwave)` under all 3 robust backends; hybrid grid error; `DomainWarning` surfaced by the un-jitted check; `RT_correction` gone |
| **M2** | Fitter dispatch + geometry threading | `fit_one`/`chisq_fit.fit` run end-to-end under all 4 backend values; legacy 4-tuples still work; `geom=None` + robust raises; nadir fallback ≡ explicit nadir; non-nadir changes Rrs |
| **M3** | Free `B_p` | free-`B_p` round-trip on synthetic truth; ndim/walker/stats bookkeeping; `fit_Bp` + `gordon` raises |
| **M4** | Ed wiring + inelastic via robust | raw Ed pair stashed backward-compatibly; robust-backend Raman/fluorescence vs BING's own path at float32 tolerance; `Geometry.Ed` actually consumed |
| **M5** | Deprecation notes, docs, throughput benchmark | full `pytest bing/tests/` green; benchmark numbers recorded (report, not thresholded); docs/skills updated |

---

## M0 — Dependency, config keys, `ObsGeometry`

**Tasks.**
- Add `'retrieve-or-bust'` to `install_requires` in `bing/setup.py`
  (setup.py:23-30). Distribution name `retrieve-or-bust`, importable package
  `robust`; JAX/Flax arrive transitively (design §5). No model files are
  added to BING — the trained emulator weights ship inside `robust`.
- `bing/rt/defs.py`: extend `rt_dict_from_p` (defs.py:5) key loop with
  **`rt_backend`** (default `"gordon"` when absent on `p`), **`fit_Bp`**
  (default `False`), **`Bp_value`** (default `0.01`) — defaults applied
  explicitly rather than the loop's `None`, so a legacy `p` object yields a
  fully valid dict. Every consumer still tolerates missing keys via
  `rt_dict.get("rt_backend", "gordon")` (design §3.1), so saved/legacy
  rt_dicts keep working unmodified.
- `bing/rt/defs.py`: new `validate_rt_dict(rt_dict, models=None, geom=None)`
  — the **fit-setup** checks (called by fitters in M2): (i) `rt_backend` ∈
  {`gordon`, `robust_ztt`, `robust_hybrid`, `robust_baseline`}; (ii)
  `fit_Bp=True` with `rt_backend="gordon"` raises (design §3.3); (iii)
  robust backend with `geom is None` raises — `theta_s` is required (CQ4);
  (iv) `robust_hybrid` with any `models[0].wave` band outside **[350, 750]
  nm** raises (design §4 — checked once here, never per forward call).
- New module `bing/rt/geometry.py`: frozen dataclass
  `ObsGeometry(theta_s, theta_v=0.0, dphi=0.0, wind=None)` (degrees; design
  §3.2 — `theta_s` positional/required, no default) with
  `to_robust() -> robust.rt.types.Geometry` (types.py:302; same units), plus
  an `Ed` pass-through slot for M4 (`to_robust(Ed=...)` keyword rather than a
  stored field, keeping the dataclass pure per-pixel metadata). Docstring:
  "Fixed per-pixel viewing/illumination geometry. Never fit."

**Deliverable.** `bing` imports `robust`; all new configuration surface
exists and validates; no fitter behavior changed yet.
**Gate.** New `test_evaluate_robust.py` (first tests): `import robust.rt`
succeeds; `rt_dict_from_p` on a legacy-style `p` yields the three new keys
with defaults; `validate_rt_dict` raises on each illegal combination above
and passes on legal ones; `ObsGeometry(theta_s=30.)` round-trips through
`to_robust()` (values, degrees); `ObsGeometry()` with no `theta_s` is a
`TypeError`. Existing suite stays green.

## M1 — The forward adapter: `calc_Rrs_from_models_robust`

**Tasks.**
- New function in `bing/evaluate.py`, directly below `calc_Rrs_from_models`
  (evaluate.py:94), with the design §3.4 signature:
  `calc_Rrs_from_models_robust(a_model, a_params, bb_model, bb_params,
  rt_dict, geom=None, Bp=None, debug=False, full_return=False)`.
  Implementation follows the design's mapping table exactly:
  - `a = a_model.eval_a(a_params)`, `bb = bb_model.eval_bb(bb_params)`
    (existing BING code, unchanged) →
    `robust.rt.types.IOPs.from_total_bb(a, bb, wave=a_model.wave, a_ph=...)`
    (types.py:131) — robust derives `bb_p = bb − bb_w(λ)` itself. `a_ph`
    (`10**a_params[..., -1:] * a_model.a_ph`, as at evaluate.py:222) is
    supplied only when fluorescence is on.
  - `PhaseParams(B_p = Bp if Bp is not None else rt_dict["Bp_value"])`
    (types.py:250-275).
  - `geometry = geom.to_robust()`; when the caller has only a solar zenith,
    `ObsGeometry(theta_s)` already encodes nadir viewing
    (≡ `Geometry.nadir(theta_s)`, types.py:337). `geom=None` never reaches
    the robust call in fit paths (M0 validation raises earlier); the function
    itself also raises on `None` so direct calls fail the same way.
  - `Inelastic(raman=..., fluorescence=..., phi_C=...)` (types.py:471) from
    `rt_dict["include_Raman"]` / `include_Chl_fl` / `phi_C`; `None` when both
    are off (bit-identical elastic path by construction).
  - backend suffix → `mode="ztt"`/`"hybrid"` into `robust.rt.forward`
    (hybrid.py:364, batched over leading axes — one call for a whole
    `(nsamples, nparam)` chain, matching `calc_Rrs_from_models`'s shape
    contract), or `robust.rt.baselines.Rrs_gordon` (baselines.py:97,
    signature-compatible). `mode="emulator"` is not exposed (design §3.1).
- **JIT strategy** *(plan choice, design §7.1)*: a module-level
  `functools.lru_cache`d builder `_robust_forward_jit(mode, inelastic_key,
  wave_key)` returns a `jax.jit`-wrapped closure with the wavelength grid
  and `Inelastic`/`PhaseParams` treedef baked in — one compile per
  configuration, reused for the whole fit (treedefs change when optional
  fields flip `None`↔set, so the key must include the inelastic config;
  `wave_key = wave.tobytes()`). No `vmap` needed: `forward` is natively
  batched. NumPy crosses to JAX only here (float32, CQ1 — stated in the
  docstring); results return as `np.asarray` for the likelihood arithmetic.
- **Un-jitted domain check** *(design §4)*: helper
  `robust_domain_check(a_model, a_params, bb_model, bb_params, rt_dict,
  geom, Bp=None)` in `evaluate.py` that calls the **un-jitted**
  `robust.rt.forward` once on concrete arrays so the emulator's
  `DomainWarning` (hybrid.py:130-162) can actually fire — it is silent under
  `jit` (traced inputs). Fitters call it **twice per fit** in M2: on the
  initial guess before sampling, and on the posterior median after — never
  inside the hot loop.
- **Drop `RT_correction`** (design §6): delete the block at
  evaluate.py:203-209, the key from any `rt_dict` construction, and any call
  sites that set it (grep both `bing/` and `papers/`; report — don't edit —
  `papers/` hits, per the don't-widen-scope rule).

**Deliverable.** A working, JIT'd, batched robust forward adapter callable
standalone; the Gordon path minus its deprecated fudge knob.
**Gate.** `test_evaluate_robust.py`: (i) **parity** — on ≥ 3 L23 spectra,
`rt_backend="robust_baseline"` matches `calc_Rrs_from_models` (elastic,
constant Gordon) at `rtol ≤ 1e-5` (both use G1=0.0949/G2=0.0794 and the same
A_Rrs/B_Rrs conversion; float32 sets the tolerance); (ii) **shapes** — for
each of the 3 robust backends, `(nparam,)` params → `(nwave,)` Rrs and
`(nsamples, nparam)` → `(nsamples, nwave)`, finite and positive-typical;
(iii) `full_return=True` returns `(Rrs, a, bb)` like the Gordon twin; (iv)
**grid** — `validate_rt_dict` with `robust_hybrid` on a grid reaching 760 nm
raises, on 350–750 passes, and `robust_ztt`/`robust_baseline` accept either;
(v) **domain** — `robust_domain_check` on a deliberately turbid IOP set emits
`DomainWarning` (`pytest.warns`), and the jitted path on the same inputs does
not error; (vi) `rt_dict` containing a stale `RT_correction` key is ignored
(no multiplication, no KeyError) and the block is gone; (vii) second call
with identical config hits the lru_cache (cache_info check) — no recompile.

## M2 — Fitter dispatch and geometry threading

**Tasks.**
- **Two-line dispatch** at both call sites (design §3.4): in
  `inference.log_prob` (branch replacing the single call at inference.py:105)
  and `chisq_fit.fit_func` (chisq_fit.py:174-175):
  `rt_dict.get("rt_backend", "gordon") == "gordon"` → existing call,
  unchanged; else → `calc_Rrs_from_models_robust(..., geom=geom, Bp=Bp)`.
  `log_prob` gains optional trailing `geom=None` (and `Bp` handling in M3);
  same for `fit_func`'s keyword set.
- **Observation tuple grows one optional trailing element** (design §3.2):
  `(Rrs, varRrs, params, idx[, geom])`. `fit_one` (unpack at
  inference.py:215) and `chisq_fit.fit` (unpack at chisq_fit.py:111) accept
  either length (`geom = items[4] if len(items) > 4 else None`); `fit_batch`
  docs (inference.py:480-486) gain the fifth element. *(Plan choice, design
  §7.2: keep the tuple — minimal, matches every existing call site; no
  dataclass promotion now.)* `geom` rides by value into `emcee`'s `args`
  list (`run_emcee`, inference.py:443) exactly like `Rrs`/`varRrs` — the
  sampler never sees it as a dimension.
- **Setup validation call**: `fit_one` and `chisq_fit.fit` call
  `validate_rt_dict(rt_dict, models=models, geom=geom)` (M0) before running —
  this is where the hybrid grid error and the missing-`theta_s` error
  surface, once per fit.
- **Domain check wiring**: `fit_one` runs `robust_domain_check` on `p0`
  before `run_emcee` and on the posterior median after (robust backends
  only); `chisq_fit.fit` runs it on `p0` only.
- Docstrings for `fit_one`, `fit_batch`, `chisq_fit.fit`, `log_prob` updated
  for the new element/keys.

**Deliverable.** All four `rt_backend` values selectable end-to-end from
`fit_one`/`fit_batch`/`chisq_fit.fit`, geometry threaded, legacy calls
untouched.
**Gate.** (i) end-to-end smoke fit: one L23 spectrum through `fit_one`
(small nsteps) and `chisq_fit.fit` under **each** of the four backend values
— chains/params finite, correct shape; (ii) legacy 4-tuple + default
(gordon) rt_dict reproduces today's behavior (regression against a pinned
pre-change result); (iii) robust backend + 4-tuple (no geom) raises at setup
with a message naming `theta_s`; (iv) `ObsGeometry(theta_s=30.)` (implicit
nadir) and `ObsGeometry(30., 0., 0.)` give identical `log_prob`; (v)
`ObsGeometry(30., theta_v=40., dphi=90.)` changes Rrs under `robust_ztt`
(non-nadir sensitivity); (vi) `fit_batch` with a mixed list of 4- and
5-tuples runs (gordon) / raises correctly (robust).

## M3 — `B_p` as an optional free MCMC parameter

**Tasks.**
- When `rt_dict["fit_Bp"]` is True, `B_p` is appended as the **last** element
  of the combined vector — `[a_params..., bb_params..., B_p]` (design §3.3).
  `log_prob` and `chisq_fit.fit_func` **peel the tail first**
  (`Bp = params[-1]; params = params[:-1]`) so the existing
  `aparams`/`bparams` split (inference.py:93-94, chisq_fit.py:170-171) is
  untouched; `Bp` flows to `calc_Rrs_from_models_robust(..., Bp=Bp)`.
- **Prior** *(plan choice, design §7.3)*: `B_p` is a linear-space parameter
  (a ratio, like slopes — not a log10 amplitude), `uniform` flavor over
  **[0.004, 0.05]** by default, inside `PhaseParams.validate`'s definitional
  (0, 1] bound. Evaluated in `log_prob` alongside the model priors (an
  out-of-range tail → `-np.inf` before any forward call). p0 seeds at
  `rt_dict["Bp_value"]` (0.01) — nonzero, so `init_walkers`'s floor gives it
  spread.
- **Bookkeeping**: `init_mcmc` gains optional `rt_dict` and adds 1 to ndim
  (inference.py:159-160) when `fit_Bp`; `prior_bounds`/`init_walkers`
  clipping extended by the `B_p` bounds; `reconstruct_from_chains`
  (evaluate.py:245) strips the tail column before model evaluation and
  forwards it as `Bp`; parameter name `'B_p'` appended for `calc_stats`
  names and corner plots (linear label — `log_params` semantics: False).
- `validate_rt_dict` already rejects `fit_Bp` + `gordon` (M0); a test pins
  it here where it matters.

**Deliverable.** Free-`B_p` fitting under any robust backend; fixed-`B_p`
(default) path byte-identical to M2.
**Gate.** (i) synthetic **round-trip**: generate Rrs with `robust_ztt` at a
known `B_p=0.02`, fit with `fit_Bp=True` — posterior median of `B_p` within
its 5–95% credible interval of truth and interval excludes the 0.004/0.05
prior edges; (ii) ndim/walker count = `sum(nparam) + 1` and chain array has
the extra column; (iii) `calc_stats` names end in `'B_p'`;
(iv) `reconstruct_from_chains` on a `fit_Bp` chain returns correctly-shaped
IOPs/Rrs; (v) `fit_Bp=True` + `rt_backend="gordon"` raises at setup;
(vi) `fit_Bp=False` results identical to an M2-pinned value.

## M4 — Ed wiring and inelastic terms through `robust`

**Tasks.**
- **`set_raman_Ed` stash (CQ2)**: add two fields to `aNWModel` beside
  `Ed_ratio_raman` (anw.py:254-260) — e.g. `wave_Ed_raw` / `Ed_raw` — and
  have `set_raman_Ed` (anw.py:480-511) store the incoming pair verbatim
  before computing the ratio exactly as today. Backward compatible: no
  signature change, `Ed_ratio_raman` unchanged, BING's own Raman path
  (evaluate.py:167) untouched.
- **Route into `Geometry.Ed`**: in `calc_Rrs_from_models_robust`, when
  `include_Raman` and the a-model carries the stashed pair, build the robust
  geometry with `Ed=(wave_Ed_raw, Ed_raw)` (types.py:320-327; robust builds
  the ratio internally, robust/rt/ed.py:154-183). When no pair is stashed,
  pass `Ed=None` — robust falls back to its packaged L23 spectra
  interpolated in `theta_s`, which is its documented default (no BING-side
  flat-Ed fallback replicated).
- **Production Ed source unchanged**: the zenith-0° generation at
  `fitting/l23.py:319-322` (and :356) keeps feeding `set_raman_Ed` as-is —
  the robust backend just re-uses what lands on the model (CQ2.2; sourcing
  per-pixel-`theta_s` irradiance is a design non-goal).
- **Fluorescence**: `include_Chl_fl` + `phi_C` flow through `Inelastic`
  (already plumbed in M1); the required `IOPs.a_ph` is asserted with a clear
  error when fluorescence is on but the a-model has no `a_ph` set
  (mirrors `set_aph` conventions).
- **conftest (CQ3)**: add `test_evaluate_robust.py` to the
  `correct_atmosphere` `collect_ignore` list (conftest.py:84-87) — its Ed
  tests import `bing.fitting.l23`. (Everything gated in M0–M3 must therefore
  not require `correct_atmosphere`; the Ed tests land only in this
  milestone, keeping earlier gates runnable without it. If reviewers prefer,
  the Ed tests can instead live in a separate `test_evaluate_robust_ed.py`
  so only that file is dropped — *(plan choice: separate file)*.)

**Deliverable.** Robust-backend inelastic physics fed by BING's real solar
spectrum; `bing.rt.raman/chl_fl` no longer on the recommended path (their
code untouched until M5's docstrings).
**Gate.** (i) after `set_raman_Ed`, the raw pair is stored and
`Ed_ratio_raman` is bit-identical to the pre-change value; (ii) with the
same Ed spectrum, `robust_baseline`+Raman vs `gordon`+Raman on L23 agree at
float32 tolerance (`rtol ≤ 5e-4` — the physics is a pinned port,
robust/tests/test_inelastic_bing_xcheck.py holds `1e-6` only under x64);
same comparison for fluorescence with matched `phi_C`; (iii) passing an Ed
pair vs `Ed=None` measurably changes the robust Raman term (the seam is
live); (iv) fluorescence-on without `a_ph` raises the clear error; (v) suite
still collects cleanly in an env without `correct_atmosphere` (the Ed test
file is dropped, nothing else).

## M5 — Deprecation notes, docs, throughput benchmark

**Tasks.**
- **Docstring pointers only** (design §6): `bing/rt/raman.py` and
  `bing/rt/chl_fl.py` module docstrings note they serve the `gordon` backend
  and are no longer the recommended inelastic path — point to
  `robust.rt.inelastic` via `rt_backend`. No code deleted.
- **Docs**: `rt_dict_from_p` docstring documents `rt_backend` (four values),
  `fit_Bp`, `Bp_value`; `.claude/skills/run-bing-fit/SKILL.md` gains the
  backend-selection knob in its config/pitfall sections (it documents
  `rt_dict` at lines 38-41/110/157 today); `inelastic-rrs/SKILL.md` gains a
  "recommended path" note (its current text describes the Gordon-only
  inelastic wiring). CLAUDE.md's rt-subpackage bullet gets one line on the
  new backend.
- **Throughput benchmark** (design §7.1 accept/optimize decision): a small
  script (suggest `dev/rob_rt/benchmark_backends.py`) timing `log_prob`
  through each backend at MCMC-realistic batch shapes; record
  calls/s vs `gordon` in the script's header. Report, not threshold — if
  `robust_hybrid` is grossly slower, that's an optimize decision for JXP,
  not a silent fix.
- Sweep: `grep -rn "RT_correction"` (must be empty in `bing/`),
  `grep -rn "rt_backend"` docs coverage, stale line-number check on the
  design doc's citations touched by this work.

**Deliverable.** Integration complete and documented; benchmark evidence in
hand for the accept/optimize call.
**Gate.** Full `pytest bing/tests/` green in `ocean14` (with and without the
`correct_atmosphere`-gated files); benchmark script runs and its numbers are
recorded in the PR description; `RT_correction` grep empty under `bing/`.

---

## Testing strategy

- **Framework/layout**: `pytest`, tests in `bing/tests/` as
  `test_evaluate_robust.py` (+ `test_evaluate_robust_ed.py` at M4), fixtures
  under `bing/tests/files/` — BING's existing layout. **No `importorskip`
  for `robust`** (CQ3). Deterministic seeds for any MCMC gate; tiny
  nsteps/nwalkers for smoke fits.
- **Tolerances are float32-honest** (CQ1): parity gates at `rtol ≤ 1e-5`
  (identical algebra, differing dtype), inelastic cross-checks at
  `rtol ≤ 5e-4`. No gate assumes the x64-only `1e-6` regime.
- **Regression pinning**: before M1's `RT_correction` deletion and M2's
  dispatch, pin current `calc_Rrs_from_models`/`log_prob` outputs on a
  reference L23 spectrum as fixture data, so "Gordon path untouched" is a
  test, not a claim.
- **Recurring gates**: shape contract `(nwave,)`/`(nsamples, nwave)` and
  legacy-rt_dict tolerance re-assert at every milestone that touches the
  call chain.

## Dependency changes

`bing/setup.py` `install_requires` (setup.py:23): add `retrieve-or-bust`
(imports as `robust`; brings JAX/Flax transitively). Nothing else — both
packages already coexist in `ocean14`, and `robust`'s emulator weights ship
with `robust`.

## Risks & de-risking

- **JIT recompiles across configs (M1).** Treedef changes (`Inelastic`
  `None`↔set, `B_p` scalar↔spectrum) each trigger one compile — correct but
  visible. The lru_cache key makes this deliberate; the M1 cache-hit test
  and M5 benchmark keep it observable.
- **`fit_batch` + ProcessPoolExecutor (M2).** Each worker process compiles
  its own JIT cache (JAX state is per-process). Acceptable for long MCMC
  fits (compile cost amortizes); the M5 benchmark measures it. If it bites,
  a fork-server/warmup note goes to JXP rather than a silent redesign.
- **Float32 in the likelihood (M1–M3).** Rrs ~1e-3 sr⁻¹ with varRrs from
  satellite noise floors is comfortably inside float32 dynamic range; the
  parity and round-trip gates would expose any real degradation. Documented
  per CQ1.
- **Turbid inputs (M1/M4).** Out-of-domain is warn-and-continue by design
  (Q8); the deliberate-turbid `DomainWarning` test keeps the warning path
  from rotting since it is silent under `jit`.
- **Scope creep into `papers/`.** `RT_correction` call sites outside
  `bing/` are reported to JXP, not edited (CLAUDE.md: don't widen scope).

## Definition of done

M5 gate passed: all four `rt_backend` values fit end-to-end through
`fit_one`/`fit_batch`/`chisq_fit.fit` with geometry threaded, `theta_s`
required, free-or-fixed `B_p`, robust-side inelastic fed by BING's solar
spectrum, `RT_correction` gone, the Gordon path otherwise regression-pinned
untouched — `pytest`-green in `ocean14`, benchmarked, documented, on the
`rob-rt-backend` branch for JXP to review and merge.

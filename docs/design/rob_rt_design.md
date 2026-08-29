# Design — `robust.rt` as a Selectable RT Backend for BING

*A buildable engineering plan for wiring retrieve-or-bust's hybrid
analytic+emulator radiative-transfer model into BING's MCMC and least-squares
fitters as a new, user-selectable backend — alongside, not in place of, the
existing Gordon (1988) forward model.*

Companion to the Q&A/Design log in
[`claude_prompts/rob_rt.md`](../../claude_prompts/rob_rt.md), which records
the full reasoning trail (15 resolved questions over three rounds) and is the
authoritative source for the decisions restated here. For the RT model itself
— the ZTT analytic backbone, the residual emulator, the inelastic terms — see
`retrieve-or-bust/design/rt_elastic_model.md` and
`retrieve-or-bust/context/RT/`; this document does not re-derive that physics.
It is about the **interface**: how BING calls `robust.rt`, what new data and
configuration flow through BING to do so, and what stays untouched.

Decisions locked in Q&A/Design (`claude_prompts/rob_rt.md`): **selectable
backend, not a replacement** (Q1); **fidelity mode chosen by the user at fit
time** (Q2); **geometry as fixed, non-fit per-pixel data with a nadir
fallback** (Q3, Q11, Q14); **a separate JIT'd forward function reusing BING's
structure, dispatched from the existing fitters** (Q4, Q13); **BING's own
inelastic modules kept but no longer recommended** (Q5); **error only when a
band falls outside the emulator's range** (Q6, Q12); **`B_p` optionally free
in the MCMC** (Q7); **warn-and-continue out of domain** (Q8); **drop
`RT_correction`** (Q9); **`bing` takes `retrieve-or-bust` as a real runtime
dependency** (Q10); **band-averaging out of scope but not precluded** (Q15).

---

## 1. Goals and non-goals

**Goals.**
- BING's two fitting entry points — the MCMC likelihood
  (`bing.fitting.inference.log_prob`, bing/fitting/inference.py:52) and the
  least-squares fitter (`bing.fitting.chisq_fit.fit_func`,
  bing/fitting/chisq_fit.py:123) — can forward-model Rrs through
  `robust.rt.forward()` (robust/rt/hybrid.py:364) instead of the Gordon
  relation, selected by a new `rt_dict["rt_backend"]` flag.
- All three `robust` fidelity levels are user-selectable at fit time:
  `ztt` (analytic backbone only), `hybrid` (ztt + learned emulator
  correction, the most accurate vs L23 truth), and `baseline` (the
  Gordon-compatible refit in `robust/rt/baselines.py`, the like-for-like
  validation swap).
- New plumbing to carry optional, fixed per-pixel viewing geometry
  (`theta_s`, `theta_v`, `dphi`) through the fitters, with a nadir-viewing
  fallback whenever it is not supplied.
- `PhaseParams.B_p` (the particulate backscattering ratio,
  robust/rt/types.py:250-275) exposed as an optional free MCMC parameter,
  with a constant default when not fit.
- The `robust` path is batched and JIT'd so MCMC throughput (10^5–10^7
  forward calls per fit) is not degraded by the emulator.
- The existing Gordon path — `evaluate.calc_Rrs_from_models`
  (bing/evaluate.py:94) and everything below it — is **left untouched** and
  remains the default backend.

**Non-goals (this doc).**
- **Band-averaged Rrs.** Band-center point evaluation only. Explicitly not
  precluded: because both `ztt` and `hybrid` evaluate natively at arbitrary
  wavelengths, SRF-weighted numerical integration over several point
  evaluations per instrument band could be added later **without any
  retraining** — a natural fast-follow for real MODIS/SeaWiFS fits, but not
  part of this design.
- **Per-instrument emulator retraining.** Not needed inside the emulator's
  350–750 nm training range (§4) and out of scope regardless.
- **Sourcing real satellite viewing geometry.** No geometry columns exist
  anywhere in `bing/data/` today; this design only builds the plumbing to
  accept per-pixel geometry optionally. Loading it from L2 metadata for a
  given instrument is future work.
- **Retiring the Gordon backend** or modifying its numerics in any way.
- The inversion methodology itself, priors, or model selection — unchanged.

---

## 2. Architecture

One new function, one new flag, one grown data tuple. Both fitters keep their
current structure and dispatch on `rt_dict["rt_backend"]`:

```
      MCMC sample / least-squares params  (aparams, bparams [, B_p])
                          │
        ┌─────────────────┴──────────────────┐
        │ inference.log_prob (inference.py:52)│
        │ chisq_fit.fit_func (chisq_fit.py:123)│
        └─────────────────┬──────────────────┘
                          │  rt_dict["rt_backend"] ?
            ┌─────────────┴───────────────────────────┐
        "gordon" (default)                  "robust_ztt" | "robust_hybrid"
            │                                    | "robust_baseline"
            ▼                                            ▼
 ┌───────────────────────────┐        ┌────────────────────────────────────┐
 │ calc_Rrs_from_models      │        │ calc_Rrs_from_models_robust  (NEW) │
 │ (evaluate.py:94)          │        │ (evaluate.py, adjacent)            │
 │ — UNCHANGED —             │        │  eval_a/eval_bb (existing BING)    │
 │ Gordon rrs = G1·u + G2·u² │        │  → IOPs / PhaseParams / Geometry   │
 │ + bing.rt raman/chl_fl    │        │  → jit'd robust.rt.forward()       │
 └───────────────────────────┘        │    (hybrid.py:364; batched walkers)│
                                      │  inelastic via robust.rt.inelastic │
                                      └────────────────────────────────────┘
                          │                              │
                          └──────────► Rrs(λ) ◄──────────┘
                                (same shape contract:
                          (nwave,) or (nsamples, nwave))

  fixed per-pixel data:  (Rrs, varRrs, params, idx [, geom])   — geom NEW,
  never sampled; geom=None → Geometry.nadir(theta_s)
```

The new function reuses BING's existing machinery wherever it exists: the
`a_model.eval_a` / `bb_model.eval_bb` parameter-to-IOP evaluation, the
`rt_dict` configuration style, and the batch shape convention of
`calc_Rrs_from_models` (single vector → `(nwave,)`; chain/walker batch →
`(nsamples, nwave)`). Only the IOP→Rrs step changes.

---

## 3. Interface and data model

### 3.1 Backend selection: one new `rt_dict` key

`rt_dict` is BING's flat flag dictionary, built by `rt_dict_from_p`
(bing/rt/defs.py:5) and threaded by value through `emcee` and
`curve_fit` into both fitters. We follow that style with a **single
combined selector** rather than two keys:

```python
rt_dict["rt_backend"] : str
    "gordon"           # default — existing path, calc_Rrs_from_models
    "robust_ztt"       # robust.rt.forward(..., mode="ztt")
    "robust_hybrid"    # robust.rt.forward(..., mode="hybrid")
    "robust_baseline"  # robust.rt.baselines.Rrs_gordon (Gordon-compatible refit)
```

Why one key and not `rt_backend` + `robust_mode`: (i) `rt_dict` is a flat
dict of scalars — a single string keeps dispatch a one-line comparison at
both call sites; (ii) it makes illegal states unrepresentable (no
`robust_mode` dangling while `rt_backend="gordon"`); (iii) a missing key
defaults to `"gordon"` via `rt_dict.get("rt_backend", "gordon")`, so every
saved/legacy `rt_dict` keeps working unmodified. `rt_dict_from_p` adds
`rt_backend` (plus `fit_Bp` / `Bp_value`, §3.3) to its key list.

Note the mapping to `robust`: `MODES = ("ztt", "emulator", "hybrid")`
(robust/rt/hybrid.py:62) — `"ztt"` and `"hybrid"` are `forward()` modes,
while the Gordon-compatible baseline is a separate module
(`robust.rt.baselines.Rrs_gordon`, robust/rt/baselines.py:97, deliberately
signature-compatible with `forward`). `mode="emulator"` (the bare correction
term) is not exposed to BING users — it is a diagnostic, not a forward model.

### 3.2 Per-pixel geometry: new fixed, non-fit data

`robust.rt` requires explicit BRDF geometry
(`Geometry(theta_s, theta_v, dphi, wind=None, Ed=None)`,
robust/rt/types.py:302). BING has none today: observations flow through the
fitters as bare `(Rrs, varRrs, params, idx)` tuples
(bing/fitting/inference.py:179-183, 215; `fit_batch` items,
inference.py:480-486), and the `params` element is the MCMC initial-guess
vector, not a metadata container. Geometry is therefore **new plumbing**,
and per the resolved Q&A (Q14) it is **fixed, known, per-pixel data — never
a sampled parameter**. It rides alongside `Rrs`/`varRrs`, not inside the
parameter vector.

**New container** — `ObsGeometry`, a small frozen dataclass in a new module
`bing/rt/geometry.py`:

```python
@dataclass(frozen=True)
class ObsGeometry:
    """Fixed per-pixel viewing/illumination geometry, degrees. Never fit."""
    theta_s: float          # solar zenith; 0 = sun overhead
    theta_v: float = 0.0    # sensor zenith; 0 = nadir (default)
    dphi:    float = 0.0    # sensor-sun relative azimuth
    wind:    float | None = None   # m/s, optional

    def to_robust(self) -> "robust.rt.types.Geometry":
        """Map onto robust.rt.types.Geometry (same units, degrees)."""
```

**Threading.** The observation tuple grows by one optional trailing
element:

```python
items = (Rrs, varRrs, params, idx)              # legacy, still valid
items = (Rrs, varRrs, params, idx, geom)        # geom: ObsGeometry | None
```

`fit_one` (inference.py:214-215) unpacks either length; `fit_batch`'s
`items` docs gain the optional fifth element. `geom` is passed by value down
to `calc_Rrs_from_models_robust` the same way `Rrs`/`varRrs` already travel
into `log_prob` via the `emcee` `args` list, so the sampler never sees it as
a dimension.

**Fallback (always).** When `geom is None` — or when only a solar zenith is
known — the robust path uses nadir viewing:
`Geometry.nadir(theta_s)` (robust/rt/types.py:337, sets
`theta_v = dphi = 0`). When nothing at all is supplied, `theta_s = 0.0` is
the documented default (the only geometry-adjacent precedent in BING is a
hard-coded 30° in a test fixture,
`bing/tests/files/gen_l23_inelastic_fixture.py`; whether the package default
should instead be 30° is left to the Coding Plan, §7). The Gordon backend
ignores `geom` entirely — its geometry remains baked into fixed mean-cosine
constants.

### 3.3 `B_p` as an optional free MCMC parameter

`robust.rt.PhaseParams` carries a single field, `B_p = bb_p / b_p`
(dimensionless, ~0.005–0.03; robust/rt/types.py:250-275). It broadcasts
against a per-scene scalar or a spectrum, so a scalar MCMC parameter is
directly supported. Two new `rt_dict` keys:

```python
rt_dict["fit_Bp"]   : bool   # default False — B_p held constant
rt_dict["Bp_value"] : float  # constant/default value, e.g. 0.01
```

- **Fixed (default):** `PhaseParams(B_p=rt_dict["Bp_value"])` inside the
  robust forward call; the parameter vector is unchanged.
- **Free (`fit_Bp=True`):** `B_p` is appended as the **last** element of the
  combined MCMC vector — `[a_params..., bb_params..., B_p]` — leaving the
  existing `aparams = params[:models[0].nparam]` /
  `bparams = params[models[0].nparam:]` split (inference.py:93-94) intact
  after the tail is peeled off. `log_prob` and `chisq_fit.fit_func` strip it
  before the split, and it gets its own prior (bounded within (0, 1], the
  definitional range enforced by `PhaseParams.validate`; a tighter default
  range ~[0.004, 0.05] is a Coding Plan choice). `ndim`, walker counts, and
  chain post-processing (`calc_stats` names, corner plots) pick up the extra
  column.
- `fit_Bp=True` with `rt_backend="gordon"` is a configuration error (raise at
  fit setup): Gordon has no phase-function input.

### 3.4 The new forward function: `calc_Rrs_from_models_robust`

New function in `bing/evaluate.py`, directly below `calc_Rrs_from_models` so
the two backends live side by side at the module's single choke point. Same
signature shape, plus the two new inputs:

```python
def calc_Rrs_from_models_robust(a_model, a_params, bb_model, bb_params,
                                rt_dict: dict,
                                geom: ObsGeometry | None = None,
                                Bp: float | np.ndarray | None = None,
                                debug: bool = False,
                                full_return: bool = False):
    """
    Rrs from model parameters via the robust.rt backend.

    Same contract as calc_Rrs_from_models: a_params/bb_params are (nparam,)
    or (nsamples, nparam); returns Rrs of shape (nwave,) or (nsamples, nwave)
    [with (Rrs, a, bb) when full_return].  geom is fixed per-pixel geometry
    (None -> nadir fallback); Bp overrides rt_dict['Bp_value'] when B_p is
    being fit (the stripped tail of the MCMC vector).
    """
```

Internal mapping (BING object → `robust.rt` input):

| BING side | robust side |
|---|---|
| `a = a_model.eval_a(a_params)`, `bb = bb_model.eval_bb(bb_params)` (existing code, unchanged) | `IOPs.from_total_bb(a, bb, wave=a_model.wave, a_ph=...)` (robust/rt/types.py:131) — robust derives the water/particle split `bb_p = bb − bb_w(λ)` itself |
| fluorescence: `aph = 10**a_params[..., -1:] * a_model.a_ph` (as in evaluate.py:222) | `IOPs.a_ph` (required by the fluorescence kernel) |
| `Bp` argument, else `rt_dict["Bp_value"]` | `PhaseParams(B_p=...)` |
| `geom.to_robust()`, else `Geometry.nadir(theta_s)` | `geometry` |
| `rt_dict["include_Raman"]`, `rt_dict["include_Chl_fl"]`, `rt_dict["phi_C"]` | `Inelastic(raman=..., fluorescence=..., phi_C=...)` (robust/rt/types.py:471); `None` when both are off |
| `rt_dict["rt_backend"]` suffix | `mode="ztt"` / `mode="hybrid"` in `forward()`; or `baselines.Rrs_gordon` |
| `a_model.wave` | `wave` |

The core call, for the `ztt`/`hybrid` backends:

```python
Rrs = robust.rt.forward(iops, phase_params, geometry, wave,
                        mode=mode,                       # "ztt" | "hybrid"
                        inelastic=inelastic,             # or None
                        on_out_of_domain="warn")         # §4
```

`forward` (robust/rt/hybrid.py:364) and `rrs_forward`
(robust/rt/hybrid.py:165) are differentiable JAX functions **batched over
leading axes**, so a whole walker/chain batch `(nsamples, nwave)` of IOPs is
one call — no Python loop over samples, exactly matching
`calc_Rrs_from_models`'s batch convention. The function wraps the robust
call in a module-level `jax.jit` closure cached per
`(mode, inelastic-config, wave-grid)` — the pytree treedefs of `PhaseParams`
and `Inelastic` change when optional fields flip between `None` and set, so
each configuration compiles once and is then reused for the entire fit (the
exact caching/vmap strategy is a Coding Plan item, §7). NumPy arrays from
`eval_a`/`eval_bb` cross the JAX boundary at this one point; outputs come
back as NumPy for the likelihood arithmetic.

**Dispatch at the call sites.** Both callers gain the same two-line branch;
nothing else in them changes:

```python
# inference.log_prob (inference.py:105) and chisq_fit.fit_func
if rt_dict.get("rt_backend", "gordon") == "gordon":
    pred = bing_eval.calc_Rrs_from_models(models[0], aparams, models[1],
                                          bparams, rt_dict)
else:
    pred = bing_eval.calc_Rrs_from_models_robust(models[0], aparams, models[1],
                                                 bparams, rt_dict,
                                                 geom=geom, Bp=Bp)
```

### 3.5 Inelastic terms

When `rt_backend` selects `robust`, Raman and chlorophyll fluorescence come
from `robust.rt.inelastic` (composed inside `rrs_forward`,
robust/rt/hybrid.py), configured by the same existing `rt_dict` flags
(`include_Raman`, `include_Chl_fl`, `phi_C`). This is safe by construction:
`robust`'s Raman factor and fluorescence kernel are explicit JAX ports of
BING's fixed physics, pinned bit-for-bit against
`bing.rt.rrs.calc_raman_correction_factor` / `calc_Rrs_fluorescence` at
`rtol ≤ 1e-6` on 150 samples
(`retrieve-or-bust/robust/tests/test_inelastic_bing_xcheck.py`). The
`Geometry.Ed` field (robust/rt/types.py:320-327) is the seam through which a
real downwelling-irradiance spectrum enters the Raman ratio; wiring BING's
`set_raman_Ed` spectra into it is a Coding Plan detail.

---

## 4. Wavelength-grid and out-of-domain policy

**Grid policy** (resolved Q6/Q12 — no per-instrument model suite is needed):

- `rt_backend="robust_ztt"` and `"robust_baseline"`: **no restriction**.
  The ZTT backbone is purely analytic and grid-agnostic — its only
  wavelength dependence is clamped interpolation valid over 350–800 nm
  (robust/rt/ztt.py:802-811); the baseline is wavelength-independent Gordon
  algebra.
- `rt_backend="robust_hybrid"`: **error at fit setup iff any requested band
  falls outside [350, 750] nm** — the emulator's training range. Inside that
  range, evaluation at arbitrary band centers is ordinary interpolation the
  emulator is designed for: it is deliberately pointwise in λ, "defined on
  any wavelength grid ... with λ as an input it can interpolate in"
  (robust/rt/emulator.py:75-79). The check runs once against `a_model.wave`
  when the fit is configured, not per forward call.
- No per-instrument emulator retraining, and no band-averaging (§1
  non-goals; band-averaging via SRF-weighted multi-point integration remains
  open as a fast-follow).

**Out-of-domain policy** (resolved Q8): when the emulator's input features
leave its L23 training domain (turbid coastal water will do this), the
policy is **warn and continue** — never hard-fail, never silently substitute
a different mode. This is `forward()`'s existing
`on_out_of_domain="warn"` behavior (a `DomainWarning` naming the offending
features, robust/rt/hybrid.py:130-162). One implementation consequence: the
domain check is *silent when inputs are traced* (i.e., inside `jit`;
hybrid.py:143), so the JIT'd sampling loop cannot itself emit the warning.
The warn-and-continue contract is met by running one explicit, un-jitted
domain check per fit — on the initial-guess parameters before sampling, and
again on the posterior median after — rather than per MCMC step (which would
also flood stderr at 10^6 calls).

---

## 5. Dependency and packaging

- `bing` takes `retrieve-or-bust` as a **real runtime dependency** (not
  test-only, not optional): add it to `install_requires` in `bing/setup.py`
  (setup.py:23). The distribution name is `retrieve-or-bust`; the importable
  package is `robust`. This transitively brings in JAX/Flax, which `robust`
  already declares.
- This reverses today's soft coupling, where only `robust`'s cross-check
  tests reference `bing` via `importorskip`. That test-side reference is
  unaffected (it is not a packaging dependency).
- Both packages run in the `ocean14` conda environment; no new environment
  is introduced.
- `robust`'s trained emulator weights ship with `robust` itself; BING adds
  no model files.

---

## 6. Deprecations

- **`bing/rt/raman.py` and `bing/rt/chl_fl.py`: kept, not recommended.**
  They remain in the codebase and continue to serve the `gordon` backend
  unchanged, but they are no longer the default/recommended inelastic path:
  whenever `rt_backend` selects `robust`, all inelastic physics comes from
  `robust.rt.inelastic` (already cross-checked to `rtol ≤ 1e-6`, §3.5).
  Their docstrings gain a pointer to the robust path; no code is deleted.
- **`RT_correction`: dropped entirely.** The multiplicative fudge in
  `calc_Rrs_from_models` (bing/evaluate.py:204-209), flagged in-code "THIS
  SHOULD BE REMOVED", is removed as part of this integration — the code
  block, the `rt_dict` key, and any call sites that set it. This is the one
  edit made to the Gordon path, and it is a deletion of a deprecated knob,
  not a numerical change to the model.

---

## 7. Open items for the Coding Plan

Implementation work this design implies but does not resolve:

1. **JIT/vmap strategy.** Exact placement of the `jax.jit` boundary in
   `calc_Rrs_from_models_robust`, the compile-cache keying across
   `(mode, inelastic, wave)` configurations, `float32` vs `float64`, and a
   throughput benchmark against the Gordon path (accept/optimize decision).
2. **Tuple vs dataclass.** Final shape of the grown observation package —
   optional fifth tuple element (minimal, as designed) vs promoting
   `(Rrs, varRrs, params, idx, geom)` to a small dataclass; and the default
   `theta_s` when no geometry is supplied (0° as documented, or 30° to match
   the existing inelastic fixture).
3. **`B_p` prior.** Functional form and default range for the free-`B_p`
   prior, plus the bookkeeping (parameter names in `calc_stats`, corner
   plots, chain I/O column counts).
4. **Ed wiring.** Route BING's `set_raman_Ed` solar spectra into
   `Geometry.Ed` for the robust Raman ratio.
5. **Test plan.** At minimum: `robust_baseline` vs `gordon` parity on L23
   spectra (the like-for-like swap); geometry threading (nadir fallback ==
   explicit nadir; non-nadir changes Rrs); the [350, 750] nm hybrid grid
   error; the out-of-domain warning on a deliberately turbid input; free-`B_p`
   round-trip on synthetic truth; and a `fit_one`/`fit_batch` end-to-end run
   under each backend value.
6. **Docs.** Update the `run-bing-fit` / `inelastic-rrs` skill docs and
   `rt_dict_from_p` docstring for the new keys.

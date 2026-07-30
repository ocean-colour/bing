# Backscattering models for turbid waters

## Goals

Give BING a non-water backscattering (`bb_nw`) family that can represent
**turbid, mineral-dominated water** (inland lakes, estuaries, turbid
coastal), so that BING — and IOPtics, which consumes BING's models —
can fit GLORIA-like hyperspectral Rrs instead of failing on it.

Concretely, this prompt doc covers four deliverables:

1. **`Pow2`** — a two-component (mineral + organic) particulate
   backscattering model in `bing/models/bbnw.py`.
2. **`PowFlex`** — a free-slope single power law (a *parameters-only*
   change: existing `Pow` class, wider `beta` prior). This is the
   **ablation/control** that proves two components are genuinely needed.
3. **`maxfev`** plumbed through `bing.fitting.chisq_fit.fit` so callers
   can raise scipy's evaluation budget (approved by JXP; IOPtics will
   expose it on `AlgorithmSpec`).
4. Tests, docs, and a small benchmark under `dev/` showing (a) no
   regression on open-ocean/L23 spectra and (b) a material improvement
   on turbid spectra.

**Non-goals.** Do *not* touch the absorption side (`a_w`, `a_ph`,
CDOM/NAP) — see "Why" below: it has been shown to have no leverage on
the wavelengths that fail. Do not change the Gordon coefficients in this
pass (a separate question; see Open Questions).

## Why (the evidence that motivates this)

This work comes out of an IOPtics investigation of chi-squared fits to
the GLORIA hyperspectral in-situ dataset. Full report, with figures,
numbers and DOI-verified references:
[../../IOPtics/reports/gloria_fits_report.md](../../IOPtics/reports/gloria_fits_report.md)
(script: `IOPtics/reports/scripts/gloria_fits_report.py`).

The findings that define this task:

- **Fits of turbid GLORIA spectra fail, and the wall is backscatter.**
  In the Gordon relation `Rrs ≈ G·b_b/(a+b_b)`, GLORIA's green
  (~560–570 nm) and NIR (~700 nm) Rrs peaks are particulate
  backscatter shining through *minima* of total absorption. Inverting
  the observed turbid Rrs through the fit's own absorption shows the
  **required** `b_b` rising to ~**0.2–0.4 m⁻¹** in the red, while the
  fitted single power-law `b_bp` sits flat at ~**0.013 m⁻¹** — short by
  an order of magnitude, in a spectral shape (rising toward the red)
  that a *decreasing* power law cannot produce without destroying the
  blue fit.
- **It is the functional FORM, not the parameter range.** `Bnw` is
  already `log_uniform(-6, 5)` — effectively unbounded. Re-fitting with
  deliberately over-wide priors (amplitudes 1e-8..1e8) changed the
  median reduced chi² by nothing (≈2.5e2 either way), under both
  Levenberg-Marquardt and MCMC.
- **Absorption is not the lever.** CDOM and NAP absorption decay as
  `exp(-S(λ-440))` and are ~1% (CDOM) of their 440-nm value by 700 nm;
  in the actual fits `a_dg` is ~8% of total absorption at 560 nm and
  ~0.06% at 700 nm. Widening CDOM/NAP priors *cannot* move 500–750 nm.
  Beyond ~570 nm absorption is owned by pure-water `a_w` (which rises
  ~500-fold from 440 to 750 nm; Pope & Fry 1997) plus the `a_ph` 675-nm
  band — both already in the model and adequate.
- **Noise inflation is bookkeeping, not a cure.** An error floor
  `σ = max(σ_measured, f·|Rrs|)` drops median chi²_ν from 247 → 72
  (f=5%) → 20 (f=10%), but leaves convergence (15/40) and the true
  median relative Rrs misfit (~**48%**) unchanged.
- **The current models fit *clear* GLORIA fine** (chi²_ν ≈ 0.1, ~6%
  misfit). Quality collapses as the Rrs peak moves red; turbid spectra
  are missed by 80–91% in the green-red.

The peer-reviewed literature says the same thing, from measurements:

- The particulate-backscatter spectral slope **flattens** toward
  wavelength-independence when inorganic minerals dominate — Snyder et
  al. (2008) [doi:10.1364/AO.47.000666]; Gordon et al. (2009) report
  `b_bp ∝ λ^-n` with `n ≈ 0.4–1.0` even in oligotrophic-to-mesotrophic
  water [doi:10.1364/OE.17.016192]; Doxaran et al. (2009) show turbid
  coastal `b_bp` is nearly flat and that the power-law *form* itself is
  distorted by residual particulate absorption
  [doi:10.4319/lo.2009.54.4.1257].
- The **backscattering ratio** `b̃_bp` is a composition signature:
  ~0.005 organic to ~0.03 mineral — Twardowski et al. (2001)
  [doi:10.1029/2000JC000404]; Boss et al. (2004)
  [doi:10.1029/2002JC001514]; Whitmire et al. (2007)
  [doi:10.1364/OE.15.007019]; McKee et al. (2009) report ~0.02–0.04 in
  mineral-rich coastal water [doi:10.1364/AO.48.004663]; Sullivan &
  Twardowski (2009) on the near-constant backward VSF shape
  [doi:10.1364/AO.48.006811].
- Backscatter **magnitude** in turbid water is set by suspended mineral
  sediment, with mass-specific backscattering ~10× the organic value —
  Neukermans et al. (2012) [doi:10.4319/lo.2012.57.1.0124]; Babin et al.
  (2003, *L&O*) [doi:10.4319/lo.2003.48.2.0843].
- The red/NIR reflectance peaks are backscatter through absorption
  minima and are the basis of SPM retrieval — Doxaran et al. (2002)
  [doi:10.1016/S0034-4257(01)00341-8]; Nechad et al. (2010)
  [doi:10.1016/j.rse.2009.11.022]; Gitelson (1992)
  [doi:10.1080/01431169208904125]; Gons (1999)
  [doi:10.1021/es9809657].

Net: turbid particulate backscatter is **larger in magnitude**,
**flatter (or rising) in spectral slope**, and **higher in
backscattering ratio** than the open-ocean regime `Pow`/`Lee`/`GSM`
encode. Hence a two-component mineral+organic `b_bp`, and a free-slope
control.

## Skills

Consider the skills in `.claude/skills/` — in particular
[add-bbnw-model](../.claude/skills/add-bbnw-model/SKILL.md).

⚠ **That skill has drifted from the code.** Its scaffold shows each
subclass overriding `eval_bbnw` and calling `super().__init__(wave)`.
The real code has `bbNWModel.__init__(self, wave, prior_dicts)` (priors
are built in the base) and dispatches evaluation from a **string
`if/elif` on `self.name` inside the base-class `eval_bbnw`**. Follow the
code map below, not the skill's scaffold — and please fix the skill (and
the `CLAUDE.md` template at "New Backscattering Model Template") as part
of the docs pass.

## Code

Coding guidelines:

- Use Python; PEP 8; keep lines under 80 characters.
- Reuse existing code (`bing/models/functions.py` already has the
  power law) rather than re-deriving it.
- Add inline comments explaining the physics choices, not the syntax.
- Import statements at the top of the file.
- Every method gets a docstring describing inputs and outputs; keep it
  numpydoc-style to match `bbnw.py`.
- Amplitudes are stored/fitted in **log10**; exponents stay linear.

## Environment / Testing

Use the `ocean14` conda environment. `conda activate` may not work
non-interactively — call the environment interpreter directly, e.g.

```bash
python -m pytest bing/tests/test_bbnw.py -q
```

Run the full suite before declaring a task done. Note that some tests
need the L23 data tree; keep new heavy tests fast (a single spectrum,
chi-squared not MCMC, where possible).

## Code map — read before coding

Everything below is in the `bing` package, **verified against the source
on 2026-07-26** (Prompt 1). Line numbers are as of that verification;
corrections from the first draft are marked ✎.

- **`bing/models/bbnw.py`**
  - `init_model(model_name, wave, prior_dicts=None)` (`bbnw.py:64`) —
    the `model_dict` factory literal is at `bbnw.py:76`, dispatch at
    `:83`. A new class must be added to that dict. ✎ `prior_dicts`
    **defaults to `None`**, which is why `init_model('Pow2', wave)`
    works with no priors at all.
  - `bbNWModel.__init__(wave, prior_dicts)` (`bbnw.py:200`) — calls
    `init_raman()` (`:205`) then `init_bbw()` (`:208`), and asserts
    `len(self.pnames) == self.nparam` (`:215`). Subclasses just forward
    to it; **do not** rebuild `bb_w`. ✎ Priors are built
    **conditionally** — `if prior_dicts is not None:` (`:211`) — so with
    no priors `self.priors` stays `None` and every fit path later dies
    on `model.priors.priors` (`l23.py:387`, `ioptics/run.py:36,46`).
    "Constructible without priors" is true; "usable" is not.
  - ✎ **`prior_dicts` is a required positional on the bb subclasses**
    (`bbNWPow.__init__(self, wave, prior_dicts)`, `bbnw.py:454`), unlike
    the a-models (`aNWExp.__init__(self, wave, prior_dicts=None)`). So
    `bbNWPow(wave)` raises `TypeError`. **Give the new classes
    `prior_dicts:list=None`** so they match `anw.py` and can be
    constructed directly in tests.
  - `bbNWModel.eval_bbnw(params, wave=None)` — ✎ **superseded by
    Prompt 11**: this was a string dispatch on `self.name`; it is now a
    template method that resolves the grid and delegates to each model's
    own `_eval_bbnw(params, wave)` (base raises `NotImplementedError`).
    Adding a model means writing that method plus an `init_model` entry.
    **Note the `wave` kwarg**: `eval_bb_ex` (`bbnw.py:288`)
    calls `eval_bbnw(params, wave=self.wave_ex)` (`:300`) to get
    backscatter at Raman *excitation* wavelengths. `Pow` and `Cst`
    honour it; **`Lee`, `GSM` *and* `Every` ignore it** — ✎ `Every` was
    missing from the first draft. Lee/GSM ignore it because their
    branches return `functions.gen_basis(params[...,-1:],
    [self.basis_func])` (`:269-271`) and `basis_func` was baked on
    `self.wave` at construction (`:523`, `:622`). **A new analytic model
    must honour `wave`.**
  - `eval_bb` (`bbnw.py:275`) returns `self.bb_w + self.eval_bbnw(
    params)` (`:286`); it takes no `wave` kwarg.
  - `init_guess(bb_nw)` — returns a starting parameter vector with
    amplitudes in **linear** space (see the contracts below).
- **`bing/models/functions.py::powerlaw(wave, params, pivot=600.)`**
  (`functions.py:61`) — `10**params[...,0] · (pivot/λ)^params[...,1]`.
  ✎ `np.outer` is used **only for the amplitude factor**; the exponent
  term broadcasts via `(params[...,1]).reshape(-1,1)` (`:73-74`).
  Verified empirically that it works on a 2-column *slice* of a wider
  array — `powerlaw(wave, p[..., 0:2])` and `powerlaw(wave, p[..., 2:4])`
  both return `(1, nwave)` for a 1-D size-4 `p` and `(nsample, nwave)`
  for a `(nsample, 4)` `p`, with correct values. ✎ Caveat: it is **not**
  "always (nsample, nwave)" — for ≥3-D params the `reshape(-1,1)`
  flattens leading dims (`(2,3,2)` → `(6, nwave)`). Harmless for 1-D
  vectors and 2-D chains, which is all BING passes.
- **`bing/parameters/standard.py`** — the combo factories.
  `expb_pow()` (`standard.py:8`) is the template; note
  `bpriors[1] = dict(flavor='uniform', pmin=0., pmax=2.)` (`:21`) —
  **`beta ≥ 0` means the slope can only *decrease* with wavelength. That
  single line is the constraint this whole task is about.**
- **`bing/parameters/p_ntuple.py`** — `gen(**kwargs)` (`:33-53`) is
  model-agnostic; it only merges defaults and fills `scl_noise`.
  Nothing needs changing for a new model. ✎ But see landmine 3 below:
  **nothing anywhere validates `len(bpriors) == model.nparam`**, and on
  the chi-squared path a mismatch corrupts p0 *silently*.
- **`bing/fitting/chisq_fit.py::fit(items, models, rt_dict,
  bounds=None)`** (`chisq_fit.py:42`) — `curve_fit` at `:95-97`; this is
  where `maxfev` goes. Note `:89-90` (`if bounds is None: bounds =
  (-np.inf, np.inf)`) is what decides which scipy branch runs.
- **`bing/fitting/l23.py`** — ✎ chi-squared bounds are built from the
  **raw prior dicts'** `pmin`/`pmax` at `l23.py:612-618` (not 611), and
  `p0` amplitudes are log10'd by **prior flavor** via the **Prior
  objects on the models** at `l23.py:384-390` (not 385), flavor test at
  `:388`. That asymmetry matters: a `gaussian` prior gives `KeyError`
  from the bounds path, and `apriors=None` (as in `standard.k2b`) gives
  `TypeError`.

### The three contracts that bite

1. **p0 is linear; the caller log10s by prior flavor.** `init_guess`
   returns amplitudes in linear units. BING (`l23.py:388`), IOPtics
   (`ioptics/run.py::_log_mask`, `:47`) and `bing/scripts/fit_Rrs.py`
   (`:112-113`, which log10s bb params **unconditionally**) then log10
   the amplitude slots. So the **order of `pnames` must match the order
   of the prior dicts**, and each amplitude slot must carry a `log_*`
   prior while each exponent slot carries a linear one. Both failure
   directions are silent: an eta declared `log_uniform` starts at
   `log10(0.0) = -inf` (or clipped to the bound), and an amplitude
   declared `uniform` is never log10'd yet `eval_bbnw` still does
   `10**`, so it is wrong by orders of magnitude.
2. **Chi-squared bounds come from `pmin`/`pmax`.** Both BING
   (`l23.py:614`) and IOPtics (`ioptics/run.py::_prior_bounds`, `:37`)
   read `prior.pmin`/`.pmax`. So new parameters must use `uniform` or
   `log_uniform` priors — a `gaussian` or `ratio` prior has no `pmin`,
   and IOPtics' `np.array(lows, dtype=float)` turns the resulting `None`
   into a **silent NaN** bound. No `RatioPrior` linking `Bmin`↔`Borg`,
   however tempting for degeneracy-breaking: an extra prior beyond
   `nparam` also makes `_log_mask` longer than `p0` → `IndexError`.
3. **`uniform` with a negative `pmin` is safe.** Worth stating because
   `UniformPrior` subclasses `LogUniformPrior` (`priors.py:158`), which
   looks alarming. It overrides only the `flavor` string;
   `UniformPrior.calc is LogUniformPrior.calc` and that `calc` just does
   `-inf` outside `[pmin, pmax]`, `0` inside (`priors.py:139-152`) — **no
   logarithm anywhere**. `uniform` and `log_uniform` are mathematically
   identical; the distinction is pure metadata driving contract 1. A
   negative-`pmin` uniform prior is already in production
   (`IOPtics/reports/scripts/gloria_fits_report.py:443` uses
   `uniform(-1, 4)` for beta).

### Landmines found in the Prompt-1 audit (read this)

Ranked by likelihood of biting. Everything here was verified in the
source; the first is the one that would have wasted the most time.

1. **MCMC walker init FREEZES any parameter whose p0 is exactly 0.**
   `bing/fitting/inference.py:321`:
   ```python
   p0 = np.tile(p0, (nwalkers, 1))
   p0 += p0*np.random.uniform(-1e-2, 1e-2, size=p0.shape)
   ```
   The perturbation is **multiplicative**, so a p0 entry of `0.0` gets
   *zero* spread across walkers. emcee's stretch move proposes along the
   vector between two walkers, so a dimension with zero inter-walker
   spread **never moves for the whole run** — and `fit_one` passes
   `skip_check=True` (`:231`), disabling the one emcee check that might
   have flagged it. The acceptance fraction stays healthy; the posterior
   for that parameter is a delta function at the initial guess with a
   zero-width credible interval. **Directly relevant:** the obvious
   `init_guess` seed for a flat mineral term is `eta_min = 0.0`.
   → **Mitigation in this pass: never seed an exponent at exactly 0**
   (use e.g. `0.05`). The real fix (additive, floored perturbation plus
   a clip into the priors) changes every existing MCMC run's
   initialization, so it is a JXP decision — see Open Questions.
   Related: even non-zero small values give an absurdly tight ball
   (`0.015` → std `9e-5`), and perturbed walkers are never clipped into
   the priors.
2. **Plotting exponentiates and log-labels *every* parameter.**
   `bing/plotting.py:262` prints `f'{model.pnames[ss]} = {10**params[ip]}'`
   for all params, and the corner path (`plotting.py:449-456`, mirrored
   in `l23.py:519-524`) does `coeff = 10**coeff` and wraps every label
   in `\log_{10}(...)`. This is **already wrong for `Pow`'s `beta`**; for
   a model that is half linear exponents it mis-reports half the bb
   panel (`eta_min ∈ [-0.5, 0.5]` renders as `[0.32, 3.16]`). Silent.
   → Minimal backward-compatible fix: declare a `log_params` boolean
   list on the model classes and have plotting honour it via
   `getattr(model, 'log_params', None)`, falling back to today's
   behaviour. (IOPtics' own corner path is clean — it never
   exponentiates: `ioptics/plotting.py:229-249`.)
3. **Nothing validates `len(prior_dicts) == model.nparam`.**
   `Priors.__init__` just does `self.nparam = len(pdicts)`
   (`priors.py:258-259`); the bb base asserts only
   `len(pnames) == nparam`. A stale 2-element `bpriors` against a
   4-param model raises `IndexError` deep inside `Priors.calc`
   (`priors.py:288`) — thousands of MCMC steps in, or inside a pool
   worker, and in IOPtics `run_batch(strict=False)` it is swallowed as
   an anonymous `fit_failed` row. On the **chi-squared** path it is
   worse than a crash: `l23.py:386-390` walks the prior list with its
   own counter, so a wrong-length list log10s the **wrong slots**.
   → One-line guard in the new classes (or the base):
   `assert self.priors is None or self.priors.nparam == self.nparam`.
4. **`set_standard_priors` blankets all bb params with
   `log_uniform(-6, 5)`** (`priors.py:364-367`) unless `p.bpriors` is
   not None — and IOPtics' `AlgorithmSpec.build_models` always calls it
   (`ioptics/algorithms/spec.py:188`). Omit `bpriors` from a combo and
   both etas silently become log-flavored (contract 1) *and* unbounded
   over `[-6, 5]`. → Always ship explicit `bpriors`; consider a default
   `prior_dicts` on the class. Good news: the "fixed beta" pathway
   (`priors.py:377-380`) is gated on `p.model_names[1] == 'Pow'`, so
   `p.beta`/`AlgorithmSpec.beta` can never mis-target index 1 of a new
   4-param block.
5. **`l23.py:842` keys extras on the literal name `'beta'`**
   (`if 'beta' in models[1].pnames:`). For a model with no `beta` the
   `beta`/`sig_beta` keys are **absent entirely** from `extras` (not
   NaN), so `process_all`'s aggregated CSV loses those columns —
   a `KeyError` risk downstream rather than a quiet NaN.
   → Extend that block to also record `eta_*`, or accept the absence
   knowingly.
6. **Latent pre-existing bug worth knowing (not ours to fix here).**
   Because `Lee`/`GSM`/`Every` ignore the `wave` kwarg, `eval_bb_ex`
   adds `bb_w_ex` (excitation grid) to `bb_nw` evaluated on the
   **emission** grid. It is silent only because `wave_ex.size ==
   wave.size`. It affects Raman fits with those three models today.
   See Open Questions.
7. **Keep `pnames` globally unique across the a- and bb-models.**
   `ioptics/evaluate.py:73` builds `params` as a **dict** keyed by
   pname, so a collision silently drops a column. Checked every
   `anw.py` pnames list: `Bmin`/`eta_min`/`Borg`/`eta_org` collide with
   nothing.
8. **`Priors.gen_bounds()` rejects any flavor but `'uniform'`**
   (`priors.py:311-316`, raises `ValueError: Unknown prior flavor:
   log_uniform`). The bb path never calls it today, but the natural way
   to write a multi-param `init_guess` is to copy `anw.py:1313-1315`
   (`bounds = self.priors.gen_bounds()`), which would crash on the new
   mixed prior list. Don't.
9. **Thin walker count.** `inference.py:160` sets
   `nwalkers = max(16, 2*ndim)`, i.e. **16** for ExpBricaud(3) + a
   4-param bb model (ndim=7). Legal, but thin for a model with a new
   amplitude/slope degeneracy — consider bumping for the benchmark.

**Useful precedent.** `aNWChase.eval_adg` (`bing/models/anw.py:1270-1275`)
already sums two `functions.exponential` calls on `params[...,:2]` and
`params[...,2:4]` — the exact slicing pattern the two-component bb model
needs. (It hardcodes `self.wave`, so don't copy that part.)

**Subsystems audited and confirmed parameter-count-agnostic:**
`inference.log_prob`/`ndim` (`:93-94,159`), `chisq_fit.fit_func`
(`:148-149`), `evaluate.calc_stats`, `evaluate.reconstruct_*`,
`thin_burn_chains`, `bing/io.py` save/load (`pnames`-driven),
`priors.priors_from_models`/`split_priors`, `stats.py` AIC/BIC
(`nparam`-summed), `models/utils.init`, and on the IOPtics side
`evaluate.py` (`na`/`k` from `nparam`), `metrics.py`, `diagnostics.py`,
`io.save_chain`, `algorithms/spec.py`, `registry.py`, and
`run.initial_guess` (its single-power-law QAA seed is only the *observed*
`bb_nw` estimate handed to `models[1].init_guess`, not an assumption
about the parameterization).

## Proposed changes

### 1. `PowFlex` — free-slope power law (parameters only, no new class)

The cheapest possible turbid extension, and the **control experiment**:
same `Pow` form, but let the slope flatten or *rise* toward the red.

```python
def expb_powflex(**kwargs):
    """ExpBricaud + a single power law with a FREE (possibly negative)
    slope, admitting the flat/'white' mineral limit of turbid water."""
    apriors = [dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
    apriors[1] = dict(flavor='uniform', pmin=0.01, pmax=0.02)
    bpriors = [dict(flavor='log_uniform', pmin=-6, pmax=5)]*2
    # beta < 0 => bb_nw RISES with wavelength (mineral/'white' limit).
    bpriors[1] = dict(flavor='uniform', pmin=-1., pmax=2.)
    params = dict(model_names=['ExpBricaud', 'Pow'],
                  apriors=apriors, bpriors=bpriors,
                  sSdg=0.002, set_Sdg=False)
    params.update(kwargs)
    return p_ntuple.gen(**params)
```

No class, no `init_model` entry, no `eval_bbnw` branch — `functions.
powerlaw` already handles a negative exponent. The GLORIA report
predicts this will **not** be enough (a single power law cannot make the
required rising, structured shape while keeping the blue), which is
exactly why it is worth measuring: it isolates "range" from "form".

### 2. `Pow2` — two-component mineral + organic backscatter (the fix)

Per JXP's answers: the **mineral term pivots at 700 nm** (where it does
its work), the organic term keeps BING's 600 nm, and **both a 4-parameter
and a 3-parameter (fixed-flat mineral) variant get prototyped**.

```
Pow2      bb_nw(λ) = 10^B_min · (700/λ)^η_min + 10^B_org · (600/λ)^η_org
Pow2Flat  bb_nw(λ) = 10^B_min                 + 10^B_org · (600/λ)^η_org
```

- **Mineral term**: large amplitude, ~flat slope (η_min ≈ 0) — supplies
  the 0.1–0.4 m⁻¹ the red demands (Snyder 2008; Doxaran 2009;
  Neukermans 2012). With the 700 nm pivot, `B_min` *is* the mineral
  backscatter at 700 nm, which makes the prior and the fitted value
  directly comparable to the report's "required `b_b`" numbers.
- **Organic term**: smaller amplitude, steeper slope — preserves the
  blue behaviour that already fits open-ocean spectra.
- **`Pow2Flat`** is `Pow2` with `η_min ≡ 0`, i.e. literally a constant
  plus a power law (`Cst` + `Pow`). It is the 3-parameter arm of the
  "prototype both" answer and the degeneracy-safe fallback; note the
  mineral pivot is irrelevant when the term is flat, so the 700 nm
  choice only affects `Pow2`.

| model | nparam | pnames |
|---|---|---|
| `Pow2` | 4 | `['Bmin', 'eta_min', 'Borg', 'eta_org']` |
| `Pow2Flat` | 3 | `['Bmin', 'Borg', 'eta_org']` |

Proposed priors (round numbers are fine per JXP):

| param | flavor | range | rationale |
|---|---|---|---|
| `Bmin` | log_uniform | −6 … 5 | as `Bnw` today; amplitude at 700 nm |
| `eta_min` | uniform | −0.5 … 0.5 | flat/"white" mineral limit |
| `Borg` | log_uniform | −6 … 5 | as `Bnw` today; amplitude at 600 nm |
| `eta_org` | uniform | 0.5 … 2 | open-ocean particle slope |

**Keep the two η ranges disjoint** (mineral ≤ 0.5 ≤ organic). The two
terms are otherwise exchangeable, and a symmetric prior would give the
posterior a label-switching degeneracy that wrecks MCMC interpretation
and makes the chi-squared covariance meaningless.

Implementation sketch (follow the real base class, not the skill; note
the four deliberate details flagged by the audit — `prior_dicts=None`,
the priors-length guard, `log_params`, and the **non-zero** `eta_min`
seed):

```python
class bbNWPow2(bbNWModel):
    """Two-component particulate backscattering (mineral + organic).

    bb_nw(λ) = 10^Bmin·(700/λ)^eta_min + 10^Borg·(600/λ)^eta_org

    The near-flat mineral term supplies the large red-end backscatter
    of turbid, mineral-dominated water, which a single decreasing
    power law cannot produce without breaking the blue.  Its pivot is
    700 nm so that Bmin is the mineral backscatter in the red, where
    it is constrained.

    References
    ----------
    - Snyder, W. A. et al. (2008), Appl. Opt. 47, 666.
    - Twardowski, M. S. et al. (2001), JGR 106, 14129.
    - Doxaran, D. et al. (2009), Limnol. Oceanogr. 54, 1257.
    - Neukermans, G. et al. (2012), Limnol. Oceanogr. 57, 124.
    """
    name = 'Pow2'
    nparam = 4
    pnames = ['Bmin', 'eta_min', 'Borg', 'eta_org']
    # Which slots are log10 amplitudes (drives p0 and plot labels).
    log_params = [True, False, True, False]
    pivot = 600.       # organic/reference pivot; kept scalar because
                       # external scripts read models[1].pivot
    pivot_min = 700.   # mineral pivot (see docstring)

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        bbNWModel.__init__(self, wave, prior_dicts)
        # Guard the silent-corruption failure mode (landmine 3).
        assert self.priors is None or self.priors.nparam == self.nparam

    def init_guess(self, bb_nw:np.ndarray):
        # Split the observed bb_nw between the two components at their
        # own pivots.  Amplitudes stay LINEAR (the caller log10s the
        # log-flavored slots).  eta_min is seeded slightly OFF zero:
        # inference.py's multiplicative walker perturbation gives a
        # p0 of exactly 0.0 zero spread, which freezes that dimension
        # for the entire MCMC run (landmine 1).
        i700 = np.argmin(np.abs(self.wave - self.pivot_min))
        i600 = np.argmin(np.abs(self.wave - self.pivot))
        p0_bb = np.array([max(bb_nw[i700]/2., 1e-5), 0.05,
                          max(bb_nw[i600]/2., 1e-5), 1.])
        assert p0_bb.size == self.nparam
        return p0_bb
```

and the dispatch branch in `bbNWModel.eval_bbnw` — note it uses the
`wave` argument, **not** `self.wave`, so Raman's `eval_bb_ex` is correct:

```python
        elif self.name == 'Pow2':
            # Mineral (flat, red-pivoted) + organic (steep) power laws.
            # Slicing params[..., 0:2] / [..., 2:4] preserves
            # functions.powerlaw's (nsample, nwave) contract for 1-D
            # and 2-D input alike (verified).
            return (functions.powerlaw(wave, params[..., 0:2],
                                       pivot=self.pivot_min) +
                    functions.powerlaw(wave, params[..., 2:4],
                                       pivot=self.pivot))
        elif self.name == 'Pow2Flat':
            # Flat mineral term + organic power law (eta_min fixed 0).
            return (functions.constant(wave, params[..., 0:1]) +
                    functions.powerlaw(wave, params[..., 1:3],
                                       pivot=self.pivot))
```

plus `'Pow2': bbNWPow2` and `'Pow2Flat': bbNWPow2Flat` in
`init_model`'s `model_dict`, and `standard.expb_pow2()` /
`standard.expb_pow2flat()` factories carrying the `bpriors` above
explicitly (never rely on `set_standard_priors`' blanket default —
landmine 4).

### 3. `maxfev` through `chisq_fit.fit`

JXP-approved (option (a)): add the kwarg to BING rather than have
IOPtics call `curve_fit` itself.

```python
def fit(items:tuple, models:list, rt_dict:dict, bounds:tuple=None,
        maxfev:int=None):
    ...
    kwargs = {} if maxfev is None else dict(maxfev=maxfev)
    ans, cov = curve_fit(partial_func, None, Rrs, p0=params,
                         sigma=np.sqrt(varRrs), full_output=False,
                         bounds=bounds, **kwargs)
```

`maxfev` is the right spelling for **both** scipy branches: with finite
`bounds` `curve_fit` uses `least_squares`/'trf' and internally renames
`maxfev` → `max_nfev` (verified in scipy **1.18.0**,
`_minpack_py.py:1031-1033`; the branch is chosen at `:923-928`), while
the unbounded 'lm' branch passes it to `leastsq`. Both branches were
confirmed to accept `maxfev` and to raise `RuntimeError` on exhaustion,
so test 8 is sound. Document the default (`None` = scipy's default) and
that raising it changes only *whether* LM returns, not fit quality —
on GLORIA a ~40× bump moved convergence 12.5% → 37.5% with no
improvement in misfit.

### 4. `log_params` + honour it in plotting

Fallout from landmine 2: `bing/plotting.py:262` prints every parameter as
`10**param`, and the corner path (`plotting.py:449-456`,
`l23.py:519-524`) exponentiates and `\log_{10}`-labels every column. That
is already wrong for `Pow`'s `beta`; for a model that is half linear
exponents it mis-reports half the bb panel with no error.

Minimal, backward-compatible fix:

- declare `log_params` (a per-parameter boolean list) on the new bb
  classes — and, while there, on `bbNWPow`/`bbNWLee`/`bbNWGSM`/`bbNWCst`
  so `beta` stops being mis-printed;
- in plotting, read it defensively:
  `log_params = getattr(model, 'log_params', [True]*model.nparam)` —
  models that don't declare it keep today's behaviour exactly.

Only exponentiate/label the slots where `log_params` is `True`. Keep this
change confined to display code: **do not** wire `log_params` into the p0
log10 step in this pass (that is a behavioural change to every fit; see
Open Questions).

### 5. (Optional, ask first) refactor `eval_bbnw` to polymorphic dispatch

Adding a model currently means editing a string `if/elif` in the base
class. Cleaner: each subclass overrides `eval_bbnw(params, wave=None)`
and the base raises `NotImplementedError`. This touches all five
existing models, so treat it as a **separate, opt-in** step after
`Pow2` works and the tests are green — not bundled in.

✔ **Done 2026-07-29 (approved).** Implemented as a *template method*
rather than a plain override: the public `eval_bbnw(params, wave=None)`
stays on the base class and resolves the grid, then delegates to each
model's `_eval_bbnw(params, wave)`. Rationale in the log — having each
subclass re-resolve `wave` itself would recreate the exact bug fixed in
Prompt 4. Verified **bit-identical** output for all seven models.

## Tests

Create `bing/tests/test_bbnw.py` (confirmed: the module does **not**
exist yet; mirror `bing/tests/test_anw.py`, which is 73 lines, module-level
`wave = np.arange(350, 755, 5.)`, four plain test functions, no fixtures).
⚠ `test_anw.py` constructs classes directly (`bing_anw.aNWExp(wave)`);
that only works because the a-models default `prior_dicts=None`. Use
`init_model(...)` here, or give the new bb classes the same default (both,
ideally — see the sketch above). Run every test for **both** `Pow2` and
`Pow2Flat`, parameterized where practical:

1. **Construction / registry.** `init_model('Pow2', wave)` returns a
   model with `nparam == 4`, `pnames` as specified, and
   `len(pnames) == nparam` (the base-class assert); `Pow2Flat` gives 3.
   Also assert `init_model('Pow2', wave)` leaves `model.priors is None`
   without raising (the factory's `prior_dicts` default).
2. **Shape contract.** `eval_bbnw` returns `(1, nwave)` for a 1-D
   parameter vector and `(nsample, nwave)` for a chain-shaped array;
   all values finite and > 0.
3. **`wave` kwarg honoured.** `eval_bbnw(params, wave=model.wave_ex)`
   returns values on the excitation grid, and `eval_bb_ex(params)`
   equals `bb_w_ex + eval_bbnw(params, wave=wave_ex)`.
4. **Reduces to `Pow`.** With the *mineral* amplitude driven to a
   negligible value (e.g. `Bmin = -12`), `Pow2` matches `Pow` evaluated
   at `(Borg, eta_org)` to tight tolerance — same pivot (600 nm), so
   this is an exact comparison. This is the regression guard that the
   new branch is a strict generalization. (The mirror check — killing
   the organic term — must compare against `Pow` with `pivot=700`, so
   do it against a hand-written `10**Bmin*(700/wave)**eta_min` instead.)
5. **Turbid shape is reachable.** A mineral-dominated parameter set
   produces `bb_nw` that is flat-or-rising from 400 → 750 nm and of
   order 0.1–0.4 m⁻¹ in the red — the thing `Pow` with `beta ∈ [0,2]`
   cannot do. Assert `bb_nw(700) / bb_nw(440) > 1` for a negative
   `eta_min`.
6. **Priors attach and bound the fit.** `init_model('Pow2', wave,
   prior_dicts)` gives `model.priors.priors[1].flavor == 'uniform'`, and
   every prior exposes `pmin`/`pmax` (the chi-squared bounds contract).
7. **p0 round-trip.** `init_guess` returns 4 linear values; log10-ing
   the log-flavored slots (as `l23.py` does) and evaluating reproduces
   the input `bb_nw` at the pivot to within a factor of ~2.
8. **`maxfev` plumbing.** `chisq_fit.fit(..., maxfev=<tiny>)` raises
   `RuntimeError` (budget exceeded) while the same call with a generous
   `maxfev` succeeds — proves the kwarg reaches scipy in the bounded
   branch. Keep this on a cheap synthetic spectrum, not L23.
9. **End-to-end, guarded.** One `expb_pow2` chi-squared fit through the
   existing L23 machinery (mirror `test_l23_fitting.py`), asserting
   finite `Rrs_model` and that chi²_ν is **no worse** than `expb_pow` on
   the same clear spectrum. Skip if the L23 data tree is absent.

Three more that exist purely because of the audit:

10. **`init_guess` never seeds an exponent at exactly 0.** Assert
    `init_guess(...)[1] != 0.0` for `Pow2`. This looks pedantic; it is
    guarding landmine 1 (a zero seed silently freezes that MCMC
    dimension for the entire run). Add the reason as a comment so nobody
    "simplifies" the seed back to 0.
11. **Priors-length guard fires.** Constructing `Pow2` with a 2-element
    prior list raises (rather than silently mis-log10-ing p0 later).
12. **`log_params` matches the priors.** For each `standard.expb_pow2*`
    factory, assert `[p['flavor'].startswith('log') for p in bpriors]`
    equals the model's `log_params`. This is the one cheap test that
    catches contract 1 breaking in either direction.

Also confirm the existing suite still passes — the new models are purely
additive, so `test_anw.py`, `test_l23_fitting.py`, `test_evaluate.py`,
`test_io.py`, `test_chl_fl.py` and `test_raman.py` should be untouched.

## Benchmark (dev/)

Add `dev/turbid_bbp/` with a script (plus figures) that answers "does
this actually help?":

- **Synthetic recovery.** Generate turbid-like Rrs from a *known* `Pow2`
  truth via `bing.evaluate.calc_Rrs_from_models`, add noise, and fit
  with `Pow`, `PowFlex`, and `Pow2`. Report recovered parameters and
  relative Rrs misfit per model. Expected: `Pow`/`PowFlex` plateau,
  `Pow2` recovers.
  ⚠ **Make the truth genuinely two-component**, i.e. give the mineral
  and organic terms *comparable* amplitudes so the organic term
  dominates the blue and the mineral term the red, and the log-log slope
  of `bb_nw` changes across the band. Prompt 3 hit this: a truth with
  `Borg` 50× below `Bmin` is effectively a *single* rising power law, so
  `PowFlex` recovers it to ~4e-5 and the comparison proves nothing. With
  balanced amplitudes, `Pow`/`PowFlex` plateau at ~1e-3 while `Pow2`
  recovers exactly (both cases are quantified in the Prompt-3 log).
- **No regression.** Fit a handful of L23 (open-ocean) spectra with
  `expb_pow` vs `expb_pow2` and show chi²_ν and IOP accuracy are not
  degraded — a 4-parameter bb model must not buy turbid performance at
  the cost of the clear-water case.
- **Figures.** `bb_nw(λ)` for each model over 400–750 nm (log y), and
  observed-vs-model Rrs panels for a clear and a turbid spectrum. Use
  matplotlib, `Agg` backend, and save PNGs next to the script.

Identifiability is the risk to watch: 4 bb + 3 a parameters against
tightly-quoted in-situ noise may be degenerate under chi-squared. If it
is, report it — the mitigations on the table are MCMC instead of LM, an
inflated noise floor (`σ = max(σ_measured, f·|Rrs|)`, already approved
on the IOPtics side and always to be labelled as inflated), and/or
fixing `eta_min` to cut a parameter.

## Docs

- `docs/models.rst` — add `Pow2` to the implemented-models section with
  its equation, parameters, priors, and the turbid-water rationale;
  mention `PowFlex` as a prior variant of `Pow`.
- `docs/parameters.rst` — document `standard.expb_pow2()` and
  `standard.expb_powflex()` alongside `expb_pow`/`giop`/`gsm`.
- `docs/fitting.rst` — document the new `maxfev` kwarg on
  `chisq_fit.fit`, including that it affects convergence only.
- The **References** section at the bottom of `docs/models.rst` (there
  is no `references.rst` in `docs/`, despite the `index.rst` toctree
  entry — flag that separately if you want it created) — add the
  turbid-backscatter citations listed under "Why" above, with DOIs.
- `CLAUDE.md` — add `bbNWPow2` to the "Available Backscattering Models"
  table, add the new combos to "Standard Model Combinations", and fix
  the stale "New Backscattering Model Template" (it shows
  `super().__init__(wave)` and a subclass-level `eval_bbnw`, neither of
  which matches the code).
- `.claude/skills/add-bbnw-model/SKILL.md` — same correction, plus the
  p0-is-linear and bounds-from-`pmin`/`pmax` contracts, which the skill
  omits today.

## Downstream (IOPtics) — for context, not work in this repo

IOPtics needs **no core change**: `AlgorithmSpec.bbnw_model` is a plain
string BING resolves, and `AlgorithmSpec.from_standard(name)` just calls
`bing.parameters.standard.<name>()`. So each new model is one line:

```python
registry.register(AlgorithmSpec.from_standard('expb_pow2',
                                              label='ExpB_Pow2'))
```

Two consequences worth knowing while designing the models here:

- **These must land on `bing` `main`.** IOPtics' CI installs `bing` from
  git, so anything left in a local working tree is invisible there
  (same follow-up as `Rrs_to_rrs` / `correct_atmosphere` /
  `load_gloria`).
- **IOPtics silently drops the new shape parameters from the results
  table.** ✎ Corrected citation: the hard-coded name list is
  `ioptics/evaluate.py:79` (`for key in ('Sdg', 'beta')`), not
  `io.py:174` (which merely reads `med_sig('beta')` from those scalars).
  `eta_min`/`eta_org` therefore never reach `results_scalar.parquet`, and
  the `beta`/`sig_beta`/`beta_truth` columns come out all-NaN for the new
  models. Nothing is lost on disk — the etas *are* in `result.params`
  (`evaluate.py:71-73`, name-driven) and in the saved chain — but any
  cross-algorithm slope comparison shows the new models as blank. An
  IOPtics-side fix (widen the tuple, or per-model "reportable shape
  params"), not a BING one.
- **`bing`-side equivalent:** `l23.py:842` keys `extras` on the literal
  `'beta'`, so `process_all`'s aggregated CSV loses those columns
  entirely (`KeyError` risk, not NaN) — landmine 5.
- **Cosmetic config to remember later:** IOPtics'
  `report/standard.py:165-166` hard-codes `"(expb_pow, k=5)"` /
  `"(giop, k=3)"` prose and `metrics.compute`'s default
  `dbic_pair=('expb_pow','giop')` (`metrics.py:635`) won't include the
  new models.

## Prompts

1. Read this doc. **No code yet:** confirm the code map against the
   current source, flag anything I got wrong, and post your
   implementation plan plus questions in Open Questions / Logs.  Also note my answers to the open questions below.  Use Opus
   ✔ **Done 2026-07-26** — code map corrected in place (✎ marks), audit
   landmines added, plan + new questions below. Prompts renumbered from
   here (a `log_params`/plotting step was added as 5).
2. Re-read this doc. See my answers to the open questions below. Implement **item 1 (`PowFlex`)** —
   `standard.expb_powflex()` — plus its test. Log your work.
   ✔ **Done 2026-07-27** — `standard.expb_powflex()` + `test_bbnw.py`
   (7 tests). Steps renumbered below: your approved Q8 and Q6 fixes are
   now steps **4** and **7**, so each behavioural change lands with its
   own test and log entry instead of riding along with a model commit.
3. Re-read this doc. Implement **item 2**: the `bbNWPow2` and
   `bbNWPow2Flat` classes, their `init_model` entries, the two
   `eval_bbnw` branches, `standard.expb_pow2()` /
   `standard.expb_pow2flat()`, and the `test_bbnw.py` tests 1–7 and
   10–12. Per **Q7**, put the priors-length assert in the **base class**
   `bbNWModel.__init__` (not per-model as the sketch shows), and add a
   test that a wrong-length `bpriors` raises. Log your work.
   ✔ **Done 2026-07-27** — both models + factories + 25 tests. Q7 landed
   as `bbNWModel.check_priors()`, also called from `set_standard_priors`
   (the path the real fitters use); see the log.
4. Re-read this doc. **Q8 fix:** make `Lee`, `GSM` and `Every` honour the
   `wave` kwarg in `eval_bbnw`, so `eval_bb_ex` stops adding `bb_w_ex`
   (excitation grid) to `bb_nw` evaluated on the emission grid. Test that
   `eval_bbnw(p, wave=wave_ex) != eval_bbnw(p)` for each, and that
   `eval_bb_ex` equals `bb_w_ex + eval_bbnw(p, wave=wave_ex)`. Note this
   *changes* Raman-enabled `giop`/`gsm` results — quantify the shift in
   the log. Log your work.
   ✔ **Done 2026-07-27** — `eval_basis_func(wave)` for Lee/GSM,
   log-log interpolation for `Every`, 7 tests. `bb_ex` was off by
   16–22%, but Raman-corrected Rrs moves only 0.004–0.02% and retrieved
   parameters ~1e-4 dex, so **no past result needs redoing**; see the log.
5. Re-read this doc. Implement **item 3 (`maxfev`)** in
   `chisq_fit.fit` plus test 8. Log your work.
   ✔ **Done 2026-07-27** — `maxfev` kwarg (forwarded only when set, so
   scipy's default is untouched) + new `bing/tests/test_chisq_fit.py`
   (5 tests, both scipy back ends). The `l23.py` wrapper does **not**
   expose it yet — one line if you want it; see the log.
6. Re-read this doc. Implement **item 4** — `log_params` on the bb
   classes and the defensive `getattr` in `plotting.py` (display only;
   do **not** touch the p0 log10 step). Log your work.
   ✔ **Done 2026-07-27** — `log_params` on the bb classes **and** the
   a-models with linear params (`Sdg` was mis-printed in every standard
   combo), `plotting.log_param_mask`, both display sites fixed, 19 tests
   incl. a flavor-vs-`log_params` cross-check over all 8 combos. p0 step
   untouched. `Chase2017` deliberately left undeclared; see the log.
7. Re-read this doc. **Q6 fix — now a prerequisite for `Pow2` MCMC, not a
   nicety.** Prompt 3 demonstrated it live: a `Pow2` χ² fit legitimately
   converges to `eta_min ≈ 0` for a flat mineral term, that value becomes
   the MCMC seed in the standard workflow, and the resulting chain had
   `eta_min` spread **exactly 0.0** over 400 steps while all six other
   dimensions moved. Do this before the step-9 benchmark.
   Replace `inference.py`'s multiplicative
   walker perturbation with an additive, floored one plus a clip into the
   prior bounds, so a parameter seeded at 0 is no longer frozen for the
   whole run. Test it (assert non-zero inter-walker spread in **every**
   dimension for a p0 containing 0, and that all walkers start in-prior),
   and report how it changes an existing L23 fit — this touches every
   MCMC run. Log your work.
   ✔ **Done 2026-07-28** — `inference.init_walkers` / `prior_bounds`,
   `run_emcee` gains `perturb_frac`/`perturb_floor`, 8 tests. L23 fits
   change only within MCMC run-to-run scatter (verified with two seeds
   per method; ACT unchanged). Default RNG deliberately stays the legacy
   global so `batch_fit(seed=)` stays reproducible; see the log.
8. Re-read this doc. Add the end-to-end guarded test (test 9) and run
   the full suite. Generate one or more Jupyter notebooks to visualize the results. Log your work.
   ✔ **Done 2026-07-28** — test 9 (3 tests, guarded) shows all models give
   *identical* raw chi² on clear L23 idx 170; full suite 162 passed,
   2 skipped. Two executed notebooks under `nb/TurbidWaters/`:
   `turbid_bbnw_models.ipynb` and `walker_init_fix.ipynb`.
9. Re-read this doc. Build the **Benchmark** under `dev/turbid_bbp/`,
   examine the outputs, and make a recommendation on identifiability
   (MCMC vs inflated noise vs `Pow2Flat`'s fixed `eta_min`). Per **Q10**,
   bump `nwalkers` to 32–64 **locally in the benchmark**, not in
   `init_mcmc`. Log your work.
   ✔ **Done 2026-07-28** — `dev/turbid_bbp/` + 4 figures.
   **Recommendation: `Pow2Flat` + MCMC** (χ² is degenerate, cond ~1e7;
   the inflated floor makes recovery *worse*, 5.7σ bias). **But** a single
   power law is statistically adequate against a two-component truth
   unless the noise is 4–11× tighter than GLORIA's — so `Pow2` probably
   won't fix GLORIA. `maxfev` is **required** (default budget: `Pow2` 5/8
   on clear L23; with maxfev 8/8). No L23 regression. See the log.
10. Re-read this doc. Do the **Docs** pass, including the CLAUDE.md and
    skill corrections. Log your work.
    ✔ **Done 2026-07-28** — `models.rst` (bb + a model lists corrected,
    turbid section, `log_params`, 17 DOI'd references), `parameters.rst`,
    `fitting.rst` (`maxfev` + walker init; least-squares API was wrong),
    `CLAUDE.md`, the skill. **sphinx installed into `ocean14` at your
    request** (9.1.0 + rtd-theme 3.1.0 + docutils 0.22.4) and added to
    `docs/requirements.txt` + a `[docs]` extra in `setup.py`;
    `sphinx-build` succeeds with **0 warnings in the edited files**.
11. Re-read this doc. *Only if I approve it:* refactor `eval_bbnw` to
    polymorphic per-subclass dispatch (optional item 5). Log your work.
    ✔ **Done 2026-07-29 (you approved)** — the `if/elif` is gone; each
    model implements `_eval_bbnw(params, wave)` and the base resolves the
    grid + raises `NotImplementedError`. Bit-identical output for all 7
    models; 16 new tests; suite 178 passed, 2 skipped. `CLAUDE.md`, the
    skill and the code map updated (they documented the old mechanism).
12. Please generate the hooks to run the tests as CI on GitHub.  I will then turn them on on GitHub.  Log your work.
    ✔ **Done 2026-07-29** — `.github/workflows/tests.yml` (tests matrix
    3.11–3.13 + docs build) and `bing/tests/conftest.py`, which turns the
    missing-L23-dataset failure into skips so CI is green with honest
    skips (70 passed / 110 skipped without data; 178 passed / 2 skipped
    with). Validated in a clean venv. **⚠ Found an upstream ocpy bug:
    `ocpy/hydrolight/` lacks `__init__.py`, so a pip-installed ocpy has no
    `ocpy.hydrolight`** — CI works around it with an editable clone; the
    one-line upstream fix is noted in the workflow. See the log.
13. **docs** Can you clean up the doc warnings, i.e. make changes to remove them all.  Log your work.
    ✔ **Done 2026-07-30** — 45 warnings + 12 errors → **zero**;
    `sphinx -W` now succeeds in both `ocean14` and the clean CI docs env,
    and `-W` is enabled in the workflow. Six autodoc API pages plus
    `contributing`/`changelog`/`references`/`examples` written, three RST
    structural bugs fixed, 14 pseudo-citations converted, and 8 docstring
    bugs fixed (four of them mine). `tutorials/index.rst` was rewritten:
    it had described a YouTube channel and APIs that do not exist. See the
    log.

## Open Questions

> Questions for JXP. Pose them; do not self-answer. Decisions and
> rationale go in the Logs.

- **Priors: literature-anchored or generic?** The table above uses
  round numbers (`eta_min ∈ [-0.5, 0.5]`, `eta_org ∈ [0.5, 2]`). Should
  they instead be anchored to specific published ranges (Twardowski 2001
  refractive-index/backscattering-ratio ranges; Snyder 2008 / Gordon
  2009 slopes)?
>A. It is ok to use round numbers for the priors for now
- **Pivot wavelength.** Keep BING's 600 nm for both components, or pivot
  the mineral term in the red (e.g. 700 nm) where it does its work?
>A. Let's use 700nm for the mineral term.
- **Fix `eta_min`?** Fixing it (e.g. at 0, i.e. a spectrally flat
  mineral term reusing the `Cst` form) drops `Pow2` to 3 parameters and
  removes most of the degeneracy risk. Prototype both, or start fixed?
>A. Prototype both.
- **A published turbid scheme as a third model?** e.g. a
  turbidity-dependent slope, or reusing the `Lee` dynamic-`Y` machinery
  with a turbid-water `Y` estimator. Worth adding now or later?
>A. We will consider this later.
- **Gordon coefficients in very turbid water.** The GLORIA report raises
  whether the fixed Gordon `G` coefficients are being pushed outside
  their validity range at these backscatter levels. Out of scope for
  this pass, but do you want it investigated separately?
>A. We won't worry about this for now.

### New questions from the Prompt-1 audit (2026-07-26)

These come out of verifying the code map; each is a **pre-existing BING
behaviour** that the new models expose. I have not changed any of them.

- **Q6. Fix the MCMC walker initialization?** `inference.py:321` perturbs
  p0 *multiplicatively* (`p0 += p0*uniform(-1e-2, 1e-2)`), so a parameter
  seeded at exactly 0 gets zero spread across walkers and — because
  emcee's stretch move proposes along walker-to-walker vectors — **never
  moves for the entire run**, silently, with a healthy acceptance
  fraction and a zero-width credible interval. `fit_one` also passes
  `skip_check=True` (`:231`), which disables emcee's own guard. I can
  work around it in this pass by seeding `eta_min = 0.05` instead of 0
  (that is what the sketch does). Do you also want the underlying fix
  (additive, floored perturbation + clip into the priors)? It would
  change the initialization of **every** existing MCMC fit, which is why
  I am not doing it unasked.
>A. Yes, this sounds like a good fix.  But please be sure to test it.

- **Q7. Add the priors-length assert to the base class or just the new
  models?** Nothing validates `len(bpriors) == model.nparam`; on the
  chi-squared path a mismatch silently log10s the wrong p0 slots
  (`l23.py:386-390`). I checked all four `standard.py` combos and they
  are consistent, so a one-line assert in `bbNWModel.__init__` would be
  safe — but it *is* a core edit that could trip someone's bespoke
  script. New models only, or the base class?
>A. Apply to the base class.

- **Q8. `Lee`/`GSM`/`Every` ignore the `wave` kwarg — fix now, later, or
  file it?** Consequence: `eval_bb_ex` (`bbnw.py:300`) adds `bb_w_ex` on
  the *excitation* grid to `bb_nw` evaluated on the *emission* grid, so
  **Raman fits using `giop`/`gsm` are quietly slightly wrong today**. It
  is silent only because the two grids have equal length. Unrelated to
  turbid water, but I found it while checking the contract the new models
  must satisfy, and `giop` is in the IOPtics leaderboard. Want a separate
  fix + test, or shall I just log it?
>A. Fix now + test.

- **Q9. Naming: `Pow2Flat` for the 3-parameter arm?** It is literally
  `Cst` + `Pow` (flat mineral + organic power law). Alternatives:
  `Pow2Fix`, `CstPow`. Say if you prefer one; it becomes a public model
  name and a `standard.expb_pow2flat()` factory.
>A. Yes that name is fine.

- **Q10. Walkers for a 7-parameter fit.** `inference.py:160` gives
  `nwalkers = max(16, 2*ndim)` → **16** for ExpBricaud(3) + `Pow2`(4).
  That is thin for a model with a fresh amplitude/slope degeneracy. Bump
  it for the turbid benchmark (e.g. 32–64), and if so, only locally in
  the benchmark or in `init_mcmc`?
>A. Bump it for the turbid benchmark (e.g. 32–64)

## Logging

The "Logs" section will record Claude's work. Please use the following
format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-07-30 (Prompt 13: docs warnings — 57 to zero, `-W` now enforced)

`sphinx-build` was reporting **45 warnings + 12 errors**. It now builds
clean, and **`sphinx -W` (warnings as errors) succeeds** — verified both
in `ocean14` and in a clean venv matching the CI docs job. I enabled `-W`
in the workflow, which is what stops the count creeping back up. Suite
unchanged: 178 passed, 2 skipped.

#### What was wrong, and what I did

**Structural RST (12 errors).**
- `raman.rst`: 9 × "Inconsistent title style: skip from level 3 to 5".
  The nine function entries under *Raman Rrs Functions* (a level-2
  heading) used the level-4 adornment, skipping level 3. Re-levelled
  `^` → `~`.
- `radiative_transfer.rst`: 2 × "Undefined substitution referenced: G2".
  The text wrote `|G2| < 1` meaning absolute value; RST reads `|…|` as a
  substitution. Now inline literals.
- `save_load.rst`: "Malformed table". The `.npz` table's first column was
  18 characters wide but two rows (``` ``a_lo`` / ``a_hi`` ```,
  ``` ``bb_lo`` / ``bb_hi`` ```) needed 19 and 21, so they spilled across
  the column boundary. Table rewritten at width 21.

**Citations (14 warnings).** `chlorophyll_fluorescence.rst` and
`raman.rst` each carried a `.. [Label]` bibliography, giving 12
"not referenced" warnings plus 2 "duplicate citation" (both files defined
`SathyendranathPlatt1998` and `OOWB`). I checked first: **no `[Label]_`
reference exists anywhere in the docs**, so these are bibliographies
written in citation syntax rather than actual citations. Converted to
bullet lists — same text, no citation semantics.

**Duplicate object descriptions (4).** `calc_Rrs_fluorescence`,
`get_emission_spectrum`, `summary_at_wavelength` and
`calc_raman_correction_factor` are each hand-documented on two pages;
added `:no-index:` to the secondary copy.

**Config (2).** `html_static_path` pointed at a non-existent `_static/`
(created it with a `.gitkeep`), and `display_version` is not a valid
option in sphinx-rtd-theme 3.x (removed).

**Missing documents (25).** Three toctrees referenced 17 documents that
were never written, which also produced 8 broken `:doc:` links. Handled
case by case rather than uniformly:

- **Six API pages created** (`models_api`, `fitting_api`,
  `parameters_api`, `evaluation_api`, `visualization_api`,
  `utilities_api`) using `automodule`, so they are generated from the real
  docstrings rather than hand-maintained. `core` was dropped from the
  toctree — it had no content distinct from the prose already on
  `api/index.rst`. `bing.fitting.l23` is deliberately excluded, with a
  note saying why: it imports `correct_atmosphere`, which the docs build
  does not install. `bing.rt.*` and `bing.io` are also excluded because
  they are hand-documented elsewhere and autodoc would duplicate them.
- **Four top-level pages written**: `contributing` (environment, running
  the tests, the data-skip behaviour, conventions), `changelog`
  (organised by theme, since there are no releases yet, recording the
  turbid-water work and the fitting corrections), `references`
  (the papers BING implements and where each is used, with DOIs), and
  `examples` (a map of `nb/`, `dev/` and `papers/`).
- **`tutorials/index.rst` rewritten** — see the honesty note below.
- `api/index.rst`'s "Module Structure" block listed files that do not
  exist (`models/base.py`, `bing/utils.py`, `rt.py` as a module) and
  omitted `rt/`, `io.py`, `noise.py`, `preproc.py`, `stats.py`. Corrected
  against the actual tree.

**Then autodoc introduced 62 new warnings**, which took two rounds:
- 52 × "duplicate object description of `<Model>.name`, `.nparam`, …".
  Cause: `anw.py`/`bbnw.py` document attributes twice — once via PEP-224
  attribute docstrings that autodoc picks up, once via the class
  docstring's `Attributes:` section that napoleon turns into
  `py:attribute` directives. Dropping `:undoc-members:` was not enough;
  the fix is `napoleon_use_ivar = True` in `conf.py`, which renders those
  sections as `:ivar:` fields instead. One line, no source churn.
- 2 × from `models.rst`'s `.. py:class:: bing.models.anw` blocks, which
  are prose containers rather than real class docs → `:no-index:`.
- 8 × genuine docstring bugs, and **four of them were mine** from earlier
  prompts: `init_walkers` and `run_emcee` wrote `|p0|` for absolute value
  (substitution reference again — the same trap as `|G2|`) and a bullet
  list without a preceding blank line. Fixed those, plus four
  pre-existing ones in `anw.py` (indented formulas after `:` instead of
  `::`, and over-indented continuation lines in
  `aNWExpBricaudFree.set_aph`) and `functions.py` (`gen_basis` mixed two
  docstring styles).

#### One honesty note

`tutorials/index.rst` did not just reference six missing tutorials — much
of its content described things that do not exist: a YouTube channel with
"video walkthroughs", a `notebooks/` directory (it is `nb/`), a
`github.com/yourusername/bing` clone URL, per-tutorial FAQ sections, and
three code examples calling APIs with the wrong signatures
(`models[0].eval(...)`, `chisq_fit.fit(models, wavelengths, …)`,
`import fitting as m_fitting`). I **deleted** that rather than only
silencing the warnings, and replaced it with a reading order through the
pages that do exist, pointers to the executed notebooks, the real data
sources, and an explicit "still to be written" section. Flagging it
because it is a deletion of content, not a fix.

#### Verification

| build | result |
|---|---|
| `ocean14` | **build succeeded** — 0 warnings (was 45 + 12 errors) |
| `ocean14`, `sphinx -W` | **build succeeded** |
| clean venv (CI docs env, no `correct_atmosphere`), `sphinx -W` | **build succeeded** |
| `pytest bing/tests` | 178 passed, 2 skipped (docstring-only source edits) |

The workflow's docs job now runs `python -m sphinx -W -b html`, replacing
the comment that said to turn `-W` on once the warnings were cleared.

### 2026-07-29 (Prompt 12: GitHub Actions CI hooks)

Added `.github/workflows/tests.yml` (there was no `.github/` in bing) and
`bing/tests/conftest.py`. **Ready for you to enable on GitHub.** Every
step was validated in a *clean virtualenv*, not just written — which is
how the one real blocker surfaced (see below).

**Style follows your existing repos**, not something invented here:
`ocpy/.github/workflows/tests.yml` (lightweight curated deps,
`pip install -e . --no-deps`, an explicitly limited scope with the reason
in a comment) and `correct-atmosphere/.github/workflows/ci.yml`
(`concurrency` cancel-in-progress, pip caching, a matrix).

#### The central problem: the suite needs a dataset CI cannot have

Not a minor subset. `bbNWModel.init_bbw` loads `Hydrolight400.nc` for
pure-water backscattering, so **constructing any backscattering model
requires the L23 data**, which ocpy locates via `$OS_COLOR`. Measured
with `env -u OS_COLOR`: **74 failed, 31 errors**, 70 passed.

Rather than decorate ~105 tests with a marker that future tests would
forget, `bing/tests/conftest.py`:

- converts *that one specific failure* — an `OSError` whose message names
  `Hydrolight` — into a **skip**, in fixtures or test bodies alike, via a
  narrow `pytest_runtest_makereport` wrapper. Everything else still fails
  normally, so a real error stays a real error;
- drops from collection the three modules that cannot even be *imported*
  without `correct_atmosphere` (`test_evaluate`, `test_io`,
  `test_l23_fitting`) — an ImportError can't be turned into a skip;
- exports `needs_l23` for tests that would rather declare it explicitly;
- documents all of the above at the top, including where the data comes
  from (Dryad doi:10.6076/D1630T, ~17 MB per file).

Verified in all three states:

| condition | result |
|---|---|
| ocean14, data present (unchanged baseline) | **178 passed, 2 skipped** |
| ocean14, `env -u OS_COLOR` | **70 passed, 110 skipped, 0 failed** |
| clean venv, no data, no `correct_atmosphere` | **62 passed, 80 skipped** in 17 s |

(62+80 < 70+110 because the venv lacks `correct_atmosphere`, so
`collect_ignore` drops those three modules' 38 tests entirely.)

#### ⚠ The blocker: `ocpy.hydrolight` is missing from a pip-installed ocpy

I built a throwaway venv to test the dependency list rather than trust
the YAML, and the first run failed at collection:

```
ocpy/water/scattering.py:4: from ocpy.hydrolight import loisel23
E   ModuleNotFoundError: No module named 'ocpy.hydrolight'
```

**Root cause: `ocpy/hydrolight/` has no `__init__.py`**, so setuptools'
`find_packages()` omits it and `pip install git+.../ocpy.git` yields an
ocpy without that subpackage. It is on `origin/main` and imports fine on
your machine *only* because an editable install leaves the source tree on
`sys.path`, where Python treats it as a namespace package. This affects
anyone pip-installing ocpy, not just CI.

The workflow works around it by cloning ocpy and installing it
**editable** (`git clone --depth 1 … && pip install --no-deps -e ../ocpy`),
verified to make `ocpy.hydrolight` importable. **The proper fix is one
line upstream** — add `ocpy/ocpy/hydrolight/__init__.py` and commit —
after which CI can revert to the one-liner `pip install --no-deps
git+https://github.com/ocean-colour/ocpy.git`. There is a comment in the
workflow saying exactly that, so it does not become mystery scaffolding.

#### The workflow

- **`tests`** job, matrix Python **3.11 / 3.12 / 3.13** (setup.py requires
  ≥3.11 and the code uses `X | Y` annotations), ubuntu-latest.
  Installs a curated stack **derived from the package's actual imports**
  (I enumerated them: numpy, scipy, pandas, matplotlib, xarray,
  h5netcdf/h5py, emcee, corner, tqdm, ipython, plus scikit-learn for
  ocpy) then `pip install -e . --no-deps`. The `--no-deps` matters:
  bing's `install_requires` pulls healpy, umap-learn, llvmlite, boto3 and
  `timm==0.3.2`, none of which the tests need and which will not resolve
  cleanly on current Pythons. Runs `pytest bing/tests -v -ra` with
  `MPLBACKEND=Agg`.
- **`docs`** job: builds the Sphinx HTML and uploads it as an artifact.
  It also installs bing and ocpy, because `docs/api/io_api.rst` uses
  `autofunction` and would otherwise silently lose those pages.
- A **commented-out `full-tests` job** at the bottom with the recipe for
  running the skipped tests: cache `Hydrolight400.nc` and point
  `$OS_COLOR` at its parent. Left commented because there is no stable
  direct-download URL for the Dryad file — dropping a copy in a bucket
  you control would make it live.
- `concurrency` with cancel-in-progress, and `cache: pip`.

Everything above was run verbatim in the clean venv: the dependency
install, both `python -c` diagnostic lines (the backslash-continued one
included — it is easy to get an IndentationError there), the pytest
invocation, and the Sphinx build.

#### Deliberate choices, and two things for you to decide

- **No `-W` on the Sphinx build.** The docs carry ~57 pre-existing
  warnings (missing `api/*.rst` stubs, the malformed table in
  `save_load.rst`, `raman.rst` title levels). Turning `-W` on now would
  make CI red immediately; the comment says to enable it once those are
  cleaned up so regressions start failing.
- **No lint / mypy / coverage jobs.** `correct-atmosphere` has them, but
  bing has no `pyproject.toml`, no type annotations to speak of, and
  those jobs there are all `|| true` — i.e. decorative. Say the word and
  I will add real ones.
- **Triggers are `main`/`master`/`develop` + pull requests**, so pushes to
  `turbid_bbp` will *not* fire; you would see CI on a PR. Want the
  feature branch added?
- The matrix is ubuntu-only, unlike `correct-atmosphere`'s three-OS
  matrix. bing's dependency stack (h5netcdf, healpy-adjacent ocpy) is
  fiddlier on Windows and the tests are pure numerics — happy to add
  macOS if you want the coverage.

### 2026-07-29 (Prompt 11: polymorphic `eval_bbnw` dispatch — approved)

Refactored the string `if/elif` out of `bbNWModel.eval_bbnw`. One source
file plus tests and the three docs that described the old mechanism.
**16 new tests; full suite 178 passed, 2 skipped.** Output verified
**bit-identical** for every model.

**One design decision I made differently from the sketch, deliberately.**
The plan said "each subclass overrides `eval_bbnw(params, wave=None)` and
the base raises `NotImplementedError`". Implemented that way, each of the
seven models would have to repeat `wave = self.wave if wave is None else
wave` — which is *precisely* the line whose absence caused the Prompt-4
bug, where `Lee`/`GSM`/`Every` quietly evaluated on `self.wave` and made
`eval_bb_ex` mix the Raman emission and excitation grids. Recreating that
opportunity seven times over seemed like the wrong trade for a refactor
whose whole point is safety.

So it is a **template method** instead:

- the public `eval_bbnw(params, wave=None)` stays on the base class,
  resolves `wave=None → self.wave`, and delegates;
- each model implements **`_eval_bbnw(params, wave)`**, where `wave` is
  guaranteed non-None;
- the base `_eval_bbnw` raises `NotImplementedError` naming the class.

The public signature and behaviour are unchanged, the `if/elif` is gone,
and adding a model is now "write a method on your class + one
`init_model` entry" — the goal of the item. Grid resolution lives in
exactly one place and cannot be forgotten. Say the word if you'd rather
have the literal form.

**What moved where.** `Cst` → `functions.constant`; `Pow` →
`functions.powerlaw`; `Pow2` → the two-pivot sum; `Pow2Flat` → constant +
power law; `Every` → delegates to its existing `eval_channels`;
`Lee`/`GSM` → a one-line delegation to a new base-class helper
`_eval_basis_bbnw`, since their bodies were identical (amplitude ×
`eval_basis_func(wave)`) and only the basis differs. That keeps the
class hierarchy flat — no new intermediate class — while having the
shared logic written once.

**Verification.** Before trusting the tests I checked equivalence
directly: reimplemented the old dispatch verbatim in a scratch script and
compared against the refactored code for all **7 models × 2 grids
(native and Raman excitation) × 1-D and chain-shaped params** — 28
comparisons, all `atol=0, rtol=0` **bit-identical**, with `eval_bb` and
`eval_bb_ex` consistency asserted alongside. I also confirmed nothing
outside `bbnw.py` subclasses these models or calls the private path:
every caller in `bing/`, `papers/`, `dev/` and IOPtics uses the public
`eval_bbnw`.

**New tests** (16, in `test_bbnw.py`):
- `test_every_model_implements_eval` — parameterised over all seven
  models, asserts each *actually overrides* `_eval_bbnw` rather than
  inheriting the base. This is the guard that matters now: with dispatch
  gone, a model that forgets its method would previously have hit a
  `ValueError: Unknown model`, and now must fail on the base's
  `NotImplementedError` — so the test proves the safety net is wired.
- `test_base_class_refuses_to_evaluate` — defines a throwaway subclass
  with no `_eval_bbnw` and asserts `NotImplementedError`.
- `test_basis_models_share_one_implementation` — `Lee`/`GSM` both expose
  `_eval_basis_bbnw`, and it honours the requested grid.
- `test_public_eval_resolves_the_grid` — for every model, `eval_bbnw(p)`,
  `eval_bbnw(p, wave=self.wave)` and `_eval_bbnw(p, self.wave)` agree, so
  the delegation is transparent.

**Docs corrected** — the Prompt-10 pass had just documented the *old*
mechanism, so leaving them would have been worse than not having written
them:
- `CLAUDE.md` — "New Backscattering Model Template": contract item 2 now
  says implement `_eval_bbnw`, the template code carries the method, and
  the "add a branch to `eval_bbnw`" block is gone.
- `.claude/skills/add-bbnw-model/SKILL.md` — API contract item 2,
  "contract 1" (rewritten from "evaluation is dispatched on `self.name`"
  to "implement `_eval_bbnw`, not `eval_bbnw`"), the scaffold, the
  two-component example, the Lee-style section (now pointing at
  `_eval_basis_bbnw`), a new pitfall for overriding the *public* method
  by mistake, and the checklist.
- This doc's Code map entry, marked ✎ superseded.
- `docs/models.rst` needed no change — it documents the models, never the
  dispatch mechanism.

### 2026-07-28 (Prompt 10: docs pass + sphinx installed and building)

Documentation only — no package code changed. **`sphinx-build` succeeds
with zero warnings in every file I touched.** Suite still 162 passed,
2 skipped.

**Sphinx installed, per your instruction.** Neither `sphinx` nor
`docutils` was present in `ocean14`, so I could not validate RST at first;
you asked me to add them. Installed into `ocean14`: **sphinx 9.1.0**,
**sphinx-rtd-theme 3.1.0**, **docutils 0.22.4**. `docs/requirements.txt`
already listed sphinx and the theme (ReadTheDocs installs from it), so the
gap was purely local. It now also lists `docutils>=0.18`, records the
verified versions, and **documents two entries that were listed but are
not actually used** rather than deleting them silently:
`sphinxcontrib-napoleon` (superseded — `conf.py` uses the built-in
`sphinx.ext.napoleon`, and the standalone package is unmaintained) and
`sphinx-autodoc-typehints` (absent from `conf.py`'s `extensions`). Also
added `extras_require={'docs': [...]}` to `setup.py` so
`pip install -e ".[docs]"` works locally, with a note to keep it in sync.

**Build result:** `build succeeded, 57 warnings` — **none of them in
`models.rst`, `parameters.rst` or `fitting.rst`**. I verified the new
content actually renders (the `log-params` anchor, the math blocks, the
DOI list, the new sections) rather than trusting the absence of warnings.

#### `docs/models.rst`

- **Backscattering section rewritten and corrected.** It documented
  models that do not exist under names the code does not accept
  (`PowerLaw`, `Constant`) and omitted `Every`. Now all seven — `Pow`,
  `Lee`, `GSM`, `Cst`, `Every`, `Pow2`, `Pow2Flat` — with the real
  `init_model` names, real `pnames`, equations, and the note that
  positive `beta` means a *decreasing* `bb_nw`.
- `Pow2` and `Pow2Flat` get their equations, parameters, default priors,
  and the reason the two exponent ranges are disjoint.
- New **"Turbid water: why two components"** section: the physical
  argument with citations, plus the three practical findings from the
  Prompt-9 benchmark — prefer `Pow2Flat` + MCMC, an inflated noise floor
  does *not* help identifiability, and raise `maxfev`.
- New **"Which parameters are log10"** section (anchored `log-params`)
  documenting the `log_params` convention, that display code reads it and
  the fitters do not, and that the two conventions must be kept
  consistent.
- **Absorption section also corrected** while I was there: it listed a
  nonexistent `QAA` model and wrong parameter names (`A_ph`, `E_ph`,
  `a_dg_443`). Now the real set with real `pnames`, including the
  `Chase2017` caveat about sitting outside the prior-flavour convention.
- **References**: 5 bare one-line entries → 17 full citations with
  **DOIs**, split into "absorption and reflectance models" and
  "backscattering in turbid and mineral-dominated water" (the latter
  being the evidence base for `Pow2`).

#### `docs/parameters.rst`

- New **"Turbid-Water Configurations"** section for `expb_pow2`,
  `expb_pow2flat` and `expb_powflex`, including *why* `Pow2Flat` is the
  recommended starting point and the warning to always pass `bpriors`
  explicitly rather than relying on `set_standard_priors`' blanket
  `log_uniform(-6, 5)`.
- Fixed two errors: `standard.gsm_gsm(...)` (no such function — it is
  `standard.gsm`, and it takes no `beta`) and
  `model_names == ['ExpBricaud', 'PowerLaw']` → `['ExpBricaud', 'Pow']`.

#### `docs/fitting.rst`

- **The least-squares section documented an API that does not exist**, so
  I could not add `maxfev` to it honestly. It showed
  `fit(models, wavelengths, Rrs, Rrs_err, p0=..., method='trf',
  bounds=[(lo, hi), ...], max_nfev=1000)` returning a dict with
  `'x'`/`'chisq'`/`'rchisq'`, plus a `chisq_fit.calc_chisq` that does not
  exist. The real signature is
  `fit(items, models, rt_dict, bounds=None, maxfev=None) -> (ans, cov,
  idx)` with `items = (Rrs, varRrs, p0, idx)` and bounds as
  `(low_array, high_array)`. Rewritten to that, with a worked example
  that computes χ²_ν from the returned parameters.
- **`maxfev` documented** with both caveats that matter: it changes
  *whether* the fit returns rather than how well the model fits, and
  parameter-rich models need it (the 5/8 and 6/8 → 8/8 convergence
  numbers from the benchmark). Plus the note that one spelling covers
  both scipy back ends.
- New **"Walker initialization"** subsection for `init_walkers` /
  `prior_bounds` / `perturb_frac` / `perturb_floor`, with a `.. warning::`
  explaining the frozen-dimension failure mode and the note that the
  legacy global RNG is deliberate so `batch_fit(seed=...)` stays
  reproducible.

#### `CLAUDE.md`

- `bbNWPow2` and `bbNWPow2Flat` added to the backscattering table, plus a
  turbid-water guidance paragraph (prefer `Pow2Flat` + MCMC; χ² is
  degenerate; raise `maxfev`) linking the benchmark and notebooks.
- The three new combos added to "Standard Model Combinations".
- New "Which parameters are log10 (`log_params`)" subsection.
- **"New Backscattering Model Template" rewritten.** The old one could not
  work: `super().__init__(wave)` (the base takes `(wave, prior_dicts)`)
  and a subclass-level `eval_bbnw` (evaluation is dispatched by a string
  `if/elif` in the *base* class). Now shows the real pattern — class
  attributes including `log_params`, `prior_dicts=None`, the base-class
  branch using the `wave` **argument**, the `init_model` entry, and the
  `standard.<combo>()` factory — with the p0-is-linear and
  never-seed-zero rules stated where they apply.

#### `.claude/skills/add-bbnw-model/SKILL.md`

Same corrections, plus what the skill was missing entirely:

- New **"Three contracts that bite"** section: base-class dispatch;
  p0-is-linear-and-log10'd-by-prior-flavour (with both silent failure
  directions spelled out); bounds-from-`pmin`/`pmax` (so no `gaussian` or
  `ratio` priors on bb parameters, and why).
- The Lee-style scaffold fixed, now with `eval_basis_func(wave)` and an
  explicit ⚠ not to evaluate the cached `basis_func` in the branch —
  which is exactly the bug fixed in Prompt 4.
- "Add a default combo to standard.py (optional)" → **not optional**,
  with the `set_standard_priors` fallback explained.
- The test section now points at `bing/tests/test_bbnw.py`, which exists,
  and at the reusable helpers in it.
- **Pitfalls rewritten and ordered by how much time they cost**, covering
  every trap this stage actually hit: ignoring `wave`, flavour/`log_params`
  disagreement, zero seeds, gaussian priors, wrong-length `bpriors`,
  `gen_bounds()`, `super().__init__(wave)`.
- Verification checklist extended with the Raman-grid check, the
  `log_params`-vs-flavours check, the prior-length check, and a
  default-`maxfev` convergence check.

#### Pre-existing doc problems I did NOT fix (flagging, not fixing)

The build's 57 warnings are all in files outside this stage's scope:
`docs/api/*.rst` referenced by the api toctree do not exist (7 warnings);
`index.rst` references missing `examples`, `contributing`, `changelog`
and `references` documents (4 — the last being the `references.rst` I
flagged back in Prompt 1); `save_load.rst:38` has a **malformed table**;
`raman.rst` has 9 "inconsistent title style" errors and duplicate
citations with `chlorophyll_fluorescence.rst`; `radiative_transfer.rst`
has two undefined `|G2|` substitutions; and the MCMC half of
`fitting.rst` still documents `init_mcmc`/`set_standard_priors`
signatures that do not match the code (I corrected only the least-squares
half and the walker-init part, i.e. the surfaces this stage changed).
Say the word if you want a separate docs-accuracy pass.

### 2026-07-28 (Prompt 9: benchmark + identifiability recommendation)

Built `dev/turbid_bbp/turbid_bbp.py` (four figures alongside it). No
package code changed. **The benchmark returns one clear recommendation
and one uncomfortable negative result — please read the second one.**

#### RECOMMENDATION (identifiability)

**Use `Pow2Flat` (3 parameters, `eta_min` fixed at 0) with MCMC.** Not
χ², and *not* the inflated noise floor. Evidence, one turbid spectrum,
48 walkers per Q10:

| strategy | σ(Bmin) | σ(eta_min) | σ(Borg) | σ(eta_org) | worst bias |
|---|---|---|---|---|---|
| χ², tight noise | 0.91 | **1.98** | 0.94 | 2.01 | — (unconstrained) |
| MCMC, tight noise | 0.15 | 0.24 | 0.70 | 0.21 | 1.1σ |
| MCMC, inflated 5% floor | 0.18 | 0.20 | **1.50** | 0.39 | **5.7σ** |
| **MCMC, `Pow2Flat`** | **0.16** | fixed | **0.15** | **0.19** | **0.3σ** |

- **χ² on the 4-parameter model is degenerate**, not merely imprecise:
  `Bmin`–`Borg` correlation **−1.00**, `Bmin`–`eta_min` +0.98,
  correlation-matrix condition number **1.1e7**, and `eta_org` pinned on
  its prior bound. The covariance is meaningless there.
- **MCMC tames it** (condition number never enters, correlations drop to
  −0.74) but `Borg` stays poor and `eta_min` lands ~1σ off truth.
- **The inflated noise floor makes recovery *worse*, not better** — it is
  the one result I did not expect. `Adg` comes back −0.59 against a truth
  of −0.30 and `Bmin` is **5.7σ** off. Loosening σ widens the likelihood
  and lets the fit wander along the degenerate direction. Keep the floor
  strictly for making χ²_ν *interpretable* (as the IOPtics side already
  labels it) and never as a way to constrain a degenerate model.
- **`Pow2Flat` + MCMC recovers every parameter within 0.3σ** with the
  tightest errors of any strategy. Fixing `eta_min` removes the
  degenerate direction outright.

#### THE NEGATIVE RESULT: this synthetic evidence does not justify Pow2

At GLORIA-like noise, **a single power law is statistically adequate
against a two-component truth**. Sweeping `eta_min` across its whole
prior range, with the amplitudes balanced (the *most* two-component case
— either extreme reduces to a single power law):

| eta_min | Pow χ²_ν | Pow2 χ²_ν | σ_crit | vs measured σ |
|---|---|---|---|---|
| −0.50 | 1.05 | 1.03 | 3.9e-5 | 3.9× tighter |
| −0.25 | 0.95 | 0.96 | 3.1e-5 | 4.8× |
| 0.00 | 1.05 | 1.00 | 2.4e-5 | 6.1× |
| +0.25 | 0.97 | 0.99 | 1.8e-5 | 8.2× |
| +0.50 | 1.00 | 0.99 | 1.3e-5 | 11.4× |

`σ_crit` is the noise level at which `Pow` *would* start to be rejected
(rms of its structural residual against the noiseless truth; for a wrong
model E[χ²_ν] ≈ 1 + mean(r²)/σ²). **The measurement error would have to
be 4–11× tighter than GLORIA's ~1.5e-4 sr⁻¹.**

Why this matters for the original goal: `Pow`'s structural misfit against
a `Pow2` truth is ~**2e-3** relative, while real turbid GLORIA spectra
are missed by ~**48%** — *250× larger*. **So `Pow2` is very unlikely to
be what fixes GLORIA.** A related realisation while reading these
numbers: the report framed the problem partly as backscatter *magnitude*
(required 0.2–0.4 m⁻¹ vs fitted 0.013), but `Pow`'s amplitude prior is
`log_uniform(-6, 5)` — it can reach 0.4 m⁻¹ trivially, and in this
benchmark it does (the synthetic truth runs 0.26 → 0.12 m⁻¹ and `Pow`
tracks it). What a single power law cannot produce is a *shape* far from
a power law — and no mineral+organic sum in the plausible range is far
enough to matter at real noise.

**Suggested next step (not done — your call):** fit real turbid GLORIA
with `Pow2Flat`+MCMC and see whether the 48% misfit actually drops. If it
does not, the remaining suspects are the ones currently deferred — the
fixed Gordon coefficients at these backscatter levels (Q5) and the
absorption side — not the `bb_nw` parameterisation. That is an IOPtics
task, since GLORIA loading lives there.

#### maxfev turns out to be required, not optional

A second, cleaner win for Prompt 5's work. On the 8 clear L23 spectra,
with **scipy's default budget**: `Pow` 8/8, `PowFlex` 8/8, **`Pow2Flat`
6/8, `Pow2` 5/8**. With `maxfev=40000`: **8/8 for all four.** The
two-component models simply cannot be run through χ² at the default
budget. (`fit_with_LM` still does not expose `maxfev`, so the benchmark
reimplements its ~10 lines to pass it — the one-line addition I flagged
as not-done in Prompt 5 now has a concrete customer.)

#### No regression on open ocean

Paired on all 8 L23 spectra (compared only where *every* model
converged — otherwise the medians describe different samples):

| model | converged | median χ²_ν | median \|Δa_nw\|/a | median \|Δbb\|/bb |
|---|---|---|---|---|
| `Pow` | 8/8 | 0.0079 | 0.116 | 0.023 |
| `PowFlex` | 8/8 | 0.0079 | 0.116 | 0.023 |
| `Pow2Flat` | 8/8 | 0.0081 | 0.116 | 0.023 |
| `Pow2` | 8/8 | 0.0082 | 0.116 | 0.023 |

Identical IOP accuracy, χ²_ν differing only by the degrees-of-freedom
divisor. The figure shows the four curves lying on top of each other
across all 8 spectra.

#### Two mistakes of my own worth recording

1. **My first L23 comparison was invalid.** With a broad `except
   Exception` and no `maxfev`, `Pow2` "converged" on 4/8 and posted a
   *better* median χ²_ν than `Pow` — a pure selection effect, since only
   the easy spectra survived. Fixed by narrowing the catch, raising the
   budget, and comparing on the common subset.
2. **Narrowing the catch immediately exposed a bug in my own index
   list:** `L23_IDX` included 3500, but L23 has 3320 spectra, so that
   entry had been silently swallowed for every model. Now 3200.

#### Files

`dev/turbid_bbp/turbid_bbp.py` plus `turbid_bbnw_recovery.png`,
`turbid_rrs_panels.png`, `turbid_identifiability.png`,
`turbid_l23_regression.png`. Per **Q10** the MCMC runs use
`nwalkers=48` **locally**; `init_mcmc`'s default (16 for 7 parameters) is
untouched. Runtime ~7 min. The Prompt-8 notebooks show the *noiseless*
demonstration; this benchmark adds the noise, identifiability and
convergence layers that change the conclusion.

### 2026-07-28 (Prompt 8: end-to-end L23 test + visualization notebooks)

Added test 9 and two executed notebooks. **3 new tests; full suite 162
passed, 2 skipped, in 2m40s.** No source changes.

**Test 9 (`test_bbnw.py`)** — a module-scoped fixture fits L23 idx **170**
(the clear, blue-peaked spectrum the existing L23 tests use) through
`fit_l23.fit_with_LM` with `expb_pow`, `expb_pow2` and `expb_pow2flat`,
then three assertions:
- `Rrs_model` finite and positive, `bb_nw` finite and positive;
- the premise is asserted rather than assumed — the spectrum's Rrs really
  does peak below 500 nm, so "clear" is checked, not claimed;
- **"no worse than `expb_pow`"** in a form that is actually principled.
  Raw chi² must be no worse (1% slack for optimizer wobble), because
  `Pow2` contains `Pow` exactly. Reduced chi² is allowed to rise by
  *exactly* the degrees-of-freedom factor `(n-k_pow)/(n-k_new)` and no
  more — rather than picking a tolerance that happens to pass.
- Plus a check that the clear-water solution is the *same* solution: the
  recovered `bb_nw` agrees with the single power law's to 10% at 440,
  555 and 700 nm, so `Pow2` is not wandering into a degenerate corner.

Measured on idx 170: **all four models give identical raw chi² = 0.1357**;
chi²_ν goes 0.00242 (k=5) → 0.00247 (k=6) → 0.00251 (k=7), i.e. purely
the ν penalty. The turbid models cost nothing on open-ocean water.

Guarded as asked: the fixture imports `bing.fitting.l23` inside a
`try` and skips on `ImportError`, and skips on `FileNotFoundError`/`OSError`
from the fit, so `test_bbnw.py` still runs on a checkout with no L23 tree.
(Here they run — 39 tests in that module, none skipped.)

**Notebooks** — new `nb/TurbidWaters/`, both **executed with outputs and
figures embedded** (via `jupyter nbconvert --execute`), verified
programmatically to contain **zero error outputs**:

1. **`turbid_bbnw_models.ipynb`** (14 cells, 3 figures) — what the new
   models are and why two components are needed.
   - Shape gallery: what a single power law can do (β = 0, 1, 2, and
     PowFlex's β = −0.5) vs `Pow2` shown *decomposed* into its mineral
     and organic terms.
   - The mechanism, numerically: for a two-component `bb_nw` the local
     log-log slope varies across the band (**1.43 at 420 nm → 1.26 at
     550 → 1.08 at 700**), and a single power law has one slope by
     construction.
   - The controlled experiment: median relative Rrs misfit **7.40e-4 for
     `Pow`, 7.40e-4 for `PowFlex` (identical), 8.2e-16 for `Pow2Flat`,
     2.1e-16 for `Pow2`** — with the punchline made explicit, that
     PowFlex's optimum is `beta = +1.1973`, *inside* `expb_pow`'s
     existing range, so widening the range buys literally nothing.
   - The L23 no-regression table and a truth-vs-recovered `bb_nw` panel.
2. **`walker_init_fix.ipynb`** (13 cells, 3 figures) — the Q6 bug and fix.
   - The walker ball itself: `eta_min` spread **0.00e+00 (old) vs
     5.79e-04 (new)**, with every well-scaled parameter *bit-identical*
     between the two (6.05e-03, 4.15e-03, ...), which is the visual proof
     that the fix is surgical. `Sdg` widens 8.1e-05 → 5.4e-04.
   - Real 250-step chains, old vs new: the old `eta_min` trace is a flat
     line with spread exactly 0.000e+00 **at acceptance 0.36** — the
     "confidently wrong uncertainty" failure mode, shown rather than
     described — vs 5.75e-02 for the new one.
   - The L23 disturbance check, recomputed in-notebook with two seeds per
     method, reproducing the Prompt-7 table to the digit, plus a bar chart
     showing the old-vs-new shift sitting at the same height as each
     method's own seed-to-seed scatter.

I put these under `nb/TurbidWaters/` to match the existing per-topic
layout (`nb/ChlFl`, `nb/Raman`, ...). Both are self-contained: they define
the pre-fix behaviour inline (`old_init_walkers`, and the multiplicative
ball) so the comparison can be re-run at any time without checking out
old code.

**Suite.** 162 passed, 2 skipped (the two unconditional
`pytest.skip()`s in `test_l23_fitting.py`). Nothing regressed.

### 2026-07-28 (Prompt 7: Q6 — walker init no longer freezes zero-seeded params)

Implemented the Q6 fix. One source file (`bing/fitting/inference.py`)
plus a new test module. **8 new tests; full suite 159 passed, 2 skipped,
in 2m32s.**

**The change.** Two new functions, so the logic is testable without
running a sampler:
- `prior_bounds(models)` — per-parameter (lower, upper) from the models'
  priors, with `±inf` wherever a bound is unavailable (no priors
  attached, or a flavor like `gaussian` that inherits `pmin = None` from
  the `Prior` base). Nothing raises, nothing becomes NaN.
- `init_walkers(p0, nwalkers, models=None, frac=1e-2, floor=1e-3,
  rng=None)` — walkers are `p0 + U(-1, 1) * max(|p0|*frac, floor)`,
  then clipped into the prior bounds.

`run_emcee` now calls it, and exposes `perturb_frac` / `perturb_floor`
so the step-9 benchmark can widen the ball for a degenerate model
without touching the defaults.

**Why this shape.** For `|p0|*frac` above the floor the distribution is
**identical to the old one** (the old `p0 + p0*U(-frac, frac)` is a
symmetric interval of half-width `|p0|*frac`, negative `p0` included),
so well-scaled parameters are perturbed exactly as before. The floor
only bites near zero: a slope seeded at `0.015` went from a half-width
of 1.5e-4 to 1e-3, and one seeded at `0.0` from **exactly 0** to 1e-3.
Clipping additionally guarantees every walker starts with a finite
log-probability, which the old version did not.

**One constraint worth recording.** The default RNG stays the **legacy
global** `np.random`, not a `Generator`. `bing.fitting.l23.batch_fit`
calls `np.random.seed(seed)` for reproducibility (`l23.py:664`), and
since the old perturbation used `np.random.uniform`, that seeding
governed walker init. Switching to `default_rng()` would have silently
broken `batch_fit(seed=...)`. `init_walkers` therefore defaults to
`np.random` and accepts an injectable `rng` for tests
(`rng = np.random if rng is None else rng` — both provide `uniform`).
There is a test pinning the `np.random.seed` reproducibility.

**Effect on a real L23 fit, as asked.** `expb_pow`, PACE, idx=170,
8000 steps — run **twice per method with different seeds**, because
otherwise a single old-vs-new pair tells you nothing about whether a
difference is systematic or just MCMC noise:

| run | Adg | Sdg | Aph | Bnw | beta | max ACT |
|---|---|---|---|---|---|---|
| OLD-a | −1.6419 | 0.0135 | −2.1727 | −3.5119 | 0.9885 | 400 |
| OLD-b | −1.6362 | 0.0133 | −2.1817 | −3.5129 | 0.9863 | 438 |
| NEW-a | −1.6447 | 0.0137 | −2.1510 | −3.5153 | 0.9669 | 413 |
| NEW-b | −1.6424 | 0.0136 | −2.1605 | −3.5114 | 1.0034 | 439 |

Seed-to-seed |Δmedian| within a method reaches 0.009 (Aph) and 0.037
(beta); the old-vs-new difference is 0.022 for both — i.e. **the same
order as each method's own run-to-run scatter**, not a systematic shift.
Integrated autocorrelation time is unchanged (400–439 either way), so
the wider `Sdg` ball costs nothing in mixing, and the `Sdg` posterior
itself is identical across all four runs (0.0133–0.0137) despite its
initial ball being ~7× wider. **Conclusion: no existing L23 result
changes beyond MCMC noise; the fix only bites where a dimension was
previously dead.** (Aside: max ACT ≈ 400 against 8000 steps means these
short test fits are only marginally converged — pre-existing, and the
reason run-to-run scatter is visible at all.)

**Tests** — new `bing/tests/test_inference.py`:
- **the regression**: for a p0 containing an exact 0, every dimension
  has non-zero inter-walker spread, and the same test *demonstrates the
  old rule failing* (recomputes the multiplicative ball inline and
  asserts that column's std is exactly 0);
- the ball matches the historical 1% width for well-scaled parameters
  (and genuinely fills it), while the floor bites only near zero;
- all walkers land in-prior even when p0 sits **exactly on a prior
  boundary** (`eta_min = pmin = −0.5` with a deliberately wide floor),
  and clipping does not flatten the clipped dimension;
- `prior_bounds` tolerates a `gaussian` prior (→ `±inf`, left unclipped)
  and a model with no priors at all;
- `np.random.seed` reproducibility, per the constraint above;
- **end-to-end through `run_emcee`**: a real 250-step sample on a
  synthetic spectrum whose truth has `eta_min = 0`, asserting every
  column moves and the formerly frozen one actually explores.

A test-authoring note, since it nearly produced a false positive: my
first version of the gaussian-prior test used
`standard.expb_pow(beta=1.0)`, assuming that yields a gaussian prior on
`beta`. It does **not** — that branch of `set_standard_priors`
(`priors.py:377`) is an `elif` after `if p.bpriors is not None`, and
`expb_pow` always supplies `bpriors`, so the gaussian is unreachable
there. This is the same gating I noted in the Prompt-1 audit (it is why
`p.beta` can never mis-target index 1 of a 4-param bb block). The test
now builds the gaussian prior dict explicitly and asserts the flavor
before relying on it.

**Observation, not changed:** `fit_one` passes `skip_check=True`
(`inference.py:231`), which disables emcee's initial-state validation.
Now that walkers are guaranteed in-prior and non-degenerate, that guard
could probably be re-enabled — it would have caught this bug — but it
risks raising on models with no priors attached, so I left it alone.
Say if you want it revisited.

### 2026-07-28 (`correct_atmosphere` resolved — FULL suite green, gaps closed)

JXP moved the `correct-atmosphere` checkout onto a branch carrying the
`Ed` work. **Confirmed and verified.** No code changed in this pass; this
entry closes the verification gap that prompts 2–6 each had to flag.

**Confirmation.** `/Users/xavier/Oceanography/python/correct-atmosphere`
is now on branch **`develop`**, whose history includes *"Merge pull
request #12 from ocean-colour/Ed"* (`8d5c549`);
`correct_atmosphere/downwelling.py` is present and
`from correct_atmosphere import downwelling` succeeds. Collection went
from **3 modules erroring** to **153 tests collected**.

**Full suite: 151 passed, 2 skipped, 0 failed, in 2m34s.** The 2 skips
are unconditional `pytest.skip()` calls written into
`test_l23_fitting.py` (`:965`, `:974` — "Requires pre-existing chain
files from batch_fit"), i.e. pre-existing by design, not data- or
change-related.

**What this actually closes.** `test_evaluate.py`, `test_io.py` and
`test_l23_fitting.py` are precisely the modules that exercise the code
paths I could previously only verify by driving them directly. All four
of those verifications now have real test coverage behind them:

- **Prompt 4 (Q8, the `wave` kwarg).** `test_raman_fitting_LM` and
  `test_raman_fitting_MCMC` fit L23 spectra through the Raman path with
  `Lee`/`GSM`, i.e. exactly the `eval_bb_ex` call that was mixing grids.
  They pass with the corrected `bb_ex`, and no assertion tolerance
  tripped — consistent with the measured impact (`bb_ex` off by 16–22%,
  but Rrs by <0.02% and retrieved parameters by ~1e-4 dex). Independent
  support for "the fix is right and no past result needs redoing".
- **Prompt 3 (`check_priors` in `set_standard_priors`).** `l23`'s
  `prep_one_l23` calls `set_standard_priors` for every fit, so the hook I
  added now runs on every L23 test — including `expb_pow`, variable-Gordon,
  Raman and Chl-fluorescence variants. Green, so the guard is inert for
  correct prior lists, which is what I had only been able to argue by
  enumerating combos.
- **Prompt 6 (`log_params` display).** `test_l23_fitting` calls
  `plotting.show_fits` (visible in the warning trace at
  `plotting.py:327`), so the changed display code is exercised end to end.
- **Prompt 5 (`maxfev`).** Every L23 χ² fit calls `chisq_fit.fit` *without*
  `maxfev`, confirming the added keyword leaves scipy's default path
  untouched for existing callers.

**Standing caveat now retired.** Earlier entries say the suite runs
"80 / 91 / 96 / 115 passed with 3 modules uncollectable" — those numbers
were the *runnable subset*. The current, complete figure is **151 passed
+ 2 skipped**. The `origin/Ed` follow-up noted in the Prompt-2 log is
done; the remaining publish-to-`main` follow-ups (bing's turbid models,
`ocpy`'s `load_gloria`) are unaffected and still open.

### 2026-07-27 (Prompt 6: `log_params` + display code honours it)

Implemented item 4. Display-only, as instructed — **the p0 log10 step is
untouched.** Four source files plus a new test module. **19 new tests,
115 in the runnable suite, all passing in 3.4 s.**

**The convention.** `log_params` is a per-parameter list of booleans
declared on the model classes, documented on both base classes
(`bbNWModel`, `aNWModel`) with an explicit note that it is read by
*display* code and deliberately **not** by the fitters' p0 conversion,
which keys on prior flavor. `None` means "all log10", the historical
default, so any model that does not declare it behaves exactly as before.

**Declared where a parameter is genuinely linear** (I traced each eval
branch rather than guessing):
- bb: `Pow` `[True, False]` (beta), `Pow2` `[True, False, True, False]`,
  `Pow2Flat` `[True, True, False]`; `Cst`/`GSM`/`Lee` `[True]` and
  `Every` `[True]*nparam` declared explicitly for the record.
- a: `Exp` `[True, False]` (Snw), `ExpBricaud` and `ExpBricaudFix`
  `[True, False, True]` (Sdg), `ExpBricaudFree`
  `[True, False, True, True]`, `ExpNMF` `[True, False, True, True]`.

**Scope note — I extended this to the a-models, beyond the letter of
item 4** (which named the bb classes). Reason: `Sdg` is linear and
appears in *every* standard combo, so a bb-only fix would have left the
parameter panel printing `Sdg = 1.04` (that being `10**0.015`) while
proudly showing the correct `beta`. The change is one declarative
attribute per class with a fallback that leaves undeclared models
untouched, so the risk is minimal — but say the word if you want it
confined to bb.

**Left undeclared on purpose: `Chase2017`/`Chase2017Mini`.** I checked
before assuming, and Chase is deliberately outside the flavor
convention: **all** of its 28 parameters are log10 quantities (the
Gaussian block goes through `functions.gaussian`, which does `10**` on
amplitude, width *and* centre), yet its priors are declared
`flavor='uniform'` with `np.log10(...)` bounds and its `init_guess`
returns log10 values directly (`anw.py:1319-1322`). So Chase opts out of
the flavor-driven p0 log10 step on purpose, and for *display* the
all-log10 fallback is already correct for it. Nothing to declare. Worth
knowing if the deferred "drive p0 off `log_params`" change is ever
attempted: Chase would need care, because for it flavor and log-ness
genuinely disagree.

**Display sites fixed.**
- `bing/plotting.py` — new documented helper `log_param_mask(models)`
  (accepts a list or a bare model, concatenates in parameter order,
  falls back to all-log10). The `show_params` text block now prints
  `10**p` only for amplitudes and `p` for linear parameters.
- `bing/plotting.py::corner_plot` — `show_log=False` now exponentiates
  **only the log columns** (previously `coeff = 10**coeff` hit every
  column), and labels wrap in `\log_{10}(...)` only for log columns.
  Related correction: labels are no longer log10-wrapped when
  `show_log=False`, since in that mode the plotted values are linear —
  previously every panel was labelled log10 regardless.
- `bing/fitting/l23.py:519-524` — the mirrored corner block got the same
  label treatment, inlined (2 lines) rather than importing
  `bing.plotting`, so the fitting module still doesn't pull in
  matplotlib. **Note this block is inside `if False:`**, i.e. currently
  unreachable debug code; fixed anyway so reviving it isn't misleading.
- Also fixed a pre-existing LaTeX typo in both places: the label was
  `$\log_{10}(Adg$)` with the closing paren *outside* math mode.

**Tests** — new `bing/tests/test_plotting.py` (sets the Agg backend
before importing `bing.plotting`):
- the fallback, for a class with no `log_params` at all and for one that
  sets it to `None`;
- declared values read back correctly, and concatenation across
  `[a_model, bb_model]` in parameter order;
- length/type consistency for every bb model;
- **the cross-check I care about most:** for all 8 combos in
  `standard.py`, `log_param_mask(models)` must equal the mask derived
  from the *prior flavors*. The figures and the fitters now provably
  agree about which parameters are log10, for every shipped combo. This
  is what would catch a future model declaring one and priming the other.
- a real render: build 7200-step synthetic chains, call `corner_plot`
  twice, and read the labels back off the axes — 4 of 7 log-labelled
  with `show_log=True` (`Sdg`, `eta_min`, `eta_org` bare), none with
  `show_log=False`.
- the `show_params` formatting rule, asserting `Sdg = 0.015` and
  `beta = 1.000` survive as linear values.

**Suite.** 115 passed in 3.4 s (was 96). The same three modules
(`test_evaluate`, `test_io`, `test_l23_fitting`) remain uncollectable on
the unchanged `from correct_atmosphere import downwelling` — still
waiting on `origin/Ed`.

### 2026-07-27 (Prompt 5: `maxfev` through `chisq_fit.fit`)

Implemented item 3. One source file (`bing/fitting/chisq_fit.py`, a
keyword plus docs) and a new test module. **5 new tests, 96 in the
runnable suite, all passing in 2.5 s.**

**The change.** `chisq_fit.fit(items, models, rt_dict, bounds=None,
maxfev=None)`. The keyword is only forwarded when it is not None
(`kwargs = {} if maxfev is None else dict(maxfev=maxfev)`), so scipy's
own default is left completely untouched for every existing caller —
appended keyword with a default, so nothing positional moves. I checked
all callers first: `bing/fitting/l23.py:625`,
`ioptics/run.py:153` and the tests all pass `bounds=` by keyword, so
none of them are affected.

Documented in the module example, the `Parameters` block, and a `Notes`
paragraph recording *why* one spelling covers both back ends (with
finite bounds curve_fit uses least_squares/'trf' and renames `maxfev` to
`max_nfev` internally; unbounded goes to leastsq/'lm' under the original
name), plus the caveat that raising it changes only *whether* the fit
returns, not fit quality.

**Tests** — new `bing/tests/test_chisq_fit.py` (not `test_bbnw.py`; this
is fitting machinery, not a model). Everything runs on a noiseless
synthetic `expb_pow` spectrum, so no L23 tree and no `correct_atmosphere`:
- `test_maxfev_exhausted_bounded` / `..._unbounded` — a tiny budget
  raises `RuntimeError` in **both** back ends. The messages differ,
  which is the direct evidence the keyword reaches each one:
  bounded gives *"Optimal parameters not found: The maximum number of
  function evaluations is exceeded."*, unbounded gives *"... Number of
  calls to function has reached maxfev = 2."*
- `test_maxfev_generous_succeeds` — `maxfev=40000` converges and
  recovers the truth to 1e-3.
- `test_maxfev_none_matches_generous` — the kwarg is **inert** when the
  budget is ample: omitting it and passing a large value agree to
  `rtol=1e-10`, in both branches. This is the regression guard for
  existing callers.
- `test_maxfev_budget_is_the_binding_constraint` — sweeps budgets and
  asserts that a small one fails while a larger one succeeds, so the test
  proves a *causal* relationship rather than just "big number works".
  For the record, on this 5-parameter synthetic spectrum the smallest
  converging budget is **maxfev=20** bounded and **50** unbounded (trf
  needs ~6 evaluations per iteration for the numerical Jacobian, lm a bit
  more), and in both cases the recovered parameters are exact to 7e-16 —
  i.e. the budget is genuinely all-or-nothing, exactly as the GLORIA
  investigation found.

**Deliberately not done (one line each, say the word).**
- **`bing/fitting/l23.py`'s wrapper does not expose `maxfev`.** Adding it
  would mean either a new kwarg on `fit_chisq` or a new `p_ntuple` field,
  and the turbid work reaches `chisq_fit.fit` through IOPtics rather than
  through the L23 wrapper. Out of item 3's scope, so flagged not done.
- **The IOPtics side** (`AlgorithmSpec.maxfev` → `fit_chisq`) is an
  IOPtics change and belongs in that repo's stage plan; BING now offers
  the hook it needs.

### 2026-07-27 (Prompt 4: Q8 — `Lee`/`GSM`/`Every` now honour the `wave` kwarg)

Fixed the latent mixed-grid bug. One source file
(`bing/models/bbnw.py`) plus tests. **36 tests in `test_bbnw.py`, 91 in
the runnable suite, all passing in 2.6 s.**

**The bug.** `eval_bb_ex` (`bbnw.py:300`) asks for backscatter on the
Raman *excitation* grid via `eval_bbnw(params, wave=self.wave_ex)`, but
`Lee`/`GSM` returned `functions.gen_basis(params[...,-1:],
[self.basis_func])` where `basis_func` was baked on `self.wave` at
construction, and `Every` returned `10**params` — the per-channel
amplitudes, also on `self.wave`. So `eval_bb_ex` was adding pure-water
backscatter on the *excitation* grid to particle backscatter on the
*emission* grid. Silent, because `wave_ex.size == wave.size`.

**The fix.**
- New `bbNWModel.eval_basis_func(wave=None)` (base raises
  `NotImplementedError`), overridden by `bbNWGSM` (`(pivot/λ)^eta`) and
  `bbNWLee` (`(pivot/λ)^Y`). The dispatch now calls
  `self.eval_basis_func(wave)` instead of reading the cache, so the
  basis is recomputed on whatever grid is requested.
- `set_basis_func` on both classes now assigns
  `self.basis_func = self.eval_basis_func()`, so the cached attribute
  (which the docstrings and `evaluate.py:363` rely on) stays exactly
  what it was — single source of truth, no behaviour change on the
  native grid.
- `bbNWLee.eval_basis_func` raises a clear `ValueError` when `Y` is
  unset, instead of dying inside `np.outer` with an opaque `TypeError`.
- New `bbNWEvery.eval_channels(params, wave=None)`. `Every`'s parameters
  *are* bb_nw per channel, so another grid needs an interpolation
  choice: I interpolate linearly in **log10(bb_nw) vs log10(λ)** — i.e.
  locally power-law, the behaviour every other model here assumes — with
  extrapolation, since `wave_ex` extends blueward of the model grid
  (350 vs 400 nm here). That choice is **exact for a power law**,
  verified to 1e-10 including in the extrapolated region, and it returns
  the amplitudes untouched on the native grid.
- Bonus one-liner while I was in that branch: `Every` now returns
  `(nsample, nwave)` like every other model, instead of `(nwave,)` for
  1-D input. Flagging it since it is a (latent, shape-only) behaviour
  change beyond the `wave` fix.

**Quantifying the shift, as asked.** Two-stage answer, because the two
stages differ by three orders of magnitude:

| quantity (giop / gsm, Raman on) | change from the fix |
|---|---|
| `bb_ex` (excitation-grid backscatter) | median **16%**, max **22%** |
| Raman-corrected **Rrs** | median **0.004%**, max **0.02%** (at 605 nm) |
| **retrieved parameters** (χ² refit) | **1e-4 dex**, i.e. 0.010% (giop) / 0.015% (gsm) in bb amplitude |

So the quantity that was wrong was wrong by a *lot*, but Raman is a small
additive correction to Rrs, so the error never propagated: refitting a
synthetic Raman-on spectrum with the buggy vs fixed model shifts
parameters by ~1e-4 dex, far below any noise level. **Conclusion: the fix
is correct and worth having, but no existing `giop`/`gsm` result needs
redoing.** Method: I reproduced the pre-fix behaviour by monkeypatching
`eval_bbnw` on a model instance, so both numbers come from the same run
rather than from comparing across a code revert.

The **elastic** path is untouched by construction: with no `wave` kwarg
the basis is evaluated on `self.wave`, which is what the cache held.
Verified identical.

**Checked, and NOT a parallel bug:** the a-side. `aNWModel.eval_a_ex`
(`anw.py:405`) passes `wave=self.wave_ex` through to `eval_anw`, and the
`Cst`/`Exp`/`ExpFix`/`ExpBricaud*` branches all honour it (`a_ph` via
`set_aph(Chl, wave=wave)`); confirmed empirically that `a_ex != a_em` for
both GIOP and GSM. Two a-side notes for later, not touched here:
`anw.py:329`'s `Every` branch still returns `10**params` (same class of
bug, but `Every` is in no standard combo), and for **fixed-Chl** a-models
`set_aph` is skipped inside `eval_anw`, so `a_ph` comes from whatever grid
it was last set on — worth a look when someone next touches the a-side.

**Tests** (7 new, parameterized where it made sense):
`test_eval_bbnw_honours_wave` over `Pow`/`Lee`/`GSM`/`Every`/`Pow2`/
`Pow2Flat` asserts the excitation-grid result differs, the shape contract
holds for 1-D and chains, and `eval_bb_ex == bb_w_ex + eval_bbnw(...,
wave=wave_ex)`; `test_cst_is_grid_invariant` covers the one legitimate
exception (flat model — same values, but still the right shape);
`test_basis_models_recompute_on_grid` checks the recomputed basis against
the analytic `(pivot/λ)^exponent` on *both* grids and that the cached
`basis_func` still matches the native grid; `test_lee_without_Y_raises`;
`test_every_interpolation_is_power_law_exact` pins the interpolation
choice (and asserts we really are extrapolating).

**Suite.** 91 passed in 2.6 s (was 80). `test_evaluate.py`, `test_io.py`,
`test_l23_fitting.py` still cannot be collected on the unchanged
`from correct_atmosphere import downwelling` (needs `origin/Ed`). Those
three are exactly the modules that exercise `Lee`/`GSM` through the L23
Raman path, so instead of leaving that untested I drove the same code
directly: `calc_Rrs_from_models` with `include_Raman=True` for `giop`,
`gsm`, `expb_pow` and `expb_pow2`, in both 1-D and 3-chain form — all
finite, correct shapes, and row 0 of the chain result matches the 1-D
result.

### 2026-07-27 (Prompt 3: `Pow2` + `Pow2Flat`, base-class prior guard)

Implemented item 2. Four files: `bing/models/bbnw.py` (the two classes,
two `eval_bbnw` branches, `init_model` entries, the Q7 guard),
`bing/priors/priors.py` (one hook, see below), `bing/parameters/
standard.py` (two factories), `bing/tests/test_bbnw.py` (tests 1–7,
10–12 plus one more). **25 tests in the module, 80 in the runnable
suite, all passing in 2.3 s.**

**The models.**
- **`bbNWPow2`** — 4 params `['Bmin', 'eta_min', 'Borg', 'eta_org']`,
  `bb_nw = 10^Bmin·(700/λ)^eta_min + 10^Borg·(600/λ)^eta_org`. `pivot`
  stays the scalar 600 (external scripts read `models[1].pivot`) with the
  mineral pivot as a separate `pivot_min = 700`. Both branches use the
  `wave` **argument**, not `self.wave`, so `eval_bb_ex`/Raman is correct
  — unlike `Lee`/`GSM`/`Every` (step 4).
- **`bbNWPow2Flat`** — 3 params `['Bmin', 'Borg', 'eta_org']`,
  subclassing `bbNWPow2`, with the mineral term spectrally flat
  (`functions.constant` + `functions.powerlaw`). Subclassing gets the
  pivots and docstring lineage for free; `nparam`/`pnames`/`log_params`/
  `init_guess` are overridden.
- Both declare `log_params` (`[True, False, True, False]` and
  `[True, True, False]`) for step 6's plotting fix, and seed `eta_min` at
  **0.05** rather than 0 (landmine 1), with the reason in a comment.
- `standard.expb_pow2()` / `expb_pow2flat()` carry explicit `bpriors` —
  never relying on `set_standard_priors`' blanket `log_uniform(-6,5)`
  (landmine 4) — with the eta ranges disjoint ([-0.5, 0.5] mineral,
  [0.5, 2] organic) to kill the label-switching degeneracy.

**Q7 — implemented slightly differently than the sketch, deliberately.**
The sketch put an assert in each new class's `__init__`. While
implementing I checked where priors actually get attached and found that
**both production paths attach them post-construction**:
`bing.fitting.l23` (`:288-291`) and IOPtics' `AlgorithmSpec.build_models`
(`spec.py:186-188`) build models with **no** prior dicts and then call
`bing_priors.set_standard_priors(models, p)`. A constructor-only assert
would therefore have guarded almost nothing real — only direct
`init_model(..., prior_dicts)` calls, i.e. mostly the tests. So:
- the check lives in the base class as **`bbNWModel.check_priors()`**
  (raising `ValueError` with the model name and both counts — better than
  a bare `assert`, which vanishes under `python -O` and carries no
  message), called from `bbNWModel.__init__`; and
- `set_standard_priors` calls it after attaching, guarded by
  `hasattr(..., 'check_priors')` so a-models are untouched
  (`priors.py:390-397`).

That second hook is one line beyond what Q7 asked for, so: **flagging it
explicitly.** I verified it cannot regress anything — every existing
combo (`expb_pow`, `expbf_pow`, `giop`, `gsm`, `k2b`) plus `Every`
(nparam = 61 on a 61-band grid) passes through `set_standard_priors` with
matching counts. Extra priors are safe too: `othera_priors` are *appended
afterwards* and only ever to `models[0]`, so the strict equality check
never sees them, and `Priors.calc` keeps its existing `len(priors) >
params.size` support for `ratio` extras. Say the word if you'd rather the
hook came out.

**Tests** (`test_bbnw.py`, parameterized over both models where the doc
said "both"): construction with no priors + direct construction (1);
`(1, nwave)` / `(nsample, nwave)` shapes and `eval_bb = bb_w + bb_nw`
(2); the `wave` kwarg genuinely changing the result and `eval_bb_ex ==
bb_w_ex + eval_bbnw(..., wave=wave_ex)` (3); reduction to `Pow` with the
mineral term off, and to a hand-written 700 nm law with the organic term
off (4); the turbid shape reachable at 0.1–0.4 m⁻¹ in the red with a
proof that `Pow` cannot do it (5); priors attached with `pmin`/`pmax`
present and the eta ranges disjoint (6); the p0 linear→log10-by-flavor
round trip landing in-bounds and in-prior, reproducing bb_nw at the pivot
within a factor 2, with no slot at exactly 0 (7 + 10); the prior-length
guard firing for both too-short and too-long lists **and via
`set_standard_priors`** (11); `log_params` matching the factory flavors
(12).

**One extra test, because it is the whole thesis.**
`test_two_comp_beats_single_powerlaw_chisq` builds a synthetic Rrs whose
true `bb_nw` is a genuine *sum* — steep organic dominating the blue, flat
mineral dominating the red, so the log-log slope changes with wavelength
— and fits it four ways:

| model | median relative Rrs misfit |
|---|---|
| `expb_pow2` | **1.1e-15** (exact; all 4 bb params recovered) |
| `expb_pow2flat` | **1.1e-16** (exact; truth's mineral term is flat) |
| `expb_pow` | 1.3e-3 (max 8.7e-3) |
| `expb_powflex` | **1.3e-3 — identical to `expb_pow`** |

That last row is the point: `PowFlex`'s optimum has `beta = 1.151 > 0`,
i.e. *inside* `expb_pow`'s existing range, so **widening the slope range
buys literally nothing** when the true shape is a sum of two power laws,
while adding the second component nails it. "Form, not range" is now a
regression test, not an argument.

**A trap I fell into first, now written into the Benchmark section.** My
initial synthetic truth used a tiny organic amplitude (0.005 vs 0.25
mineral), which made the truth *effectively a single rising power law* —
and `PowFlex` recovered it to 4e-5 while `Pow2` gave 1e-16. That case
looks like a win for one component and is simply not discriminating. The
benchmark must use **comparable** amplitudes so both terms matter across
the band; I added that as an explicit requirement in Benchmark (dev/).

**⚠ The Q6 walker bug is worse than "cosmetic", and Pow2 will trip it in
normal use.** An MCMC smoke run (32 walkers, 400 steps, seeded at the
truth) came back with acceptance 0.35 and every dimension moving —
**except `eta_min`, spread exactly 0.000e+00, frozen for the whole run**,
because the truth's `eta_min` is 0. That is not a contrived seed: the χ²
fit above *legitimately converges to `eta_min = -0.000`* for a flat
mineral term, and the standard workflow feeds the χ² result to the MCMC
as p0. So a flat-mineral `Pow2` fit will report a delta-function
posterior with a zero-width credible interval for `eta_min` every time,
silently. My `init_guess` seed of 0.05 does **not** protect the
chisq→MCMC path. Step 7 (your approved Q6 fix) is therefore a
prerequisite for trusting any `Pow2` MCMC, including the step-9
benchmark; I have noted that on step 7 rather than pulling it forward
into this commit.

**Suite.** `test_anw`, `test_bbnw`, `test_chl_fl`, `test_raman` →
**80 passed in 2.3 s** (was 62 before this task). `test_evaluate.py`,
`test_io.py`, `test_l23_fitting.py` still cannot be collected on the
unchanged pre-existing `from correct_atmosphere import downwelling`
(needs `origin/Ed` merged — see the previous log entry). Because those
three exercise the `set_standard_priors` path I touched, I verified that
path directly instead: all 8 standard combos + `Every` attach priors with
matching counts.

### 2026-07-27 (Prompt 2: `PowFlex` — free-slope power law + tests)

Implemented item 1. Two files touched, both additive: a new factory in
`bing/parameters/standard.py` and a new test module
`bing/tests/test_bbnw.py`. **No existing code changed**, so nothing that
currently works can regress.

**`standard.expb_powflex()`** (`standard.py:32`, placed right after
`expb_pow`). Identical to `expb_pow` except one line:
`bpriors[1] = dict(flavor='uniform', pmin=-1., pmax=2.)` instead of
`pmin=0.`. In `(pivot/λ)^beta` a positive beta is a *decreasing* bb_nw,
so relaxing the floor to −1 is exactly what admits the flat/"white"
mineral limit — bb_nw may now be flat or **rise** toward the red. The
docstring says so, cites Snyder 2008 / Gordon 2009 / Doxaran 2009, and
states the purpose: this widens the *range* of the existing single power
law without changing its *form*, so a fit that still fails here
implicates the form rather than the priors. No class, no `init_model`
entry, no `eval_bbnw` branch — confirmed `functions.powerlaw` handles a
negative exponent unchanged.

**`bing/tests/test_bbnw.py`** — 7 tests, all passing, **1.3 s** total
(no MCMC, no L23 fitting path). The module is the home for the `Pow2`
tests in step 3. Two shared helpers deliberately *mirror production
code* rather than importing it, so the tests fail if the contracts drift:
`bb_bounds()` reproduces `l23.py:612-618` (bounds read from the **raw
prior dicts**) and `log10_by_flavor()` reproduces `l23.py:384-390` (p0
amplitude slots chosen by prior **flavor**).

- `test_powflex_priors_admit_negative_slope` — the factory matches
  `expb_pow` in models and a-priors, differing *only* in beta's range;
  asserts `pmin=-1` here vs `pmin=0` there, and that the flavor pattern
  is `[log, linear]` (contract 1).
- `test_powflex_prior_evaluates_negative_beta` — the `Prior` object
  accepts beta = 0 and −0.5 and still rejects −1.5 and 2.5, while
  `expb_pow`'s prior rejects −0.5 outright.
- `test_powflex_bounds_reach_negative_beta` — the χ² bound actually
  handed to `curve_fit` reaches −1 (this is the one that would catch a
  prior added in the wrong place).
- `test_powflex_rising_bbnw` — the physics: beta < 0 gives
  `bb_nw(700) > bb_nw(440)`, beta = 0 is spectrally flat, beta > 0 is
  the unchanged open-ocean case; exact-value check against
  `10**Bnw·(600/λ)^beta`; plus the `(nsample, nwave)` contract for
  chain-shaped params.
- `test_powflex_p0_within_bounds` — `init_guess` returns a **linear**
  amplitude, the flavor-driven log10 lands the seed inside the bounds
  and in-prior, and **no slot is exactly 0** (the landmine-1 guard, with
  the reason in a comment).
- `test_powflex_recovers_rising_bbnw_chisq` — the end-to-end control
  experiment (see below).
- `test_powflex_prior_count_matches_model` — `len(bpriors) == nparam ==
  priors.nparam`, since nothing in BING validates it (landmine 3).

**The result worth recording.** I built a synthetic Rrs from a truth with
beta = **−0.4** (rising bb_nw) and fit it both ways:

| priors | fitted beta | max relative Rrs residual |
|---|---|---|
| `expb_powflex` | **−0.4000** (exact) | **7.8e-16** (machine precision) |
| `expb_pow` | **0.0000** (pinned at its floor) | **2.2e-2** |

`expb_pow` doesn't just fit slightly worse — it jams beta on the boundary
and then *distorts the absorption* to compensate (`Adg` −1.00 → −0.81),
which is the same "absorption absorbs the blame for missing backscatter"
signature the GLORIA report saw. That is now a committed test, not a
one-off, and it needs neither L23 nor `correct_atmosphere`. It also
confirms the widened prior propagates correctly through *both* consumers
(the `Prior` objects and the `curve_fit` bounds).

**Suite.** `bing/tests/test_anw.py`, `test_bbnw.py`, `test_chl_fl.py`,
`test_raman.py` → **62 passed in 1.5 s** (61 before this task + the 7 new
minus... precisely: 55 pre-existing + 7 new; the module count moved 4 →
5). **Three modules cannot be collected** — `test_evaluate.py`,
`test_io.py`, `test_l23_fitting.py` — all on the same import:
`bing/fitting/l23.py:51` does `from correct_atmosphere import
downwelling`. This is **pre-existing and unrelated to this change** (my
diff is one new factory and one new test file; the import is at
l23 module scope).

> **Diagnosis for JXP, since only you can fix it.** After you installed
> `correct-atmosphere` the error changed from `ModuleNotFoundError` to
> `ImportError: cannot import name 'downwelling'`. The installed checkout
> (`/Users/xavier/Oceanography/python/correct-atmosphere`, on branch
> `main`) has no `downwelling` module at all — `main` ships
> `rayleigh/gases/glint/whitecaps/aerosols/...`. It lives on the
> **`origin/Ed`** branch: `git ls-tree -r --name-only origin/Ed` shows
> `correct_atmosphere/downwelling.py`, `tests/test_downwelling.py`,
> `docs/downwelling.rst`, `nb/05_downwelling_irradiance.ipynb`. So `Ed`
> needs merging into `main` (or installing from that branch) before those
> three bing test modules can be collected anywhere. I did not touch your
> checkout.

**Your Q6–Q10 answers are recorded and slotted** (none of them were in
this step's scope, so no behavioural code changed):
- **Q6** (fix the multiplicative walker perturbation, with a test) and
  **Q8** (make `Lee`/`GSM`/`Every` honour the `wave` kwarg, with a test)
  are both approved and are *behaviour* changes to existing fits. They
  are not part of item 1; I have added them to the Prompts list as their
  own steps so each lands with its own test and log entry rather than
  being smuggled into a model commit.
- **Q7** — the priors-length assert goes in the **base class**
  `bbNWModel.__init__`; folded into step 3 (the sketch's per-class assert
  moves to the base).
- **Q9** — `Pow2Flat` name confirmed; no change needed.
- **Q10** — bump `nwalkers` to 32–64 **locally in the turbid benchmark**
  (step 7), not in `init_mcmc`.

### 2026-07-26 (Prompt 1: verify the code map, fold in JXP's answers, plan)

**No code changed** — this was the read-only verification pass. Per the
prompt I used **Opus**: two Opus subagents ran in parallel, one
re-deriving every claim in the Code map from the source (with empirical
runs in `ocean14`, scipy 1.18.0), one auditing both repos for assumptions
that break when a bb model has **4** parameters not named `Bnw`/`beta`. I
then spot-verified the load-bearing findings myself
(`inference.py:310-330`, `plotting.py:258-266` and `:448-458`,
`priors.py:360-385`, `l23.py:840-848`, `inference.py:160`).

**Code map: 7 of 12 claims verified as written, 5 corrected** (marked ✎
in place):

1. `init_model`'s `prior_dicts` **defaults to `None`** (`bbnw.py:64`) —
   which is what makes the no-priors construction in test 1 legal.
2. The base `__init__` builds priors **conditionally** (`bbnw.py:211`);
   with none, `self.priors` stays `None` and every fit path later dies on
   `model.priors.priors`. Also: `prior_dicts` is a **required positional**
   on the bb subclasses (unlike the a-models), so `bbNWPow(wave)` is a
   `TypeError` — the new classes should default it to `None`.
3. **`Every` also ignores the `wave` kwarg**, not just `Lee`/`GSM`; `Cst`
   honours it. And this is a **live bug**, not a stylistic
   inconsistency — `eval_bb_ex` adds `bb_w_ex` on the excitation grid to
   `bb_nw` on the emission grid, silent because the grids are the same
   length. Raman + `giop`/`gsm` fits are affected today → Q8.
4. `functions.powerlaw` uses `np.outer` only for the **amplitude**; the
   exponent broadcasts via `reshape(-1,1)`. It is therefore **not**
   "always (nsample, nwave)" — ≥3-D params get their leading dims
   flattened. Irrelevant for the 1-D and 2-D shapes BING passes, but I
   had overstated it. The two-slice pattern (`p[...,0:2]`, `p[...,2:4]`)
   was verified empirically to give `(1, nwave)` / `(nsample, nwave)`
   with correct values.
5. Line numbers: bounds are `l23.py:612-618`, the p0 log10 block is
   `l23.py:384-390`. And the `beta`-name extraction I attributed to
   `ioptics/io.py:174` is actually `ioptics/evaluate.py:79`.

**Nine landmines found** and written into the doc as a new "Landmines"
subsection. The one that matters most, and that I would have walked
straight into:

> `inference.py:321` perturbs walkers **multiplicatively**
> (`p0 += p0*uniform(-1e-2,1e-2)`). A parameter seeded at exactly `0.0`
> gets **zero** spread, and emcee's stretch move can never move a
> dimension with zero inter-walker spread. So the natural
> `init_guess` seed for a flat mineral term — `eta_min = 0.0` — would
> have frozen that dimension for the whole run, reporting a
> zero-width credible interval with a perfectly healthy acceptance
> fraction. An Opus agent confirmed it both by simulating the tile+
> perturb (std exactly 0.0 across 16 walkers) and by running emcee
> 2000×16 on a broad 2-D Gaussian (`dim1 range: 0.0 0.0` while dim0
> explored freely). `fit_one` passes `skip_check=True`, disabling the
> only emcee check that might have caught it.

The doc now seeds `eta_min = 0.05` with a comment explaining why, and
test 10 asserts the seed is non-zero so nobody "simplifies" it back.
Others: plotting exponentiates/log-labels **every** parameter
(`plotting.py:262`, `:449-456`) — already wrong for `Pow`'s `beta`;
nothing validates `len(bpriors) == nparam` (silent p0 corruption on the
χ² path); `set_standard_priors` blankets bb params with
`log_uniform(-6,5)` if `bpriors` is omitted; `l23.py:842` keys extras on
the literal `'beta'`; `Priors.gen_bounds()` raises for any flavor but
`'uniform'`, so the `anw.py` `init_guess` idiom must not be copied.

**One concern I raised and then disproved**, worth recording so it isn't
re-litigated: `UniformPrior` subclasses `LogUniformPrior`
(`priors.py:158`) and overrides only the `flavor` string, so I suspected
a negative `pmin` might be mishandled. It is not —
`UniformPrior.calc is LogUniformPrior.calc` and that `calc` takes **no
logarithm** (`-inf` outside `[pmin, pmax]`, `0` inside). `uniform` and
`log_uniform` are mathematically identical; the distinction is pure
metadata driving the p0 log10 step. A negative-`pmin` uniform prior is
already in production in the GLORIA report script.

**JXP's answers folded in.**
- *Round-number priors are fine* → prior table kept as-is.
- *Mineral pivot 700 nm* → `Pow2` is now
  `10^Bmin·(700/λ)^eta_min + 10^Borg·(600/λ)^eta_org`. Implementation
  detail this forces: `self.pivot` must stay a **scalar 600** because
  four external scripts read `models[1].pivot` as "the wavelength where
  `params[0]` is the amplitude"
  (`papers/phytoplankton/Figures/py/figs_phyto.py:1541`,
  `figs_oo_poster.py:1211`, `posters/SBG_2025/py/sbg_poster_2025.py:376`,
  `papers/biomass/Analysis/py/fit_giop.py:174`), so the mineral pivot is
  a **separate** `pivot_min = 700.` attribute rather than making `pivot`
  a tuple. IOPtics reads `.pivot` nowhere.
- *Prototype both* → added **`Pow2Flat`** (3 params, `eta_min ≡ 0`,
  i.e. `Cst` + `Pow`) alongside the 4-param `Pow2`, with its own
  `eval_bbnw` branch and `standard.expb_pow2flat()`. Note the 700 nm
  pivot only affects `Pow2` — a flat term has no pivot. Naming is Q9.
- *No third turbid scheme, no Gordon work* → left out of scope.

**Plan.** Prompts renumbered (2–9) with a new step 5 for the
`log_params`/plotting fix; step 3 now builds both models plus tests
1–7 and 10–12. Sequence: `PowFlex` (control) → `Pow2` + `Pow2Flat` →
`maxfev` → `log_params`/plotting → guarded L23 end-to-end → benchmark
→ docs → optional dispatch refactor. Five new questions (Q6–Q10) are in
Open Questions; the two I would most like answered before step 3 are
**Q6** (fix the walker perturbation, or just avoid zero seeds?) and
**Q9** (the `Pow2Flat` name, since it becomes public API).

**Useful precedent found:** `aNWChase.eval_adg` (`anw.py:1270-1275`)
already sums two `functions.exponential` calls over `params[...,:2]` /
`params[...,2:4]` — the exact slicing pattern the two-component bb model
needs (it hardcodes `self.wave`, so don't copy that part).

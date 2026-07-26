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

Everything below is in the `bing` package. Line numbers are as of this
writing.

- **`bing/models/bbnw.py`**
  - `init_model(model_name, wave, prior_dicts)` — the `model_dict`
    factory (`bbnw.py:76`). A new class must be added here.
  - `bbNWModel.__init__(wave, prior_dicts)` (`bbnw.py:200`) — calls
    `init_raman()` then `init_bbw()`, builds `self.priors` from
    `prior_dicts`, and asserts `len(self.pnames) == self.nparam`.
    Subclasses just forward to it; **do not** rebuild `bb_w`.
  - `bbNWModel.eval_bbnw(params, wave=None)` (`bbnw.py:240`) — the
    string dispatch. **Note the `wave` kwarg**: `eval_bb_ex`
    (`bbnw.py:288`) calls `eval_bbnw(params, wave=self.wave_ex)` to get
    backscatter at Raman *excitation* wavelengths. `Pow` honours it;
    the basis-function models (`Lee`, `GSM`) silently ignore it (a
    pre-existing inconsistency — do not copy it). **A new analytic
    model must honour `wave`.**
  - `eval_bb = bb_w + eval_bbnw` (`bbnw.py:275`).
  - `init_guess(bb_nw)` — returns a starting parameter vector with
    amplitudes in **linear** space (see the p0 contract below).
- **`bing/models/functions.py::powerlaw(wave, params, pivot=600.)`**
  (`functions.py:61`) — `10**params[...,0] · (pivot/λ)^params[...,1]`,
  via `np.outer`, so the return is always `(nsample, nwave)` even for a
  1-D parameter vector. Reuse it; it works on a 2-column *slice* of a
  wider parameter array (`params[..., 0:2]`) for both 1-D and 2-D input.
- **`bing/parameters/standard.py`** — the combo factories.
  `expb_pow()` (`standard.py:8`) is the template; note
  `bpriors[1] = dict(flavor='uniform', pmin=0., pmax=2.)` — **`beta ≥ 0`
  means the slope can only *decrease* with wavelength. That single line
  is the constraint this whole task is about.**
- **`bing/parameters/p_ntuple.py`** — `gen(**kwargs)` is model-agnostic;
  it just carries `model_names`, `apriors`, `bpriors`, etc. Nothing
  there needs changing for a new model, but `len(bpriors)` **must**
  equal the new model's `nparam`.
- **`bing/fitting/chisq_fit.py::fit(items, models, rt_dict, bounds)`**
  (`chisq_fit.py:42`) — wraps `scipy.optimize.curve_fit`; this is where
  `maxfev` goes.
- **`bing/fitting/l23.py`** — bounds for the chi-squared fit are built
  from the prior dicts' `pmin`/`pmax` (`l23.py:611-618`), and `p0`
  amplitudes are log10'd by **prior flavor**, not by position
  (`l23.py:385-390`).

### The two contracts that bite

1. **p0 is linear; the caller log10s by prior flavor.** `init_guess`
   returns amplitudes in linear units. Both BING (`l23.py:385`) and
   IOPtics (`ioptics/run.py::_log_mask`) then log10 exactly the slots
   whose prior `flavor` starts with `log`. So the **order of `pnames`
   must match the order of the prior dicts**, and each amplitude slot
   must carry a `log_*` prior while each exponent slot carries a linear
   one. Get this wrong and the fit silently starts in the wrong space.
2. **Chi-squared bounds come from `pmin`/`pmax`.** Both BING and
   IOPtics (`ioptics/run.py::_prior_bounds`) read `prior.pmin`/`.pmax`.
   So new parameters must use `uniform` or `log_uniform` priors —
   a `gaussian` prior has no `pmin` and will break the chi-squared path.

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

```
bb_nw(λ) = 10^B_min · (600/λ)^η_min  +  10^B_org · (600/λ)^η_org
```

- **Mineral term**: large amplitude, ~flat slope (η_min ≈ 0) — supplies
  the 0.1–0.4 m⁻¹ the red demands (Snyder 2008; Doxaran 2009;
  Neukermans 2012).
- **Organic term**: smaller amplitude, steeper slope — preserves the
  blue behaviour that already fits open-ocean spectra.

4 parameters: `['Bmin', 'eta_min', 'Borg', 'eta_org']`, pivot 600 nm
(BING's `Pow` convention).

Proposed priors:

| param | flavor | range | rationale |
|---|---|---|---|
| `Bmin` | log_uniform | −6 … 5 | as `Bnw` today |
| `eta_min` | uniform | −0.5 … 0.5 | flat/"white" mineral limit |
| `Borg` | log_uniform | −6 … 5 | as `Bnw` today |
| `eta_org` | uniform | 0.5 … 2 | open-ocean particle slope |

**Keep the two η ranges disjoint** (mineral ≤ 0.5 ≤ organic). The two
terms are otherwise exchangeable, and a symmetric prior would give the
posterior a label-switching degeneracy that wrecks MCMC interpretation
and makes the chi-squared covariance meaningless.

Implementation sketch (follow the real base class, not the skill):

```python
class bbNWPow2(bbNWModel):
    """Two-component particulate backscattering (mineral + organic).

    bb_nw(λ) = 10^Bmin·(600/λ)^eta_min + 10^Borg·(600/λ)^eta_org

    The near-flat mineral term supplies the large red-end backscatter
    of turbid, mineral-dominated water, which a single decreasing
    power law cannot produce without breaking the blue.

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
    pivot = 600.

    def __init__(self, wave:np.ndarray, prior_dicts:list):
        bbNWModel.__init__(self, wave, prior_dicts)

    def init_guess(self, bb_nw:np.ndarray):
        # Split the observed bb_nw at the pivot between the two
        # components; amplitudes stay LINEAR (the caller log10s the
        # log-flavored slots).
        ipiv = np.argmin(np.abs(self.wave - self.pivot))
        half = max(bb_nw[ipiv]/2., 1e-5)
        p0_bb = np.array([half, 0., half, 1.])
        assert p0_bb.size == self.nparam
        return p0_bb
```

and the dispatch branch in `bbNWModel.eval_bbnw`:

```python
        elif self.name == 'Pow2':
            # Mineral (flat) + organic (steep) power laws.  Slicing
            # params[..., 0:2] / [..., 2:4] keeps functions.powerlaw's
            # (nsample, nwave) contract for 1-D and 2-D input alike.
            return (functions.powerlaw(wave, params[..., 0:2],
                                       pivot=self.pivot) +
                    functions.powerlaw(wave, params[..., 2:4],
                                       pivot=self.pivot))
```

plus `'Pow2': bbNWPow2` in `init_model`'s `model_dict` and a
`standard.expb_pow2()` factory with the 4 `bpriors` above.

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
`maxfev` → `max_nfev` (verified in scipy 1.18,
`scipy/optimize/_minpack_py.py`), while the unbounded 'lm' branch passes
it to `leastsq`. Document the default (`None` = scipy's default) and
that raising it changes only *whether* LM returns, not fit quality —
on GLORIA a ~40× bump moved convergence 12.5% → 37.5% with no
improvement in misfit.

### 4. (Optional, ask first) refactor `eval_bbnw` to polymorphic dispatch

Adding a model currently means editing a string `if/elif` in the base
class. Cleaner: each subclass overrides `eval_bbnw(params, wave=None)`
and the base raises `NotImplementedError`. This touches all five
existing models, so treat it as a **separate, opt-in** step after
`Pow2` works and the tests are green — not bundled in.

## Tests

Create `bing/tests/test_bbnw.py` (the module does not exist yet; mirror
`bing/tests/test_anw.py`):

1. **Construction / registry.** `init_model('Pow2', wave)` returns a
   model with `nparam == 4`, `pnames` as specified, and
   `len(pnames) == nparam` (the base-class assert).
2. **Shape contract.** `eval_bbnw` returns `(1, nwave)` for a 1-D
   parameter vector and `(nsample, nwave)` for a chain-shaped array;
   all values finite and > 0.
3. **`wave` kwarg honoured.** `eval_bbnw(params, wave=model.wave_ex)`
   returns values on the excitation grid, and `eval_bb_ex(params)`
   equals `bb_w_ex + eval_bbnw(params, wave=wave_ex)`.
4. **Reduces to `Pow`.** With the organic amplitude driven to a
   negligible value (e.g. `Borg = -12`), `Pow2` matches `Pow` evaluated
   at `(Bmin, eta_min)` to tight tolerance. This is the regression guard
   that the new branch is a strict generalization.
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

Also confirm the existing suite still passes — `Pow2` is additive, so
`test_anw.py`, `test_l23_fitting.py`, `test_evaluate.py`, `test_raman.py`
should be untouched.

## Benchmark (dev/)

Add `dev/turbid_bbp/` with a script (plus figures) that answers "does
this actually help?":

- **Synthetic recovery.** Generate turbid-like Rrs from a *known* `Pow2`
  truth via `bing.evaluate.calc_Rrs_from_models`, add noise, and fit
  with `Pow`, `PowFlex`, and `Pow2`. Report recovered parameters and
  relative Rrs misfit per model. Expected: `Pow`/`PowFlex` plateau,
  `Pow2` recovers.
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
- **`ioptics/io.py` extracts a scalar named `beta`** from the fitted
  parameter names (`io.py:174`). `Pow2` has no `beta`, so that column
  will be NaN for it. Either accept that or decide (with JXP) whether
  one of the new exponents should be reported under `beta`.

## Prompts

1. Read this doc. **No code yet:** confirm the code map against the
   current source, flag anything I got wrong, and post your
   implementation plan plus questions in Open Questions / Logs.  Also note my answers to the open questions below.  Use Opus
2. Re-read this doc. Implement **item 1 (`PowFlex`)** —
   `standard.expb_powflex()` — plus its test. Log your work.
3. Re-read this doc. Implement **item 2 (`Pow2`)**: the class, the
   `init_model` entry, the `eval_bbnw` branch, `standard.expb_pow2()`,
   and the `test_bbnw.py` tests 1–7. Log your work.
4. Re-read this doc. Implement **item 3 (`maxfev`)** in
   `chisq_fit.fit` plus test 8. Log your work.
5. Re-read this doc. Add the end-to-end guarded test (test 9) and run
   the full suite. Log your work.
6. Re-read this doc. Build the **Benchmark** under `dev/turbid_bbp/`,
   examine the outputs, and make a recommendation on identifiability
   (MCMC vs inflated noise vs fixing `eta_min`). Log your work.
7. Re-read this doc. Do the **Docs** pass, including the CLAUDE.md and
   skill corrections. Log your work.
8. Re-read this doc. *Only if I approve it:* refactor `eval_bbnw` to
   polymorphic per-subclass dispatch (optional item 4). Log your work.

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

## Logging

The "Logs" section will record Claude's work. Please use the following
format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

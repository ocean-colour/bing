# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

BING (Bayesian INferences with Gordon coefficients) is a Python package for ocean color remote sensing analysis, specializing in bio-optical parameter retrieval through Bayesian inference. It implements Gordon's semi-analytical bio-optical models with MCMC sampling to estimate inherent optical properties (IOPs) from remote sensing reflectance (Rrs) measurements.

**Key Scientific Formula** (elastic; Gordon 1988):
`Rrs(λ) = G₁ * bb(λ) / (a(λ) + bb(λ)) + G₂ * [bb(λ) / (a(λ) + bb(λ))]²`

where `G1_STANDARD=0.0949`, `G2_STANDARD=0.0794` are the standard Gordon coefficients (constants in [bing/rt/rrs.py](bing/rt/rrs.py)). Wavelength-dependent G₁(λ)/G₂(λ) are also supported via `bing.rt.rrs.wave_dependent_gordon`. `a` is absorption, `bb` is backscattering. The full forward model can additionally include **Raman scattering** ([bing/rt/raman.py](bing/rt/raman.py)) and **chlorophyll fluorescence** ([bing/rt/chl_fl.py](bing/rt/chl_fl.py)) inelastic contributions.

## Development Commands

### Installation
```bash
# Development installation
pip install -e .

# With test dependencies
pip install -e .[dev]
```

### Testing
```bash
# Run all tests
pytest bing/tests/

# Run specific test file
pytest bing/tests/test_anw.py

# Run specific test function
pytest bing/tests/test_anw.py::test_init

# Run with verbose output
pytest -v bing/tests/
```

### Using the CLI
```bash
# Fit Rrs data from a CSV table
bing_fit_Rrs input_table.csv Exp,Pow --outroot output --satellite PACE --fit_method mcmc

# Input CSV format: wave,Rrs,sigRrs,[optional: anw,bbnw]
```

## Architecture Overview

### Core Data Flow
```
Rrs measurements (satellite/in-situ)
    ↓
[Bio-optical Models] (models/)
    ├─ Absorption: a(λ) = a_w(λ) + a_nw(λ)
    └─ Backscattering: bb(λ) = bb_w(λ) + bb_nw(λ)
    ↓
[Radiation Transfer] (rt/)
    ├─ Elastic Gordon (rt/rrs.py): Rrs from a, bb
    ├─ Raman scattering correction (rt/raman.py)
    └─ Chlorophyll fluorescence (rt/chl_fl.py)
    ↓
[Fitting Algorithms] (fitting/)
    ├─ MCMC via emcee (inference.py) - full posterior
    └─ Least-squares (chisq_fit.py) - initial guess
    ↓
[Analysis] (evaluate.py)
    ├─ Extract statistics from chains/fits
    └─ Reconstruct a, bb with uncertainties
    ↓
Fitted IOPs + Uncertainties
```

### Module Organization

**[bing/models/](bing/models/)** - Bio-optical model implementations
- `anw.py` (~1345 lines): Absorption models (Cst, Every, Exp, ExpFix, Bricaud, ExpBricaud, ExpBricaudFix, ExpBricaudFree, GIOP, ExpNMF, GSM, Chase, ChaseMini). Module-level `init_model(name, wave, prior_dicts)` is the entry point.
- `bbnw.py` (~638 lines): Backscattering models (Cst, Every, Pow, GSM, Lee). Also exposes `init_model(name, wave, prior_dicts)`.
- `functions.py`: Generic spectral basis functions (exponential, power-law, Gaussian)
- `utils.py`: Top-level `init(model_names, wave)` builds both models in a single call

**[bing/rt/](bing/rt/)** - Radiation transfer (subpackage)
- `rrs.py`: Gordon elastic model (`calc_Rrs(a, bb)`), wavelength-dependent Gordon coefficients, fluorescence-aware Rrs builders, and `A_Rrs=0.52`/`B_Rrs=1.7` conversion constants
- `raman.py`: Raman scattering coefficients/redistribution + `calc_Rrs_with_raman`
- `chl_fl.py`: Low-level chlorophyll fluorescence emission (`calc_R_fluorescence`, `calc_fluorescence_line_height`, `fluorescence_backscattering_coeff`)
- `defs.py`: Shared definitions/constants for the subpackage

**[bing/fitting/](bing/fitting/)** - Parameter estimation algorithms
- `inference.py` (~427 lines): MCMC sampling with emcee (`log_prob`, `run_emcee`, `fit_one`, `fit_batch`, `init_mcmc`)
- `chisq_fit.py`: Least-squares optimization via `scipy.optimize.curve_fit`
- `l23.py`: Specialized fitting for Loisel et al. 2023 synthetic dataset

**[bing/parameters/](bing/parameters/)** - Model configuration system
- `standard.py`: Pre-configured model combinations (`expb_pow`, `giop`, `gsm`, etc.)
- `p_ntuple.py`: Named-tuple generator for complete fitting configurations

**[bing/priors/](bing/priors/)** - Bayesian prior distributions
- `priors.py`: Prior classes (`LogUniformPrior`, `UniformPrior`, `GaussianPrior`, `RatioPrior`)
- `adg.py`: Special handling for `a_dg` (dissolved + detrital absorption)

**[bing/](bing/)** - Core utilities
- `evaluate.py`: Post-fitting analysis (`calc_stats`, `reconstruct_from_chains`)
- `plotting.py`: Spectral fit visualization with uncertainties
- `noise.py`: Satellite-specific noise modeling (PACE, MODIS, SeaWiFS, SBG)
- `stats.py`: Chi-squared and information criteria calculations
- `preproc.py`: Wavelength interpolation to satellite bands

## Critical Implementation Details

### Parameter Convention
**All model parameters are stored and fitted in log10 space** for numerical stability and prior range handling. When implementing new models or working with parameters:

```python
# Parameters come in as log10
params = [-1.2, 0.015, 0.3]  # Example: log10(Adg), Sdg, log10(Aph)

# Models convert internally for evaluation
Adg = 10**params[0]  # Convert to linear space
Sdg = params[1]       # Spectral slopes stay linear
```

### Two-Model Architecture
BING always fits pairs of models: `[absorption_model, backscattering_model]`. Parameter arrays are split:

```python
aparams = params[:models[0].nparam]      # First N params for absorption
bparams = params[models[0].nparam:]      # Remaining params for backscattering
```

When adding new models, ensure `nparam` and `pnames` attributes are set correctly.

### Model Evaluation Shape Convention
All model evaluation functions return arrays of shape `(nsample, nwave)` for batch processing:

```python
# Single spectrum evaluation
a_nw = model.eval_anw(params)  # Shape: (1, nwave)

# Batch evaluation (e.g., from MCMC chains)
chains = np.array([[...], [...], ...])  # Shape: (nsamples, nparams)
a_nw = model.eval_anw(chains)           # Shape: (nsamples, nwave)
```

### Water Component Initialization
Water absorption and backscattering are computed once at model initialization and stored as `model.a_w` and `model.bb_w`. Total IOPs = water + non-water:

```python
a_total = model.a_w + a_nw     # Total absorption
bb_total = model.bb_w + bb_nw  # Total backscattering
```

Water components use IOCCG reference data and are interpolated to model wavelengths.

### Chlorophyll Treatment
Models with chlorophyll dependency (e.g., ExpBricaud, Bricaud) require:

1. Setting phytoplankton absorption coefficients: `model.set_aph(Chl)`
2. Chlorophyll derived from fitted amplitude: `Chl = 10^param / 0.05582`
3. Uses Bricaud et al. (1995) coefficients interpolated to model wavelengths

### Prior Handling
Priors are attached to models and return log-probabilities:

```python
prior_dict = {'flavor': 'log_uniform', 'pmin': -6, 'pmax': 5}
prior_dicts = [prior_dict] * model.nparam
model.priors = bing_priors.Priors(prior_dicts)

log_prior = model.priors.calc(params)  # Returns -np.inf if out of bounds
```

### MCMC Workflow Pattern
Standard fitting workflow for a single spectrum:

```python
# 1. Configuration
params = standard.expb_pow(satellite='PACE', nsteps=40000, nburn=1000)
models = model_utils.init(params.model_names, wavelengths)

# 2. Set priors
for jj in range(2):
    prior_dicts = [bing_priors.default] * models[jj].nparam
    models[jj].priors = bing_priors.Priors(prior_dicts)

# 3. Initial guess via least-squares
p0_best, cov, _ = chisq_fit.fit(
    (Rrs, varRrs, p0_init, 0), models, bounds=(pmin, pmax))

# 4. MCMC refinement
pdict = inference.init_mcmc(models, nsteps=40000, nburn=1000)
chains, idx = inference.fit_one(
    (Rrs, varRrs, p0_best, 0), models=models, pdict=pdict, chains_only=True)

# 5. Analysis
stats = evaluate.calc_stats(chains, models[0].pnames + models[1].pnames)
a, bb, a_lo, a_hi, bb_lo, bb_hi, Rrs_pred, sigRrs = \
    evaluate.reconstruct_from_chains(models, chains, perc=(5, 95))
```

### Batch Processing Pattern
For large-scale processing, use parallel batch fitting:

```python
# Prepare items as list of tuples
items = [(Rrs[i], varRrs[i], p0[i], i) for i in range(n_spectra)]

# Parallel fitting
chains_list, indices = inference.fit_batch(
    models, pdict, items, n_cores=10)

# Results preserve indices for assembly
for chains, idx in zip(chains_list, indices):
    process_results(chains, idx)
```

## Model System

### Available Absorption Models ([bing/models/anw.py](bing/models/anw.py))

| Model (class) | Name | Params | Description / Use Case |
|---|---|---|---|
| `aNWCst` | `Cst` | 1 | Spectrally flat — sanity / null baseline |
| `aNWEvery` | `Every` | Nwave | Free amplitude per wavelength — non-parametric |
| `aNWExp` | `Exp` | 2 | Dissolved/detrital exponential with free slope |
| `aNWExpFix` | `ExpFix` | 1 | Exponential with fixed slope |
| `aNWBricaud` | `Bricaud` | 1 | Pure phytoplankton via Bricaud (1995) |
| `aNWExpBricaud` | `ExpBricaud` | 3 | Adg·exp(-Sdg·λ) + Aph(Chl) — CDOM + phytoplankton, the standard combo |
| `aNWExpBricaudFix` | `ExpBricaudFix` | 2 | ExpBricaud with Sdg fixed |
| `aNWExpBricaudFree` | `ExpBricaudFree` | 3 | ExpBricaud with Chl as a free parameter |
| `aNWGIOP` | `GIOP` | 2 | GIOP framework (Werdell et al. 2013) — satellite-processing standard |
| `aNWGSM` | `GSM` | 2 | Garver-Siegel-Maritorena |
| `aNWExpNMF` | `ExpNMF` | varies | Exponential + NMF basis for phytoplankton |
| `aNWChase` | `Chase` | varies | Chase et al. multi-Gaussian phytoplankton + Exp |
| `aNWChaseMini` | `ChaseMini` | fewer | Reduced-basis Chase variant |

### Available Backscattering Models ([bing/models/bbnw.py](bing/models/bbnw.py))

| Model (class) | Name | Params | Description / Use Case |
|---|---|---|---|
| `bbNWCst` | `Cst` | 1 | Spectrally flat — simple particle model |
| `bbNWEvery` | `Every` | Nwave | Free amplitude per wavelength — non-parametric |
| `bbNWPow` | `Pow` | 2 | Bnw · (600/λ)^β — power-law particles (most common) |
| `bbNWGSM` | `GSM` | 1 | Fixed spectral slope (1.0337 at 443 nm) |
| `bbNWLee` | `Lee` | 1 | Bnw · (600/λ)^Y(Rrs) — Lee et al. 2002 dynamic slope (requires `set_basis_func(Y)`) |
| `bbNWPow2` | `Pow2` | 4 | Bmin · (700/λ)^η_min + Borg · (600/λ)^η_org — two-component mineral + organic, for turbid water |
| `bbNWPow2Flat` | `Pow2Flat` | 3 | Bmin + Borg · (600/λ)^η_org — as `Pow2` with η_min fixed at 0 |

**Turbid water**: prefer `Pow2Flat` with MCMC. The 4-parameter `Pow2` is
degenerate under chi-squared against realistic in-situ noise (Bmin/Borg
anti-correlated at −1.00, condition number ~1e7), and an inflated noise
floor makes recovery *worse*, not better. Raise `maxfev` for either — at
scipy's default budget they fail to converge on many spectra. Benchmark:
[dev/turbid_bbp/turbid_bbp.py](dev/turbid_bbp/turbid_bbp.py); notebooks in
[nb/TurbidWaters/](nb/TurbidWaters/).

### Standard Model Combinations (bing/parameters/standard.py)

```python
# Most commonly used
params = standard.expb_pow(satellite='PACE')  # ExpBricaud + Power-law

# Other configurations
params = standard.giop()      # GIOP + Lee
params = standard.gsm()       # GSM + GSM
params = standard.k2b()       # Bricaud + Constant

# Turbid / mineral-dominated water
params = standard.expb_pow2()      # ExpBricaud + Pow2 (4 bb params)
params = standard.expb_pow2flat()  # ExpBricaud + Pow2Flat (3 bb params)
params = standard.expb_powflex()   # ExpBricaud + Pow, beta free to go
                                   # negative: the control experiment
```

### Which parameters are log10 (`log_params`)

Amplitudes are held in log10, exponents/slopes in linear space. Models
declare this as a `log_params` list of booleans (e.g. `[True, False,
True, False]` for `Pow2`); `None` means all-log10, the historical
default. **Display code** reads it via `bing.plotting.log_param_mask` so
figures only exponentiate and log-label the parameters that really are
log10. It is deliberately *not* used by the p0 conversion in the
fitters, which keys on the prior flavor (`log_uniform` vs `uniform`) —
so keep the two consistent when adding a model. (`Chase2017` is the
documented exception: all its parameters are log10 but its priors are
`uniform` over log10 bounds.)

## Adding New Models

### New Absorption Model Template
```python
class aNWYourModel(aNWModel):
    def __init__(self, wave, prior_dicts=None):
        super().__init__(wave)
        self.nparam = 2  # Number of parameters
        self.pnames = ['param1', 'param2']
        self.uses_Chl = False  # Set True if chlorophyll-dependent

        if prior_dicts is not None:
            self.priors = Priors(prior_dicts)

    def eval_anw(self, params):
        """Evaluate non-water absorption.

        Args:
            params: Parameters (log10 if amplitude, linear if slope/exponent)
                    Shape: (nparam,) or (nsamples, nparam)

        Returns:
            a_nw: Non-water absorption, shape (nsamples, nwave)
        """
        params = np.atleast_2d(params)
        nsample = params.shape[0]

        # Your model implementation
        # Remember: amplitudes are log10, so convert with 10**param
        amplitude = 10**params[:, 0:1]  # Shape (nsample, 1)
        slope = params[:, 1:2]

        # Vectorized calculation
        a_nw = amplitude * np.exp(-slope * (self.wave - 440))

        return a_nw  # Shape (nsample, nwave)
```

### New Backscattering Model Template

Two things differ from the absorption side, and both bite:

1. The base `bbNWModel.__init__` takes `(wave, prior_dicts)` and builds
   the priors itself — do **not** call `super().__init__(wave)` and do
   not rebuild `bb_w`.
2. Implement **`_eval_bbnw(params, wave)`**, not `eval_bbnw`. The public
   `eval_bbnw(params, wave=None)` lives on the base class, resolves
   `wave=None` to `self.wave`, and delegates — so your method always
   receives a real grid and must use it (`eval_bb_ex` passes the Raman
   *excitation* wavelengths). The base `_eval_bbnw` raises
   `NotImplementedError`, so a model that forgets fails loudly.

```python
class bbNWYourModel(bbNWModel):
    """One-line description: bb_nw(λ) = <equation>."""
    name = 'YourModel'          # the init_model key
    nparam = 2
    pnames = ['Bnw', 'exponent']
    log_params = [True, False]  # which slots are log10 amplitudes
    pivot = 600.
    uses_basis_params = False   # True if it needs set_basis_func(...)

    # prior_dicts defaults to None so tests can construct directly
    def __init__(self, wave, prior_dicts=None):
        bbNWModel.__init__(self, wave, prior_dicts)

    def _eval_bbnw(self, params, wave):
        """bb_nw on the GIVEN grid, shape (nsample, nwave).

        Use `wave`, never self.wave: eval_bb_ex passes the Raman
        excitation wavelengths.  functions.powerlaw/constant/gen_basis
        already return (nsample, nwave) for 1-D and chain-shaped params.
        """
        return functions.powerlaw(wave, params, pivot=self.pivot)

    def init_guess(self, bb_nw):
        """Starting parameters, amplitudes in LINEAR space.

        The caller (bing.fitting.l23, ioptics.run) log10s the slots
        whose prior flavor starts with 'log'.  Never seed a parameter at
        exactly 0: the MCMC walker ball would have zero spread there and
        that dimension would never move.
        """
        i_piv = np.argmin(np.abs(self.wave - self.pivot))
        return np.array([max(bb_nw[i_piv], 1e-5), 1.])
```

Then register the class:

```python
    model_dict = {..., 'YourModel': bbNWYourModel}
```

Finally add a `standard.<combo>()` factory that passes `bpriors`
explicitly (one dict per parameter, `log_uniform` for amplitudes and
`uniform` for linear exponents). The base class asserts
`len(pnames) == nparam` and, via `check_priors()`, that the attached
priors match `nparam` — a mismatch otherwise corrupts p0 silently.

## Data Sources and External Dependencies

### Satellite Integration (via ocpy package)
- **PACE OCI**: Hyperspectral, 5nm resolution (400-700nm)
- **MODIS Aqua**: Multispectral, selected bands
- **SeaWiFS**: Historical ocean color sensor
- **SBG**: Future hyperspectral mission

Noise specifications loaded via `ocpy.satellites.{pace,modis,seawifs}.gen_noise_vector(wavelengths)`

### Argo BGC Integration
Processing matched satellite-float observations:
```python
matched = pd.read_csv('matched_argo_bgc_profiles_bbp.csv')
for idx, profile in matched.iterrows():
    m_fitting.doit(profile, f"fits/Argo_{profile.cruise}_{profile.profile:03d}.npz")
```

### Loisel et al. (2023) Synthetic Dataset
Hydrolight radiative transfer simulations for validation:
```python
from bing.fitting import l23

# Load synthetic spectrum with true IOPs
data_dict = l23.load_one_l23(idx=170, step=1, ds=None, wv_max=700, wv_min=400)
# Returns: wave, Rrs, a, bb, Chl, Sdg, Y (true parameters)

# Fit and compare to truth
chains = l23.fit_one(params, idx=170)
```

## Testing

### Test Structure ([bing/tests/](bing/tests/))
- `test_anw.py`: Absorption model validation (initialization, evaluation, priors, batch processing)
- `test_l23_fitting.py`: Loisel et al. 2023 synthetic-dataset fitting validation
- `test_raman.py`: Raman scattering coefficients and `calc_Rrs_with_raman`
- `test_chl_fl.py`: Chlorophyll fluorescence emission model
- `files/`: Reference inputs for tests

### Running Specific Tests
```bash
# Test all absorption models
pytest bing/tests/test_anw.py -v

# Test inelastic processes
pytest bing/tests/test_raman.py
pytest bing/tests/test_chl_fl.py

# Test L23 fitting end-to-end
pytest bing/tests/test_l23_fitting.py

# Test a specific function
pytest bing/tests/test_anw.py::test_expnmf
```

When adding a new absorption or backscattering model, extend `test_anw.py` / add a peer file: cover (a) instantiation via `init_model`, (b) shape-correct `eval_*` on both single and batched params, (c) prior attachment, and (d) round-trip through `bing.rt.rrs.calc_Rrs`.

## Data Formats

### Input CSV for fit_Rrs.py
```csv
wave,Rrs,sigRrs,anw,bbnw
400,0.010,0.0005,0.045,0.0021
405,0.011,0.0006,0.043,0.0020
...
```
Required: `wave`, `Rrs`. Optional: `sigRrs` (if not simulating satellite), `anw`, `bbnw` (for initial guess).

### Output NPZ Structure
```python
results = {
    'wavelength': np.array([...]),        # Model wavelengths
    'Rrs': np.array([...]),              # Measured Rrs
    'sigRrs': np.array([...]),           # Uncertainties
    'chains': np.array([...]),           # MCMC chains (nsamples, nparams)
    'ans': np.array([...]),              # Least-squares best-fit (if chisq method)
    'cov': np.array([...]),              # Covariance matrix (if chisq method)
}
```

## Common Pitfalls

1. **Forgetting log10 conversion**: Amplitudes are in log10 space. Always use `10**param` for linear values; slopes/exponents stay linear.

2. **Shape mismatches**: Models expect 2D parameter arrays. Use `np.atleast_2d(params)` for single spectra; `eval_anw` / `eval_bbnw` always return `(nsample, nwave)`.

3. **Prior out-of-bounds**: MCMC returns `-np.inf` for invalid priors. Check prior ranges match parameter scales (log10 vs. linear).

4. **Water components**: Don't reinitialize `a_w` and `bb_w` during fitting — they're computed once at model creation.

5. **Wavelength correspondence**: Always initialize models with the same wavelength array used for Rrs data. The `rt` submodules and noise generators must use that grid too.

6. **Chlorophyll models**: If `model.uses_Chl == True`, must call `model.set_aph(Chl)` before evaluation.

7. **Lee backscatter model**: If `model.uses_basis_params == True`, must call `model.set_basis_func(Y)` before evaluation.

8. **`rt` is a package, not a module**: Import as `from bing.rt import rrs` (or `from bing.rt.rrs import calc_Rrs`). Old code that did `from bing import rt; rt.calc_Rrs(...)` still works via re-exports in `bing/rt/__init__.py`, but new code should target the submodule directly.

9. **Inelastic contributions are opt-in**: Standard `calc_Rrs(a, bb)` is purely elastic. To include Raman / fluorescence use `calc_Rrs_with_raman` or `calc_Rrs_with_fluorescence` from `bing.rt` — and remember to add their parameters to the fit (or fix them) so priors and `nparam` stay consistent.

10. **Wavelength-dependent G₁/G₂**: If you call `wave_dependent_gordon(wave)`, pass the same `wave` grid the model uses. Mismatched grids silently extrapolate.

## Working in this repo

- **Branch context**: ongoing work lives on feature branches like `more_Gordon`; the working area for that effort is [dev/Gordon/](dev/Gordon/). Look there before adding new exploration notebooks.
- **Notebooks vs. package code**: scratch and figures go in `dev/`, `nb/`, or `papers/*/Analysis/`. Anything reusable belongs in the `bing/` package with a matching test.
- **Don't widen scope**: bug fixes and new models should land as focused changes. Don't refactor the Gordon model, prior system, or `eval_*` shape contract opportunistically — they're load-bearing for every consumer in `papers/`.
- **External dep boundary**: noise specs and satellite band definitions live in the sibling [ocpy](../ocpy/) package (`ocpy.satellites.*`). Don't duplicate that data in `bing/`.

## Repository Structure

```
bing/
├── models/          # Bio-optical models (absorption, backscattering)
├── rt/              # Radiation transfer subpackage
│   ├── rrs.py       #   Gordon elastic Rrs + fluorescence Rrs builders
│   ├── raman.py     #   Raman scattering
│   ├── chl_fl.py    #   Chlorophyll fluorescence
│   └── defs.py      #   Shared constants
├── fitting/         # MCMC and least-squares algorithms
├── parameters/      # Model configuration system
├── priors/          # Bayesian prior distributions
├── scripts/         # Command-line interface scripts (fit_Rrs.py)
├── tests/           # Unit tests (anw, l23, raman, chl_fl)
├── data/            # Reference data (water IOPs, phytoplankton coefficients)
├── evaluate.py      # Post-fitting analysis
├── plotting.py      # Visualization
├── noise.py         # Satellite noise modeling
├── stats.py         # Statistical utilities
└── preproc.py       # Preprocessing utilities

bin/                 # Executable scripts
└── bing_fit_Rrs     # Main CLI entry point

papers/              # Research applications (data + analysis scripts)
├── biomass/         #   PACE-Argo BGC validation
├── phytoplankton/   #   Model-comparison study
├── bing_2.0/        #   Large-scale synthetic dataset benchmarking
└── Solar/           #   Solar-induced fluorescence / inelastic studies

dev/                 # Active development scratch (e.g. dev/Gordon/)
docs/                # Sphinx documentation
nb/                  # Jupyter notebooks for development
prompts/             # Project-specific Claude prompts and skills
```

## Key References

- Gordon et al. (1988): Semi-analytical model formulation
- Bricaud et al. (1995): Phytoplankton absorption parameterization
- Lee et al. (2002): Backscattering and QAA algorithm
- Werdell et al. (2013): GIOP framework
- Loisel et al. (2023): Synthetic Hydrolight dataset
- Foreman-Mackey et al. (2013): emcee MCMC implementation

## Documentation

Full documentation: https://oc-bing.readthedocs.io
- Installation guide
- API reference
- Tutorials with examples
- Model descriptions
- Algorithm details

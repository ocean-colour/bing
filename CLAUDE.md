# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

BING (Bayesian INferences with Gordon coefficients) is a Python package for ocean color remote sensing analysis, specializing in bio-optical parameter retrieval through Bayesian inference. It implements Gordon's semi-analytical bio-optical models with MCMC sampling to estimate inherent optical properties (IOPs) from remote sensing reflectance (Rrs) measurements.

**Key Scientific Formula**: `Rrs(λ) = G₀ * bb(λ) / (a(λ) + bb(λ)) + G₁ * [bb(λ) / (a(λ) + bb(λ))]²`

where G₀=0.0949, G₁=0.0794 are Gordon coefficients, a is absorption, and bb is backscattering.

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
[Gordon Formula] (rt.py)
    ├─ Calculates model Rrs from a and bb
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

**bing/models/** - Bio-optical model implementations
- `anw.py` (1019 lines): Absorption models (ExpBricaud, GIOP, GSM, Bricaud, etc.)
- `bbnw.py` (368 lines): Backscattering models (Power-law, Lee, GSM, etc.)
- `functions.py`: Generic spectral basis functions (exponential, power-law, Gaussian)
- `utils.py`: Model initialization and configuration

**bing/fitting/** - Parameter estimation algorithms
- `inference.py`: MCMC sampling with emcee (log_prob, run_emcee, fit_one, fit_batch)
- `chisq_fit.py`: Least-squares optimization via scipy.optimize.curve_fit
- `l23.py`: Specialized fitting for Loisel et al. 2023 synthetic dataset

**bing/parameters/** - Model configuration system
- `standard.py`: Pre-configured model combinations (expb_pow, giop, gsm, etc.)
- `p_ntuple.py`: Named tuple generator for complete fitting configurations

**bing/priors/** - Bayesian prior distributions
- `priors.py`: Prior classes (LogUniformPrior, UniformPrior, GaussianPrior, RatioPrior)
- `adg.py`: Special handling for a_dg (dissolved + detrital absorption)

**bing/** - Core utilities
- `rt.py`: Gordon's radiation transfer model (`calc_Rrs(a, bb)`)
- `evaluate.py`: Post-fitting analysis (calc_stats, reconstruct_from_chains)
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

### Available Absorption Models (bing/models/anw.py)

| Model | Params | Description | Use Case |
|-------|--------|-------------|----------|
| **ExpBricaud** | 3 | Adg*exp(-Sdg*λ) + Aph(Chl) | Standard mixed absorption (CDOM + phytoplankton) |
| **ExpBricaudFree** | 3 | Like ExpBricaud but Chl as free parameter | When chlorophyll not constrained |
| **GIOP** | 2 | Standard GIOP algorithm | Industry standard for satellite processing |
| **GSM** | 2 | Garver-Siegel-Maritorena | Alternative semi-analytical model |
| **Bricaud** | 1 | Pure phytoplankton absorption | Phytoplankton-dominated waters |
| **Exp** | 2 | Exponential with free slope | Dissolved/detrital matter only |

### Available Backscattering Models (bing/models/bbnw.py)

| Model | Params | Description | Use Case |
|-------|--------|-------------|----------|
| **Pow** | 2 | Bnw * (600/λ)^beta | Power-law particles (most common) |
| **Lee** | 1 | Bnw * (600/λ)^Y(Rrs) | Lee et al. 2002 with dynamic slope |
| **GSM** | 1 | Bnw * (443/λ)^1.0337 | Fixed spectral slope |
| **Cst** | 1 | Spectrally flat | Simple particle model |

### Standard Model Combinations (bing/parameters/standard.py)

```python
# Most commonly used
params = standard.expb_pow(satellite='PACE')  # ExpBricaud + Power-law

# Other configurations
params = standard.giop()      # GIOP + Lee
params = standard.gsm()       # GSM + GSM
params = standard.k2b()       # Bricaud + Constant
```

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
```python
class bbNWYourModel(bbNWModel):
    def __init__(self, wave, prior_dicts=None):
        super().__init__(wave)
        self.nparam = 2
        self.pnames = ['Bnw', 'exponent']
        self.uses_basis_params = False  # Set True if needs dynamic parameters

        if prior_dicts is not None:
            self.priors = Priors(prior_dicts)

    def eval_bbnw(self, params):
        """Evaluate non-water backscattering."""
        params = np.atleast_2d(params)

        # Your implementation
        amplitude = 10**params[:, 0:1]
        exponent = params[:, 1:2]

        bb_nw = amplitude * (600.0 / self.wave)**exponent

        return bb_nw
```

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

### Test Structure
- `test_anw.py`: Absorption model validation (initialization, evaluation, priors, batch processing)
- `test_inference.py`: MCMC workflow testing
- `test_l23_fitting.py`: Loisel dataset validation

### Running Specific Tests
```bash
# Test all absorption models
pytest bing/tests/test_anw.py -v

# Test MCMC inference
pytest bing/tests/test_inference.py

# Test specific model
pytest bing/tests/test_anw.py::test_expnmf
```

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

1. **Forgetting log10 conversion**: Amplitudes are in log10 space. Always use `10**param` for linear values.

2. **Shape mismatches**: Models expect 2D parameter arrays. Use `np.atleast_2d(params)` for single spectra.

3. **Prior out-of-bounds**: MCMC returns `-np.inf` for invalid priors. Check prior ranges match parameter scales.

4. **Water components**: Don't reinitialize `a_w` and `bb_w` during fitting - they're computed once at model creation.

5. **Wavelength correspondence**: Always initialize models with the same wavelength array used for Rrs data.

6. **Chlorophyll models**: If `model.uses_Chl == True`, must call `model.set_aph(Chl)` before evaluation.

7. **Lee backscatter model**: If `model.uses_basis_params == True`, must call `model.set_basis_func(Y)` before evaluation.

## Repository Structure

```
bing/
├── models/          # Bio-optical models (absorption, backscattering)
├── fitting/         # MCMC and least-squares algorithms
├── parameters/      # Model configuration system
├── priors/          # Bayesian prior distributions
├── scripts/         # Command-line interface scripts
├── tests/           # Unit tests
├── data/            # Reference data (water IOPs, phytoplankton coefficients)
├── rt.py            # Gordon radiation transfer
├── evaluate.py      # Post-fitting analysis
├── plotting.py      # Visualization
├── noise.py         # Satellite noise modeling
├── stats.py         # Statistical utilities
└── preproc.py       # Preprocessing utilities

bin/                 # Executable scripts
├── bing_fit_Rrs     # Main CLI entry point

papers/              # Research applications
├── biomass/         # PACE-Argo BGC validation
├── phytoplankton/   # Model comparison study
└── bing_2.0/        # Large-scale synthetic dataset benchmarking

docs/                # Sphinx documentation
nb/                  # Jupyter notebooks for development
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

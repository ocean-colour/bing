# BING - Bayesian INferences with Gordon coefficients

[![Documentation Status](https://readthedocs.org/projects/bing/badge/?version=latest)](https://bing.readthedocs.io/en/latest/?badge=latest)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

The **Bayesian INferences with Gordon coefficients (BING)** package is a comprehensive Python toolkit for ocean color remote sensing analysis, specializing in bio-optical parameter retrieval through Bayesian inference methods. BING implements Gordon's semi-analytical bio-optical models with advanced statistical fitting techniques, with particular emphasis on NASA's PACE (Plankton, Aerosol, Cloud, ocean Ecosystem) mission data.

## 🌊 Key Features

- **🛰️ Satellite Data Processing**: Native support for PACE OCI, MODIS, SeaWiFS, and future SBG missions
- **🔬 Bio-optical Modeling**: Implementation of Gordon's formulations with multiple absorption and backscattering models
- **📊 Bayesian Inference**: MCMC sampling with emcee for full posterior distributions
- **🎯 Parameter Retrieval**: Robust estimation of IOPs from remote sensing reflectance
- **🌐 Argo BGC Integration**: Tools for matching satellite observations with BGC-Argo float profiles  
- **📈 Uncertainty Quantification**: Comprehensive error propagation and statistical analysis
- **🎨 Visualization**: Publication-ready plotting utilities for spectral fits and residuals
- **⚡ Performance**: Parallel processing support for large-scale ocean color analysis

## 📚 Documentation

**Full documentation is available at [https://oc-bing.readthedocs.io](https://oc-bing.readthedocs.io)**

The documentation includes:
- [Getting Started Guide](https://bing.readthedocs.io/en/latest/getting_started.html)
- [Installation Instructions](https://bing.readthedocs.io/en/latest/installation.html)
- [API Reference](https://bing.readthedocs.io/en/latest/api/index.html)
- [Tutorials](https://bing.readthedocs.io/en/latest/tutorials/index.html)
- [Model Descriptions](https://bing.readthedocs.io/en/latest/models.html)

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (when available)
pip install bing-ocean

# Or install from source
git clone https://github.com/yourusername/bing.git
cd bing
pip install -e .
```

### Basic Usage

```python
import numpy as np
from bing.parameters import standard
from bing.models import utils as model_utils
from bing.fitting import chisq_fit, inference

# Define wavelengths
wavelengths = np.arange(400, 701, 5)

# Load standard parameter configuration
params = standard.expb_pow(satellite='PACE')

# Initialize bio-optical models
models = model_utils.init(params.model_names, wavelengths)

# Load your Rrs data (example with synthetic data)
Rrs_measured = np.array([...])  # Your measured Rrs
Rrs_uncertainty = np.array([...])  # Measurement uncertainties

# Perform least-squares fitting
result = chisq_fit.fit(models, wavelengths, Rrs_measured, Rrs_uncertainty)
print(f"Best-fit parameters: {result['x']}")

# Run MCMC for uncertainty quantification
pdict = inference.init_mcmc(models, nsteps=10000, nburn=1000)
chains = inference.fit_one((Rrs_measured, Rrs_uncertainty, result['x'], 0), 
                          models=models, pdict=pdict, chains_only=True)
```

## 🔬 Bio-optical Models

BING implements various bio-optical models based on Gordon's formulation:

### Gordon's Semi-Analytical Model

The core of BING is based on Gordon et al.'s relationship:

```
Rrs(λ) = G₀ * bb(λ) / (a(λ) + bb(λ)) + G₁ * [bb(λ) / (a(λ) + bb(λ))]²
```

Where G₀ and G₁ are the Gordon coefficients, typically 0.0949 and 0.0794 respectively.

### Absorption Models
- **ExpBricaud**: Exponential model with chlorophyll dependency (Bricaud et al., 1995)
- **GIOP**: Generalized IOP framework (Werdell et al., 2013)
- **GSM**: Garver-Siegel-Maritorena semi-analytical model
- **QAA**: Quasi-Analytical Algorithm (Lee et al., 2002)

### Backscattering Models
- **PowerLaw**: Power-law spectral dependency
- **Lee**: Lee et al. (2002) formulation
- **Constant**: Spectrally flat backscattering

## 🛠️ Advanced Features

### PACE Data Processing

```python
from ocpy.pace import io as pace_io
import fitting as m_fitting

# Load PACE OCI Level 2 data
xds, flags = pace_io.load_oci_l2('path/to/PACE_OCI.L2.nc')

# Extract and fit Rrs at specific location
lat, lon = 25.0, -80.0
rrs_data = extract_pace_rrs(xds, lat, lon)

# Process with BING
m_fitting.doit(rrs_data, 'output.npz')
```

### Argo BGC Matching

```python
import pandas as pd

# Load matched Argo profiles
matched = pd.read_csv('matched_argo_bgc_profiles_bbp.csv')

# Process matched data
for idx, profile in matched.iterrows():
    outfile = f"fits/Argo_{profile.cruise}_{profile.profile:03d}_fits.npz"
    m_fitting.doit(profile, outfile, nclosest=5)
```

### Loisel et al. (2023) Synthetic Dataset

BING includes specialized routines for working with the Loisel et al. (2023) synthetic dataset:

```python
from bing.fitting import l23
from bing.parameters import standard

# Configure for L23 dataset
params = standard.expb_pow(satellite='PACE', nsteps=40000)

# Fit synthetic spectrum
idx = 170  # Spectrum index
chains, models, prep_dict, idx, extras = l23.fit_one(params, idx)

# Save results
outfile = l23.chain_filename(params, idx=idx)
l23.save_chains(chains, idx, outfile, extras=extras)
```

## 📊 Outputs

BING provides comprehensive outputs including:

- **Fitted Parameters**: IOPs (absorption, backscattering coefficients)
- **Derived Products**: Chlorophyll-a, CDOM, particulate backscattering
- **Uncertainties**: Full posterior distributions from MCMC
- **Diagnostics**: χ², residuals, convergence metrics
- **Visualizations**: Spectral fits, corner plots, residual analysis

Example output structure:
```python
results = {
    'wavelength': np.array([...]),      # Wavelengths (nm)
    'Rrs': np.array([...]),             # Measured Rrs
    'Rrs_unc': np.array([...]),         # Uncertainties
    'fitted_params': np.array([...]),   # Best-fit parameters
    'chains': np.array([...]),          # MCMC chains
    'statistics': {
        'chi2': float,                   # Chi-squared
        'rmse': float,                   # Root mean square error
        'r2': float                      # Coefficient of determination
    }
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/bing.git
cd bing

# Create a development environment
conda create -n bing-dev python=3.9
conda activate bing-dev

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use BING in your research, please cite:

```bibtex
@software{bing2024,
  title={BING: Bayesian INferences with Gordon coefficients},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/bing},
  note={Python package for ocean color remote sensing analysis}
}
```

## 🙏 Acknowledgments

- NASA Ocean Biology Processing Group for PACE data access
- Argo Data Management Team for BGC-Argo profiles  
- Loisel et al. (2023) for the synthetic dataset
- Gordon and colleagues for foundational bio-optical formulations
- The emcee developers for the MCMC implementation

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/bing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/bing/discussions)
- **Email**: bing-dev@example.com

## 🔗 Related Projects

- [OCPY](https://github.com/oceancolor/ocpy) - Ocean Color Python tools
- [earthaccess](https://github.com/nsidc/earthaccess) - NASA Earthdata access
- [emcee](https://github.com/dfm/emcee) - The MCMC Hammer

## 📈 Project Status

BING is under active development. Current version: 2.0.0

### Recent Updates
- Added support for PACE OCI hyperspectral data
- Implemented Raman scattering corrections
- Enhanced Argo BGC matching algorithms
- Improved parallel processing capabilities

### Roadmap
- [ ] GUI interface for interactive fitting
- [ ] Machine learning models for initial parameter estimation
- [ ] Support for geostationary ocean color sensors
- [ ] Integration with cloud computing platforms
- [ ] Automated quality control workflows

---

**Note**: For detailed information about the mathematical formulations, algorithm descriptions, and scientific background, please refer to the [full documentation](https://bing.readthedocs.io).

.. _api:

=============
API Reference
=============

This section provides detailed API documentation for all BING modules.

.. toctree::
   :maxdepth: 2

   models_api
   fitting_api
   parameters_api
   evaluation_api
   visualization_api
   utilities_api
   io_api

Core Modules
------------

The BING package is organized into the following main modules:

**bing.models**
    Bio-optical models for absorption and backscattering

**bing.fitting**
    Parameter estimation algorithms (least-squares, MCMC)

**bing.parameters**
    Standard parameter sets and configurations

**bing.evaluate**
    Model evaluation and statistical analysis

**bing.plotting**
    Visualization utilities for results

**bing.priors**
    Prior distributions for Bayesian inference

**bing.rt**
    Radiative transfer calculations

Quick Reference
---------------

Most Common Functions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Model initialization
    from bing.models import utils as model_utils
    models = model_utils.init(model_names, wavelengths)
    
    # Parameter sets
    from bing.parameters import standard
    params = standard.expb_pow(satellite='PACE')
    
    # Fitting
    from bing.fitting import chisq_fit
    result = chisq_fit.fit(models, wave, Rrs, uncertainty)
    
    # MCMC inference
    from bing.fitting import inference
    chains = inference.fit_one(data, models, pdict)
    
    # Evaluation
    from bing import evaluate
    stats = evaluate.calc_stats(chains, models, wavelengths)
    
    # Plotting
    from bing import plotting
    plotting.show_fits(models, chains, **kwargs)

Module Structure
----------------

.. code-block:: text

    bing/
    ├── models/
    │   ├── anw.py          # Non-water absorption models
    │   ├── bbnw.py         # Non-water backscattering models
    │   ├── functions.py    # Spectral basis functions
    │   └── utils.py        # Model construction helpers
    ├── fitting/
    │   ├── chisq_fit.py    # Least-squares fitting
    │   ├── inference.py    # MCMC inference (emcee)
    │   └── l23.py          # Loisel et al. 2023 driver
    ├── parameters/
    │   ├── standard.py     # Standard model combinations
    │   └── p_ntuple.py     # Parameter tuple generator
    ├── priors/
    │   ├── priors.py       # Prior distributions
    │   └── adg.py          # a_dg-specific priors
    ├── rt/
    │   ├── rrs.py          # Gordon relation, Rrs builders
    │   ├── raman.py        # Raman scattering
    │   ├── chl_fl.py       # Chlorophyll fluorescence
    │   └── defs.py         # rt_dict definitions
    ├── evaluate.py         # Reconstruction and statistics
    ├── io.py               # Saving and loading fits
    ├── noise.py            # Satellite noise models
    ├── plotting.py         # Figures
    ├── preproc.py          # Wavelength preprocessing
    └── stats.py            # Information criteria

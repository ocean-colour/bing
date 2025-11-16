.. _api:

=============
API Reference
=============

This section provides detailed API documentation for all BING modules.

.. toctree::
   :maxdepth: 2
   
   core
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
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── anw.py          # Non-water absorption models
    │   ├── bbnw.py         # Non-water backscattering models
    │   ├── base.py         # Base model classes
    │   ├── functions.py    # Model functions
    │   └── utils.py        # Model utilities
    ├── fitting/
    │   ├── __init__.py
    │   ├── chisq_fit.py    # Chi-square fitting
    │   ├── inference.py    # MCMC inference
    │   └── l23.py          # Loisel et al. 2023 fitting
    ├── parameters/
    │   ├── __init__.py
    │   ├── standard.py     # Standard parameter sets
    │   └── p_ntuple.py     # Parameter tuple generator
    ├── priors/
    │   ├── __init__.py
    │   └── priors.py       # Prior distributions
    ├── evaluate.py         # Evaluation functions
    ├── plotting.py         # Plotting utilities
    ├── rt.py              # Radiative transfer
    └── utils.py           # General utilities

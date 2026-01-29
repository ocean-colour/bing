.. BING documentation master file

==================================================
BING - Bayesian INference with Gordon coefficients
==================================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

Welcome to the BING documentation. BING is a comprehensive Python package for ocean color remote sensing analysis, 
specializing in bio-optical modeling, spectral fitting, and satellite data processing with a focus on 
NASA's PACE (Plankton, Aerosol, Cloud, ocean Ecosystem) mission data.

Features
--------

* **Ocean Color Analysis**: Process and analyze remote sensing reflectance (Rrs) data
* **Model Fitting**: Bayesian and least-squares fitting of bio-optical models
* **PACE Integration**: Native support for PACE OCI data processing
* **Uncertainty Quantification**: Comprehensive error propagation and uncertainty analysis

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   installation
   tutorials/index
   api/index
   models
   fitting
   parameters
   data_processing
   examples
   contributing
   changelog
   raman
   chlorophyll_fluorescence
   references

Quick Start
-----------

.. code-block:: python

    import numpy as np
    from bing import evaluate
    from bing.parameters import standard
    from bing.models import utils as model_utils
    from bing.fitting import chisq_fit, inference

    # Load standard parameters
    params = standard.expb_pow(satellite='PACE')
    
    # Initialize models
    models = model_utils.init(params.model_names, wavelengths)
    
    # Perform fitting
    result = chisq_fit.fit(Rrs_data, Rrs_uncertainty, models)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
--------

If you use BING in your research, please cite:

.. code-block:: bibtex

    @software{bing2024,
      title={BING: Biogeochemical Index Network Generator},
      author={Your Name},
      year={2024},
      url={https://github.com/yourusername/bing}
    }

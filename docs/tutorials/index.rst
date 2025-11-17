.. _tutorials:

=========
Tutorials
=========

These tutorials provide step-by-step guides for common BING workflows.

.. toctree::
   :maxdepth: 2
   
   basic_fitting
   pace_processing
   argo_matching
   model_comparison
   uncertainty_analysis
   visualization

Getting Started
---------------

If you're new to BING, we recommend starting with these tutorials in order:

1. :doc:`basic_fitting` - Learn the fundamentals of fitting bio-optical models
2. :doc:`pace_processing` - Process PACE OCI data
3. :doc:`argo_matching` - Match satellite data with Argo floats
4. :doc:`model_comparison` - Compare different model formulations
5. :doc:`uncertainty_analysis` - Quantify uncertainties in retrieved parameters
6. :doc:`visualization` - Create publication-quality figures

Tutorial Format
---------------

Each tutorial includes:

* **Objectives** - What you'll learn
* **Prerequisites** - Required knowledge and data
* **Code examples** - Complete, runnable code
* **Expected output** - What results to expect
* **Exercises** - Practice problems to test understanding

Example Data
------------

Tutorial data can be downloaded from:

* PACE OCI sample data: `NASA Earthdata <https://earthdata.nasa.gov/>`_
* Argo BGC profiles: `Argo Data Management <https://argo.ucsd.edu/>`_
* Loisel et al. (2023) synthetic dataset: Included in BING

Jupyter Notebooks
-----------------

Interactive Jupyter notebooks for all tutorials are available in the 
`BING repository <https://github.com/yourusername/bing/tree/main/notebooks>`_.

To run the notebooks:

.. code-block:: bash

    git clone https://github.com/yourusername/bing.git
    cd bing/notebooks
    jupyter notebook

Quick Examples
--------------

Basic Fitting Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from bing.parameters import standard
    from bing.models import utils as model_utils
    from bing.fitting import chisq_fit
    
    # Setup
    wavelengths = np.arange(400, 701, 10)
    params = standard.expb_pow()
    models = model_utils.init(params.model_names, wavelengths)
    
    # Simulate data (replace with real data)
    true_params = [0.01, 0.65, 0.015, 0.001]
    Rrs_true = models[0].eval(true_params[:2]) + models[1].eval(true_params[2:])
    Rrs_noise = Rrs_true + np.random.normal(0, 0.0005, len(wavelengths))
    Rrs_unc = np.full_like(Rrs_noise, 0.0005)
    
    # Fit
    result = chisq_fit.fit(models, wavelengths, Rrs_noise, Rrs_unc)
    print(f"Fitted: {result['x']}")
    print(f"True:   {true_params}")

PACE Processing Example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ocpy.pace import io as pace_io
    import fitting as m_fitting
    import pandas as pd
    
    # Load matched data
    matched = pd.read_csv('matched_argo_bgc_profiles.csv')
    
    # Process first match
    imatched = matched.iloc[0]
    outfile = f"fits/{imatched.cruise}_{imatched.profile:03d}.npz"
    
    # Run fitting
    m_fitting.doit(imatched, outfile)
    
    # Load results
    results = np.load(outfile)
    print(f"Available data: {list(results.keys())}")

Visualization Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bing import plotting
    import matplotlib.pyplot as plt
    
    # After fitting...
    plotting.show_fits(
        models, chains,
        Chl=1.5, Y=443,
        Rrs_true={'wave': wavelengths, 'spec': Rrs_measured},
        perc=(16, 84),
        savefig='fit_results.png'
    )
    plt.show()

Common Issues
-------------

**Memory errors with large datasets**
    Use data chunking or reduce MCMC steps

**Convergence failures**
    Check initial parameters and adjust bounds

**Missing dependencies**
    Ensure all required packages are installed

**Authentication errors**
    Configure NASA Earthdata credentials

Support
-------

For tutorial-specific questions:

* Check the FAQ section in each tutorial
* Post issues on GitHub
* Contact the BING development team

Video Tutorials
---------------

Video walkthroughs of key tutorials are available on our 
`YouTube channel <https://youtube.com/bingocean>`_.

Contributing Tutorials
----------------------

We welcome community-contributed tutorials! See :doc:`/contributing` 
for guidelines on submitting new tutorials.

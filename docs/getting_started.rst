.. _getting_started:

===============
Getting Started
===============

This guide provides a quick introduction to using BING for ocean color remote sensing analysis.

Overview
--------

BING (Biogeochemical Index Network Generator) is designed to:

1. Process ocean color remote sensing data from satellites (especially PACE)
2. Fit bio-optical models to spectral data
3. Match satellite observations with in-situ measurements (e.g., Argo floats)
4. Perform uncertainty quantification and statistical analysis

Core Concepts
-------------

Remote Sensing Reflectance (Rrs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remote sensing reflectance is the fundamental measurement in ocean color:

.. math::

    R_{rs}(\lambda) = \frac{L_w(\lambda)}{E_d(\lambda)}

where :math:`L_w` is water-leaving radiance and :math:`E_d` is downwelling irradiance.

Inherent Optical Properties (IOPs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BING models the relationship between Rrs and IOPs:

* **a(λ)**: Total absorption coefficient
* **bb(λ)**: Total backscattering coefficient
* **anw(λ)**: Non-water absorption
* **bbnw(λ)**: Non-water backscattering

Bio-optical Models
~~~~~~~~~~~~~~~~~~

BING implements various models for IOPs:

* **Exponential models** for absorption
* **Power-law models** for backscattering
* **Empirical models** (GSM, GIOP)

Basic Workflow
--------------

1. Load Data
~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from grab_pace_granules import load_from_json
    
    # Load matched Argo profiles
    matched = pd.read_csv('matched_argo_bgc_profiles_bbp.csv')
    
    # Load PACE granules
    granules, pace_data = load_from_json('PACE_50clouds.json')

2. Initialize Models
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bing.parameters import standard
    from bing.models import utils as model_utils
    import numpy as np
    
    # Define wavelengths
    wavelengths = np.arange(400, 701, 5)
    
    # Get standard parameters
    params = standard.expb_pow(satellite='PACE', add_noise=True)
    
    # Initialize models
    models = model_utils.init(params.model_names, wavelengths)

3. Prepare Data
~~~~~~~~~~~~~~~

.. code-block:: python

    from ocpy.pace import io as pace_io
    
    # Load PACE L2 data
    xds, flags = pace_io.load_oci_l2('path/to/pace/file.nc')
    
    # Extract Rrs at specific location
    lat, lon = 40.0, -70.0
    rrs_data = xds.Rrs.sel(latitude=lat, longitude=lon, method='nearest')
    
    # Get uncertainties
    rrs_uncertainty = xds.Rrs_unc.sel(latitude=lat, longitude=lon, method='nearest')

4. Perform Fitting
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bing.fitting import chisq_fit
    from bing.fitting import inference as bing_inf
    
    # Least-squares fit for initial guess
    try:
        lm_result = chisq_fit.fit(
            models, wavelengths, rrs_data.values, 
            rrs_uncertainty.values
        )
        p0 = lm_result['x']
    except RuntimeError:
        print("Least-squares fit failed, using default initial guess")
        p0 = np.ones(4)
    
    # MCMC fitting for uncertainty quantification
    pdict = bing_inf.init_mcmc(models, nsteps=10000, nburn=1000)
    chains = bing_inf.fit_one((rrs_data, rrs_uncertainty, p0, 0), 
                              models=models, pdict=pdict)

5. Analyze Results
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bing import evaluate
    from bing import plotting
    
    # Calculate statistics
    stats = evaluate.calc_stats(chains, models, wavelengths)
    
    # Plot results
    plotting.show_fits(
        models, chains, 
        Chl=1.0, Y=443,  # Example values
        Rrs_true=dict(wave=wavelengths, spec=rrs_data),
        perc=(16, 84)  # Confidence intervals
    )

Example: Processing PACE Data
------------------------------

Here's a complete example processing PACE OCI data:

.. code-block:: python

    import os
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    
    # BING imports
    from bing.parameters import standard
    from bing.models import utils as model_utils
    from bing.fitting import inference as bing_inf
    from bing.fitting import chisq_fit
    from bing import evaluate
    from bing import plotting
    
    # OCPY imports
    from ocpy.pace import io as pace_io
    
    # Load PACE file
    pace_file = os.path.join(os.getenv('OS_COLOR'), 
                             'PACE/L2_AOP/PACE_OCI.20240315T183000.L2.OC_AOP.V2.nc')
    xds, flags = pace_io.load_oci_l2(pace_file)
    
    # Define location of interest
    lat, lon = 25.0, -80.0  # Near Florida
    
    # Extract data at location
    idx_lat = np.argmin(np.abs(xds.latitude.values - lat))
    idx_lon = np.argmin(np.abs(xds.longitude.values - lon))
    
    # Get Rrs spectrum
    wavelengths = xds.wavelength.values
    good_bands = (wavelengths >= 400) & (wavelengths <= 700)
    wave = wavelengths[good_bands]
    
    rrs = xds.Rrs[idx_lat, idx_lon, good_bands].values
    rrs_unc = xds.Rrs_unc[idx_lat, idx_lon, good_bands].values
    
    # Check for valid data
    if np.all(np.isnan(rrs)):
        print("No valid data at this location")
    else:
        # Initialize models
        params = standard.expb_pow(satellite='PACE')
        models = model_utils.init(params.model_names, wave)
        
        # Fit the data
        result = chisq_fit.fit(models, wave, rrs, rrs_unc)
        
        # Print results
        print(f"Fitted parameters: {result['x']}")
        print(f"Chi-squared: {result['chisq']}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(wave, rrs, yerr=rrs_unc, fmt='o', label='PACE OCI')
        ax.plot(wave, result['model_Rrs'], 'r-', label='Fitted Model')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Rrs (sr$^{-1}$)')
        ax.legend()
        plt.show()

Working with Matched Data
-------------------------

BING includes tools for matching satellite data with in-situ measurements:

.. code-block:: python

    import fitting as m_fitting
    
    # Load matched Argo profile
    matched = pd.read_csv('matched_argo_bgc_profiles_bbp.csv')
    imatched = matched.iloc[0]  # First matched profile
    
    # Process the matched data
    outfile = f"fits/Argo_{imatched.cruise}_{imatched.profile:03d}_fits.npz"
    m_fitting.doit(imatched, outfile, nclosest=5)
    
    # Load and analyze results
    fit_data = np.load(outfile)
    print("Available keys:", fit_data.keys())

Next Steps
----------

* Explore the :doc:`/tutorials/index` for detailed examples
* Review the :doc:`/api/index` for comprehensive function documentation
* Learn about different :doc:`models` available in BING
* Understand the :doc:`fitting` algorithms
* Check :doc:`examples` for real-world applications

Tips and Best Practices
-----------------------

1. **Data Quality**: Always check quality flags in satellite data
2. **Wavelength Selection**: Focus on 400-700 nm for ocean color
3. **Initial Guesses**: Good initial parameters improve convergence
4. **Uncertainty**: Include measurement uncertainties for robust fitting
5. **Validation**: Compare results with in-situ measurements when available

Getting Help
------------

* GitHub Issues: Report bugs or request features
* Documentation: This documentation site
* Examples: Jupyter notebooks in the repository

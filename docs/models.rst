.. _models:

==================
Bio-optical Models
==================

BING implements various bio-optical models for ocean color remote sensing analysis.
This document describes the available models and their mathematical formulations.

Overview
--------

Bio-optical models in BING relate Inherent Optical Properties (IOPs) to Remote Sensing Reflectance (Rrs):

.. math::

    R_{rs}(\lambda) = f \cdot \frac{b_b(\lambda)}{a(\lambda) + b_b(\lambda)}

where:
- :math:`a(\lambda)` is the total absorption coefficient
- :math:`b_b(\lambda)` is the total backscattering coefficient
- :math:`f` is a factor depending on the solar zenith angle and viewing geometry

In addition to this elastic forward model, BING optionally adds two inelastic
contributions to Rrs: Raman scattering by water molecules and chlorophyll
fluorescence emission near 685 nm. These are enabled through the
``include_Raman`` and ``include_Chl_fl`` flags on the parameter named-tuple
(see :ref:`parameters`) and are computed by :mod:`bing.rt`. The chlorophyll
fluorescence path depends on the ``correct_atmosphere`` package for the
downwelling irradiance spectrum -- see :doc:`chlorophyll_fluorescence` and
:doc:`radiative_transfer`.

Model Components
----------------

Total Absorption
~~~~~~~~~~~~~~~~

The total absorption coefficient is decomposed as:

.. math::

    a(\lambda) = a_w(\lambda) + a_{nw}(\lambda)

where:
- :math:`a_w(\lambda)` is pure water absorption
- :math:`a_{nw}(\lambda)` is non-water absorption

Non-water Absorption Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Exponential Model (Bricaud)**

.. math::

    a_{ph}(\lambda) = A_{ph} \cdot [Chl]^{E_{ph}} \cdot a_{ph}^*(\lambda)

where :math:`a_{ph}^*(\lambda)` is the chlorophyll-specific absorption coefficient.

**CDOM Absorption**

.. math::

    a_{dg}(\lambda) = a_{dg}(\lambda_0) \cdot \exp[-S_{dg}(\lambda - \lambda_0)]

where :math:`S_{dg}` is the spectral slope.

Total Backscattering
~~~~~~~~~~~~~~~~~~~~

.. math::

    b_b(\lambda) = b_{bw}(\lambda) + b_{bnw}(\lambda)

where:
- :math:`b_{bw}(\lambda)` is water molecular backscattering
- :math:`b_{bnw}(\lambda)` is particulate backscattering

Non-water Backscattering Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Power Law Model**

.. math::

    b_{bnw}(\lambda) = b_{bnw}(\lambda_0) \cdot \left(\frac{\lambda_0}{\lambda}\right)^Y

where :math:`Y` is the spectral slope parameter.

**Lee Model**

Based on Lee et al. (2002):

.. math::

    b_{bp}(\lambda) = b_{bp}(443) \cdot \left(\frac{443}{\lambda}\right)^Y

Implemented Models
------------------

Absorption Models (anw)
~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: bing.models.anw

    **ExpBricaud**
        Exponential Bricaud model for phytoplankton absorption
        
        Parameters:
            - ``A_ph``: Amplitude coefficient
            - ``E_ph``: Exponent for chlorophyll dependency
            - ``S_dg``: CDOM spectral slope

    **GIOP**
        Generalized IOP model absorption
        
        Parameters:
            - ``a_dg_443``: CDOM absorption at 443 nm
            - ``a_ph_443``: Phytoplankton absorption at 443 nm
            - ``S_dg``: CDOM spectral slope

    **GSM**
        Garver-Siegel-Maritorena model absorption
        
        Parameters:
            - ``Chl``: Chlorophyll concentration
            - ``a_dg_443``: CDOM absorption at 443 nm

    **QAA**
        Quasi-Analytical Algorithm absorption
        
        Parameters:
            - ``a_total_ref``: Total absorption at reference wavelength
            - ``S``: Spectral slope

Backscattering Models (bbnw)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: bing.models.bbnw

    **PowerLaw**
        Power law backscattering model
        
        Parameters:
            - ``b_bp_ref``: Particulate backscattering at reference wavelength
            - ``Y``: Spectral slope

    **Lee**
        Lee et al. (2002) backscattering model
        
        Parameters:
            - ``b_bp_443``: Particulate backscattering at 443 nm
            - ``Y``: Spectral slope (default: 1.0)

    **GSM**
        GSM backscattering model
        
        Parameters:
            - ``b_bp_443``: Particulate backscattering at 443 nm

    **Constant**
        Wavelength-independent backscattering
        
        Parameters:
            - ``b_bp``: Constant backscattering value

Model Initialization
--------------------

Initialize models using the utilities module:

.. code-block:: python

    from bing.models import utils as model_utils
    import numpy as np
    
    # Define wavelengths
    wavelengths = np.arange(400, 701, 5)
    
    # Initialize absorption and backscattering models
    model_names = ['ExpBricaud', 'PowerLaw']
    models = model_utils.init(model_names, wavelengths)
    
    # Access individual models
    anw_model = models[0]  # Absorption model
    bbnw_model = models[1]  # Backscattering model

Model Configuration
-------------------

Setting Chlorophyll
~~~~~~~~~~~~~~~~~~~

For models that use chlorophyll:

.. code-block:: python

    # Set chlorophyll concentration
    anw_model.set_aph(Chl=1.5)  # mg/m³

Custom Parameters
~~~~~~~~~~~~~~~~~

Override default parameters:

.. code-block:: python

    from bing.models.anw import ExpBricaud
    
    # Custom initialization
    model = ExpBricaud(
        wave=wavelengths,
        A_ph=0.05,
        E_ph=0.65,
        S_dg=0.015
    )

Model Evaluation
----------------

Forward Modeling
~~~~~~~~~~~~~~~~

Calculate Rrs from model parameters:

.. code-block:: python

    from bing import rt as bing_rt
    
    # Set parameters
    params = [0.01, 0.65, 0.015, 0.001, 1.2]  # Example parameters
    
    # Evaluate models
    anw = anw_model.eval(params[:3])
    bbnw = bbnw_model.eval(params[3:])
    
    # Add water contributions
    aw = absorption.a_water(wavelengths)
    bbw = scattering.b_water(wavelengths) * 0.5
    
    # Total IOPs
    a_total = aw + anw
    bb_total = bbw + bbnw
    
    # Calculate Rrs
    Rrs = bing_rt.calc_Rrs(a_total, bb_total)

Model Comparison
~~~~~~~~~~~~~~~~

Compare different model combinations:

.. code-block:: python

    from bing.parameters import standard
    
    # Different model combinations
    configs = [
        standard.expb_pow(),    # ExpBricaud + PowerLaw
        standard.giop(),         # GIOP + Lee
        standard.gsm_gsm(),      # GSM + GSM
    ]
    
    for config in configs:
        models = model_utils.init(config.model_names, wavelengths)
        # Perform fitting and analysis...

Performance Considerations
--------------------------

Model Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **ExpBricaud + PowerLaw**: Good for general cases, especially with chlorophyll data
2. **GIOP**: NASA's operational model, well-validated
3. **GSM**: Empirical model, good for Case 1 waters
4. **QAA**: Semi-analytical, good for clear waters

Computational Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~

- Simple models (Constant, PowerLaw): Fast evaluation
- Complex models (GSM, QAA): More computational overhead
- MCMC fitting: Use appropriate number of steps based on model complexity

Model Validation
----------------

Residual Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bing import evaluate
    
    # Calculate residuals
    residuals = (measured_Rrs - modeled_Rrs) / uncertainty
    
    # Statistical metrics
    rmse = np.sqrt(np.mean(residuals**2))
    bias = np.mean(residuals)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"Bias: {bias:.4f}")

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.model_selection import KFold
    
    # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True)
    
    for train_idx, test_idx in kf.split(data):
        train_data = data[train_idx]
        test_data = data[test_idx]
        
        # Fit model on training data
        # Evaluate on test data

Advanced Topics
---------------

Custom Model Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create custom models by subclassing base classes:

.. code-block:: python

    from bing.models.base import BaseModel
    
    class CustomAbsorption(BaseModel):
        def __init__(self, wave, **kwargs):
            super().__init__(wave, **kwargs)
            self.n_params = 3  # Number of parameters
        
        def eval(self, params):
            # Custom evaluation logic
            a0, slope, offset = params
            return a0 * np.exp(-slope * (self.wave - 443)) + offset

Model Coupling
~~~~~~~~~~~~~~

Couple absorption and backscattering models:

.. code-block:: python

    # Coupled parameterization
    def coupled_model(params, wave):
        # Shared parameters
        Chl = params[0]
        
        # Absorption depends on Chl
        anw = 0.05 * Chl**0.65 * absorption_spectrum(wave)
        
        # Backscattering also depends on Chl
        bbnw = 0.001 * Chl**0.5 * backscatter_spectrum(wave)
        
        return anw, bbnw

References
----------

1. Bricaud et al. (1995) - Phytoplankton absorption
2. Lee et al. (2002) - Backscattering models
3. Maritorena et al. (2002) - GSM algorithm
4. Werdell et al. (2013) - GIOP framework
5. Lee et al. (2002) - QAA algorithm

.. _parameters:

==========
Parameters
==========

BING provides pre-configured parameter sets for common ocean color analysis scenarios.

Overview
--------

The parameters module provides:

* Standard parameter configurations for different satellites
* Model combinations for various water types
* Prior distributions for Bayesian inference
* Noise models for uncertainty estimation

Standard Parameter Sets
-----------------------

The :mod:`bing.parameters.standard` module provides ready-to-use configurations:

ExpBricaud + PowerLaw
~~~~~~~~~~~~~~~~~~~~~

Most commonly used combination:

.. code-block:: python

    from bing.parameters import standard
    
    # Basic configuration
    params = standard.expb_pow(
        satellite='PACE',    # 'PACE', 'MODIS', 'SeaWiFS', 'SBG'
        add_noise=True,      # Add realistic noise
        nsteps=40000,        # MCMC steps
        nburn=1000          # Burn-in period
    )
    
    print(params.model_names)  # ['ExpBricaud', 'PowerLaw']
    print(params.wv_min, params.wv_max)  # 400.0 700.0

GIOP Configuration
~~~~~~~~~~~~~~~~~~

NASA's operational algorithm:

.. code-block:: python

    params = standard.giop(
        satellite='PACE',
        set_Sdg=True,        # Fix CDOM slope
        sSdg=0.018           # CDOM slope value
    )

GSM Configuration
~~~~~~~~~~~~~~~~~

Empirical model for Case 1 waters:

.. code-block:: python

    params = standard.gsm_gsm(
        satellite='MODIS',
        beta=1.0             # Backscattering slope constraint
    )

Custom Parameter Sets
---------------------

Creating Custom Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the parameter tuple generator:

.. code-block:: python

    from bing.parameters import p_ntuple
    
    # Define custom configuration
    custom_params = p_ntuple.gen(
        model_names=['ExpBricaud', 'Lee'],
        satellite='PACE',
        wv_min=400.0,
        wv_max=700.0,
        add_noise=True,
        nsteps=20000,
        nburn=500,
        scl_noise=1.0,
        apriors=[
            {'flavor': 'log_uniform', 'pmin': -6, 'pmax': 2},
            {'flavor': 'uniform', 'pmin': 0.3, 'pmax': 1.0},
            {'flavor': 'gaussian', 'mean': 0.015, 'std': 0.003}
        ],
        bpriors=[
            {'flavor': 'log_uniform', 'pmin': -6, 'pmax': 0}
        ]
    )

Parameter Structure
~~~~~~~~~~~~~~~~~~~

Parameter tuples contain:

.. code-block:: python

    # Access parameters
    params.model_names      # List of model names
    params.satellite       # Satellite configuration
    params.wv_min         # Minimum wavelength
    params.wv_max         # Maximum wavelength
    params.add_noise      # Noise flag
    params.scl_noise      # Noise scaling
    params.nsteps         # MCMC steps
    params.nburn          # Burn-in steps
    params.apriors        # Absorption priors
    params.bpriors        # Backscattering priors
    params.set_Sdg        # CDOM slope flag
    params.sSdg           # CDOM slope value
    params.beta           # Backscattering constraint
    # Radiative-transfer options
    params.variable_Gordon # Wavelength-dependent Gordon coefficients
    params.include_Raman   # Include Raman scattering correction
    params.include_Chl_fl  # Include chlorophyll fluorescence emission
    params.phi_C           # Fluorescence quantum yield (default 0.02)
    params.double_gaussian # Double-Gaussian (685+730 nm) emission line

Radiative-Transfer Options
~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter tuple carries the radiative-transfer flags consumed by the
forward model and the fitting routines. These are extracted into a small
"rt dict" by :func:`bing.rt.defs.rt_dict_from_p`:

.. code-block:: python

    from bing.parameters import p_ntuple
    from bing.rt import defs as rt_defs

    p = p_ntuple.gen(
        model_names=['ExpBricaud', 'Pow'],
        variable_Gordon=True,
        include_Raman=False,
        include_Chl_fl=True,    # Enable chlorophyll fluorescence
        phi_C=0.02,             # Quantum yield
        double_gaussian=True,   # Use 685 + 730 nm emission peaks
    )

    rt_dict = rt_defs.rt_dict_from_p(p)
    # {'variable_Gordon': True, 'include_Raman': False,
    #  'include_Chl_fl': True,  'phi_C': 0.02,
    #  'double_gaussian': True}

The fields are:

- ``variable_Gordon`` (bool, default ``True``) -- use wavelength-dependent
  :math:`G_1(\lambda), G_2(\lambda)`. See :ref:`radiative_transfer`.
- ``include_Raman`` (bool, default ``False``) -- add the Sathyendranath &
  Platt (1998) Raman scattering correction. See :ref:`raman`.
- ``include_Chl_fl`` (bool, default ``False``) -- add chlorophyll
  fluorescence emission to the forward Rrs. Depends on the
  `correct_atmosphere <https://github.com/ocean-colour/correct-atmosphere>`_
  package. See :doc:`chlorophyll_fluorescence`.
- ``phi_C`` (float, default ``0.02``) -- fluorescence quantum yield.
- ``double_gaussian`` (bool, default ``True``) -- if ``True``, use the
  double-Gaussian emission line (PS II at 685 nm and PS I at 730 nm).

See :ref:`rt-dict-from-p` for the full description of ``rt_dict_from_p``
and how the dictionary is consumed by the fitting routines.

Satellite-Specific Settings
---------------------------

PACE
~~~~

.. code-block:: python

    params = standard.expb_pow(satellite='PACE')
    # Optimized for 400-700 nm range
    # Higher spectral resolution

MODIS
~~~~~

.. code-block:: python

    params = standard.expb_pow(satellite='MODIS')
    # Limited bands in visible
    # Adjusted noise model

SeaWiFS
~~~~~~~

.. code-block:: python

    params = standard.expb_pow(satellite='SeaWiFS')
    # Historical data compatibility
    # 8 visible bands

Future Missions (SBG)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    params = standard.expb_pow(satellite='SBG')
    # Surface Biology and Geology
    # Hyperspectral configuration

Prior Distributions
-------------------

Prior Types
~~~~~~~~~~~

BING supports several prior distribution types:

.. code-block:: python

    # Uniform prior
    uniform_prior = {
        'flavor': 'uniform',
        'pmin': 0.0,
        'pmax': 1.0
    }
    
    # Log-uniform prior (scale-invariant)
    log_uniform_prior = {
        'flavor': 'log_uniform',
        'pmin': -6,  # log10(min)
        'pmax': 2    # log10(max)
    }
    
    # Gaussian prior
    gaussian_prior = {
        'flavor': 'gaussian',
        'mean': 0.5,
        'std': 0.1
    }
    
    # Truncated Gaussian
    truncated_gaussian = {
        'flavor': 'truncated_gaussian',
        'mean': 0.5,
        'std': 0.1,
        'pmin': 0.0,
        'pmax': 1.0
    }

Setting Model-Specific Priors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bing.priors import priors as bing_priors
    
    # Standard priors for models
    model_priors = bing_priors.set_standard_priors(models)
    
    # Custom priors for ExpBricaud model
    custom_apriors = [
        {'flavor': 'log_uniform', 'pmin': -6, 'pmax': 2},  # A_ph
        {'flavor': 'uniform', 'pmin': 0.3, 'pmax': 1.0},   # E_ph
        {'flavor': 'gaussian', 'mean': 0.015, 'std': 0.003} # S_dg
    ]
    
    # Custom priors for PowerLaw model
    custom_bpriors = [
        {'flavor': 'log_uniform', 'pmin': -6, 'pmax': 0},  # b_bp
        {'flavor': 'uniform', 'pmin': 0.0, 'pmax': 2.0}    # Y
    ]

Noise Models
------------

Instrument-Specific Noise
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # PACE noise model
    def pace_noise(Rrs, scale=1.0):
        """PACE-specific noise model"""
        base_noise = 5e-4
        relative_noise = 0.02
        return scale * np.sqrt(
            (base_noise)**2 + (relative_noise * Rrs)**2
        )
    
    # MODIS noise model
    def modis_noise(Rrs, scale=1.0):
        """MODIS-specific noise model"""
        base_noise = 1e-3
        relative_noise = 0.05
        return scale * np.sqrt(
            (base_noise)**2 + (relative_noise * Rrs)**2
        )

Adding Synthetic Noise
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bing.utils import add_noise
    
    # Add realistic noise to synthetic data
    Rrs_noisy = add_noise(
        Rrs_true, 
        params.satellite,
        scale=params.scl_noise
    )

Parameter Validation
--------------------

Checking Parameters
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def validate_parameters(params):
        """Validate parameter configuration"""
        
        # Check wavelength range
        assert params.wv_min < params.wv_max
        assert params.wv_min >= 350
        assert params.wv_max <= 900
        
        # Check MCMC settings
        assert params.nsteps > params.nburn
        assert params.nburn >= 100
        
        # Check model compatibility
        valid_models = ['ExpBricaud', 'PowerLaw', 'Lee', 
                       'GIOP', 'GSM', 'QAA', 'Constant']
        for model in params.model_names:
            assert model in valid_models
        
        # Check priors
        if params.apriors:
            assert len(params.apriors) == expected_n_params_abs
        if params.bpriors:
            assert len(params.bpriors) == expected_n_params_bb
        
        return True

Parameter Bounds
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Physical bounds for parameters
    bounds = {
        'A_ph': (1e-6, 10),      # Phytoplankton amplitude
        'E_ph': (0.3, 1.0),      # Phytoplankton exponent
        'S_dg': (0.01, 0.025),   # CDOM slope
        'b_bp': (1e-6, 0.1),     # Particulate backscattering
        'Y': (0.0, 3.0),         # Backscattering slope
        'Chl': (0.01, 100),      # Chlorophyll concentration
        'a_dg_443': (1e-6, 1.0)  # CDOM absorption at 443nm
    }

Best Practices
--------------

1. **Model Selection**
   
   - Use ExpBricaud+PowerLaw for general purposes
   - Use GIOP for NASA compatibility
   - Use GSM for empirical analysis

2. **Prior Selection**
   
   - Use log-uniform for scale parameters
   - Use uniform for bounded parameters
   - Use Gaussian when mean is known

3. **MCMC Settings**
   
   - Start with nsteps=10000 for testing
   - Use nsteps=40000+ for publication
   - Set nburn to 10-25% of nsteps

4. **Noise Configuration**
   
   - Use satellite-specific noise models
   - Scale noise based on data quality
   - Include noise in synthetic studies

5. **Wavelength Range**
   
   - 400-700 nm for ocean color
   - Extend to 350-750 for full visible
   - Consider NIR for turbid waters

Examples
--------

High Chlorophyll Waters
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configuration for productive waters
    params = standard.expb_pow(
        satellite='PACE',
        add_noise=True
    )
    
    # Adjust priors for high Chl
    params.apriors[0] = {'flavor': 'log_uniform', 'pmin': -4, 'pmax': 2}

Clear Ocean Waters
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configuration for oligotrophic waters
    params = p_ntuple.gen(
        model_names=['QAA', 'Lee'],
        satellite='MODIS',
        apriors=[
            {'flavor': 'log_uniform', 'pmin': -8, 'pmax': -2}
        ]
    )

Coastal Waters
~~~~~~~~~~~~~~

.. code-block:: python

    # Configuration for complex coastal waters
    params = standard.giop(
        satellite='PACE',
        set_Sdg=False  # Variable CDOM slope
    )

References
----------

* Standard parameter sets based on literature values
* Prior distributions from field measurements
* Noise models from instrument specifications

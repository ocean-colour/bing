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

As above, the names are the strings :func:`bing.models.anw.init_model`
accepts and the parameters are the model's ``pnames``.

.. py:class:: bing.models.anw
   :no-index:

    **ExpBricaud**
        Exponential CDOM/detrital term plus Bricaud et al. (1995)
        phytoplankton absorption -- the most commonly used model

        Parameters:
            - ``Adg``: log10 CDOM+detrital amplitude at 400 nm
            - ``Sdg``: exponential slope (linear)
            - ``Aph``: log10 phytoplankton amplitude at 440 nm

    **ExpBricaudFix**
        As ``ExpBricaud`` but with ``Chl`` fixed externally rather than
        derived from ``Aph`` (same three parameters)

    **ExpBricaudFree**
        As ``ExpBricaud`` with ``Chl`` a free parameter

        Parameters: ``Adg``, ``Sdg``, ``Chl``, ``Aph``

    **GIOP**
        GIOP-style absorption: exponential ``a_dg`` with a fixed slope
        plus a Bricaud ``a_ph`` basis

        Parameters:
            - ``Aexp``: log10 ``a_dg`` amplitude
            - ``Aph``: log10 phytoplankton amplitude

    **GSM**
        Garver-Siegel-Maritorena absorption

        Parameters:
            - ``Aexp``: log10 ``a_dg`` amplitude
            - ``Chl``: log10 chlorophyll

    **Exp**, **ExpFix**, **Cst**, **Every**
        Simpler forms: exponential with a free (``Exp``: ``Anw``,
        ``Snw``) or fixed (``ExpFix``: ``Aexp``) slope, a spectrally flat
        term (``Cst``), and a non-parametric one-amplitude-per-channel
        model (``Every``).

    **Chase2017**, **Chase2017Mini**
        Multi-Gaussian phytoplankton decomposition (28 and 12
        parameters). Note these hold *all* parameters in log10, including
        the Gaussian widths and centres, while declaring ``uniform``
        priors over log10 bounds -- so they sit outside the
        prior-flavour convention described in :ref:`log-params`.

Backscattering Models (bbnw)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model names below are the strings accepted by
:func:`bing.models.bbnw.init_model`, and the parameter names are the
model's ``pnames``. Amplitudes are fitted in log10 space; spectral
exponents are linear (see :ref:`log-params`).

.. py:class:: bing.models.bbnw
   :no-index:

    **Pow**
        Power law, the most common choice (2 parameters)

        .. math::

            b_{b,nw}(\lambda) = 10^{B_{nw}}
            \left(\frac{600}{\lambda}\right)^{\beta}

        Parameters:
            - ``Bnw``: log10 amplitude at the 600 nm pivot
            - ``beta``: spectral exponent (linear). Note that positive
              ``beta`` means a *decreasing* ``bb_nw``; the ``expb_pow``
              prior floors it at 0.

    **Lee**
        Power law with the exponent taken from Lee et al. (2002)
        rather than fitted (1 parameter)

        Parameters:
            - ``Bnw``: log10 amplitude at 600 nm

        ``Y`` is supplied externally via ``set_basis_func(Y)`` (done for
        you by :func:`bing.models.utils.init_other_bits`).

    **GSM**
        Power law with the globally fixed exponent
        :math:`\eta = 1.0337` at a 443 nm pivot (1 parameter)

        Parameters:
            - ``Bnw``: log10 amplitude at 443 nm

    **Cst**
        Spectrally flat backscattering (1 parameter)

        Parameters:
            - ``Bnw``: log10 amplitude

    **Every**
        Non-parametric: one free amplitude per wavelength

        Parameters:
            - ``Bnw_<wave>``: log10 amplitude in each channel

    **Pow2**
        Two-component mineral + organic power laws, for turbid water
        (4 parameters)

        .. math::

            b_{b,nw}(\lambda) = 10^{B_{min}}
            \left(\frac{700}{\lambda}\right)^{\eta_{min}}
            + 10^{B_{org}}
            \left(\frac{600}{\lambda}\right)^{\eta_{org}}

        Parameters:
            - ``Bmin``: log10 mineral amplitude at 700 nm
            - ``eta_min``: mineral exponent (linear), near 0
            - ``Borg``: log10 organic amplitude at 600 nm
            - ``eta_org``: organic exponent (linear), 0.5-2

        Default priors (``standard.expb_pow2``): amplitudes
        ``log_uniform(-6, 5)``, ``eta_min`` ``uniform(-0.5, 0.5)``,
        ``eta_org`` ``uniform(0.5, 2)``. The two exponent ranges are
        deliberately **disjoint**: the terms are otherwise exchangeable
        and a symmetric prior leaves the posterior with a
        label-switching degeneracy.

    **Pow2Flat**
        ``Pow2`` with the mineral exponent fixed at 0, i.e. a constant
        mineral term plus an organic power law (3 parameters)

        .. math::

            b_{b,nw}(\lambda) = 10^{B_{min}} + 10^{B_{org}}
            \left(\frac{600}{\lambda}\right)^{\eta_{org}}

        Parameters:
            - ``Bmin``: log10 mineral amplitude (wavelength independent)
            - ``Borg``: log10 organic amplitude at 600 nm
            - ``eta_org``: organic exponent (linear)

Turbid water: why two components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open-ocean parameterisations assume a single, decreasing power law. In
turbid, mineral-dominated water the measured particulate backscattering
is larger in magnitude, **flatter** in spectral slope -- sometimes rising
toward the red -- and higher in backscattering ratio (Snyder et al. 2008;
Doxaran et al. 2009; Neukermans et al. 2012; see `References`_). A single
power law has one slope by construction, whereas a sum of a flat mineral
term and a steeper organic term has a slope that *varies* across the
band, which is the behaviour those measurements report.

Two practical notes from benchmarking these models
(``dev/turbid_bbp/turbid_bbp.py``):

* **Prefer** ``Pow2Flat`` **with MCMC.** Under chi-squared the
  4-parameter ``Pow2`` is degenerate against realistic in-situ noise:
  ``Bmin`` and ``Borg`` come back anti-correlated at −1.00 with a
  correlation-matrix condition number of order :math:`10^7`, so the
  reported covariance is meaningless. Fixing ``eta_min`` (i.e.
  ``Pow2Flat``) removes that degenerate direction and recovers every
  parameter to within a fraction of its uncertainty.
* **An inflated noise floor does not help identifiability.** Widening
  :math:`\sigma` makes recovery *worse* by letting the fit slide along
  the degenerate direction. Use an error floor only to make
  :math:`\chi^2_\nu` interpretable, and label it as inflated.
* Raise ``maxfev`` when fitting these models by least squares -- at
  scipy's default budget ``Pow2`` fails to converge on a substantial
  fraction of spectra (see :doc:`fitting`).

Also available as a *prior* variant rather than a new class:
``standard.expb_powflex`` keeps the single ``Pow`` form but widens the
``beta`` prior to ``uniform(-1, 2)`` so the slope may flatten or rise.
It is useful as a control -- if a fit that fails with ``expb_pow``
also fails with ``expb_powflex``, the problem is the functional *form*
rather than the prior *range*.

.. _log-params:

Which parameters are log10
~~~~~~~~~~~~~~~~~~~~~~~~~~

Amplitudes are held in log10, exponents and slopes in linear space. Each
model declares this as a ``log_params`` list of booleans, e.g.
``[True, False, True, False]`` for ``Pow2``. A model that does not
declare it is treated as all-log10, which is the historical default.

Display code reads it via
:func:`bing.plotting.log_param_mask`, so figures exponentiate and label
only the parameters that really are log10. It is deliberately *not* used
by the initial-guess conversion in the fitters, which keys on the prior
flavour (``log_uniform`` vs ``uniform``) instead -- so when adding a
model, keep those two conventions consistent with each other.

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

Absorption and reflectance models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Bricaud, A. et al. (1995). Variability in the chlorophyll-specific
   absorption coefficients of natural phytoplankton.
   *J. Geophys. Res.* 100, 13321-13332. doi:10.1029/95JC00463
2. Lee, Z. et al. (2002). Deriving inherent optical properties from
   water color. *Appl. Opt.* 41, 5755-5772. doi:10.1364/AO.41.005755
3. Maritorena, S. et al. (2002). Optimization of a semianalytical ocean
   color model for global-scale applications. *Appl. Opt.* 41,
   2705-2714. doi:10.1364/AO.41.002705
4. Werdell, P. J. et al. (2013). Generalized ocean color inversion model
   for retrieving marine inherent optical properties. *Appl. Opt.* 52,
   2019-2037. doi:10.1364/AO.52.002019
5. Pope, R. M. & Fry, E. S. (1997). Absorption spectrum (380-700 nm) of
   pure water. II. Integrating cavity measurements. *Appl. Opt.* 36,
   8710-8723. doi:10.1364/AO.36.008710

Backscattering in turbid and mineral-dominated water
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These motivate the two-component ``Pow2`` / ``Pow2Flat`` models.

6. Snyder, W. A. et al. (2008). Optical scattering and backscattering by
   organic and inorganic particulates in U.S. coastal waters.
   *Appl. Opt.* 47, 666-677. doi:10.1364/AO.47.000666 -- the
   particulate backscattering slope flattens toward
   wavelength-independence where inorganic particles dominate.
7. Twardowski, M. S. et al. (2001). A model for estimating bulk
   refractive index from the optical backscattering ratio.
   *J. Geophys. Res.* 106, 14129-14142. doi:10.1029/2000JC000404
8. Boss, E. et al. (2004). Particulate backscattering ratio at LEO 15.
   *J. Geophys. Res.* 109, C01014. doi:10.1029/2002JC001514
9. Whitmire, A. L. et al. (2007). Spectral variability of the particulate
   backscattering ratio. *Opt. Express* 15, 7019-7031.
   doi:10.1364/OE.15.007019
10. Gordon, H. R. et al. (2009). Spectra of particulate backscattering in
    natural waters. *Opt. Express* 17, 16192-16208.
    doi:10.1364/OE.17.016192 -- reports :math:`b_{bp} \propto
    \lambda^{-n}` with :math:`n \approx 0.4-1.0`, already below the
    commonly assumed 1-2.
11. Sullivan, J. M. & Twardowski, M. S. (2009). Angular shape of the
    oceanic particulate volume scattering function in the backward
    direction. *Appl. Opt.* 48, 6811-6819. doi:10.1364/AO.48.006811
12. McKee, D. et al. (2009). Optical water type discrimination and
    tuning remote sensing band-ratio algorithms. *Appl. Opt.* 48,
    4663-4675. doi:10.1364/AO.48.004663
13. Doxaran, D. et al. (2009). Spectral variations of light scattering by
    marine particles in coastal waters. *Limnol. Oceanogr.* 54,
    1257-1271. doi:10.4319/lo.2009.54.4.1257 -- in turbid coastal water
    the scattering spectrum is nearly flat and a power law is further
    distorted by residual particulate absorption.
14. Neukermans, G. et al. (2012). In situ variability of mass-specific
    beam attenuation and backscattering of marine particles.
    *Limnol. Oceanogr.* 57, 124-144. doi:10.4319/lo.2012.57.1.0124 --
    mass-specific backscattering is roughly an order of magnitude higher
    for mineral than organic particles.
15. Babin, M. et al. (2003). Light scattering properties of marine
    particles in coastal and open ocean waters. *Limnol. Oceanogr.* 48,
    843-859. doi:10.4319/lo.2003.48.2.0843
16. Doxaran, D. et al. (2002). Spectrophotometric properties of turbid
    coastal waters. *Remote Sens. Environ.* 81, 149-161.
    doi:10.1016/S0034-4257(01)00341-8
17. Nechad, B. et al. (2010). Calibration and validation of a generic
    multisensor algorithm for mapping of total suspended matter.
    *Remote Sens. Environ.* 114, 854-866. doi:10.1016/j.rse.2009.11.022

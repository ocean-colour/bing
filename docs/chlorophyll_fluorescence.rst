=========================
Chlorophyll Fluorescence
=========================

A set of Python methods for calculating chlorophyll fluorescence inelastic
emission and its contribution to remote sensing reflectance (Rrs) in seawater.

Overview
========

Chlorophyll a (Chl-a) fluorescence occurs when light absorbed by photosynthetic
pigments in phytoplankton is re-emitted at longer wavelengths. The primary
emission peak is centered at 685 nm, with a secondary peak around 730-740 nm.

The fluorescence emission adds to the elastic backscattering signal and can
contribute significantly to remote sensing reflectance in the red wavelengths,
particularly in waters with high chlorophyll concentration or under conditions
of photoinhibition.

This module implements formulations from:

- **Gordon (1979)** -- Original diffuse reflectance augmentation by chlorophyll
  fluorescence at 685 nm
- **Maritorena et al. (2000)** -- Fluorescence quantum yield determinations
- **Behrenfeld et al. (2009)** -- Satellite-detected fluorescence and global
  phytoplankton physiology
- **Bricaud et al. (1995)** -- Phytoplankton absorption parameterization
- **Sathyendranath & Platt (1998)** -- Inelastic scattering radiative transfer

Key characteristics:

- **Excitation range**: 370-690 nm (absorbed by photosynthetic pigments)
- **Primary emission**: 685 nm (PS II), FWHM ~25 nm
- **Secondary emission**: ~730 nm (PS I), FWHM ~50 nm
- **Quantum yield**: 0.005-0.07 (varies with irradiance and physiology)


Dependencies
============

The chlorophyll fluorescence implementation depends on the
``correct_atmosphere`` package for downwelling irradiance spectra used to
weight the excitation wavelengths.

Install via:

.. code-block:: bash

    pip install git+https://github.com/ocean-colour/correct-atmosphere.git

The package is imported as:

.. code-block:: python

    from correct_atmosphere import downwelling


Module Architecture
===================

The chlorophyll fluorescence implementation is organized into two modules:

**Low-level methods** (``bing.rt.chl_fl``):
    Core physical calculations including emission line shapes, quantum yields,
    fluorescence coefficients, phase functions, and wavelength redistribution.
    Also includes the Gordon (1979) / Sathyendranath & Platt (1998) reflectance
    formulation ``calc_R_fluorescence`` and the Fluorescence Line Height
    calculation.

**Rrs methods** (``bing.rt.rrs``):
    The integrated function ``calc_Rrs_fluorescence`` that integrates the
    fluorescence contribution to remote sensing reflectance over excitation
    wavelengths, using the Bricaud parameterization for phytoplankton
    absorption and a supplied downwelling irradiance spectrum.


Quick Start
===========

The simplest way to compute the fluorescence contribution to Rrs:

.. code-block:: python

    import numpy as np
    from bing.rt import calc_Rrs_fluorescence
    from correct_atmosphere import downwelling

    # Emission wavelengths (where fluorescence appears)
    wave = np.linspace(650, 750, 101)

    # Excitation wavelengths (where light is absorbed)
    wave_ex = np.arange(400.0, 681.0, 1.0)

    # IOPs at emission and excitation wavelengths (set by user from a model)
    a_em  = ...   # Total absorption at emission wavelengths  [m^-1]
    bb_em = ...   # Total backscattering at emission wavelengths [m^-1]
    a_ex  = ...   # Total absorption at excitation wavelengths [m^-1]
    bb_ex = ...   # Total backscattering at excitation wavelengths [m^-1]
    aph_ex = ...  # Phytoplankton absorption at excitation wavelengths [m^-1]

    # Downwelling irradiance at excitation and (peak) emission
    Ed_ex = downwelling.solar_irradiance(wave_ex)
    Ed_em = downwelling.solar_irradiance(685.)

    # Fluorescence contribution to Rrs
    Rrs_fl = calc_Rrs_fluorescence(
        wave, a_em, bb_em,
        a_ex, bb_ex, aph_ex,
        wave_ex, Ed_ex, Ed_em,
        phi_C=0.02,
        double_gaussian=True,
    )

    # Fluorescence Line Height (FLH) from satellite Rrs (MERIS/OLCI bands)
    from bing.rt import calc_fluorescence_line_height
    FLH = calc_fluorescence_line_height(
        Rrs_665=0.001, Rrs_680=0.0015, Rrs_709=0.0005)


Top-level Rrs Method
====================

calc_Rrs_fluorescence
---------------------

.. py:function:: calc_Rrs_fluorescence(wavelength, a_em, bb_em, a_ex, bb_ex, aph_ex, wavelength_ex, Ed_ex, Ed_em, mu_d=None, mu_f=None, phi_C=0.02, double_gaussian=True)

   Calculate the contribution to Rrs from chlorophyll fluorescence,
   integrating over excitation wavelengths.

   :param wavelength: Emission wavelength(s) :math:`\lambda` in nm
       (typically 650-750 nm).
   :type wavelength: float or ndarray
   :param a_em: Total absorption at emission wavelength(s) [m\ :sup:`-1`].
   :type a_em: float or ndarray
   :param bb_em: Total backscattering at emission wavelength(s) [m\ :sup:`-1`].
   :type bb_em: float or ndarray
   :param a_ex: Total absorption at excitation wavelengths [m\ :sup:`-1`].
   :type a_ex: ndarray
   :param bb_ex: Total backscattering at excitation wavelengths [m\ :sup:`-1`].
   :type bb_ex: ndarray
   :param aph_ex: Phytoplankton absorption at excitation wavelengths [m\ :sup:`-1`].
   :type aph_ex: ndarray
   :param wavelength_ex: Excitation wavelengths :math:`\lambda'` in nm.
   :type wavelength_ex: ndarray
   :param Ed_ex: Downwelling irradiance at excitation wavelengths
       (units cancel in the normalization).
   :type Ed_ex: ndarray
   :param Ed_em: Downwelling irradiance at the (peak) emission wavelength.
   :type Ed_em: float or ndarray
   :param mu_d: Mean cosine for downwelling irradiance. Default: ``raman.MU_D_DEFAULT``.
   :type mu_d: float, optional
   :param mu_f: Mean cosine for fluorescence emission (isotropic = 0.5).
   :type mu_f: float, optional
   :param phi_C: Fluorescence quantum yield. Default 0.02.
   :type phi_C: float
   :param double_gaussian: If True, use the double-Gaussian emission line
       (685 + 730 nm peaks). Default True.
   :type double_gaussian: bool
   :returns: Fluorescence contribution to Rrs in sr\ :sup:`-1`.
   :rtype: ndarray

   The function integrates the contribution from each excitation wavelength
   following Gordon (1979) / Sathyendranath & Platt (1998), shapes the result
   with the chlorophyll emission line, then converts the subsurface
   reflectance to above-surface Rrs.

   The function supports both single spectra (1D ``a_ex``) and MCMC chains
   (2D ``a_ex`` with shape ``(nsamples, nwave_ex)``).


Low-level Methods (``bing.rt.chl_fl``)
======================================

Emission Line Shape
-------------------

.. py:function:: emission_line_single_gaussian(wavelength, lambda_center=685.0, sigma=10.6)

   Single Gaussian emission line :math:`h_C(\lambda)`, normalized so that
   :math:`\int h_C(\lambda)\,d\lambda = 1`.

.. py:function:: emission_line_double_gaussian(wavelength, lambda_primary=685.0, sigma_primary=10.6, lambda_secondary=730.0, sigma_secondary=21.2, weight_primary=0.75)

   Weighted sum of two Gaussians for the PS II (685 nm) and PS I (730 nm)
   peaks:

   .. math::

       h_C(\lambda) = W\,G(\lambda;\lambda_1,\sigma_1)
                    + (1-W)\,G(\lambda;\lambda_2,\sigma_2)


Quantum Yield Methods
---------------------

.. py:function:: quantum_yield_constant(phi=0.02)

   Return a constant quantum yield :math:`\Phi_C`.

.. py:function:: quantum_yield_irradiance_dependent(PAR, phi_max=0.07, phi_min=0.01, E_k=100.0)

   Irradiance-dependent quantum yield (Morrison 2003):

   .. math::

       \Phi_C(E) = \Phi_{min} + (\Phi_{max} - \Phi_{min}) \frac{E_k}{E + E_k}

.. py:function:: quantum_yield_depth_profile(depth, K_PAR=0.05, PAR_surface=500.0, phi_max=0.07, phi_min=0.01, E_k=100.0)

   Depth profile combining Beer-Lambert PAR attenuation with the
   irradiance-dependent quantum yield.


Absorption Efficiency
---------------------

.. py:function:: absorption_efficiency(wavelength_ex, a_ph)

   Returns 1.0 within the photosynthetic excitation range
   (370-690 nm) and 0.0 outside.

.. py:function:: absorption_efficiency_weighted(wavelength_ex, a_ph, a_ph_ref=None)

   Weights by relative phytoplankton absorption :math:`a_{ph}/a_{ph,\rm ref}`
   within the excitation range.


Fluorescence Coefficients
-------------------------

.. py:function:: fluorescence_scattering_coeff(a_ph, phi_C=0.02)

   :math:`b_C(\lambda') = \Phi_C\,a_{ph}(\lambda')`.

.. py:function:: fluorescence_backscattering_coeff(a_ph, phi_C=0.02)

   :math:`b_{bC} = 0.5\,b_C = 0.5\,\Phi_C\,a_{ph}` (isotropic emission).


Phase Function
--------------

.. py:function:: fluorescence_phase_function(psi)

   Isotropic phase function :math:`\tilde{\beta}_C(\psi) = 1/(4\pi)`.

.. py:function:: fluorescence_backscatter_fraction()

   Returns 0.5 (the isotropic backscatter fraction).


Volume Scattering Function
--------------------------

.. py:function:: fluorescence_vsf(wavelength_ex, wavelength_em, psi, a_ph, phi_C=0.02, double_gaussian=False)

   Combines the scattering coefficient, wavelength redistribution and angular
   distribution into :math:`\beta_C(\lambda', \lambda, \psi)`.


Reflectance Contribution
------------------------

.. py:function:: calc_R_fluorescence(a_em, bb_em, a_ex, bb_ex, a_ph_ex, Ed_ratio=1.0, phi_C=0.02, mu_d=0.9, mu_f=0.5)

   Fluorescence subsurface reflectance contribution following Sathyendranath
   & Platt (1998):

   .. math::

       R^F(\lambda,0) = \frac{E_d(\lambda')}{E_d(\lambda)}
                      \cdot \frac{b_{bF}(\lambda')/\mu_d}{K(\lambda') + \kappa^F(\lambda)}

   where :math:`b_{bF} = 0.5\,\Phi_C\,a_{ph}` is the fluorescence
   backscattering coefficient, :math:`K = (a + b_b)/\mu_d` is the downwelling
   attenuation, and :math:`\kappa^F = (a + b_b)/\mu_f` is the upwelling
   attenuation for fluorescence.

.. py:function:: calc_R_fluorescence_integrated(wavelength_em, wavelength_ex, a_em, bb_em, a_ex, bb_ex, a_ph_ex, Ed, phi_C=0.02, mu_d=0.9, mu_f=0.5, double_gaussian=False)

   Subsurface fluorescence reflectance integrated over excitation wavelengths
   using a trapezoidal rule.


Fluorescence Line Height (FLH)
------------------------------

.. py:function:: calc_fluorescence_line_height(Rrs_665, Rrs_680, Rrs_709, lambda_665=665.0, lambda_680=680.0, lambda_709=709.0)

   Compute the Fluorescence Line Height. The baseline at the fluorescence
   band is linearly interpolated between 665 nm and 709 nm, and subtracted
   from the measured Rrs(680).

   Reference wavelengths vary by sensor:

   - **MODIS**: 667, 678, 748 nm
   - **MERIS / OLCI**: 665, 681, 709 nm

.. py:function:: calc_normalized_fluorescence_line_height(Rrs_665, Rrs_680, Rrs_709, lambda_665=665.0, lambda_680=680.0, lambda_709=709.0)

   Normalized FLH (nFLH), defined as the FLH divided by the baseline
   reflectance.


Convenience Methods
-------------------

.. py:function:: get_emission_spectrum(wavelength_ex, wavelength_em_range=None, n_points=100, double_gaussian=False)

   Return ``(wavelength_em, intensity)`` arrays for plotting the emission
   line shape.

.. py:function:: summary_at_wavelength(wavelength_ex, a_ph, phi_C=0.02)

   Return a dictionary of fluorescence quantities at a single excitation
   wavelength.


Physical Constants
==================

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Constant
     - Value
     - Description
   * - ``LAMBDA_FL_PRIMARY``
     - 685 nm
     - Primary emission wavelength (PS II)
   * - ``LAMBDA_FL_SECONDARY``
     - 730 nm
     - Secondary emission wavelength (PS I)
   * - ``SIGMA_FL_PRIMARY``
     - 10.6 nm
     - Primary peak standard deviation
   * - ``SIGMA_FL_SECONDARY``
     - 21.2 nm
     - Secondary peak standard deviation
   * - ``FWHM_FL_PRIMARY``
     - 25 nm
     - Primary peak FWHM
   * - ``FWHM_FL_SECONDARY``
     - 50 nm
     - Secondary peak FWHM
   * - ``WEIGHT_PRIMARY``
     - 0.75
     - Weight of primary peak in double-Gaussian
   * - ``PHI_FL_DEFAULT``
     - 0.02
     - Default quantum yield (HydroLight value)
   * - ``PHI_FL_HIGH_LIGHT``
     - 0.01
     - Quantum yield at high irradiance
   * - ``PHI_FL_LOW_LIGHT``
     - 0.07
     - Quantum yield at low irradiance
   * - ``LAMBDA_EX_MIN``
     - 370 nm
     - Minimum excitation wavelength
   * - ``LAMBDA_EX_MAX``
     - 690 nm
     - Maximum excitation wavelength


Using Fluorescence in the BING Fitting Workflow
===============================================

Chlorophyll fluorescence is enabled in BING fits through the parameter
configuration object generated by :func:`bing.parameters.p_ntuple.gen`. The
relevant fields are:

- ``include_Chl_fl`` -- include the fluorescence contribution in the forward
  model. Default ``False``.
- ``phi_C`` -- fluorescence quantum yield. Default 0.02.
- ``double_gaussian`` -- if ``True``, use the double-Gaussian (685+730 nm)
  emission line. Default ``True``.

These settings are extracted into a small dictionary, the **rt dict**, by
:func:`bing.rt.defs.rt_dict_from_p` (see :ref:`rt-dict-from-p`). The rt dict
is then passed through the chisq and MCMC fitting machinery to control which
radiative-transfer corrections are applied during forward modeling.


Physical Background
===================

Fluorescence Process
--------------------

Chlorophyll fluorescence occurs through:

1. **Absorption**: Light at 370-690 nm absorbed by photosynthetic pigments
2. **Excitation**: Chlorophyll molecules raised to higher energy states
3. **Energy dissipation**: Photosynthesis, NPQ (heat), or fluorescence
4. **Emission**: 685 nm (PS II) and 730 nm (PS I) Gaussian peaks


Quantum Yield Variability
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Condition
     - :math:`\Phi_C`
     - Notes
   * - High light (surface)
     - 0.005-0.01
     - NPQ reduces yield
   * - Low light (depth)
     - 0.05-0.07
     - Maximum yield
   * - Typical / average
     - 0.02
     - HydroLight default
   * - Nutrient stress
     - Variable
     - Can increase yield


Emission Spectrum Shape
-----------------------

**Single Gaussian (PS II only)**:

.. math::

   h_C(\lambda) = \frac{1}{\sigma\sqrt{2\pi}}
                  \exp\!\left[-\frac{(\lambda - 685)^2}{2\sigma^2}\right]

with :math:`\sigma = 10.6` nm (FWHM = 25 nm).

**Double Gaussian (PS II + PS I)**:

.. math::

   h_C(\lambda) = 0.75 \cdot G(685, 10.6) + 0.25 \cdot G(730, 21.2)


Reflectance Formulation
-----------------------

The fluorescence contribution to subsurface reflectance follows the
Gordon (1979) / Sathyendranath & Platt (1998) formulation:

.. math::

   R^F(\lambda, 0) = \frac{E_d(\lambda')}{E_d(\lambda)} \cdot
   \frac{b_{bF}(\lambda')/\mu_d(\lambda')}{K(\lambda') + \kappa^F(\lambda)}

with :math:`b_{bF} = 0.5\,\Phi_C\,a_{ph}`,
:math:`K = (a + b_b)/\mu_d`, and :math:`\kappa^F = (a + b_b)/\mu_f`.


Comparison with Raman Scattering
================================

Both chlorophyll fluorescence and Raman scattering are inelastic processes
that redistribute light to longer wavelengths.

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Property
     - Fluorescence
     - Raman
   * - Source
     - Phytoplankton pigments
     - Water molecules
   * - Excitation range
     - 370-690 nm
     - All visible wavelengths
   * - Emission
     - 685, 730 nm (fixed)
     - Shifted by ~3400 cm\ :sup:`-1`
   * - Angular distribution
     - Isotropic (:math:`b_b/b = 0.5`)
     - Nearly isotropic (:math:`b_b/b = 0.5`)
   * - Magnitude
     - Depends on Chl, varies widely
     - Fixed for given water type
   * - Variability
     - High (physiology-dependent)
     - Low (temperature / salinity)


Limitations and Notes
=====================

1. **Quantum yield uncertainty**: :math:`\Phi_C` varies significantly with
   phytoplankton physiology, light history, and nutrient status. The default
   value of 0.02 is approximate.
2. **PS I contribution**: The 730 nm peak is typically smaller than the
   primary PS II peak, but can be significant in certain conditions.
3. **Absorption model**: The Bricaud parameterization represents average
   phytoplankton. Specific phytoplankton groups may differ.
4. **Excitation irradiance**: The integrated Rrs method requires a
   downwelling irradiance spectrum (provided by ``correct_atmosphere``).
5. **Depth integration**: For satellite applications, the signal represents
   a depth-integrated fluorescence weighted by the light field.


References
==========

.. [Gordon1979] Gordon, H.R. (1979). "Diffuse reflectance of the ocean: the
   theory of its augmentation by chlorophyll a fluorescence at 685 nm,"
   *Appl. Opt.* 18, 1161-1166.

.. [Bricaud1995] Bricaud, A., Babin, M., Morel, A., and Claustre, H. (1995).
   "Variability in the chlorophyll-specific absorption coefficients of natural
   phytoplankton: Analysis and parameterization," *J. Geophys. Res.* 100,
   13321-13332.

.. [Maritorena2000] Maritorena, S., Morel, A., and Gentili, B. (2000).
   "Determination of the fluorescence quantum yield by oceanic phytoplankton
   in their natural habitat," *Appl. Opt.* 39, 6725-6737.

.. [SathyendranathPlatt1998] Sathyendranath, S. and Platt, T. (1998).
   "Ocean-colour model incorporating transspectral processes," *Appl. Opt.*
   37, 2216-2227.

.. [Behrenfeld2009] Behrenfeld, M.J. et al. (2009). "Satellite-detected
   fluorescence reveals global physiology of ocean phytoplankton,"
   *Biogeosciences* 6, 779-794.

.. [OOWB] Ocean Optics Web Book:
   https://www.oceanopticsbook.info/view/scattering/level-2/chlorophyll-fluorescence


See Also
========

* :py:mod:`bing.rt.chl_fl` -- low-level fluorescence methods
* :py:mod:`bing.rt.rrs` -- top-level Rrs method ``calc_Rrs_fluorescence``
* :py:mod:`bing.rt.defs` -- :func:`rt_dict_from_p` builds the rt dict
* :ref:`radiative_transfer` -- Gordon model and Raman correction
* :ref:`raman` -- Raman scattering corrections

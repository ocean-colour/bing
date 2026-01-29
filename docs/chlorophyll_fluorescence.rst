=========================
Chlorophyll Fluorescence
=========================

A Python module for calculating chlorophyll fluorescence inelastic emission
and its contribution to remote sensing reflectance (Rrs) in seawater.

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

- **Gordon (1979)** — Original diffuse reflectance augmentation by chlorophyll
  fluorescence at 685 nm
- **Maritorena et al. (2000)** — Fluorescence quantum yield determinations
- **Behrenfeld et al. (2009)** — Satellite-detected fluorescence and global
  phytoplankton physiology
- **Bricaud et al. (1995)** — Phytoplankton absorption parameterization
- **Sathyendranath & Platt (1998)** — Inelastic scattering radiative transfer

Key characteristics:

- **Excitation range**: 370-690 nm (absorbed by photosynthetic pigments)
- **Primary emission**: 685 nm (PS II), FWHM ~25 nm
- **Secondary emission**: ~730 nm (PS I), FWHM ~50 nm
- **Quantum yield**: 0.005-0.07 (varies with irradiance and physiology)


Quick Start
===========

.. code-block:: python

    import numpy as np
    from bing.rt import (
        calc_Rrs_fluorescence,
        calc_Rrs_fluorescence_simple,
        calc_fluorescence_spectrum,
        calc_fluorescence_line_height,
    )

    # Calculate fluorescence Rrs spectrum for Chl = 1 mg m^-3
    wavelengths = np.linspace(650, 750, 100)
    Rrs_fl = calc_Rrs_fluorescence_simple(wavelengths, Chl=1.0)
    print(f"Peak fluorescence Rrs: {Rrs_fl.max():.2e} sr^-1")

    # Get full fluorescence spectrum with components
    wavelength, Rrs_fl, components = calc_fluorescence_spectrum(
        Chl=1.0, return_components=True
    )

    # Calculate Fluorescence Line Height (FLH) from satellite data
    FLH = calc_fluorescence_line_height(
        Rrs_665=0.001,  # sr^-1
        Rrs_680=0.0015,
        Rrs_709=0.0005
    )


Module Architecture
===================

The chlorophyll fluorescence implementation is organized into two modules:

**Low-level functions** (``bing.rt.chl_fl``):
    Core physical calculations including emission line shapes, quantum yields,
    fluorescence coefficients, phase functions, and wavelength redistribution.

**High-level Rrs functions** (``bing.rt.rrs``):
    Convenient functions for calculating fluorescence contribution to remote
    sensing reflectance, integrated over excitation wavelengths, using the
    Bricaud parameterization for phytoplankton absorption.


Module Contents
===============

Physical Constants
------------------

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
     - Primary peak full width at half max
   * - ``FWHM_FL_SECONDARY``
     - 50 nm
     - Secondary peak full width at half max
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


Fluorescence Rrs Functions (bing.rt.rrs)
----------------------------------------

calc_Rrs_fluorescence
^^^^^^^^^^^^^^^^^^^^^

.. function:: calc_Rrs_fluorescence(wavelength, Chl, phi_C=0.02, wavelength_ex=None, mu_d=0.9, mu_f=0.5, double_gaussian=False)

   Calculate Rrs contribution from chlorophyll fluorescence integrating over
   excitation wavelengths.

   :param wavelength: Emission wavelength(s) in nm (typically 650-750 nm)
   :type wavelength: float or ndarray
   :param Chl: Chlorophyll-a concentration in mg m\ :sup:`-3`
   :type Chl: float
   :param phi_C: Fluorescence quantum yield (0-1)
   :type phi_C: float
   :param wavelength_ex: Excitation wavelengths for integration. If None, uses 400-680 nm
   :type wavelength_ex: ndarray or None
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_f: Mean cosine for fluorescence emission (isotropic = 0.5)
   :type mu_f: float
   :param double_gaussian: If True, use double Gaussian emission (685 + 730 nm peaks)
   :type double_gaussian: bool
   :returns: Fluorescence contribution to Rrs in sr\ :sup:`-1`
   :rtype: float or ndarray

   **Example:**

   .. code-block:: python

       wavelengths = np.linspace(650, 750, 100)
       Rrs_fl = calc_Rrs_fluorescence(wavelengths, Chl=1.0)
       print(f"Peak fluorescence Rrs: {Rrs_fl.max():.2e} sr^-1")


calc_Rrs_fluorescence_simple
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: calc_Rrs_fluorescence_simple(wavelength, Chl, phi_C=0.02, lambda_ex=440.0, mu_d=0.9, mu_f=0.5, double_gaussian=False)

   Fast single-excitation model for fluorescence Rrs calculation.

   This is a faster version that assumes all excitation occurs at a single
   wavelength (typically 440 nm, the blue absorption peak).

   :param wavelength: Emission wavelength(s) in nm
   :type wavelength: float or ndarray
   :param Chl: Chlorophyll-a concentration in mg m\ :sup:`-3`
   :type Chl: float
   :param phi_C: Fluorescence quantum yield
   :type phi_C: float
   :param lambda_ex: Excitation wavelength in nm
   :type lambda_ex: float
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_f: Mean cosine for fluorescence emission
   :type mu_f: float
   :param double_gaussian: If True, use double Gaussian emission
   :type double_gaussian: bool
   :returns: Fluorescence contribution to Rrs in sr\ :sup:`-1`
   :rtype: float or ndarray

   **Example:**

   .. code-block:: python

       wavelengths = np.linspace(650, 750, 100)
       Rrs_fl = calc_Rrs_fluorescence_simple(wavelengths, Chl=1.0)
       peak_idx = np.argmax(Rrs_fl)
       print(f"Peak at {wavelengths[peak_idx]:.0f} nm: {Rrs_fl[peak_idx]:.2e} sr^-1")


calc_Rrs_with_fluorescence
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: calc_Rrs_with_fluorescence(wavelength, a, bb, Chl, phi_C=0.02, in_G1=None, in_G2=None, mu_d=0.9, mu_f=0.5, double_gaussian=False)

   Calculate total Rrs including both elastic scattering and fluorescence.

   :param wavelength: Wavelength(s) in nm
   :type wavelength: float or ndarray
   :param a: Total absorption coefficient in m\ :sup:`-1`
   :type a: float or ndarray
   :param bb: Total backscattering coefficient in m\ :sup:`-1`
   :type bb: float or ndarray
   :param Chl: Chlorophyll-a concentration in mg m\ :sup:`-3`
   :type Chl: float
   :param phi_C: Fluorescence quantum yield
   :type phi_C: float
   :param in_G1, in_G2: Gordon coefficients. If None, use defaults
   :type in_G1, in_G2: float or None
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_f: Mean cosine for fluorescence emission
   :type mu_f: float
   :param double_gaussian: If True, use double Gaussian emission
   :type double_gaussian: bool
   :returns: Total Rrs (elastic + fluorescence) in sr\ :sup:`-1`
   :rtype: float or ndarray

   **Example:**

   .. code-block:: python

       wavelength = np.linspace(400, 750, 100)
       a = 0.1 * np.ones_like(wavelength)  # Simplified
       bb = 0.002 * np.ones_like(wavelength)
       Rrs_total = calc_Rrs_with_fluorescence(wavelength, a, bb, Chl=1.0)


calc_fluorescence_spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: calc_fluorescence_spectrum(Chl, wavelength=None, phi_C=0.02, double_gaussian=False, return_components=False)

   Calculate the full fluorescence Rrs spectrum for given Chl concentrations.

   Convenience function that returns the fluorescence spectrum across the
   emission range (typically 650-750 nm).

   :param Chl: Chlorophyll-a concentration(s) in mg m\ :sup:`-3`
   :type Chl: float or ndarray
   :param wavelength: Emission wavelengths in nm. If None, uses 650-750 nm at 1 nm resolution
   :type wavelength: ndarray or None
   :param phi_C: Fluorescence quantum yield
   :type phi_C: float
   :param double_gaussian: If True, use double Gaussian emission
   :type double_gaussian: bool
   :param return_components: If True, also return component spectra
   :type return_components: bool
   :returns: Fluorescence Rrs spectrum. If return_components=True, also returns
             (wavelength, Rrs_fl, components_dict)
   :rtype: ndarray or tuple

   **Example:**

   .. code-block:: python

       # Multiple Chl concentrations
       Chl_values = [0.1, 1.0, 10.0]  # mg m^-3
       Rrs_fl = calc_fluorescence_spectrum(Chl_values)
       print(f"Shape: {Rrs_fl.shape}")  # (3, 101)

       # With components
       wavelength, Rrs_fl, components = calc_fluorescence_spectrum(
           1.0, return_components=True
       )


calc_fluorescence_correction_factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: calc_fluorescence_correction_factor(wavelength, a, bb, Chl, phi_C=0.02, in_G1=None, in_G2=None)

   Calculate the multiplicative correction factor for fluorescence.

   Returns the ratio of total Rrs (elastic + fluorescence) to elastic Rrs,
   useful for understanding the fluorescence contribution.

   :param wavelength: Wavelength(s) in nm
   :type wavelength: float or ndarray
   :param a: Total absorption coefficient in m\ :sup:`-1`
   :type a: float or ndarray
   :param bb: Total backscattering coefficient in m\ :sup:`-1`
   :type bb: float or ndarray
   :param Chl: Chlorophyll-a concentration in mg m\ :sup:`-3`
   :type Chl: float
   :param phi_C: Fluorescence quantum yield
   :type phi_C: float
   :param in_G1, in_G2: Gordon coefficients
   :type in_G1, in_G2: float or None
   :returns: Correction factor: (Rrs_elastic + Rrs_fluorescence) / Rrs_elastic
   :rtype: float or ndarray

   .. note::
      Values > 1 indicate wavelengths where fluorescence adds signal.
      The correction is typically largest near 685 nm (the fluorescence peak).


IOP Functions (bing.rt.rrs)
---------------------------

calc_a_ph_bricaud
^^^^^^^^^^^^^^^^^

.. function:: calc_a_ph_bricaud(wavelength, Chl)

   Calculate phytoplankton absorption using Bricaud et al. (1995) parameterization.

   :param wavelength: Wavelength(s) in nm
   :type wavelength: float or ndarray
   :param Chl: Chlorophyll-a concentration in mg m\ :sup:`-3`
   :type Chl: float or ndarray
   :returns: Phytoplankton absorption coefficient a\ :sub:`ph` in m\ :sup:`-1`
   :rtype: float or ndarray

   The Bricaud model parameterizes phytoplankton absorption as:

   .. math::

       a_{ph}(\lambda, Chl) = A(\lambda) \times Chl^{E(\lambda)}

   where A(λ) and E(λ) are wavelength-dependent coefficients derived from
   measurements of diverse phytoplankton populations.

   **Example:**

   .. code-block:: python

       from bing.rt.rrs import calc_a_ph_bricaud

       # Single wavelength and Chl
       a_ph_440 = calc_a_ph_bricaud(440, 1.0)  # Blue peak

       # Spectrum
       wavelengths = np.arange(400, 700, 5)
       a_ph_spectrum = calc_a_ph_bricaud(wavelengths, 1.0)


calc_a_water
^^^^^^^^^^^^

.. function:: calc_a_water(wavelength)

   Calculate pure water absorption coefficient.

   Uses Pope & Fry (1997) and Smith & Baker (1981) data.

   :param wavelength: Wavelength(s) in nm
   :type wavelength: float or ndarray
   :returns: Pure water absorption coefficient a\ :sub:`w` in m\ :sup:`-1`
   :rtype: float or ndarray


calc_bb_water
^^^^^^^^^^^^^

.. function:: calc_bb_water(wavelength)

   Calculate pure water backscattering coefficient.

   Uses Morel (1974) wavelength dependence:

   .. math::

       b_{bw}(\lambda) = b_{bw}(500) \times (500/\lambda)^{4.32}

   :param wavelength: Wavelength(s) in nm
   :type wavelength: float or ndarray
   :returns: Pure water backscattering coefficient b\ :sub:`bw` in m\ :sup:`-1`
   :rtype: float or ndarray


Low-Level Functions (bing.rt.chl_fl)
------------------------------------

Emission Line Shape
~~~~~~~~~~~~~~~~~~~

emission_line_single_gaussian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: emission_line_single_gaussian(wavelength, lambda_center=685.0, sigma=10.6)

   Calculate single Gaussian emission line shape h\ :sub:`C`\ (λ).

   :param wavelength: Emission wavelength(s) in nm
   :type wavelength: float or ndarray
   :param lambda_center: Center wavelength of emission peak in nm
   :type lambda_center: float
   :param sigma: Standard deviation of Gaussian in nm
   :type sigma: float
   :returns: Normalized emission line shape h\ :sub:`C`\ (λ) in nm\ :sup:`-1`
   :rtype: float or ndarray

   The emission line shape is normalized such that ∫ h\ :sub:`C`\ (λ) dλ = 1.

   **Example:**

   .. code-block:: python

       from bing.rt.chl_fl import emission_line_single_gaussian

       wavelength = np.linspace(650, 750, 100)
       h_C = emission_line_single_gaussian(wavelength)
       plt.plot(wavelength, h_C)
       plt.xlabel('Wavelength [nm]')
       plt.ylabel('Emission line shape [nm$^{-1}$]')


emission_line_double_gaussian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: emission_line_double_gaussian(wavelength, lambda_primary=685.0, sigma_primary=10.6, lambda_secondary=730.0, sigma_secondary=21.2, weight_primary=0.75)

   Calculate double Gaussian emission line shape h\ :sub:`C`\ (λ).

   This model includes both the primary PS II peak at 685 nm and the
   secondary PS I peak at 730 nm.

   :param wavelength: Emission wavelength(s) in nm
   :type wavelength: float or ndarray
   :param lambda_primary: Center of primary peak in nm
   :type lambda_primary: float
   :param sigma_primary: Standard deviation of primary Gaussian in nm
   :type sigma_primary: float
   :param lambda_secondary: Center of secondary peak in nm
   :type lambda_secondary: float
   :param sigma_secondary: Standard deviation of secondary Gaussian in nm
   :type sigma_secondary: float
   :param weight_primary: Weight of primary peak (0-1)
   :type weight_primary: float
   :returns: Normalized emission line shape h\ :sub:`C`\ (λ) in nm\ :sup:`-1`
   :rtype: float or ndarray

   The emission is modeled as:

   .. math::

       h_C(\lambda) = W \times G(\lambda; \lambda_1, \sigma_1) + (1-W) \times G(\lambda; \lambda_2, \sigma_2)


Quantum Yield Functions
~~~~~~~~~~~~~~~~~~~~~~~

quantum_yield_constant
^^^^^^^^^^^^^^^^^^^^^^

.. function:: quantum_yield_constant(phi=0.02)

   Return constant quantum yield Φ\ :sub:`C`.

   :param phi: Quantum yield value (default 0.02)
   :type phi: float
   :returns: Quantum yield (dimensionless, 0-1)
   :rtype: float


quantum_yield_irradiance_dependent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: quantum_yield_irradiance_dependent(PAR, phi_max=0.07, phi_min=0.01, E_k=100.0)

   Calculate irradiance-dependent quantum yield Φ\ :sub:`C`\ (E).

   Quantum yield decreases with increasing irradiance due to
   nonphotochemical quenching (NPQ).

   :param PAR: Photosynthetically active radiation [μmol photons m\ :sup:`-2` s\ :sup:`-1`]
   :type PAR: float or ndarray
   :param phi_max: Maximum quantum yield at low light
   :type phi_max: float
   :param phi_min: Minimum quantum yield at high light
   :type phi_min: float
   :param E_k: Half-saturation irradiance [same units as PAR]
   :type E_k: float
   :returns: Irradiance-dependent quantum yield
   :rtype: float or ndarray

   Based on Morrison (2003) model:

   .. math::

       \Phi_C(E) = \Phi_{min} + (\Phi_{max} - \Phi_{min}) \times \frac{E_k}{E + E_k}

   **Example:**

   .. code-block:: python

       PAR = np.linspace(0, 1000, 100)
       phi_C = quantum_yield_irradiance_dependent(PAR)
       plt.plot(PAR, phi_C)
       plt.xlabel('PAR [μmol photons m$^{-2}$ s$^{-1}$]')
       plt.ylabel('Quantum yield')


quantum_yield_depth_profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: quantum_yield_depth_profile(depth, K_PAR=0.05, PAR_surface=500.0, phi_max=0.07, phi_min=0.01, E_k=100.0)

   Calculate depth-dependent quantum yield Φ\ :sub:`C`\ (z).

   Combines Beer-Lambert light attenuation with irradiance-dependent
   quantum yield.

   :param depth: Depth(s) in meters (positive downward)
   :type depth: float or ndarray
   :param K_PAR: Diffuse attenuation coefficient for PAR [m\ :sup:`-1`]
   :type K_PAR: float
   :param PAR_surface: Surface PAR [μmol photons m\ :sup:`-2` s\ :sup:`-1`]
   :type PAR_surface: float
   :param phi_max: Maximum quantum yield at low light
   :type phi_max: float
   :param phi_min: Minimum quantum yield at high light
   :type phi_min: float
   :param E_k: Half-saturation irradiance
   :type E_k: float
   :returns: Depth-dependent quantum yield
   :rtype: float or ndarray


Fluorescence Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~

fluorescence_scattering_coeff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: fluorescence_scattering_coeff(a_ph, phi_C=0.02)

   Calculate the fluorescence scattering coefficient b\ :sub:`C`\ (λ').

   :param a_ph: Phytoplankton absorption coefficient [m\ :sup:`-1`]
   :type a_ph: float or ndarray
   :param phi_C: Quantum yield
   :type phi_C: float
   :returns: Fluorescence scattering coefficient b\ :sub:`C` in m\ :sup:`-1`
   :rtype: float or ndarray

   The coefficient describes how much light at excitation wavelength λ' is
   converted to fluorescence per unit path length:

   .. math::

       b_C(\lambda') = \Phi_C \times a_{ph}(\lambda')


fluorescence_backscattering_coeff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: fluorescence_backscattering_coeff(a_ph, phi_C=0.02)

   Calculate the fluorescence backscattering coefficient b\ :sub:`bC`\ (λ').

   :param a_ph: Phytoplankton absorption coefficient [m\ :sup:`-1`]
   :type a_ph: float or ndarray
   :param phi_C: Quantum yield
   :type phi_C: float
   :returns: Fluorescence backscattering coefficient b\ :sub:`bC` in m\ :sup:`-1`
   :rtype: float or ndarray

   For isotropic fluorescence emission:

   .. math::

       b_{bC} = 0.5 \times b_C = 0.5 \times \Phi_C \times a_{ph}


Phase Function
~~~~~~~~~~~~~~

fluorescence_phase_function
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: fluorescence_phase_function(psi)

   Calculate the fluorescence phase function β̃\ :sub:`C`\ (ψ).

   Fluorescence emission is isotropic (equal in all directions).

   :param psi: Scattering angle(s) in radians (0 = forward, π = backward)
   :type psi: float or ndarray
   :returns: Phase function β̃\ :sub:`C`\ (ψ) in sr\ :sup:`-1`
   :rtype: float or ndarray

   For isotropic emission:

   .. math::

       \tilde{\beta}_C(\psi) = \frac{1}{4\pi} \text{ sr}^{-1}


fluorescence_backscatter_fraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: fluorescence_backscatter_fraction()

   Return the backscatter fraction for isotropic fluorescence emission.

   :returns: Backscatter fraction (0.5 for isotropic emission)
   :rtype: float


Reflectance Contribution
~~~~~~~~~~~~~~~~~~~~~~~~

calc_R_fluorescence
^^^^^^^^^^^^^^^^^^^

.. function:: calc_R_fluorescence(a_em, bb_em, a_ex, bb_ex, a_ph_ex, Ed_ratio=1.0, phi_C=0.02, mu_d=0.9, mu_f=0.5)

   Calculate fluorescence contribution to reflectance at the sea surface.

   Based on Gordon (1979) and Sathyendranath & Platt (1998) formulation.

   :param a_em: Total absorption at emission wavelength [m\ :sup:`-1`]
   :type a_em: float or ndarray
   :param bb_em: Total backscattering at emission wavelength [m\ :sup:`-1`]
   :type bb_em: float or ndarray
   :param a_ex: Total absorption at excitation wavelength [m\ :sup:`-1`]
   :type a_ex: float or ndarray
   :param bb_ex: Total backscattering at excitation wavelength [m\ :sup:`-1`]
   :type bb_ex: float or ndarray
   :param a_ph_ex: Phytoplankton absorption at excitation wavelength [m\ :sup:`-1`]
   :type a_ph_ex: float or ndarray
   :param Ed_ratio: Ratio of downwelling irradiance Ed(λ')/Ed(λ)
   :type Ed_ratio: float or ndarray
   :param phi_C: Quantum yield
   :type phi_C: float
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_f: Mean cosine for fluorescence emission (isotropic = 0.5)
   :type mu_f: float
   :returns: Fluorescence reflectance contribution R\ :sup:`F`\ (λ, 0)
   :rtype: float or ndarray

   Following the Sathyendranath & Platt (1998) formulation:

   .. math::

       R^F(\lambda, 0) = \frac{E_d(\lambda')}{E_d(\lambda)} \times \frac{b_{bF}(\lambda')/\mu_d}{K(\lambda') + \kappa^F(\lambda)}


Fluorescence Line Height
~~~~~~~~~~~~~~~~~~~~~~~~

calc_fluorescence_line_height
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: calc_fluorescence_line_height(Rrs_665, Rrs_680, Rrs_709, lambda_665=665.0, lambda_680=680.0, lambda_709=709.0)

   Calculate Fluorescence Line Height (FLH) from Rrs measurements.

   FLH is a standard product derived from satellite ocean color measurements
   that isolates the chlorophyll fluorescence signal from the background.

   :param Rrs_665: Remote sensing reflectance at ~665 nm [sr\ :sup:`-1`]
   :type Rrs_665: float or ndarray
   :param Rrs_680: Remote sensing reflectance at ~680 nm [sr\ :sup:`-1`]
   :type Rrs_680: float or ndarray
   :param Rrs_709: Remote sensing reflectance at ~709 nm [sr\ :sup:`-1`]
   :type Rrs_709: float or ndarray
   :param lambda_665: Wavelength of first baseline band
   :type lambda_665: float
   :param lambda_680: Wavelength of fluorescence band
   :type lambda_680: float
   :param lambda_709: Wavelength of second baseline band
   :type lambda_709: float
   :returns: Fluorescence Line Height [sr\ :sup:`-1`]
   :rtype: float or ndarray

   The FLH is calculated as:

   .. math::

       FLH = R_{rs}(680) - R_{rs,baseline}(680)

   where the baseline is linearly interpolated between 665 nm and 709 nm.

   Reference wavelengths vary by sensor:

   - **MODIS**: 667, 678, 748 nm
   - **MERIS/OLCI**: 665, 681, 709 nm

   **Example:**

   .. code-block:: python

       # From satellite Rrs data
       FLH = calc_fluorescence_line_height(
           Rrs_665=0.001,
           Rrs_680=0.0015,
           Rrs_709=0.0005
       )
       print(f"FLH = {FLH:.4e} sr^-1")


calc_normalized_fluorescence_line_height
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: calc_normalized_fluorescence_line_height(Rrs_665, Rrs_680, Rrs_709, lambda_665=665.0, lambda_680=680.0, lambda_709=709.0)

   Calculate normalized Fluorescence Line Height (nFLH).

   Normalized FLH accounts for variations in surface irradiance by
   dividing by the baseline reflectance.

   :param Rrs_665: Remote sensing reflectance at ~665 nm [sr\ :sup:`-1`]
   :type Rrs_665: float or ndarray
   :param Rrs_680: Remote sensing reflectance at ~680 nm [sr\ :sup:`-1`]
   :type Rrs_680: float or ndarray
   :param Rrs_709: Remote sensing reflectance at ~709 nm [sr\ :sup:`-1`]
   :type Rrs_709: float or ndarray
   :returns: Normalized Fluorescence Line Height [dimensionless]
   :rtype: float or ndarray


Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

get_emission_spectrum
^^^^^^^^^^^^^^^^^^^^^

.. function:: get_emission_spectrum(wavelength_ex, wavelength_em_range=None, n_points=100, double_gaussian=False)

   Get the chlorophyll fluorescence emission spectrum for a given excitation.

   :param wavelength_ex: Excitation wavelength in nm
   :type wavelength_ex: float
   :param wavelength_em_range: (min, max) emission wavelength range in nm. If None, uses (640, 800) nm
   :type wavelength_em_range: tuple or None
   :param n_points: Number of points in the spectrum
   :type n_points: int
   :param double_gaussian: If True, use double Gaussian model
   :type double_gaussian: bool
   :returns: (wavelength_em, intensity) arrays
   :rtype: tuple of ndarray

   **Example:**

   .. code-block:: python

       from bing.rt.chl_fl import get_emission_spectrum

       lambda_em, intensity = get_emission_spectrum(440)
       plt.plot(lambda_em, intensity)
       plt.xlabel('Emission wavelength [nm]')
       plt.ylabel('Relative intensity')


summary_at_wavelength
^^^^^^^^^^^^^^^^^^^^^

.. function:: summary_at_wavelength(wavelength_ex, a_ph, phi_C=0.02)

   Get a summary of fluorescence parameters at a given excitation wavelength.

   :param wavelength_ex: Excitation wavelength in nm
   :type wavelength_ex: float
   :param a_ph: Phytoplankton absorption at excitation wavelength [m\ :sup:`-1`]
   :type a_ph: float
   :param phi_C: Quantum yield
   :type phi_C: float
   :returns: Dictionary of fluorescence parameters
   :rtype: dict

   **Example:**

   .. code-block:: python

       >>> from bing.rt.chl_fl import summary_at_wavelength
       >>> summary_at_wavelength(440, a_ph=0.05)
       {
           'excitation_wavelength_nm': 440,
           'emission_peak_primary_nm': 685.0,
           'emission_peak_secondary_nm': 730.0,
           'emission_fwhm_primary_nm': 25.0,
           'emission_fwhm_secondary_nm': 50.0,
           'in_excitation_range': True,
           'quantum_yield': 0.02,
           'phytoplankton_absorption_m-1': 0.05,
           'fluorescence_scattering_coeff_m-1': 0.001,
           'fluorescence_backscatter_coeff_m-1': 0.0005,
           'backscatter_fraction': 0.5
       }


Physical Background
===================

Fluorescence Process
--------------------

Chlorophyll fluorescence occurs through the following mechanism:

1. **Absorption**: Light at wavelengths 370-690 nm is absorbed by
   photosynthetic pigments (primarily chlorophyll a and b)

2. **Excitation**: Absorbed photons excite chlorophyll molecules to
   higher energy states

3. **Energy dissipation**: Excited energy can be:

   - Used for photosynthesis
   - Dissipated as heat (nonphotochemical quenching)
   - Re-emitted as fluorescence

4. **Emission**: Fluorescence is emitted at 685 nm (PS II) and 730 nm (PS I)


Quantum Yield Variability
-------------------------

The fluorescence quantum yield (Φ\ :sub:`C`) varies significantly:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Condition
     - Φ\ :sub:`C`
     - Notes
   * - High light (surface)
     - 0.005-0.01
     - NPQ reduces yield
   * - Low light (depth)
     - 0.05-0.07
     - Maximum yield
   * - Typical/average
     - 0.02
     - HydroLight default
   * - Nutrient stress
     - Variable
     - Can increase yield


Emission Spectrum Shape
-----------------------

The emission spectrum is modeled using Gaussian functions:

**Single Gaussian (PS II only)**:

.. math::

   h_C(\lambda) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left[-\frac{(\lambda - 685)^2}{2\sigma^2}\right]

with σ = 10.6 nm (FWHM = 25 nm).

**Double Gaussian (PS II + PS I)**:

.. math::

   h_C(\lambda) = 0.75 \times G(685, 10.6) + 0.25 \times G(730, 21.2)

The double Gaussian model includes the secondary PS I peak at 730 nm,
which can be important for accurate modeling in certain conditions.


Reflectance Formulation
-----------------------

The fluorescence contribution to reflectance follows the Gordon (1979)
and Sathyendranath & Platt (1998) formulation for inelastic scattering:

.. math::

   R^F(\lambda, 0) = \frac{E_d(\lambda')}{E_d(\lambda)} \times
   \frac{b_{bF}(\lambda')/\mu_d(\lambda')}{K(\lambda') + \kappa^F(\lambda)}

where:

- b\ :sub:`bF` = 0.5 × Φ\ :sub:`C` × a\ :sub:`ph` is the fluorescence backscattering
- K(λ') = (a(λ') + b\ :sub:`b`\ (λ'))/μ\ :sub:`d` is downwelling attenuation
- κ\ :sup:`F`\ (λ) = (a(λ) + b\ :sub:`b`\ (λ))/μ\ :sub:`f` is upwelling attenuation


Bricaud Parameterization
------------------------

Phytoplankton absorption is calculated using the Bricaud et al. (1995)
parameterization:

.. math::

   a_{ph}(\lambda, Chl) = A(\lambda) \times Chl^{E(\lambda)}

where A(λ) and E(λ) are wavelength-dependent coefficients. Key values:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Wavelength (nm)
     - A
     - E
     - Notes
   * - 440
     - 0.0654
     - 0.668
     - Blue absorption peak
   * - 550
     - 0.0110
     - 0.715
     - Green minimum
   * - 675
     - 0.0260
     - 0.775
     - Red absorption peak
   * - 685
     - 0.0210
     - 0.776
     - Near fluorescence emission


Example: Full Fluorescence Analysis
===================================

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from bing.rt import (
        calc_Rrs,
        calc_Rrs_fluorescence_simple,
        calc_Rrs_with_fluorescence,
        calc_fluorescence_correction_factor,
        calc_a_ph_bricaud,
        calc_a_water,
        calc_bb_water,
    )
    from bing.rt.chl_fl import (
        emission_line_single_gaussian,
        emission_line_double_gaussian,
        quantum_yield_irradiance_dependent,
        quantum_yield_depth_profile,
    )

    # 1. Compare emission line shapes
    wavelength_em = np.linspace(640, 800, 200)

    h_single = emission_line_single_gaussian(wavelength_em)
    h_double = emission_line_double_gaussian(wavelength_em)

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(wavelength_em, h_single, label='Single Gaussian')
    plt.plot(wavelength_em, h_double, label='Double Gaussian')
    plt.xlabel('Emission wavelength [nm]')
    plt.ylabel('h$_C$(λ) [nm$^{-1}$]')
    plt.legend()
    plt.title('Emission Line Shape')

    # 2. Quantum yield depth profile
    depth = np.linspace(0, 100, 50)
    phi_C = quantum_yield_depth_profile(depth)

    plt.subplot(122)
    plt.plot(phi_C, depth)
    plt.xlabel('Quantum yield Φ$_C$')
    plt.ylabel('Depth [m]')
    plt.gca().invert_yaxis()
    plt.title('Quantum Yield Profile')
    plt.tight_layout()
    plt.show()

    # 3. Fluorescence Rrs for different Chl concentrations
    wavelength = np.linspace(650, 750, 100)
    Chl_values = [0.1, 0.5, 1.0, 5.0, 10.0]

    plt.figure(figsize=(8, 5))
    for Chl in Chl_values:
        Rrs_fl = calc_Rrs_fluorescence_simple(wavelength, Chl)
        plt.plot(wavelength, Rrs_fl * 1e4, label=f'Chl = {Chl} mg m$^{{-3}}$')

    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Rrs$_{fl}$ [×10$^{-4}$ sr$^{-1}$]')
    plt.legend()
    plt.title('Fluorescence Rrs vs Chlorophyll')
    plt.show()

    # 4. Fluorescence correction factor
    wavelength_full = np.linspace(400, 750, 200)
    Chl = 1.0

    # Calculate IOPs
    a_ph = calc_a_ph_bricaud(wavelength_full, Chl)
    a_w = calc_a_water(wavelength_full)
    bb_w = calc_bb_water(wavelength_full)

    a_total = a_w + a_ph
    bb_total = bb_w

    correction = calc_fluorescence_correction_factor(
        wavelength_full, a_total, bb_total, Chl
    )

    plt.figure(figsize=(8, 4))
    plt.plot(wavelength_full, correction)
    plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Correction factor')
    plt.title(f'Fluorescence Correction Factor (Chl = {Chl} mg m$^{{-3}}$)')
    plt.show()


Comparison with Raman Scattering
================================

Both chlorophyll fluorescence and Raman scattering are inelastic processes
that redistribute light to longer wavelengths. However, they differ in
several important ways:

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
     - Isotropic (b\ :sub:`b`/b = 0.5)
     - Nearly isotropic (b\ :sub:`b`/b = 0.5)
   * - Magnitude
     - Depends on Chl, varies widely
     - Fixed for given water type
   * - Variability
     - High (physiology-dependent)
     - Low (temperature/salinity)


Limitations and Notes
=====================

1. **Quantum yield uncertainty**: The quantum yield varies significantly
   with phytoplankton physiology, light history, and nutrient status.
   The default value of 0.02 is approximate.

2. **PS I contribution**: The secondary peak at 730 nm is typically smaller
   than the primary PS II peak, but can be significant in certain conditions.

3. **Absorption model**: The Bricaud parameterization represents average
   phytoplankton. Specific phytoplankton groups may differ.

4. **Clear water assumption**: The simplified model assumes particle
   backscattering is small compared to water backscattering.

5. **Depth integration**: For satellite applications, the signal represents
   depth-integrated fluorescence weighted by the light field.


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
   "Ocean-colour model incorporating transspectral processes," *Appl. Opt.* 37,
   2216-2227.

.. [Behrenfeld2009] Behrenfeld, M.J. et al. (2009). "Satellite-detected
   fluorescence reveals global physiology of ocean phytoplankton,"
   *Biogeosciences* 6, 779-794.

.. [PopeFry1997] Pope, R.M. and Fry, E.S. (1997). "Absorption spectrum
   (380-700 nm) of pure water. II. Integrating cavity measurements,"
   *Appl. Opt.* 36, 8710-8723.

.. [Morel1974] Morel, A. (1974). "Optical properties of pure water and pure
   sea water," *Optical Aspects of Oceanography*, Academic Press, 1-24.

.. [OOWB] Ocean Optics Web Book:
   https://www.oceanopticsbook.info/view/scattering/level-2/chlorophyll-fluorescence

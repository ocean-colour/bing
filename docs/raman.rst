================
Raman Scattering 
================

A Python module for calculating Raman scattering coefficients, backscattering 
coefficients, and related optical properties for pure water and seawater.

Overview
========

Raman scattering is an inelastic scattering process where incident light excites 
water molecules to a virtual energy state, which then decays to a vibrational 
level above the ground state. The emitted light has a longer wavelength (lower 
energy) than the incident light, with a characteristic wavenumber shift of 
~3400 cm\ :sup:`-1` corresponding to the O-H stretching mode of water.

This module implements the formulations from:

- **Bartlett et al. (1998)** — Raman scattering coefficient measurements and 
  wavelength dependence
- **Walrafen (1967)** — Wavenumber distribution function (emission spectrum shape)
- **Ge et al. (1993)** — Depolarization ratio and phase function
- **Mobley (1994)** — *Light and Water* textbook formulations

The next version will include temperature and salinity dependence 
based on **Artlett & Pask (2017)**.

And CDOM.


Quick Start
===========

.. code-block:: python

    import numpy as np
    from bing.rt.raman import (
        raman_scattering_coeff,
        raman_backscattering_coeff,
        excitation_to_emission_wavelength,
        summary_at_wavelength
    )

    # Raman scattering coefficient at 488 nm
    b_R = raman_scattering_coeff(488)
    print(f"b_R(488 nm) = {b_R:.2e} m⁻¹")  # 2.60e-04 m⁻¹

    # Backscattering coefficient
    b_bR = raman_backscattering_coeff(488)
    print(f"b_bR(488 nm) = {b_bR:.2e} m⁻¹")  # 1.30e-04 m⁻¹

    # Emission wavelength for 488 nm excitation
    lambda_em = excitation_to_emission_wavelength(488)
    print(f"Emission center: {lambda_em:.1f} nm")  # ~585 nm

    # Full summary
    summary = summary_at_wavelength(488)


Module Contents
===============

Constants
---------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Constant
     - Value
     - Description
   * - ``B_RAMAN_488_BARTLETT``
     - 2.7×10\ :sup:`-4` m\ :sup:`-1`
     - Bartlett et al. (1998) measurement
   * - ``B_RAMAN_488_DESIDERIO``
     - 2.4×10\ :sup:`-4` m\ :sup:`-1`
     - Desiderio (2000) measurement
   * - ``B_RAMAN_488_HYDROLIGHT``
     - 2.6×10\ :sup:`-4` m\ :sup:`-1`
     - HydroLight default value
   * - ``DEPOLARIZATION_RATIO``
     - 0.17
     - For water at ~3400 cm\ :sup:`-1`
   * - ``WAVENUMBER_SHIFT_CENTER``
     - 3400 cm\ :sup:`-1`
     - Center of O-H stretch band


Functions
---------

Scattering Coefficients
~~~~~~~~~~~~~~~~~~~~~~~

raman_scattering_coeff
^^^^^^^^^^^^^^^^^^^^^^

.. function:: raman_scattering_coeff(wavelength_excitation, reference_value=2.6e-4, units='energy')

   Calculate the total Raman scattering coefficient b\ :sub:`R`\ (λ').

   :param wavelength_excitation: Excitation wavelength(s) λ' in nm
   :type wavelength_excitation: float or ndarray
   :param reference_value: Reference value at 488 nm in m\ :sup:`-1`
   :type reference_value: float
   :param units: ``'energy'`` (exponent -5.5) or ``'photon'`` (exponent -5.3)
   :type units: str
   :returns: b\ :sub:`R` in m\ :sup:`-1`
   :rtype: float or ndarray

   **Example:**

   .. code-block:: python

       # Single wavelength
       b_R = raman_scattering_coeff(400)  # 7.76e-04 m⁻¹

       # Array of wavelengths
       wavelengths = np.array([400, 450, 500, 550])
       b_R = raman_scattering_coeff(wavelengths)


raman_backscattering_coeff
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: raman_backscattering_coeff(wavelength_excitation, reference_value=2.6e-4, units='energy')

   Calculate the Raman backscattering coefficient b\ :sub:`bR`\ (λ').

   The backscattering coefficient is the integral of the volume scattering 
   function over the backward hemisphere (scattering angles from 90° to 180°).

   :param wavelength_excitation: Excitation wavelength(s) λ' in nm
   :type wavelength_excitation: float or ndarray
   :param reference_value: Reference value at 488 nm in m\ :sup:`-1`
   :type reference_value: float
   :param units: ``'energy'`` or ``'photon'``
   :type units: str
   :returns: b\ :sub:`bR` in m\ :sup:`-1`
   :rtype: float or ndarray

   .. note::
      For the Raman phase function, the backscattering ratio 
      b\ :sub:`bR`/b\ :sub:`R` = 0.5 exactly.


Wavelength Functions
~~~~~~~~~~~~~~~~~~~~

excitation_to_emission_wavelength
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: excitation_to_emission_wavelength(lambda_ex, delta_nu=3400)

   Convert excitation wavelength to emission wavelength.

   :param lambda_ex: Excitation wavelength(s) in nm
   :type lambda_ex: float or ndarray
   :param delta_nu: Wavenumber shift in cm\ :sup:`-1`
   :type delta_nu: float
   :returns: Emission wavelength(s) in nm
   :rtype: float or ndarray

   **Example:**

   .. code-block:: python

       excitation_to_emission_wavelength(488)   # → 585.1 nm
       excitation_to_emission_wavelength(400)   # → 463.0 nm
       excitation_to_emission_wavelength(550)   # → 676.5 nm


emission_to_excitation_wavelength
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: emission_to_excitation_wavelength(lambda_em, delta_nu=3400)

   Inverse conversion from emission to excitation wavelength.

   :param lambda_em: Emission wavelength(s) in nm
   :type lambda_em: float or ndarray
   :param delta_nu: Wavenumber shift in cm\ :sup:`-1`
   :type delta_nu: float
   :returns: Excitation wavelength(s) in nm
   :rtype: float or ndarray


wavelength_redistribution
^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: wavelength_redistribution(lambda_ex, lambda_em)

   Calculate the wavelength redistribution function f\ :sub:`R`\ (λ', λ) in 
   nm\ :sup:`-1`.

   This gives the probability density that light at excitation wavelength λ', 
   if Raman scattered, will be emitted at wavelength λ.

   :param lambda_ex: Excitation wavelength λ' in nm
   :type lambda_ex: float
   :param lambda_em: Emission wavelength(s) λ in nm
   :type lambda_em: float or ndarray
   :returns: f\ :sub:`R` in nm\ :sup:`-1`
   :rtype: float or ndarray

   **Example:**

   .. code-block:: python

       # Emission spectrum for 488 nm excitation
       lambda_em = np.linspace(550, 620, 100)
       f_R = wavelength_redistribution(488, lambda_em)


Phase Function
~~~~~~~~~~~~~~

raman_phase_function
^^^^^^^^^^^^^^^^^^^^

.. function:: raman_phase_function(psi, rho=0.17, normalize=True)

   Calculate the Raman phase function β\ :sub:`R`\ (ψ) in sr\ :sup:`-1`.

   :param psi: Scattering angle(s) in radians
   :type psi: float or ndarray
   :param rho: Depolarization ratio
   :type rho: float
   :param normalize: If True, normalize so ∫β sin(ψ) dψ = 1/(2π)
   :type normalize: bool
   :returns: Phase function in sr\ :sup:`-1`
   :rtype: float or ndarray

   The phase function has the form:

   .. math::

       \beta_R(\psi) = \frac{1 + \delta \cos^2\psi}{4\pi \times \text{normalization}}

   where δ = (1 - ρ)/(1 + ρ) ≈ 0.71 for ρ = 0.17.


raman_phase_function_simple
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: raman_phase_function_simple(psi)

   Simplified form from the Ocean Optics Web Book:

   .. math::

       \beta_R(\psi) = \frac{1 + 0.53 \cos^2\psi}{4\pi \times 1.177}

   :param psi: Scattering angle(s) in radians
   :type psi: float or ndarray
   :returns: Phase function in sr\ :sup:`-1`
   :rtype: float or ndarray


Volume Scattering Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

raman_vsf
^^^^^^^^^

.. function:: raman_vsf(wavelength_excitation, wavelength_emission, psi, reference_value=2.6e-4, units='energy')

   Calculate the full Raman volume scattering function 
   β\ :sub:`R`\ (λ', λ, ψ) in m\ :sup:`-1` sr\ :sup:`-1` nm\ :sup:`-1`.

   This combines all three components:

   .. math::

       \beta_R(\lambda', \lambda, \psi) = b_R(\lambda') \times f_R(\lambda', \lambda) \times \tilde{\beta}_R(\psi)

   :param wavelength_excitation: Excitation wavelength λ' in nm
   :type wavelength_excitation: float
   :param wavelength_emission: Emission wavelength λ in nm
   :type wavelength_emission: float
   :param psi: Scattering angle(s) in radians
   :type psi: float or ndarray
   :param reference_value: Reference value at 488 nm in m\ :sup:`-1`
   :type reference_value: float
   :param units: ``'energy'`` or ``'photon'``
   :type units: str
   :returns: Volume scattering function in m\ :sup:`-1` sr\ :sup:`-1` nm\ :sup:`-1`
   :rtype: float or ndarray


Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

get_emission_spectrum
^^^^^^^^^^^^^^^^^^^^^

.. function:: get_emission_spectrum(wavelength_excitation, wavelength_emission_range=None, n_points=100)

   Get the Raman emission spectrum for a given excitation wavelength.

   :param wavelength_excitation: Excitation wavelength in nm
   :type wavelength_excitation: float
   :param wavelength_emission_range: (min, max) emission wavelength range in nm
   :type wavelength_emission_range: tuple or None
   :param n_points: Number of points in the spectrum
   :type n_points: int
   :returns: (wavelength_emission, intensity) arrays
   :rtype: tuple of ndarray

   **Example:**

   .. code-block:: python

       lambda_em, intensity = get_emission_spectrum(488)
       plt.plot(lambda_em, intensity)


summary_at_wavelength
^^^^^^^^^^^^^^^^^^^^^

.. function:: summary_at_wavelength(wavelength, units='energy')

   Get a dictionary of all Raman parameters at a given excitation wavelength.

   :param wavelength: Excitation wavelength in nm
   :type wavelength: float
   :param units: ``'energy'`` or ``'photon'``
   :type units: str
   :returns: Dictionary of Raman parameters
   :rtype: dict

   **Example:**

   .. code-block:: python

       >>> summary_at_wavelength(488)
       {
           'excitation_wavelength_nm': 488,
           'emission_center_nm': 585.08,
           'wavelength_shift_nm': 97.08,
           'wavenumber_shift_cm-1': 3400.0,
           'scattering_coeff_m-1': 2.6e-04,
           'backscattering_coeff_m-1': 1.3e-04,
           'backscattering_ratio': 0.5,
           'depolarization_ratio': 0.17,
           'units': 'energy'
       }


Physical Background
===================

Wavelength Dependence
---------------------

The Raman scattering coefficient follows a strong power-law wavelength dependence:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Units
     - Excitation Form
     - Emission Form
   * - Energy
     - b\ :sub:`R` ∝ (λ')\ :sup:`-5.5`
     - b\ :sub:`R` ∝ λ\ :sup:`-4.8`
   * - Photon
     - b\ :sub:`R` ∝ (λ')\ :sup:`-5.3`
     - b\ :sub:`R` ∝ λ\ :sup:`-4.6`

This strong wavelength dependence means Raman scattering is much more significant 
at blue wavelengths than at red wavelengths.


Emission Spectrum
-----------------

The emission spectrum is modeled using the Walrafen (1967) four-Gaussian 
parameterization:

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - Component
     - Weight
     - Center (cm\ :sup:`-1`)
     - FWHM (cm\ :sup:`-1`)
   * - 1
     - 0.41
     - 3250
     - 210
   * - 2
     - 0.39
     - 3425
     - 175
   * - 3
     - 0.10
     - 3530
     - 140
   * - 4
     - 0.10
     - 3625
     - 140

This produces the characteristic peak-and-shoulder shape of the water Raman 
emission band.


Wavelength Shifts
-----------------

The wavelength shift increases with excitation wavelength due to the fixed 
wavenumber shift:

.. list-table::
   :header-rows: 1
   :widths: 33 33 34

   * - λ' (nm)
     - λ (nm)
     - Shift (nm)
   * - 350
     - 397
     - 47
   * - 400
     - 463
     - 63
   * - 450
     - 531
     - 81
   * - 488
     - 585
     - 97
   * - 500
     - 602
     - 102
   * - 550
     - 677
     - 127


Backscattering
--------------

The Raman phase function is nearly symmetric about 90°, resulting in a 
backscattering ratio of exactly 0.5. This is similar to Rayleigh (molecular) 
scattering and quite different from particle scattering (which is strongly 
forward-peaked).


Example: Radiative Transfer Source Term
=======================================

In radiative transfer calculations, Raman scattering appears as a source term. 
For unpolarized light:

.. code-block:: python

    import numpy as np
    from raman_seawater import raman_vsf

    def raman_source_term(L_excitation, lambda_ex, lambda_em, psi):
        """
        Calculate Raman scattering source term for RT equation.
        
        Parameters
        ----------
        L_excitation : ndarray
            Radiance at excitation wavelength [W m⁻² sr⁻¹ nm⁻¹]
        lambda_ex : float
            Excitation wavelength [nm]
        lambda_em : float  
            Emission wavelength [nm]
        psi : ndarray
            Scattering angles [radians]
        
        Returns
        -------
        S_raman : ndarray
            Source term contribution [W m⁻³ sr⁻¹ nm⁻¹]
        """
        beta_R = raman_vsf(lambda_ex, lambda_em, psi)
        # Integrate over solid angle (simplified)
        S_raman = beta_R * L_excitation
        return S_raman


Example: Effect on Remote Sensing Reflectance
=============================================

Raman scattering can contribute significantly to remote sensing reflectance, 
especially in clear waters at red wavelengths:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from raman_seawater import raman_scattering_coeff, excitation_to_emission_wavelength

    # Wavelengths where Raman contributes to Rrs
    emission_wavelengths = np.array([550, 600, 650, 700])

    # Find corresponding excitation wavelengths
    for lam_em in emission_wavelengths:
        # Light at ~450-500 nm scatters into these wavelengths
        lam_ex = 1e7 / (1e7/lam_em + 3400)  # Approximate
        b_R = raman_scattering_coeff(lam_ex)
        print(f"Light at {lam_ex:.0f} nm → {lam_em} nm, b_R = {b_R:.2e} m⁻¹")


Limitations and Notes
=====================

1. **Temperature/Salinity Dependence**: The current implementation uses 
   Walrafen (1967) parameters for pure water at 25°C. The emission spectrum 
   shape varies slightly with temperature and salinity (see Artlett & Pask 2017).

2. **Pure Water vs Seawater**: Bartlett et al. (1998) found no statistically 
   significant difference between pure water and seawater Raman scattering 
   coefficients.

3. **Polarization**: The phase function averages over all polarization states. 
   For polarized radiative transfer, a full Mueller matrix treatment is required.

4. **Energy vs Photon Units**: Use ``units='energy'`` for radiative transfer 
   codes like HydroLight that work in energy units (W m\ :sup:`-2` nm\ :sup:`-1`), 
   and ``units='photon'`` for Monte Carlo simulations that track photon numbers.


References
==========

.. [Bartlett1998] Bartlett, J.S., Voss, K.J., Sathyendranath, S., and Vodacek, A. 
   (1998). "Raman scattering by pure water and seawater," *Appl. Opt.* 37, 
   3324-3332.

.. [Desiderio2000] Desiderio, R.A. (2000). "Application of the Raman scattering 
   coefficient of water to calculations in marine optics," *Appl. Opt.* 39, 
   1893-1894.

.. [Ge1993] Ge, Y., Gordon, H.R., and Voss, K.J. (1993). "Simulation of 
   inelastic-scattering contributions to the irradiance field in the ocean: 
   variation in Fraunhofer line depths," *Appl. Opt.* 32, 4028-4036.

.. [Mobley1994] Mobley, C.D. (1994). *Light and Water: Radiative Transfer in 
   Natural Waters*. Academic Press.

.. [Walrafen1967] Walrafen, G.E. (1967). "Raman spectral studies of the effects 
   of temperature on water structure," *J. Chem. Phys.* 47, 114-126.

.. [Artlett2017] Artlett, C.P. and Pask, H.M. (2017). "Optical remote sensing 
   of water temperature using Raman spectroscopy," *Opt. Express* 25, 2840-2851.

.. [OOWB] Ocean Optics Web Book: 
   https://www.oceanopticsbook.info/view/scattering/level-2/raman-scattering

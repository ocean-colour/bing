.. _raman:

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
- **Sathyendranath & Platt (1998)** — Ocean-colour model incorporating
  transspectral processes (Raman contribution to Rrs)

The module is organized into two parts:

**Low-level functions** (``bing.rt.raman``):
    Core Raman scattering coefficients, phase functions, emission spectra,
    and wavelength redistribution.

**Rrs functions** (``bing.rt.rrs``):
    Higher-level functions for calculating Raman contribution to remote
    sensing reflectance, including first-order and second-order terms.


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
   :no-index:

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
   :no-index:

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


Raman Rrs Functions (bing.rt.rrs)
---------------------------------

These functions calculate the contribution of Raman scattering to remote sensing
reflectance (Rrs), implementing the Sathyendranath & Platt (1998) formulation.

calc_Rrs_with_raman
~~~~~~~~~~~~~~~~~~~

.. function:: calc_Rrs_with_raman(a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=1.0, in_G1=None, in_G2=None, mu_d=0.9, mu_u=0.4, mu_R=0.5, include_second_order=True)

   Calculate remote sensing reflectance (Rrs) including Raman correction.

   This function computes Rrs using the Gordon model for elastic scattering
   and adds the Raman contribution from Sathyendranath & Platt (1998).

   :param a_em: Total absorption coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type a_em: float or ndarray
   :param bb_em: Total backscattering coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type bb_em: float or ndarray
   :param a_ex: Total absorption coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type a_ex: float or ndarray
   :param bb_ex: Total backscattering coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type bb_ex: float or ndarray
   :param bb_R: Raman backscattering coefficient at λ' [m\ :sup:`-1`]. Can be computed using ``bing.rt.raman.raman_backscattering_coeff()``
   :type bb_R: float or ndarray
   :param Ed_ratio: Ratio Ed(λ')/Ed(λ). Default is 1.0
   :type Ed_ratio: float or ndarray
   :param in_G1, in_G2: Gordon coefficients. If None, use defaults (0.0949, 0.0794)
   :type in_G1, in_G2: float or None
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_u: Mean cosine for upwelling irradiance
   :type mu_u: float
   :param mu_R: Mean cosine for Raman-scattered light
   :type mu_R: float
   :param include_second_order: If True, include second-order Raman terms. Default is True
   :type include_second_order: bool
   :returns: Remote sensing reflectance Rrs [sr\ :sup:`-1`]
   :rtype: float or ndarray

   **Example:**

   .. code-block:: python

       from bing.rt import raman
       from bing.rt.rrs import calc_Rrs_with_raman

       # At 520 nm emission with 443 nm excitation
       a_520, bb_520 = 0.05, 0.002
       a_443, bb_443 = 0.03, 0.003
       bb_R = raman.raman_backscattering_coeff(443)
       Rrs = calc_Rrs_with_raman(a_520, bb_520, a_443, bb_443, bb_R)


calc_R_raman_first_order
~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: calc_R_raman_first_order(a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=1.0, mu_d=0.9, mu_R=0.5)

   Calculate first-order Raman reflectance (Term 1).

   This is Eq. (11) from Sathyendranath & Platt (1998).

   :param a_em: Absorption coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type a_em: float or ndarray
   :param bb_em: Elastic backscattering coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type bb_em: float or ndarray
   :param a_ex: Absorption coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type a_ex: float or ndarray
   :param bb_ex: Elastic backscattering coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type bb_ex: float or ndarray
   :param bb_R: Raman backscattering coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type bb_R: float or ndarray
   :param Ed_ratio: Ratio of downwelling irradiance Ed(λ')/Ed(λ). Default is 1.0
   :type Ed_ratio: float or ndarray
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_R: Mean cosine for Raman-scattered light
   :type mu_R: float
   :returns: First-order Raman reflectance R\ :sup:`R`\ (λ, 0)
   :rtype: float or ndarray

   The first-order term represents direct Raman scattering:

   .. math::

       R^R(\lambda, 0) = \frac{E_d(\lambda')}{E_d(\lambda)} \times
       \frac{b_b^R(\lambda')/\mu_d(\lambda')}{K(\lambda') + \kappa^R(\lambda)}


calc_R_raman_RE
~~~~~~~~~~~~~~~

.. function:: calc_R_raman_RE(a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=1.0, s_E=1.0, mu_d=0.9, mu_R=0.5)

   Calculate second-order Raman-then-Elastic reflectance (Term 2).

   This is Eq. (18) from Sathyendranath & Platt (1998): a downward
   Raman-scattering event followed by an upward elastic-scattering event.

   :param a_em: Absorption coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type a_em: float or ndarray
   :param bb_em: Elastic backscattering coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type bb_em: float or ndarray
   :param a_ex: Absorption coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type a_ex: float or ndarray
   :param bb_ex: Elastic backscattering coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type bb_ex: float or ndarray
   :param bb_R: Raman backscattering coefficient at λ' [m\ :sup:`-1`]
   :type bb_R: float or ndarray
   :param Ed_ratio: Ratio Ed(λ')/Ed(λ). Default is 1.0
   :type Ed_ratio: float or ndarray
   :param s_E: Shape factor for elastic scattering
   :type s_E: float
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_R: Mean cosine for Raman-scattered light
   :type mu_R: float
   :returns: Second-order Raman-Elastic reflectance R\ :sup:`RE`\ (λ, 0)
   :rtype: float or ndarray

   .. note::
      This term is typically ~10% of the first-order Raman term.


calc_R_raman_ER
~~~~~~~~~~~~~~~

.. function:: calc_R_raman_ER(a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=1.0, s_E=1.0, mu_d=0.9, mu_u=0.4, mu_R=0.5)

   Calculate second-order Elastic-then-Raman reflectance (Term 3).

   This is Eq. (23) from Sathyendranath & Platt (1998): an upward
   elastic-scattering event followed by an upward Raman-scattering event.

   :param a_em: Absorption coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type a_em: float or ndarray
   :param bb_em: Elastic backscattering coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type bb_em: float or ndarray
   :param a_ex: Absorption coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type a_ex: float or ndarray
   :param bb_ex: Elastic backscattering coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type bb_ex: float or ndarray
   :param bb_R: Raman backscattering coefficient at λ' [m\ :sup:`-1`]
   :type bb_R: float or ndarray
   :param Ed_ratio: Ratio Ed(λ')/Ed(λ). Default is 1.0
   :type Ed_ratio: float or ndarray
   :param s_E: Shape factor for elastic scattering
   :type s_E: float
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_u: Mean cosine for upwelling irradiance
   :type mu_u: float
   :param mu_R: Mean cosine for Raman-scattered light
   :type mu_R: float
   :returns: Second-order Elastic-Raman reflectance R\ :sup:`ER`\ (λ, 0)
   :rtype: float or ndarray

   .. note::
      This term is typically ~10% of the first-order Raman term.


calc_R_raman_total
~~~~~~~~~~~~~~~~~~

.. function:: calc_R_raman_total(a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=1.0, s_E=1.0, mu_d=0.9, mu_u=0.4, mu_R=0.5, include_second_order=True)

   Calculate total Raman contribution to reflectance at the sea surface.

   This combines the first-order Raman term (Term 1) and optionally the
   two second-order terms (Terms 2 and 3) from Sathyendranath & Platt (1998).

   :param a_em: Absorption coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type a_em: float or ndarray
   :param bb_em: Elastic backscattering coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type bb_em: float or ndarray
   :param a_ex: Absorption coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type a_ex: float or ndarray
   :param bb_ex: Elastic backscattering coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type bb_ex: float or ndarray
   :param bb_R: Raman backscattering coefficient at λ' [m\ :sup:`-1`]
   :type bb_R: float or ndarray
   :param Ed_ratio: Ratio Ed(λ')/Ed(λ). Default is 1.0
   :type Ed_ratio: float or ndarray
   :param s_E: Shape factor for elastic scattering
   :type s_E: float
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_u: Mean cosine for upwelling irradiance
   :type mu_u: float
   :param mu_R: Mean cosine for Raman-scattered light
   :type mu_R: float
   :param include_second_order: If True, include second-order terms (RE and ER). Default is True
   :type include_second_order: bool
   :returns: Total Raman reflectance contribution
   :rtype: float or ndarray

   Total Raman reflectance = R\ :sup:`R` + R\ :sup:`RE` + R\ :sup:`ER`

   .. note::
      The second-order Raman-Raman terms (Terms 4 and 5) are neglected as they
      contribute only ~1% of the first-order term (see Section 4.A of the paper).


calc_R_total_with_raman
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: calc_R_total_with_raman(a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=1.0, s_E=1.0, mu_d=0.9, mu_u=0.4, mu_R=0.5, include_second_order=True)

   Calculate total reflectance including elastic and Raman contributions.

   This implements Eq. (26) from Sathyendranath & Platt (1998), the simplified
   model for clear waters.

   :param a_em: Absorption coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type a_em: float or ndarray
   :param bb_em: Elastic backscattering coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type bb_em: float or ndarray
   :param a_ex: Absorption coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type a_ex: float or ndarray
   :param bb_ex: Elastic backscattering coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type bb_ex: float or ndarray
   :param bb_R: Raman backscattering coefficient at λ' [m\ :sup:`-1`]
   :type bb_R: float or ndarray
   :param Ed_ratio: Ratio Ed(λ')/Ed(λ). Default is 1.0
   :type Ed_ratio: float or ndarray
   :param s_E: Shape factor for elastic scattering
   :type s_E: float
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_u: Mean cosine for upwelling irradiance
   :type mu_u: float
   :param mu_R: Mean cosine for Raman-scattered light
   :type mu_R: float
   :param include_second_order: If True, include second-order Raman terms. Default is True
   :type include_second_order: bool
   :returns: Total reflectance R(λ, 0) = R\ :sup:`E`\ (λ) + R\ :sup:`R`\ (λ) + R\ :sup:`RE`\ (λ) + R\ :sup:`ER`\ (λ)
   :rtype: float or ndarray


calc_raman_correction_factor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: calc_raman_correction_factor(a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=1.0, s_E=1.0, mu_d=0.9, mu_u=0.4, mu_R=0.5, include_second_order=True)
   :no-index:

   Calculate the multiplicative correction factor for Raman scattering.

   This returns the ratio of total reflectance (with Raman) to elastic
   reflectance, useful for correcting Rrs retrievals.

   :param a_em: Absorption coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type a_em: float or ndarray
   :param bb_em: Elastic backscattering coefficient at emission wavelength λ [m\ :sup:`-1`]
   :type bb_em: float or ndarray
   :param a_ex: Absorption coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type a_ex: float or ndarray
   :param bb_ex: Elastic backscattering coefficient at excitation wavelength λ' [m\ :sup:`-1`]
   :type bb_ex: float or ndarray
   :param bb_R: Raman backscattering coefficient at λ' [m\ :sup:`-1`]
   :type bb_R: float or ndarray
   :param Ed_ratio: Ratio Ed(λ')/Ed(λ). Default is 1.0
   :type Ed_ratio: float or ndarray
   :param s_E: Shape factor for elastic scattering
   :type s_E: float
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_u: Mean cosine for upwelling irradiance
   :type mu_u: float
   :param mu_R: Mean cosine for Raman-scattered light
   :type mu_R: float
   :param include_second_order: If True, include second-order Raman terms. Default is True
   :type include_second_order: bool
   :returns: Correction factor: (R\ :sup:`E` + R\ :sub:`raman`) / R\ :sup:`E`
   :rtype: float or ndarray

   To remove Raman contribution from measured reflectance:

   .. code-block:: python

       R_elastic = R_measured / correction_factor

   .. note::
      Typical correction factors range from 1.0 to ~1.25, with largest
      corrections in clear (oligotrophic) waters at longer wavelengths.


calc_R_elastic
~~~~~~~~~~~~~~

.. function:: calc_R_elastic(a, bb, s=1.0, mu_d=0.9, mu_u=0.4)

   Calculate elastic-scattering reflectance at the sea surface.

   This is Eq. (5) from Sathyendranath & Platt (1998), the standard
   elastic scattering term (Term 0 in Table 1).

   :param a: Absorption coefficient [m\ :sup:`-1`]
   :type a: float or ndarray
   :param bb: Backscattering coefficient [m\ :sup:`-1`]
   :type bb: float or ndarray
   :param s: Shape factor for scattering (s = 1 for isotropic/Rayleigh)
   :type s: float
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_u: Mean cosine for upwelling irradiance
   :type mu_u: float
   :returns: Elastic reflectance R\ :sup:`E`\ (λ, 0)
   :rtype: float or ndarray

   .. math::

       R^E(\lambda, 0) = \frac{\mu_u \times s}{\mu_u + \mu_d} \times \frac{b_b}{a + b_b}

   This is equivalent to the Gordon model when appropriate G factors are used.


calc_attenuation_coeffs
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: calc_attenuation_coeffs(a, bb, mu_d=0.9, mu_u=0.4, mu_R=0.5)

   Calculate diffuse attenuation coefficients for elastic and Raman scattering.

   Following Sathyendranath & Platt (1998) Eqs. (3) and (4).

   :param a: Absorption coefficient [m\ :sup:`-1`]
   :type a: float or ndarray
   :param bb: Backscattering coefficient [m\ :sup:`-1`]
   :type bb: float or ndarray
   :param mu_d: Mean cosine for downwelling irradiance
   :type mu_d: float
   :param mu_u: Mean cosine for upwelling irradiance
   :type mu_u: float
   :param mu_R: Mean cosine for Raman-scattered light (typically 0.5 for isotropic)
   :type mu_R: float
   :returns: Dictionary containing attenuation coefficients:

             - ``'K'``: Downwelling attenuation coefficient
             - ``'kappa_E'``: Upwelling attenuation for elastic scatter
             - ``'kappa_R'``: Upwelling attenuation for Raman scatter
             - ``'K_R'``: Downwelling attenuation after Raman scatter

   :rtype: dict


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


Integration with BING Fitting
=============================

BING supports Raman scattering correction during bio-optical parameter retrieval.
When enabled, the fitting algorithms account for the Raman contribution to measured
Rrs, improving IOP retrievals especially in clear oligotrophic waters at longer
wavelengths.

Enabling Raman Correction in Fitting
------------------------------------

When setting up fitting parameters, enable Raman correction:

.. code-block:: python

    from bing.parameters import standard
    from bing.fitting import l23

    # Enable Raman correction in parameter configuration
    params = standard.expb_pow(
        satellite='PACE',
        nsteps=40000,
        include_Raman=True,  # Enable Raman correction
        variable_Gordon=True  # Recommended with Raman correction
    )

    # Fit using L23 synthetic dataset
    chains, models, prep_dict, idx, extras = l23.fit_one(params, idx=170)

How Raman Correction Works in Fitting
-------------------------------------

When ``include_Raman=True``, the forward model:

1. Computes IOPs at emission wavelengths (model wavelengths) using the bio-optical models
2. Computes IOPs at Raman excitation wavelengths (~3400 cm\ :sup:`-1` shift)
3. Calculates the Raman correction factor at each wavelength
4. Applies the correction to the elastic Rrs

The models automatically initialize Raman-related quantities:

.. code-block:: python

    # Models automatically compute excitation wavelengths
    print(f"Emission wavelengths: {models[0].wave[:5]}...")
    print(f"Excitation wavelengths: {models[0].wave_ex[:5]}...")

    # Backscattering model provides Raman backscattering coefficient
    print(f"bb_R: {models[1].bb_R[:5]}...")

Direct Calculation with calc_Rrs
--------------------------------

The main ``calc_Rrs`` function in ``bing.rt.rrs`` accepts optional Raman parameters:

.. code-block:: python

    from bing.rt import rrs
    from bing.rt import raman
    import numpy as np

    # Define wavelengths
    wave = np.arange(400, 701, 5)
    wave_ex = raman.emission_to_excitation_wavelength(wave)

    # IOPs at emission and excitation wavelengths
    a_em = ...      # Absorption at emission wavelengths
    bb_em = ...     # Backscattering at emission wavelengths
    a_ex = ...      # Absorption at excitation wavelengths
    bb_ex = ...     # Backscattering at excitation wavelengths

    # Raman backscattering coefficient
    bb_R = raman.raman_backscattering_coeff(wave_ex)

    # Calculate Rrs with Raman correction
    Rrs = rrs.calc_Rrs(a_em, bb_em,
                       a_ex=a_ex, bb_ex=bb_ex, bb_R=bb_R)

Using the Correction Factor
---------------------------

You can also compute the Raman correction factor separately:

.. code-block:: python

    from bing.rt.rrs import calc_raman_correction_factor

    # Calculate correction factor
    corr = calc_raman_correction_factor(a_em, bb_em, a_ex, bb_ex, bb_R)

    # Apply to elastic Rrs
    Rrs_with_raman = Rrs_elastic * corr

    # Or remove Raman from measured Rrs
    Rrs_elastic_only = Rrs_measured / corr

Typical Correction Magnitudes
-----------------------------

The Raman correction factor varies with water type and wavelength:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Water Type
     - Correction Range
     - Notes
   * - Oligotrophic
     - 1.05 - 1.25
     - Largest corrections at red wavelengths
   * - Mesotrophic
     - 1.02 - 1.15
     - Moderate corrections
   * - Eutrophic
     - 1.01 - 1.08
     - Absorption dominates, smaller corrections

.. note::
   Raman corrections are most important for clear waters (Chl < 0.5 mg m\ :sup:`-3`) at
   wavelengths > 500 nm. For turbid waters, the correction is typically < 5%.


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

* Sathyendranath, S. and Platt, T. (1998).
  "Ocean-colour model incorporating transspectral processes," *Appl. Opt.* 37,
  2216-2227.

* Bartlett, J.S., Voss, K.J., Sathyendranath, S., and Vodacek, A.
  (1998). "Raman scattering by pure water and seawater," *Appl. Opt.* 37,
  3324-3332.

* Desiderio, R.A. (2000). "Application of the Raman scattering 
  coefficient of water to calculations in marine optics," *Appl. Opt.* 39,
  1893-1894.

* Ge, Y., Gordon, H.R., and Voss, K.J. (1993). "Simulation of 
  inelastic-scattering contributions to the irradiance field in the ocean:
  variation in Fraunhofer line depths," *Appl. Opt.* 32, 4028-4036.

* Mobley, C.D. (1994). *Light and Water: Radiative Transfer in 
  Natural Waters*. Academic Press.

* Walrafen, G.E. (1967). "Raman spectral studies of the effects 
  of temperature on water structure," *J. Chem. Phys.* 47, 114-126.

* Artlett, C.P. and Pask, H.M. (2017). "Optical remote sensing 
  of water temperature using Raman spectroscopy," *Opt. Express* 25, 2840-2851.

* Ocean Optics Web Book: 
  https://www.oceanopticsbook.info/view/scattering/level-2/raman-scattering

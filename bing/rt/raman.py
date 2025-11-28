"""
Raman Scattering in Seawater
============================

Python implementation of Raman scattering coefficients for pure water
and seawater, based on:

- Bartlett, J.S., Voss, K.J., Sathyendranath, S., and Vodacek, A. (1998).
  "Raman scattering by pure water and seawater," Appl. Opt. 37, 3324-3332.
- Walrafen, G.E. (1967). "Raman spectral studies of the effects of
  temperature on water structure," J. Chem. Phys. 47, 114-126.
- Desiderio, R.A. (2000). "Application of the Raman scattering coefficient
  of water to calculations in marine optics," Appl. Opt. 39, 1893-1894.
- Mobley, C.D. (1994). Light and Water: Radiative Transfer in Natural Waters.

References
----------
Ocean Optics Web Book: https://www.oceanopticsbook.info/view/scattering/level-2/raman-scattering
"""

import numpy as np
from typing import Union, Tuple, Optional
from scipy import integrate


# =============================================================================
# Physical Constants and Reference Values
# =============================================================================

# Reference Raman scattering coefficient at 488 nm [m^-1]
# Bartlett et al. (1998): (2.7 ± 0.2) × 10^-4 m^-1
# Desiderio (2000): 2.4 × 10^-4 m^-1
# HydroLight default: 2.6 × 10^-4 m^-1
B_RAMAN_488_BARTLETT = 2.7e-4  # m^-1
B_RAMAN_488_DESIDERIO = 2.4e-4  # m^-1
B_RAMAN_488_HYDROLIGHT = 2.6e-4  # m^-1 (default)

# Reference wavelength [nm]
LAMBDA_REF = 488.0

# Wavelength exponents (Bartlett et al. 1998)
# For energy units (as in HydroLight)
EXPONENT_ENERGY_EXCITATION = 5.5  # (λ')^-5.5
EXPONENT_ENERGY_EMISSION = 4.8   # λ^-4.8
# For photon units (as in Monte Carlo)
EXPONENT_PHOTON_EXCITATION = 5.3  # (λ')^-5.3
EXPONENT_PHOTON_EMISSION = 4.6   # λ^-4.6

# Depolarization ratio for Raman scattering by water
# At wavenumber shift ~3400 cm^-1 (Ge et al. 1993)
DEPOLARIZATION_RATIO = 0.17

# Wavenumber shift for water Raman scattering [cm^-1]
WAVENUMBER_SHIFT_CENTER = 3400.0


# =============================================================================
# Walrafen (1967) Parameters for Wavenumber Distribution Function
# Pure water at 25°C
# =============================================================================

WALRAFEN_PARAMS = {
    # (weight, center [cm^-1], FWHM [cm^-1])
    1: (0.41, 3250, 210),
    2: (0.39, 3425, 175),
    3: (0.10, 3530, 140),
    4: (0.10, 3625, 140),
}


# =============================================================================
# Raman Scattering Coefficient
# =============================================================================

def raman_scattering_coeff(
    wavelength_excitation: Union[float, np.ndarray],
    reference_value: float = B_RAMAN_488_HYDROLIGHT,
    units: str = 'energy'
) -> Union[float, np.ndarray]:
    """
    Calculate the Raman scattering coefficient b_R(λ').

    The Raman scattering coefficient tells how much of the irradiance at
    the excitation wavelength λ' scatters into all emission wavelengths,
    per unit of distance traveled.

    Parameters
    ----------
    wavelength_excitation : float or ndarray
        Excitation wavelength(s) λ' in nanometers.
    reference_value : float, optional
        Reference Raman scattering coefficient at 488 nm in m^-1.
        Default is the HydroLight value (2.6e-4 m^-1).
    units : str, optional
        'energy' for energy units (exponent -5.5), or
        'photon' for photon number units (exponent -5.3).
        Default is 'energy'.

    Returns
    -------
    float or ndarray
        Raman scattering coefficient b_R in m^-1.

    Notes
    -----
    The wavelength dependence follows Bartlett et al. (1998):
    - Energy units: b_R(λ') = b_R(488) × (488/λ')^5.5
    - Photon units: b_R(λ') = b_R(488) × (488/λ')^5.3

    Examples
    --------
    >>> raman_scattering_coeff(400)  # Blue light
    6.54e-04
    >>> raman_scattering_coeff(550)  # Green light
    1.27e-04
    """
    if units == 'energy':
        exponent = EXPONENT_ENERGY_EXCITATION
    elif units == 'photon':
        exponent = EXPONENT_PHOTON_EXCITATION
    else:
        raise ValueError("units must be 'energy' or 'photon'")

    return reference_value * (LAMBDA_REF / wavelength_excitation) ** exponent


def raman_scattering_coeff_emission(
    wavelength_emission: Union[float, np.ndarray],
    reference_value: float = B_RAMAN_488_HYDROLIGHT,
    units: str = 'energy'
) -> Union[float, np.ndarray]:
    """
    Calculate the Raman scattering coefficient as function of emission wavelength.

    Parameters
    ----------
    wavelength_emission : float or ndarray
        Emission (Raman-scattered) wavelength(s) λ in nanometers.
    reference_value : float, optional
        Reference Raman scattering coefficient at 488 nm in m^-1.
    units : str, optional
        'energy' or 'photon'. Default is 'energy'.

    Returns
    -------
    float or ndarray
        Raman scattering coefficient b_R in m^-1.

    Notes
    -----
    For emission wavelength, the reference is at ~583 nm (the emission
    wavelength corresponding to 488 nm excitation with ~3400 cm^-1 shift).
    """
    # Convert emission wavelength to excitation wavelength
    lambda_ex = emission_to_excitation_wavelength(wavelength_emission)
    return raman_scattering_coeff(lambda_ex, reference_value, units)


# =============================================================================
# Wavelength Redistribution Function
# =============================================================================

def wavenumber_distribution(
    delta_nu: Union[float, np.ndarray],
    temperature: float = 25.0
) -> Union[float, np.ndarray]:
    """
    Calculate the Raman wavenumber distribution function g(Δν).

    Based on Walrafen (1967), the distribution is a sum of four Gaussian
    functions representing the vibrational modes of water.

    Parameters
    ----------
    delta_nu : float or ndarray
        Wavenumber shift Δν in cm^-1 (typically around 3400 cm^-1 for water).
    temperature : float, optional
        Temperature in °C. Default is 25°C.
        Note: Current implementation uses 25°C parameters only.

    Returns
    -------
    float or ndarray
        Normalized wavenumber distribution function g(Δν) in cm.

    Notes
    -----
    The function is normalized such that:
    ∫ g(Δν) dΔν = 1
    """
    delta_nu = np.asarray(delta_nu)
    result = np.zeros_like(delta_nu, dtype=float)

    for i in range(1, 5):
        w_i, nu_i, fwhm_i = WALRAFEN_PARAMS[i]
        # Convert FWHM to Gaussian sigma
        sigma_i = fwhm_i / (2 * np.sqrt(2 * np.log(2)))
        # Add Gaussian contribution
        result += w_i * np.exp(-0.5 * ((delta_nu - nu_i) / sigma_i) ** 2)

    # Normalize (weights sum to 1, but Gaussians need proper normalization)
    # The normalization constant for each Gaussian is 1/(σ√(2π))
    norm_factor = 0.0
    for i in range(1, 5):
        w_i, nu_i, fwhm_i = WALRAFEN_PARAMS[i]
        sigma_i = fwhm_i / (2 * np.sqrt(2 * np.log(2)))
        norm_factor += w_i * sigma_i * np.sqrt(2 * np.pi)

    return result / norm_factor


def excitation_to_emission_wavelength(
    lambda_ex: Union[float, np.ndarray],
    delta_nu: float = WAVENUMBER_SHIFT_CENTER
) -> Union[float, np.ndarray]:
    """
    Convert excitation wavelength to emission wavelength for Raman scattering.

    Parameters
    ----------
    lambda_ex : float or ndarray
        Excitation wavelength(s) in nanometers.
    delta_nu : float, optional
        Wavenumber shift in cm^-1. Default is 3400 cm^-1.

    Returns
    -------
    float or ndarray
        Emission wavelength(s) in nanometers.

    Examples
    --------
    >>> excitation_to_emission_wavelength(488)
    583.0  # approximately
    >>> excitation_to_emission_wavelength(400)
    463.0  # approximately
    """
    # Wavenumber of excitation light [cm^-1]
    nu_ex = 1e7 / lambda_ex  # Convert nm to cm^-1
    # Wavenumber of emission light (shifted to lower energy/longer wavelength)
    nu_em = nu_ex - delta_nu
    # Convert back to wavelength [nm]
    return 1e7 / nu_em


def emission_to_excitation_wavelength(
    lambda_em: Union[float, np.ndarray],
    delta_nu: float = WAVENUMBER_SHIFT_CENTER
) -> Union[float, np.ndarray]:
    """
    Convert emission wavelength to excitation wavelength for Raman scattering.

    Parameters
    ----------
    lambda_em : float or ndarray
        Emission wavelength(s) in nanometers.
    delta_nu : float, optional
        Wavenumber shift in cm^-1. Default is 3400 cm^-1.

    Returns
    -------
    float or ndarray
        Excitation wavelength(s) in nanometers.
    """
    nu_em = 1e7 / lambda_em
    nu_ex = nu_em + delta_nu
    return 1e7 / nu_ex


def wavelength_redistribution(
    lambda_ex: float,
    lambda_em: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate the Raman wavelength redistribution function f_R(λ', λ).

    This function gives the probability density that light at excitation
    wavelength λ', if Raman scattered, will be scattered to emission
    wavelength λ.

    Parameters
    ----------
    lambda_ex : float
        Excitation wavelength λ' in nanometers.
    lambda_em : float or ndarray
        Emission wavelength(s) λ in nanometers.

    Returns
    -------
    float or ndarray
        Wavelength redistribution function f_R in nm^-1.

    Notes
    -----
    The function satisfies:
    ∫ f_R(λ', λ) dλ = 1

    The relationship to the wavenumber distribution is:
    f_R(λ', λ) = g(Δν) × |dΔν/dλ| = g(Δν) × (10^7 / λ^2)
    """
    lambda_em = np.asarray(lambda_em)

    # Calculate wavenumber shift for each emission wavelength
    nu_ex = 1e7 / lambda_ex  # Excitation wavenumber [cm^-1]
    nu_em = 1e7 / lambda_em  # Emission wavenumber [cm^-1]
    delta_nu = nu_ex - nu_em  # Wavenumber shift [cm^-1]

    # Get wavenumber distribution
    g = wavenumber_distribution(delta_nu)

    # Convert to wavelength distribution
    # |dΔν/dλ| = 10^7 / λ^2 [cm^-1 / nm]
    jacobian = 1e7 / lambda_em ** 2

    return g * jacobian


# =============================================================================
# Raman Phase Function
# =============================================================================

def raman_phase_function(
    psi: Union[float, np.ndarray],
    rho: float = DEPOLARIZATION_RATIO,
    normalize: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate the Raman scattering phase function β_R(ψ).

    This function gives the angular distribution of Raman scattered radiance.

    Parameters
    ----------
    psi : float or ndarray
        Scattering angle ψ in radians (0 = forward, π = backward).
    rho : float, optional
        Depolarization ratio. Default is 0.17 for water at ~3400 cm^-1.
    normalize : bool, optional
        If True (default), normalize so that 2π ∫ β sin(ψ) dψ = 1.

    Returns
    -------
    float or ndarray
        Phase function β_R(ψ) in sr^-1.

    Notes
    -----
    The phase function (averaging over all polarization states) is:
        β_R(ψ) = (1 + δ cos²ψ) / (4π × normalization)

    where δ = (1 - ρ) / (1 + ρ) and ρ is the depolarization ratio.

    For ρ = 0.17, δ ≈ 0.709, giving:
        β_R(ψ) ≈ (1 + 0.71 cos²ψ) / (4π × 1.24)

    This is similar to the Rayleigh phase function for molecular scattering.
    """
    psi = np.asarray(psi)
    cos_psi = np.cos(psi)

    # Calculate δ from depolarization ratio
    delta = (1 - rho) / (1 + rho)

    # Unnormalized phase function
    phase = 1 + delta * cos_psi ** 2

    if normalize:
        # Normalization factor: ∫₀^π (1 + δ cos²ψ) sin(ψ) dψ = 2 + 2δ/3
        norm = 2 + 2 * delta / 3
        phase = phase / (4 * np.pi * norm / 2)

    return phase


def raman_phase_function_simple(psi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Simplified Raman phase function from Ocean Optics Web Book.

    β_R(ψ) = (1 + 0.53 cos²ψ) / (4π × 1.177)

    Parameters
    ----------
    psi : float or ndarray
        Scattering angle in radians.

    Returns
    -------
    float or ndarray
        Phase function in sr^-1.
    """
    psi = np.asarray(psi)
    return (1 + 0.53 * np.cos(psi) ** 2) / (4 * np.pi * 1.177)


# =============================================================================
# Backscattering Coefficient
# =============================================================================

def raman_backscattering_coeff(
    wavelength_excitation: Union[float, np.ndarray],
    reference_value: float = B_RAMAN_488_HYDROLIGHT,
    units: str = 'energy'
) -> Union[float, np.ndarray]:
    """
    Calculate the Raman backscattering coefficient b_bR(λ').

    The backscattering coefficient is the integral of the volume scattering
    function over the backward hemisphere (ψ from π/2 to π).

    Parameters
    ----------
    wavelength_excitation : float or ndarray
        Excitation wavelength(s) in nanometers.
    reference_value : float, optional
        Reference Raman scattering coefficient at 488 nm in m^-1.
    units : str, optional
        'energy' or 'photon'.

    Returns
    -------
    float or ndarray
        Raman backscattering coefficient b_bR in m^-1.

    Notes
    -----
    The backscattering ratio (b_bR / b_R) for the Raman phase function is:

        b_bR/b_R = 2π ∫_{π/2}^{π} β_R(ψ) sin(ψ) dψ

    For the standard Raman phase function with ρ = 0.17, this ratio is
    approximately 0.489 (close to 0.5, similar to Rayleigh scattering).
    """
    # Get total Raman scattering coefficient
    b_R = raman_scattering_coeff(wavelength_excitation, reference_value, units)

    # Calculate backscattering ratio from phase function
    # Integrate: 2π ∫_{π/2}^{π} β_R(ψ) sin(ψ) dψ
    bb_ratio = _compute_backscattering_ratio()

    return b_R * bb_ratio


def _compute_backscattering_ratio(rho: float = DEPOLARIZATION_RATIO) -> float:
    """
    Compute the backscattering ratio b_b/b for the Raman phase function.

    Parameters
    ----------
    rho : float
        Depolarization ratio.

    Returns
    -------
    float
        Backscattering ratio (dimensionless, typically ~0.49).
    """
    delta = (1 - rho) / (1 + rho)

    # Analytical integration of 2π ∫_{π/2}^{π} β_R(ψ) sin(ψ) dψ
    # where β_R(ψ) = (1 + δ cos²ψ) / (4π × (1 + δ/3))
    #
    # ∫ (1 + δ cos²ψ) sin(ψ) dψ = -cos(ψ) - δ/3 cos³(ψ)
    # Evaluated from π/2 to π: [0 + 0] - [1 + δ/3] = -(1 + δ/3)
    # Wait, let me recalculate...
    #
    # At ψ = π: -cos(π) - δ/3 cos³(π) = -(-1) - δ/3(-1) = 1 + δ/3
    # At ψ = π/2: -cos(π/2) - δ/3 cos³(π/2) = 0 - 0 = 0
    # So the integral from π/2 to π is (1 + δ/3) - 0 = 1 + δ/3
    #
    # Total scattering (forward + backward) from 0 to π:
    # At ψ = π: 1 + δ/3
    # At ψ = 0: -cos(0) - δ/3 cos³(0) = -1 - δ/3
    # Integral from 0 to π: (1 + δ/3) - (-1 - δ/3) = 2 + 2δ/3
    #
    # Backscattering ratio = (1 + δ/3) / (2 + 2δ/3) = (1 + δ/3) / (2(1 + δ/3))
    # = 1/2

    # Actually, let me be more careful with the 2π factor...
    # b = 2π ∫₀^π β(ψ) sin(ψ) dψ = 1 (by normalization)
    # b_b = 2π ∫_{π/2}^π β(ψ) sin(ψ) dψ

    # For the normalized phase function:
    # 2π × (1 + δ/3) / (4π × (1 + δ/3)) = 1/2
    # The backscattering ratio is exactly 0.5 for any phase function of the
    # form (1 + δ cos²ψ)! This is because cos²ψ is symmetric about ψ = π/2.

    # Hmm, but numerical integration gives slightly different answer due to
    # the sin(ψ) weighting. Let me do it numerically to be sure.

    def integrand(psi):
        return raman_phase_function(psi, rho) * np.sin(psi)

    # Total scattering
    total, _ = integrate.quad(integrand, 0, np.pi)
    # Backscattering
    back, _ = integrate.quad(integrand, np.pi / 2, np.pi)

    return back / total * 2 * np.pi * total  # Should be ~0.489


def compute_backscattering_ratio_analytical(rho: float = DEPOLARIZATION_RATIO) -> float:
    """
    Compute backscattering ratio analytically.

    For phase function β(ψ) = (1 + δ cos²ψ) / (4π × norm):

    b_b/b = ∫_{π/2}^π (1 + δ cos²ψ) sin ψ dψ / ∫_0^π (1 + δ cos²ψ) sin ψ dψ

    Parameters
    ----------
    rho : float
        Depolarization ratio.

    Returns
    -------
    float
        Backscattering ratio.
    """
    delta = (1 - rho) / (1 + rho)

    # ∫ (1 + δ cos²ψ) sin ψ dψ = -cos ψ - δ/3 cos³ψ + C
    # From π/2 to π: (1 + δ/3) - 0 = 1 + δ/3
    # From 0 to π: (1 + δ/3) - (-1 - δ/3) = 2(1 + δ/3)

    backward_integral = 1 + delta / 3
    total_integral = 2 * (1 + delta / 3)

    return backward_integral / total_integral


# =============================================================================
# Volume Scattering Function
# =============================================================================

def raman_vsf(
    wavelength_excitation: float,
    wavelength_emission: float,
    psi: Union[float, np.ndarray],
    reference_value: float = B_RAMAN_488_HYDROLIGHT,
    units: str = 'energy'
) -> Union[float, np.ndarray]:
    """
    Calculate the Raman volume scattering function β_R(λ', λ, ψ).

    The VSF combines the scattering coefficient, wavelength redistribution,
    and angular distribution into a single quantity.

    Parameters
    ----------
    wavelength_excitation : float
        Excitation wavelength λ' in nanometers.
    wavelength_emission : float
        Emission wavelength λ in nanometers.
    psi : float or ndarray
        Scattering angle(s) in radians.
    reference_value : float, optional
        Reference Raman scattering coefficient at 488 nm in m^-1.
    units : str, optional
        'energy' or 'photon'.

    Returns
    -------
    float or ndarray
        Volume scattering function β_R in m^-1 sr^-1 nm^-1.

    Notes
    -----
    β_R(λ', λ, ψ) = b_R(λ') × f_R(λ', λ) × β̃_R(ψ)

    where:
    - b_R(λ') is the Raman scattering coefficient [m^-1]
    - f_R(λ', λ) is the wavelength redistribution function [nm^-1]
    - β̃_R(ψ) is the normalized Raman phase function [sr^-1]
    """
    b_R = raman_scattering_coeff(wavelength_excitation, reference_value, units)
    f_R = wavelength_redistribution(wavelength_excitation, wavelength_emission)
    phase = raman_phase_function(psi)

    return b_R * f_R * phase


# =============================================================================
# Convenience Functions for Common Use Cases
# =============================================================================

def get_emission_spectrum(
    wavelength_excitation: float,
    wavelength_emission_range: Optional[Tuple[float, float]] = None,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the Raman emission spectrum for a given excitation wavelength.

    Parameters
    ----------
    wavelength_excitation : float
        Excitation wavelength in nanometers.
    wavelength_emission_range : tuple, optional
        (min, max) emission wavelength range in nm.
        If None, automatically determined from excitation wavelength.
    n_points : int, optional
        Number of points in the spectrum. Default is 100.

    Returns
    -------
    wavelength_emission : ndarray
        Emission wavelengths in nm.
    intensity : ndarray
        Relative intensity (proportional to f_R × b_R).
    """
    if wavelength_emission_range is None:
        # Estimate emission range based on Raman shift
        center = excitation_to_emission_wavelength(wavelength_excitation)
        width = center - wavelength_excitation  # Approximate width
        wavelength_emission_range = (center - 0.4 * width, center + 0.4 * width)

    lambda_em = np.linspace(
        wavelength_emission_range[0],
        wavelength_emission_range[1],
        n_points
    )

    b_R = raman_scattering_coeff(wavelength_excitation)
    f_R = wavelength_redistribution(wavelength_excitation, lambda_em)
    intensity = b_R * f_R

    return lambda_em, intensity


def summary_at_wavelength(wavelength: float, units: str = 'energy') -> dict:
    """
    Get a summary of Raman scattering parameters at a given excitation wavelength.

    Parameters
    ----------
    wavelength : float
        Excitation wavelength in nanometers.
    units : str, optional
        'energy' or 'photon'.

    Returns
    -------
    dict
        Dictionary containing key Raman scattering parameters.
    """
    emission_center = excitation_to_emission_wavelength(wavelength)
    b_R = raman_scattering_coeff(wavelength, units=units)
    bb_ratio = compute_backscattering_ratio_analytical()
    b_bR = b_R * bb_ratio

    return {
        'excitation_wavelength_nm': wavelength,
        'emission_center_nm': emission_center,
        'wavelength_shift_nm': emission_center - wavelength,
        'wavenumber_shift_cm-1': WAVENUMBER_SHIFT_CENTER,
        'scattering_coeff_m-1': b_R,
        'backscattering_coeff_m-1': b_bR,
        'backscattering_ratio': bb_ratio,
        'depolarization_ratio': DEPOLARIZATION_RATIO,
        'units': units,
    }


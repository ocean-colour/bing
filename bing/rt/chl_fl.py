"""
Chlorophyll Fluorescence in Seawater
=====================================

Python implementation of chlorophyll fluorescence inelastic emission for
ocean color remote sensing, based on:

- Gordon, H.R. (1979). "Diffuse reflectance of the ocean: the theory of its
  augmentation by chlorophyll a fluorescence at 685 nm," Appl. Opt. 18, 1161-1166.
- Maritorena, S., Morel, A., and Gentili, B. (2000). "Determination of the
  fluorescence quantum yield by oceanic phytoplankton in their natural habitat,"
  Appl. Opt. 39, 6725-6737.
- Behrenfeld, M.J. et al. (2009). "Satellite-detected fluorescence reveals
  global physiology of ocean phytoplankton," Biogeosciences 6, 779-794.
- Ocean Optics Web Book: Chlorophyll Fluorescence

Theory
------
Chlorophyll a (Chl-a) fluorescence occurs when light absorbed by photosynthetic
pigments in phytoplankton is re-emitted at longer wavelengths. The primary
emission peak is centered at 685 nm, with a secondary peak around 730-740 nm.

The fluorescence emission adds to the elastic backscattering signal and can
contribute significantly to remote sensing reflectance in the red wavelengths,
particularly in waters with high chlorophyll concentration or under conditions
of photoinhibition.

Key characteristics:
- Excitation: 370-690 nm (absorbed by photosynthetic pigments)
- Primary emission: 685 nm (FWHM ~25 nm)
- Secondary emission: ~730 nm (FWHM ~50 nm)
- Quantum yield: 0.005-0.07 (varies with irradiance and physiology)

References
----------
Ocean Optics Web Book: https://www.oceanopticsbook.info/view/scattering/level-2/chlorophyll-fluorescence
"""

import numpy as np
from typing import Union, Optional, Tuple


# =============================================================================
# Physical Constants and Reference Values
# =============================================================================

# Primary chlorophyll fluorescence emission wavelength [nm]
LAMBDA_FL_PRIMARY = 685.0

# Secondary fluorescence emission wavelength [nm] (PS I contribution)
LAMBDA_FL_SECONDARY = 730.0

# Gaussian width parameters
# Primary emission line (FWHM = 25 nm)
SIGMA_FL_PRIMARY = 10.6  # Standard deviation [nm]
FWHM_FL_PRIMARY = 25.0   # Full width at half maximum [nm]

# Secondary emission line (FWHM = 50 nm)
SIGMA_FL_SECONDARY = 21.2  # Standard deviation [nm]
FWHM_FL_SECONDARY = 50.0   # Full width at half maximum [nm]

# Weight of primary peak in double-Gaussian model
WEIGHT_PRIMARY = 0.75
WEIGHT_SECONDARY = 0.25

# Quantum yield reference values
# High irradiance (surface, photoinhibition): 0.005-0.01
# Low irradiance (depth): up to 0.07
# HydroLight default: 0.02
PHI_FL_HIGH_LIGHT = 0.01    # High irradiance (near surface)
PHI_FL_LOW_LIGHT = 0.07     # Low irradiance (depth)
PHI_FL_DEFAULT = 0.02       # HydroLight default value

# Excitation wavelength range [nm]
LAMBDA_EX_MIN = 370.0
LAMBDA_EX_MAX = 690.0


# =============================================================================
# Emission Line Shape Functions
# =============================================================================

def emission_line_single_gaussian(
    wavelength: Union[float, np.ndarray],
    lambda_center: float = LAMBDA_FL_PRIMARY,
    sigma: float = SIGMA_FL_PRIMARY
) -> Union[float, np.ndarray]:
    """
    Calculate single Gaussian emission line shape h_C(λ).

    Parameters
    ----------
    wavelength : float or ndarray
        Emission wavelength(s) in nanometers.
    lambda_center : float, optional
        Center wavelength of emission peak in nm. Default is 685 nm.
    sigma : float, optional
        Standard deviation of Gaussian in nm. Default is 10.6 nm.

    Returns
    -------
    float or ndarray
        Normalized emission line shape h_C(λ) in nm^-1.

    Notes
    -----
    The emission line shape is normalized such that:
    ∫ h_C(λ) dλ = 1

    This function represents the probability density that fluorescence,
    if emitted, will be at wavelength λ.
    """
    wavelength = np.asarray(wavelength)

    # Gaussian emission line (normalized)
    h_C = (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
          np.exp(-0.5 * ((wavelength - lambda_center) / sigma) ** 2)

    return h_C


def emission_line_double_gaussian(
    wavelength: Union[float, np.ndarray],
    lambda_primary: float = LAMBDA_FL_PRIMARY,
    sigma_primary: float = SIGMA_FL_PRIMARY,
    lambda_secondary: float = LAMBDA_FL_SECONDARY,
    sigma_secondary: float = SIGMA_FL_SECONDARY,
    weight_primary: float = WEIGHT_PRIMARY
) -> Union[float, np.ndarray]:
    """
    Calculate double Gaussian emission line shape h_C(λ).

    This model includes both the primary PS II peak at 685 nm and the
    secondary PS I peak at 730 nm.

    Parameters
    ----------
    wavelength : float or ndarray
        Emission wavelength(s) in nanometers.
    lambda_primary : float, optional
        Center wavelength of primary peak in nm. Default is 685 nm.
    sigma_primary : float, optional
        Standard deviation of primary Gaussian in nm. Default is 10.6 nm.
    lambda_secondary : float, optional
        Center wavelength of secondary peak in nm. Default is 730 nm.
    sigma_secondary : float, optional
        Standard deviation of secondary Gaussian in nm. Default is 21.2 nm.
    weight_primary : float, optional
        Weight of primary peak (0-1). Default is 0.75.

    Returns
    -------
    float or ndarray
        Normalized emission line shape h_C(λ) in nm^-1.

    Notes
    -----
    h_C(λ) = W × G(λ; λ₁, σ₁) + (1-W) × G(λ; λ₂, σ₂)

    where W is the weight of the primary peak and G is a normalized Gaussian.
    """
    wavelength = np.asarray(wavelength)
    weight_secondary = 1.0 - weight_primary

    # Primary Gaussian (PS II, 685 nm)
    g1 = (1.0 / (sigma_primary * np.sqrt(2 * np.pi))) * \
         np.exp(-0.5 * ((wavelength - lambda_primary) / sigma_primary) ** 2)

    # Secondary Gaussian (PS I, 730 nm)
    g2 = (1.0 / (sigma_secondary * np.sqrt(2 * np.pi))) * \
         np.exp(-0.5 * ((wavelength - lambda_secondary) / sigma_secondary) ** 2)

    # Weighted sum
    h_C = weight_primary * g1 + weight_secondary * g2

    return h_C


# =============================================================================
# Absorption Efficiency Function
# =============================================================================

def absorption_efficiency(
    wavelength_ex: Union[float, np.ndarray],
    a_ph: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate the absorption efficiency g_C(λ') for fluorescence excitation.

    This function describes what fraction of absorbed light at excitation
    wavelength λ' can contribute to fluorescence emission.

    Parameters
    ----------
    wavelength_ex : float or ndarray
        Excitation wavelength(s) in nanometers.
    a_ph : float or ndarray
        Phytoplankton absorption coefficient at excitation wavelength(s) [m^-1].

    Returns
    -------
    float or ndarray
        Absorption efficiency g_C(λ'), dimensionless.

    Notes
    -----
    The absorption efficiency accounts for the fact that only light absorbed
    by photosynthetically active pigments can contribute to chlorophyll
    fluorescence. This is typically approximated as:

    g_C(λ') = a_ph(λ') / a_ph(440)

    where a_ph(440) is the phytoplankton absorption at the blue peak.

    In simpler models, g_C(λ') = 1 for all wavelengths in the excitation
    range (370-690 nm), meaning all absorbed light equally contributes
    to fluorescence regardless of excitation wavelength.
    """
    wavelength_ex = np.asarray(wavelength_ex)
    a_ph = np.asarray(a_ph)

    # Simple model: unit efficiency within excitation range
    # More sophisticated models would weight by absorption spectrum
    g_C = np.where(
        (wavelength_ex >= LAMBDA_EX_MIN) & (wavelength_ex <= LAMBDA_EX_MAX),
        np.ones_like(wavelength_ex),
        np.zeros_like(wavelength_ex)
    )

    return g_C


def absorption_efficiency_weighted(
    wavelength_ex: Union[float, np.ndarray],
    a_ph: Union[float, np.ndarray],
    a_ph_ref: Optional[float] = None
) -> Union[float, np.ndarray]:
    """
    Calculate weighted absorption efficiency g_C(λ') for fluorescence.

    This version weights the absorption efficiency by the relative
    phytoplankton absorption compared to a reference wavelength.

    Parameters
    ----------
    wavelength_ex : float or ndarray
        Excitation wavelength(s) in nanometers.
    a_ph : float or ndarray
        Phytoplankton absorption coefficient at excitation wavelength(s) [m^-1].
    a_ph_ref : float, optional
        Reference phytoplankton absorption for normalization [m^-1].
        If None, uses max(a_ph).

    Returns
    -------
    float or ndarray
        Weighted absorption efficiency g_C(λ'), dimensionless.
    """
    wavelength_ex = np.asarray(wavelength_ex)
    a_ph = np.asarray(a_ph)

    if a_ph_ref is None:
        a_ph_ref = np.max(a_ph)

    # Avoid division by zero
    if a_ph_ref == 0:
        return np.zeros_like(a_ph)

    # Weight by relative absorption within excitation range
    g_C = np.where(
        (wavelength_ex >= LAMBDA_EX_MIN) & (wavelength_ex <= LAMBDA_EX_MAX),
        a_ph / a_ph_ref,
        np.zeros_like(a_ph)
    )

    return g_C


# =============================================================================
# Quantum Yield Functions
# =============================================================================

def quantum_yield_constant(phi: float = PHI_FL_DEFAULT) -> float:
    """
    Return constant quantum yield Φ_C.

    Parameters
    ----------
    phi : float, optional
        Quantum yield value. Default is 0.02.

    Returns
    -------
    float
        Quantum yield (dimensionless, 0-1).
    """
    return phi


def quantum_yield_irradiance_dependent(
    PAR: Union[float, np.ndarray],
    phi_max: float = PHI_FL_LOW_LIGHT,
    phi_min: float = PHI_FL_HIGH_LIGHT,
    E_k: float = 100.0
) -> Union[float, np.ndarray]:
    """
    Calculate irradiance-dependent quantum yield Φ_C(E).

    Quantum yield decreases with increasing irradiance due to
    nonphotochemical quenching (NPQ).

    Parameters
    ----------
    PAR : float or ndarray
        Photosynthetically active radiation [μmol photons m^-2 s^-1]
        or [W m^-2]. Alternatively, can be a relative measure (0-1).
    phi_max : float, optional
        Maximum quantum yield at low light. Default is 0.07.
    phi_min : float, optional
        Minimum quantum yield at high light. Default is 0.01.
    E_k : float, optional
        Half-saturation irradiance [same units as PAR]. Default is 100.

    Returns
    -------
    float or ndarray
        Irradiance-dependent quantum yield.

    Notes
    -----
    Based on Morrison (2003) model:
    Φ_C(E) = Φ_min + (Φ_max - Φ_min) × E_k / (E + E_k)

    At low light (E → 0): Φ_C → Φ_max
    At high light (E → ∞): Φ_C → Φ_min
    """
    PAR = np.asarray(PAR)

    phi_C = phi_min + (phi_max - phi_min) * E_k / (PAR + E_k)

    return phi_C


def quantum_yield_depth_profile(
    depth: Union[float, np.ndarray],
    K_PAR: float = 0.05,
    PAR_surface: float = 500.0,
    phi_max: float = PHI_FL_LOW_LIGHT,
    phi_min: float = PHI_FL_HIGH_LIGHT,
    E_k: float = 100.0
) -> Union[float, np.ndarray]:
    """
    Calculate depth-dependent quantum yield Φ_C(z).

    Combines Beer-Lambert light attenuation with irradiance-dependent
    quantum yield.

    Parameters
    ----------
    depth : float or ndarray
        Depth(s) in meters (positive downward).
    K_PAR : float, optional
        Diffuse attenuation coefficient for PAR [m^-1]. Default is 0.05.
    PAR_surface : float, optional
        Surface PAR [μmol photons m^-2 s^-1]. Default is 500.
    phi_max : float, optional
        Maximum quantum yield at low light. Default is 0.07.
    phi_min : float, optional
        Minimum quantum yield at high light. Default is 0.01.
    E_k : float, optional
        Half-saturation irradiance. Default is 100.

    Returns
    -------
    float or ndarray
        Depth-dependent quantum yield.
    """
    depth = np.asarray(depth)

    # PAR at depth (Beer-Lambert)
    PAR_z = PAR_surface * np.exp(-K_PAR * depth)

    # Irradiance-dependent quantum yield
    phi_C = quantum_yield_irradiance_dependent(PAR_z, phi_max, phi_min, E_k)

    return phi_C


# =============================================================================
# Wavelength Redistribution Function
# =============================================================================

def wavelength_redistribution(
    wavelength_ex: float,
    wavelength_em: Union[float, np.ndarray],
    phi_C: float = PHI_FL_DEFAULT,
    double_gaussian: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate the fluorescence wavelength redistribution function f_C(λ', λ).

    This function gives the probability density that light absorbed at
    excitation wavelength λ', if it fluoresces, will be emitted at
    wavelength λ.

    Parameters
    ----------
    wavelength_ex : float
        Excitation wavelength λ' in nanometers.
    wavelength_em : float or ndarray
        Emission wavelength(s) λ in nanometers.
    phi_C : float, optional
        Quantum yield (efficiency). Default is 0.02.
    double_gaussian : bool, optional
        If True, use double Gaussian model. Default is False (single Gaussian).

    Returns
    -------
    float or ndarray
        Wavelength redistribution function f_C(λ', λ) in nm^-1.

    Notes
    -----
    f_C(λ', λ) = Φ_C × g_C(λ') × h_C(λ) × (λ'/λ)

    where:
    - Φ_C is the quantum yield
    - g_C(λ') is the absorption efficiency (=1 for 370-690 nm)
    - h_C(λ) is the emission line shape
    - λ'/λ converts from quantum to energy units

    The integral ∫ f_C(λ', λ) dλ = Φ_C × g_C(λ') gives the total
    fluorescence probability at excitation wavelength λ'.
    """
    wavelength_em = np.asarray(wavelength_em)

    # Absorption efficiency (1 within excitation range, 0 outside)
    g_C = 1.0 if LAMBDA_EX_MIN <= wavelength_ex <= LAMBDA_EX_MAX else 0.0

    # Emission line shape
    if double_gaussian:
        h_C = emission_line_double_gaussian(wavelength_em)
    else:
        h_C = emission_line_single_gaussian(wavelength_em)

    # Wavelength ratio (quantum to energy conversion)
    lambda_ratio = wavelength_ex / wavelength_em

    # Complete redistribution function
    f_C = phi_C * g_C * h_C * lambda_ratio

    return f_C


# =============================================================================
# Fluorescence Phase Function
# =============================================================================

def fluorescence_phase_function(
    psi: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate the fluorescence phase function β̃_C(ψ).

    Fluorescence emission is isotropic (equal in all directions).

    Parameters
    ----------
    psi : float or ndarray
        Scattering angle ψ in radians (0 = forward, π = backward).

    Returns
    -------
    float or ndarray
        Phase function β̃_C(ψ) in sr^-1.

    Notes
    -----
    For isotropic emission: β̃_C(ψ) = 1/(4π) sr^-1

    This satisfies the normalization:
    2π ∫₀^π β̃_C(ψ) sin(ψ) dψ = 1
    """
    psi = np.asarray(psi)
    return np.full_like(psi, 1.0 / (4 * np.pi), dtype=float)


def fluorescence_backscatter_fraction() -> float:
    """
    Return the backscatter fraction for isotropic fluorescence emission.

    Returns
    -------
    float
        Backscatter fraction (0.5 for isotropic emission).

    Notes
    -----
    For isotropic emission, exactly half of the fluorescence is emitted
    into the backward hemisphere:

    b_bf / b_f = 2π ∫_{π/2}^π β̃_C(ψ) sin(ψ) dψ = 0.5
    """
    return 0.5


# =============================================================================
# Fluorescence Scattering Coefficient
# =============================================================================

def fluorescence_scattering_coeff(
    a_ph: Union[float, np.ndarray],
    phi_C: float = PHI_FL_DEFAULT
) -> Union[float, np.ndarray]:
    """
    Calculate the fluorescence scattering coefficient b_C(λ').

    This coefficient describes how much light at excitation wavelength λ'
    is converted to fluorescence per unit path length.

    Parameters
    ----------
    a_ph : float or ndarray
        Phytoplankton absorption coefficient at excitation wavelength [m^-1].
    phi_C : float, optional
        Quantum yield. Default is 0.02.

    Returns
    -------
    float or ndarray
        Fluorescence scattering coefficient b_C in m^-1.

    Notes
    -----
    b_C(λ') = Φ_C × a_ph(λ')

    where Φ_C is the quantum yield and a_ph is the phytoplankton absorption.
    """
    return phi_C * np.asarray(a_ph)


def fluorescence_backscattering_coeff(
    a_ph: Union[float, np.ndarray],
    phi_C: float = PHI_FL_DEFAULT
) -> Union[float, np.ndarray]:
    """
    Calculate the fluorescence backscattering coefficient b_bC(λ').

    Parameters
    ----------
    a_ph : float or ndarray
        Phytoplankton absorption coefficient at excitation wavelength [m^-1].
    phi_C : float, optional
        Quantum yield. Default is 0.02.

    Returns
    -------
    float or ndarray
        Fluorescence backscattering coefficient b_bC in m^-1.

    Notes
    -----
    For isotropic fluorescence emission:
    b_bC = 0.5 × b_C = 0.5 × Φ_C × a_ph
    """
    b_f = fluorescence_scattering_coeff(a_ph, phi_C)
    return 0.5 * b_f


# =============================================================================
# Volume Scattering Function
# =============================================================================

def fluorescence_vsf(
    wavelength_ex: float,
    wavelength_em: float,
    psi: Union[float, np.ndarray],
    a_ph: float,
    phi_C: float = PHI_FL_DEFAULT,
    double_gaussian: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate the fluorescence volume scattering function β_C(λ', λ, ψ).

    The VSF combines the scattering coefficient, wavelength redistribution,
    and angular distribution.

    Parameters
    ----------
    wavelength_ex : float
        Excitation wavelength λ' in nanometers.
    wavelength_em : float
        Emission wavelength λ in nanometers.
    psi : float or ndarray
        Scattering angle(s) in radians.
    a_ph : float
        Phytoplankton absorption at excitation wavelength [m^-1].
    phi_C : float, optional
        Quantum yield. Default is 0.02.
    double_gaussian : bool, optional
        If True, use double Gaussian emission. Default is False.

    Returns
    -------
    float or ndarray
        Volume scattering function β_C in m^-1 sr^-1 nm^-1.

    Notes
    -----
    β_C(λ', λ, ψ) = b_C(λ') × f_C(λ', λ) × β̃_C(ψ)

    where:
    - b_C(λ') = Φ_C × a_ph(λ') is the fluorescence scattering coefficient
    - f_C(λ', λ) is the wavelength redistribution function
    - β̃_C(ψ) = 1/(4π) is the isotropic phase function
    """
    b_C = fluorescence_scattering_coeff(a_ph, phi_C)
    f_C = wavelength_redistribution(wavelength_ex, wavelength_em, phi_C, double_gaussian)
    phase = fluorescence_phase_function(psi)

    # Note: f_C already includes phi_C, but b_C also includes it
    # The correct formulation avoids double-counting:
    # β_C = a_ph × f_C × β̃_C  (absorption drives the process)
    return a_ph * f_C * phase


# =============================================================================
# Reflectance Contribution Functions
# =============================================================================

def calc_R_fluorescence(
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: Union[float, np.ndarray],
    bb_ex: Union[float, np.ndarray],
    a_ph_ex: Union[float, np.ndarray],
    Ed_ratio: Union[float, np.ndarray] = 1.0,
    phi_C: float = PHI_FL_DEFAULT,
    mu_d: float = 0.9,
    mu_f: float = 0.5
) -> Union[float, np.ndarray]:
    """
    Calculate fluorescence contribution to reflectance at the sea surface.

    Based on Gordon (1979) and Sathyendranath & Platt (1998) formulation
    for inelastic scattering.

    Parameters
    ----------
    a_em : float or ndarray
        Total absorption coefficient at emission wavelength λ [m^-1].
    bb_em : float or ndarray
        Total backscattering coefficient at emission wavelength λ [m^-1].
    a_ex : float or ndarray
        Total absorption coefficient at excitation wavelength λ' [m^-1].
    bb_ex : float or ndarray
        Total backscattering coefficient at excitation wavelength λ' [m^-1].
    a_ph_ex : float or ndarray
        Phytoplankton absorption at excitation wavelength λ' [m^-1].
    Ed_ratio : float or ndarray, optional
        Ratio of downwelling irradiance Ed(λ')/Ed(λ). Default is 1.0.
    phi_C : float, optional
        Quantum yield. Default is 0.02.
    mu_d : float, optional
        Mean cosine for downwelling irradiance. Default is 0.9.
    mu_f : float, optional
        Mean cosine for fluorescence emission (isotropic). Default is 0.5.

    Returns
    -------
    float or ndarray
        Fluorescence reflectance contribution R^F(λ, 0).

    Notes
    -----
    Following the same formulation as Raman scattering (Eq. 11 in S&P 1998):

    R^F(λ, 0) = [Ed(λ')/Ed(λ)] × [b_bF(λ')/μ_d(λ')] × 1/[K(λ') + κ^F(λ)]

    where:
    - b_bF(λ') = 0.5 × Φ_C × a_ph(λ') is the fluorescence backscattering
    - K(λ') = (a(λ') + bb(λ'))/μ_d is downwelling attenuation
    - κ^F(λ) = (a(λ) + bb(λ))/μ_f is upwelling attenuation for fluorescence
    """
    # Fluorescence backscattering coefficient
    bb_F = fluorescence_backscattering_coeff(a_ph_ex, phi_C)

    # Attenuation coefficients
    K_ex = (a_ex + bb_ex) / mu_d        # K(λ')
    kappa_F_em = (a_em + bb_em) / mu_f  # κ^F(λ)

    # Fluorescence reflectance (analogous to Raman first-order)
    R_F = Ed_ratio * (bb_F / mu_d) / (K_ex + kappa_F_em)

    return R_F


def calc_R_fluorescence_integrated(
    wavelength_em: Union[float, np.ndarray],
    wavelength_ex: np.ndarray,
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: np.ndarray,
    bb_ex: np.ndarray,
    a_ph_ex: np.ndarray,
    Ed: np.ndarray,
    phi_C: float = PHI_FL_DEFAULT,
    mu_d: float = 0.9,
    mu_f: float = 0.5,
    double_gaussian: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate fluorescence reflectance integrating over excitation wavelengths.

    This function integrates the fluorescence contribution from all excitation
    wavelengths that can contribute to emission at the given emission wavelength(s).

    Parameters
    ----------
    wavelength_em : float or ndarray
        Emission wavelength(s) λ in nanometers.
    wavelength_ex : ndarray
        Array of excitation wavelengths λ' in nanometers.
    a_em : float or ndarray
        Total absorption at emission wavelength(s) [m^-1].
    bb_em : float or ndarray
        Total backscattering at emission wavelength(s) [m^-1].
    a_ex : ndarray
        Total absorption at excitation wavelengths [m^-1].
    bb_ex : ndarray
        Total backscattering at excitation wavelengths [m^-1].
    a_ph_ex : ndarray
        Phytoplankton absorption at excitation wavelengths [m^-1].
    Ed : ndarray
        Downwelling irradiance at excitation wavelengths [W m^-2 nm^-1] or relative.
    phi_C : float, optional
        Quantum yield. Default is 0.02.
    mu_d : float, optional
        Mean cosine for downwelling irradiance. Default is 0.9.
    mu_f : float, optional
        Mean cosine for fluorescence emission. Default is 0.5.
    double_gaussian : bool, optional
        If True, use double Gaussian emission model. Default is False.

    Returns
    -------
    float or ndarray
        Integrated fluorescence reflectance R^F(λ, 0).

    Notes
    -----
    The total fluorescence at emission wavelength λ is:

    R^F(λ) = ∫ h_C(λ) × Ed(λ') × (λ'/λ) × [b_bF(λ')/μ_d] / [K(λ') + κ^F(λ)] dλ'

    where the integral is over all excitation wavelengths.
    """
    wavelength_em = np.atleast_1d(wavelength_em)
    wavelength_ex = np.asarray(wavelength_ex)
    a_ex = np.asarray(a_ex)
    bb_ex = np.asarray(bb_ex)
    a_ph_ex = np.asarray(a_ph_ex)
    Ed = np.asarray(Ed)

    # Ensure emission arrays are properly shaped
    a_em = np.atleast_1d(a_em)
    bb_em = np.atleast_1d(bb_em)

    # Attenuation at emission wavelength (upwelling fluorescence)
    kappa_F_em = (a_em + bb_em) / mu_f  # Shape: (n_em,)

    # Initialize output
    R_F = np.zeros_like(wavelength_em, dtype=float)

    # Integrate over excitation wavelengths
    for i, lambda_em in enumerate(wavelength_em):
        # Emission line shape at this emission wavelength
        if double_gaussian:
            h_C = emission_line_double_gaussian(lambda_em)
        else:
            h_C = emission_line_single_gaussian(lambda_em)

        # Skip if emission is negligible
        if h_C < 1e-10:
            continue

        # Sum contributions from all excitation wavelengths
        integrand = np.zeros(len(wavelength_ex))

        for j, lambda_ex in enumerate(wavelength_ex):
            # Check if excitation wavelength is in valid range
            if lambda_ex < LAMBDA_EX_MIN or lambda_ex > LAMBDA_EX_MAX:
                continue

            # Fluorescence backscattering coefficient
            bb_F = fluorescence_backscattering_coeff(a_ph_ex[j], phi_C)

            # Downwelling attenuation at excitation wavelength
            K_ex = (a_ex[j] + bb_ex[j]) / mu_d

            # Wavelength ratio (energy conversion)
            lambda_ratio = lambda_ex / lambda_em

            # Contribution from this excitation wavelength
            integrand[j] = Ed[j] * h_C * lambda_ratio * (bb_F / mu_d) / (K_ex + kappa_F_em[i])

        # Integrate using trapezoidal rule
        if len(wavelength_ex) > 1:
            R_F[i] = np.trapz(integrand, wavelength_ex)
        else:
            R_F[i] = integrand[0]

    return R_F if len(R_F) > 1 else R_F[0]


# =============================================================================
# Fluorescence Line Height (FLH) Calculation
# =============================================================================

def calc_fluorescence_line_height(
    Rrs_665: Union[float, np.ndarray],
    Rrs_680: Union[float, np.ndarray],
    Rrs_709: Union[float, np.ndarray],
    lambda_665: float = 665.0,
    lambda_680: float = 680.0,
    lambda_709: float = 709.0
) -> Union[float, np.ndarray]:
    """
    Calculate Fluorescence Line Height (FLH) from Rrs measurements.

    FLH is a standard product derived from satellite ocean color measurements
    that isolates the chlorophyll fluorescence signal from the background.

    Parameters
    ----------
    Rrs_665 : float or ndarray
        Remote sensing reflectance at ~665 nm [sr^-1].
    Rrs_680 : float or ndarray
        Remote sensing reflectance at ~680 nm (near fluorescence peak) [sr^-1].
    Rrs_709 : float or ndarray
        Remote sensing reflectance at ~709 nm [sr^-1].
    lambda_665 : float, optional
        Wavelength of first baseline band. Default is 665 nm.
    lambda_680 : float, optional
        Wavelength of fluorescence band. Default is 680 nm.
    lambda_709 : float, optional
        Wavelength of second baseline band. Default is 709 nm.

    Returns
    -------
    float or ndarray
        Fluorescence Line Height [sr^-1].

    Notes
    -----
    FLH = Rrs(680) - Rrs_baseline(680)

    where the baseline is linearly interpolated between 665 nm and 709 nm:
    Rrs_baseline(680) = Rrs(665) + (Rrs(709) - Rrs(665)) × (680-665)/(709-665)

    This baseline subtraction removes the elastic backscattering contribution,
    leaving primarily the fluorescence signal.

    Reference wavelengths vary by sensor:
    - MODIS: 667, 678, 748 nm
    - MERIS/OLCI: 665, 681, 709 nm
    """
    # Linear interpolation weight
    w = (lambda_680 - lambda_665) / (lambda_709 - lambda_665)

    # Baseline at fluorescence band
    Rrs_baseline = Rrs_665 + w * (Rrs_709 - Rrs_665)

    # Fluorescence line height
    FLH = Rrs_680 - Rrs_baseline

    return FLH


def calc_normalized_fluorescence_line_height(
    Rrs_665: Union[float, np.ndarray],
    Rrs_680: Union[float, np.ndarray],
    Rrs_709: Union[float, np.ndarray],
    lambda_665: float = 665.0,
    lambda_680: float = 680.0,
    lambda_709: float = 709.0
) -> Union[float, np.ndarray]:
    """
    Calculate normalized Fluorescence Line Height (nFLH).

    Normalized FLH accounts for variations in surface irradiance by
    dividing by the baseline reflectance.

    Parameters
    ----------
    Rrs_665 : float or ndarray
        Remote sensing reflectance at ~665 nm [sr^-1].
    Rrs_680 : float or ndarray
        Remote sensing reflectance at ~680 nm [sr^-1].
    Rrs_709 : float or ndarray
        Remote sensing reflectance at ~709 nm [sr^-1].
    lambda_665 : float, optional
        Wavelength of first baseline band. Default is 665 nm.
    lambda_680 : float, optional
        Wavelength of fluorescence band. Default is 680 nm.
    lambda_709 : float, optional
        Wavelength of second baseline band. Default is 709 nm.

    Returns
    -------
    float or ndarray
        Normalized Fluorescence Line Height [dimensionless].
    """
    FLH = calc_fluorescence_line_height(
        Rrs_665, Rrs_680, Rrs_709,
        lambda_665, lambda_680, lambda_709
    )

    # Baseline at fluorescence band
    w = (lambda_680 - lambda_665) / (lambda_709 - lambda_665)
    Rrs_baseline = Rrs_665 + w * (Rrs_709 - Rrs_665)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        nFLH = np.where(Rrs_baseline > 0, FLH / Rrs_baseline, 0.0)

    return nFLH


# =============================================================================
# Emission Spectrum Convenience Functions
# =============================================================================

def get_emission_spectrum(
    wavelength_ex: float,
    wavelength_em_range: Optional[Tuple[float, float]] = None,
    n_points: int = 100,
    double_gaussian: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the chlorophyll fluorescence emission spectrum for a given excitation.

    Parameters
    ----------
    wavelength_ex : float
        Excitation wavelength in nanometers.
    wavelength_em_range : tuple, optional
        (min, max) emission wavelength range in nm.
        If None, uses (640, 800) nm.
    n_points : int, optional
        Number of points in the spectrum. Default is 100.
    double_gaussian : bool, optional
        If True, use double Gaussian model. Default is False.

    Returns
    -------
    wavelength_em : ndarray
        Emission wavelengths in nm.
    intensity : ndarray
        Relative emission intensity (emission line shape).
    """
    if wavelength_em_range is None:
        wavelength_em_range = (640.0, 800.0)

    lambda_em = np.linspace(
        wavelength_em_range[0],
        wavelength_em_range[1],
        n_points
    )

    if double_gaussian:
        intensity = emission_line_double_gaussian(lambda_em)
    else:
        intensity = emission_line_single_gaussian(lambda_em)

    return lambda_em, intensity


def summary_at_wavelength(
    wavelength_ex: float,
    a_ph: float,
    phi_C: float = PHI_FL_DEFAULT
) -> dict:
    """
    Get a summary of fluorescence parameters at a given excitation wavelength.

    Parameters
    ----------
    wavelength_ex : float
        Excitation wavelength in nanometers.
    a_ph : float
        Phytoplankton absorption at excitation wavelength [m^-1].
    phi_C : float, optional
        Quantum yield. Default is 0.02.

    Returns
    -------
    dict
        Dictionary containing key fluorescence parameters.
    """
    in_excitation_range = LAMBDA_EX_MIN <= wavelength_ex <= LAMBDA_EX_MAX

    b_C = fluorescence_scattering_coeff(a_ph, phi_C) if in_excitation_range else 0.0
    bb_C = fluorescence_backscattering_coeff(a_ph, phi_C) if in_excitation_range else 0.0

    return {
        'excitation_wavelength_nm': wavelength_ex,
        'emission_peak_primary_nm': LAMBDA_FL_PRIMARY,
        'emission_peak_secondary_nm': LAMBDA_FL_SECONDARY,
        'emission_fwhm_primary_nm': FWHM_FL_PRIMARY,
        'emission_fwhm_secondary_nm': FWHM_FL_SECONDARY,
        'in_excitation_range': in_excitation_range,
        'quantum_yield': phi_C,
        'phytoplankton_absorption_m-1': a_ph,
        'fluorescence_scattering_coeff_m-1': b_C,
        'fluorescence_backscatter_coeff_m-1': bb_C,
        'backscatter_fraction': 0.5,
    }




def calc_Rrs_with_fluorescence(
    wavelength: Union[float, np.ndarray],
    a: Union[float, np.ndarray],
    bb: Union[float, np.ndarray],
    Chl: float,
    phi_C: float = 0.02,
    in_G1: Optional[float] = None,
    in_G2: Optional[float] = None,
    mu_d: Optional[float] = None,
    mu_f: Optional[float] = None,
    double_gaussian: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate total Rrs including elastic scattering and fluorescence.

    Parameters
    ----------
    wavelength : float or ndarray
        Wavelength(s) in nanometers.
    a : float or ndarray
        Total absorption coefficient [m^-1].
    bb : float or ndarray
        Total backscattering coefficient [m^-1].
    Chl : float
        Chlorophyll-a concentration in mg m^-3.
    phi_C : float, optional
        Fluorescence quantum yield. Default is 0.02.
    in_G1, in_G2 : float, optional
        Gordon coefficients. If None, use defaults.
    mu_d : float, optional
        Mean cosine for downwelling irradiance. Default is 0.9.
    mu_f : float, optional
        Mean cosine for fluorescence emission. Default is 0.5.
    double_gaussian : bool, optional
        If True, use double Gaussian emission. Default is False.

    Returns
    -------
    float or ndarray
        Total Rrs (elastic + fluorescence) in sr^-1.

    Examples
    --------
    >>> wavelength = np.linspace(400, 750, 100)
    >>> a = 0.1 * np.ones_like(wavelength)  # Simplified
    >>> bb = 0.002 * np.ones_like(wavelength)
    >>> Rrs_total = calc_Rrs_with_fluorescence(wavelength, a, bb, Chl=1.0)
    """
    # Elastic Rrs (Gordon model)
    Rrs_elastic = calc_Rrs(a, bb, in_G1, in_G2)

    # Fluorescence Rrs
    Rrs_fl = calc_Rrs_fluorescence_simple(
        wavelength, Chl, phi_C,
        mu_d=mu_d, mu_f=mu_f,
        double_gaussian=double_gaussian
    )

    return Rrs_elastic + Rrs_fl


def calc_fluorescence_spectrum(
    Chl: Union[float, np.ndarray],
    wavelength: Optional[np.ndarray] = None,
    phi_C: float = 0.02,
    double_gaussian: bool = False,
    return_components: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Calculate the full fluorescence Rrs spectrum for given Chl concentrations.

    This is a convenience function that returns the fluorescence spectrum
    across the emission range (typically 650-750 nm).

    Parameters
    ----------
    Chl : float or ndarray
        Chlorophyll-a concentration(s) in mg m^-3.
    wavelength : ndarray, optional
        Emission wavelengths in nm. If None, uses 650-750 nm at 1 nm resolution.
    phi_C : float, optional
        Fluorescence quantum yield. Default is 0.02.
    double_gaussian : bool, optional
        If True, use double Gaussian emission. Default is False.
    return_components : bool, optional
        If True, also return component spectra. Default is False.

    Returns
    -------
    Rrs_fl : ndarray
        Fluorescence Rrs spectrum. Shape is (len(Chl), len(wavelength)) if
        Chl is an array, otherwise (len(wavelength),).
    components : dict, optional
        If return_components=True, dictionary containing:
        - 'wavelength': wavelength array
        - 'emission_shape': normalized emission line shape
        - 'a_water': water absorption at emission wavelengths
        - 'a_ph': phytoplankton absorption at emission wavelengths

    Examples
    --------
    >>> Chl_values = [0.1, 1.0, 10.0]  # mg m^-3
    >>> Rrs_fl = calc_fluorescence_spectrum(Chl_values)
    >>> print(f"Shape: {Rrs_fl.shape}")  # (3, 101)

    >>> wavelength, Rrs_fl, components = calc_fluorescence_spectrum(
    ...     1.0, return_components=True)
    """
    from . import chl_fl

    if wavelength is None:
        wavelength = np.arange(650, 751, 1.0)

    Chl = np.atleast_1d(Chl)

    # Output array
    Rrs_fl = np.zeros((len(Chl), len(wavelength)))

    for i, chl in enumerate(Chl):
        Rrs_fl[i, :] = calc_Rrs_fluorescence_simple(
            wavelength, chl, phi_C, double_gaussian=double_gaussian
        )

    Rrs_fl = np.squeeze(Rrs_fl)

    if return_components:
        # Get emission line shape
        if double_gaussian:
            emission_shape = chl_fl.emission_line_double_gaussian(wavelength)
        else:
            emission_shape = chl_fl.emission_line_single_gaussian(wavelength)

        components = {
            'wavelength': wavelength,
            'emission_shape': emission_shape,
            'a_water': calc_a_water(wavelength),
            'a_ph': calc_a_ph_bricaud(wavelength, Chl[0] if len(Chl) == 1 else 1.0),
        }
        return wavelength, Rrs_fl, components

    return Rrs_fl


def calc_fluorescence_correction_factor(
    wavelength: Union[float, np.ndarray],
    a: Union[float, np.ndarray],
    bb: Union[float, np.ndarray],
    Chl: float,
    phi_C: float = 0.02,
    in_G1: Optional[float] = None,
    in_G2: Optional[float] = None
) -> Union[float, np.ndarray]:
    """
    Calculate the multiplicative correction factor for fluorescence.

    This returns the ratio of total Rrs (elastic + fluorescence) to elastic Rrs,
    useful for understanding the fluorescence contribution.

    Parameters
    ----------
    wavelength : float or ndarray
        Wavelength(s) in nanometers.
    a : float or ndarray
        Total absorption coefficient [m^-1].
    bb : float or ndarray
        Total backscattering coefficient [m^-1].
    Chl : float
        Chlorophyll-a concentration in mg m^-3.
    phi_C : float, optional
        Fluorescence quantum yield. Default is 0.02.
    in_G1, in_G2 : float, optional
        Gordon coefficients. If None, use defaults.

    Returns
    -------
    float or ndarray
        Correction factor: (Rrs_elastic + Rrs_fluorescence) / Rrs_elastic

    Notes
    -----
    Values > 1 indicate wavelengths where fluorescence adds signal.
    The correction is typically largest near 685 nm (the fluorescence peak)
    and approaches 1 at wavelengths away from the emission band.
    """
    Rrs_elastic = calc_Rrs(a, bb, in_G1, in_G2)

    Rrs_fl = calc_Rrs_fluorescence_simple(wavelength, Chl, phi_C)

    return (Rrs_elastic + Rrs_fl) / Rrs_elastic
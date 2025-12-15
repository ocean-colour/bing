"""
Radiation Transfer methods for BING
====================================

This module implements radiative transfer models for computing remote sensing
reflectance (Rrs) from inherent optical properties, including:

1. Standard Gordon (1988) elastic scattering model
2. Raman scattering corrections based on Sathyendranath & Platt (1998)

References
----------
- Gordon, H.R. et al. (1988). "A semianalytic radiance model of ocean color,"
  J. Geophys. Res. 93, 10909-10924.
- Sathyendranath, S. and Platt, T. (1998). "Ocean-color model incorporating
  transspectral processes," Appl. Opt. 37, 2216-2227.
"""

import numpy as np
from typing import Union, Optional, Tuple

# Conversion from rrs to Rrs
A_Rrs, B_Rrs = 0.52, 1.7

# Gordon factors
G1, G2 = 0.0949, 0.0794  # Standard Gordon factors

# Default mean cosines for Raman scattering calculations
# Following Sathyendranath & Platt (1998) Section 4.C
MU_D_DEFAULT = 0.9  # Mean cosine for downwelling irradiance (clear sky, high sun)
MU_U_DEFAULT = 0.5  # Mean cosine for upwelling irradiance (diffuse)
MU_R_DEFAULT = 0.5  # Mean cosine for Raman-scattered light (isotropic)


def calc_Rrs(a, bb, in_G1=None, in_G2=None):
    """
    Calculates the remote sensing reflectance (Rrs) using the given absorption (a) and backscattering (bb) coefficients.

    Parameters:
        a (float or array-like): Absorption coefficient.
        bb (float or array-like): Backscattering coefficient.
        in_G1 (float or array-like, optional): G1 value. Default is None.
        in_G2 (float or array-like, optional): G2 value. Default is None.

    Returns:
        float or array-like: Remote Sensing Reflectance (Rrs) value.
    """
    # u
    u = bb / (a+bb)
    # rrs
    if in_G1 is not None:
        t1 = in_G1 * u
    else: 
        t1 = G1 * u
    if in_G2 is not None:
        t2 = in_G2 * u*u
    else:
        t2 = G2 * u*u
    rrs = t1 + t2
    # Done
    Rrs = A_Rrs*rrs / (1 - B_Rrs*rrs)
    return Rrs


# =============================================================================
# Raman Scattering Correction Functions
# Based on Sathyendranath & Platt (1998), Applied Optics 37, 2216-2227
# =============================================================================

def calc_attenuation_coeffs(
    a: Union[float, np.ndarray],
    bb: Union[float, np.ndarray],
    mu_d: float = MU_D_DEFAULT,
    mu_u: float = MU_U_DEFAULT,
    mu_R: float = MU_R_DEFAULT
) -> dict:
    """
    Calculate diffuse attenuation coefficients for elastic and Raman scattering.

    Following Sathyendranath & Platt (1998) Eqs. (3) and (4), the attenuation
    coefficients are approximated as (a + bb) / mu, where mu is the relevant
    mean cosine for the light stream direction.

    Parameters
    ----------
    a : float or ndarray
        Absorption coefficient [m^-1].
    bb : float or ndarray
        Backscattering coefficient [m^-1].
    mu_d : float
        Mean cosine for downwelling irradiance.
    mu_u : float
        Mean cosine for upwelling irradiance.
    mu_R : float
        Mean cosine for Raman-scattered light (typically 0.5 for isotropic).

    Returns
    -------
    dict
        Dictionary containing attenuation coefficients:
        - 'K': Downwelling attenuation coefficient
        - 'kappa_E': Upwelling attenuation for elastic scatter
        - 'kappa_R': Upwelling attenuation for Raman scatter
        - 'K_R': Downwelling attenuation after Raman scatter
    """
    K = (a + bb) / mu_d         # Eq. (3)
    kappa_E = (a + bb) / mu_u   # Eq. (4) for elastic
    kappa_R = (a + bb) / mu_R   # Eq. (4) adapted for Raman
    K_R = (a + bb) / mu_R       # Downwelling after Raman scatter

    return {
        'K': K,
        'kappa_E': kappa_E,
        'kappa_R': kappa_R,
        'K_R': K_R,
    }


def calc_R_elastic(
    a: Union[float, np.ndarray],
    bb: Union[float, np.ndarray],
    s: float = 1.0,
    mu_d: float = MU_D_DEFAULT,
    mu_u: float = MU_U_DEFAULT
) -> Union[float, np.ndarray]:
    """
    Calculate elastic-scattering reflectance at the sea surface.

    This is Eq. (5) from Sathyendranath & Platt (1998), the standard
    elastic scattering term (Term 0 in Table 1).

    Parameters
    ----------
    a : float or ndarray
        Absorption coefficient [m^-1].
    bb : float or ndarray
        Backscattering coefficient [m^-1].
    s : float
        Shape factor for scattering (s = 1 for isotropic/Rayleigh).
    mu_d : float
        Mean cosine for downwelling irradiance.
    mu_u : float
        Mean cosine for upwelling irradiance.

    Returns
    -------
    float or ndarray
        Elastic reflectance R^E(λ, 0).

    Notes
    -----
    R^E(λ, 0) = [μ_u × s / (μ_u + μ_d)] × [b_b / (a + b_b)]

    This is equivalent to the Gordon model when appropriate G factors are used.
    """
    K = (a + bb) / mu_d
    kappa_E = (a + bb) / mu_u

    # Eq. (2) / Eq. (5) form
    R_E = (s * bb) / (mu_d * (K + kappa_E))

    return R_E


def calc_R_raman_first_order(
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: Union[float, np.ndarray],
    bb_ex: Union[float, np.ndarray],
    bb_R: Union[float, np.ndarray],
    Ed_ratio: Union[float, np.ndarray] = 1.0,
    mu_d: float = MU_D_DEFAULT,
    mu_R: float = MU_R_DEFAULT
) -> Union[float, np.ndarray]:
    """
    Calculate first-order Raman reflectance (Term 1).

    This is Eq. (11) from Sathyendranath & Platt (1998).

    Parameters
    ----------
    a_em : float or ndarray
        Absorption coefficient at emission wavelength λ [m^-1].
    bb_em : float or ndarray
        Elastic backscattering coefficient at emission wavelength λ [m^-1].
    a_ex : float or ndarray
        Absorption coefficient at excitation wavelength λ' [m^-1].
    bb_ex : float or ndarray
        Elastic backscattering coefficient at excitation wavelength λ' [m^-1].
    bb_R : float or ndarray
        Raman backscattering coefficient at excitation wavelength λ' [m^-1].
        Note: For Raman scattering, s^R = 1 (symmetric phase function).
    Ed_ratio : float or ndarray
        Ratio of downwelling irradiance Ed(λ')/Ed(λ). Default is 1.0.
    mu_d : float
        Mean cosine for downwelling irradiance.
    mu_R : float
        Mean cosine for Raman-scattered light.

    Returns
    -------
    float or ndarray
        First-order Raman reflectance R^R(λ, 0).

    Notes
    -----
    R^R(λ, 0) = [Ed(λ')/Ed(λ)] × [b_b^R(λ')/μ_d(λ')] × 1/[K(λ') + κ^R(λ)]

    where K(λ') = (a(λ') + b_b(λ'))/μ_d
    and   κ^R(λ) = (a(λ) + b_b(λ))/μ_R
    """
    # Attenuation coefficients
    K_ex = (a_ex + bb_ex) / mu_d      # K(λ')
    kappa_R_em = (a_em + bb_em) / mu_R  # κ^R(λ)

    # Eq. (11)
    R_R = Ed_ratio * (bb_R / mu_d) / (K_ex + kappa_R_em)

    return R_R


def calc_R_raman_RE(
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: Union[float, np.ndarray],
    bb_ex: Union[float, np.ndarray],
    bb_R: Union[float, np.ndarray],
    Ed_ratio: Union[float, np.ndarray] = 1.0,
    s_E: float = 1.0,
    mu_d: float = MU_D_DEFAULT,
    mu_R: float = MU_R_DEFAULT
) -> Union[float, np.ndarray]:
    """
    Calculate second-order Raman-then-Elastic reflectance (Term 2).

    This is Eq. (18) from Sathyendranath & Platt (1998): a downward
    Raman-scattering event followed by an upward elastic-scattering event.

    Parameters
    ----------
    a_em : float or ndarray
        Absorption coefficient at emission wavelength λ [m^-1].
    bb_em : float or ndarray
        Elastic backscattering coefficient at emission wavelength λ [m^-1].
    a_ex : float or ndarray
        Absorption coefficient at excitation wavelength λ' [m^-1].
    bb_ex : float or ndarray
        Elastic backscattering coefficient at excitation wavelength λ' [m^-1].
    bb_R : float or ndarray
        Raman backscattering coefficient at λ' [m^-1].
    Ed_ratio : float or ndarray
        Ratio Ed(λ')/Ed(λ). Default is 1.0.
    s_E : float
        Shape factor for elastic scattering.
    mu_d : float
        Mean cosine for downwelling irradiance.
    mu_R : float
        Mean cosine for Raman-scattered light.

    Returns
    -------
    float or ndarray
        Second-order Raman-Elastic reflectance R^RE(λ, 0).

    Notes
    -----
    R^RE(λ, 0) = [Ed(λ')/Ed(λ)] × [s^E b_b^E(λ)/μ_d^R(λ)] × [b_b^R(λ')/μ_d(λ')]
                 × 1 / {[K(λ') + κ^RE(λ)] × [K^R(λ) + κ^RE(λ)]}

    This term is typically ~10% of the first-order Raman term.
    """
    # Attenuation coefficients at excitation wavelength
    K_ex = (a_ex + bb_ex) / mu_d       # K(λ')

    # Attenuation coefficients at emission wavelength
    K_R_em = (a_em + bb_em) / mu_R     # K^R(λ)
    kappa_RE_em = (a_em + bb_em) / mu_R  # κ^RE(λ) ≈ κ^R for clear water

    # Eq. (18)
    numerator = Ed_ratio * (s_E * bb_em / mu_R) * (bb_R / mu_d)
    denominator = (K_ex + kappa_RE_em) * (K_R_em + kappa_RE_em)

    R_RE = numerator / denominator

    return R_RE


def calc_R_raman_ER(
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: Union[float, np.ndarray],
    bb_ex: Union[float, np.ndarray],
    bb_R: Union[float, np.ndarray],
    Ed_ratio: Union[float, np.ndarray] = 1.0,
    s_E: float = 1.0,
    mu_d: float = MU_D_DEFAULT,
    mu_u: float = MU_U_DEFAULT,
    mu_R: float = MU_R_DEFAULT
) -> Union[float, np.ndarray]:
    """
    Calculate second-order Elastic-then-Raman reflectance (Term 3).

    This is Eq. (23) from Sathyendranath & Platt (1998): an upward
    elastic-scattering event followed by an upward Raman-scattering event.

    Parameters
    ----------
    a_em : float or ndarray
        Absorption coefficient at emission wavelength λ [m^-1].
    bb_em : float or ndarray
        Elastic backscattering coefficient at emission wavelength λ [m^-1].
    a_ex : float or ndarray
        Absorption coefficient at excitation wavelength λ' [m^-1].
    bb_ex : float or ndarray
        Elastic backscattering coefficient at excitation wavelength λ' [m^-1].
    bb_R : float or ndarray
        Raman backscattering coefficient at λ' [m^-1].
    Ed_ratio : float or ndarray
        Ratio Ed(λ')/Ed(λ). Default is 1.0.
    s_E : float
        Shape factor for elastic scattering.
    mu_d : float
        Mean cosine for downwelling irradiance.
    mu_u : float
        Mean cosine for upwelling irradiance.
    mu_R : float
        Mean cosine for Raman-scattered light.

    Returns
    -------
    float or ndarray
        Second-order Elastic-Raman reflectance R^ER(λ, 0).

    Notes
    -----
    R^ER(λ, 0) = [Ed(λ')/Ed(λ)] × [s^E b_b^E(λ')/μ_d(λ')] × [b_b^R(λ')/μ_u^E(λ')]
                 × 1 / {[K(λ') + κ^E(λ')] × [K(λ') + κ^ER(λ)]}

    This term is typically ~10% of the first-order Raman term.
    """
    # Attenuation coefficients at excitation wavelength
    K_ex = (a_ex + bb_ex) / mu_d       # K(λ')
    kappa_E_ex = (a_ex + bb_ex) / mu_u  # κ^E(λ')

    # Attenuation coefficient at emission wavelength
    kappa_ER_em = (a_em + bb_em) / mu_R  # κ^ER(λ)

    # Eq. (23)
    numerator = Ed_ratio * (s_E * bb_ex / mu_d) * (bb_R / mu_u)
    denominator = (K_ex + kappa_E_ex) * (K_ex + kappa_ER_em)

    R_ER = numerator / denominator

    return R_ER


def calc_R_raman_total(
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: Union[float, np.ndarray],
    bb_ex: Union[float, np.ndarray],
    bb_R: Union[float, np.ndarray],
    Ed_ratio: Union[float, np.ndarray] = 1.0,
    s_E: float = 1.0,
    mu_d: float = MU_D_DEFAULT,
    mu_u: float = MU_U_DEFAULT,
    mu_R: float = MU_R_DEFAULT,
    include_second_order: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate total Raman contribution to reflectance at the sea surface.

    This combines the first-order Raman term (Term 1) and optionally the
    two second-order terms (Terms 2 and 3) from Sathyendranath & Platt (1998).

    Parameters
    ----------
    a_em : float or ndarray
        Absorption coefficient at emission wavelength λ [m^-1].
    bb_em : float or ndarray
        Elastic backscattering coefficient at emission wavelength λ [m^-1].
    a_ex : float or ndarray
        Absorption coefficient at excitation wavelength λ' [m^-1].
    bb_ex : float or ndarray
        Elastic backscattering coefficient at excitation wavelength λ' [m^-1].
    bb_R : float or ndarray
        Raman backscattering coefficient at λ' [m^-1].
    Ed_ratio : float or ndarray
        Ratio Ed(λ')/Ed(λ). Default is 1.0.
    s_E : float
        Shape factor for elastic scattering.
    mu_d : float
        Mean cosine for downwelling irradiance.
    mu_u : float
        Mean cosine for upwelling irradiance.
    mu_R : float
        Mean cosine for Raman-scattered light.
    include_second_order : bool
        If True, include second-order terms (RE and ER). Default is True.

    Returns
    -------
    float or ndarray
        Total Raman reflectance contribution.

    Notes
    -----
    Total Raman reflectance = R^R + R^RE + R^ER

    The second-order Raman-Raman terms (Terms 4 and 5) are neglected as they
    contribute only ~1% of the first-order term (see Section 4.A of the paper).
    """
    # First-order Raman term (always included)
    R_R = calc_R_raman_first_order(
        a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio, mu_d, mu_R
    )

    if include_second_order:
        # Second-order Raman-Elastic term
        R_RE = calc_R_raman_RE(
            a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio, s_E, mu_d, mu_R
        )

        # Second-order Elastic-Raman term
        R_ER = calc_R_raman_ER(
            a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio, s_E, mu_d, mu_u, mu_R
        )

        return R_R + R_RE + R_ER

    return R_R


def calc_R_total_with_raman(
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: Union[float, np.ndarray],
    bb_ex: Union[float, np.ndarray],
    bb_R: Union[float, np.ndarray],
    Ed_ratio: Union[float, np.ndarray] = 1.0,
    s_E: float = 1.0,
    mu_d: float = MU_D_DEFAULT,
    mu_u: float = MU_U_DEFAULT,
    mu_R: float = MU_R_DEFAULT,
    include_second_order: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate total reflectance including elastic and Raman contributions.

    This implements Eq. (26) from Sathyendranath & Platt (1998), the simplified
    model for clear waters.

    Parameters
    ----------
    a_em : float or ndarray
        Absorption coefficient at emission wavelength λ [m^-1].
    bb_em : float or ndarray
        Elastic backscattering coefficient at emission wavelength λ [m^-1].
    a_ex : float or ndarray
        Absorption coefficient at excitation wavelength λ' [m^-1].
    bb_ex : float or ndarray
        Elastic backscattering coefficient at excitation wavelength λ' [m^-1].
    bb_R : float or ndarray
        Raman backscattering coefficient at λ' [m^-1].
    Ed_ratio : float or ndarray
        Ratio Ed(λ')/Ed(λ). Default is 1.0.
    s_E : float
        Shape factor for elastic scattering.
    mu_d : float
        Mean cosine for downwelling irradiance.
    mu_u : float
        Mean cosine for upwelling irradiance.
    mu_R : float
        Mean cosine for Raman-scattered light.
    include_second_order : bool
        If True, include second-order Raman terms. Default is True.

    Returns
    -------
    float or ndarray
        Total reflectance R(λ, 0) = R^E(λ) + R^R(λ) + R^RE(λ) + R^ER(λ).

    Notes
    -----
    For the simplified model (Eq. 26), assuming clear waters where molecular
    scattering dominates upward scatter:
    - μ_u = μ_R = 0.5
    - s = 1
    - κ^E = κ^R = K^R = κ^RE = κ^ER

    The total reflectance becomes:
    R(λ) = R^E(λ) + R^R(λ) × {1 + b_b^E(λ)/κ^E(λ) + b_b^E(λ')/[0.5(K(λ') + κ^E(λ'))]}
    """
    # Elastic term
    R_E = calc_R_elastic(a_em, bb_em, s_E, mu_d, mu_u)

    # Raman terms
    R_raman = calc_R_raman_total(
        a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio,
        s_E, mu_d, mu_u, mu_R, include_second_order
    )

    return R_E + R_raman


def calc_raman_correction_factor(
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: Union[float, np.ndarray],
    bb_ex: Union[float, np.ndarray],
    bb_R: Union[float, np.ndarray],
    Ed_ratio: Union[float, np.ndarray] = 1.0,
    s_E: float = 1.0,
    mu_d: float = MU_D_DEFAULT,
    mu_u: float = MU_U_DEFAULT,
    mu_R: float = MU_R_DEFAULT,
    include_second_order: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate the multiplicative correction factor for Raman scattering.

    This returns the ratio of total reflectance (with Raman) to elastic
    reflectance, useful for correcting Rrs retrievals.

    Parameters
    ----------
    a_em : float or ndarray
        Absorption coefficient at emission wavelength λ [m^-1].
    bb_em : float or ndarray
        Elastic backscattering coefficient at emission wavelength λ [m^-1].
    a_ex : float or ndarray
        Absorption coefficient at excitation wavelength λ' [m^-1].
    bb_ex : float or ndarray
        Elastic backscattering coefficient at excitation wavelength λ' [m^-1].
    bb_R : float or ndarray
        Raman backscattering coefficient at λ' [m^-1].
    Ed_ratio : float or ndarray
        Ratio Ed(λ')/Ed(λ). Default is 1.0.
    s_E : float
        Shape factor for elastic scattering.
    mu_d : float
        Mean cosine for downwelling irradiance.
    mu_u : float
        Mean cosine for upwelling irradiance.
    mu_R : float
        Mean cosine for Raman-scattered light.
    include_second_order : bool
        If True, include second-order Raman terms. Default is True.

    Returns
    -------
    float or ndarray
        Correction factor: (R^E + R_raman) / R^E

    Notes
    -----
    To remove Raman contribution from measured reflectance:
        R_elastic = R_measured / correction_factor

    Typical correction factors range from 1.0 to ~1.25, with largest
    corrections in clear (oligotrophic) waters at longer wavelengths.
    """
    R_E = calc_R_elastic(a_em, bb_em, s_E, mu_d, mu_u)

    R_raman = calc_R_raman_total(
        a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio,
        s_E, mu_d, mu_u, mu_R, include_second_order
    )

    return (R_E + R_raman) / R_E


def calc_Rrs_with_raman(
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: Union[float, np.ndarray],
    bb_ex: Union[float, np.ndarray],
    bb_R: Union[float, np.ndarray],
    Ed_ratio: Union[float, np.ndarray] = 1.0,
    in_G1: Optional[float] = None,
    in_G2: Optional[float] = None,
    mu_d: float = MU_D_DEFAULT,
    mu_u: float = MU_U_DEFAULT,
    mu_R: float = MU_R_DEFAULT,
    include_second_order: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate remote sensing reflectance (Rrs) including Raman correction.

    This function computes Rrs using the Gordon model for elastic scattering
    and adds the Raman contribution from Sathyendranath & Platt (1998).

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
    bb_R : float or ndarray
        Raman backscattering coefficient at λ' [m^-1].
        Can be computed using bing.rt.raman.raman_backscattering_coeff().
    Ed_ratio : float or ndarray
        Ratio Ed(λ')/Ed(λ). Default is 1.0.
    in_G1, in_G2 : float, optional
        Gordon coefficients. If None, use defaults (0.0949, 0.0794).
    mu_d : float
        Mean cosine for downwelling irradiance.
    mu_u : float
        Mean cosine for upwelling irradiance.
    mu_R : float
        Mean cosine for Raman-scattered light.
    include_second_order : bool
        If True, include second-order Raman terms. Default is True.

    Returns
    -------
    float or ndarray
        Remote sensing reflectance Rrs [sr^-1].

    Examples
    --------
    >>> from bing.rt import raman
    >>> # At 520 nm emission with 443 nm excitation
    >>> a_520, bb_520 = 0.05, 0.002
    >>> a_443, bb_443 = 0.03, 0.003
    >>> bb_R = raman.raman_backscattering_coeff(443)
    >>> Rrs = calc_Rrs_with_raman(a_520, bb_520, a_443, bb_443, bb_R)
    """
    # Elastic Rrs (Gordon model)
    Rrs_elastic = calc_Rrs(a_em, bb_em, in_G1, in_G2)

    # Raman reflectance contribution (subsurface)
    R_raman = calc_R_raman_total(
        a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio,
        s_E=1.0, mu_d=mu_d, mu_u=mu_u, mu_R=mu_R,
        include_second_order=include_second_order
    )

    # Convert Raman reflectance to Rrs
    # R_raman is subsurface reflectance; convert similar to elastic
    # Using simplified conversion: Rrs_raman ≈ R_raman / Q
    # where Q ≈ π for Lambertian, but we use the same conversion as elastic
    Rrs_raman = A_Rrs * R_raman / (1 - B_Rrs * R_raman)

    return Rrs_elastic + Rrs_raman
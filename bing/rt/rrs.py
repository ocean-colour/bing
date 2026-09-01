"""
Radiation Transfer methods for BING
====================================

This module implements radiative transfer models for computing remote sensing
reflectance (Rrs) from inherent optical properties, including:

1. Standard Gordon (1988) elastic scattering model
2. Raman scattering corrections based on Sathyendranath & Platt (1998)
3. Chlorophyll fluorescence contribution based on Gordon (1979)

References
----------
- Gordon, H.R. et al. (1988). "A semianalytic radiance model of ocean color,"
  J. Geophys. Res. 93, 10909-10924.
- Sathyendranath, S. and Platt, T. (1998). "Ocean-color model incorporating
  transspectral processes," Appl. Opt. 37, 2216-2227.
- Gordon, H.R. (1979). "Diffuse reflectance of the ocean: the theory of its
  augmentation by chlorophyll a fluorescence at 685 nm," Appl. Opt. 18, 1161-1166.
- Bricaud, A. et al. (1995). "Variability in the chlorophyll-specific absorption
  coefficients of natural phytoplankton," J. Geophys. Res. 100, 13321-13332.
"""

import os
from importlib import resources
import numpy as np
import pandas
from typing import Union, Optional, Tuple
from scipy import interpolate 

from bing.rt import raman
from bing.rt import chl_fl, raman

# Conversion from rrs to Rrs
A_Rrs, B_Rrs = 0.52, 1.7

# Gordon factors
G1_STANDARD, G2_STANDARD = 0.0949, 0.0794  # Standard Gordon factors

# Max elements in the fluorescence (n_samples, n_em, n_ex) integrand tensor
# processed in one vectorised block by calc_Rrs_fluorescence's chains path.
# Caps peak RAM (~this*8 bytes * a few copies) while keeping the per-call
# vectorisation that the MCMC log_prob hot path depends on.  5e7 -> ~0.4 GiB
# per tensor, a few GiB peak even for a 528k-sample reconstruction.
FL_CHUNK_ELEMENTS = 50_000_000

from IPython import embed


def wave_dependent_gordon(wave:np.ndarray, bounds_error:bool=True,
                          include_G0:bool=False):
    """
    Load and interpolate wavelength-dependent Gordon coefficients.

    Two CSVs are supported under ``bing/data/RT/``:

    - ``gordon_coefficients.csv`` (2-parameter): columns G1, G2.
      ``rrs(λ) = G1(λ)·u + G2(λ)·u²``.
    - ``gordon_coefficients_with_G0.csv`` (3-parameter): columns G0, G1, G2.
      ``rrs(λ) = G0(λ) + G1(λ)·u + G2(λ)·u²``.

    Parameters
    ----------
    wave : np.ndarray
        Wavelengths in nanometers at which to interpolate.
    bounds_error : bool, optional
        If True (default), raises an error if wavelengths are outside the
        tabulated range. If False, extrapolates using cubic spline.
    include_G0 : bool, optional
        If True, also load the constant-offset coefficient G0(λ) from the
        3-parameter CSV. Default False (reads the 2-parameter CSV and returns
        ``G0 = None``).

    Returns
    -------
    G1, G2, G0 : np.ndarray
        Always returns a 3-tuple. ``G0`` is ``None`` when ``include_G0`` is
        False and an array of the same shape as ``G1`` otherwise. (This
        signature is post-merge; callers that previously did
        ``G1, G2 = wave_dependent_gordon(wave)`` must unpack three values.)

    See Also
    --------
    calc_elastic_Rrs : Uses these coefficients to compute Rrs from IOPs.
    """
    fname = 'gordon_coefficients_with_G0.csv' if include_G0 else 'gordon_coefficients.csv'
    gordon_file = os.path.join(
            resources.files('bing'),
            'data', 'RT', fname)
    result = pandas.read_csv(gordon_file, comment='#')

    f_G1 = interpolate.interp1d(result['wavelength'], result['G1'], kind=3,
                                bounds_error=bounds_error)
    f_G2 = interpolate.interp1d(result['wavelength'], result['G2'], kind=3,
                                bounds_error=bounds_error)
    G1 = f_G1(wave)
    G2 = f_G2(wave)

    if include_G0:
        if 'G0' not in result.columns:
            raise IOError(f"G0 column missing from {gordon_file}")
        f_G0 = interpolate.interp1d(result['wavelength'], result['G0'], kind=3,
                                    bounds_error=bounds_error)
        return G1, G2, f_G0(wave)
    return G1, G2, None


def wave_dependent_gordon_bbp(wave: np.ndarray, bounds_error: bool = True):
    """
    Load and interpolate the 3-parameter (G1, G2, Gb) coefficients fit to
    ``rrs(λ) = G1(λ)·u + G2(λ)·u² + Gb(λ)·bbp``.

    Read from ``bing/data/RT/gordon_coefficients_with_Gb.csv``.

    Returns
    -------
    G1, G2, Gb : np.ndarray
        Wavelength-dependent coefficients. Always a 3-tuple of arrays.
    """
    gordon_file = os.path.join(
        resources.files('bing'),
        'data', 'RT', 'gordon_coefficients_with_Gb.csv')
    result = pandas.read_csv(gordon_file, comment='#')

    for col in ('G1', 'G2', 'Gb'):
        if col not in result.columns:
            raise IOError(f"{col} column missing from {gordon_file}")

    f_G1 = interpolate.interp1d(result['wavelength'], result['G1'], kind=3,
                                bounds_error=bounds_error)
    f_G2 = interpolate.interp1d(result['wavelength'], result['G2'], kind=3,
                                bounds_error=bounds_error)
    f_Gb = interpolate.interp1d(result['wavelength'], result['Gb'], kind=3,
                                bounds_error=bounds_error)
    return f_G1(wave), f_G2(wave), f_Gb(wave)


def wave_dependent_gordon_full(wave: np.ndarray, bounds_error: bool = True):
    """
    Load and interpolate the 4-parameter (G1, G2, G0, Gb) coefficients fit to
    ``rrs(λ) = G0(λ) + G1(λ)·u + G2(λ)·u² + Gb(λ)·bbp``.

    Read from ``bing/data/RT/gordon_coefficients_with_G0_Gb.csv``. Empirically
    this 4-parameter form matches or beats both 3-parameter recipes at every
    wavelength on the L23 elastic dataset.

    Returns
    -------
    G1, G2, G0, Gb : np.ndarray
        Wavelength-dependent coefficients. Always a 4-tuple of arrays.
        (Order parallels ``wave_dependent_gordon(..., include_G0=True)`` which
        returns ``(G1, G2, G0)``; Gb is appended.)
    """
    gordon_file = os.path.join(
        resources.files('bing'),
        'data', 'RT', 'gordon_coefficients_with_G0_Gb.csv')
    result = pandas.read_csv(gordon_file, comment='#')

    for col in ('G0', 'G1', 'G2', 'Gb'):
        if col not in result.columns:
            raise IOError(f"{col} column missing from {gordon_file}")

    f_G1 = interpolate.interp1d(result['wavelength'], result['G1'], kind=3,
                                bounds_error=bounds_error)
    f_G2 = interpolate.interp1d(result['wavelength'], result['G2'], kind=3,
                                bounds_error=bounds_error)
    f_G0 = interpolate.interp1d(result['wavelength'], result['G0'], kind=3,
                                bounds_error=bounds_error)
    f_Gb = interpolate.interp1d(result['wavelength'], result['Gb'], kind=3,
                                bounds_error=bounds_error)
    return f_G1(wave), f_G2(wave), f_G0(wave), f_Gb(wave)


def Rrs_to_rrs(Rrs: np.ndarray, A: float = A_Rrs, B: float = B_Rrs) -> np.ndarray:
    """
    Convert above-surface Rrs to subsurface rrs.

    Uses the relation: rrs = Rrs / (A + B * Rrs)

    Parameters
    ----------
    Rrs : np.ndarray
        Remote sensing reflectance (above surface).
    A : float
        Conversion coefficient (default 0.52).
    B : float
        Conversion coefficient (default 1.17).

    Returns
    -------
    np.ndarray
        Subsurface remote sensing reflectance.
    """
    return Rrs / (A + B * Rrs)


def rrs_to_Rrs(rrs: np.ndarray, A: float = A_Rrs, B: float = B_Rrs) -> np.ndarray:
    """
    Convert subsurface rrs to above-surface Rrs.

    Uses the relation: Rrs = A * rrs / (1 - B * rrs)

    Parameters
    ----------
    rrs : np.ndarray
        Subsurface remote sensing reflectance.
    A : float
        Conversion coefficient (default 0.52).
    B : float
        Conversion coefficient (default 1.17).

    Returns
    -------
    np.ndarray
        Remote sensing reflectance (above surface).
    """
    return A * rrs / (1 - B * rrs)

def calc_Rrs(a, bb, in_G1:float|np.ndarray=None, in_G2:float|np.ndarray=None,
    a_ex: Union[float, np.ndarray]=None,
    bb_ex: Union[float, np.ndarray]=None,
    bb_R: Union[float, np.ndarray]=None,
    in_G0: Union[float, np.ndarray, None]=None,
    in_Gb: Union[float, np.ndarray, None]=None,
    in_bbp: Union[float, np.ndarray, None]=None,
    Ed_ratio: Union[float, np.ndarray, None]=None,
    ):
    """
    Calculate remote sensing reflectance (Rrs) including optional Raman correction.

    This is the main Rrs calculation function that combines elastic scattering
    (Gordon model) with an optional Raman scattering correction based on
    Sathyendranath & Platt (1998).

    Chl fluorescence is not included in this function.

    Parameters
    ----------
    a : float or np.ndarray
        Total absorption coefficient at emission wavelength(s) [m^-1].
    bb : float or np.ndarray
        Total backscattering coefficient at emission wavelength(s) [m^-1].
    in_G1 : float or np.ndarray, optional
        First-order Gordon coefficient. If None, uses default (0.0949).
    in_G2 : float or np.ndarray, optional
        Second-order Gordon coefficient. If None, uses default (0.0794).
    a_ex : float or np.ndarray, optional
        Absorption coefficient at Raman excitation wavelength(s) [m^-1].
        Required for Raman correction.
    bb_ex : float or np.ndarray, optional
        Backscattering coefficient at Raman excitation wavelength(s) [m^-1].
        Required for Raman correction.
    bb_R : float or np.ndarray, optional
        Raman backscattering coefficient [m^-1].
        Required for Raman correction. Can be computed using
        bing.rt.raman.raman_backscattering_coeff().
    Ed_ratio : float or np.ndarray, optional
        Downwelling-irradiance ratio Ed(λ')/Ed(λ) between the Raman
        excitation and emission wavelengths. If None, a flat solar
        spectrum (ratio = 1) is assumed — this is known to distort the
        spectral shape of the Raman correction (too strong in the blue,
        too weak in the red); supply the true ratio whenever an Ed
        spectrum is available (see aNWModel.set_raman_Ed).

    Returns
    -------
    np.ndarray
        Remote sensing reflectance Rrs [sr^-1].

    Raises
    ------
    IOError
        If a_ex is provided but bb_ex or bb_R are not.

    Notes
    -----
    When Raman parameters (a_ex, bb_ex, bb_R) are provided, the elastic Rrs
    is multiplied by a correction factor that accounts for the Raman scattering
    contribution. This correction is typically 1.0-1.25, with largest values
    in clear oligotrophic waters at longer wavelengths.

    Examples
    --------
    >>> # Elastic-only calculation
    >>> Rrs = calc_Rrs(a=0.05, bb=0.002)
    >>>
    >>> # With Raman correction
    >>> from bing.rt import raman
    >>> bb_R = raman.raman_backscattering_coeff(wave_ex)
    >>> Rrs = calc_Rrs(a, bb, a_ex=a_ex, bb_ex=bb_ex, bb_R=bb_R)

    See Also
    --------
    calc_elastic_Rrs : Calculate elastic-only Rrs.
    calc_raman_correction_factor : Compute the Raman correction factor.
    """
    # Elastic
    Rrs = calc_elastic_Rrs(a, bb, in_G1=in_G1, in_G2=in_G2, in_G0=in_G0,
                           in_Gb=in_Gb, in_bbp=in_bbp)

    # Raman?
    if a_ex is not None:
        if bb_ex is None or bb_R is None:
            raise IOError("bb_ex,bb_R must be set if a_ex is provided")
        corr = calc_raman_correction_factor(
            a, bb, a_ex, bb_ex, bb_R,
            Ed_ratio=1.0 if Ed_ratio is None else Ed_ratio)
        # Apply
        Rrs *= corr

    # Return
    return Rrs



def calc_elastic_Rrs(a, bb, in_G1:float|np.ndarray=None, in_G2:float|np.ndarray=None,
                     in_G0:Union[float, np.ndarray, None]=None,
                     in_Gb:Union[float, np.ndarray, None]=None,
                     in_bbp:Union[float, np.ndarray, None]=None):
    """
    Calculates the remote sensing reflectance (Rrs) using
    the given absorption (a) and backscattering (bb) coefficients.

    Evaluates ``rrs = G0 + G1·u + G2·u² + Gb·bbp`` where u = bb/(a+bb),
    then converts to above-surface Rrs. G0 and Gb are independent optional
    third coefficients; in practice only one is used at a time.

    Parameters:
        a (float or array-like): Absorption coefficient.
        bb (float or array-like): Backscattering coefficient.
        in_G1 (float or array-like, optional): G1 value. Default uses G1_STANDARD.
        in_G2 (float or array-like, optional): G2 value. Default uses G2_STANDARD.
        in_G0 (float or array-like, optional): Constant offset. Default None.
        in_Gb (float or array-like, optional): Slope on bbp. Default None.
            Requires ``in_bbp`` when supplied.
        in_bbp (float or array-like, optional): Particulate backscatter
            (= bbnw). Required when ``in_Gb`` is provided.

    Returns:
        float or array-like: Remote Sensing Reflectance (Rrs) value.
    """
    u = bb / (a+bb)
    t1 = in_G1 * u   if in_G1 is not None else G1_STANDARD * u
    t2 = in_G2 * u*u if in_G2 is not None else G2_STANDARD * u*u
    rrs = t1 + t2
    if in_G0 is not None:
        rrs = rrs + in_G0
    if in_Gb is not None:
        if in_bbp is None:
            raise IOError("in_bbp must be supplied when in_Gb is provided")
        rrs = rrs + in_Gb * in_bbp
    return rrs_to_Rrs(rrs)


# =============================================================================
# Raman Scattering Correction Functions
# Based on Sathyendranath & Platt (1998), Applied Optics 37, 2216-2227
# =============================================================================
# NOTE: Most Raman scattering functions have been moved to bing.rt.raman module.
# This section now contains only the correction factor function that bridges
# between the Gordon model (in rrs.py) and Raman scattering (in raman.py).
# For detailed Raman scattering calculations, see bing.rt.raman.


def calc_raman_correction_factor(
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: Union[float, np.ndarray],
    bb_ex: Union[float, np.ndarray],
    bb_R: Union[float, np.ndarray],
    Ed_ratio: Union[float, np.ndarray] = 1.0,
    s_E: float = 1.0,
    mu_d: Optional[float] = None,
    mu_u: Optional[float] = None,
    mu_R: Optional[float] = None,
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
        Can be computed using bing.rt.raman.raman_backscattering_coeff().
    Ed_ratio : float or ndarray
        Ratio Ed(λ')/Ed(λ). Default is 1.0.
    s_E : float
        Shape factor for elastic scattering.
    mu_d : float, optional
        Mean cosine for downwelling irradiance. If None, uses default from raman module.
    mu_u : float, optional
        Mean cosine for upwelling irradiance. If None, uses default from raman module.
    mu_R : float, optional
        Mean cosine for Raman-scattered light. If None, uses default from raman module.
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

    This function now calls bing.rt.raman for the detailed Raman calculations.

    Examples
    --------
    >>> from bing.rt import raman
    >>> a_520, bb_520 = 0.05, 0.002
    >>> a_443, bb_443 = 0.03, 0.003
    >>> bb_R = raman.raman_backscattering_coeff(443)
    >>> correction = calc_raman_correction_factor(a_520, bb_520, a_443, bb_443, bb_R)
    """

    # Use default mean cosines if not provided
    if mu_d is None:
        mu_d = raman.MU_D_DEFAULT
    if mu_u is None:
        mu_u = raman.MU_U_DEFAULT
    if mu_R is None:
        mu_R = raman.MU_R_DEFAULT

    # Calculate elastic reflectance
    R_E = raman.calc_R_elastic(a_em, bb_em, s_E, mu_d, mu_u)

    # Calculate total Raman reflectance
    R_raman = raman.calc_R_raman_total(
        a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio,
        s_E, mu_d, mu_u, mu_R, include_second_order
    )

    return (R_E + R_raman) / R_E

# =============================================================================
# Chl Fluorescence Functions
# =============================================================================

# Default mean cosine for fluorescence (isotropic emission)
MU_F_DEFAULT = 0.5

def calc_Rrs_fluorescence(
    wavelength: Union[float, np.ndarray],
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: np.ndarray,
    bb_ex: np.ndarray,
    aph_ex: np.ndarray,
    wavelength_ex: np.ndarray,
    Ed_ex: np.ndarray,
    Ed_em: Union[float, np.ndarray],
    mu_d: Optional[float] = None,
    mu_f: Optional[float] = None,
    phi_C: float = 0.02,
    double_gaussian: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate Rrs contribution from chlorophyll fluorescence as a function of wavelength.

    This function computes the fluorescence contribution to remote sensing reflectance
    integrating over excitation wavelengths and using the Bricaud parameterization
    for phytoplankton absorption.

    Parameters
    ----------
    wavelength : float or ndarray
        Emission wavelength(s) λ in nanometers. Typically in range 650-750 nm.
    a_em: float or np.ndarray
        Total absorption coefficient at emission wavelength(s) [m^-1].
    bb_em: float or np.ndarray
        Total backscattering coefficient at emission wavelength(s) [m^-1].
    wavelength_ex : float or ndarray, optional
        Excitation wavelength(s) for integration. If None, uses 400-680 nm range.
    a_ex: float or np.ndarray
        Total absorption coefficient at excitation wavelength(s) [m^-1].
    bb_ex: float or np.ndarray
        Total backscattering coefficient at excitation wavelength(s) [m^-1].
    aph_ex: float or np.ndarray
        Phytoplankton absorption coefficient at excitation wavelength(s) [m^-1].
    mu_d : float, optional
        Mean cosine for downwelling irradiance. Default is 0.9.
    mu_f : float, optional
        Mean cosine for fluorescence emission (isotropic). Default is 0.5.
    phi_C : float, optional
        Fluorescence quantum yield (0-1). Default is 0.02.
    double_gaussian : bool, optional
        If True, use double Gaussian emission (685 + 730 nm peaks). Default is False.

    Returns
    -------
    float or ndarray
        Fluorescence contribution to Rrs in sr^-1.

    Notes
    -----
    The calculation integrates over excitation wavelengths:

    R_F(λ) = ∫ Ed(λ') × (λ'/λ) × [b_bF(λ')/μ_d] / [K(λ') + κ_F(λ)] dλ' / Ed(λ)

    where:
    - Ed(λ') is the downwelling irradiance spectrum
    - b_bF = 0.5 × Φ_C × a_ph is the fluorescence backscattering coefficient
    - K and κ_F are attenuation coefficients

    R_F is a two-flow *irradiance* reflectance (Eu/Ed).  Because the
    emission is isotropic, the upwelling radiance is uniform and
    Lu(0-) = Eu(0-)/π, so the subsurface remote-sensing reflectance is
    rrs_F = R_F/π.  The final Rrs applies the emission line shape h_C(λ)
    (Gaussian at 685 nm, optionally double Gaussian) and the standard
    rrs → Rrs conversion:

    Rrs_fl(λ) = h_C(λ) × A × (R_F/π) / (1 − B × R_F/π)

    The 1/π conversion was validated against the Loisel et al. (2023)
    HydroLight scenario differences (X4−X2); without it the term is ~3×
    too large (see retrieve-or-bust context/RT/rt_inelastic_bing_summary.md).

    ``Ed_em`` may be a scalar (legacy: Ed at the 685 nm peak) or an array
    over the emission wavelengths (preferred; exact per-λ_em
    normalization).

    Examples
    --------
    >>> wavelengths = np.linspace(650, 750, 100)
    >>> Rrs_fl = calc_Rrs_fluorescence(wavelengths, Chl=1.0)
    >>> print(f"Peak fluorescence Rrs: {Rrs_fl.max():.2e} sr^-1")
    """
    wavelength = np.atleast_1d(wavelength)

    # wavelength_ex and Ed_ex are scene properties (same across MCMC samples).
    # reconstruct_from_chains tiles them to 2D via np.outer for legacy reasons;
    # log_prob passes them as 1D.  Collapse to 1D so the broadcasting below is
    # uniform regardless of which caller invoked us.
    wavelength_ex = np.asarray(wavelength_ex)
    if wavelength_ex.ndim > 1:
        wavelength_ex = wavelength_ex[0]
    Ed_ex = np.asarray(Ed_ex)
    if Ed_ex.ndim > 1:
        Ed_ex = Ed_ex[0]

    # The caller uses 2D excitation IOPs for chains (and also for a single
    # MCMC step, where a_ex has shape (1, n_ex) from eval_a).  Some callers
    # pass aph_ex / a_em / bb_em as 1D in that same path; broadcast them so
    # all of (a_ex, bb_ex, aph_ex, a_em, bb_em) live at a common rank.
    ndim = a_ex.ndim

    # Use default mean cosines if not provided
    if mu_d is None:
        mu_d = raman.MU_D_DEFAULT
    if mu_f is None:
        mu_f = 0.5  # Default for fluorescence (isotropic)

    # Calculate fluorescence emission line shape at each emission wavelength
    if double_gaussian:
        h_C = chl_fl.emission_line_double_gaussian(wavelength)
    else:
        h_C = chl_fl.emission_line_single_gaussian(wavelength)

    # Upwelling attenuation at each emission wavelength.  The old code froze
    # this at λ=685 nm, which overestimated the 730 nm secondary peak of the
    # double-Gaussian model by ~4× because pure-water absorption is ~4×
    # larger at 730 than at 685.  See dev/ChlFl/double_gaussian.py and the
    # Logs section of prompts/chl_fl.md for the investigation.
    a_em = np.asarray(a_em)
    bb_em = np.asarray(bb_em)
    kappa_F_em = (a_em + bb_em) / mu_f

    # Downwelling attenuation at each excitation wavelength
    K_ex = (a_ex + bb_ex) / mu_d

    # Fluorescence backscattering coefficient at each excitation wavelength
    bb_F = chl_fl.fluorescence_backscattering_coeff(aph_ex, phi_C)

    # Build (n_em, n_ex) — or (n_samples, n_em, n_ex) for chains — denominator
    # K(λ') + κ_F(λ_em), keeping the proper λ_em dependence of κ_F.
    if ndim == 1:
        # K_ex: (n_ex,), kappa_F_em: (n_em,) -> denom: (n_em, n_ex)
        denom = K_ex[None, :] + kappa_F_em[:, None]
        # λ' / λ_em, per (em, ex) pair
        lambda_ratio = wavelength_ex[None, :] / wavelength[:, None]
        # integrand: (n_em, n_ex); Ed_ex and bb_F broadcast along axis 0
        integrand = (Ed_ex * lambda_ratio
                     * (bb_F / mu_d)[None, :] / denom)
        R_F = np.trapezoid(integrand, x=wavelength_ex, axis=1)
    else:
        # Chains path.  The full 3-D broadcast denom (n_samples, n_em, n_ex)
        # blows up RAM: for the standard biomass fit (n_samples~528k, n_em~60,
        # n_ex~60) that single tensor is ~14 GiB and the chained arithmetic
        # holds several live copies -> >100 GB.  But forming it one *emission
        # wavelength* at a time instead pays an n_em-long Python loop on every
        # call, which makes the n_samples==1 log_prob hot path ~9x slower and
        # the whole MCMC fit ~7x slower (see dev/ChlFl/mcmc_speed.py).
        #
        # Resolve both by chunking over samples: each chunk is evaluated with
        # the fast fully-vectorised 3-D op, but the chunk is sized so its
        # integrand stays under FL_CHUNK_ELEMENTS.  For n_samples==1 this is a
        # single vectorised block (as fast as the pre-memory-fix code); for
        # 528k samples it is many bounded-memory blocks (low RAM).  The
        # denominator K(λ') + κ_F(λ_em) is a *sum* so it can't factor across
        # the (em, ex) axes — chunking is what bounds the footprint.
        # See dev/ChlFl/memory_profile.py and the Logs in prompts/chl_fl.md.
        n_samples = a_ex.shape[0]
        if kappa_F_em.ndim == 1:
            kappa_F_em = np.broadcast_to(
                kappa_F_em, (n_samples,) + kappa_F_em.shape)
        if bb_F.ndim == 1:
            bb_F = np.broadcast_to(bb_F, (n_samples,) + bb_F.shape)

        n_em = wavelength.size
        n_ex = wavelength_ex.size
        R_F = np.empty((n_samples, n_em))
        # λ' / λ_em is sample-independent -> build once and reuse per chunk.
        lambda_ratio = (wavelength_ex[None, None, :]
                        / wavelength[None, :, None])  # (1, n_em, n_ex)
        chunk = max(1, FL_CHUNK_ELEMENTS // (n_em * n_ex))
        for lo in range(0, n_samples, chunk):
            hi = min(lo + chunk, n_samples)
            # denom: (m, n_em, n_ex) for this block of m = hi-lo samples
            denom = (K_ex[lo:hi, None, :]
                     + kappa_F_em[lo:hi, :, None])
            integrand = (Ed_ex[None, None, :] * lambda_ratio
                         * (bb_F[lo:hi, None, :] / mu_d) / denom)
            R_F[lo:hi] = np.trapezoid(integrand, x=wavelength_ex, axis=2)

    # Normalize by emission-wavelength irradiance
    R_F = R_F / Ed_em

    # Convert the two-flow *irradiance* reflectance R_F = Eu/Ed into a
    # subsurface remote-sensing reflectance rrs = Lu/Ed.  Fluorescence
    # emission is isotropic, so the upwelling radiance field is uniform
    # and Lu(0-) = Eu(0-)/pi.  Omitting this factor treats R_F as if it
    # were already a radiance-based rrs and inflates Rrs_fl by ~3x:
    # validated against the Loisel+23 HydroLight scenario pairs (X4-X2),
    # the median model/truth ratio at 685 nm is 1.01/0.96/0.87 at solar
    # zenith 0/30/60 deg with this factor, vs 3.18/3.00/2.73 without it
    # (retrieve-or-bust, context/RT/rt_inelastic_bing_summary.md).
    rrs_F = R_F / np.pi

    # Convert subsurface rrs to Rrs and apply the emission line shape.
    # rrs_F has shape (n_em,) or (n_samples, n_em); h_C broadcasts along
    # the n_em axis in both cases.
    Rrs_fl = h_C * A_Rrs * rrs_F / (1 - B_Rrs * rrs_F)

    return Rrs_fl


def calc_Rrs_fluorescence_simple(
    wavelength: Union[float, np.ndarray],
    Chl: float,
    phi_C: float = 0.02,
    lambda_ex: float = 440.0,
    mu_d: Optional[float] = None,
    mu_f: Optional[float] = None,
    double_gaussian: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate Rrs fluorescence contribution using simplified single-excitation model.

    This is a faster version that assumes all excitation occurs at a single
    wavelength (typically 440 nm, the blue absorption peak).

    Parameters
    ----------
    wavelength : float or ndarray
        Emission wavelength(s) λ in nanometers.
    Chl : float
        Chlorophyll-a concentration in mg m^-3.
    phi_C : float, optional
        Fluorescence quantum yield. Default is 0.02.
    lambda_ex : float, optional
        Excitation wavelength in nm. Default is 440 nm.
    mu_d : float, optional
        Mean cosine for downwelling irradiance. Default is 0.9.
    mu_f : float, optional
        Mean cosine for fluorescence emission. Default is 0.5.
    double_gaussian : bool, optional
        If True, use double Gaussian emission. Default is False.

    Returns
    -------
    float or ndarray
        Fluorescence contribution to Rrs in sr^-1.

    Examples
    --------
    >>> wavelengths = np.linspace(650, 750, 100)
    >>> Rrs_fl = calc_Rrs_fluorescence_simple(wavelengths, Chl=1.0)
    >>> peak_idx = np.argmax(Rrs_fl)
    >>> print(f"Peak at {wavelengths[peak_idx]:.0f} nm: {Rrs_fl[peak_idx]:.2e} sr^-1")
    """
    # JXP thinks this calculation is wrong
    raise NotImplementedError("calc_Rrs_fluorescence_simple is not implemented")
    from . import chl_fl, raman

    wavelength = np.atleast_1d(wavelength)

    # Use default mean cosines if not provided
    if mu_d is None:
        mu_d = raman.MU_D_DEFAULT
    if mu_f is None:
        mu_f = 0.5  # Default for fluorescence (isotropic)

    # IOPs at excitation wavelength
    a_ph_ex = calc_a_ph_bricaud(lambda_ex, Chl)
    a_w_ex = calc_a_water(lambda_ex)
    bb_w_ex = calc_bb_water(lambda_ex)

    a_ex = a_w_ex + a_ph_ex
    bb_ex = bb_w_ex

    # IOPs at emission wavelengths
    a_ph_em = calc_a_ph_bricaud(wavelength, Chl)
    a_w_em = calc_a_water(wavelength)
    bb_w_em = calc_bb_water(wavelength)

    a_em = a_w_em + a_ph_em
    bb_em = bb_w_em

    # Fluorescence backscattering coefficient
    bb_F = chl_fl.fluorescence_backscattering_coeff(a_ph_ex, phi_C)

    # Attenuation coefficients
    K_ex = (a_ex + bb_ex) / mu_d
    kappa_F_em = (a_em + bb_em) / mu_f

    # Emission line shape
    if double_gaussian:
        h_C = chl_fl.emission_line_double_gaussian(wavelength)
    else:
        h_C = chl_fl.emission_line_single_gaussian(wavelength)

    # Wavelength ratio
    lambda_ratio = lambda_ex / wavelength

    # Subsurface fluorescence reflectance
    R_F = h_C * lambda_ratio * (bb_F / mu_d) / (K_ex + kappa_F_em)

    # Convert to Rrs
    Rrs_fl = A_Rrs * R_F / (1 - B_Rrs * R_F)

    return np.squeeze(Rrs_fl)
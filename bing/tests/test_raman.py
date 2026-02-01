""" Tests for Raman scattering and Raman corrections """
import os

import numpy as np

import pytest
import matplotlib.pyplot as plt

from ocpy.utils import plotting
from ocpy.satellites import pace as sat_pace

from bing.rt import raman
from bing.rt import rrs
from bing.parameters import standard
from bing.models import utils as model_utils


from IPython import embed


# =============================================================================
# Tests for raman.py module
# =============================================================================

def test_raman_seawater():

    print("Raman Scattering in Seawater")
    print("=" * 50)

    # Example 1: Scattering coefficient at various wavelengths
    wavelengths = np.array([350, 400, 450, 488, 500, 550, 600])
    print("\nRaman Scattering Coefficient b_R(λ'):")
    print("-" * 40)
    for lam in wavelengths:
        b_R = raman.raman_scattering_coeff(lam)
        lam_em = raman.excitation_to_emission_wavelength(lam)
        print(f"  λ' = {lam:3.0f} nm → λ ≈ {lam_em:5.1f} nm : "
              f"b_R = {b_R:.3e} m⁻¹")

    # Example 2: Summary at 488 nm
    print("\n" + "=" * 50)
    print("Summary at λ' = 488 nm (Argon laser line):")
    print("-" * 40)
    summary = raman.summary_at_wavelength(488)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4e}" if value < 0.01 else f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Example 3: Backscattering coefficient
    print("\n" + "=" * 50)
    print("Raman Backscattering Coefficient b_bR(λ'):")
    print("-" * 40)
    for lam in wavelengths:
        b_bR = raman.raman_backscattering_coeff(lam)
        b_R = raman.raman_scattering_coeff(lam)
        print(f"  λ' = {lam:3.0f} nm : b_bR = {b_bR:.3e} m⁻¹  "
              f"(ratio = {b_bR/b_R:.3f})")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Scattering coefficient vs wavelength
    ax1 = axes[0, 0]
    lam = np.linspace(300, 700, 200)
    b_R = raman.raman_scattering_coeff(lam)
    ax1.semilogy(lam, b_R * 1e4, 'b-', linewidth=2)
    ax1.set_xlabel(r'Excitation Wavelength $\lambda$\' (nm)')
    ax1.set_ylabel(r'$b_R \, \rm (×10^{-4} m^{-1})$')
    ax1.set_title('Raman Scattering Coefficient')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=2.6, color='r', linestyle='--', label='b_R(488nm) = 2.6×10⁻⁴')
    ax1.legend()

    # Plot 2: Wavenumber distribution function
    ax2 = axes[0, 1]
    delta_nu = np.linspace(2800, 4000, 300)
    g = raman.wavenumber_distribution(delta_nu)
    ax2.plot(delta_nu, g, 'b-', linewidth=2)
    ax2.set_xlabel(r'Wavenumber Shift $\Delta \nu$ (cm⁻¹)')
    ax2.set_ylabel(r'$g(\Delta \nu$) (cm)')
    ax2.set_title('Wavenumber Distribution (Walrafen 1967)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Emission spectra for different excitation wavelengths
    ax3 = axes[1, 0]
    for lam_ex in [400, 450, 500, 550]:
        lam_em, intensity = raman.get_emission_spectrum(lam_ex)
        # Normalize for plotting
        intensity_norm = intensity / intensity.max()
        ax3.plot(lam_em, intensity_norm, label=f'λ\' = {lam_ex} nm', linewidth=2)
    ax3.set_xlabel('Emission Wavelength λ (nm)')
    ax3.set_ylabel('Relative Intensity')
    ax3.set_title('Raman Emission Spectra')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Phase function
    ax4 = axes[1, 1]
    psi = np.linspace(0, np.pi, 180)
    psi_deg = np.degrees(psi)
    phase = raman.raman_phase_function(psi)
    ax4.plot(psi_deg, phase, 'b-', linewidth=2)
    ax4.set_xlabel('Scattering Angle ψ (degrees)')
    ax4.set_ylabel('β_R(ψ) (sr⁻¹)')
    ax4.set_title('Raman Phase Function')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=90, color='r', linestyle='--', alpha=0.5, label='Forward/Backward boundary')
    ax4.axvline(x=180, color='g', linestyle='--', alpha=0.5, label='Backscatter (180°)')
    ax4.legend()

    # 
    for ax in [ax1, ax2, ax3, ax4]:
        plotting.set_fontsize(ax, 15)

    plt.tight_layout()
    plt.savefig('raman_seawater_plots.png', dpi=150)
    plt.close()

    print("\n" + "=" * 50)
    print("Plots saved to: raman_seawater_plots.png")


# =============================================================================
# Tests for Raman scattering corrections in rrs.py
# Based on Sathyendranath & Platt (1998), Applied Optics 37, 2216-2227
# =============================================================================

def test_raman_backscattering_coeff():
    """Test Raman backscattering coefficient calculation."""
    # Test at reference wavelength (488 nm)
    bb_R_488 = raman.raman_backscattering_coeff(488)
    # Should be close to reference value (half of total scattering)
    assert np.isclose(bb_R_488, raman.B_RAMAN_488_HYDROLIGHT * 0.5, rtol=0.1)

    # Test wavelength dependence (shorter wavelength = higher scattering)
    bb_R_400 = raman.raman_backscattering_coeff(400)
    bb_R_600 = raman.raman_backscattering_coeff(600)
    assert bb_R_400 > bb_R_488 > bb_R_600

    # Test array input
    wavelengths = np.array([400, 450, 500, 550])
    bb_R_arr = raman.raman_backscattering_coeff(wavelengths)
    assert bb_R_arr.shape == (4,)
    assert np.all(np.diff(bb_R_arr) < 0)  # Decreasing with wavelength


def test_calc_R_elastic():
    """Test elastic reflectance calculation."""
    a = 0.05   # m^-1
    bb = 0.002  # m^-1

    R_E = rrs.calc_R_elastic(a, bb)

    # Reflectance should be positive and reasonable (< 0.1 for these values)
    assert R_E > 0
    assert R_E < 0.1

    # Test with arrays
    a_arr = np.array([0.03, 0.05, 0.1])
    bb_arr = np.array([0.003, 0.002, 0.001])
    R_E_arr = rrs.calc_R_elastic(a_arr, bb_arr)
    assert R_E_arr.shape == (3,)
    # Higher bb/a ratio should give higher reflectance
    assert R_E_arr[0] > R_E_arr[2]


def test_calc_R_raman_first_order():
    """Test first-order Raman reflectance calculation (Eq. 11)."""
    # Emission at 520 nm, excitation at 443 nm
    a_em, bb_em = 0.05, 0.002
    a_ex, bb_ex = 0.03, 0.003
    bb_R = raman.raman_backscattering_coeff(443)

    R_R = rrs.calc_R_raman_first_order(a_em, bb_em, a_ex, bb_ex, bb_R)

    # Should be positive
    assert R_R > 0
    # Typically smaller than elastic reflectance but significant
    R_E = rrs.calc_R_elastic(a_em, bb_em)
    assert R_R < R_E
    assert R_R > 0.001  # Should be non-negligible


def test_calc_R_raman_second_order():
    """Test second-order Raman reflectance terms (Eqs. 18, 23)."""
    a_em, bb_em = 0.05, 0.002
    a_ex, bb_ex = 0.03, 0.003
    bb_R = raman.raman_backscattering_coeff(443)

    R_R = rrs.calc_R_raman_first_order(a_em, bb_em, a_ex, bb_ex, bb_R)
    R_RE = rrs.calc_R_raman_RE(a_em, bb_em, a_ex, bb_ex, bb_R)
    R_ER = rrs.calc_R_raman_ER(a_em, bb_em, a_ex, bb_ex, bb_R)

    # Second-order terms should be positive
    assert R_RE > 0
    assert R_ER > 0

    # Second-order terms should be much smaller than first-order
    # Paper says ~10% of first-order term
    assert R_RE < 0.2 * R_R
    assert R_ER < 0.2 * R_R

    # Combined second-order should be ~10-20% of first-order
    second_order_ratio = (R_RE + R_ER) / R_R
    assert 0.05 < second_order_ratio < 0.25


def test_calc_R_raman_total():
    """Test total Raman reflectance calculation."""
    a_em, bb_em = 0.05, 0.002
    a_ex, bb_ex = 0.03, 0.003
    bb_R = raman.raman_backscattering_coeff(443)

    # With second-order terms
    R_total = rrs.calc_R_raman_total(
        a_em, bb_em, a_ex, bb_ex, bb_R, include_second_order=True
    )

    # Without second-order terms
    R_first_only = rrs.calc_R_raman_total(
        a_em, bb_em, a_ex, bb_ex, bb_R, include_second_order=False
    )

    # Total should be greater than first-order only
    assert R_total > R_first_only

    # Difference should be the second-order contribution
    R_R = rrs.calc_R_raman_first_order(a_em, bb_em, a_ex, bb_ex, bb_R)
    assert np.isclose(R_first_only, R_R)


def test_calc_raman_correction_factor():
    """Test Raman correction factor calculation."""
    a_em, bb_em = 0.05, 0.002
    a_ex, bb_ex = 0.03, 0.003
    bb_R = raman.raman_backscattering_coeff(443)

    corr = rrs.calc_raman_correction_factor(
        a_em, bb_em, a_ex, bb_ex, bb_R
    )

    # Correction factor should be > 1 (Raman adds to reflectance)
    assert corr > 1.0
    # Typically < 1.5 for these conditions
    assert corr < 1.5

    # For clearer water, correction should be larger
    a_clear, bb_clear = 0.02, 0.001
    corr_clear = rrs.calc_raman_correction_factor(
        a_clear, bb_clear, a_ex, bb_ex, bb_R
    )
    assert corr_clear > corr


def test_calc_Rrs_with_raman():
    """Test Rrs calculation with Raman correction."""
    a_em, bb_em = 0.05, 0.002
    a_ex, bb_ex = 0.03, 0.003
    bb_R = raman.raman_backscattering_coeff(443)

    # Elastic only
    Rrs_elastic = rrs.calc_Rrs(a_em, bb_em)

    # With Raman
    Rrs_with_raman = rrs.calc_Rrs_with_raman(
        a_em, bb_em, a_ex, bb_ex, bb_R
    )

    # With Raman should be higher
    assert Rrs_with_raman > Rrs_elastic

    # Both should be positive and reasonable
    assert Rrs_elastic > 0
    assert Rrs_with_raman > 0
    assert Rrs_elastic < 0.1
    assert Rrs_with_raman < 0.1


def test_array_calculations():
    """Test that all functions work with array inputs."""
    # Multiple wavelengths
    wavelengths_em = np.array([480, 520, 560, 600])
    wavelengths_ex = wavelengths_em - 80  # Approximate excitation wavelengths

    a_em = np.array([0.04, 0.05, 0.07, 0.25])
    bb_em = np.array([0.003, 0.002, 0.0015, 0.001])
    a_ex = np.array([0.025, 0.03, 0.035, 0.04])
    bb_ex = np.array([0.004, 0.003, 0.0025, 0.002])
    bb_R = raman.raman_backscattering_coeff(wavelengths_ex)

    # Test all functions with arrays
    R_E = rrs.calc_R_elastic(a_em, bb_em)
    R_R = rrs.calc_R_raman_first_order(a_em, bb_em, a_ex, bb_ex, bb_R)
    R_RE = rrs.calc_R_raman_RE(a_em, bb_em, a_ex, bb_ex, bb_R)
    R_ER = rrs.calc_R_raman_ER(a_em, bb_em, a_ex, bb_ex, bb_R)
    R_total = rrs.calc_R_raman_total(a_em, bb_em, a_ex, bb_ex, bb_R)
    corr = rrs.calc_raman_correction_factor(a_em, bb_em, a_ex, bb_ex, bb_R)
    Rrs = rrs.calc_Rrs_with_raman(a_em, bb_em, a_ex, bb_ex, bb_R)

    # Check shapes
    assert R_E.shape == (4,)
    assert R_R.shape == (4,)
    assert R_RE.shape == (4,)
    assert R_ER.shape == (4,)
    assert R_total.shape == (4,)
    assert corr.shape == (4,)
    assert Rrs.shape == (4,)

    # All values should be positive
    assert np.all(R_E > 0)
    assert np.all(R_R > 0)
    assert np.all(R_RE > 0)
    assert np.all(R_ER > 0)
    assert np.all(R_total > 0)
    assert np.all(corr > 1)
    assert np.all(Rrs > 0)


def test_wavelength_dependence():
    """Test that Raman contribution is significant in clear water."""
    # Clear water conditions
    # Absorption increases with wavelength (especially red)
    wavelengths = np.array([450, 500, 550, 600])
    a_water = np.array([0.015, 0.026, 0.064, 0.24])  # Approximate pure water absorption
    bb_water = np.array([0.0025, 0.0018, 0.0013, 0.001])  # Pure water backscatter

    # Get excitation wavelengths (~80 nm shorter for ~3400 cm^-1 shift)
    wavelengths_ex = wavelengths - 80
    a_ex = np.interp(wavelengths_ex, wavelengths, a_water)
    bb_ex = np.interp(wavelengths_ex, wavelengths, bb_water)
    bb_R = raman.raman_backscattering_coeff(wavelengths_ex)

    # Calculate correction factors
    corr = rrs.calc_raman_correction_factor(
        a_water, bb_water, a_ex, bb_ex, bb_R
    )

    # Correction should be significant (> 1) at all wavelengths
    assert np.all(corr > 1.0)


def test_Ed_ratio_effect():
    """Test effect of downwelling irradiance ratio on Raman contribution."""
    a_em, bb_em = 0.05, 0.002
    a_ex, bb_ex = 0.03, 0.003
    bb_R = raman.raman_backscattering_coeff(443)

    # Ed_ratio = 1 (equal irradiance at both wavelengths)
    R_R_1 = rrs.calc_R_raman_first_order(
        a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=1.0
    )

    # Ed_ratio = 1.5 (more irradiance at excitation wavelength)
    R_R_1p5 = rrs.calc_R_raman_first_order(
        a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=1.5
    )

    # Higher Ed_ratio should give higher Raman reflectance
    assert R_R_1p5 > R_R_1
    # Should scale linearly with Ed_ratio
    assert np.isclose(R_R_1p5 / R_R_1, 1.5, rtol=0.01)


def test_mean_cosine_sensitivity():
    """Test sensitivity to mean cosine parameters."""
    a_em, bb_em = 0.05, 0.002
    a_ex, bb_ex = 0.03, 0.003
    bb_R = raman.raman_backscattering_coeff(443)

    # Default mean cosines
    R_default = rrs.calc_R_raman_total(
        a_em, bb_em, a_ex, bb_ex, bb_R
    )

    # Higher mu_d (more direct sunlight)
    R_high_mud = rrs.calc_R_raman_total(
        a_em, bb_em, a_ex, bb_ex, bb_R, mu_d=0.95
    )

    # Lower mu_d (more diffuse light)
    R_low_mud = rrs.calc_R_raman_total(
        a_em, bb_em, a_ex, bb_ex, bb_R, mu_d=0.7
    )

    # Results should vary with mu_d
    assert R_high_mud != R_default
    assert R_low_mud != R_default


def test_consistency_with_gordon():
    """Test that elastic reflectance is consistent with Gordon formula."""
    a = 0.05
    bb = 0.002

    # Gordon formula Rrs
    Rrs_gordon = rrs.calc_Rrs(a, bb)

    # Our elastic reflectance converted to Rrs
    R_E = rrs.calc_R_elastic(a, bb, mu_d=0.9, mu_u=0.5)

    # Both should be in the same ballpark (within factor of 2-3)
    # They use different formulations but should give similar magnitude
    assert Rrs_gordon > 0
    assert R_E > 0
    assert 0.1 < Rrs_gordon / R_E < 10  # Order of magnitude check


def test_raman_wavelength_conversion():
    """Test excitation/emission wavelength conversion functions."""
    # At 488 nm excitation, emission should be around 583 nm
    lambda_ex = 488
    lambda_em = raman.excitation_to_emission_wavelength(lambda_ex)

    # Should be longer wavelength (Stokes shift)
    assert lambda_em > lambda_ex

    # For 3400 cm^-1 shift, expect ~95 nm difference at 488 nm
    shift = lambda_em - lambda_ex
    assert 80 < shift < 110

    # Round-trip conversion should return original
    lambda_ex_back = raman.emission_to_excitation_wavelength(lambda_em)
    assert np.isclose(lambda_ex, lambda_ex_back, rtol=0.001)

# Raman related methods

# Init
p_expb = standard.expb_pow(satellite='SBG', add_noise=True,
                           variable_Gordon=False, include_Raman=True)
model_wave = sat_pace.wave(wv_min=p_expb.wv_min,
                                   wv_max=p_expb.wv_max)                        
model_names=['ExpBricaud', 'Pow']
a_model, _ = model_utils.init(model_names, model_wave)

# Wave ex
a_model.init_raman()

# a_ex
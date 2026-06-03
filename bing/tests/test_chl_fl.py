""" Tests for Chlorophyll Fluorescence module """

from collections import namedtuple

import numpy as np
import pytest

from bing.rt import chl_fl
from bing.rt import rrs
from bing.rt import defs as rt_defs


# =============================================================================
# Tests for emission line shape functions
# =============================================================================

def test_emission_line_single_gaussian():
    """Test single Gaussian emission line shape."""
    # Test at emission peak (685 nm)
    h_peak = chl_fl.emission_line_single_gaussian(685.0)

    # Peak should be positive
    assert h_peak > 0

    # Test away from peak - should be lower
    h_650 = chl_fl.emission_line_single_gaussian(650.0)
    h_720 = chl_fl.emission_line_single_gaussian(720.0)
    assert h_peak > h_650
    assert h_peak > h_720

    # Test normalization (integral should be ~1)
    wavelengths = np.linspace(600, 800, 1000)
    h = chl_fl.emission_line_single_gaussian(wavelengths)
    integral = np.trapezoid(h, wavelengths)
    assert np.isclose(integral, 1.0, rtol=0.01)

    # Test array input
    wave_arr = np.array([660, 685, 710])
    h_arr = chl_fl.emission_line_single_gaussian(wave_arr)
    assert h_arr.shape == (3,)
    assert h_arr[1] > h_arr[0]  # Peak at 685 nm
    assert h_arr[1] > h_arr[2]


def test_emission_line_double_gaussian():
    """Test double Gaussian emission line shape."""
    # Test at primary peak (685 nm)
    h_685 = chl_fl.emission_line_double_gaussian(685.0)

    # Test at secondary peak (730 nm)
    h_730 = chl_fl.emission_line_double_gaussian(730.0)

    # Both should be positive
    assert h_685 > 0
    assert h_730 > 0

    # Primary peak should be higher (weight = 0.75)
    assert h_685 > h_730

    # Test normalization (integral should be ~1)
    wavelengths = np.linspace(600, 850, 1000)
    h = chl_fl.emission_line_double_gaussian(wavelengths)
    integral = np.trapezoid(h, wavelengths)
    assert np.isclose(integral, 1.0, rtol=0.01)

    # Test array input
    wave_arr = np.array([660, 685, 710, 730, 760])
    h_arr = chl_fl.emission_line_double_gaussian(wave_arr)
    assert h_arr.shape == (5,)


# =============================================================================
# Tests for quantum yield functions
# =============================================================================

def test_quantum_yield_constant():
    """Test constant quantum yield."""
    phi = chl_fl.quantum_yield_constant()
    assert phi == chl_fl.PHI_FL_DEFAULT
    assert phi == 0.02

    phi_custom = chl_fl.quantum_yield_constant(0.05)
    assert phi_custom == 0.05


def test_quantum_yield_irradiance_dependent():
    """Test irradiance-dependent quantum yield."""
    # At low light, QY should approach phi_max
    phi_low = chl_fl.quantum_yield_irradiance_dependent(PAR=0.0)
    assert np.isclose(phi_low, chl_fl.PHI_FL_LOW_LIGHT, rtol=0.01)

    # At high light, QY should approach phi_min
    phi_high = chl_fl.quantum_yield_irradiance_dependent(PAR=10000.0)
    assert phi_high < chl_fl.PHI_FL_LOW_LIGHT
    assert phi_high > chl_fl.PHI_FL_HIGH_LIGHT * 0.9  # Near minimum

    # Intermediate PAR should give intermediate QY
    phi_mid = chl_fl.quantum_yield_irradiance_dependent(PAR=100.0)
    assert phi_low > phi_mid > phi_high

    # Test array input
    PAR_arr = np.array([0, 50, 100, 200, 500])
    phi_arr = chl_fl.quantum_yield_irradiance_dependent(PAR_arr)
    assert phi_arr.shape == (5,)
    assert np.all(np.diff(phi_arr) < 0)  # Decreasing with PAR


def test_quantum_yield_depth_profile():
    """Test depth-dependent quantum yield."""
    # At surface (high light), QY should be lower
    phi_surface = chl_fl.quantum_yield_depth_profile(depth=0.0)

    # At depth (low light), QY should be higher
    phi_deep = chl_fl.quantum_yield_depth_profile(depth=100.0)

    assert phi_deep > phi_surface

    # Test array input
    depths = np.array([0, 10, 25, 50, 100])
    phi_z = chl_fl.quantum_yield_depth_profile(depths)
    assert phi_z.shape == (5,)
    assert np.all(np.diff(phi_z) > 0)  # Increasing with depth


# =============================================================================
# Tests for fluorescence coefficients
# =============================================================================

def test_fluorescence_scattering_coeff():
    """Test fluorescence scattering coefficient."""
    a_ph = 0.03  # m^-1
    phi_C = 0.02

    b_C = chl_fl.fluorescence_scattering_coeff(a_ph, phi_C)

    # b_C = phi_C * a_ph
    expected = phi_C * a_ph
    assert np.isclose(b_C, expected)

    # Test array input
    a_ph_arr = np.array([0.01, 0.03, 0.1])
    b_C_arr = chl_fl.fluorescence_scattering_coeff(a_ph_arr, phi_C)
    assert b_C_arr.shape == (3,)
    assert np.allclose(b_C_arr, phi_C * a_ph_arr)


def test_fluorescence_backscattering_coeff():
    """Test fluorescence backscattering coefficient."""
    a_ph = 0.03  # m^-1
    phi_C = 0.02

    bb_C = chl_fl.fluorescence_backscattering_coeff(a_ph, phi_C)
    b_C = chl_fl.fluorescence_scattering_coeff(a_ph, phi_C)

    # bb_C = 0.5 * b_C (isotropic emission)
    assert np.isclose(bb_C, 0.5 * b_C)

    # Test value
    expected = 0.5 * phi_C * a_ph
    assert np.isclose(bb_C, expected)


def test_fluorescence_backscatter_fraction():
    """Test backscatter fraction for isotropic emission."""
    frac = chl_fl.fluorescence_backscatter_fraction()
    assert frac == 0.5


# =============================================================================
# Tests for phase function
# =============================================================================

def test_fluorescence_phase_function():
    """Test isotropic fluorescence phase function."""
    # Any angle should give 1/(4*pi)
    psi_0 = 0.0
    psi_90 = np.pi / 2
    psi_180 = np.pi

    phase_0 = chl_fl.fluorescence_phase_function(psi_0)
    phase_90 = chl_fl.fluorescence_phase_function(psi_90)
    phase_180 = chl_fl.fluorescence_phase_function(psi_180)

    expected = 1.0 / (4 * np.pi)
    assert np.isclose(phase_0, expected)
    assert np.isclose(phase_90, expected)
    assert np.isclose(phase_180, expected)

    # All angles should be equal (isotropic)
    assert np.isclose(phase_0, phase_90)
    assert np.isclose(phase_90, phase_180)

    # Test array input
    psi_arr = np.linspace(0, np.pi, 10)
    phase_arr = chl_fl.fluorescence_phase_function(psi_arr)
    assert phase_arr.shape == (10,)
    assert np.allclose(phase_arr, expected)


# =============================================================================
# Tests for wavelength redistribution
# =============================================================================

def test_wavelength_redistribution():
    """Test wavelength redistribution function."""
    lambda_ex = 440.0  # nm
    lambda_em = 685.0  # nm
    phi_C = 0.02

    f_C = chl_fl.wavelength_redistribution(lambda_ex, lambda_em, phi_C)

    # Should be positive
    assert f_C > 0

    # Test outside excitation range
    lambda_ex_outside = 700.0  # Outside 370-690 nm range
    f_C_outside = chl_fl.wavelength_redistribution(lambda_ex_outside, lambda_em, phi_C)
    assert f_C_outside == 0

    # Test array emission wavelengths
    lambda_em_arr = np.array([650, 685, 720, 750])
    f_C_arr = chl_fl.wavelength_redistribution(lambda_ex, lambda_em_arr, phi_C)
    assert f_C_arr.shape == (4,)
    # Peak should be at 685 nm
    assert f_C_arr[1] > f_C_arr[0]
    assert f_C_arr[1] > f_C_arr[2]


def test_absorption_efficiency():
    """Test absorption efficiency function."""
    a_ph = 0.03

    # Within excitation range (370-690 nm)
    g_440 = chl_fl.absorption_efficiency(440.0, a_ph)
    assert g_440 == 1.0

    g_550 = chl_fl.absorption_efficiency(550.0, a_ph)
    assert g_550 == 1.0

    # Outside excitation range
    g_350 = chl_fl.absorption_efficiency(350.0, a_ph)
    assert g_350 == 0.0

    g_700 = chl_fl.absorption_efficiency(700.0, a_ph)
    assert g_700 == 0.0

    # Test array input
    wavelengths = np.array([350, 400, 500, 690, 700])
    g_arr = chl_fl.absorption_efficiency(wavelengths, a_ph)
    expected = np.array([0, 1, 1, 1, 0])
    assert np.array_equal(g_arr, expected)


# =============================================================================
# Tests for low-level reflectance calculations (chl_fl module)
# =============================================================================

def test_calc_R_fluorescence():
    """Test fluorescence reflectance calculation."""
    # Typical values
    a_em, bb_em = 0.5, 0.002     # at 685 nm (high water absorption)
    a_ex, bb_ex = 0.1, 0.003     # at 440 nm
    a_ph_ex = 0.03               # phytoplankton absorption at 440 nm
    phi_C = 0.02

    R_F = chl_fl.calc_R_fluorescence(
        a_em, bb_em, a_ex, bb_ex, a_ph_ex, phi_C=phi_C
    )

    # Should be positive
    assert R_F > 0

    # Should be reasonable magnitude (typically small)
    assert R_F < 0.01

    # Test with Ed_ratio
    R_F_2 = chl_fl.calc_R_fluorescence(
        a_em, bb_em, a_ex, bb_ex, a_ph_ex, Ed_ratio=2.0, phi_C=phi_C
    )
    # Should scale linearly with Ed_ratio
    assert np.isclose(R_F_2, 2.0 * R_F, rtol=0.01)


def test_calc_R_fluorescence_sensitivity():
    """Test sensitivity of fluorescence reflectance to parameters."""
    a_em, bb_em = 0.5, 0.002
    a_ex, bb_ex = 0.1, 0.003
    a_ph_ex = 0.03

    # Higher chlorophyll (higher a_ph) should give higher fluorescence
    R_F_low = chl_fl.calc_R_fluorescence(
        a_em, bb_em, a_ex, bb_ex, a_ph_ex=0.01
    )
    R_F_high = chl_fl.calc_R_fluorescence(
        a_em, bb_em, a_ex, bb_ex, a_ph_ex=0.1
    )
    assert R_F_high > R_F_low

    # Higher quantum yield should give higher fluorescence
    R_F_low_phi = chl_fl.calc_R_fluorescence(
        a_em, bb_em, a_ex, bb_ex, a_ph_ex, phi_C=0.01
    )
    R_F_high_phi = chl_fl.calc_R_fluorescence(
        a_em, bb_em, a_ex, bb_ex, a_ph_ex, phi_C=0.05
    )
    assert R_F_high_phi > R_F_low_phi


def test_calc_R_fluorescence_array():
    """Test fluorescence reflectance with array inputs."""
    # Multiple emission wavelengths
    a_em = np.array([0.3, 0.5, 0.8])
    bb_em = np.array([0.002, 0.002, 0.001])
    a_ex = 0.1
    bb_ex = 0.003
    a_ph_ex = 0.03

    R_F = chl_fl.calc_R_fluorescence(a_em, bb_em, a_ex, bb_ex, a_ph_ex)

    assert R_F.shape == (3,)
    assert np.all(R_F > 0)


# =============================================================================
# Tests for Fluorescence Line Height
# =============================================================================

def test_calc_fluorescence_line_height():
    """Test Fluorescence Line Height calculation."""
    # Typical Rrs values around red peak
    Rrs_665 = 0.001
    Rrs_680 = 0.0015  # Higher due to fluorescence
    Rrs_709 = 0.0008

    FLH = chl_fl.calc_fluorescence_line_height(Rrs_665, Rrs_680, Rrs_709)

    # Should be positive (fluorescence adds to Rrs at 680)
    assert FLH > 0

    # Calculate expected baseline
    w = (680 - 665) / (709 - 665)
    baseline = Rrs_665 + w * (Rrs_709 - Rrs_665)
    expected_FLH = Rrs_680 - baseline

    assert np.isclose(FLH, expected_FLH)


def test_calc_fluorescence_line_height_no_fluorescence():
    """Test FLH when there's no fluorescence signal."""
    # Linear spectrum (no fluorescence peak)
    Rrs_665 = 0.001
    Rrs_709 = 0.0008
    # Linear interpolation to 680
    w = (680 - 665) / (709 - 665)
    Rrs_680 = Rrs_665 + w * (Rrs_709 - Rrs_665)

    FLH = chl_fl.calc_fluorescence_line_height(Rrs_665, Rrs_680, Rrs_709)

    # Should be zero (or very close)
    assert np.isclose(FLH, 0.0, atol=1e-10)


def test_calc_fluorescence_line_height_array():
    """Test FLH with array inputs."""
    Rrs_665 = np.array([0.001, 0.002, 0.0015])
    Rrs_680 = np.array([0.0015, 0.0028, 0.002])
    Rrs_709 = np.array([0.0008, 0.0015, 0.001])

    FLH = chl_fl.calc_fluorescence_line_height(Rrs_665, Rrs_680, Rrs_709)

    assert FLH.shape == (3,)
    assert np.all(FLH > 0)


def test_calc_normalized_fluorescence_line_height():
    """Test normalized FLH calculation."""
    Rrs_665 = 0.001
    Rrs_680 = 0.0015
    Rrs_709 = 0.0008

    FLH = chl_fl.calc_fluorescence_line_height(Rrs_665, Rrs_680, Rrs_709)
    nFLH = chl_fl.calc_normalized_fluorescence_line_height(Rrs_665, Rrs_680, Rrs_709)

    # nFLH should be FLH / baseline
    w = (680 - 665) / (709 - 665)
    baseline = Rrs_665 + w * (Rrs_709 - Rrs_665)
    expected_nFLH = FLH / baseline

    assert np.isclose(nFLH, expected_nFLH)


# =============================================================================
# Tests for convenience functions
# =============================================================================

def test_get_emission_spectrum():
    """Test emission spectrum generation."""
    wavelength_ex = 450.0

    wave_em, intensity = chl_fl.get_emission_spectrum(wavelength_ex)

    # Check output shapes
    assert wave_em.shape == (100,)
    assert intensity.shape == (100,)

    # Check wavelength range
    assert wave_em[0] == 640.0
    assert wave_em[-1] == 800.0

    # Peak should be around 685 nm
    peak_idx = np.argmax(intensity)
    assert 680 < wave_em[peak_idx] < 690


def test_get_emission_spectrum_double_gaussian():
    """Test emission spectrum with double Gaussian model."""
    wavelength_ex = 450.0

    wave_em_single, intensity_single = chl_fl.get_emission_spectrum(
        wavelength_ex, double_gaussian=False
    )
    wave_em_double, intensity_double = chl_fl.get_emission_spectrum(
        wavelength_ex, double_gaussian=True
    )

    # Double Gaussian should have more intensity at ~730 nm
    idx_730 = np.argmin(np.abs(wave_em_single - 730))
    assert intensity_double[idx_730] > intensity_single[idx_730]


def test_summary_at_wavelength():
    """Test summary function."""
    wavelength_ex = 440.0
    a_ph = 0.03
    phi_C = 0.02

    summary = chl_fl.summary_at_wavelength(wavelength_ex, a_ph, phi_C)

    # Check all expected keys are present
    expected_keys = [
        'excitation_wavelength_nm',
        'emission_peak_primary_nm',
        'emission_peak_secondary_nm',
        'emission_fwhm_primary_nm',
        'emission_fwhm_secondary_nm',
        'in_excitation_range',
        'quantum_yield',
        'phytoplankton_absorption_m-1',
        'fluorescence_scattering_coeff_m-1',
        'fluorescence_backscatter_coeff_m-1',
        'backscatter_fraction',
    ]

    for key in expected_keys:
        assert key in summary

    # Check values
    assert summary['excitation_wavelength_nm'] == 440.0
    assert summary['emission_peak_primary_nm'] == 685.0
    assert summary['in_excitation_range'] == True
    assert summary['quantum_yield'] == phi_C
    assert summary['backscatter_fraction'] == 0.5


def test_summary_outside_excitation_range():
    """Test summary when outside excitation range."""
    wavelength_ex = 700.0  # Outside 370-690 nm
    a_ph = 0.03

    summary = chl_fl.summary_at_wavelength(wavelength_ex, a_ph)

    assert summary['in_excitation_range'] == False
    assert summary['fluorescence_scattering_coeff_m-1'] == 0.0
    assert summary['fluorescence_backscatter_coeff_m-1'] == 0.0


# =============================================================================
# Tests for volume scattering function
# =============================================================================

def test_fluorescence_vsf():
    """Test fluorescence volume scattering function."""
    wavelength_ex = 440.0
    wavelength_em = 685.0
    psi = np.pi / 2  # 90 degrees
    a_ph = 0.03
    phi_C = 0.02

    beta_C = chl_fl.fluorescence_vsf(
        wavelength_ex, wavelength_em, psi, a_ph, phi_C
    )

    # Should be positive
    assert beta_C > 0

    # Test array of angles (should all be equal for isotropic)
    psi_arr = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    beta_arr = chl_fl.fluorescence_vsf(
        wavelength_ex, wavelength_em, psi_arr, a_ph, phi_C
    )

    assert beta_arr.shape == (5,)
    # All values should be equal (isotropic phase function)
    assert np.allclose(beta_arr, beta_arr[0])


# =============================================================================
# Tests for the top-level calc_Rrs_fluorescence (bing.rt.rrs)
# =============================================================================

# Helper: build a small set of flat IOPs and irradiance arrays for the
# integrated Rrs fluorescence calculation. Kept local to avoid hard-coding
# the downwelling spectrum, which depends on the optional correct_atmosphere
# package.
def _flat_inputs(n_em=21, n_ex=29, a0=0.5, bb0=0.002,
                 a_ex0=0.1, bb_ex0=0.003, aph_ex0=0.03, Ed0=1.0):
    # Return arguments in the positional order expected by
    # rrs.calc_Rrs_fluorescence so they can be splatted directly.
    wave = np.linspace(650.0, 750.0, n_em)
    wave_ex = np.linspace(400.0, 680.0, n_ex)
    a_em = a0 * np.ones(n_em)
    bb_em = bb0 * np.ones(n_em)
    a_ex = a_ex0 * np.ones(n_ex)
    bb_ex = bb_ex0 * np.ones(n_ex)
    aph_ex = aph_ex0 * np.ones(n_ex)
    Ed_ex = Ed0 * np.ones(n_ex)
    Ed_em = Ed0
    return wave, a_em, bb_em, a_ex, bb_ex, aph_ex, wave_ex, Ed_ex, Ed_em


def test_calc_Rrs_fluorescence_basic():
    """Top-level calc_Rrs_fluorescence: positive, peaks near 685 nm."""
    wave, a_em, bb_em, a_ex, bb_ex, aph_ex, wave_ex, Ed_ex, Ed_em = _flat_inputs()

    Rrs_fl = rrs.calc_Rrs_fluorescence(
        wave, a_em, bb_em,
        a_ex, bb_ex, aph_ex,
        wave_ex, Ed_ex, Ed_em,
        phi_C=0.02,
        double_gaussian=True,
    )

    # Shape matches emission grid
    assert Rrs_fl.shape == wave.shape

    # All non-negative and small
    assert np.all(Rrs_fl >= 0)
    assert np.max(Rrs_fl) < 0.01

    # Peak should land near the primary emission peak (685 nm)
    peak_idx = np.argmax(Rrs_fl)
    assert 680.0 <= wave[peak_idx] <= 695.0


def test_calc_Rrs_fluorescence_aph_scaling():
    """Higher phytoplankton absorption gives stronger fluorescence."""
    base = _flat_inputs(aph_ex0=0.01)
    high = _flat_inputs(aph_ex0=0.05)

    Rrs_low = rrs.calc_Rrs_fluorescence(*base, phi_C=0.02, double_gaussian=True)
    Rrs_hi  = rrs.calc_Rrs_fluorescence(*high, phi_C=0.02, double_gaussian=True)

    # Linear scaling at the peak (a_em/bb_em are identical, so the only thing
    # that changes is b_bF ~ phi_C * a_ph(ex))
    assert np.max(Rrs_hi) > np.max(Rrs_low)


def test_calc_Rrs_fluorescence_phi_scaling():
    """Higher quantum yield gives stronger fluorescence."""
    args = _flat_inputs()

    Rrs_low_phi = rrs.calc_Rrs_fluorescence(*args, phi_C=0.01, double_gaussian=True)
    Rrs_hi_phi  = rrs.calc_Rrs_fluorescence(*args, phi_C=0.05, double_gaussian=True)

    assert np.max(Rrs_hi_phi) > np.max(Rrs_low_phi)


def test_calc_Rrs_fluorescence_double_vs_single_gaussian():
    """Double Gaussian shifts some signal into the ~730 nm secondary peak.

    With κ_F evaluated per emission wavelength (flat a_em here so κ_F is the
    same at 685 and 730), the primary peak shrinks to ~0.75× the single-
    Gaussian peak because that's the area weight in the double-Gaussian shape.
    """
    wave, a_em, bb_em, a_ex, bb_ex, aph_ex, wave_ex, Ed_ex, Ed_em = _flat_inputs()

    Rrs_single = rrs.calc_Rrs_fluorescence(
        wave, a_em, bb_em, a_ex, bb_ex, aph_ex,
        wave_ex, Ed_ex, Ed_em, phi_C=0.02, double_gaussian=False)
    Rrs_double = rrs.calc_Rrs_fluorescence(
        wave, a_em, bb_em, a_ex, bb_ex, aph_ex,
        wave_ex, Ed_ex, Ed_em, phi_C=0.02, double_gaussian=True)

    # Around 730 nm, the double-Gaussian model has the secondary peak
    idx_685 = np.argmin(np.abs(wave - 685.0))
    idx_730 = np.argmin(np.abs(wave - 730.0))
    assert Rrs_double[idx_730] > Rrs_single[idx_730]

    # At the primary peak the ratio is set purely by the emission-shape weight
    # (κ_F is identical across λ_em with flat a_em).  Allow some slack because
    # the peak bin may not land exactly on 685 nm.
    primary_ratio = Rrs_double[idx_685] / Rrs_single[idx_685]
    assert 0.70 < primary_ratio < 0.80


def _stepped_em_inputs(n_em=21, n_ex=29):
    """Same as ``_flat_inputs`` but with a step in ``a_em``: low near 685 nm,
    high near 730 nm.  Mimics the steep rise of pure-water absorption between
    those two wavelengths (a_w(685)≈0.49, a_w(730)≈1.96 m^-1) and is what
    exposes the κ_F(λ_em) dependence in calc_Rrs_fluorescence.
    """
    wave = np.linspace(650.0, 750.0, n_em)
    wave_ex = np.linspace(400.0, 680.0, n_ex)
    # a_em: ~0.5 at 685, ramping linearly to ~2.0 at 730 and beyond.
    a_em = 0.5 + 1.5 * np.clip((wave - 685.0) / (730.0 - 685.0), 0.0, 1.0)
    bb_em = 0.002 * np.ones(n_em)
    a_ex = 0.1 * np.ones(n_ex)
    bb_ex = 0.003 * np.ones(n_ex)
    aph_ex = 0.03 * np.ones(n_ex)
    Ed_ex = np.ones(n_ex)
    Ed_em = 1.0
    return wave, a_em, bb_em, a_ex, bb_ex, aph_ex, wave_ex, Ed_ex, Ed_em


def test_calc_Rrs_fluorescence_per_lambda_kappa_F():
    """Regression test for the per-λ κ_F fix.

    With ``a_em(730) ≈ 4 × a_em(685)``, the upwelling attenuation at 730 nm
    is ~4× stronger than at 685 nm, so the 730-nm fluorescence shoulder
    must be much smaller than the emission-shape weight alone would predict.

    Pre-fix (κ_F frozen at 685): Rrs[730]/Rrs[685] ≈ 0.16.
    Post-fix: Rrs[730]/Rrs[685] should drop well below 0.10.
    """
    args = _stepped_em_inputs()
    wave = args[0]

    Rrs_double = rrs.calc_Rrs_fluorescence(
        *args, phi_C=0.02, double_gaussian=True)

    i685 = int(np.argmin(np.abs(wave - 685.0)))
    i730 = int(np.argmin(np.abs(wave - 730.0)))
    ratio = Rrs_double[i730] / Rrs_double[i685]

    assert 0.02 < ratio < 0.10, f'unexpected 730/685 ratio: {ratio:.3f}'


def test_calc_Rrs_fluorescence_matches_reference():
    """calc_Rrs_fluorescence must agree with an explicit per-λ_em integration.

    This pins the implementation: any future refactor that re-introduces the
    "freeze κ_F at 685 nm" shortcut will fail here.
    """
    args = _stepped_em_inputs()
    (wave, a_em, bb_em, a_ex, bb_ex, aph_ex,
     wave_ex, Ed_ex, Ed_em) = args

    # Reference: explicit loop over λ_em with κ_F(λ_em) and λ'/λ_em
    mu_d, mu_f = 0.9, 0.5
    phi_C = 0.02
    h_C = chl_fl.emission_line_double_gaussian(wave)
    kappa_F_em = (a_em + bb_em) / mu_f
    K_ex = (a_ex + bb_ex) / mu_d
    bb_F = chl_fl.fluorescence_backscattering_coeff(aph_ex, phi_C)
    R_F_ref = np.zeros_like(wave, dtype=float)
    for i, lam_em in enumerate(wave):
        lam_ratio = wave_ex / lam_em
        integrand = Ed_ex * lam_ratio * (bb_F / mu_d) / (K_ex + kappa_F_em[i])
        R_F_ref[i] = np.trapezoid(integrand, x=wave_ex)
    R_F_ref /= Ed_em
    A_Rrs, B_Rrs = 0.52, 1.7
    Rrs_ref = h_C * A_Rrs * R_F_ref / (1 - B_Rrs * R_F_ref)

    Rrs_got = rrs.calc_Rrs_fluorescence(
        *args, phi_C=phi_C, double_gaussian=True)

    np.testing.assert_allclose(Rrs_got, Rrs_ref, rtol=1e-12, atol=0.0)


def test_calc_Rrs_fluorescence_chains():
    """2D excitation inputs (MCMC chains) produce a 2D Rrs array."""
    wave, a_em, bb_em, a_ex, bb_ex, aph_ex, wave_ex, Ed_ex, Ed_em = _flat_inputs()

    n_samples = 4
    # Broadcast the per-spectrum excitation IOPs to (n_samples, nwave_ex).
    # Vary the phytoplankton absorption across samples so the result is not
    # degenerate.
    aph_ex_2d = aph_ex[None, :] * np.linspace(0.5, 2.0, n_samples)[:, None]
    a_ex_2d  = np.broadcast_to(a_ex,  (n_samples, a_ex.size)).copy()
    bb_ex_2d = np.broadcast_to(bb_ex, (n_samples, bb_ex.size)).copy()

    # For 2D excitation, the function also expects 2D emission IOPs and Ed_em
    a_em_2d  = np.broadcast_to(a_em,  (n_samples, a_em.size)).copy()
    bb_em_2d = np.broadcast_to(bb_em, (n_samples, bb_em.size)).copy()

    Rrs_fl = rrs.calc_Rrs_fluorescence(
        wave, a_em_2d, bb_em_2d,
        a_ex_2d, bb_ex_2d, aph_ex_2d,
        wave_ex, Ed_ex, Ed_em,
        phi_C=0.02,
        double_gaussian=True,
    )

    # Expect one Rrs spectrum per sample
    assert Rrs_fl.shape == (n_samples, wave.size)

    # Increasing a_ph across samples should give increasing peak Rrs_fl
    peaks = Rrs_fl.max(axis=1)
    assert np.all(np.diff(peaks) > 0)


# =============================================================================
# Tests for rt_dict_from_p (bing.rt.defs)
# =============================================================================

def test_rt_dict_from_p_defaults():
    """rt_dict_from_p extracts the RT options from a default p_ntuple."""
    from bing.parameters import p_ntuple

    p = p_ntuple.gen(model_names=['ExpBricaud', 'Pow'])
    rt_dict = rt_defs.rt_dict_from_p(p)

    # Default values defined in bing.parameters.p_ntuple.def_dict
    assert rt_dict['variable_Gordon'] is True
    assert rt_dict['variable_Gordon_G0'] is False
    assert rt_dict['variable_Gordon_bbp'] is False
    assert rt_dict['include_Raman'] is False
    assert rt_dict['include_Chl_fl'] is False
    assert rt_dict['phi_C'] == 0.02
    assert rt_dict['double_gaussian'] is True

    # No other unexpected keys are added
    assert set(rt_dict.keys()) == {
        'variable_Gordon', 'variable_Gordon_G0', 'variable_Gordon_bbp',
        'include_Raman', 'include_Chl_fl',
        'phi_C', 'double_gaussian',
    }


def test_rt_dict_from_p_chl_fl_enabled():
    """rt_dict_from_p propagates custom chlorophyll-fluorescence options."""
    from bing.parameters import p_ntuple

    p = p_ntuple.gen(
        model_names=['ExpBricaud', 'Pow'],
        include_Chl_fl=True,
        phi_C=0.05,
        double_gaussian=False,
    )
    rt_dict = rt_defs.rt_dict_from_p(p)

    assert rt_dict['include_Chl_fl'] is True
    assert rt_dict['phi_C'] == 0.05
    assert rt_dict['double_gaussian'] is False


def test_rt_dict_from_p_missing_attrs():
    """rt_dict_from_p returns None for missing attributes."""
    # A minimal named-tuple lacking all RT fields
    Minimal = namedtuple('Minimal', ['model_names'])
    p = Minimal(model_names=['ExpBricaud', 'Pow'])

    rt_dict = rt_defs.rt_dict_from_p(p)

    # All five keys are present and set to None
    for key in ('variable_Gordon', 'include_Raman', 'include_Chl_fl',
                'phi_C', 'double_gaussian'):
        assert key in rt_dict
        assert rt_dict[key] is None


# =============================================================================
# Integration tests
# =============================================================================

def test_end_to_end_fluorescence_calculation():
    """Test complete fluorescence calculation workflow with chl_fl primitives."""
    # Excitation at 440 nm, emission at 685 nm
    wavelength_ex = 440.0

    # IOPs at emission wavelength (red, high water absorption)
    a_w_em = 0.45      # Pure water absorption at 685 nm
    a_ph_em = 0.01     # Low phytoplankton absorption in red
    bb_w_em = 0.0005   # Pure water backscattering
    bb_p_em = 0.001    # Particle backscattering

    a_em = a_w_em + a_ph_em
    bb_em = bb_w_em + bb_p_em

    # IOPs at excitation wavelength (blue)
    a_w_ex = 0.01      # Pure water absorption at 440 nm
    a_ph_ex = 0.03     # Phytoplankton absorption peak
    bb_w_ex = 0.003    # Pure water backscattering
    bb_p_ex = 0.002    # Particle backscattering

    a_ex = a_w_ex + a_ph_ex
    bb_ex = bb_w_ex + bb_p_ex

    # Quantum yield (typical surface value)
    phi_C = 0.02

    # Calculate fluorescence reflectance
    R_F = chl_fl.calc_R_fluorescence(
        a_em, bb_em, a_ex, bb_ex, a_ph_ex, phi_C=phi_C
    )

    # Basic sanity checks
    assert R_F > 0
    assert R_F < 0.01  # Should be small but measurable

    # Calculate fluorescence coefficients
    b_C = chl_fl.fluorescence_scattering_coeff(a_ph_ex, phi_C)
    bb_C = chl_fl.fluorescence_backscattering_coeff(a_ph_ex, phi_C)

    assert b_C == phi_C * a_ph_ex
    assert bb_C == 0.5 * b_C

    # Check emission spectrum
    wave_em, intensity = chl_fl.get_emission_spectrum(wavelength_ex)
    peak_wave = wave_em[np.argmax(intensity)]
    assert np.isclose(peak_wave, 685.0, atol=2.0)


def test_fluorescence_vs_raman_comparison():
    """Compare fluorescence and Raman contributions (order of magnitude)."""
    from bing.rt import raman

    # Typical conditions
    a_em, bb_em = 0.5, 0.002     # at 685 nm
    a_ex, bb_ex = 0.1, 0.003     # at 440 nm
    a_ph_ex = 0.03               # phytoplankton absorption
    phi_C = 0.02

    # Fluorescence contribution
    R_F = chl_fl.calc_R_fluorescence(
        a_em, bb_em, a_ex, bb_ex, a_ph_ex, phi_C=phi_C
    )

    # Raman contribution (for comparison)
    bb_R = raman.raman_backscattering_coeff(440)

    # Both should be positive and small
    assert R_F > 0
    assert bb_R > 0

    # Fluorescence backscatter coefficient
    bb_F = chl_fl.fluorescence_backscattering_coeff(a_ph_ex, phi_C)

    # Both are typically in similar range (10^-5 to 10^-2)
    assert 1e-5 < bb_F < 1e-2
    assert 1e-5 < bb_R < 1e-2

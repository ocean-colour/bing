""" Tests for Chlorophyll Fluorescence module """
import os

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from bing.rt import chl_fl


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
    integral = np.trapz(h, wavelengths)
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
    integral = np.trapz(h, wavelengths)
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
# Tests for reflectance calculations
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
    # Note: The VSF varies with wavelength ratio, but for same em/ex wavelengths
    # the phase function contribution is equal
    assert np.allclose(beta_arr, beta_arr[0])


# =============================================================================
# Integration tests
# =============================================================================

def test_end_to_end_fluorescence_calculation():
    """Test complete fluorescence calculation workflow."""
    # Set up a realistic scenario
    # Excitation at 440 nm, emission at 685 nm
    wavelength_ex = 440.0
    wavelength_em = 685.0

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

    # Both are typically in similar range (10^-4 to 10^-3)
    assert 1e-5 < bb_F < 1e-2
    assert 1e-5 < bb_R < 1e-2


# =============================================================================
# Run all tests (for notebook integration)
# =============================================================================

def run_all_tests():
    """Run all chlorophyll fluorescence tests and return summary."""
    tests = [
        ("Emission line single Gaussian", test_emission_line_single_gaussian),
        ("Emission line double Gaussian", test_emission_line_double_gaussian),
        ("Quantum yield constant", test_quantum_yield_constant),
        ("Quantum yield irradiance dependent", test_quantum_yield_irradiance_dependent),
        ("Quantum yield depth profile", test_quantum_yield_depth_profile),
        ("Fluorescence scattering coefficient", test_fluorescence_scattering_coeff),
        ("Fluorescence backscattering coefficient", test_fluorescence_backscattering_coeff),
        ("Fluorescence backscatter fraction", test_fluorescence_backscatter_fraction),
        ("Fluorescence phase function", test_fluorescence_phase_function),
        ("Wavelength redistribution", test_wavelength_redistribution),
        ("Absorption efficiency", test_absorption_efficiency),
        ("Fluorescence reflectance", test_calc_R_fluorescence),
        ("Fluorescence reflectance sensitivity", test_calc_R_fluorescence_sensitivity),
        ("Fluorescence reflectance array", test_calc_R_fluorescence_array),
        ("Fluorescence Line Height", test_calc_fluorescence_line_height),
        ("FLH no fluorescence", test_calc_fluorescence_line_height_no_fluorescence),
        ("FLH array", test_calc_fluorescence_line_height_array),
        ("Normalized FLH", test_calc_normalized_fluorescence_line_height),
        ("Get emission spectrum", test_get_emission_spectrum),
        ("Get emission spectrum double Gaussian", test_get_emission_spectrum_double_gaussian),
        ("Summary at wavelength", test_summary_at_wavelength),
        ("Summary outside excitation range", test_summary_outside_excitation_range),
        ("Fluorescence VSF", test_fluorescence_vsf),
        ("End-to-end fluorescence", test_end_to_end_fluorescence_calculation),
        ("Fluorescence vs Raman comparison", test_fluorescence_vs_raman_comparison),
    ]

    print("=" * 60)
    print("Running All Chlorophyll Fluorescence Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{'─' * 50}")
        print(f"Test: {name}")
        print("─" * 50)
        try:
            test_func()
            print(f"✓ Passed")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Summary: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    return passed, failed

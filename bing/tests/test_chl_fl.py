""" Tests for Chlorophyll Fluorescence module """

import numpy as np

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

# =============================================================================
# Tests for rrs.py fluorescence Rrs functions
# =============================================================================

def test_calc_a_ph_bricaud():
    """Test Bricaud phytoplankton absorption parameterization."""
    from bing.rt import rrs

    # Test single wavelength, single Chl
    a_ph = rrs.calc_a_ph_bricaud(440, 1.0)
    assert a_ph > 0
    assert 0.01 < a_ph < 0.2  # Reasonable range for Chl=1

    # Test wavelength dependence (blue peak > green)
    a_ph_440 = rrs.calc_a_ph_bricaud(440, 1.0)
    a_ph_550 = rrs.calc_a_ph_bricaud(550, 1.0)
    assert a_ph_440 > a_ph_550

    # Test Chl dependence (higher Chl = higher absorption)
    a_ph_low = rrs.calc_a_ph_bricaud(440, 0.1)
    a_ph_high = rrs.calc_a_ph_bricaud(440, 10.0)
    assert a_ph_high > a_ph_low

    # Test array wavelength input
    wavelengths = np.array([400, 440, 500, 550, 675])
    a_ph_arr = rrs.calc_a_ph_bricaud(wavelengths, 1.0)
    assert a_ph_arr.shape == (5,)
    assert np.all(a_ph_arr > 0)

    # Test array Chl input
    Chl_arr = np.array([0.1, 1.0, 10.0])
    a_ph_multi = rrs.calc_a_ph_bricaud(440, Chl_arr)
    assert a_ph_multi.shape == (3,)
    assert np.all(np.diff(a_ph_multi) > 0)  # Increasing with Chl


def test_calc_a_water():
    """Test pure water absorption calculation."""
    from bing.rt import rrs

    # Test at key wavelengths
    a_w_450 = rrs.calc_a_water(450)
    a_w_550 = rrs.calc_a_water(550)
    a_w_685 = rrs.calc_a_water(685)

    # Water absorption increases with wavelength in red
    assert a_w_685 > a_w_550 > a_w_450

    # Test known approximate values
    assert 0.005 < a_w_450 < 0.02   # Blue: low absorption
    assert 0.4 < a_w_685 < 0.6     # Red: high absorption

    # Test array input
    wavelengths = np.array([400, 500, 600, 700])
    a_w_arr = rrs.calc_a_water(wavelengths)
    assert a_w_arr.shape == (4,)
    assert np.all(a_w_arr > 0)


def test_calc_bb_water():
    """Test pure water backscattering calculation."""
    from bing.rt import rrs

    # Test at reference wavelength
    bb_w_500 = rrs.calc_bb_water(500)
    assert np.isclose(bb_w_500, 0.00144, rtol=0.01)

    # Test wavelength dependence (decreases with wavelength)
    bb_w_400 = rrs.calc_bb_water(400)
    bb_w_600 = rrs.calc_bb_water(600)
    assert bb_w_400 > bb_w_500 > bb_w_600

    # Test array input
    wavelengths = np.array([400, 500, 600, 700])
    bb_w_arr = rrs.calc_bb_water(wavelengths)
    assert bb_w_arr.shape == (4,)
    assert np.all(np.diff(bb_w_arr) < 0)  # Decreasing with wavelength


def test_calc_Rrs_fluorescence_simple():
    """Test simplified fluorescence Rrs calculation."""
    from bing.rt import rrs

    wavelength = np.arange(650, 751, 5)

    # Test for Chl = 1.0 mg/m³
    Rrs_fl = rrs.calc_Rrs_fluorescence_simple(wavelength, Chl=1.0)

    # Should be array of correct shape
    assert Rrs_fl.shape == wavelength.shape

    # All values should be positive
    assert np.all(Rrs_fl >= 0)

    # Peak should be at 685 nm
    peak_idx = np.argmax(Rrs_fl)
    assert 680 <= wavelength[peak_idx] <= 690

    # Test Chl dependence
    Rrs_fl_low = rrs.calc_Rrs_fluorescence_simple(wavelength, Chl=0.1)
    Rrs_fl_high = rrs.calc_Rrs_fluorescence_simple(wavelength, Chl=10.0)
    assert np.max(Rrs_fl_high) > np.max(Rrs_fl) > np.max(Rrs_fl_low)


def test_calc_Rrs_fluorescence_simple_quantum_yield():
    """Test quantum yield effect on fluorescence Rrs."""
    from bing.rt import rrs

    wavelength = np.array([685])  # At peak

    # Higher quantum yield should give higher Rrs
    Rrs_phi_low = rrs.calc_Rrs_fluorescence_simple(wavelength, Chl=1.0, phi_C=0.01)
    Rrs_phi_high = rrs.calc_Rrs_fluorescence_simple(wavelength, Chl=1.0, phi_C=0.05)

    assert Rrs_phi_high > Rrs_phi_low

    # Should scale approximately linearly with phi_C
    ratio = Rrs_phi_high / Rrs_phi_low
    expected_ratio = 0.05 / 0.01
    assert np.isclose(ratio, expected_ratio, rtol=0.1)


def test_calc_Rrs_fluorescence_simple_double_gaussian():
    """Test double Gaussian emission model."""
    from bing.rt import rrs

    wavelength = np.arange(650, 780, 5)

    Rrs_single = rrs.calc_Rrs_fluorescence_simple(
        wavelength, Chl=1.0, double_gaussian=False
    )
    Rrs_double = rrs.calc_Rrs_fluorescence_simple(
        wavelength, Chl=1.0, double_gaussian=True
    )

    # Double Gaussian should have more signal around 730 nm
    idx_730 = np.argmin(np.abs(wavelength - 730))
    assert Rrs_double[idx_730] > Rrs_single[idx_730]

    # Primary peak (685 nm) should be lower for double Gaussian
    # (because some weight goes to secondary peak)
    idx_685 = np.argmin(np.abs(wavelength - 685))
    assert Rrs_single[idx_685] > Rrs_double[idx_685]


def test_calc_Rrs_fluorescence_integrated():
    """Test integrated fluorescence Rrs calculation."""
    from bing.rt import rrs

    wavelength = np.arange(660, 720, 5)

    # Integrated calculation
    Rrs_int = rrs.calc_Rrs_fluorescence(wavelength, Chl=1.0)

    # Should be array of correct shape
    assert Rrs_int.shape == wavelength.shape

    # All values should be positive
    assert np.all(Rrs_int >= 0)

    # Peak should still be around 685 nm
    peak_idx = np.argmax(Rrs_int)
    assert 680 <= wavelength[peak_idx] <= 690


def test_calc_Rrs_with_fluorescence():
    """Test total Rrs with fluorescence."""
    from bing.rt import rrs

    wavelength = np.arange(650, 720, 5)

    # Create simple IOPs
    a = 0.5 * np.ones_like(wavelength, dtype=float)
    bb = 0.002 * np.ones_like(wavelength, dtype=float)

    # Elastic only
    Rrs_elastic = rrs.calc_Rrs(a, bb)

    # With fluorescence
    Rrs_total = rrs.calc_Rrs_with_fluorescence(wavelength, a, bb, Chl=1.0)

    # Total should be >= elastic everywhere
    assert np.all(Rrs_total >= Rrs_elastic * 0.999)  # Small tolerance

    # Enhancement should be largest around 685 nm
    enhancement = Rrs_total - Rrs_elastic
    peak_idx = np.argmax(enhancement)
    assert 680 <= wavelength[peak_idx] <= 690


def test_calc_fluorescence_spectrum():
    """Test fluorescence spectrum convenience function."""
    from bing.rt import rrs

    # Single Chl value
    Rrs_fl = rrs.calc_fluorescence_spectrum(Chl=1.0)
    assert Rrs_fl.shape == (101,)  # Default 650-750 nm at 1 nm

    # Multiple Chl values
    Chl_arr = [0.1, 1.0, 10.0]
    Rrs_fl_multi = rrs.calc_fluorescence_spectrum(Chl_arr)
    assert Rrs_fl_multi.shape == (3, 101)

    # Higher Chl should give higher fluorescence
    assert np.max(Rrs_fl_multi[2, :]) > np.max(Rrs_fl_multi[1, :]) > np.max(Rrs_fl_multi[0, :])


def test_calc_fluorescence_spectrum_with_components():
    """Test fluorescence spectrum with component output."""
    from bing.rt import rrs

    wavelength, Rrs_fl, components = rrs.calc_fluorescence_spectrum(
        Chl=1.0, return_components=True
    )

    # Check outputs
    assert wavelength.shape == (101,)
    assert Rrs_fl.shape == (101,)

    # Check components dictionary
    assert 'wavelength' in components
    assert 'emission_shape' in components
    assert 'a_water' in components
    assert 'a_ph' in components

    # Emission shape should be normalized
    integral = np.trapz(components['emission_shape'], components['wavelength'])
    assert np.isclose(integral, 1.0, rtol=0.05)


def test_calc_fluorescence_spectrum_custom_wavelength():
    """Test fluorescence spectrum with custom wavelength array."""
    from bing.rt import rrs

    custom_wave = np.arange(670, 710, 2)
    Rrs_fl = rrs.calc_fluorescence_spectrum(Chl=1.0, wavelength=custom_wave)

    assert Rrs_fl.shape == custom_wave.shape


def test_calc_fluorescence_correction_factor():
    """Test fluorescence correction factor calculation."""
    from bing.rt import rrs

    wavelength = np.array([650, 670, 685, 700, 720])
    a = 0.5 * np.ones_like(wavelength, dtype=float)
    bb = 0.002 * np.ones_like(wavelength, dtype=float)

    corr = rrs.calc_fluorescence_correction_factor(wavelength, a, bb, Chl=1.0)

    # Correction should be >= 1 everywhere (fluorescence adds signal)
    assert np.all(corr >= 1.0)

    # Maximum correction should be around 685 nm
    peak_idx = np.argmax(corr)
    assert 680 <= wavelength[peak_idx] <= 690

    # Correction should be ~1 far from emission peak
    assert np.isclose(corr[0], 1.0, rtol=0.01)  # 650 nm
    assert np.isclose(corr[-1], 1.0, rtol=0.01)  # 720 nm


def test_calc_fluorescence_correction_factor_chl_dependence():
    """Test correction factor dependence on Chl."""
    from bing.rt import rrs

    wavelength = np.array([685])
    a = np.array([0.5])
    bb = np.array([0.002])

    corr_low = rrs.calc_fluorescence_correction_factor(wavelength, a, bb, Chl=0.1)
    corr_mid = rrs.calc_fluorescence_correction_factor(wavelength, a, bb, Chl=1.0)
    corr_high = rrs.calc_fluorescence_correction_factor(wavelength, a, bb, Chl=10.0)

    # Higher Chl should give larger correction
    assert corr_high > corr_mid > corr_low


def test_fluorescence_rrs_physical_values():
    """Test that fluorescence Rrs values are physically reasonable."""
    from bing.rt import rrs

    wavelength = np.arange(650, 751, 1)

    # Typical ocean conditions
    for Chl in [0.1, 1.0, 10.0]:
        Rrs_fl = rrs.calc_Rrs_fluorescence_simple(wavelength, Chl)

        # All values should be positive
        assert np.all(Rrs_fl >= 0)

        # Peak should be reasonable magnitude
        # Typically 10^-7 to 10^-4 sr^-1 for ocean fluorescence
        peak = np.max(Rrs_fl)
        assert 1e-8 < peak < 1e-3


def test_fluorescence_rrs_consistency():
    """Test consistency between different fluorescence Rrs calculations."""
    from bing.rt import rrs

    wavelength = np.array([685])  # Single wavelength at peak
    Chl = 1.0

    # Simple calculation
    Rrs_simple = rrs.calc_Rrs_fluorescence_simple(wavelength, Chl)

    # From spectrum
    Rrs_spectrum = rrs.calc_fluorescence_spectrum(Chl, wavelength=wavelength)

    # Should be identical
    assert np.isclose(Rrs_simple, Rrs_spectrum, rtol=0.01)


def test_bricaud_coefficients_wavelength_coverage():
    """Test Bricaud parameterization across full wavelength range."""
    from bing.rt import rrs

    # Test wavelengths from UV to NIR
    wavelengths = np.arange(350, 750, 10)

    # Should not raise errors
    a_ph = rrs.calc_a_ph_bricaud(wavelengths, 1.0)

    # All values should be positive
    assert np.all(a_ph > 0)

    # Check spectral shape (peaks in blue and red)
    idx_440 = np.argmin(np.abs(wavelengths - 440))
    idx_550 = np.argmin(np.abs(wavelengths - 550))
    idx_675 = np.argmin(np.abs(wavelengths - 675))

    # Blue peak > green minimum
    assert a_ph[idx_440] > a_ph[idx_550]
    # Red peak > green minimum
    assert a_ph[idx_675] > a_ph[idx_550]


# =============================================================================
# Run all tests (for notebook integration)
# =============================================================================

def run_all_tests():
    """Run all chlorophyll fluorescence tests and return summary."""
    tests = [
        # chl_fl.py tests
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
        # rrs.py fluorescence Rrs tests
        ("Bricaud a_ph parameterization", test_calc_a_ph_bricaud),
        ("Water absorption", test_calc_a_water),
        ("Water backscattering", test_calc_bb_water),
        ("Rrs fluorescence simple", test_calc_Rrs_fluorescence_simple),
        ("Rrs fluorescence quantum yield", test_calc_Rrs_fluorescence_simple_quantum_yield),
        ("Rrs fluorescence double Gaussian", test_calc_Rrs_fluorescence_simple_double_gaussian),
        ("Rrs fluorescence integrated", test_calc_Rrs_fluorescence_integrated),
        ("Rrs with fluorescence", test_calc_Rrs_with_fluorescence),
        ("Fluorescence spectrum", test_calc_fluorescence_spectrum),
        ("Fluorescence spectrum components", test_calc_fluorescence_spectrum_with_components),
        ("Fluorescence spectrum custom wavelength", test_calc_fluorescence_spectrum_custom_wavelength),
        ("Fluorescence correction factor", test_calc_fluorescence_correction_factor),
        ("Correction factor Chl dependence", test_calc_fluorescence_correction_factor_chl_dependence),
        ("Fluorescence Rrs physical values", test_fluorescence_rrs_physical_values),
        ("Fluorescence Rrs consistency", test_fluorescence_rrs_consistency),
        ("Bricaud wavelength coverage", test_bricaud_coefficients_wavelength_coverage),
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

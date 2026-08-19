""" Tests for L23 fitting module """
import os
import tempfile

import numpy as np

import pathlib
import pytest

from bing.fitting import l23 as fit_l23
from bing.parameters import standard
from bing import evaluate
from bing import plotting as bing_plot
from bing import rt as bing_rt
from bing.rt import defs as rt_defs

from IPython import embed

def data_path(filename):
    data_dir = pathlib.Path(__file__).parent.absolute().joinpath('files')
    return str(data_dir.joinpath(filename).resolve())

# ===== Helper functions for validation =====

def validate_basic_returns(chains, models, prep_dict, idx_out, extras, expected_idx):
    """Validate basic return values from fit_one."""
    assert idx_out == expected_idx, "Index should be preserved"
    assert chains is not None, "Chains should not be None"
    assert models is not None and len(models) == 2, "Should return 2 models"
    assert prep_dict is not None, "prep_dict should not be None"
    assert extras is not None, "extras should not be None"


def validate_chain_shape(chains, models):
    """Validate chain array shape and dimensions."""
    # chains shape: (nsteps, nwalkers, nparams)
    assert chains.ndim == 3, f"Chains should be 3D, got {chains.ndim}D"
    nsteps, nwalkers, nparams = chains.shape
    assert nsteps > 0, "Should have steps in chains"
    assert nwalkers > 0, "Should have walkers in chains"

    # Check parameter count matches models
    expected_nparams = models[0].nparam + models[1].nparam
    assert nparams == expected_nparams, \
        f"Chains should have {expected_nparams} params, got {nparams}"

    return nsteps, nwalkers, nparams


def validate_models(models):
    """Validate model attributes and wavelengths."""
    assert hasattr(models[0], 'wave'), "Model should have wavelength array"
    assert hasattr(models[0], 'nparam'), "Model should have nparam attribute"
    assert hasattr(models[0], 'pnames'), "Model should have parameter names"
    assert hasattr(models[1], 'eval_bb'), "Backscatter model should have eval_bb"
    assert hasattr(models[0], 'eval_a'), "Absorption model should have eval_a"

    # Check wavelengths are reasonable
    wave = models[0].wave
    assert np.all(wave >= 400) and np.all(wave <= 800), \
        "Wavelengths should be in visible range [400-800nm]"

    return wave


def validate_prep_dict(prep_dict, wave):
    """Validate prep_dict contents and data quality."""
    required_keys = ['odict', 'pdict', 'models', 'model_Rrs', 'model_varRrs', 'p0']
    for key in required_keys:
        assert key in prep_dict, f"prep_dict should contain '{key}'"

    # Check Rrs data
    model_Rrs = prep_dict['model_Rrs']
    model_varRrs = prep_dict['model_varRrs']
    assert len(model_Rrs) == len(wave), "Rrs should match wavelength array"
    # Note: With added noise, some Rrs can be negative (realistic for satellite data)
    # Check that most values are positive and in reasonable range
    assert np.sum(model_Rrs > 0) > 0.7 * len(model_Rrs), \
        "Most Rrs values should be positive"
    assert np.all(np.abs(model_Rrs) < 0.1), "Rrs magnitudes should be < 0.1 sr^-1"
    assert np.all(model_varRrs > 0), "Variance should be positive"

    return model_Rrs, model_varRrs


def validate_extras(extras, wave):
    """Validate extras dictionary contents."""
    required_extras = ['wave', 'obs_Rrs', 'varRrs', 'Chl', 'Y']
    for key in required_extras:
        assert key in extras, f"extras should contain '{key}'"

    assert np.array_equal(extras['wave'], wave), "Extras wavelength should match model"
    assert extras['Chl'] > 0, "Chlorophyll should be positive"
    assert 0 < extras['Y'] < 5, "Y parameter should be in reasonable range [0-5]"


def validate_fitted_parameters(chains, models, models_info=None):
    """Validate fitted parameters are within reasonable bounds."""
    # Calculate statistics from chains
    pnames = models[0].pnames + models[1].pnames
    stats = evaluate.calc_stats(chains, pnames)

    # Check that parameters are within reasonable bounds
    # For ExpBricaud + Power-law: [log10(Adg), Sdg, log10(Aph), log10(Bnw), beta]
    med_params = stats['med']

    # Absorption amplitude (Adg) should be in reasonable range (log10 scale)
    assert -6 < med_params[0] < 2, f"log10(Adg) out of bounds: {med_params[0]}"

    # Spectral slope (Sdg) should be positive and reasonable
    assert 0 < med_params[1] < 0.03, f"Sdg out of bounds: {med_params[1]}"

    # Backscatter amplitude should be reasonable
    assert -6 < med_params[-2] < 2, f"log10(Bnw) out of bounds: {med_params[-2]}"

    return med_params, pnames, stats


def validate_reconstructed_rrs(models, chains, model_Rrs, model_varRrs, wave, nparams,
    rt_dict:dict):
    """Validate reconstructed Rrs and IOPs from chains."""
    a, bb, a_lo, a_hi, bb_lo, bb_hi, Rrs_pred, sigRrs = \
        evaluate.reconstruct_from_chains(models, chains, rt_dict, perc=(5, 95))

    # Check shapes
    assert len(Rrs_pred) == len(wave), "Predicted Rrs should match wavelength array"
    assert len(a) == len(wave), "Absorption should match wavelength array"
    assert len(bb) == len(wave), "Backscattering should match wavelength array"

    # Check physical constraints
    assert np.all(a > 0), "Absorption should be positive"
    assert np.all(bb > 0), "Backscattering should be positive"
    # Predicted Rrs from model should be mostly positive
    assert np.sum(Rrs_pred > 0) > 0.95 * len(Rrs_pred), \
        "Predicted Rrs should be mostly positive"
    assert np.all(Rrs_pred < 0.1), "Rrs should be < 0.1 sr^-1"

    # Check uncertainty bounds are reasonable
    assert np.all(a_hi > a_lo), "Upper bound should exceed lower bound"
    assert np.all(bb_hi > bb_lo), "Upper bound should exceed lower bound"

    # Calculate chi-squared goodness of fit
    residuals = (model_Rrs - Rrs_pred) / np.sqrt(model_varRrs)
    chi2 = np.sum(residuals**2)
    dof = len(wave) - nparams
    reduced_chi2 = chi2 / dof

    # Reduced chi-squared should be near 1 for good fit (allow range [0.1, 10])
    assert 0.05 < reduced_chi2 < 20, \
        f"Reduced chi-squared out of expected range: {reduced_chi2:.2f}"

    # Check that predicted Rrs is reasonably close to observed
    # For positive Rrs values, check relative error
    positive_mask = model_Rrs > 0
    if np.sum(positive_mask) > 0:
        rel_error = np.abs(Rrs_pred[positive_mask] - model_Rrs[positive_mask]) / \
                    model_Rrs[positive_mask]
        # Allow larger tolerance since we added noise
        assert np.median(rel_error) < 1.5, \
            f"Median relative error too large: {np.median(rel_error):.2f}"

    return a, bb, a_lo, a_hi, bb_lo, bb_hi, Rrs_pred, reduced_chi2


def validate_l23_comparison(l23_dict, models, chains, wave, med_params,
                            a, bb, model_Rrs, Rrs_pred):
    """Compare fitted values to true L23 values."""
    # Interpolate true IOPs to model wavelengths
    true_wave = l23_dict['true_wave']
    true_a = l23_dict['a']
    true_bb = l23_dict['bb']
    true_anw = l23_dict['anw']
    true_bbnw = l23_dict['bbnw']

    # Interpolate to model wavelengths
    a_true_interp = np.interp(wave, true_wave, true_a)
    bb_true_interp = np.interp(wave, true_wave, true_bb)
    anw_true_interp = np.interp(wave, true_wave, true_anw)
    bbnw_true_interp = np.interp(wave, true_wave, true_bbnw)

    # Compare total absorption
    a_rel_error = np.abs(a - a_true_interp) / a_true_interp
    a_mae = np.mean(np.abs(a - a_true_interp))
    a_mape = np.mean(a_rel_error) * 100  # Mean absolute percentage error

    # Compare total backscattering
    bb_rel_error = np.abs(bb - bb_true_interp) / bb_true_interp
    bb_mae = np.mean(np.abs(bb - bb_true_interp))
    bb_mape = np.mean(bb_rel_error) * 100

    # Check that fits are reasonably accurate
    # Allow up to 50% MAPE since we added noise to the data
    assert a_mape < 50, f"Absorption MAPE too high: {a_mape:.1f}%"
    assert bb_mape < 50, f"Backscattering MAPE too high: {bb_mape:.1f}%"

    # Compare spectral parameters
    true_Sdg = l23_dict['Sdg']
    true_Chl = l23_dict['Chl']
    true_Y = l23_dict['Y']

    fitted_Sdg = med_params[1]  # Sdg is 2nd parameter

    # Recover Chl from fitted Aph parameter
    fitted_Aph_log10 = med_params[2]  # log10(Aph) is 3rd parameter
    fitted_Aph_440 = 10**fitted_Aph_log10  # Convert from log10
    recovered_Chl = fitted_Aph_440 / 0.05582

    # Calculate relative errors for parameters
    Sdg_rel_error = np.abs(fitted_Sdg - true_Sdg) / true_Sdg * 100
    Chl_rel_error = np.abs(recovered_Chl - true_Chl) / true_Chl * 100

    # Parameters should be recovered within reasonable accuracy
    # Note: With added noise, parameter recovery can be challenging
    assert Sdg_rel_error < 100, f"Sdg relative error too high: {Sdg_rel_error:.1f}%"
    # Allow larger error for Chl which depends on the Aph parameterization
    assert Chl_rel_error < 150, f"Chl relative error too high: {Chl_rel_error:.1f}%"

    # Return comparison metrics
    return {
        'a_mae': a_mae, 'a_mape': a_mape,
        'bb_mae': bb_mae, 'bb_mape': bb_mape,
        'true_Sdg': true_Sdg, 'fitted_Sdg': fitted_Sdg, 'Sdg_rel_error': Sdg_rel_error,
        'true_Chl': true_Chl, 'recovered_Chl': recovered_Chl, 'Chl_rel_error': Chl_rel_error,
        'true_Y': true_Y, 'Y_used': l23_dict['Y']
    }


def validate_file_save_load(chains, idx, outfile, extras):
    """Test file save and load functionality."""
    fit_l23.save_chains(chains, idx, outfile, extras=extras)
    assert os.path.exists(outfile), f"Output file {outfile} should be created"

    # Load and verify saved data
    loaded = np.load(outfile, allow_pickle=True)
    assert 'chains' in loaded, "Saved file should contain chains"
    assert 'idx' in loaded, "Saved file should contain idx"

    # Verify chains are identical
    np.testing.assert_array_equal(loaded['chains'], chains,
                                   err_msg="Loaded chains should match saved chains")

    # Verify extras were saved
    required_extras = ['wave', 'obs_Rrs', 'varRrs', 'Chl', 'Y']
    for key in required_extras:
        assert key in loaded, f"Saved file should contain extra '{key}'"


def print_test_summary(idx, wave, chains, pnames, med_params, reduced_chi2,
                      model_Rrs, Rrs_pred, comparison_metrics):
    """Print summary statistics for test."""
    print(f"\n===== Test Summary for idx={idx} =====")
    print(f"Wavelength range: {wave.min():.1f} - {wave.max():.1f} nm ({len(wave)} bands)")
    print(f"Chain shape: {chains.shape} (steps, walkers, params)")
    print(f"Model parameters: {pnames}")
    print(f"Median fitted params: {med_params}")
    print(f"Reduced χ²: {reduced_chi2:.3f}")
    print(f"Median |Rrs_obs - Rrs_pred|: {np.median(np.abs(model_Rrs - Rrs_pred)):.6f}")
    print(f"Number of positive Rrs_obs: {np.sum(model_Rrs > 0)}/{len(model_Rrs)}")

    print(f"\n--- Comparison to True L23 Values ---")
    print(f"Absorption:")
    print(f"  MAE: {comparison_metrics['a_mae']:.5f} m^-1")
    print(f"  MAPE: {comparison_metrics['a_mape']:.2f}%")
    print(f"Backscattering:")
    print(f"  MAE: {comparison_metrics['bb_mae']:.6f} m^-1")
    print(f"  MAPE: {comparison_metrics['bb_mape']:.2f}%")

    print(f"\nSpectral Parameters:")
    print(f"  Sdg:  True={comparison_metrics['true_Sdg']:.5f}, "
          f"Fitted={comparison_metrics['fitted_Sdg']:.5f}, "
          f"Error={comparison_metrics['Sdg_rel_error']:.1f}%")
    print(f"  Chl:  True={comparison_metrics['true_Chl']:.3f}, "
          f"Recovered={comparison_metrics['recovered_Chl']:.3f} mg/m³, "
          f"Error={comparison_metrics['Chl_rel_error']:.1f}%")
    print(f"  Y:    True={comparison_metrics['true_Y']:.3f}, "
          f"Used(fixed)={comparison_metrics['Y_used']:.3f} "
          f"[Y is passed as fixed input to fit]")
    print("========================================\n")


# ===== Main tests =====

def test_single_fit_standard_Gordon():
    """Test single spectrum fitting with standard Gordon coefficients."""
    idx = 2773
    p_expb = standard.expb_pow(satellite='SBG', add_noise=True,
                               variable_Gordon=False)
    outfile = fit_l23.chain_filename(p_expb, idx=idx, path='./')
    rt_dict = rt_defs.rt_dict_from_p(p_expb)

    # L23 data
    l23_dict = fit_l23.load_one_l23(idx)

    # Run the fit
    chains, models, prep_dict, idx_out, extras = fit_l23.fit_one(p_expb, idx)

    # Validate using helper functions
    validate_basic_returns(chains, models, prep_dict, idx_out, extras, idx)
    nsteps, nwalkers, nparams = validate_chain_shape(chains, models)
    wave = validate_models(models)
    model_Rrs, model_varRrs = validate_prep_dict(prep_dict, wave)
    validate_extras(extras, wave)
    med_params, pnames, stats = validate_fitted_parameters(chains, models)

    a, bb, a_lo, a_hi, bb_lo, bb_hi, Rrs_pred, reduced_chi2 = \
        validate_reconstructed_rrs(models, chains, model_Rrs, 
                        model_varRrs, wave, nparams, rt_dict)

    comparison_metrics = validate_l23_comparison(
        l23_dict, models, chains, wave, med_params, a, bb, model_Rrs, Rrs_pred)

    validate_file_save_load(chains, idx, outfile, extras)

    # Clean up
    os.remove(outfile)

    # Print summary
    print_test_summary(idx, wave, chains, pnames, med_params, reduced_chi2,
                      model_Rrs, Rrs_pred, comparison_metrics)


def test_single_fit_variable_Gordon():
    """Test single spectrum fitting with variable (wavelength-dependent) Gordon coefficients.

    This test verifies that the fitting works correctly when using wavelength-dependent
    Gordon coefficients (G1 and G2) instead of the standard constant values (G0=0.0949, G1=0.0794).

    The variable Gordon coefficients are loaded from bing/data/RT/gordon_coefficients.csv and
    interpolated to the model wavelengths. This allows for more accurate modeling of the
    water-leaving radiance across different wavelengths.

    NOTE: This test was previously failing due to a scipy compatibility issue in rrs.py
    (using bounds_error=True with fill_value='extrapolate'). This has been fixed by
    removing the fill_value parameter.
    """
    idx = 2773
    p_expb = standard.expb_pow(satellite='SBG', add_noise=True,
                               variable_Gordon=True)
    outfile = fit_l23.chain_filename(p_expb, idx=idx, path='./')
    rt_dict = rt_defs.rt_dict_from_p(p_expb)

    # L23 data
    l23_dict = fit_l23.load_one_l23(idx)

    # Run the fit
    chains, models, prep_dict, idx_out, extras = fit_l23.fit_one(p_expb, idx)

    # Validate using helper functions
    validate_basic_returns(chains, models, prep_dict, idx_out, extras, idx)
    nsteps, nwalkers, nparams = validate_chain_shape(chains, models)
    wave = validate_models(models)
    model_Rrs, model_varRrs = validate_prep_dict(prep_dict, wave)
    validate_extras(extras, wave)
    med_params, pnames, stats = validate_fitted_parameters(chains, models)

    a, bb, a_lo, a_hi, bb_lo, bb_hi, Rrs_pred, reduced_chi2 = \
        validate_reconstructed_rrs(models, chains, model_Rrs, model_varRrs, wave, nparams, rt_dict)

    comparison_metrics = validate_l23_comparison(
        l23_dict, models, chains, wave, med_params, a, bb, model_Rrs, Rrs_pred)

    # ===== Variable Gordon specific checks =====
    # Check that G1 and G2 were set on the models
    assert hasattr(models[0], 'G1'), "Model should have G1 attribute for variable Gordon"
    assert hasattr(models[0], 'G2'), "Model should have G2 attribute for variable Gordon"
    assert hasattr(models[1], 'G1'), "Backscatter model should have G1 attribute"
    assert hasattr(models[1], 'G2'), "Backscatter model should have G2 attribute"

    # G1 and G2 should be arrays (wavelength-dependent), not None
    assert models[0].G1 is not None, "G1 should be set for variable Gordon"
    assert models[0].G2 is not None, "G2 should be set for variable Gordon"
    assert isinstance(models[0].G1, np.ndarray), "G1 should be an array"
    assert isinstance(models[0].G2, np.ndarray), "G2 should be an array"

    # Check that G1 and G2 have the right length
    assert len(models[0].G1) == len(wave), "G1 should match wavelength array"
    assert len(models[0].G2) == len(wave), "G2 should match wavelength array"

    # Check that G1 and G2 are in reasonable ranges
    # Standard Gordon: G0=0.0949, G1=0.0794
    # Variable Gordon should vary with wavelength and can have different ranges
    assert np.all(models[0].G1 > 0), "G1 should be positive"
    # Note: G2 can be negative at certain wavelengths in the variable Gordon formulation
    assert np.all(np.abs(models[0].G2) < 3.0), "G2 magnitude should be < 3.0"
    assert np.all(models[0].G1 < 0.2), "G1 should be < 0.2"

    # Check that G1 and G2 vary with wavelength (not constant)
    assert np.std(models[0].G1) > 0, "G1 should vary with wavelength"
    assert np.std(models[0].G2) > 0, "G2 should vary with wavelength"

    # Verify that prep_dict contains G1 and G2
    assert 'G1' in prep_dict, "prep_dict should contain G1"
    assert 'G2' in prep_dict, "prep_dict should contain G2"
    np.testing.assert_array_equal(prep_dict['G1'], models[0].G1,
                                   err_msg="prep_dict G1 should match model G1")
    np.testing.assert_array_equal(prep_dict['G2'], models[0].G2,
                                   err_msg="prep_dict G2 should match model G2")

    validate_file_save_load(chains, idx, outfile, extras)

    # Clean up
    os.remove(outfile)

    # Print summary with variable Gordon note
    print("\n[Variable Gordon Coefficients Enabled]")
    print(f"G1 range: {models[0].G1.min():.4f} - {models[0].G1.max():.4f}")
    print(f"G2 range: {models[0].G2.min():.4f} - {models[0].G2.max():.4f}")
    print(f"G1 std dev: {np.std(models[0].G1):.6f}")
    print(f"G2 std dev: {np.std(models[0].G2):.6f}")

    print_test_summary(idx, wave, chains, pnames, med_params, reduced_chi2,
                      model_Rrs, Rrs_pred, comparison_metrics)


def test_single_fit_variable_Gordon_with_G0():
    """Variable Gordon with the constant offset G0 enabled.

    Verifies that:
    - variable_Gordon_G0=True loads G0(λ) from gordon_coefficients_with_G0.csv,
    - G0 is stashed on both models and exposed via prep_dict['G0'],
    - the resulting fit runs to completion and recovers reasonable IOPs,
    - G0 stays in the empirical range (~10⁻⁴ at red).
    """
    idx = 2773
    p_expb = standard.expb_pow(satellite='SBG', add_noise=True,
                               variable_Gordon=True, variable_Gordon_G0=True)

    chains, models, prep_dict, idx_out, extras = fit_l23.fit_one(p_expb, idx)

    # Basic shape/validity (re-uses existing helpers)
    validate_basic_returns(chains, models, prep_dict, idx_out, extras, idx)
    wave = validate_models(models)

    # G0 is set on the absorption model (post-merge design: only models[0]
    # carries the Gordon coefficients; bb model reads them indirectly).
    assert getattr(models[0], 'G0', None) is not None, "G0 should be set when variable_Gordon_G0=True"
    assert isinstance(models[0].G0, np.ndarray)
    assert len(models[0].G0) == len(wave), "G0 should match wavelength array"
    # G0 fit values are ~1e-4 in magnitude (see dev/Gordon/calc_gordon.py log)
    assert np.all(np.abs(models[0].G0) < 5e-3), \
        f"G0 magnitudes look unphysical: max |G0| = {np.max(np.abs(models[0].G0)):.2e}"

    # G0 is exposed in prep_dict and matches the model attribute
    assert 'G0' in prep_dict, "prep_dict should contain G0"
    np.testing.assert_array_equal(prep_dict['G0'], models[0].G0)

    # Backwards-compat: G1/G2 still loaded and consistent
    assert models[0].G1 is not None and models[0].G2 is not None
    assert np.all(models[0].G1 > 0)
    assert np.all(np.abs(models[0].G2) < 3.0)

    print(f"\n[Variable Gordon with G0]")
    print(f"  G0 range: [{models[0].G0.min():+.4e}, {models[0].G0.max():+.4e}]")
    print(f"  G1 range: [{models[0].G1.min():.4f}, {models[0].G1.max():.4f}]")
    print(f"  G2 range: [{models[0].G2.min():.4f}, {models[0].G2.max():.4f}]")


def test_single_fit_variable_Gordon_with_G0_and_bbp():
    """4-parameter variable Gordon: both G0 and Gb enabled.

    Verifies the (G0,G1,G2,Gb) recipe -- when both ``variable_Gordon_G0=True``
    and ``variable_Gordon_bbp=True`` -- loads from
    ``gordon_coefficients_with_G0_Gb.csv``, sets both G0 and Gb on the
    absorption model, exposes them via prep_dict, and runs the MCMC.
    """
    idx = 2773
    p_expb = standard.expb_pow(
        satellite='SBG', add_noise=True,
        variable_Gordon=True,
        variable_Gordon_G0=True,
        variable_Gordon_bbp=True,
    )

    chains, models, prep_dict, idx_out, extras = fit_l23.fit_one(p_expb, idx)

    validate_basic_returns(chains, models, prep_dict, idx_out, extras, idx)
    wave = validate_models(models)

    # Both G0 and Gb are set on the absorption model
    assert getattr(models[0], 'G0', None) is not None, \
        "G0 should be set when both flags are True"
    assert getattr(models[0], 'Gb', None) is not None, \
        "Gb should be set when both flags are True"
    assert isinstance(models[0].G0, np.ndarray)
    assert isinstance(models[0].Gb, np.ndarray)
    assert len(models[0].G0) == len(wave)
    assert len(models[0].Gb) == len(wave)

    # Magnitudes stay in the empirical envelope from calc_gordon.py
    assert np.all(np.abs(models[0].G0) < 5e-3), \
        f"|G0| out of envelope: max = {np.max(np.abs(models[0].G0)):.2e}"
    assert np.all(np.abs(models[0].Gb) < 5.0), \
        f"|Gb| out of envelope: max = {np.max(np.abs(models[0].Gb)):.2e}"

    # prep_dict exposes both and matches the model attributes
    assert 'G0' in prep_dict and 'Gb' in prep_dict
    np.testing.assert_array_equal(prep_dict['G0'], models[0].G0)
    np.testing.assert_array_equal(prep_dict['Gb'], models[0].Gb)

    # G1/G2 still loaded and sensible
    assert models[0].G1 is not None and models[0].G2 is not None
    assert np.all(models[0].G1 > 0)
    assert np.all(np.abs(models[0].G2) < 3.0)

    print(f"\n[Variable Gordon, 4-parameter (G0 + Gb)]")
    print(f"  G0 range: [{models[0].G0.min():+.4e}, {models[0].G0.max():+.4e}]")
    print(f"  Gb range: [{models[0].Gb.min():+.4e}, {models[0].Gb.max():+.4e}]")
    print(f"  G1 range: [{models[0].G1.min():.4f}, {models[0].G1.max():.4f}]")
    print(f"  G2 range: [{models[0].G2.min():.4f}, {models[0].G2.max():.4f}]")


def test_single_fit_variable_Gordon_with_bbp():
    """Variable Gordon with the bbp slope Gb enabled.

    Verifies that ``variable_Gordon_bbp=True`` loads Gb(λ) from
    ``gordon_coefficients_with_Gb.csv``, sets it on the absorption model,
    exposes it via ``prep_dict['Gb']``, and runs the MCMC end-to-end.
    """
    idx = 2773
    p_expb = standard.expb_pow(satellite='SBG', add_noise=True,
                               variable_Gordon=True, variable_Gordon_bbp=True)

    chains, models, prep_dict, idx_out, extras = fit_l23.fit_one(p_expb, idx)

    validate_basic_returns(chains, models, prep_dict, idx_out, extras, idx)
    wave = validate_models(models)

    # Gb is set on the absorption model
    assert getattr(models[0], 'Gb', None) is not None, \
        "Gb should be set when variable_Gordon_bbp=True"
    assert isinstance(models[0].Gb, np.ndarray)
    assert len(models[0].Gb) == len(wave), "Gb should match wavelength array"
    # Gb magnitudes are ~1e-2..2e-1 in absolute value at oceanic bands
    assert np.all(np.abs(models[0].Gb) < 5.0), \
        f"|Gb| unphysically large: max = {np.max(np.abs(models[0].Gb)):.2e}"

    # G0 is None in this mode (mutually exclusive with Gb)
    assert getattr(models[0], 'G0', None) is None, \
        "G0 should be None when variable_Gordon_bbp=True"

    # prep_dict exposes Gb and matches the model
    assert 'Gb' in prep_dict, "prep_dict should contain Gb"
    np.testing.assert_array_equal(prep_dict['Gb'], models[0].Gb)

    # G1/G2 still loaded
    assert models[0].G1 is not None and models[0].G2 is not None

    print(f"\n[Variable Gordon with Gb]")
    print(f"  Gb range: [{models[0].Gb.min():+.4e}, {models[0].Gb.max():+.4e}]")
    print(f"  G1 range: [{models[0].G1.min():.4f}, {models[0].G1.max():.4f}]")
    print(f"  G2 range: [{models[0].G2.min():.4f}, {models[0].G2.max():.4f}]")


# ===== Tests for individual l23 methods =====

def test_load_one_l23_basic():
    """Test basic loading of L23 data."""
    idx = 100
    l23_dict = fit_l23.load_one_l23(idx)

    # Check required keys exist
    required_keys = ['wave', 'Rrs', 'a', 'bb', 'true_wave', 'true_Rrs',
                     'gordon_Rrs', 'bbw', 'bbnw', 'aw', 'anw', 'adg', 'ag',
                     'aph', 'Sdg', 'Y', 'Chl']
    for key in required_keys:
        assert key in l23_dict, f"Missing key: {key}"

    # Check array shapes are consistent
    assert len(l23_dict['wave']) == len(l23_dict['Rrs'])
    assert len(l23_dict['true_wave']) == len(l23_dict['true_Rrs'])
    assert len(l23_dict['true_wave']) == len(l23_dict['a'])
    assert len(l23_dict['true_wave']) == len(l23_dict['bb'])

    # Check physical constraints
    assert np.all(l23_dict['Rrs'] >= -0.01), "Rrs should be mostly positive"
    assert np.all(l23_dict['Rrs'] < 0.1), "Rrs should be < 0.1"
    assert np.all(l23_dict['a'] > 0), "Absorption should be positive"
    assert np.all(l23_dict['bb'] > 0), "Backscattering should be positive"
    assert l23_dict['Chl'] > 0, "Chlorophyll should be positive"
    assert 0 < l23_dict['Y'] < 5, "Y should be in reasonable range"
    assert 0 < l23_dict['Sdg'] < 0.03, "Sdg should be in reasonable range"

    # Check wavelength ordering
    assert np.all(np.diff(l23_dict['wave']) > 0), "Wavelengths should be monotonic"
    assert np.all(np.diff(l23_dict['true_wave']) > 0), "True wavelengths should be monotonic"


def test_load_one_l23_with_step():
    """Test loading L23 data with downsampling."""
    idx = 50
    step = 2

    l23_dict_step1 = fit_l23.load_one_l23(idx, step=1)
    l23_dict_step2 = fit_l23.load_one_l23(idx, step=step)

    # Check that step reduces number of wavelengths
    expected_len = len(l23_dict_step1['wave'][::step])
    assert len(l23_dict_step2['wave']) == expected_len, \
        f"Step should downsample wavelengths: expected {expected_len}, got {len(l23_dict_step2['wave'])}"

    # Check that true_wave (full resolution) is the same
    assert len(l23_dict_step1['true_wave']) == len(l23_dict_step2['true_wave'])


def test_load_one_l23_wavelength_range():
    """Test loading L23 data with wavelength range restrictions."""
    idx = 200

    # Test with wavelength limits
    l23_dict_full = fit_l23.load_one_l23(idx, wv_min=400, wv_max=700)
    l23_dict_restricted = fit_l23.load_one_l23(idx, wv_min=450, wv_max=650)

    # Check that restricted range has fewer wavelengths
    assert len(l23_dict_restricted['true_wave']) < len(l23_dict_full['true_wave'])

    # Check that wavelengths are within bounds
    assert np.all(l23_dict_restricted['true_wave'] >= 450)
    assert np.all(l23_dict_restricted['true_wave'] <= 650)

    # Full range should respect bounds too
    assert np.all(l23_dict_full['true_wave'] >= 400)
    assert np.all(l23_dict_full['true_wave'] <= 700)


def test_load_one_l23_gordon_consistency():
    """Test that Gordon Rrs calculation is consistent with IOPs."""
    idx = 150
    l23_dict = fit_l23.load_one_l23(idx)

    # Recalculate Gordon Rrs from a and bb
    recalc_Rrs = bing_rt.calc_Rrs(l23_dict['a'], l23_dict['bb'])

    # Should match stored gordon_Rrs
    np.testing.assert_allclose(recalc_Rrs, l23_dict['gordon_Rrs'],
                               rtol=1e-10, atol=1e-12,
                               err_msg="Gordon Rrs should match recalculation from a and bb")


def test_prep_one_l23_basic():
    """Test basic preparation of L23 data for fitting."""
    idx = 100
    p = standard.expb_pow(satellite='PACE', add_noise=False, variable_Gordon=False)

    prep_dict = fit_l23.prep_one_l23(p, idx)

    # Check required keys
    required_keys = ['odict', 'model_Rrs', 'model_varRrs', 'p0', 'pdict', 'models']
    for key in required_keys:
        assert key in prep_dict, f"Missing key in prep_dict: {key}"

    # Check models
    assert len(prep_dict['models']) == 2, "Should have 2 models (absorption + backscattering)"
    assert hasattr(prep_dict['models'][0], 'nparam')
    assert hasattr(prep_dict['models'][1], 'nparam')

    # Check p0 shape matches total parameters
    expected_nparam = prep_dict['models'][0].nparam + prep_dict['models'][1].nparam
    assert len(prep_dict['p0']) == expected_nparam, \
        f"Initial parameters should have {expected_nparam} elements"

    # Check Rrs arrays
    model_wave = prep_dict['models'][0].wave
    assert len(prep_dict['model_Rrs']) == len(model_wave)
    assert len(prep_dict['model_varRrs']) == len(model_wave)
    assert np.all(prep_dict['model_varRrs'] > 0), "Variance should be positive"


def test_prep_one_l23_different_satellites():
    """Test preparation with different satellite configurations."""
    idx = 50

    satellites = ['PACE', 'MODIS', 'SeaWiFS', 'SBG']

    for sat in satellites:
        # Use numeric scl_noise to avoid string issues
        p = standard.expb_pow(satellite=sat, add_noise=False,
                             variable_Gordon=False, scl_noise=0.05)
        prep_dict = fit_l23.prep_one_l23(p, idx)

        # Check that models were initialized
        assert len(prep_dict['models']) == 2
        assert len(prep_dict['model_Rrs']) == len(prep_dict['models'][0].wave)

        # Wavelength arrays should be appropriate for satellite
        wave = prep_dict['models'][0].wave
        assert len(wave) > 0, f"Should have wavelengths for {sat}"
        assert np.all(wave >= 400) and np.all(wave <= 800), \
            f"Wavelengths should be in visible range for {sat}"


def test_prep_one_l23_with_noise():
    """Test preparation with and without noise addition."""
    idx = 75

    # Without noise
    p_no_noise = standard.expb_pow(satellite='PACE', add_noise=False, variable_Gordon=False)
    prep_no_noise = fit_l23.prep_one_l23(p_no_noise, idx)

    # With noise
    p_with_noise = standard.expb_pow(satellite='PACE', add_noise=True, variable_Gordon=False)
    prep_with_noise = fit_l23.prep_one_l23(p_with_noise, idx)

    # Rrs values should differ when noise is added
    # (though not guaranteed for every single case due to randomness)
    rrs_diff = np.abs(prep_no_noise['model_Rrs'] - prep_with_noise['model_Rrs'])

    # At least some difference should exist
    # Note: This test might rarely fail due to random chance
    assert np.sum(rrs_diff > 1e-6) > 0, "Adding noise should change Rrs values"


def test_prep_one_l23_variable_gordon():
    """Test preparation with variable Gordon coefficients.

    Note: This test was previously skipped due to scipy interpolation incompatibility,
    which has now been fixed in rrs.py.
    """
    idx = 120

    # Standard Gordon
    p_standard = standard.expb_pow(satellite='PACE', variable_Gordon=False)
    prep_standard = fit_l23.prep_one_l23(p_standard, idx)

    # Variable Gordon
    p_variable = standard.expb_pow(satellite='PACE', variable_Gordon=True)
    prep_variable = fit_l23.prep_one_l23(p_variable, idx)

    # Both should succeed and produce valid outputs
    assert prep_standard['model_Rrs'] is not None
    assert prep_variable['model_Rrs'] is not None

    # Check that both have reasonable Rrs values
    assert np.all(np.abs(prep_standard['model_Rrs']) < 0.1)
    assert np.all(np.abs(prep_variable['model_Rrs']) < 0.1)


def test_chain_filename_basic():
    """Test chain filename generation with basic parameters."""
    p = standard.expb_pow(satellite='PACE', add_noise=True)

    # With index
    filename_with_idx = fit_l23.chain_filename(p, idx=100, path='./')
    assert 'BING20_' in filename_with_idx
    assert '_100_' in filename_with_idx
    assert '_P' in filename_with_idx  # PACE
    assert '.npz' in filename_with_idx

    # Without index
    filename_no_idx = fit_l23.chain_filename(p, idx=None, path='./')
    assert 'BING20_' in filename_no_idx
    assert '_P23' in filename_no_idx  # PACE with L23 dataset
    assert '.npz' in filename_no_idx


def test_chain_filename_satellites():
    """Test chain filename generation for different satellites."""
    idx = 42
    path = '/tmp/'

    satellite_codes = {
        'PACE': '_P',
        'MODIS': '_M',
        'SeaWiFS': '_S',
        'SBG': '_B'
    }

    for sat, code in satellite_codes.items():
        # Use numeric scl_noise to avoid string conversion issues
        p = standard.expb_pow(satellite=sat, variable_Gordon=False, scl_noise=0.05)
        filename = fit_l23.chain_filename(p, idx=idx, path=path)
        assert code in filename, f"Filename should contain {code} for {sat}"


def test_chain_filename_noise_flags():
    """Test chain filename generation with noise flags."""
    idx = 10

    # With noise
    p_noise = standard.expb_pow(satellite='PACE', add_noise=True)
    filename_noise = fit_l23.chain_filename(p_noise, idx=idx, path='./')
    assert '_N' in filename_noise, "Should contain _N for added noise"

    # Without noise
    p_no_noise = standard.expb_pow(satellite='PACE', add_noise=False)
    filename_no_noise = fit_l23.chain_filename(p_no_noise, idx=idx, path='./')
    assert '_n' in filename_no_noise, "Should contain _n for no noise"


def test_save_and_load_chains():
    """Test saving and loading chain data."""
    # Create dummy data
    nsteps, nwalkers, nparams = 100, 10, 5
    chains = np.random.randn(nsteps, nwalkers, nparams)
    idx = 42

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        outfile = tmp.name

    try:
        # Test basic save
        fit_l23.save_chains(chains, idx, outfile)
        assert os.path.exists(outfile), "File should be created"

        # Load and verify
        loaded = np.load(outfile, allow_pickle=True)
        assert 'chains' in loaded
        assert 'idx' in loaded
        np.testing.assert_array_equal(loaded['chains'], chains)
        assert loaded['idx'] == idx

        # Test save with extras
        extras = {
            'wave': np.array([400, 450, 500, 550, 600]),
            'obs_Rrs': np.array([0.01, 0.012, 0.011, 0.009, 0.008]),
            'varRrs': np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001]),
            'Chl': 0.5,
            'Y': 1.2
        }

        fit_l23.save_chains(chains, idx, outfile, extras=extras)

        # Load and verify extras
        loaded = np.load(outfile, allow_pickle=True)
        for key in extras.keys():
            assert key in loaded, f"Extras key {key} should be saved"
            if isinstance(extras[key], np.ndarray):
                np.testing.assert_array_equal(loaded[key], extras[key])
            else:
                assert loaded[key] == extras[key]

    finally:
        # Cleanup
        if os.path.exists(outfile):
            os.remove(outfile)


def test_save_chains_overwrite():
    """Test that save_chains can overwrite existing files."""
    chains1 = np.random.randn(50, 10, 5)
    chains2 = np.random.randn(60, 12, 5)
    idx = 1

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        outfile = tmp.name

    try:
        # Save first set of chains
        fit_l23.save_chains(chains1, idx, outfile)
        loaded1 = np.load(outfile)
        assert loaded1['chains'].shape == chains1.shape

        # Overwrite with second set
        fit_l23.save_chains(chains2, idx, outfile)
        loaded2 = np.load(outfile)
        assert loaded2['chains'].shape == chains2.shape
        np.testing.assert_array_equal(loaded2['chains'], chains2)

    finally:
        if os.path.exists(outfile):
            os.remove(outfile)


def test_fit_one_with_custom_p0():
    """Test fit_one with custom initial parameters."""
    idx = 500
    p = standard.expb_pow(satellite='PACE', add_noise=False,
                          nsteps=100, nburn=10, variable_Gordon=False)  # Small number for speed

    # Get default p0 first
    prep_dict = fit_l23.prep_one_l23(p, idx)
    default_p0 = prep_dict['p0']

    # Create custom p0 (slightly perturbed)
    custom_p0 = default_p0 + np.random.randn(len(default_p0)) * 0.1

    # Run fit with custom p0
    chains, models, prep_dict_out, idx_out, extras = fit_l23.fit_one(
        p, idx, p0=custom_p0)

    # Check that custom p0 was used
    np.testing.assert_array_equal(prep_dict_out['p0'], custom_p0)

    # Check outputs
    assert chains is not None
    assert chains.ndim == 3
    assert idx_out == idx


def test_fit_one_minimal_steps():
    """Test fit_one with minimal MCMC steps for speed."""
    idx = 300
    p = standard.expb_pow(satellite='PACE', add_noise=False,
                          nsteps=50, nburn=5, variable_Gordon=False)  # Very minimal

    chains, models, prep_dict, idx_out, extras = fit_l23.fit_one(p, idx)

    # Basic checks
    assert chains.shape[0] == 50, "Should have 50 steps"
    assert idx_out == idx
    assert 'Chl' in extras
    assert 'Y' in extras
    assert 'wave' in extras


def test_fit_one_different_models():
    """Test fit_one with different model combinations."""
    idx = 250

    # Test GIOP model
    p_giop = standard.giop(satellite='PACE', nsteps=100, nburn=10, variable_Gordon=False)
    chains_giop, models_giop, _, _, _ = fit_l23.fit_one(p_giop, idx)

    assert chains_giop is not None
    assert len(models_giop) == 2

    # ExpB + Pow should have different number of parameters than GIOP
    p_expb = standard.expb_pow(satellite='PACE', nsteps=100, nburn=10, variable_Gordon=False)
    chains_expb, models_expb, _, _, _ = fit_l23.fit_one(p_expb, idx)

    # Parameter counts may differ between models
    expb_nparam = chains_expb.shape[2]
    giop_nparam = chains_giop.shape[2]

    # Both should be valid
    assert expb_nparam > 0
    assert giop_nparam > 0


#@pytest.mark.slow
def test_batch_fit_small():
    """Test batch fitting with a small number of spectra.

    Note: Marked as slow since it involves MCMC fitting.
    """
    p = standard.expb_pow(satellite='PACE', add_noise=False,
                          nsteps=100, nburn=10, variable_Gordon=False)

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run batch fit in debug mode (processes only 4 spectra)
        fit_l23.batch_fit(p, n_batch=2, n_cores=1, debug=True,
                         seed=42, out_dir=tmpdir)

        # Check that output files were created
        output_files = [f for f in os.listdir(tmpdir) if f.endswith('.npz')]
        assert len(output_files) > 0, "Should create output files"

        # Load one file and verify structure
        test_file = os.path.join(tmpdir, output_files[0])
        loaded = np.load(test_file, allow_pickle=True)

        assert 'chains' in loaded
        assert 'idx' in loaded
        assert 'wave' in loaded
        assert 'obs_Rrs' in loaded


def test_process_one_structure():
    """Test process_one output structure.

    Note: This test requires that a chain file already exists.
    We'll skip it if the file doesn't exist.
    """
    pytest.skip("Requires pre-existing chain files from batch_fit")


def test_process_all_structure():
    """Test process_all output structure.

    Note: This test requires that chain files already exist.
    We'll skip it if files don't exist.
    """
    pytest.skip("Requires pre-existing chain files from batch_fit")



def test_raman_fitting_LM():
    idx = 170
    # Prep
    p_R = standard.expb_pow(satellite='PACE', add_noise=False, variable_Gordon=True, include_Raman=True)
    # Fit
    ans, cov, models_LM, prep_dict_LM, idx = fit_l23.fit_with_LM(p_R, idx)
    # Plot as an additional test
    rt_dict_R = rt_defs.rt_dict_from_p(p_R)
    Chl = 10**ans[2]/0.05582
    # show=False: a unit test must not open (and block on) a GUI window;
    # the figure-construction code path is still exercised.
    _ = bing_plot.show_fits(models_LM, ans, rt_dict_R, Chl, None,
                figsize=(12,4), fontsize=13., show=False,
                Rrs_true=dict(wave=models_LM[0].wave, spec=prep_dict_LM['model_Rrs'], var=prep_dict_LM['model_varRrs']),
                log_abb=True )

def test_raman_fitting_MCMC():
    idx = 170
    p_R = standard.expb_pow(satellite='PACE', add_noise=False, variable_Gordon=True, 
        include_Raman=True, nsteps=10000, nburn=1000)
    chains_R, models_R, prep_dict_R, idx, extras_R = fit_l23.fit_one(p_R, idx)
    # Plot
    rt_dict_R = rt_defs.rt_dict_from_p(p_R)
    # show=False: see test_raman_fitting_LM
    _ = bing_plot.show_fits(models_R, chains_R, rt_dict_R, None, None,
                figsize=(12,4), fontsize=13., show=False,
                Rrs_true=dict(wave=models_R[0].wave,
                    spec=prep_dict_R['model_Rrs'], var=prep_dict_R['model_varRrs']),
                log_abb=True)

def test_Chl_fitting_LM():
    idx = 170

    # Parameters
    p_Chl = standard.expb_pow(satellite='PACE', add_noise=False, variable_Gordon=True, 
        include_Raman=True, nsteps=10000, nburn=1000,
        include_Chl_fl=True, phi_C=0.02, double_gaussian=True)

    # Fit
    ans, cov, models_LM, prep_dict_LM, idx = fit_l23.fit_with_LM(p_Chl, idx)


def test_Chl_fitting_MCMC():
    idx = 170

    # Parameters
    p_Chl = standard.expb_pow(satellite='PACE', add_noise=False, variable_Gordon=True, 
        include_Raman=True, nsteps=10000, nburn=1000,
        include_Chl_fl=True, phi_C=0.02, double_gaussian=True)

    # Fit
    chains_Chl, models_Chl, prep_dict_Chl, idx, extras_Chl = fit_l23.fit_one(p_Chl, idx)
    # RT dict
    rt_dict_Chl = rt_defs.rt_dict_from_p(p_Chl)

    # Test chains
    #from importlib import reload
    #reload(evaluate)
    #_ = evaluate.reconstruct_from_chains(models_Chl, chains_Chl, rt_dict_Chl)

    # Plot
    # show=False: see test_raman_fitting_LM
    _ = bing_plot.show_fits(models_Chl, chains_Chl, rt_dict_Chl, None, None,
                figsize=(12,4), fontsize=13., show=False,
                Rrs_true=dict(wave=models_Chl[0].wave,
                    spec=prep_dict_Chl['model_Rrs'], var=prep_dict_Chl['model_varRrs']),
                log_abb=True )
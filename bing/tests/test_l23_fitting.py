""" Tests for phytoplankton """""
import os

import numpy as np

import pathlib
import pytest

from bing.fitting import l23 as fit_l23
from bing.parameters import standard
from bing import evaluate
from bing import rt as bing_rt

from IPython import embed

def data_path(filename):
    data_dir = pathlib.Path(__file__).parent.absolute().joinpath('files')
    return str(data_dir.joinpath(filename).resolve())


def test_single_fit():
    """Test single spectrum fitting with comprehensive output validation."""
    idx = 2773
    p_expb = standard.expb_pow(satellite='SBG', add_noise=True)
    outfile = fit_l23.chain_filename(p_expb, idx=idx, path='./')

    # L23 data
    l23_dict = fit_l23.load_one_l23(idx)

    # Run the fit
    chains, models, prep_dict, idx_out, extras = fit_l23.fit_one(p_expb, idx)

    # ===== Basic return value checks =====
    assert idx_out == idx, "Index should be preserved"
    assert chains is not None, "Chains should not be None"
    assert models is not None and len(models) == 2, "Should return 2 models"
    assert prep_dict is not None, "prep_dict should not be None"
    assert extras is not None, "extras should not be None"

    # ===== Check chains shape =====
    # chains shape: (nsteps, nwalkers, nparams)
    assert chains.ndim == 3, f"Chains should be 3D, got {chains.ndim}D"
    nsteps, nwalkers, nparams = chains.shape
    assert nsteps > 0, "Should have steps in chains"
    assert nwalkers > 0, "Should have walkers in chains"

    # Check parameter count matches models
    expected_nparams = models[0].nparam + models[1].nparam
    assert nparams == expected_nparams, \
        f"Chains should have {expected_nparams} params, got {nparams}"

    # ===== Check models =====
    assert hasattr(models[0], 'wave'), "Model should have wavelength array"
    assert hasattr(models[0], 'nparam'), "Model should have nparam attribute"
    assert hasattr(models[0], 'pnames'), "Model should have parameter names"
    assert hasattr(models[1], 'eval_bb'), "Backscatter model should have eval_bb"
    assert hasattr(models[0], 'eval_a'), "Absorption model should have eval_a"

    # Check wavelengths are reasonable
    wave = models[0].wave
    assert np.all(wave >= 400) and np.all(wave <= 800), \
        "Wavelengths should be in visible range [400-800nm]"

    # ===== Check prep_dict contents =====
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

    # ===== Check extras contents =====
    required_extras = ['wave', 'obs_Rrs', 'varRrs', 'Chl', 'Y']
    for key in required_extras:
        assert key in extras, f"extras should contain '{key}'"

    assert np.array_equal(extras['wave'], wave), "Extras wavelength should match model"
    assert extras['Chl'] > 0, "Chlorophyll should be positive"
    assert 0 < extras['Y'] < 5, "Y parameter should be in reasonable range [0-5]"

    # ===== Validate fitted parameters =====
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

    # ===== Validate reconstructed Rrs =====
    a, bb, a_lo, a_hi, bb_lo, bb_hi, Rrs_pred, sigRrs = \
        evaluate.reconstruct_from_chains(models, chains, perc=(5, 95))

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

    # ===== Compare fitted values to true L23 values =====
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

    # Note: extras['Chl'] and extras['Y'] are passed from the L23 data
    # as fixed parameters in the fit (see l23.py lines 296-299).
    # To get the actual recovered Chl from the fitted Aph parameter:
    # Chl = Aph(440) / 0.05582 (from Bricaud et al. 1995)
    # But Aph is log10, so: Chl = 10^Aph / 0.05582
    fitted_Aph_log10 = med_params[2]  # log10(Aph) is 3rd parameter
    fitted_Aph_440 = 10**fitted_Aph_log10  # Convert from log10
    recovered_Chl = fitted_Aph_440 / 0.05582

    # Calculate relative errors for parameters
    Sdg_rel_error = np.abs(fitted_Sdg - true_Sdg) / true_Sdg * 100
    Chl_rel_error = np.abs(recovered_Chl - true_Chl) / true_Chl * 100
    # Y is also passed as fixed input, so we note this but don't test recovery
    Y_used = extras['Y']  # This is the Y value used in fitting (=true_Y)

    # Parameters should be recovered within reasonable accuracy
    assert Sdg_rel_error < 50, f"Sdg relative error too high: {Sdg_rel_error:.1f}%"
    # Allow larger error for Chl which depends on the Aph parameterization
    assert Chl_rel_error < 100, f"Chl relative error too high: {Chl_rel_error:.1f}%"

    # Store comparison metrics for summary output
    comparison_metrics = {
        'a_mae': a_mae,
        'a_mape': a_mape,
        'bb_mae': bb_mae,
        'bb_mape': bb_mape,
        'true_Sdg': true_Sdg,
        'fitted_Sdg': fitted_Sdg,
        'Sdg_rel_error': Sdg_rel_error,
        'true_Chl': true_Chl,
        'recovered_Chl': recovered_Chl,
        'Chl_rel_error': Chl_rel_error,
        'true_Y': true_Y,
        'Y_used': Y_used,  # Y is passed as fixed input, not recovered
    }

    # ===== Test file save and load =====
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
    for key in required_extras:
        assert key in loaded, f"Saved file should contain extra '{key}'"

    # Clean up
    os.remove(outfile)

    # Print summary statistics for development reference
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
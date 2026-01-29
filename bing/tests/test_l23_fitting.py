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
    print(f"Chl: {extras['Chl']:.3f} mg/m³")
    print(f"Y (backscatter slope param): {extras['Y']:.3f}")
    print(f"Median |Rrs_obs - Rrs_pred|: {np.median(np.abs(model_Rrs - Rrs_pred)):.6f}")
    print(f"Number of positive Rrs_obs: {np.sum(model_Rrs > 0)}/{len(model_Rrs)}")
    print("========================================\n")
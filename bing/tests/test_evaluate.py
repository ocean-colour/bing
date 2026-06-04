"""Tests for bing.evaluate.

These tests focus on the post-fitting analysis layer in
``bing/evaluate.py``. They are structured around an end-to-end L23 fit
(see ``test_l23_fitting.py`` for the template) so that the refactored
``reconstruct_from_chains`` -- which now delegates the forward model to
``calc_Rrs_from_models`` -- is exercised the same way callers in
``papers/`` and the notebooks invoke it.

Coverage:
    * ``thin_burn_chains`` -- shape / burn-in semantics on synthetic chains.
    * ``calc_stats`` -- median / percentile recovery on Gaussian chains.
    * ``calc_Rrs_from_models`` -- forward-model sanity on a 1-D param vector.
    * ``reconstruct_from_chains`` -- end-to-end through an L23 MCMC fit, for
      standard Gordon, variable Gordon, and Raman-enabled rt_dicts.
    * The key consistency check that motivates the refactor: the median /
      std of Rrs returned by ``reconstruct_from_chains`` matches what
      ``calc_Rrs_from_models`` produces directly from the same chains.
"""
import numpy as np

import pytest

from bing.fitting import l23 as fit_l23
from bing.parameters import standard
from bing import evaluate
from bing.rt import defs as rt_defs


# ===== Synthetic / unit tests =====

def test_thin_burn_chains_shape():
    """thin_burn_chains drops burn-in, thins, and flattens walkers."""
    nsteps, nwalkers, nparam = 200, 8, 3
    rng = np.random.default_rng(0)
    chains = rng.standard_normal((nsteps, nwalkers, nparam))

    out = evaluate.thin_burn_chains(chains, burn=50, thin=2)

    # Expected: ((200 - 50) // 2) * 8 = 75 * 8 = 600 samples
    expected_samples = ((nsteps - 50) // 2) * nwalkers
    assert out.shape == (expected_samples, nparam)

    # Values come from the post-burn slice
    expected_first = chains[50, 0, :]
    np.testing.assert_array_equal(out[0], expected_first)


def test_thin_burn_chains_default_burn():
    """Default burn=7000 must produce a non-empty slice on long chains."""
    nsteps, nwalkers, nparam = 10000, 4, 5
    chains = np.zeros((nsteps, nwalkers, nparam))
    out = evaluate.thin_burn_chains(chains)
    # (10000 - 7000) * 4 = 12000 samples
    assert out.shape == (12000, nparam)


def test_calc_stats_recovers_gaussian():
    """calc_stats should recover the median / percentiles of a known dist."""
    nsteps, nwalkers, nparam = 9000, 4, 2
    rng = np.random.default_rng(42)
    mu = np.array([1.0, -2.0])
    sigma = np.array([0.5, 1.5])
    # Constant-in-time chains so burn-in is irrelevant.
    samples = rng.standard_normal((nsteps, nwalkers, nparam)) * sigma + mu

    stats = evaluate.calc_stats(samples, names=['m1', 'm2'], perc=(16, 84))

    assert stats['names'] == ['m1', 'm2']
    np.testing.assert_allclose(stats['med'], mu, atol=0.05)
    # ~1-sigma percentiles
    np.testing.assert_allclose(stats['p16'], mu - sigma, atol=0.1)
    np.testing.assert_allclose(stats['p84'], mu + sigma, atol=0.1)


# ===== L23 end-to-end fixtures + tests =====

# Use chains long enough to survive the default thin_burn_chains(burn=7000).
# 8000 steps with ~16 walkers leaves ~16k post-burn samples, which is more
# than enough for the refactor checks (which only test internal consistency,
# not posterior shape).
_NSTEPS = 8000
_NBURN = 100


def _run_l23_fit(**p_kwargs):
    """Run a small L23 fit; returns (chains, models, prep_dict, rt_dict)."""
    p = standard.expb_pow(satellite='PACE', add_noise=False,
                          nsteps=_NSTEPS, nburn=_NBURN, **p_kwargs)
    chains, models, prep_dict, _, _ = fit_l23.fit_one(p, idx=170)
    rt_dict = rt_defs.rt_dict_from_p(p)
    return chains, models, prep_dict, rt_dict


@pytest.fixture(scope="module")
def l23_fit_standard():
    """One L23 fit with standard (constant) Gordon coefficients."""
    return _run_l23_fit(variable_Gordon=False)


@pytest.fixture(scope="module")
def l23_fit_variable():
    """One L23 fit with wavelength-dependent Gordon coefficients."""
    return _run_l23_fit(variable_Gordon=True)


@pytest.fixture(scope="module")
def l23_fit_raman():
    """One L23 fit with variable Gordon + Raman correction."""
    return _run_l23_fit(variable_Gordon=True, include_Raman=True)


def _validate_reconstruction(a_med, bb_med, a_lo, a_hi, bb_lo, bb_hi,
                             Rrs_pred, sigRrs, wave):
    """Shared physical / shape checks on reconstruct_from_chains output."""
    nwave = len(wave)
    assert a_med.shape == (nwave,)
    assert bb_med.shape == (nwave,)
    assert a_lo.shape == (nwave,)
    assert a_hi.shape == (nwave,)
    assert bb_lo.shape == (nwave,)
    assert bb_hi.shape == (nwave,)
    assert Rrs_pred.shape == (nwave,)
    assert sigRrs.shape == (nwave,)

    assert np.all(a_med > 0), "median absorption should be positive"
    assert np.all(bb_med > 0), "median backscattering should be positive"
    assert np.all(a_hi >= a_lo), "absorption credible band must be ordered"
    assert np.all(bb_hi >= bb_lo), "backscattering credible band must be ordered"
    assert np.all(np.isfinite(Rrs_pred))
    assert np.all(np.abs(Rrs_pred) < 0.1)
    assert np.all(sigRrs >= 0)


def test_reconstruct_from_chains_standard_Gordon(l23_fit_standard):
    """End-to-end smoke test of reconstruct_from_chains with standard Gordon."""
    chains, models, prep_dict, rt_dict = l23_fit_standard

    out = evaluate.reconstruct_from_chains(models, chains, rt_dict,
                                           perc=(5, 95))
    a_med, bb_med, a_lo, a_hi, bb_lo, bb_hi, Rrs_pred, sigRrs = out

    wave = models[0].wave
    _validate_reconstruction(a_med, bb_med, a_lo, a_hi,
                             bb_lo, bb_hi, Rrs_pred, sigRrs, wave)

    # Median Rrs should be in the right ballpark relative to the data fit
    obs_Rrs = prep_dict['model_Rrs']
    mask = obs_Rrs > 0
    rel = np.abs(Rrs_pred[mask] - obs_Rrs[mask]) / obs_Rrs[mask]
    # Loose tolerance: short MCMC may not have fully converged.
    assert np.median(rel) < 0.5, \
        f"median relative error too large: {np.median(rel):.3f}"


def test_reconstruct_from_chains_variable_Gordon(l23_fit_variable):
    """reconstruct_from_chains works with wavelength-dependent G1, G2."""
    chains, models, _, rt_dict = l23_fit_variable

    out = evaluate.reconstruct_from_chains(models, chains, rt_dict,
                                           perc=(5, 95))
    a_med, bb_med, a_lo, a_hi, bb_lo, bb_hi, Rrs_pred, sigRrs = out

    wave = models[0].wave
    _validate_reconstruction(a_med, bb_med, a_lo, a_hi,
                             bb_lo, bb_hi, Rrs_pred, sigRrs, wave)

    # Sanity: variable-Gordon G arrays were actually loaded onto the model
    assert models[0].G1 is not None
    assert models[0].G2 is not None
    assert np.std(models[0].G1) > 0
    assert np.std(models[0].G2) > 0


def test_reconstruct_from_chains_with_Raman(l23_fit_raman):
    """Raman branch in reconstruct_from_chains.

    Exercises ``eval_a_ex`` / ``eval_bb_ex`` and the Raman correction inside
    ``calc_Rrs_from_models``.
    """
    chains, models, _, rt_dict = l23_fit_raman
    assert rt_dict['include_Raman'] is True

    out = evaluate.reconstruct_from_chains(models, chains, rt_dict,
                                           perc=(5, 95))
    a_med, bb_med, a_lo, a_hi, bb_lo, bb_hi, Rrs_pred, sigRrs = out

    wave = models[0].wave
    _validate_reconstruction(a_med, bb_med, a_lo, a_hi,
                             bb_lo, bb_hi, Rrs_pred, sigRrs, wave)


def test_reconstruct_matches_calc_Rrs_from_models(l23_fit_standard):
    """The refactor's key invariant.

    ``reconstruct_from_chains`` is now defined to forward Rrs assembly to
    ``calc_Rrs_from_models``. Computing the same quantity directly from the
    flattened chains via ``calc_Rrs_from_models`` and taking the median /
    std must match what ``reconstruct_from_chains`` returns -- to floating-
    point tolerance, since the underlying calculation is identical.
    """
    chains, models, _, rt_dict = l23_fit_standard

    _, _, _, _, _, _, Rrs_recon, sigRrs_recon = \
        evaluate.reconstruct_from_chains(models, chains, rt_dict,
                                         perc=(5, 95))

    # Replicate the chain flattening that reconstruct_from_chains does
    flat = evaluate.thin_burn_chains(chains)
    nparam_a = models[0].nparam
    aparams = flat[..., :nparam_a]
    bparams = flat[..., nparam_a:]
    Rrs_all = evaluate.calc_Rrs_from_models(models[0], aparams,
                                            models[1], bparams, rt_dict)
    Rrs_med_direct = np.median(Rrs_all, axis=0)
    sigRrs_direct = np.std(Rrs_all, axis=0)

    np.testing.assert_allclose(Rrs_recon, Rrs_med_direct,
                               rtol=1e-12, atol=1e-15,
                               err_msg="median Rrs mismatch between "
                                       "reconstruct_from_chains and "
                                       "calc_Rrs_from_models")
    np.testing.assert_allclose(sigRrs_recon, sigRrs_direct,
                               rtol=1e-12, atol=1e-15,
                               err_msg="sigRrs mismatch between "
                                       "reconstruct_from_chains and "
                                       "calc_Rrs_from_models")


def test_reconstruct_matches_calc_Rrs_from_models_raman(l23_fit_raman):
    """Same invariant as above but for the Raman code path."""
    chains, models, _, rt_dict = l23_fit_raman

    _, _, _, _, _, _, Rrs_recon, _ = \
        evaluate.reconstruct_from_chains(models, chains, rt_dict)

    flat = evaluate.thin_burn_chains(chains)
    nparam_a = models[0].nparam
    Rrs_all = evaluate.calc_Rrs_from_models(
        models[0], flat[..., :nparam_a],
        models[1], flat[..., nparam_a:], rt_dict)
    Rrs_med_direct = np.median(Rrs_all, axis=0)

    np.testing.assert_allclose(Rrs_recon, Rrs_med_direct,
                               rtol=1e-12, atol=1e-15)


def test_calc_Rrs_from_models_single_param(l23_fit_standard):
    """calc_Rrs_from_models accepts a 1-D parameter vector (log_prob path)."""
    chains, models, _, rt_dict = l23_fit_standard

    flat = evaluate.thin_burn_chains(chains)
    p_med = np.median(flat, axis=0)
    aparams = p_med[:models[0].nparam]
    bparams = p_med[models[0].nparam:]

    Rrs = evaluate.calc_Rrs_from_models(models[0], aparams,
                                        models[1], bparams, rt_dict)

    # eval_a / eval_bb return (1, nwave) for 1-D params; calc_Rrs preserves
    # that batch axis. Just check the trailing dim + that values are physical.
    assert Rrs.shape[-1] == len(models[0].wave)
    Rrs_flat = np.squeeze(Rrs)
    assert np.all(np.isfinite(Rrs_flat))
    assert np.all(np.abs(Rrs_flat) < 0.1)


def test_reconstruct_chisq_fits_basic(l23_fit_standard):
    """reconstruct_chisq_fits returns (Rrs, a, bb) of the right shapes."""
    chains, models, _, rt_dict = l23_fit_standard
    flat = evaluate.thin_burn_chains(chains)
    p_med = np.median(flat, axis=0)

    # ExpBricaud uses Chl via set_aph; recover it from the median Aph param
    # the same way the rest of the codebase does (Chl = 10^Aph / 0.05582).
    Chl = 10**p_med[models[0].nparam - 1] / 0.05582

    # 1-D path
    Rrs, a, bb = evaluate.reconstruct_chisq_fits(models, p_med, rt_dict,
                                                 Chl=Chl)
    nwave = len(models[0].wave)
    assert Rrs.shape == (nwave,)
    assert a.shape == (nwave,)
    assert bb.shape == (nwave,)
    assert np.all(a > 0)
    assert np.all(bb > 0)

    # 2-D path: stack two copies; rows should be identical.
    params2 = np.stack([p_med, p_med], axis=0)
    Chl2 = np.array([Chl, Chl])
    Rrs2, a2, bb2 = evaluate.reconstruct_chisq_fits(models, params2, rt_dict,
                                                    Chl=Chl2)
    assert Rrs2.shape == (2, nwave)
    assert a2.shape == (2, nwave)
    assert bb2.shape == (2, nwave)
    np.testing.assert_allclose(Rrs2[0], Rrs2[1], rtol=1e-12)

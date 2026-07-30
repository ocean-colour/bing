""" Tests for MCMC inference helpers

Concentrated on walker initialization.  The perturbation used to be
multiplicative, so a parameter seeded at exactly 0 got zero spread
across walkers and -- because emcee proposes along walker-to-walker
vectors -- never moved for the whole run, silently and with a healthy
acceptance fraction.  These tests pin the fix.
"""
import numpy as np

import pytest

from bing.fitting import inference as bing_inf
from bing.models import utils as model_utils
from bing.parameters import standard
from bing.fitting import chisq_fit
from bing.rt import defs as rt_defs

wave = np.arange(400., 705., 5.)

# ExpBricaud + Pow2: Adg, Sdg, Aph, Bmin, eta_min, Borg, eta_org.
# eta_min sits at exactly 0 -- a flat mineral term, which a chi-squared
# fit legitimately converges to, and the case that used to freeze.
P0_WITH_ZERO = np.array([-1.0, 0.015, -0.7, -1.3, 0.0, -1.0, 1.8])


@pytest.fixture(scope="module")
def pow2_models():
    p = standard.expb_pow2()
    return model_utils.init(p.model_names, wave, (p.apriors, p.bpriors))


def test_every_dimension_gets_spread():
    # The regression: no dimension may be degenerate, including the one
    # seeded at exactly 0
    walkers = bing_inf.init_walkers(P0_WITH_ZERO, 32,
                                    rng=np.random.default_rng(3))
    assert walkers.shape == (32, P0_WITH_ZERO.size)

    spread = walkers.std(axis=0)
    assert np.all(spread > 0.), f"dead dimension(s): {np.where(spread == 0.)}"

    # The zero-seeded slot gets the floor, not nothing
    izero = int(np.where(P0_WITH_ZERO == 0.)[0][0])
    assert spread[izero] > 1e-4
    assert np.all(np.abs(walkers[:, izero]) <= 1e-3 + 1e-12)

    # Under the old multiplicative rule that slot was exactly frozen
    old = np.tile(P0_WITH_ZERO, (32, 1))
    old = old + old*np.random.default_rng(3).uniform(-1e-2, 1e-2,
                                                     size=old.shape)
    assert old[:, izero].std() == 0.


def test_scale_matches_history_for_well_scaled_params():
    # For |p0|*frac above the floor the ball is the historical 1% one,
    # so existing fits are not perturbed differently
    p0 = np.array([-2.0, 1.5, 3.0])       # all |p0|*1e-2 > 1e-3
    walkers = bing_inf.init_walkers(p0, 512,
                                    rng=np.random.default_rng(0))
    half_width = np.abs(walkers - p0).max(axis=0)
    assert np.all(half_width <= np.abs(p0)*1e-2 + 1e-12)
    # and it really does fill that interval
    assert np.all(half_width > 0.9*np.abs(p0)*1e-2)


def test_floor_only_bites_near_zero():
    p0 = np.array([-2.0, 0.015, 0.0])
    walkers = bing_inf.init_walkers(p0, 512,
                                    rng=np.random.default_rng(1))
    half_width = np.abs(walkers - p0).max(axis=0)
    # Well-scaled parameter keeps the relative ball ...
    assert half_width[0] <= 0.02 + 1e-12
    # ... while the small and zero ones get the absolute floor
    assert half_width[1] > 1.5e-4      # old rule gave only 1.5e-4
    assert half_width[1] <= 1e-3 + 1e-12
    assert half_width[2] > 5e-4


def test_walkers_start_in_prior(pow2_models):
    # Every walker must have a finite log-prior, even when p0 sits
    # exactly on a prior boundary (eta_min pmin = -0.5)
    p0 = P0_WITH_ZERO.copy()
    p0[4] = -0.5
    walkers = bing_inf.init_walkers(p0, 64, models=pow2_models,
                                    floor=0.05,
                                    rng=np.random.default_rng(5))

    na = pow2_models[0].nparam
    for walker in walkers:
        lnp = (pow2_models[0].priors.calc(walker[:na]) +
               pow2_models[1].priors.calc(walker[na:]))
        assert np.isfinite(lnp)

    low, high = bing_inf.prior_bounds(pow2_models)
    assert np.all(walkers >= low) and np.all(walkers <= high)
    # Clipping must not flatten the dimension it clipped
    assert walkers[:, 4].std() > 0.


def test_unclipped_without_models():
    # No models -> no bounds to respect, and no crash
    p0 = np.array([-1.0, 0.0])
    walkers = bing_inf.init_walkers(p0, 16, models=None,
                                    rng=np.random.default_rng(2))
    assert walkers.shape == (16, 2)
    assert np.all(walkers.std(axis=0) > 0.)


def test_prior_bounds_tolerates_missing_bounds(pow2_models):
    low, high = bing_inf.prior_bounds(pow2_models)
    assert low.size == sum(m.nparam for m in pow2_models)
    assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))

    # A gaussian prior inherits pmin/pmax = None from the Prior base, so
    # that slot must come back infinite rather than raising or turning
    # into NaN.  (Note standard.expb_pow(beta=...) does NOT produce one:
    # that branch of set_standard_priors is gated behind bpriors being
    # None, and expb_pow always supplies bpriors.)
    apriors = [dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
    apriors[1] = dict(flavor='uniform', pmin=0.01, pmax=0.02)
    bpriors = [dict(flavor='log_uniform', pmin=-6, pmax=5),
               dict(flavor='gaussian', mean=1.0, sigma=0.1)]
    models = model_utils.init(['ExpBricaud', 'Pow'], wave,
                              (apriors, bpriors))
    assert models[1].priors.priors[-1].flavor == 'gaussian'
    low, high = bing_inf.prior_bounds(models)
    assert not np.any(np.isnan(low)) and not np.any(np.isnan(high))
    assert np.isneginf(low[-1]) and np.isposinf(high[-1])

    # ... and such a parameter is simply left unclipped
    walkers = bing_inf.init_walkers(np.zeros(low.size), 16,
                                    models=models,
                                    rng=np.random.default_rng(4))
    assert walkers[:, -1].std() > 0.

    # No priors attached at all is also fine
    bare = model_utils.init(['ExpBricaud', 'Pow'], wave)
    low, high = bing_inf.prior_bounds(bare)
    assert np.all(np.isneginf(low)) and np.all(np.isposinf(high))


def test_legacy_seed_still_reproducible():
    # bing.fitting.l23.batch_fit calls np.random.seed for
    # reproducibility, so the default RNG must remain the legacy global
    np.random.seed(1234)
    first = bing_inf.init_walkers(P0_WITH_ZERO, 16)
    np.random.seed(1234)
    second = bing_inf.init_walkers(P0_WITH_ZERO, 16)
    assert np.allclose(first, second)

    np.random.seed(4321)
    third = bing_inf.init_walkers(P0_WITH_ZERO, 16)
    assert not np.allclose(first, third)


def test_emcee_moves_every_dimension(pow2_models):
    """End-to-end: the frozen dimension now samples.

    Short real run through run_emcee on a synthetic spectrum whose truth
    has eta_min = 0 exactly.  Before the fix that column came back with
    a standard deviation of exactly 0 across 400 steps while the other
    six explored normally.
    """
    p = standard.expb_pow2(wv_min=400., wv_max=700.,
                           variable_Gordon=False)
    models = model_utils.init(p.model_names, wave,
                              (p.apriors, p.bpriors))
    models[0].set_aph(np.array([1.0]))
    rt_dict = rt_defs.rt_dict_from_p(p)

    Rrs = chisq_fit.fit_func(wave, *P0_WITH_ZERO, models=models,
                             rt_dict=rt_dict)
    varRrs = (0.02*Rrs)**2

    np.random.seed(7)
    sampler = bing_inf.run_emcee(models, Rrs, varRrs, rt_dict,
                                 nwalkers=16, nburn=50, nsteps=250,
                                 skip_check=True,
                                 p0=P0_WITH_ZERO.copy())
    chains = sampler.get_chain()
    assert chains.shape == (250, 16, P0_WITH_ZERO.size)

    spread = chains.reshape(-1, P0_WITH_ZERO.size).std(axis=0)
    assert np.all(spread > 0.), \
        f"frozen dimension(s): {np.where(spread == 0.)}"
    # The formerly frozen slot must actually explore, not just jitter
    assert spread[4] > 1e-4
    assert np.mean(sampler.acceptance_fraction) > 0.05

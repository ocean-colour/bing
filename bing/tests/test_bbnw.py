""" Tests for non-water backscattering (bb_nw) models

Covers the turbid-water work: the free-slope ``PowFlex`` prior set
(``standard.expb_powflex``) and, as they land, the two-component
``Pow2`` / ``Pow2Flat`` models.

The PowFlex tests deliberately assert the *discriminating* property --
that a negative power-law exponent is reachable, so bb_nw may flatten or
rise toward the red (the "white"/mineral limit of turbid water) -- and
compare against ``expb_pow``, which floors beta at 0.
"""
import numpy as np

import pytest

from bing.models import bbnw as bing_bbnw
from bing.models import utils as model_utils
from bing.parameters import standard
from bing.priors import priors as bing_priors
from bing.fitting import chisq_fit
from bing.rt import defs as rt_defs

wave = np.arange(350, 755, 5.)
# Fit grid: inside the Gordon/water tables, so no trimming is needed
wave_fit = np.arange(400., 705., 5.)


def bb_bounds(p):
    """Chi-squared bb bounds, built exactly as bing.fitting.l23 does.

    l23.py reads pmin/pmax from the *raw prior dicts* (not the Prior
    objects on the model), so this reproduces the bound curve_fit
    actually sees.

    Parameters
    ----------
    p : namedtuple
        A BING parameter tuple (e.g. from standard.expb_powflex).

    Returns
    -------
    tuple of np.ndarray
        (lower, upper) bounds for the bb parameters.
    """
    low = np.array([item['pmin'] for item in p.bpriors], dtype=float)
    high = np.array([item['pmax'] for item in p.bpriors], dtype=float)
    return low, high


def log10_by_flavor(p0, prior_dicts):
    """log10 the amplitude slots of p0, the way the fitters do.

    Mirrors bing.fitting.l23 (:384-390) and ioptics.run._log_mask: the
    slots to convert are chosen by prior *flavor*, not by position.

    Parameters
    ----------
    p0 : np.ndarray
        Initial guess with amplitudes in linear space.
    prior_dicts : list of dict
        The matching prior dicts, in parameter order.

    Returns
    -------
    np.ndarray
        p0 with the log-flavored slots converted to log10.
    """
    out = np.array(p0, dtype=float)
    for kk, pdict in enumerate(prior_dicts):
        if pdict['flavor'][0:3] == 'log':
            out[kk] = np.log10(out[kk])
    return out


def fit_setup(p):
    """Build the pieces a synthetic chi-squared fit needs.

    Parameters
    ----------
    p : namedtuple
        A BING parameter tuple with apriors/bpriors set.

    Returns
    -------
    tuple
        (models, rt_dict, bounds) ready for chisq_fit.fit, with bounds
        assembled from the prior dicts as bing.fitting.l23 does.
    """
    models = model_utils.init(p.model_names, wave_fit,
                              (p.apriors, p.bpriors))
    models[0].set_aph(np.array([1.0]))  # Bricaud aph needs a Chl
    rt_dict = rt_defs.rt_dict_from_p(p)
    low = np.array([d['pmin'] for d in p.apriors] +
                   [d['pmin'] for d in p.bpriors], dtype=float)
    high = np.array([d['pmax'] for d in p.apriors] +
                    [d['pmax'] for d in p.bpriors], dtype=float)
    return models, rt_dict, (low, high)


@pytest.fixture(scope="module")
def pow_model():
    """A Pow bb model carrying the PowFlex priors (loads L23 bb_w once)."""
    p = standard.expb_powflex()
    return bing_bbnw.init_model('Pow', wave, p.bpriors)


def test_powflex_priors_admit_negative_slope():
    p = standard.expb_powflex()
    ref = standard.expb_pow()

    # Same models, same a-priors -- only beta's range differs
    assert p.model_names == ['ExpBricaud', 'Pow']
    assert p.model_names == ref.model_names
    assert p.apriors == ref.apriors
    assert len(p.bpriors) == 2

    # The one substantive difference: beta may go negative
    assert p.bpriors[1]['flavor'] == 'uniform'
    assert np.isclose(p.bpriors[1]['pmin'], -1.)
    assert np.isclose(p.bpriors[1]['pmax'], 2.)
    assert np.isclose(ref.bpriors[1]['pmin'], 0.)  # what we relax

    # Amplitude stays log-flavored, slope stays linear.  This pattern is
    # load-bearing: the fitters log10 p0 slots by flavor, so swapping
    # these silently starts the fit in the wrong space.
    flavors = [pd['flavor'][0:3] == 'log' for pd in p.bpriors]
    assert flavors == [True, False]


def test_powflex_prior_evaluates_negative_beta():
    priors = bing_priors.Priors(standard.expb_powflex().bpriors)
    ref = bing_priors.Priors(standard.expb_pow().bpriors)

    # Flat and rising slopes are now in-prior ...
    assert priors.priors[1].calc(0.) == 0.
    assert priors.priors[1].calc(-0.5) == 0.
    # ... while expb_pow rejects the same value outright
    assert ref.priors[1].calc(-0.5) == -np.inf

    # Still bounded on both sides
    assert priors.priors[1].calc(-1.5) == -np.inf
    assert priors.priors[1].calc(2.5) == -np.inf


def test_powflex_bounds_reach_negative_beta():
    low, high = bb_bounds(standard.expb_powflex())
    ref_low, _ = bb_bounds(standard.expb_pow())

    # curve_fit must be *allowed* to explore a rising bb_nw
    assert np.isclose(low[1], -1.)
    assert np.isclose(high[1], 2.)
    assert np.isclose(ref_low[1], 0.)
    assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))


def test_powflex_rising_bbnw(pow_model):
    i440 = np.argmin(np.abs(wave - 440.))
    i700 = np.argmin(np.abs(wave - 700.))

    # Negative beta -> bb_nw RISES with wavelength (mineral/white limit)
    bb_rise = pow_model.eval_bbnw(np.array([-1., -0.5]))
    assert bb_rise.shape == (1, wave.size)
    assert bb_rise[0, i700] > bb_rise[0, i440]
    # Exact form check: 10**Bnw * (600/wave)**beta
    assert np.isclose(bb_rise[0, i700],
                      0.1*(pow_model.pivot/wave[i700])**(-0.5))

    # beta = 0 -> spectrally flat
    bb_flat = pow_model.eval_bbnw(np.array([-1., 0.]))
    assert np.allclose(bb_flat, 0.1)

    # Positive beta -> the open-ocean decreasing power law (unchanged)
    bb_fall = pow_model.eval_bbnw(np.array([-1., 1.]))
    assert bb_fall[0, i700] < bb_fall[0, i440]

    # 2-D (chain-shaped) params keep the (nsample, nwave) contract
    chains = np.array([[-1., -0.5], [-1., 1.]])
    bb_2d = pow_model.eval_bbnw(chains)
    assert bb_2d.shape == (2, wave.size)
    assert np.allclose(bb_2d[0], bb_rise[0])


def test_powflex_p0_within_bounds(pow_model):
    p = standard.expb_powflex()

    # A rising, turbid-looking bb_nw to seed from
    bb_nw = 0.05*(wave/600.)**0.3
    p0 = pow_model.init_guess(bb_nw)
    assert p0.size == pow_model.nparam

    # Amplitude comes back LINEAR; the caller log10s it by flavor
    assert np.isclose(p0[0], bb_nw[np.argmin(np.abs(wave - 600.))])
    p0_fit = log10_by_flavor(p0, p.bpriors)
    assert np.isclose(p0_fit[0], np.log10(p0[0]))
    assert np.isclose(p0_fit[1], p0[1])  # slope untouched

    # The seed must be feasible for curve_fit ...
    low, high = bb_bounds(p)
    assert np.all(p0_fit >= low) and np.all(p0_fit <= high)
    # ... and in-prior for MCMC
    assert pow_model.priors.calc(p0_fit) == 0.

    # No slot seeded at exactly 0: inference.py perturbs walkers
    # multiplicatively (p0 += p0*U(-1e-2,1e-2)), so a 0 seed gets zero
    # spread and that dimension never moves for the whole MCMC run.
    assert np.all(p0_fit != 0.)


def test_powflex_recovers_rising_bbnw_chisq():
    """End-to-end: a rising-bb spectrum is recoverable, and expb_pow can't.

    Builds a synthetic Rrs from a truth with beta = -0.4 (bb_nw rising
    toward the red, the turbid/mineral limit), then fits it twice: once
    with the PowFlex priors and once with expb_pow's beta >= 0 floor.
    This is the whole point of the control experiment -- it exercises
    priors -> curve_fit bounds -> recovery in one shot.

    variable_Gordon is off so the forward model is the plain elastic
    Gordon relation (the wavelength-dependent coefficients would need
    G1/G2 seeded on the a-model).
    """
    truth = np.array([-1.0, 0.015, -0.7, -1.0, -0.4])
    p0 = np.array([-0.5, 0.015, -0.5, -0.5, 0.5])

    p = standard.expb_powflex(wv_min=400., wv_max=700.,
                              variable_Gordon=False)
    models, rt_dict, bounds = fit_setup(p)
    Rrs = chisq_fit.fit_func(wave_fit, *truth, models=models,
                             rt_dict=rt_dict)
    varRrs = (0.02*Rrs)**2

    ans, _, _ = chisq_fit.fit((Rrs, varRrs, p0, 0), models, rt_dict,
                              bounds=bounds)
    pred = chisq_fit.fit_func(wave_fit, *ans, models=models,
                              rt_dict=rt_dict)
    assert np.allclose(ans, truth, atol=1e-3)
    assert ans[-1] < 0.       # the slope expb_pow's prior forbids
    assert np.max(np.abs(pred - Rrs)/Rrs) < 1e-6

    # Control: expb_pow pins beta at its floor and cannot follow the data
    q = standard.expb_pow(wv_min=400., wv_max=700.,
                          variable_Gordon=False)
    models_q, rt_q, bounds_q = fit_setup(q)
    ans_q, _, _ = chisq_fit.fit((Rrs, varRrs, p0, 0), models_q, rt_q,
                                bounds=bounds_q)
    pred_q = chisq_fit.fit_func(wave_fit, *ans_q, models=models_q,
                                rt_dict=rt_q)
    assert np.isclose(ans_q[-1], 0., atol=1e-6)   # stuck on the boundary
    assert np.max(np.abs(pred_q - Rrs)/Rrs) > 1e-3


def test_powflex_prior_count_matches_model(pow_model):
    p = standard.expb_powflex()

    # Nothing in BING validates this, and a mismatch silently log10s the
    # wrong p0 slots on the chi-squared path -- so assert it here.
    assert len(p.bpriors) == pow_model.nparam
    assert pow_model.priors.nparam == pow_model.nparam
    assert pow_model.pnames == ['Bnw', 'beta']


# ---------------------------------------------------------------------
# Pow2 / Pow2Flat -- two-component (mineral + organic) backscattering
# ---------------------------------------------------------------------

# (name, nparam, pnames, log_params, factory)
TWO_COMP = [
    ('Pow2', 4, ['Bmin', 'eta_min', 'Borg', 'eta_org'],
     [True, False, True, False], standard.expb_pow2),
    ('Pow2Flat', 3, ['Bmin', 'Borg', 'eta_org'],
     [True, True, False], standard.expb_pow2flat),
]

# Mineral-dominated parameter sets: a large, flat-to-rising mineral term
# plus a small steep organic one.  Keyed by model name.
TURBID_PARAMS = {
    'Pow2': np.array([np.log10(0.25), -0.2, np.log10(0.005), 1.2]),
    'Pow2Flat': np.array([np.log10(0.25), np.log10(0.005), 1.2]),
}


@pytest.fixture(scope="module")
def two_comp_models():
    """Both two-component models, priors attached, built once."""
    out = {}
    for name, _, _, _, factory in TWO_COMP:
        out[name] = bing_bbnw.init_model(name, wave, factory().bpriors)
    return out


@pytest.mark.parametrize("name,nparam,pnames,log_params,factory", TWO_COMP)
def test_two_comp_init(name, nparam, pnames, log_params, factory):
    # Test 1: construction / registry, with no priors at all
    model = bing_bbnw.init_model(name, wave)
    assert model.name == name
    assert model.nparam == nparam
    assert model.pnames == pnames
    assert len(model.pnames) == model.nparam
    assert model.priors is None          # factory's prior_dicts default
    assert model.log_params == log_params
    assert np.isclose(model.pivot, 600.)
    assert np.isclose(model.pivot_min, 700.)

    # Water backscattering comes from the base class -- not recomputed
    assert model.bb_w is not None
    assert model.bb_w.size == wave.size

    # Direct construction works too (prior_dicts defaults to None),
    # unlike the older bb classes
    direct = getattr(bing_bbnw, 'bbNW' + name)(wave)
    assert direct.nparam == nparam


@pytest.mark.parametrize("name,nparam,pnames,log_params,factory", TWO_COMP)
def test_two_comp_shapes(name, nparam, pnames, log_params, factory,
                         two_comp_models):
    # Test 2: shape contract for 1-D and chain-shaped params
    model = two_comp_models[name]
    params = TURBID_PARAMS[name]

    bb_1d = model.eval_bbnw(params)
    assert bb_1d.shape == (1, wave.size)
    assert np.all(np.isfinite(bb_1d)) and np.all(bb_1d > 0)

    chains = np.vstack([params, params + 0.1])
    bb_2d = model.eval_bbnw(chains)
    assert bb_2d.shape == (2, wave.size)
    assert np.all(np.isfinite(bb_2d)) and np.all(bb_2d > 0)
    assert np.allclose(bb_2d[0], bb_1d[0])

    # eval_bb just adds pure water
    assert np.allclose(model.eval_bb(params), model.bb_w + bb_1d)


@pytest.mark.parametrize("name,nparam,pnames,log_params,factory", TWO_COMP)
def test_two_comp_honours_wave_kwarg(name, nparam, pnames, log_params,
                                     factory, two_comp_models):
    # Test 3: the wave kwarg must be honoured, or Raman is wrong.
    # (Lee/GSM/Every ignore it today -- see the doc's landmine 6.)
    model = two_comp_models[name]
    params = TURBID_PARAMS[name]

    bb_em = model.eval_bbnw(params)
    bb_ex = model.eval_bbnw(params, wave=model.wave_ex)
    assert bb_ex.shape == bb_em.shape
    # The excitation grid is bluer, so the values must actually differ
    assert not np.allclose(bb_ex, bb_em)

    # And eval_bb_ex must be bb_w_ex + bb_nw on the excitation grid
    assert np.allclose(model.eval_bb_ex(params), model.bb_w_ex + bb_ex)


def test_pow2_reduces_to_pow(two_comp_models):
    # Test 4: killing the organic term leaves a plain power law, and
    # killing the mineral term leaves the 600 nm-pivoted organic one.
    model = two_comp_models['Pow2']
    pow_model = bing_bbnw.init_model('Pow', wave)

    # Mineral off -> exactly Pow(Borg, eta_org) at the same pivot
    only_org = model.eval_bbnw(np.array([-12., 0., -2., 1.1]))
    ref_org = pow_model.eval_bbnw(np.array([-2., 1.1]))
    assert np.allclose(only_org, ref_org, rtol=1e-9)

    # Organic off -> the 700 nm-pivoted mineral law (hand-written, since
    # Pow's pivot is 600)
    only_min = model.eval_bbnw(np.array([-1., -0.3, -12., 1.1]))
    ref_min = 0.1*(700./wave)**(-0.3)
    assert np.allclose(only_min[0], ref_min, rtol=1e-9)

    # Pow2Flat likewise reduces to a constant plus Pow
    flat = two_comp_models['Pow2Flat']
    only_org_f = flat.eval_bbnw(np.array([-12., -2., 1.1]))
    assert np.allclose(only_org_f, ref_org, rtol=1e-9)


@pytest.mark.parametrize("name,nparam,pnames,log_params,factory", TWO_COMP)
def test_two_comp_turbid_shape(name, nparam, pnames, log_params, factory,
                               two_comp_models):
    # Test 5: the turbid shape the open-ocean Pow (beta in [0,2]) cannot
    # make -- flat-to-rising, and of order 0.1-0.4 m^-1 in the red.
    model = two_comp_models[name]
    params = TURBID_PARAMS[name]
    i440 = np.argmin(np.abs(wave - 440.))
    i700 = np.argmin(np.abs(wave - 700.))

    bb_nw = model.eval_bbnw(params)[0]
    assert 0.1 < bb_nw[i700] < 0.4          # the magnitude the red needs
    assert bb_nw[i700] > 0.9*bb_nw[i440]    # flat or rising

    # Pow2's mineral exponent can make it genuinely rise
    if name == 'Pow2':
        assert bb_nw[i700] > bb_nw[i440]

    # A single Pow with the same red amplitude cannot: with beta >= 0 it
    # must be >= its 700 nm value everywhere blueward.
    pow_model = bing_bbnw.init_model('Pow', wave)
    pow_bb = pow_model.eval_bbnw(np.array([np.log10(bb_nw[i700]), 0.5]))[0]
    assert pow_bb[i440] > pow_bb[i700]


@pytest.mark.parametrize("name,nparam,pnames,log_params,factory", TWO_COMP)
def test_two_comp_priors(name, nparam, pnames, log_params, factory,
                         two_comp_models):
    # Test 6: priors attach, and every one exposes pmin/pmax so the
    # chi-squared bounds path works
    model = two_comp_models[name]
    p = factory()

    assert model.priors is not None
    assert model.priors.nparam == nparam
    for prior in model.priors.priors:
        assert prior.pmin is not None and prior.pmax is not None
        assert prior.pmax > prior.pmin

    # The exponent priors are linear-uniform; disjoint for Pow2
    ieta_org = pnames.index('eta_org')
    assert model.priors.priors[ieta_org].flavor == 'uniform'
    if name == 'Pow2':
        ieta_min = pnames.index('eta_min')
        assert model.priors.priors[ieta_min].flavor == 'uniform'
        assert np.isclose(p.bpriors[ieta_min]['pmin'], -0.5)
        assert np.isclose(p.bpriors[ieta_min]['pmax'], 0.5)
        # Disjoint ranges break the label-switching degeneracy
        assert p.bpriors[ieta_min]['pmax'] <= p.bpriors[ieta_org]['pmin']

    low, high = bb_bounds(p)
    assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))


@pytest.mark.parametrize("name,nparam,pnames,log_params,factory", TWO_COMP)
def test_two_comp_p0_round_trip(name, nparam, pnames, log_params, factory,
                                two_comp_models):
    # Tests 7 and 10: init_guess returns LINEAR amplitudes that survive
    # the flavor-driven log10, land in bounds, and reproduce the input
    # bb_nw at the pivot to within a factor of ~2.  And no slot is
    # seeded at exactly 0 (landmine 1: a 0 seed freezes that MCMC
    # dimension for the whole run, because the walker perturbation in
    # bing.fitting.inference is multiplicative).
    model = two_comp_models[name]
    p = factory()

    bb_in = 0.05*(wave/600.)**0.2          # mildly rising, turbid-ish
    p0 = model.init_guess(bb_in)
    assert p0.size == nparam
    assert np.all(p0 != 0.)

    p0_fit = log10_by_flavor(p0, p.bpriors)
    for kk, is_log in enumerate(log_params):
        if is_log:
            assert np.isclose(p0_fit[kk], np.log10(p0[kk]))
        else:
            assert np.isclose(p0_fit[kk], p0[kk])
    assert np.all(p0_fit != 0.)

    # Feasible for curve_fit and in-prior for MCMC
    low, high = bb_bounds(p)
    assert np.all(p0_fit >= low) and np.all(p0_fit <= high)
    assert model.priors.calc(p0_fit) == 0.

    # And the seed is the right order of magnitude at the pivot
    i600 = np.argmin(np.abs(wave - 600.))
    bb_p0 = model.eval_bbnw(p0_fit)[0]
    assert 0.5 < bb_p0[i600]/bb_in[i600] < 2.


@pytest.mark.parametrize("name,nparam,pnames,log_params,factory", TWO_COMP)
def test_two_comp_prior_length_guard(name, nparam, pnames, log_params,
                                     factory):
    # Test 11: a wrong-length prior list must fail loudly at attach
    # time, not silently mis-log10 p0 slots much later
    short = [dict(flavor='log_uniform', pmin=-6, pmax=5)]*2
    with pytest.raises(ValueError):
        bing_bbnw.init_model(name, wave, short)

    too_many = [dict(flavor='log_uniform', pmin=-6, pmax=5)]*(nparam + 1)
    with pytest.raises(ValueError):
        bing_bbnw.init_model(name, wave, too_many)

    # The guard also covers set_standard_priors, which attaches priors
    # after construction (the path the real fitters take)
    p = factory()
    models = model_utils.init(p.model_names, wave)
    bad = p._replace(bpriors=short)
    with pytest.raises(ValueError):
        bing_priors.set_standard_priors(models, bad)


def test_two_comp_beats_single_powerlaw_chisq():
    """The core claim: a two-component bb shape needs two components.

    Builds a synthetic Rrs whose true bb_nw is a *sum* of a steep
    organic power law (dominating the blue) and a flat mineral term
    (dominating the red), so its log-log slope changes with wavelength.
    Pow2 and Pow2Flat recover it exactly; a single power law cannot,
    **and widening that power law's slope range does not help** --
    PowFlex lands on the same optimum as expb_pow.  That is the
    "form, not range" argument as a regression test.
    """
    # bb_nw: organic 0.10 at 600 with eta=1.8, mineral 0.05 flat
    truth = np.array([-0.3, 0.015, -0.4,
                      np.log10(0.05), 0.0, np.log10(0.10), 1.8])
    p2 = standard.expb_pow2(wv_min=400., wv_max=700.,
                            variable_Gordon=False)
    models, rt_dict, bounds = fit_setup(p2)
    Rrs = chisq_fit.fit_func(wave_fit, *truth, models=models,
                             rt_dict=rt_dict)
    varRrs = (0.02*Rrs)**2

    def misfit(p, p0):
        m, rt, bnds = fit_setup(p)
        ans, _, _ = chisq_fit.fit((Rrs, varRrs, p0, 0), m, rt,
                                  bounds=bnds)
        pred = chisq_fit.fit_func(wave_fit, *ans, models=m, rt_dict=rt)
        return float(np.median(np.abs(pred - Rrs)/Rrs)), ans

    # Two components: exact recovery
    m2, ans2 = misfit(p2, np.array([-0.5, 0.015, -0.5, -1., 0.05, -2., 1.]))
    assert m2 < 1e-6
    assert np.allclose(ans2, truth, atol=1e-3)

    mflat, _ = misfit(
        standard.expb_pow2flat(wv_min=400., wv_max=700.,
                               variable_Gordon=False),
        np.array([-0.5, 0.015, -0.5, -1., -2., 1.]))
    assert mflat < 1e-6          # the truth has a flat mineral term

    # One component: plateaus, and the free slope buys nothing
    single_p0 = np.array([-0.5, 0.015, -0.5, -1., 0.5])
    mpow, anspow = misfit(
        standard.expb_pow(wv_min=400., wv_max=700.,
                          variable_Gordon=False), single_p0)
    mflex, ansflex = misfit(
        standard.expb_powflex(wv_min=400., wv_max=700.,
                             variable_Gordon=False), single_p0)
    assert mpow > 1e-4
    assert mflex > 1e-4
    assert mpow/m2 > 100. and mflex/m2 > 100.
    # PowFlex's optimum has beta > 0, i.e. inside expb_pow's range too
    assert np.allclose(anspow, ansflex, atol=1e-3)


# ---------------------------------------------------------------------
# End-to-end through the L23 machinery (guarded: needs the data tree)
# ---------------------------------------------------------------------

# The same clear, blue-peaked spectrum the existing L23 tests use
L23_IDX = 170


@pytest.fixture(scope="module")
def l23_lm_fits():
    """Chi-squared fits of one clear L23 spectrum, per bb model.

    Skips rather than fails when the L23 data tree (or one of the l23
    module's dependencies) is missing, so this file still runs on a bare
    checkout.

    Returns
    -------
    dict
        Keyed by combo name, each with the fitted parameters, models,
        prep dict, predicted Rrs and rt_dict.
    """
    try:
        from bing.fitting import l23 as fit_l23
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"bing.fitting.l23 unavailable: {exc}")

    out = {}
    for combo in ('expb_pow', 'expb_pow2', 'expb_pow2flat'):
        p = getattr(standard, combo)(satellite='PACE', add_noise=False,
                                     variable_Gordon=True)
        try:
            ans, _, models, prep, _ = fit_l23.fit_with_LM(p, L23_IDX)
        except (FileNotFoundError, OSError) as exc:  # pragma: no cover
            pytest.skip(f"L23 data tree unavailable: {exc}")
        rt_dict = rt_defs.rt_dict_from_p(p)
        pred = chisq_fit.fit_func(models[0].wave, *ans, models=models,
                                  rt_dict=rt_dict)
        out[combo] = dict(ans=ans, models=models, prep=prep, pred=pred,
                          rt_dict=rt_dict)
    return out


def chi2_of(fit):
    """Raw chi-squared and its reduced form for one entry of l23_lm_fits."""
    Rrs = fit['prep']['model_Rrs']
    var = fit['prep']['model_varRrs']
    chi2 = float(np.sum((fit['pred'] - Rrs)**2/var))
    ndof = Rrs.size - fit['ans'].size
    return chi2, chi2/ndof


def test_l23_pow2_fit_is_finite(l23_lm_fits):
    # Test 9, part 1: the fit completes and produces physical output
    ref = l23_lm_fits['expb_pow']
    Rrs = ref['prep']['model_Rrs']
    wave = ref['models'][0].wave
    # Premise of this test: a clear spectrum, peaking in the blue
    assert wave[np.argmax(Rrs)] < 500.

    for combo in ('expb_pow2', 'expb_pow2flat'):
        fit = l23_lm_fits[combo]
        assert np.all(np.isfinite(fit['pred']))
        assert np.all(fit['pred'] > 0.)
        assert np.all(np.isfinite(fit['ans']))

        bb_nw = fit['models'][1].eval_bbnw(fit['ans'][3:])
        assert np.all(np.isfinite(bb_nw)) and np.all(bb_nw > 0.)


@pytest.mark.parametrize("combo", ['expb_pow2', 'expb_pow2flat'])
def test_l23_two_comp_no_worse_than_pow(combo, l23_lm_fits):
    """Test 9, part 2: the extra components must not cost accuracy.

    Pow2 contains Pow exactly (kill the mineral term and the pivots
    agree), so its *raw* chi-squared cannot legitimately be worse. The
    reduced chi-squared may rise purely because it divides by fewer
    degrees of freedom, so the allowance below is exactly that
    bookkeeping factor and nothing more.
    """
    chi2_ref, chi2nu_ref = chi2_of(l23_lm_fits['expb_pow'])
    chi2_new, chi2nu_new = chi2_of(l23_lm_fits[combo])

    # Raw chi-squared: no worse (1% slack for optimizer wobble)
    assert chi2_new <= chi2_ref*1.01

    # Reduced: no worse beyond the degrees-of-freedom penalty
    n = l23_lm_fits['expb_pow']['prep']['model_Rrs'].size
    k_ref = l23_lm_fits['expb_pow']['ans'].size
    k_new = l23_lm_fits[combo]['ans'].size
    dof_penalty = (n - k_ref)/(n - k_new)
    assert chi2nu_new <= chi2nu_ref*dof_penalty*1.01

    # And the clear-water solution really is the same one: the recovered
    # bb_nw agrees with the single power law it should reduce to
    wave = l23_lm_fits['expb_pow']['models'][0].wave
    bb_ref = l23_lm_fits['expb_pow']['models'][1].eval_bbnw(
        l23_lm_fits['expb_pow']['ans'][3:])[0]
    bb_new = l23_lm_fits[combo]['models'][1].eval_bbnw(
        l23_lm_fits[combo]['ans'][3:])[0]
    for wv_ref in (440., 555., 700.):
        iwv = int(np.argmin(np.abs(wave - wv_ref)))
        assert np.isclose(bb_new[iwv], bb_ref[iwv], rtol=0.1)


# ---------------------------------------------------------------------
# Every model must honour the wave kwarg (Raman excitation grid)
# ---------------------------------------------------------------------

def build_with_params(name):
    """A model of each flavour plus a valid 1-D parameter vector.

    Parameters
    ----------
    name : str
        Model name accepted by bbnw.init_model.

    Returns
    -------
    tuple
        (model, params) with any basis function already set.
    """
    model = bing_bbnw.init_model(name, wave)
    if name == 'Lee':
        model.set_basis_func(1.0)
        params = np.array([-2.0])
    elif name == 'Every':
        # log10 of a power law, so the log-log interpolation has an
        # analytic answer to be checked against
        params = np.log10(0.01*(600./wave)**1.0)
    elif name in ('Pow2', 'Pow2Flat'):
        params = TURBID_PARAMS[name]
    elif name == 'Pow':
        params = np.array([-2.0, 1.0])
    else:
        params = np.array([-2.0])
    return model, params


# Cst is spectrally flat, so it is grid-invariant by construction and
# excluded from the "must differ" assertion.
GRID_MODELS = ['Pow', 'Lee', 'GSM', 'Every', 'Pow2', 'Pow2Flat']


@pytest.mark.parametrize("name", GRID_MODELS)
def test_eval_bbnw_honours_wave(name):
    """Every model must evaluate on the wavelengths it is given.

    Before this fix Lee, GSM and Every ignored the kwarg and returned
    values on self.wave, so eval_bb_ex added pure-water backscattering
    on the *excitation* grid to particle backscattering on the
    *emission* grid.  It was silent because the grids have equal length.
    """
    model, params = build_with_params(name)

    bb_em = model.eval_bbnw(params)
    bb_ex = model.eval_bbnw(params, wave=model.wave_ex)

    # Shape contract, including for the non-parametric Every
    assert bb_em.shape == (1, wave.size)
    assert bb_ex.shape == (1, wave.size)

    # The excitation grid is bluer, so values must actually change
    assert not np.allclose(bb_ex, bb_em)

    # ... and eval_bb_ex must combine both quantities on that grid
    assert np.allclose(model.eval_bb_ex(params),
                       model.bb_w_ex + bb_ex)

    # Chain-shaped params too
    chains = np.vstack([params, params + 0.1])
    assert model.eval_bbnw(chains, wave=model.wave_ex).shape == \
        (2, wave.size)


def test_cst_is_grid_invariant():
    # The flat model is the one legitimate exception: same value on any
    # grid, but it must still return the right shape.
    model, params = build_with_params('Cst')
    bb_ex = model.eval_bbnw(params, wave=model.wave_ex)
    assert bb_ex.shape == (1, wave.size)
    assert np.allclose(bb_ex, model.eval_bbnw(params))
    assert np.allclose(model.eval_bb_ex(params), model.bb_w_ex + bb_ex)


@pytest.mark.parametrize("name,exp_attr", [('Lee', 'Y'), ('GSM', 'eta')])
def test_basis_models_recompute_on_grid(name, exp_attr):
    # The basis models are power laws, so the recomputed basis has an
    # analytic form -- check it rather than just "it differs".
    model, params = build_with_params(name)
    expon = getattr(model, exp_attr)

    for grid in (model.wave, model.wave_ex):
        assert np.allclose(model.eval_basis_func(grid),
                           (model.pivot/grid)**expon)
        assert np.allclose(model.eval_bbnw(params, wave=grid)[0],
                           10**params[-1]*(model.pivot/grid)**expon)

    # The cached attribute is still the native-grid basis (other code
    # and the docstrings refer to it)
    assert np.allclose(model.basis_func, model.eval_basis_func())
    assert np.allclose(model.basis_func, (model.pivot/wave)**expon)


def test_lee_without_Y_raises():
    # Previously this died inside np.outer with an opaque TypeError
    model = bing_bbnw.init_model('Lee', wave)
    assert model.Y is None
    with pytest.raises(ValueError):
        model.eval_bbnw(np.array([-2.0]))


def test_every_interpolation_is_power_law_exact():
    # Every's params ARE bb_nw per channel, so another grid needs an
    # interpolation choice.  We interpolate linearly in log-log space,
    # which is exact for a power law -- including blueward of the model
    # grid, where the excitation wavelengths live.
    model, params = build_with_params('Every')
    assert model.wave_ex.min() < wave.min()      # extrapolating

    bb_ex = model.eval_bbnw(params, wave=model.wave_ex)[0]
    exact = 0.01*(600./model.wave_ex)**1.0
    assert np.allclose(bb_ex, exact, rtol=1e-10)

    # On the native grid it must return the amplitudes untouched
    assert np.allclose(model.eval_bbnw(params, wave=wave)[0], 10**params)
    assert np.allclose(model.eval_bbnw(params)[0], 10**params)


@pytest.mark.parametrize("name,nparam,pnames,log_params,factory", TWO_COMP)
def test_two_comp_log_params_match_priors(name, nparam, pnames, log_params,
                                          factory, two_comp_models):
    # Test 12: the model's log_params must agree with the factory's
    # prior flavors, in both directions.  This is the cheap test that
    # catches contract 1 (p0 log10-ing keyed on flavor) breaking.
    p = factory()
    flavors = [pd['flavor'].startswith('log') for pd in p.bpriors]
    assert flavors == log_params
    assert flavors == two_comp_models[name].log_params
    assert len(flavors) == nparam
    assert p.model_names == ['ExpBricaud', name]

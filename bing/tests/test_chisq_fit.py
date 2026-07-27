""" Tests for the least-squares (chi-squared) fitting module

Currently focused on the ``maxfev`` evaluation budget, which exists for
turbid spectra where the optimizer runs out of function evaluations
before converging.  Everything here runs on a cheap synthetic spectrum,
so no L23 data tree is needed.
"""
import numpy as np

import pytest

from bing.models import utils as model_utils
from bing.parameters import standard
from bing.fitting import chisq_fit
from bing.rt import defs as rt_defs

wave = np.arange(400., 705., 5.)

# ExpBricaud (Adg, Sdg, Aph) + Pow (Bnw, beta)
TRUTH = np.array([-1.0, 0.015, -0.7, -2.0, 1.0])
P0 = np.array([-0.5, 0.013, -0.4, -1.5, 0.6])


@pytest.fixture(scope="module")
def synthetic():
    """A noiseless synthetic Rrs plus everything needed to refit it.

    Returns
    -------
    dict
        models, rt_dict, bounds, Rrs and varRrs for a single spectrum.
        variable_Gordon is off so the plain elastic Gordon relation is
        used (the wavelength-dependent coefficients would need G1/G2
        seeded on the absorption model).
    """
    p = standard.expb_pow(wv_min=400., wv_max=700.,
                          variable_Gordon=False)
    models = model_utils.init(p.model_names, wave, (p.apriors, p.bpriors))
    models[0].set_aph(np.array([1.0]))
    rt_dict = rt_defs.rt_dict_from_p(p)

    Rrs = chisq_fit.fit_func(wave, *TRUTH, models=models,
                             rt_dict=rt_dict)
    low = np.array([d['pmin'] for d in p.apriors] +
                   [d['pmin'] for d in p.bpriors], dtype=float)
    high = np.array([d['pmax'] for d in p.apriors] +
                    [d['pmax'] for d in p.bpriors], dtype=float)
    return dict(models=models, rt_dict=rt_dict, bounds=(low, high),
                Rrs=Rrs, varRrs=(0.02*Rrs)**2)


def run_fit(syn, bounds, **kwargs):
    """Fit the synthetic spectrum, optionally passing maxfev."""
    items = (syn['Rrs'], syn['varRrs'], P0.copy(), 0)
    return chisq_fit.fit(items, syn['models'], syn['rt_dict'],
                         bounds=bounds, **kwargs)


def test_maxfev_exhausted_bounded(synthetic):
    # Bounded => curve_fit uses least_squares ('trf'), which takes
    # max_nfev; scipy renames our maxfev for us.  A tiny budget must
    # surface as RuntimeError rather than a silently unconverged answer.
    with pytest.raises(RuntimeError):
        run_fit(synthetic, synthetic['bounds'], maxfev=2)


def test_maxfev_exhausted_unbounded(synthetic):
    # Unbounded => the 'lm' back end, where the keyword goes to leastsq
    # under its original name.  Both paths must honour it.
    with pytest.raises(RuntimeError):
        run_fit(synthetic, None, maxfev=2)


def test_maxfev_generous_succeeds(synthetic):
    # A generous budget converges, and recovers the truth
    ans, cov, idx = run_fit(synthetic, synthetic['bounds'], maxfev=40000)
    assert np.allclose(ans, TRUTH, atol=1e-3)
    assert cov.shape == (TRUTH.size, TRUTH.size)
    assert idx == 0


def test_maxfev_none_matches_generous(synthetic):
    # The kwarg must be inert when the budget is ample: omitting it and
    # passing a large value give the same answer, i.e. we are not
    # perturbing scipy's defaults for existing callers.
    ref, _, _ = run_fit(synthetic, synthetic['bounds'])
    big, _, _ = run_fit(synthetic, synthetic['bounds'], maxfev=40000)
    assert np.allclose(ref, big, rtol=1e-10)

    # ... in the unbounded branch too
    ref_u, _, _ = run_fit(synthetic, None)
    big_u, _, _ = run_fit(synthetic, None, maxfev=40000)
    assert np.allclose(ref_u, big_u, rtol=1e-10)


def test_maxfev_budget_is_the_binding_constraint(synthetic):
    # The point of the kwarg: a spectrum that fails on a small budget
    # succeeds on a larger one, with nothing else changed.  Find the
    # smallest budget that works, and confirm a smaller one fails.
    ok = None
    for budget in (10, 30, 100, 300, 1000):
        try:
            run_fit(synthetic, synthetic['bounds'], maxfev=budget)
        except RuntimeError:
            continue
        ok = budget
        break
    assert ok is not None, "no budget in the sweep converged"
    assert ok > 10, "the sweep never exercised an exhausted budget"

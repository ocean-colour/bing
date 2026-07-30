""" Tests for display helpers in bing.plotting

Focused on log_param_mask, which decides what the figures exponentiate
and how they label axes.  Getting it wrong is silent: a linear slope
printed as 10**slope looks like a plausible number.
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')   # before bing.plotting imports pyplot

import pytest

from bing import plotting
from bing.models import anw as bing_anw
from bing.models import bbnw as bing_bbnw
from bing.models import utils as model_utils
from bing.parameters import standard
from bing.priors import priors as bing_priors

wave = np.arange(400., 705., 5.)

# Every combo in standard.py
COMBOS = ['expb_pow', 'expbf_pow', 'giop', 'gsm', 'k2b',
          'expb_powflex', 'expb_pow2', 'expb_pow2flat']


class FakeModel:
    """A model that predates log_params, to exercise the fallback."""
    nparam = 3
    pnames = ['a', 'b', 'c']


def test_mask_falls_back_to_all_log():
    # Anything that does not declare log_params keeps the historical
    # behaviour: treat every parameter as a log10 amplitude
    assert plotting.log_param_mask(FakeModel()) == [True, True, True]
    assert plotting.log_param_mask([FakeModel()]) == [True]*3

    # An explicit None means the same thing
    fake = FakeModel()
    fake.log_params = None
    assert plotting.log_param_mask(fake) == [True]*3


def test_mask_reads_declared_values():
    pow_model = bing_bbnw.init_model('Pow', wave)
    assert plotting.log_param_mask(pow_model) == [True, False]

    pow2 = bing_bbnw.init_model('Pow2', wave)
    assert plotting.log_param_mask(pow2) == [True, False, True, False]

    # Concatenates across models, in parameter order
    anw = bing_anw.init_model('ExpBricaud', wave)
    assert plotting.log_param_mask([anw, pow2]) == \
        [True, False, True] + [True, False, True, False]

    # A bare model and a one-element list agree
    assert plotting.log_param_mask(pow2) == \
        plotting.log_param_mask([pow2])


@pytest.mark.parametrize("name", ['Cst', 'Pow', 'Lee', 'GSM', 'Every',
                                  'Pow2', 'Pow2Flat'])
def test_bb_log_params_length(name):
    # A declared mask must cover exactly the parameters
    model = bing_bbnw.init_model(name, wave)
    assert len(plotting.log_param_mask(model)) == model.nparam
    if model.log_params is not None:
        assert len(model.log_params) == model.nparam
        assert all(isinstance(v, bool) for v in model.log_params)


@pytest.mark.parametrize("combo", COMBOS)
def test_mask_agrees_with_prior_flavors(combo):
    """The display convention must agree with the fitting convention.

    The fitters decide which p0 slots to log10 from the prior *flavor*
    (log_uniform -> log10); the figures decide what to exponentiate from
    log_params.  If those two ever disagree, one of them is lying about
    the same parameter.  Checked for every standard combo.

    (Not checked for Chase2017, which is deliberately outside this
    convention: all 28 of its parameters are log10 quantities but its
    priors are declared 'uniform' with log10 bounds, and its init_guess
    returns log10 directly.  It declares no log_params, so it falls back
    to all-log10, which is correct for it.)
    """
    p = getattr(standard, combo)()
    models = model_utils.init(p.model_names, wave)
    bing_priors.set_standard_priors(models, p)

    from_flavor = []
    for model in models:
        for prior in model.priors.priors[:model.nparam]:
            from_flavor.append(str(prior.flavor)[:3] == 'log')

    assert plotting.log_param_mask(models) == from_flavor


def test_corner_plot_labels_only_log_params():
    """Render a corner plot and read the labels back off the axes.

    With show_log=True the log10 amplitudes carry a log10(...) label and
    the linear exponents do not; with show_log=False nothing does,
    because only the log columns were exponentiated.
    """
    from matplotlib import pyplot as plt

    p = standard.expb_pow2()
    models = model_utils.init(p.model_names, wave,
                              (p.apriors, p.bpriors))
    truth = np.array([-1.0, 0.015, -0.7, -1.3, 0.0, -1.0, 1.8])
    rng = np.random.default_rng(7)
    # corner_plot burns the first 7000 steps internally
    chains = truth + 0.01*rng.standard_normal((7200, 16, truth.size))

    n = truth.size
    for show_log, n_expected in ((True, 4), (False, 0)):
        fig = plotting.corner_plot(chains, models=models,
                                   show_log=show_log)
        axes = np.array(fig.axes).reshape(n, n)
        labels = [axes[n-1, kk].get_xlabel() for kk in range(n)]
        logged = [r'\log_{10}' in lb for lb in labels]
        assert sum(logged) == n_expected
        if show_log:
            # Adg, Aph, Bmin, Borg are log; Sdg, eta_min, eta_org are not
            assert logged == [True, False, True, True, False, True,
                              False]
            assert labels[1] == 'Sdg'
            assert labels[-1] == 'eta_org'
        plt.close(fig)


def test_show_params_text_uses_the_mask():
    """The printed value must be 10**p for amplitudes and p for slopes.

    show_fits needs a full fit to run, so this checks the formatting
    rule directly against the mask rather than rendering a figure.
    """
    models = [bing_anw.init_model('ExpBricaud', wave),
              bing_bbnw.init_model('Pow', wave)]
    params = np.array([-1.0, 0.015, -0.7, -2.0, 1.0])
    is_log = plotting.log_param_mask(models)
    assert is_log == [True, False, True, True, False]

    shown = []
    ip = 0
    for model in models:
        for ss in range(model.nparam):
            val = (f'{10**params[ip]:.2f}' if is_log[ip]
                   else f'{params[ip]:.3f}')
            shown.append(f'{model.pnames[ss]} = {val}')
            ip += 1

    # Sdg and beta keep their linear values ...
    assert 'Sdg = 0.015' in shown
    assert 'beta = 1.000' in shown
    # ... and would have been badly wrong under the old all-log rule
    assert not np.isclose(10**0.015, 0.015)
    # ... while amplitudes are still exponentiated
    assert 'Adg = 0.10' in shown

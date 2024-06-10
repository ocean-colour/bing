""" Tests for phytoplankton """""
import os

import numpy as np

import pytest

from big.models import anw as big_anw

from IPython import embed

wave = np.arange(350, 755, 5.)

def test_init():
    # Wavelengths from 350 to 755 in steps of 5
    anwExp = big_anw.aNWExp(wave, 'log')

    # Check that a_w is set
    assert anwExp.a_w is not None
    # Check a value too
    assert np.isclose(anwExp.a_w[0], 0.0071, atol=1e-5)
    #pytest.set_trace()

def test_eval():
    anwExp = big_anw.aNWExp(wave, 'log')

    # Evaluatee a_nw on a flat array
    #   The code always returns a multi-dimensional array
    a_nw = anwExp.eval_anw(np.array([-1., np.log10(0.015)]))
    assert np.isclose(a_nw[0][0], 0.2117, atol=1e-5)

    # Now a 2D param array
    a_nw = anwExp.eval_anw(np.array([[-1., np.log10(0.015)], 
                                     [-1., np.log10(0.015)]]))
    assert a_nw.shape == (2, wave.size)
    assert np.isclose(a_nw[1][0], 0.2117, atol=1e-5)

def test_priors():
    anwExp = big_anw.aNWExp(wave, 'log')

    # Check priors
    assert anwExp.priors is not None
    assert anwExp.priors.approach == 'log'
    assert anwExp.priors.nparam == 2
    assert anwExp.priors.priors.shape == (2,2)
    assert np.all(anwExp.priors.priors[:,0] == -6)
    assert np.all(anwExp.priors.priors[:,1] == 5)
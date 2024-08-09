""" Tests for phytoplankton """""
import os

import numpy as np

import pathlib
import pytest

from bing.models import anw as bing_anw
from bing.models import bbnw as bing_bbnw
from bing import inference as bing_inf

from IPython import embed

def data_path(filename):
    data_dir = pathlib.Path(__file__).parent.absolute().joinpath('files')
    return str(data_dir.joinpath(filename).resolve())


def test_fit():
    # Prep
    idx = 170
    fit_file = data_path('fit_one.npz')
    d = np.load(fit_file)
    wave = d['wave']
    gordon_Rrs = d['gordon_Rrs']
    varRrs = d['varRrs']
    p0 = d['p0']
    
    items = [(gordon_Rrs, varRrs, p0, idx)]

    model_names = ['Exp', 'Pow']


    # Load models
    anw_model = bing_anw.init_model(model_names[0], wave)
    bbnw_model = bing_bbnw.init_model(model_names[1], wave)
    models = [anw_model, bbnw_model]


    # Initialize the MCMC
    nsteps:int=1000 
    nburn:int=100
    pdict = bing_inf.init_mcmc(models, nsteps=nsteps, nburn=nburn)

    chains, new_idx = bing_inf.fit_one(items[0], models=models, pdict=pdict, chains_only=True)

    # Test
    assert chains.shape == (nsteps, 16, 4)
    assert new_idx == 170
""" Tests for phytoplankton """""
import os

import numpy as np

import pathlib
import pytest

from bing.fitting import l23 as fit_l23
from bing.parameters import standard

from IPython import embed

def data_path(filename):
    data_dir = pathlib.Path(__file__).parent.absolute().joinpath('files')
    return str(data_dir.joinpath(filename).resolve())


def test_single_fit():
    idx = 2773
    p_expb = standard.expb_pow(satellite='SBG', add_noise=True)
    outfile = fit_l23.chain_filename(p_expb, idx=idx, path='./')
    chains, models, prep_dict, idx, extras = fit_l23.fit_one(p_expb, idx)
    fit_l23.save_chains(chains, idx, outfile, extras=extras)
    # Clean up
    os.remove(outfile)
""" Simple utility functions for models """

import numpy as np

from boring.models import anw as boring_anw
from boring.models import bbnw as boring_bbnw

def init(model_names:list, model_wave:np.ndarray):
    anw_model = boring_anw.init_model(model_names[0], model_wave)
    bbnw_model = boring_bbnw.init_model(model_names[1], model_wave)
    models = [anw_model, bbnw_model]

    return models
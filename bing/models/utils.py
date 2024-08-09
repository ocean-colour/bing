""" Simple utility functions for models """

import numpy as np

from bing.models import anw as bing_anw
from bing.models import bbnw as bing_bbnw

import numpy as np

def init(model_names: list, model_wave: np.ndarray,
         prior_dicts: tuple = (None, None)) -> list:
    """
    Initialize models with given model names, model wave, and prior dictionaries.

    Args:
        model_names (list): A list of model names.
        model_wave (np.ndarray): An array representing the model wave.
        prior_dicts (tuple, optional): A tuple of prior dictionaries. Defaults to (None, None).

    Returns:
        list: A list of initialized models.
    """
    anw_model = bing_anw.init_model(model_names[0], model_wave,
                                      prior_dicts[0])
    bbnw_model = bing_bbnw.init_model(model_names[1], model_wave,
                                        prior_dicts[1])
    models = [anw_model, bbnw_model]

    return models
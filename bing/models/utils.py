""" Simple utility functions for models """

import numpy as np

from ocpy.chl import band_ratios
from ocpy.iop import zlee

from bing.models import anw as bing_anw
from bing.models import bbnw as bing_bbnw

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

def init_other_bits(models:list, Chl:float=None, Y:float=None,
                    Rrs:np.ndarray=None, update_dict:dict=None,
                    verbose:bool=False) -> dict:
    """
    Initialize other bits for the models.

    Args:
        models (list): A list of models.
        Chl (float, optional): The chlorophyll value. Defaults to None.
        Y (float, optional): The Y value. Defaults to None.
        Rrs (np.ndarray, optional): The Rrs values. Defaults to None.
        update_dict (dict, optional): A dictionary to be updated. Defaults to None.
        verbose (bool, optional): Flag to show verbose output. Defaults to False.

    Returns:
        dict: A dictionary of any updated items
    """
    ret_items = {}

    # Internals (some of which depend on Rrs)
    if models[0].uses_Chl:
        if models[0].name == 'GIOP' and Rrs is not None:
            # Calculate Chl from Rrs
            OC_Chl = band_ratios.oc4(models[0].wave, Rrs)
            Chl = OC_Chl
            if update_dict is not None:
                if verbose:
                    print(f'Using Chl = {OC_Chl} instead of {update_dict["Chl"]}')
                update_dict['Chl'] = Chl
            ret_items['Chl'] = Chl
        #
        models[0].set_aph(Chl)

    if models[1].uses_basis_params:  # Lee
        # GIOP?
        if models[0].name == 'GIOP' and Rrs is not None:
            Y = zlee.Y_from_Rrs(models[1].wave, Rrs)
            if update_dict is not None:
                if verbose:
                    print(f'Using Y = {Y} instead of {update_dict["Y"]}')
                update_dict['Y'] = Y
            ret_items['Y'] = Y
        # Go forth
        models[1].set_basis_func(Y)

    # Return
    return ret_items
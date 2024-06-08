""" Classes and methods to deal with priors """

import numpy as np

from abc import ABCMeta

class Prior:
    """
    Abstract base class for a prior

    Attributes:

    """
    __metaclass__ = ABCMeta

    flavor:str = None
    """
    Approach to the prior
    """

    def __init__(self, pdict:dict):
        self.init_from_dict(pdict)

    def init_from_dict(self, pdict:dict):
        """
        Initialize the prior from a dictionary

        Args:
            pdict (dict): The dictionary containing the prior information
        """

class UniformPrior(Prior):
    """
    Class for a uniform prior

    Attributes:

    """
    flavor:str = 'uniform'
    """
    Approach to the prior
    """

    pmn:float = None
    """
    Minimum value for the prior
    """

    pmx:float = None
    """
    Maximum value for the prior
    """
    def __init__(self, pdict:dict):
        Prior.__init__(self, pdict)

    def init_from_dict(self, pdict:dict):
        """
        Initialize the prior from a dictionary

        Args:
            pdict (dict): The dictionary containing the prior information
        """
        self.pmin = pdict['pmin']
        self.pmax = pdict['pmax']

    def calc(self, params:np.ndarray):
        """
        Calculate the prior for the parameters

        Args:
            params (np.ndarray): The parameters

        Returns:
            bool: True if the parameters are within the prior, False otherwise
        """
        if (params[0] < self.pmin) or (params[1] > self.pmax):
            return -np.inf
        else:
            return 0

class Priors:

    nparam:int = None
    """
    The number of parameters for the model
    """

    priors:list = None
    """
    The priors for the model
    """

    def __init__(self, pdicts:list):
        self.nparam = len(pdicts)
        self.set_priors(pdicts)

    def set_priors(self, pdicts):
        """
        Set the priors for the model

        """
        for pdict in pdicts:
            if pdict['flavor'] == 'uniform':
                self.priors = UniformPrior(pdict)
            else:
                raise ValueError(f"Unknown prior flavor: {pdict['flavor']}")

    def calc(self, params:np.ndarray):
        for kk,prior in enumerate(self.priors):
            return prior.calc(params[kk])
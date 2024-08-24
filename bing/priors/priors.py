""" Classes and methods to deal with priors """

import numpy as np

from abc import ABCMeta

default = dict(flavor='uniform', pmin=-6, pmax=5)

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

    def __repr__(self):
        return f"<Prior: {self.flavor}>"


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

    def calc(self, param:float):
        """
        Calculate the prior for the parameters

        Args:
            params (np.ndarray): The parameters

        Returns:
            bool: True if the parameters are within the prior, False otherwise
        """
        if (param < self.pmin) or (param > self.pmax):
            return -np.inf
        else:
            return 0


    def __repr__(self):
        return f"<Prior: {self.flavor}, pmin={self.pmin:0.3f}, pmax={self.pmax:0.3f} >"


class Priors:

    nparam:int = None
    """
    The number of parameters for the model
    """

    priors:list = None
    """
    The priors for the model
    """

    pdicts:list = None
    """
    The prior dictionaries
    """

    def __init__(self, pdicts:list):
        self.nparam = len(pdicts)
        self.set_priors(pdicts)

    def set_priors(self, pdicts):
        """
        Set the priors for the model

        """
        self.priors = []
        for pdict in pdicts:
            if pdict['flavor'] == 'uniform':
                self.priors.append(UniformPrior(pdict))
            else:
                raise ValueError(f"Unknown prior flavor: {pdict['flavor']}")

    def calc(self, params:np.ndarray):
        prior_sum = 0.
        for kk,prior in enumerate(self.priors):
            prior_sum += prior.calc(params[kk])
        #
        return prior_sum

    def gen_bounds(self):
        """
        Generate the bounds for the prior

        Returns:
            tuple: A tuple containing the minimum and maximum values for the prior
        """
        pmins = []
        pmaxs = []
        for kk,prior in enumerate(self.priors):
            if prior.flavor == 'uniform':
                pmins.append(prior.pmin)
                pmaxs.append(prior.pmax)
            else:
                raise ValueError(f"Unknown prior flavor: {prior.flavor}")
        # Return
        return np.array(pmins), np.array(pmaxs)


    def __repr__(self):
        rstr =  "<Priors: \n"
        for prior in self.priors:
            rstr += f"  {prior}\n"
        rstr += ">"
        return rstr
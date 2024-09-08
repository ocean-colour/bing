""" Classes and methods to deal with priors """

import numpy as np

from abc import ABCMeta

default = dict(flavor='log_uniform', pmin=-6, pmax=5)

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

    pmin:float = None
    """
    Minimum value for the prior
    """

    pmax:float = None
    """
    Maximum value for the prior
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


class GaussianPrior(Prior):
    """
    Class for a uniform prior

    Attributes:

    """
    flavor:str = 'guassian'
    """
    Approach to the prior
    """

    mean:float = None
    """
    The mean value for the prior
    """

    sigma:float = None
    """
    The standard deviation for the prior
    """

    def __init__(self, pdict:dict):
        Prior.__init__(self, pdict)

    def init_from_dict(self, pdict:dict):
        """
        Initialize the prior from a dictionary

        Args:
            pdict (dict): The dictionary containing the prior information
                mean (float): The mean value for the prior
                sigma (float): The standard deviation for the prior
        """
        # Optional
        if 'pmin' in pdict:
            self.pmin = pdict['pmin']
        if 'pmax' in pdict:
            self.pmax = pdict['pmax']
        # Requred
        self.mean = pdict['mean']
        self.sigma = pdict['sigma']

    def calc(self, param:float):
        """
        Calculate the prior for the parameters

        Args:
            params (np.ndarray): The parameters

        Returns:
            bool: True if the parameters are within the prior, False otherwise
        """
        # Optional
        if self.pmin is not None and param < self.pmin:
            return -np.inf
        if self.pmax is not None and param > self.pmax:
            return -np.inf

        # Required
        return -0.5*((param - self.mean)/self.sigma)**2


    def __repr__(self):
        return f"<Prior: {self.flavor}, mean={self.mean:0.3f}, sigma={self.sigma:0.3f} >"

class LogUniformPrior(Prior):
    """
    Class for a uniform prior

    Attributes:

    """
    flavor:str = 'log_uniform'
    """
    Approach to the prior
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

class UniformPrior(LogUniformPrior):
    """
    Class for a uniform prior

    This is identical to a log_uniform prior except
    that the prior is linear space rather than log space

    Attributes:

    """
    flavor:str = 'uniform'
    """
    Approach to the prior
    """

    def __init__(self, pdict:dict):
        Prior.__init__(self, pdict)

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
            if pdict['flavor'] == 'log_uniform':
                self.priors.append(LogUniformPrior(pdict))
            elif pdict['flavor'] == 'uniform':
                self.priors.append(UniformPrior(pdict))
            elif pdict['flavor'] == 'gaussian':
                self.priors.append(GaussianPrior(pdict))
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
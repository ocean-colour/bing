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
    Class for a Gaussian prior

    Attributes:

    """
    flavor:str = 'gaussian'
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



class RatioPrior(Prior):
    """
    Class for a Ratio prior, e.g. CDOM/aph

    """
    flavor:str = 'ratio'
    """
    Approach to the prior
    """

    ratio:float = None
    """
    The expected ratio
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
                ratio (float): The mean value for the prior
                sigma (float): The standard deviation for the prior
                i0 (int): Index of the first parameter in the parameter array
                i1 (int): Index of the second parameter in the parameter array
        """
        # Required
        self.ratio = pdict['ratio']
        self.sigma = pdict['sigma']
        self.i0 = pdict['i0']
        self.i1 = pdict['i1']

    def calc(self, params:np.ndarray):
        """
        Calculate the prior for the parameters

        Args:
            params (np.ndarray): The parameters

        Returns:
            float: prior
        """
        # Calculate ratio (assumed log10)
        p0 = 10**(params[self.i0])
        p1 = 10**(params[self.i1])
        pred_0 = self.ratio*p1

        # Required
        return -0.5*((pred_0 - p0)/(self.sigma*p0))**2


    def __repr__(self):
        return f"<Prior: {self.flavor}, mean={self.ratio:0.3f}, sigma={self.sigma:0.3f} >"


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

    def add_prior(self, pdict):
        if pdict['flavor'] == 'log_uniform':
            self.priors.append(LogUniformPrior(pdict))
        elif pdict['flavor'] == 'uniform':
            self.priors.append(UniformPrior(pdict))
        elif pdict['flavor'] == 'gaussian':
            self.priors.append(GaussianPrior(pdict))
        elif pdict['flavor'] == 'ratio':
            self.priors.append(RatioPrior(pdict))
        else:
            raise ValueError(f"Unknown prior flavor: {pdict['flavor']}")

    def set_priors(self, pdicts):
        """
        Set the priors for the model

        """
        self.priors = []
        for pdict in pdicts:
            self.add_prior(pdict)

    def calc(self, params:np.ndarray):
        prior_sum = 0.

        # Individual priors
        for kk,param in enumerate(params):
            prior_sum += self.priors[kk].calc(param)

        # Extras
        if len(self.priors) > params.size:
            for kk in range(params.size, len(self.priors)):
                prior = self.priors[kk]
                if getattr(prior, "flavor", None) == "ratio":
                    prior_sum += prior.calc(params)
                else:
                    raise TypeError(f"Prior at index {kk} with flavor '{getattr(prior, 'flavor', None)}' does not support full params array input.")
        
        # Return
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


def set_standard_priors(models, p):
    """
    Set standard priors for the given models based on the provided parameters.

    This function configures the prior distributions for the parameters of 
    the models. It supports special cases for specific model names and 
    parameter configurations, and allows customization of priors through 
    the `p` object.

    Args:
        models (list): A list of model objects. Each model should have attributes 
            `nparam` (number of parameters), `name` (model name), and `pnames` 
            (list of parameter names).
        p (object): An object containing prior configuration attributes:
            - `apriors` (list or None): Custom priors for the first model.
            - `bpriors` (list or None): Custom priors for the second model.
            - `model_names` (list): Names of the models.
            - `beta` (float or None): Value for the beta parameter in the second model.
            - `set_Sdg` (bool): Whether to set the Sdg parameter.
            - `Sdg` (float): Mean value for the Sdg parameter.
            - `sSdg` (float): Standard deviation for the Sdg parameter.

    Returns:
        None: The function modifies the `priors` attribute of the models in-place.

    Notes:
        - For the first model (`jj == 0`), if its name is 'ExpBricaud', a specific 
          prior is set for the second parameter.
        - For the second model (`jj == 1`), if its name is 'Pow' and `p.beta` is 
          provided, a Gaussian prior is set for the second parameter.
        - If `p.set_Sdg` is True, a Gaussian prior is set for the 'Sdg' parameter 
          in the first model.
    """

    # Set priors
    prior_dict = dict(flavor='log_uniform', pmin=-6, pmax=5)
    # Loop on a, bb
    for jj in range(2):
        prior_dicts = [prior_dict]*models[jj].nparam
        # Special cases
        if jj == 0 and p.apriors is not None:
            prior_dicts = p.apriors
        elif jj == 1 and p.bpriors is not None:
            prior_dicts = p.bpriors
        elif jj == 0 and models[0].name == 'ExpBricaud':
            prior_dicts[1] = dict(flavor='log_uniform',
                                pmin=np.log10(0.007),
                                pmax=np.log10(0.02))
        elif jj == 1 and p.model_names[1] == 'Pow' and \
            p.beta is not None:
            prior_dicts[1] = dict(flavor='gaussian',
                                mean=p.beta, sigma=0.1)

        # Sdg
        if p.set_Sdg and jj==0:
            print(f"Using Sdg = {p.Sdg}")
            # Find Sdg
            ii = models[0].pnames.index('Sdg')
            prior_dicts[ii] = dict(flavor='gaussian',
                                mean=p.Sdg, sigma=p.sSdg)
        # Finish
        models[jj].priors = Priors(prior_dicts)
        # Validate the count where the model knows how (bb models).
        # Priors are attached here, after construction, so the model's
        # own constructor check never sees this path.  Extra priors
        # (othera_priors) are appended by the caller afterwards.
        if hasattr(models[jj], 'check_priors'):
            models[jj].check_priors()


def priors_from_models(models):
    """Extract prior dicts from a pair of models for serialisation.

    Walks ``models[i].priors.priors`` and returns a flat list of plain
    dicts (absorption first, then backscattering).  The order matches
    ``models[0].pnames + models[1].pnames`` so the saved list can be
    split back later with :func:`split_priors`.

    If a model has no priors attached, the BING default prior is used as
    a fallback so the saved fit can still be reloaded.
    """
    pdicts = []
    for model in models:
        if model.priors is None:
            # Fall back to BING's default prior so the saved fit can be
            # reloaded even if the caller never set priors.
            pdicts.extend([dict(default)] * model.nparam)
            continue
        for prior in model.priors.priors:
            pdict = {"flavor": prior.flavor}
            # Range bounds are present on all uniform / log-uniform
            # priors and optionally on Gaussian priors.
            if getattr(prior, "pmin", None) is not None:
                pdict["pmin"] = float(prior.pmin)
            if getattr(prior, "pmax", None) is not None:
                pdict["pmax"] = float(prior.pmax)
            # Gaussian-specific fields
            if getattr(prior, "mean", None) is not None:
                pdict["mean"] = float(prior.mean)
            if getattr(prior, "sigma", None) is not None:
                pdict["sigma"] = float(prior.sigma)
            # Ratio-specific fields
            if getattr(prior, "ratio", None) is not None:
                pdict["ratio"] = float(prior.ratio)
                pdict["i0"] = int(prior.i0)
                pdict["i1"] = int(prior.i1)
            pdicts.append(pdict)
    return pdicts


def split_priors(prior_dicts, models):
    """Split a flat prior list back into ``[a_priors, b_priors]``.

    Uses ``models[0].nparam`` to know where the absorption priors end.
    Inverse of :func:`priors_from_models`.
    """
    n_a = models[0].nparam
    return prior_dicts[:n_a], prior_dicts[n_a:]
""" Models for non-water backscattering """
import numpy as np

from scipy.interpolate import interp1d

from oceancolor.water import scattering as water_bb
from oceancolor.hydrolight import loisel23

from abc import ABCMeta

from boring.models import functions
from boring import priors as boring_priors

def init_model(model_name:str, wave:np.ndarray, prior_dicts:list=None):
    """
    Initialize a model for non-water absorption

    Args:
        model_name (str): The name of the model
        wave (np.ndarray): The wavelengths
        prior_choice (str): The choice of priors

    Returns:
        bbNWModel: The model
    """
    model_dict = {'Cst': bbNWCst, 'Pow': bbNWPow, 'Lee': bbNWLee}
    if model_name not in model_dict.keys():
        raise ValueError(f"Unknown model: {model_name}")
    else:
        return model_dict[model_name](wave, prior_dicts)

class bbNWModel:
    """
    Abstract base class for non-water backscattering

    Attributes:

    """
    __metaclass__ = ABCMeta

    name:str = None
    """
    The name of the model
    """

    nparam:int = None
    """
    The number of parameters for the model
    """

    bb_w:np.ndarray = None
    """
    The backscattering of water
    """

    priors:boring_priors.Priors = None
    """
    The priors for the model
    """

    basis_func:np.ndarray = None
    """
    The basis function for the model
    """

    uses_basis_params:bool = False
    """
    Whether the model uses basis parameters
    """

    def __init__(self, wave:np.ndarray, prior_dicts:list):
        self.wave = wave
        self.internals = {}

        # Initialize water
        self.init_bbw()

        # Set priors
        if prior_dicts is not None:
            self.priors = boring_priors.Priors(prior_dicts)

    def init_bbw(self):
        """
        Initialize the backscattering coefficient of water

        Args:

        Returns:
            np.ndarray: The absorption coefficient of water
        """
        # TODO -- replace this with a proper calculation!
        #_, _, b_w = water_bb.betasw_ZHH2009(
        #    self.wave, 20, [0], 33)
        #self.bb_w = b_w/2.
        idx = 0
        ds = loisel23.load_ds(4,0)
        wave = ds.Lambda.data
        bbw = ds.bb.data[idx,:]-ds.bbnw.data[idx,:]
        # Interpolate
        f = interp1d(wave, bbw, kind='linear', fill_value='extrapolate')
        self.bb_w = f(self.wave)
        
    def eval_bbnw(self, params:np.ndarray):
        """
        Evaluate the non-water backscattering coefficients

        Parameters:
            params (np.ndarray): The parameters for the model

            Pow:
                params[0] = log10(Bnw)
                params[1] = log10(beta)
            Cst:
                params[0] = log10(Bnw)

        Returns:
            np.ndarray: The non-water backscattering coefficient
        """
        if self.name == 'Pow':
            return functions.powerlaw(self.wave, params, pivot=self.pivot)
        elif self.name == 'Cst':
            return functions.constant(self.wave, params)
        elif self.name == 'Lee':
            return functions.gen_basis(params[...,-1:], [self.basis_func])
        else:
            raise ValueError(f"Unknown model: {self.name}")


    def eval_bb(self, params:np.ndarray):
        """
        Evaluate the absorption coefficient

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The absorption coefficient
        """
        return self.bb_w + self.eval_bbnw(params)

    def init_guess(self, bb_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            bb_nw (np.ndarray): The non-water absorption coefficient
        """

    def set_basis_func(self, param:float):
        """
        Set the basis function for the model

        Parameters:
            param (float): The basis function
        """


class bbNWCst(bbNWModel):
    """
    Constant model for non-water scattering
        Bnw

    Attributes:

    """
    name = 'Cst'
    nparam = 1

    def __init__(self, wave:np.ndarray, prior_dicts:list):
        bbNWModel.__init__(self, wave, prior_dicts)

    def init_guess(self, bb_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            bb_nw (np.ndarray): The non-water scattering coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        i600 = np.argmin(np.abs(self.wave-600))
        p0_bb = np.array([bb_nw[i600]])
        # Return
        assert p0_bb.size == self.nparam
        return p0_bb
        
class bbNWPow(bbNWModel):
    """
    Power-law model for non-water backscattering
        bb_nw = Bnw * (600/wave)^beta

    Attributes:

    """
    name = 'Pow'
    nparam = 2
    pivot = 600.

    def __init__(self, wave:np.ndarray, prior_dicts:list):
        bbNWModel.__init__(self, wave, prior_dicts)

    def init_guess(self, bb_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        i600 = np.argmin(np.abs(self.wave-self.pivot))
        p0_bb = np.array([bb_nw[i600], 1.])
        assert p0_bb.size == self.nparam

        # Return
        return p0_bb 

class bbNWLee(bbNWModel):
    """
    Power-law model for non-water backscattering

        bb_nw = Bnw * (600/wave)^Y

     with Y calculated from Lee+2002

        Y = 2.2 * (1 - 1.2 * np.exp(-0.9 * rrs[440]/rrs[555]))

    Attributes:

    """
    name = 'Lee'
    nparam = 1
    pivot = 600.
    uses_basis_params = True

    def __init__(self, wave:np.ndarray, prior_dicts:list):
        bbNWModel.__init__(self, wave, prior_dicts)

        # Lee+2002
        self.Y = None

    def set_basis_func(self, Y:float):
        self.Y = Y
        self.basis_func = (self.pivot/self.wave)**self.Y

    def init_guess(self, bb_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        i600 = np.argmin(np.abs(self.wave-self.pivot))
        p0_bb = np.array([bb_nw[i600]])
        assert p0_bb.size == self.nparam

        # Return
        return p0_bb 
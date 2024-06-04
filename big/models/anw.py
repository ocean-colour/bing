""" Models for non-water absorption """
import numpy as np

from abc import ABCMeta

from oceancolor.water import absorption as water_abs
from big import priors as big_priors

def init_model(model_name:str, wave:np.ndarray, prior_choice:str):
    """
    Initialize a model for non-water absorption

    Args:
        model_name (str): The name of the model
        wave (np.ndarray): The wavelengths
        prior_choice (str): The choice of priors

    Returns:
        aNWModel: The model
    """
    if model_name == 'Exp':
        return aNWExp(wave, prior_choice)
    else:
        raise ValueError(f"Unknown model: {model_name}")

class aNWModel:
    """
    Abstract base class for non-water absoprtion

    Attributes:

    """
    __metaclass__ = ABCMeta

    name = None
    """
    The name of the model
    """

    nparam:int = None
    """
    The number of parameters for the model
    """

    a_w:np.ndarray = None
    """
    The absorption coefficient of water
    """

    prior_approach:str = None
    """
    Approach to priors
    """

    priors:big_priors.Priors = None
    """
    The priors for the model
    """

    def __init__(self, wave:np.ndarray, prior_choice:str):
        self.wave = wave
        self.internals = {}

        # Initialize water
        self.init_aw()

        # Set priors
        self.priors = big_priors.Priors(prior_choice, self.nparam)

    def init_aw(self, data:str='IOCCG'):
        """
        Initialize the absorption coefficient of water

        Args:
            data (str, optional): The data source to use. Defaults to 'IOCCG'.

        Returns:
            np.ndarray: The absorption coefficient of water
        """
        self.a_w = water_abs.a_water(self.wave, data=data)

    def eval_anw(self, params:np.ndarray):
        """
        Evaluate the non-water absorption coefficient

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The non-water absorption coefficient
                This is always a multi-dimensional array
        """

    def eval_a(self, params:np.ndarray):
        """
        Evaluate the absorption coefficient

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The absorption coefficient
        """
        return self.a_w + self.eval_anw(params)

    def init_guess(self, a_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient
        """

        
class aNWExp(aNWModel):
    """
    Exponential model for non-water absorption
        Anw * exp(-Snw*(wave-400))

    Attributes:

    """
    name = 'Exp'
    nparam = 2

    def __init__(self, wave:np.ndarray, prior_choice:str):
        aNWModel.__init__(self, wave, prior_choice)

    def eval_anw(self, params:np.ndarray):
        """
        Evaluate the model

        Parameters:
            params (np.ndarray): The parameters for the model
                params[0] = log10(Anw)
                params[1] = log10(Snw)

        Returns:
            np.ndarray: The non-water absorption coefficient 
        """
        a_nw = np.outer(10**params[...,0], np.ones_like(self.wave)) *\
            np.exp(np.outer(-10**params[...,1],self.wave-400.))

        return a_nw

    def init_guess(self, a_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        i400 = np.argmin(np.abs(self.wave-400))
        p0_a = np.array([a_nw[i400], 0.017])
        # Return
        return p0_a
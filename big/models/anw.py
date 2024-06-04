""" Models for non-water absorption """
import numpy as np

from oceancolor.water import absorption as water_abs

from abc import ABCMeta

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

    def __init__(self, wave:np.ndarray):
        self.wave = wave
        self.internals = {}

        # Initialize water
        self.init_aw()

    def meval(self, params:np.ndarray):
        return

    def init_aw(self, data:str='IOCCG'):
        """
        Initialize the absorption coefficient of water

        Args:
            data (str, optional): The data source to use. Defaults to 'IOCCG'.

        Returns:
            np.ndarray: The absorption coefficient of water
        """
        self.aw = water_abs.a_water(self.wave, data=data)

    def eval_anw(self, params:np.ndarray):
        """
        Evaluate the non-water absorption coefficient

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The non-water absorption coefficient
        """
        return np.zeros_like(self.wave)


    def eval_a(self, params:np.ndarray):
        """
        Evaluate the absorption coefficient

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The absorption coefficient
        """
        return self.aw + self.eval_anw(params)
        
class aNWExp(aNWModel):
    """
    Exponential model for non-water absorption
        Anw * exp(-Snw*(wave-400))

    Attributes:

    """
    name = 'Exp'

    def __init__(self, wave:np.ndarray):
        aNWModel.__init__(self, wave)

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
        anw = np.outer(10**params[...,0], np.ones_like(self.wave)) *\
            np.exp(np.outer(-10**params[...,1],self.wave-400.))

        return anw

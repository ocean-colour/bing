""" Models for non-water backscattering """
import numpy as np

from oceancolor.water import scattering as water_bb

from abc import ABCMeta

class bbNWModel:
    """
    Abstract base class for non-water backscattering

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
        self.init_bbw()

    def init_bbw(self):
        """
        Initialize the backscattering coefficient of water

        Args:

        Returns:
            np.ndarray: The absorption coefficient of water
        """
        self.bb_w = water_bb.betasw_ZHH2009(
            self.wave, 20, [0], 33)

    def eval_bbnw(self, params:np.ndarray):
        """
        Evaluate the non-water backscattering coefficients

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The non-water backscattering coefficient
        """
        return np.zeros_like(self.wave)


    def eval_bb(self, params:np.ndarray):
        """
        Evaluate the absorption coefficient

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The absorption coefficient
        """
        return self.bb_w + self.eval_bbnw(params)
        
class bbNWPow(bbNWModel):
    """
    Power-law model for non-water backscattering
        bb_nw = Bnw * (600/wave)^beta

    Attributes:

    """
    name = 'Exp'

    def __init__(self, wave:np.ndarray):
        bbNWModel.__init__(self, wave)

    def eval_bbnw(self, params:np.ndarray):
        """
        Evaluate the model

        Parameters:
            params (np.ndarray): The parameters for the model
                params[0] = log10(Bnw)
                params[1] = log10(beta)

        Returns:
            np.ndarray: The non-water absorption coefficient 
        """
        bb_nw = np.outer(10**params[...,0], np.ones_like(self.wave)) *\
                       (600./self.wave)**(10**params[...,1]).reshape(-1,1)

        return bb_nw

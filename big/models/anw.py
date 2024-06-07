""" Models for non-water absorption """
import numpy as np

from abc import ABCMeta

from scipy.interpolate import interp1d

from oceancolor.water import absorption as water_abs
from oceancolor.ph import absorption as ph_absorption

from big import priors as big_priors

from big.models import functions

from IPython import embed

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
    elif model_name == 'Cst':
        return aNWCst(wave, prior_choice)
    elif model_name == 'ExpBricaud':
        return aNWExpBricaud(wave, prior_choice)
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
    a_ph:np.ndarray = None
    """
    The absorption coefficient for phytoplankton
    """

    pivot:float = None
    """
    Pivot wavelength 
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

            Cst:
                params[...,0] = log10(Anw)
            Exp:
                params[...,0] = log10(Anw)
                params[...,1] = log10(Snw)

        Returns:
            np.ndarray: The non-water absorption coefficient
                This is always a multi-dimensional array
        """
        if self.name == 'Cst':
            return functions.constant(self.wave, params)
        elif self.name == 'Exp':
            return functions.exponential(self.wave, params, pivot=self.pivot)
        elif self.name == 'ExpBricaud':
            a_dg = functions.exponential(self.wave, params, pivot=self.pivot)
            a_ph = functions.gen_basis(params[...,-1:], self.a_ph)
            return a_dg + a_ph
        else:
            raise ValueError(f"Unknown model: {self.name}")

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

class aNWCst(aNWModel):
    """
    Constant model for non-water absorption
        Anw

    Attributes:

    """
    name = 'Cst'
    nparam = 1

    def __init__(self, wave:np.ndarray, prior_choice:str):
        aNWModel.__init__(self, wave, prior_choice)

    def init_guess(self, a_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        i400 = np.argmin(np.abs(self.wave-400))
        p0_a = np.array([a_nw[i400]])
        # Return
        return p0_a
        
class aNWExp(aNWModel):
    """
    Exponential model for non-water absorption
        Anw * exp(-Snw*(wave-400))

    Attributes:

    """
    name = 'Exp'
    nparam = 2
    pivot = 400.

    def __init__(self, wave:np.ndarray, prior_choice:str):
        aNWModel.__init__(self, wave, prior_choice)

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

class aNWExpBricaud(aNWModel):
    """
    Exponential model + aph for non-water absorption
        adg = Adg * exp(-Sdg*(wave-400))
        aph = A_B * chlA**B_B

    Attributes:

    """
    name = 'ExpBricaud'
    nparam = 3
    pivot = 400.

    def __init__(self, wave:np.ndarray, prior_choice:str):
        aNWModel.__init__(self, wave, prior_choice)

    def set_aph(self, Chla):
        # ##################################
        # Bricaud
        b1998 = ph_absorption.load_bricaud1998()

        # Interpolate
        f_b1998_A = interp1d(b1998['lambda'], b1998.Aphi, bounds_error=False, fill_value=0.)
        f_b1998_E = interp1d(b1998['lambda'], b1998.Ephi, bounds_error=False, fill_value=0.)

        # Apply
        L23_A = f_b1998_A(self.wave)
        L23_E = f_b1998_E(self.wave)

        self.a_ph = L23_A * Chla**L23_E

        # Normalize at 440
        iwave = np.argmin(np.abs(self.wave-440))
        self.a_ph /= self.a_ph[iwave]

    def init_guess(self, a_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        i400 = np.argmin(np.abs(self.wave-400))
        p0_a = np.array([a_nw[i400]/2., 0.017, a_nw[i400]/2.])
        # Return
        return p0_a
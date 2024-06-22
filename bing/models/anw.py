""" Models for non-water absorption """
import numpy as np
import warnings

from abc import ABCMeta

from scipy.interpolate import interp1d

from oceancolor.water import absorption as water_abs
from oceancolor.ph import absorption as ph_absorption

from ihop.iops import io as iops_io

from bing import priors as bing_priors
from bing.models import functions

from IPython import embed

def init_model(model_name:str, wave:np.ndarray, prior_dicts:list=None):
    """
    Initialize a model for non-water absorption

    Args:
        model_name (str): The name of the model
        wave (np.ndarray): The wavelengths
        prior_dicts (list): The choice of priors

    Returns:
        aNWModel: The model
    """
    model_dict = {'Exp': aNWExp, 'Cst': aNWCst, 'ExpBricaud': aNWExpBricaud,
                  'GIOP': aNWGIOP, 'ExpNMF': aNWExpNMF, 'ExpFix': aNWExpFix,
                  'GSM': aNWGSM}
    if model_name not in model_dict.keys():
        raise ValueError(f"Unknown model: {model_name}")
    else:
        return model_dict[model_name](wave, prior_dicts)

class aNWModel:
    """
    Abstract base class for non-water absoprtion

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

    uses_Chl:bool = False
    """
    Does the model use chlorophyll?
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

    priors:bing_priors.Priors = None
    """
    The priors for the model
    """
    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        self.wave = wave
        self.internals = {}

        # Initialize water
        self.init_aw()

        # Set priors
        if prior_dicts is not None:
            self.priors = bing_priors.Priors(prior_dicts)

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
        elif self.name == 'ExpFix':
            return functions.exponential(self.wave, params, pivot=self.pivot, S=self.Sdg)
        elif self.name == 'ExpBricaud':
            a_dg = functions.exponential(self.wave, params, pivot=self.pivot)
            a_ph = functions.gen_basis(params[...,-1:], [self.a_ph])
            return a_dg + a_ph
        elif self.name in ['GIOP', 'GSM']:
            a_dg = functions.exponential(self.wave, params, pivot=self.pivot, S=self.Sdg)
            a_ph = functions.gen_basis(params[...,-1:], [self.a_ph])
            return a_dg + a_ph
        elif self.name == 'ExpNMF':
            a_dg = functions.exponential(self.wave, params, pivot=self.pivot)
            a_ph = functions.gen_basis(params[...,-2:], 
                                       [self.W1, self.W2])
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
    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWModel.__init__(self, wave, prior_dicts)

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

class aNWExpFix(aNWModel):
    """
    Exponential model for non-water absorption with fixed S
        Aexp * exp(-Sexp*(wave-400))

    Attributes:

    """
    name = 'ExpFix'
    nparam = 1
    pivot = 400.

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWModel.__init__(self, wave, prior_dicts)
        self.Sdg = 0.018

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

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWModel.__init__(self, wave, prior_dicts)

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
    Exponential model + Bricaud aph for non-water absorption
        adg = Adg * exp(-Sdg*(wave-400))
        aph = A_B * chlA**B_B

    Attributes:

    """
    name = 'ExpBricaud'
    nparam = 3
    pivot = 400.
    uses_Chl = True

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWModel.__init__(self, wave, prior_dicts)

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
        assert p0_a.size == self.nparam
        # Return
        return p0_a


class aNWGIOP(aNWModel):
    """
    GIOP (Werdell+2013)
    Exponential model with Sdg fixed + Bricaud aph for non-water absorption
        adg = Adg * exp(-Sdg*(wave-400))
            Sdg = 0.018
        aph = Aph * [A_B * chlA**E_B]

    Attributes:

    """
    name = 'GIOP'
    nparam = 2
    pivot = 400.
    uses_Chl = True

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWModel.__init__(self, wave, prior_dicts)

        # Sdg
        self.Sdg = 0.018

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
        p0_a = np.array([a_nw[i400]/2., a_nw[i400]/2.])
        assert p0_a.size == self.nparam
        # Return
        return p0_a

class aNWExpNMF(aNWModel):
    """
    Exponential model + NMF aph for non-water absorption
        adg = Adg * exp(-Sdg*(wave-400))
        aph = H1*W1 + H2*W2

    """
    name = 'ExpNMF'
    nparam = 4
    pivot = 400.

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWModel.__init__(self, wave, prior_dicts)

        # Set the basis functions
        self.set_w1w2()

    def set_w1w2(self):
        warnings.warn("Need to remove the dependency on IHOP")

        # ##################################
        # NMF for aph
        # Load the decomposition of aph
        aph_file = iops_io.loisel23_filename('nmf', 'aph', 2, 4, 0)
        d_aph = np.load(aph_file)
        NMF_W1=d_aph['M'][0]
        NMF_W2=d_aph['M'][1]

        # Interpolate onto our wavelengths
        self.W1 = np.interp(self.wave, d_aph['wave'], NMF_W1)
        self.W2 = np.interp(self.wave, d_aph['wave'], NMF_W2)

    def init_guess(self, a_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        i400 = np.argmin(np.abs(self.wave-400))
        p0_a = np.array([a_nw[i400]/2., 0.017, a_nw[i400]/4., 
                         a_nw[i400]/4.])
        assert p0_a.size == self.nparam
        # Return
        return p0_a

class aNWGSM(aNWModel):
    """
    GSM (Manitorena+2002)
    Exponential model with Sdg fixed + Bricaud aph for non-water absorption
        adg = Adg * exp(-Sdg*(wave-400))
            Sdg = 0.0206
        aph = Chl * a_ph*
            with a_ph* an interpolation of Maritorena+2002 values

    Attributes:

    """
    name = 'GSM'
    nparam = 2
    pivot = 443.
    uses_Chl = True

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWModel.__init__(self, wave, prior_dicts)

        # Sdg 
        self.Sdg = 0.0206

    def set_aph(self, Chla):
        # ##################################
        # Maritorena+2002
        interp_wv = [412., 443., 490., 510., 555.]
        interp_aph_star = [0.00665, 0.05582, 0.02055, 0.01910, 0.01015]

        # Interpolate
        f = interp1d(interp_wv, interp_aph_star, kind='linear', fill_value='extrapolate')

        # Apply
        aph_star = f(self.wave)

        # Truncate at 400
        aph_star[self.wave < 400] = 0.

        # Minimum value is 0
        aph_star[aph_star < 0.] = 0.

        self.a_ph = aph_star
        self.Chla = Chla

    def init_guess(self, a_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        ipivot = np.argmin(np.abs(self.wave-self.pivot))
        p0_a = np.array([a_nw[ipivot]/2., self.Chla])
        assert p0_a.size == self.nparam
        # Return
        return p0_a
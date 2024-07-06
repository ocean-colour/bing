""" Models for non-water absorption """
import numpy as np
import warnings

from abc import ABCMeta

from scipy.interpolate import interp1d

from ocpy.water import absorption as water_abs
from ocpy.ph import absorption as ph_absorption

from bing import priors as bing_priors
from bing.models import functions

from IPython import embed

def init_model(model_name:str, wave:np.ndarray, 
               prior_dicts:list=None):
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
                  'GSM': aNWGSM, 'Every': aNWEvery,
                  'ExpB': aNWExp, 'Chase2017': aNWChase}
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

    pnames:list = None
    """
    The names of the parameters
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

        # Checks
        assert len(self.pnames) == self.nparam

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
        elif self.name == 'Every':
            return 10**params
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
    pnames = ['Anw']

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

class aNWEvery(aNWModel):
    """
    Fully flexible model that has one parameter for every wavelength channel
        Anw -- one per channel

    Attributes:

    """
    name = 'Every'
    nparam = None

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):

        # Set nparam
        self.nparam = wave.size
        self.pnames = [f'Anw_{wave[i]}' for i in range(wave.size)]
        
        aNWModel.__init__(self, wave, prior_dicts)


    def init_guess(self, a_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        # Return
        return a_nw


class aNWExpFix(aNWModel):
    """
    Exponential model for non-water absorption with fixed S
        Aexp * exp(-Sexp*(wave-400))

    Attributes:

    """
    name = 'ExpFix'
    nparam = 1
    pivot = 400.
    pnames = ['Aexp']

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
    pnames = ['Anw', 'Snw']
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
    pnames = ['Adg', 'Sdg', 'Aph']
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
        aexp = Aexp * exp(-Sexp*(wave-400))
            Sexp = 0.018
        aph = Aph * [A_B * chlA**E_B]

    Attributes:

    """
    name = 'GIOP'
    nparam = 2
    pnames = ['Aexp', 'Aph']
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
        aexp = Aexp * exp(-Sexp*(wave-400))
        aph = H1*W1 + H2*W2

    """
    name = 'ExpNMF'
    nparam = 4
    pnames = ['Aexp', 'Sexp', 'H1', 'H2']
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
    pnames = ['Aexp', 'Chl']
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

class aNWChase(aNWModel):
    """
    Chase+2017 

    Exponential model with Sdg fixed + Bricaud aph for non-water absorption
        aNAP = CNAP * exp(-SNAP*(wave-400))
        aCDOM = CCDOM * exp(-SCDOM*(wave-400))

        aph = Sum aph_i * exp((wave-wave_i)**2/2/sigma_i**2)
            8 Gaussians

    Attributes:

    """
    name = 'Chase2017'
    nparam = 28
    pnames = ['CNAP', 'SNAP', 'CCDOM', 'SCDOM', 
              'aph384', 'aph413', 'aph435', 'aph461', 
              'aph464', 'aph490', 'aph532', 'aph583', 
              'sig384', 'sig413', 'sig435', 'sig461',
              'sig464', 'sig490', 'sig532', 'sig583',
              'cen384', 'cen413', 'cen435', 'cen461',
              'cen464', 'cen490', 'cen532', 'cen583']
    pivot = 400.

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWModel.__init__(self, wave, prior_dicts)

        # Gaussian centroids
        self.init_priors()

    def init_priors(self):

        # NAP and CDOM
        prior_dicts = [
            dict(flavor='uniform', pmin=-6, pmax=np.log10(0.05)), # CNAP
            dict(flavor='uniform', pmin=np.log10(0.005), pmax=np.log10(0.016)), # SNAP
            dict(flavor='uniform', pmin=np.log10(0.01), pmax=np.log10(0.8)), # CCDOM
            dict(flavor='uniform', pmin=np.log10(0.005), pmax=np.log10(0.02)), # SCDOM
        ]

        # APH
        prior_dicts += [dict(flavor='uniform', pmin=-6, pmax=np.log10(0.5))]*8

        # SIGMA
        prior_dicts += [
            dict(flavor='uniform', pmin=np.log10(22.), pmax=np.log10(24.)), # 384
            dict(flavor='uniform', pmin=np.log10(8.), pmax=np.log10(10.)), # 
            dict(flavor='uniform', pmin=np.log10(13.), pmax=np.log10(15.)), # 
            dict(flavor='uniform', pmin=np.log10(10.), pmax=np.log10(12.)), # 
            dict(flavor='uniform', pmin=np.log10(18.), pmax=np.log10(20.)), # 
            dict(flavor='uniform', pmin=np.log10(18.), pmax=np.log10(20.)), # 
            dict(flavor='uniform', pmin=np.log10(19.), pmax=np.log10(21.)), # 
            dict(flavor='uniform', pmin=np.log10(19.), pmax=np.log10(21.)), # 
        ]

        # CEN
        prior_dicts += [
            dict(flavor='uniform', pmin=np.log10(383), pmax=np.log10(385)), # 384
            dict(flavor='uniform', pmin=np.log10(412), pmax=np.log10(414)), # 
            dict(flavor='uniform', pmin=np.log10(434), pmax=np.log10(436)), # 
            dict(flavor='uniform', pmin=np.log10(460), pmax=np.log10(462)), # 
            dict(flavor='uniform', pmin=np.log10(463), pmax=np.log10(465)), # 
            dict(flavor='uniform', pmin=np.log10(489), pmax=np.log10(491)), # 
            dict(flavor='uniform', pmin=np.log10(531), pmax=np.log10(533)), # 
            dict(flavor='uniform', pmin=np.log10(582), pmax=np.log10(584)), # 
        ]
        # Set
        self.priors = bing_priors.Priors(prior_dicts)

    def eval_anw(self, params:np.ndarray):
        """
        Evaluate the non-water absorption coefficient

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The non-water absorption coefficient
                This is always a multi-dimensional array
        """
        # NAP
        aNAP = functions.exponential(self.wave, params[...,:2], pivot=self.pivot)
        # CDOM
        aCDOM = functions.exponential(self.wave, params[...,2:4], pivot=self.pivot)

        # 
        atot = aNAP + aCDOM

        # Aph
        aphs = params[...,4:12]
        sigs = params[...,12:20]
        cens = params[...,20:]

        for aph, sig, cen in zip(aphs, sigs, cens):
            # Repackage
            params = np.array([aph, sig, cen])
            atot += functions.gaussian(self.wave, params)

        return atot

    def init_guess(self, a_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        bounds = self.priors.gen_bounds()
        p0_a = (bounds[0] + bounds[1])/2

        # Amplitudes
        i400 = np.argmin(np.abs(self.wave-400))
        p0_a[0] = np.log10(a_nw[i400]/3.)
        p0_a[2] = np.log10(a_nw[i400]/3.)

        p0_a[4:12] = np.log10(a_nw[i400]/3.)

        # Insist everything is in bounds
        p0_a = np.clip(p0_a, bounds[0], bounds[1])

        # Check
        assert p0_a.size == self.nparam
        # Return
        return p0_a
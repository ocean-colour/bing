"""
Non-Water Backscattering Models for BING
=========================================

This module implements various bio-optical models for non-water backscattering
(bb_nw) used in ocean color remote sensing retrievals. Non-water backscattering
is primarily caused by particles in the water column, including:

- Phytoplankton cells and their internal structures
- Non-algal particles (NAP): detritus, minerals, sediments
- Bubbles (in surface waters)

The spectral shape of particle backscattering typically follows a power-law
dependence on wavelength, with the exponent related to particle size distribution.

Available Models
----------------
- **Cst**: Spectrally constant backscattering
- **Pow**: Power-law model with free exponent (most common)
- **Lee**: Power-law with exponent from Lee et al. (2002) formula
- **GSM**: Power-law with fixed exponent (Maritorena et al. 2002)
- **Every**: Fully flexible (one parameter per wavelength)

Parameter Conventions
---------------------
All amplitude parameters (Bnw) are stored and fitted in log10 space.
Spectral exponents (beta, Y) remain in linear space.

References
----------
- Lee, Z. et al. (2002). "Deriving inherent optical properties from water color,"
  Appl. Opt. 41, 5755-5772.
- Maritorena, S. et al. (2002). "Ocean color chlorophyll algorithms for SeaWiFS,"
  J. Geophys. Res. 107, 3108.
- Morel, A. and Maritorena, S. (2001). "Bio-optical properties of oceanic waters,"
  J. Geophys. Res. 106, 7163-7180.

Examples
--------
>>> from bing.models import bbnw
>>> import numpy as np
>>> wave = np.arange(400, 701, 5)
>>>
>>> # Initialize power-law model
>>> model = bbnw.init_model('Pow', wave)
>>>
>>> # Evaluate at given parameters (log10 space for amplitude)
>>> params = np.array([-2.5, 1.0])  # log10(Bnw), beta
>>> bb_nw = model.eval_bbnw(params)
"""
import numpy as np

from scipy.interpolate import interp1d

from ocpy.water import scattering as water_bb
from ocpy.hydrolight import loisel23

from abc import ABCMeta

from bing.models import functions
from bing.priors import priors as bing_priors
from bing.rt import raman

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
    model_dict = {'Cst': bbNWCst, 'Pow': bbNWPow, 
                  'Lee': bbNWLee, 'GSM': bbNWGSM,
                  'Every': bbNWEvery} 

    if model_name not in model_dict.keys():
        raise ValueError(f"Unknown model: {model_name}")
    else:
        return model_dict[model_name](wave, prior_dicts)

class bbNWModel:
    """
    Abstract base class for non-water backscattering models.

    This class defines the interface and common functionality for all
    non-water backscattering models in BING. Subclasses implement specific
    parameterizations (power-law, constant, Lee algorithm, etc.).

    All models share a common structure:
    1. Initialization sets up wavelengths, water backscattering, and Raman parameters
    2. Priors are attached for Bayesian inference
    3. eval_bbnw() computes non-water backscattering from parameters
    4. eval_bb() adds water backscattering to get total backscattering

    Parameters are typically stored in log10 space for amplitudes.

    Attributes
    ----------
    name : str
        Model identifier (e.g., 'Pow', 'Lee', 'GSM')
    wave : np.ndarray
        Wavelengths at which the model operates [nm]
    nparam : int
        Number of free parameters
    pnames : list of str
        Names of the parameters
    bb_w : np.ndarray
        Pure water backscattering coefficient at model wavelengths [m^-1]
    bb_w_ex : np.ndarray
        Pure water backscattering at Raman excitation wavelengths [m^-1]
    wave_ex : np.ndarray
        Raman excitation wavelengths corresponding to model wavelengths [nm]
    bb_R : np.ndarray
        Raman backscattering coefficient at excitation wavelengths [m^-1]
    priors : bing.priors.Priors
        Prior distributions for Bayesian inference
    G1, G2 : float or np.ndarray or None
        Gordon coefficients for radiative transfer (can be wavelength-dependent)
    basis_func : np.ndarray or None
        Pre-computed spectral basis function for basis-based models
    uses_basis_params : bool
        Whether the model requires external parameters for the basis function
    internals : dict
        Storage for intermediate calculations

    See Also
    --------
    bbNWPow : Power-law model with free exponent
    bbNWLee : Power-law with Lee et al. (2002) dynamic exponent
    bbNWGSM : GSM model with fixed exponent
    """
    __metaclass__ = ABCMeta

    name:str = None
    """
    The name of the model
    """

    G1:float | np.ndarray = None
    """
    Gordon G1 coefficients
        If None, the default will be used (if the Gordon approx is done)
    """

    G2:float | np.ndarray = None
    """
    Gordon G2 coefficients
        If None, the default will be used (if the Gordon approx is done)
    """

    wave_ex:np.ndarray = None
    """
    Excitation wavelengths for Raman scattering
    """

    bb_R:np.ndarray = None
    """
    Raman backscattering coefficient
    """

    nparam:int = None
    """
    The number of parameters for the model
    """

    pnames:list = None
    """
    The names of the parameters
    """

    bb_w:np.ndarray = None
    """
    The backscattering of water
    """

    bb_w_ex:np.ndarray = None
    """
    The backscattering of water at Raman excitation wavelengths
    """

    priors:bing_priors.Priors = None
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

        # Initialize for Raman
        self.init_raman()

        # Initialize water
        self.init_bbw()

        # Set priors
        if prior_dicts is not None:
            self.priors = bing_priors.Priors(prior_dicts)

        # Checks
        assert len(self.pnames) == self.nparam

    def init_bbw(self):
        """
        Initialize the backscattering coefficient of water

        Args:

        Returns:
            np.ndarray: The backscattering coefficient of water
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
        # Raman
        self.bb_w_ex = f(self.wave_ex)
        
    def eval_bbnw(self, params:np.ndarray, wave:np.ndarray=None):
        """
        Evaluate the non-water backscattering coefficients

        Parameters:
            params (np.ndarray): The parameters for the model

            Pow:
                params[0] = log10(Bnw)
                params[1] = beta
            Cst:
                params[0] = log10(Bnw)

            wave (np.ndarray, optional): Wavelengths for evaluation

        Returns:
            np.ndarray: The non-water backscattering coefficient
        """
        # Wavelengths for evaluation
        if wave is None:
            wave = self.wave  # Model values

        if self.name == 'Pow':
            return functions.powerlaw(wave, params, pivot=self.pivot)
        elif self.name == 'Every':
            return 10**params
        elif self.name == 'Cst':
            return functions.constant(wave, params)
        elif self.name == 'Lee':
            return functions.gen_basis(params[...,-1:], [self.basis_func])
        elif self.name == 'GSM':
            return functions.gen_basis(params[...,-1:], [self.basis_func])
        else:
            raise ValueError(f"Unknown model: {self.name}")

    def eval_bb(self, params:np.ndarray):
        """
        Evaluate the backscattering coefficient

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The backscattering coefficients
        """
        # Add water and return
        return self.bb_w + self.eval_bbnw(params)

    def eval_bb_ex(self, params:np.ndarray):
        """
        Evaluate the backscatattering coefficient at Raman 
        excitation wavelengths    

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The backscatattering coefficients
        """
        # Add water and return
        return self.bb_w_ex + self.eval_bbnw(params, wave=self.wave_ex)

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

    def init_raman(self):
        """
        Initialize wavelengths and coefficients for Raman scattering calculations.

        Computes excitation wavelengths and Raman backscattering coefficients
        needed for the Raman scattering correction to Rrs.

        Sets
        ----
        wave_ex : np.ndarray
            Excitation wavelengths corresponding to self.wave via Raman shift.
            For water, the Raman shift is ~3400 cm^-1.
        bb_R : np.ndarray
            Raman backscattering coefficient at each excitation wavelength [m^-1].
            Computed using the Bartlett et al. (1998) parameterization.

        See Also
        --------
        bing.rt.raman.emission_to_excitation_wavelength : Wavelength conversion
        bing.rt.raman.raman_backscattering_coeff : Raman backscattering calculation
        """
        self.wave_ex = raman.emission_to_excitation_wavelength(self.wave)
        self.bb_R = raman.raman_backscattering_coeff(self.wave_ex)

    def __repr__(self):
        return f"<bbNWModel: {self.name}, nparam={self.nparam}>"

class bbNWCst(bbNWModel):
    """
    Constant model for non-water scattering
        Bnw

    Attributes:

    """
    name = 'Cst'
    nparam = 1
    pnames = ['Bnw']

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

class bbNWEvery(bbNWModel):
    """
    Fully flexible model that has one parameter for every wavelength channel
        Bnw -- one per channel

    Attributes:

    """
    name = 'Every'
    nparam = None

    def __init__(self, wave:np.ndarray, prior_dicts:list):
        # Set nparam
        self.nparam = wave.size
        self.pnames = [f'Bnw_{wave[i]}' for i in range(wave.size)]

        bbNWModel.__init__(self, wave, prior_dicts)


    def init_guess(self, bb_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            bb_nw (np.ndarray): The non-water scattering coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        return bb_nw
        
class bbNWPow(bbNWModel):
    """
    Power-law model for non-water backscattering.

    The most commonly used backscattering model, where particle backscattering
    follows a power-law dependence on wavelength. The exponent is related to
    the particle size distribution (Junge slope).

    Model equation:
        bb_nw(λ) = Bnw × (600/λ)^β

    Parameters (in fitting space)
    -----------------------------
    Bnw : float (log10)
        Backscattering amplitude at 600 nm [m^-1]
    beta : float (linear)
        Spectral exponent, typically 0-2. Higher values indicate smaller
        particles (steeper Junge slope).

    Attributes
    ----------
    name : str
        'Pow'
    nparam : int
        2 (Bnw, beta)
    pnames : list
        ['Bnw', 'beta']
    pivot : float
        Reference wavelength = 600 nm

    Notes
    -----
    - For Rayleigh scattering (very small particles), β ≈ 4
    - For typical oceanic particles, β ≈ 0.5-2
    - Smaller β indicates larger particles (e.g., sediments)

    Examples
    --------
    >>> model = bbNWPow(wave)
    >>> params = np.array([-2.5, 1.2])  # log10(Bnw), beta
    >>> bb_nw = model.eval_bbnw(params)
    """
    name = 'Pow'
    nparam = 2
    pnames = ['Bnw', 'beta']
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

class bbNWGSM(bbNWModel):
    """
    Garver-Siegel-Maritorena (GSM) backscattering model.

    Power-law model with a globally-optimized fixed spectral exponent from
    the GSM algorithm (Maritorena et al. 2002). Reduces parameter degeneracy
    by fixing the spectral shape.

    Model equation:
        bb_nw(λ) = Bnw × (443/λ)^1.0337

    Parameters (in fitting space)
    -----------------------------
    Bnw : float (log10)
        Backscattering amplitude at 443 nm [m^-1]

    Attributes
    ----------
    name : str
        'GSM'
    nparam : int
        1 (only Bnw)
    pnames : list
        ['Bnw']
    pivot : float
        Reference wavelength = 443 nm
    eta : float
        Fixed spectral exponent = 1.0337
    basis_func : np.ndarray
        Pre-computed (443/λ)^η spectral shape

    References
    ----------
    Maritorena, S. et al. (2002). "Ocean color chlorophyll algorithms for
    SeaWiFS," J. Geophys. Res. 107, 3108.
    """
    name = 'GSM'
    nparam = 1
    pnames = ['Bnw']
    pivot = 443.

    def __init__(self, wave:np.ndarray, prior_dicts:list):
        bbNWModel.__init__(self, wave, prior_dicts)

        # Manitorena+2002
        self.eta = 1.0337
        self.set_basis_func()

    def set_basis_func(self):
        self.basis_func = (self.pivot/self.wave)**self.eta

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

class bbNWLee(bbNWModel):
    """
    Lee et al. (2002) power-law model with dynamic spectral exponent.

    Uses the QAA (Quasi-Analytical Algorithm) approach where the backscattering
    spectral exponent Y is estimated from the ratio of subsurface reflectance
    at blue and green wavelengths, rather than being fitted directly.

    Model equation:
        bb_nw(λ) = Bnw × (600/λ)^Y

    where Y is computed from reflectance:
        Y = 2.2 × (1 - 1.2 × exp(-0.9 × rrs(440)/rrs(555)))

    Parameters (in fitting space)
    -----------------------------
    Bnw : float (log10)
        Backscattering amplitude at 600 nm [m^-1]

    Attributes
    ----------
    name : str
        'Lee'
    nparam : int
        1 (only Bnw)
    pnames : list
        ['Bnw']
    pivot : float
        Reference wavelength = 600 nm
    uses_basis_params : bool
        True - Y must be set before evaluation
    Y : float
        Spectral exponent computed from reflectance ratio
    basis_func : np.ndarray
        Pre-computed (600/λ)^Y spectral shape

    Notes
    -----
    Before calling eval_bbnw(), you must set the spectral exponent using
    either compute_Y() or set_basis_func(). This is typically done during
    data preparation using the observed Rrs.

    References
    ----------
    Lee, Z. et al. (2002). "Deriving inherent optical properties from water
    color: a multiband quasi-analytical algorithm for optically deep waters,"
    Appl. Opt. 41, 5755-5772.

    Examples
    --------
    >>> model = bbNWLee(wave)
    >>> model.compute_Y(rrs_440=0.01, rrs_555=0.005)  # From observed Rrs
    >>> params = np.array([-2.5])  # log10(Bnw)
    >>> bb_nw = model.eval_bbnw(params)
    """
    name = 'Lee'
    nparam = 1
    pnames = ['Bnw']
    pivot = 600.
    uses_basis_params = True

    def __init__(self, wave:np.ndarray, prior_dicts:list):
        bbNWModel.__init__(self, wave, prior_dicts)

        # Lee+2002
        self.Y = None

    def compute_Y(self, rrs_440:float, rrs_555:float):
        """ Compute Y from Lee+2002 given rrs           

        Parameters:
            rrs_440 (float): The remote sensing reflectance at 440 nm
            rrs_555 (float): The remote sensing reflectance at 555 nm

        """
        Y = 2.2 * (1 - 1.2 * np.exp(-0.9 * rrs_440/rrs_555))
        self.set_basis_func(Y)

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
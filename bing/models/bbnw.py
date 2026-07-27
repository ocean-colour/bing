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
- **Pow2**: Two-component mineral + organic power laws (turbid water)
- **Pow2Flat**: As Pow2 with a spectrally flat mineral term (3 params)

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
                  'Every': bbNWEvery,
                  'Pow2': bbNWPow2, 'Pow2Flat': bbNWPow2Flat}

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

    log_params:list = None
    """
    Which parameters are log10 amplitudes (True) vs linear (False)

    ``None`` means "all log10", which is the historical default and is
    what display code assumes for any model that does not declare this.

    This is consumed by **display** code (see bing.plotting) to decide
    which values to exponentiate and which labels to wrap in
    log10(...).  It is deliberately *not* used by the p0 conversion in
    the fitters, which keys on the prior flavor instead.
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
        self.check_priors()

    def check_priors(self):
        """
        Confirm the attached priors match the model's parameter count.

        Nothing else in BING validates this, and a mismatch is worse
        than a crash: the chi-squared path (bing.fitting.l23) walks the
        prior list with its own counter to decide which p0 slots are
        log10 amplitudes, so a wrong-length list silently converts the
        wrong parameters.  Fail loudly instead.

        Called on construction and by
        bing.priors.priors.set_standard_priors, which attaches priors
        after the model is built.

        Raises:
            ValueError: If the number of priors differs from nparam.
        """
        if self.priors is None:
            return
        if self.priors.nparam != self.nparam:
            raise ValueError(
                f"{self.name}: got {self.priors.nparam} priors for "
                f"{self.nparam} parameters ({self.pnames})")

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
            Pow2:
                params[0] = log10(Bmin)   # mineral, pivot 700 nm
                params[1] = eta_min
                params[2] = log10(Borg)   # organic, pivot 600 nm
                params[3] = eta_org
            Pow2Flat:
                params[0] = log10(Bmin)   # mineral, spectrally flat
                params[1] = log10(Borg)   # organic, pivot 600 nm
                params[2] = eta_org

            wave (np.ndarray, optional): Wavelengths for evaluation

        Returns:
            np.ndarray: The non-water backscattering coefficient
        """
        # Wavelengths for evaluation
        if wave is None:
            wave = self.wave  # Model values

        if self.name == 'Pow':
            return functions.powerlaw(wave, params, pivot=self.pivot)
        elif self.name == 'Pow2':
            # Mineral (near-flat, red-pivoted) + organic (steep) power
            # laws.  Slicing params[..., 0:2] / [..., 2:4] preserves
            # functions.powerlaw's (nsample, nwave) contract for 1-D
            # and 2-D (chain-shaped) input alike.
            return (functions.powerlaw(wave, params[..., 0:2],
                                       pivot=self.pivot_min) +
                    functions.powerlaw(wave, params[..., 2:4],
                                       pivot=self.pivot))
        elif self.name == 'Pow2Flat':
            # As Pow2 with eta_min fixed at 0, i.e. a constant mineral
            # term (its pivot is then irrelevant) + an organic power law.
            return (functions.constant(wave, params[..., 0:1]) +
                    functions.powerlaw(wave, params[..., 1:3],
                                       pivot=self.pivot))
        elif self.name == 'Every':
            return self.eval_channels(params, wave=wave)
        elif self.name == 'Cst':
            return functions.constant(wave, params)
        elif self.name in ['Lee', 'GSM']:
            # Evaluate the basis on the REQUESTED grid.  self.basis_func
            # is cached on self.wave, so using it here silently mixed
            # grids whenever wave was the Raman excitation grid.
            return functions.gen_basis(params[...,-1:],
                                       [self.eval_basis_func(wave)])
        else:
            raise ValueError(f"Unknown model: {self.name}")

    def eval_basis_func(self, wave:np.ndarray=None):
        """
        Evaluate the model's spectral basis function on a wavelength grid.

        Basis-function models (Lee, GSM) cache their shape on
        ``self.wave`` in ``self.basis_func``.  This method recomputes it
        for an arbitrary grid, which is what makes ``eval_bb_ex`` (the
        Raman excitation wavelengths) correct.

        Parameters:
            wave (np.ndarray, optional): Wavelengths for evaluation.
                Defaults to the model wavelengths.

        Returns:
            np.ndarray: The basis function on wave

        Raises:
            NotImplementedError: If the model has no basis function.
        """
        raise NotImplementedError(
            f"{self.name} has no basis function")

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
    log_params = [True]

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
        self.log_params = [True]*wave.size

        bbNWModel.__init__(self, wave, prior_dicts)

    def eval_channels(self, params:np.ndarray, wave:np.ndarray=None):
        """
        Evaluate the per-channel amplitudes, interpolating if needed.

        The parameters *are* bb_nw, one per model wavelength, so any
        other grid (e.g. the Raman excitation wavelengths) requires an
        interpolation choice.  We interpolate linearly in log10(bb_nw)
        vs log10(wave) -- i.e. locally power-law, the behaviour every
        other model here assumes -- and allow extrapolation, since the
        excitation grid extends blueward of the model grid.

        Parameters:
            params (np.ndarray): log10(bb_nw), one per model wavelength
            wave (np.ndarray, optional): Wavelengths for evaluation.
                Defaults to the model wavelengths.

        Returns:
            np.ndarray: bb_nw with shape (nsample, wave.size)
        """
        vals = 10**np.atleast_2d(params)
        if wave is None:
            return vals
        wave = np.atleast_1d(wave)
        if wave.size == self.wave.size and np.allclose(wave, self.wave):
            return vals
        # Power-law-like interpolation in log-log space
        f = interp1d(np.log10(self.wave), np.log10(vals), axis=-1,
                     kind='linear', fill_value='extrapolate')
        return 10**f(np.log10(wave))

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
    # beta is a linear exponent, not a log10 amplitude
    log_params = [True, False]
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

class bbNWPow2(bbNWModel):
    """
    Two-component (mineral + organic) particulate backscattering.

    Turbid, mineral-dominated water needs particulate backscatter that
    is both larger in magnitude and flatter -- sometimes rising -- with
    wavelength than a single decreasing power law can supply.  This
    model splits bb_nw into a near-flat mineral term and a steeper
    organic term, so the red can be lifted without wrecking the blue.

    Model equation:
        bb_nw(λ) = Bmin × (700/λ)^eta_min + Borg × (600/λ)^eta_org

    The mineral term is pivoted at 700 nm so that Bmin *is* the mineral
    backscatter in the red, where turbid reflectance constrains it; the
    organic term keeps BING's 600 nm convention.

    Parameters (in fitting space)
    -----------------------------
    Bmin : float (log10)
        Mineral backscattering amplitude at 700 nm [m^-1]
    eta_min : float (linear)
        Mineral spectral exponent, ~0 (flat/"white").  Negative values
        make bb_nw rise toward the red.
    Borg : float (log10)
        Organic backscattering amplitude at 600 nm [m^-1]
    eta_org : float (linear)
        Organic spectral exponent, the open-ocean 0.5-2 range.

    Attributes
    ----------
    name : str
        'Pow2'
    nparam : int
        4 (Bmin, eta_min, Borg, eta_org)
    pnames : list
        ['Bmin', 'eta_min', 'Borg', 'eta_org']
    log_params : list of bool
        Which parameters are log10 amplitudes
    pivot : float
        Organic/reference wavelength = 600 nm.  Kept a scalar because
        external analysis scripts read models[1].pivot.
    pivot_min : float
        Mineral pivot wavelength = 700 nm

    Notes
    -----
    Keep the eta_min and eta_org priors on *disjoint* ranges (e.g.
    [-0.5, 0.5] and [0.5, 2]).  The two terms are otherwise
    exchangeable, and a symmetric prior gives the posterior a
    label-switching degeneracy.

    References
    ----------
    - Snyder, W. A. et al. (2008). "Optical scattering and backscattering
      by organic and inorganic particulates in U.S. coastal waters,"
      Appl. Opt. 47, 666-677.
    - Twardowski, M. S. et al. (2001). "A model for estimating bulk
      refractive index from the optical backscattering ratio,"
      J. Geophys. Res. 106, 14129-14142.
    - Doxaran, D. et al. (2009). "Spectral variations of light scattering
      by marine particles in coastal waters," Limnol. Oceanogr. 54,
      1257-1271.
    - Neukermans, G. et al. (2012). "In situ variability of mass-specific
      beam attenuation and backscattering of marine particles,"
      Limnol. Oceanogr. 57, 124-144.

    Examples
    --------
    >>> model = bbnw.init_model('Pow2', wave)
    >>> params = np.array([-1.0, 0.0, -2.0, 1.0])
    >>> bb_nw = model.eval_bbnw(params)
    """
    name = 'Pow2'
    nparam = 4
    pnames = ['Bmin', 'eta_min', 'Borg', 'eta_org']
    # Which slots are log10 amplitudes (the rest are linear exponents)
    log_params = [True, False, True, False]
    pivot = 600.
    pivot_min = 700.

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        bbNWModel.__init__(self, wave, prior_dicts)

    def init_guess(self, bb_nw:np.ndarray):
        """
        Initialize the model with a guess

        Splits the observed bb_nw between the two components, each
        evaluated at its own pivot.

        Parameters:
            bb_nw (np.ndarray): The non-water backscattering coefficient

        Returns:
            np.ndarray: Initial guess, amplitudes in LINEAR space (the
                caller log10s the log-flavored slots)
        """
        i700 = np.argmin(np.abs(self.wave-self.pivot_min))
        i600 = np.argmin(np.abs(self.wave-self.pivot))
        # eta_min is seeded slightly OFF zero on purpose: the MCMC
        # walker perturbation in bing.fitting.inference is
        # multiplicative (p0 += p0*U(-1e-2,1e-2)), so a parameter seeded
        # at exactly 0 gets zero spread across walkers and that
        # dimension never moves for the whole run.
        p0_bb = np.array([max(bb_nw[i700]/2., 1e-5), 0.05,
                          max(bb_nw[i600]/2., 1e-5), 1.])
        assert p0_bb.size == self.nparam

        # Return
        return p0_bb

class bbNWPow2Flat(bbNWPow2):
    """
    Two-component backscattering with a spectrally FLAT mineral term.

    :class:`bbNWPow2` with eta_min fixed at 0, i.e. a constant mineral
    term plus an organic power law:

        bb_nw(λ) = Bmin + Borg × (600/λ)^eta_org

    Dropping the mineral exponent removes one parameter and most of the
    amplitude/slope degeneracy between the two terms, at the cost of not
    being able to represent a *rising* mineral contribution.  Use it as
    the identifiability-safe arm when comparing against Pow2.

    Parameters (in fitting space)
    -----------------------------
    Bmin : float (log10)
        Mineral backscattering amplitude [m^-1], wavelength independent
    Borg : float (log10)
        Organic backscattering amplitude at 600 nm [m^-1]
    eta_org : float (linear)
        Organic spectral exponent, the open-ocean 0.5-2 range

    Attributes
    ----------
    name : str
        'Pow2Flat'
    nparam : int
        3 (Bmin, Borg, eta_org)
    pnames : list
        ['Bmin', 'Borg', 'eta_org']

    References
    ----------
    See :class:`bbNWPow2`.
    """
    name = 'Pow2Flat'
    nparam = 3
    pnames = ['Bmin', 'Borg', 'eta_org']
    log_params = [True, True, False]

    def init_guess(self, bb_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            bb_nw (np.ndarray): The non-water backscattering coefficient

        Returns:
            np.ndarray: Initial guess, amplitudes in LINEAR space (the
                caller log10s the log-flavored slots)
        """
        i700 = np.argmin(np.abs(self.wave-self.pivot_min))
        i600 = np.argmin(np.abs(self.wave-self.pivot))
        p0_bb = np.array([max(bb_nw[i700]/2., 1e-5),
                          max(bb_nw[i600]/2., 1e-5), 1.])
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
    log_params = [True]
    pivot = 443.

    def __init__(self, wave:np.ndarray, prior_dicts:list):
        bbNWModel.__init__(self, wave, prior_dicts)

        # Manitorena+2002
        self.eta = 1.0337
        self.set_basis_func()

    def set_basis_func(self):
        """Cache the basis function on the model wavelengths."""
        self.basis_func = self.eval_basis_func()

    def eval_basis_func(self, wave:np.ndarray=None):
        """
        Evaluate (pivot/wave)**eta on an arbitrary wavelength grid.

        Parameters:
            wave (np.ndarray, optional): Wavelengths for evaluation.
                Defaults to the model wavelengths.

        Returns:
            np.ndarray: The GSM spectral shape on wave
        """
        if wave is None:
            wave = self.wave
        return (self.pivot/wave)**self.eta

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
    log_params = [True]
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
        """Set the spectral exponent and cache the basis function."""
        self.Y = Y
        self.basis_func = self.eval_basis_func()

    def eval_basis_func(self, wave:np.ndarray=None):
        """
        Evaluate (pivot/wave)**Y on an arbitrary wavelength grid.

        Parameters:
            wave (np.ndarray, optional): Wavelengths for evaluation.
                Defaults to the model wavelengths.

        Returns:
            np.ndarray: The Lee spectral shape on wave

        Raises:
            ValueError: If Y has not been set yet (call compute_Y or
                set_basis_func first).
        """
        if self.Y is None:
            raise ValueError(
                "Lee: Y is not set -- call compute_Y() or "
                "set_basis_func(Y) before evaluating")
        if wave is None:
            wave = self.wave
        return (self.pivot/wave)**self.Y

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
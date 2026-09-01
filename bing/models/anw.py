"""
Non-Water Absorption Models for BING
=====================================

This module implements various bio-optical models for non-water absorption
(a_nw) used in ocean color remote sensing retrievals. Non-water absorption
consists of contributions from:

- Phytoplankton pigments (a_ph): Primarily chlorophyll-a and accessory pigments
- Colored dissolved organic matter (CDOM, a_g): Exponentially decaying with wavelength
- Non-algal particles/detritus (NAP, a_d): Also exponentially decaying

The combined dissolved + detrital absorption is often modeled together as a_dg.

Available Models
----------------
- **Cst**: Spectrally constant absorption
- **Exp**: Single exponential decay (for a_dg-dominated waters)
- **ExpFix**: Exponential with fixed spectral slope
- **Bricaud**: Phytoplankton-only using Bricaud et al. (1995) parameterization
- **ExpBricaud**: Exponential a_dg + Bricaud a_ph (most common for BING)
- **ExpBricaudFix**: Like ExpBricaud but with fixed chlorophyll
- **ExpBricaudFree**: Like ExpBricaud but with Chl as free parameter
- **GIOP**: Fixed-slope exponential + Bricaud (Werdell et al. 2013)
- **GSM**: Garver-Siegel-Maritorena model (Maritorena et al. 2002)
- **ExpNMF**: Exponential + NMF basis functions for a_ph
- **Chase2017**: Gaussian decomposition (Chase et al. 2017)
- **Every**: Fully flexible (one parameter per wavelength)

Parameter Conventions
---------------------
All amplitude parameters are stored and fitted in log10 space for numerical
stability. Spectral slopes (S, Sdg) remain in linear space.

References
----------
- Bricaud, A. et al. (1995). "Variability in the chlorophyll-specific absorption
  coefficients of natural phytoplankton," J. Geophys. Res. 100, 13321-13332.
- Werdell, P.J. et al. (2013). "Generalized ocean color inversion model (GIOP),"
  Appl. Opt. 52, 2019-2037.
- Maritorena, S. et al. (2002). "Ocean color chlorophyll algorithms for SeaWiFS,"
  J. Geophys. Res. 107, 3108.
- Chase, A.P. et al. (2017). "Decomposition of in situ particulate absorption
  spectra," Methods in Oceanography 7, 110-124.

Examples
--------
>>> from bing.models import anw
>>> import numpy as np
>>> wave = np.arange(400, 701, 5)
>>>
>>> # Initialize ExpBricaud model
>>> model = anw.init_model('ExpBricaud', wave)
>>> model.set_aph(Chl=1.0)  # Set chlorophyll for Bricaud parameterization
>>>
>>> # Evaluate at given parameters (log10 space for amplitudes)
>>> params = np.array([-1.0, 0.017, -1.2])  # log10(Adg), Sdg, log10(Aph)
>>> a_nw = model.eval_anw(params)
"""
import numpy as np
import warnings

from abc import ABCMeta

from scipy.interpolate import interp1d

from ocpy.water import absorption as water_abs
from ocpy.ph import absorption as ph_absorption

from bing.priors import priors as bing_priors
from bing.models import functions
from bing.rt import raman, rrs

from IPython import embed

# ##################################
# Bricaud
b1998 = ph_absorption.load_bricaud1998()

# Interpolate
f_b1998_A = interp1d(b1998['lambda'], b1998.Aphi, bounds_error=False, fill_value=0.)
f_b1998_E = interp1d(b1998['lambda'], b1998.Ephi, bounds_error=False, fill_value=0.)

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
    model_dict = {'Exp': aNWExp, 'Cst': aNWCst, 
                  'ExpBricaudFix': aNWExpBricaudFix,
                  'ExpBricaudFree': aNWExpBricaudFree,
                  'ExpBricaud': aNWExpBricaud,
                  'GIOP': aNWGIOP, 'ExpNMF': aNWExpNMF, 'ExpFix': aNWExpFix,
                  'GSM': aNWGSM, 'Every': aNWEvery,
                  'ExpB': aNWExp, 'Chase2017': aNWChase, 
                  'Chase2017Mini': aNWChaseMini,
                  'Bricaud': aNWBricaud,
                  }
    if model_name not in model_dict.keys():
        raise ValueError(f"Unknown model: {model_name}")
    else:
        return model_dict[model_name](wave, prior_dicts)

class aNWModel:
    """
    Abstract base class for non-water absorption models.

    This class defines the interface and common functionality for all
    non-water absorption models in BING. Subclasses implement specific
    bio-optical parameterizations (exponential, Bricaud, etc.).

    All models share a common structure:
    1. Initialization sets up wavelengths, water absorption, and Raman parameters
    2. Priors are attached for Bayesian inference
    3. eval_anw() computes non-water absorption from parameters
    4. eval_a() adds water absorption to get total absorption

    Parameters are typically stored and fitted in log10 space for amplitudes
    to ensure positivity and improve sampling efficiency.

    Attributes
    ----------
    name : str
        Model identifier (e.g., 'Exp', 'ExpBricaud', 'GIOP')
    wave : np.ndarray
        Wavelengths at which the model operates [nm]
    nparam : int
        Number of free parameters
    pnames : list of str
        Names of the parameters
    a_w : np.ndarray
        Pure water absorption coefficient at model wavelengths [m^-1]
    a_w_ex : np.ndarray
        Pure water absorption at Raman excitation wavelengths [m^-1]
    wave_ex : np.ndarray
        Raman excitation wavelengths corresponding to model wavelengths [nm]
    priors : bing.priors.Priors
        Prior distributions for Bayesian inference
    uses_Chl : bool
        Whether model requires chlorophyll input for phytoplankton absorption
    fix_Chl : bool
        If uses_Chl, whether chlorophyll is fixed or fitted
    G1, G2 : float or np.ndarray or None
        Gordon coefficients for radiative transfer (can be wavelength-dependent)
    G0 : float or np.ndarray or None
        Gordon G0 coefficients for radiative transfer (can be wavelength-dependent)
        If None, the default will be used (if the Gordon approx is done)
        If variable_Gordon_G0 is True, then G0 will be fitted.
        If variable_Gordon_G0 is False, then G0 will be fixed to the default.
    pivot : float
        Reference wavelength for spectral parameterizations [nm]
    internals : dict
        Storage for intermediate calculations

    See Also
    --------
    aNWExp : Exponential decay model
    aNWExpBricaud : Exponential + Bricaud phytoplankton
    aNWGIOP : GIOP algorithm implementation
    """
    __metaclass__ = ABCMeta

    name:str = None
    """
    The name of the model
    """

    G0:float | np.ndarray = None
    """
    Gordon G1 coefficients
        If None, the default will be used (if the Gordon approx is done)
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

    Gb:float | np.ndarray = None
    """
    Gordon Gb coefficient (slope on particulate backscatter bbp)
        If None, the bbp term is omitted (i.e. rrs = G0 + G1·u + G2·u²).
        If variable_Gordon_bbp is True, Gb(λ) is loaded from
        ``gordon_coefficients_with_Gb.csv``.
    """

    wave_ex:np.ndarray = None
    """
    Excitation wavelengths for Raman scattering
    """

    nparam:int = None
    """
    The number of parameters for the model
    """

    pnames:list = None
    """
    The names of the parameters
    """

    log_params:list = None
    """
    Which parameters are log10 amplitudes (True) vs linear (False)

    ``None`` means "all log10", the historical default and what display
    code assumes for any model that does not declare this.  Read by
    **display** code (bing.plotting); deliberately not used by the p0
    conversion in the fitters, which keys on the prior flavor.
    """

    uses_Chl:bool = False
    """
    Does the model use chlorophyll (for absorption)?
    """

    fix_Chl:bool = None
    """
    If Chl, is it fixed?
    """

    i_Chl_ex:np.ndarray = None
    """
    The indices of the excitation wavelengths for chlorophyll fluorescence
    """

    Ed_ex:np.ndarray = None
    """
    The downwelling irradiance at the excitation wavelengths
    """

    Ed_em:float = None
    """
    The downwelling irradiance at the emission wavelength(s).
    An array over the model wave grid (preferred) or a scalar at the
    685 nm peak (legacy).
    """

    Ed_ratio_raman:np.ndarray = None
    """
    Downwelling-irradiance ratio Ed(wave_ex)/Ed(wave) between the Raman
    excitation and emission (model) wavelengths. Set by set_raman_Ed().
    When None, the Raman correction falls back to a flat solar spectrum
    (ratio = 1), which distorts its spectral shape.
    """

    wave_Ed_raw:np.ndarray = None
    """
    Wavelengths [nm] of the raw Ed spectrum last passed to set_raman_Ed(),
    stored verbatim (the caller's array, uncopied). None until
    set_raman_Ed() is called. Consumed (with Ed_raw) by the robust RT
    backend, which builds its own Ed ratio internally from the raw pair.
    """

    Ed_raw:np.ndarray = None
    """
    Raw downwelling-irradiance spectrum at wave_Ed_raw last passed to
    set_raman_Ed(), stored verbatim (the caller's array, uncopied). None
    until set_raman_Ed() is called. Consumed (with wave_Ed_raw) by the
    robust RT backend; BING's own Raman path keeps using Ed_ratio_raman.
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

        # Initialize for Raman
        self.init_raman()

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
        self.a_w_ex = water_abs.a_water(self.wave_ex, data=data)

    def eval_anw(self, params:np.ndarray, retsub_comps:bool=False,
                 wave:np.ndarray=None):
        """
        Evaluate the non-water absorption coefficient

        Parameters:
            params (np.ndarray): The parameters for the model
            retsub_comps (bool, optional): Return the sub-components. Default is False.

            Cst:
                params[...,0] = log10(Anw)
            Bricaud:
                params[...,0] = log10(Aph) 
            Exp:
                params[...,0] = log10(Anw)
                params[...,1] = log10(Snw)
            ExpBricaud:
                params[...,0] = log10(Adg)
                params[...,1] = log10(Sdg)
                params[...,2] = log10(Aph)
            wave (np.ndarray, optional): Wavelengths for evaluation

        Returns:
            np.ndarray: The non-water absorption coefficient
                This is always a multi-dimensional array
        """
        # Wavelengths for evaluation
        if wave is None:
            wave = self.wave  # Model values

        if self.name == 'Cst':
            return functions.constant(wave, params)
        elif self.name == 'Every':
            return 10**params
        elif self.name == 'Exp':
            return functions.exponential(wave, params, pivot=self.pivot)
        elif self.name == 'ExpFix':
            return functions.exponential(wave, params, pivot=self.pivot, S=self.Sdg)
        elif self.name == 'Bricaud':
            Chl = 10**params[...,-1:] / 0.05582
            self.set_aph(Chl, wave=wave)
            if len(params.shape) == 2:
                a_ph = (10**params[...,-1:]) * self.a_ph
            else:
                a_ph = functions.gen_basis(params[...,-1:], [self.a_ph])
            return a_ph
        elif self.name in ['ExpBricaudFix', 'ExpBricaud', 'ExpBricaudFree']:
            # a_dg
            a_dg = functions.exponential(wave, params, pivot=self.pivot)
            # a_ph
            if not self.fix_Chl:
                if self.name == 'ExpBricaud':
                    Chl = 10**params[...,-1:] / 0.05582
                elif self.name == 'ExpBricaudFree':
                    Chl = 10**params[...,-2] 
                else:
                    raise ValueError(f"Unknown model: {self.name}")
                self.set_aph(Chl, wave=wave)
            if len(params.shape) == 2:
                a_ph = (10**params[...,-1:]) * self.a_ph
            else:
                a_ph = functions.gen_basis(params[...,-1:], [self.a_ph])
            # Finish
            if retsub_comps:
                return a_dg, a_ph
            else:
                return a_dg + a_ph
        elif self.name in ['GIOP', 'GSM']:
            a_dg = functions.exponential(wave, params, pivot=self.pivot, S=self.Sdg)
            a_ph = functions.gen_basis(params[...,-1:], [self.a_ph])
            if retsub_comps:
                return a_dg, a_ph
            else:
                return a_dg + a_ph
        elif self.name == 'ExpNMF':
            a_dg = functions.exponential(wave, params, pivot=self.pivot)
            a_ph = functions.gen_basis(params[...,-2:], 
                                       [self.W1, self.W2])
            if retsub_comps:
                return a_dg, a_ph
            else:
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

    def eval_a_ex(self, params:np.ndarray):
        """
        Evaluate the absorption coefficient at Raman 
        excitation wavelengths    

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The absorption coefficient
        """
        # Add water and return
        return self.a_w_ex + self.eval_anw(params, wave=self.wave_ex)

    def init_guess(self, a_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient
        """

    def init_var_gordon(self, include_G0:bool=False, include_Gb:bool=False):
        """
        Initialize the variable Gordon parameters.

        Four recipes are supported, chosen by the two flags:

        - ``include_G0=False, include_Gb=False`` -- 2-parameter
            ``rrs = G1·u + G2·u²``  (loads ``gordon_coefficients.csv``).
        - ``include_G0=True,  include_Gb=False`` -- 3-parameter, constant offset
            ``rrs = G0 + G1·u + G2·u²``  (loads ``gordon_coefficients_with_G0.csv``).
        - ``include_G0=False, include_Gb=True``  -- 3-parameter, bbp slope
            ``rrs = G1·u + G2·u² + Gb·bbp``  (loads ``gordon_coefficients_with_Gb.csv``).
        - ``include_G0=True,  include_Gb=True``  -- 4-parameter
            ``rrs = G0 + G1·u + G2·u² + Gb·bbp``  (loads ``gordon_coefficients_with_G0_Gb.csv``).
        """
        if include_G0 and include_Gb:
            self.G1, self.G2, self.G0, self.Gb = rrs.wave_dependent_gordon_full(self.wave)
        elif include_Gb:
            self.G1, self.G2, self.Gb = rrs.wave_dependent_gordon_bbp(self.wave)
            self.G0 = None
        else:
            self.G1, self.G2, self.G0 = rrs.wave_dependent_gordon(
                self.wave, include_G0=include_G0)
            self.Gb = None

    def init_raman(self):
        """
        Initialize wavelengths for Raman scattering calculations.

        Computes the excitation wavelengths that correspond to each emission
        (model) wavelength via the Raman shift (~3400 cm^-1 for water).
        These are needed for computing the Raman correction to Rrs.

        Sets
        ----
        wave_ex : np.ndarray
            Excitation wavelengths corresponding to self.wave via Raman shift.
            For example, emission at 550 nm corresponds to excitation at ~470 nm.

        See Also
        --------
        bing.rt.raman.emission_to_excitation_wavelength : Wavelength conversion function
        """
        self.wave_ex = raman.emission_to_excitation_wavelength(self.wave)

    def set_raman_Ed(self, wave_Ed:np.ndarray, Ed:np.ndarray):
        """
        Set the downwelling-irradiance ratio used by the Raman correction.

        Interpolates the supplied Ed spectrum onto the model (emission)
        grid and the Raman excitation grid, and stores

            Ed_ratio_raman = Ed(wave_ex) / Ed(wave)

        When this attribute is set, the production Raman path
        (evaluate.calc_Rrs_from_models -> rrs.calc_Rrs) uses the true
        solar-spectrum ratio instead of the flat-Ed (ratio = 1) fallback.
        Validated against the Loisel+23 X2/X1 HydroLight pairs, the flat
        fallback distorts the spectral shape of the Raman correction
        (~ +60% increment error at 490 nm, -15% and worse in the red);
        the true ratio removes most of that error.

        The incoming pair is also stashed verbatim (uncopied) on
        ``wave_Ed_raw`` / ``Ed_raw`` before the ratio is computed. The
        robust RT backend reads that raw pair to build its own internal
        Ed ratio (``robust.rt.ed``); BING's own Raman path is unaffected
        and keeps consuming ``Ed_ratio_raman``.

        Parameters
        ----------
        wave_Ed : np.ndarray
            Wavelengths of the Ed spectrum [nm]. Must cover both the
            model grid and the Raman excitation grid; note wave_ex
            extends ~50 nm blueward of the model grid (e.g. a 400 nm
            emission edge needs Ed down to ~352 nm).
        Ed : np.ndarray
            Downwelling irradiance at wave_Ed (any consistent units;
            only the ratio is used).

        Sets
        ----
        wave_Ed_raw, Ed_raw : np.ndarray
            The incoming pair, verbatim.
        Ed_ratio_raman : np.ndarray
            Ed(wave_ex) / Ed(wave) on the model grids.
        """
        # Stash the raw pair verbatim for consumers that build their own
        # ratio (the robust RT backend's Geometry.Ed seam).
        self.wave_Ed_raw = wave_Ed
        self.Ed_raw = Ed
        # Ratio computation unchanged.
        if self.wave_ex is None:
            self.init_raman()
        f_Ed = interp1d(wave_Ed, Ed, kind='linear', bounds_error=True)
        self.Ed_ratio_raman = f_Ed(self.wave_ex) / f_Ed(self.wave)

    def init_Chl_fluorescence(self, wv_ex_range:tuple=(400, 700),
        Ed:np.ndarray=None, Ed_em=None):
        """
        Initialize the chlorophyll fluorescence parameters

        Parameters:
            wv_ex_range (tuple, optional): The range of excitation wavelengths. Defaults to (400, 700).
            Ed (np.ndarray): Downwelling irradiance on the model wave grid.
            Ed_em (float or np.ndarray, optional): Downwelling irradiance at
                the emission wavelength(s). If None (preferred), the full
                Ed vector on the model grid is used, giving the exact
                per-wavelength normalization of the fluorescence term.
                A scalar (legacy: Ed at the 685 nm peak) is still accepted.
        """
        # Grab the indices
        i_Chl_ex = np.where((self.wave >= wv_ex_range[0]) & (self.wave <= wv_ex_range[1]))[0]
        #i_Chl_em = np.where((self.wave >= wv_em_range[0]) & (self.wave <= wv_em_range[1]))[0]

        # Multi-spectral checks here (not ready for multi-spectral yet)

        # Downwelling
        if Ed is None:
            raise IOError("Need to calculate here")
        self.Ed_ex = Ed[i_Chl_ex]
        # Per-lambda_em Ed (array) unless a legacy scalar is supplied
        self.Ed_em = Ed if Ed_em is None else Ed_em

        # Save em
        self.i_Chl_ex = i_Chl_ex
        #self.i_Chl_em = i_Chl_em

    def __repr__(self):
        return f"<aNWModel: {self.name}, nparam={self.nparam}>"

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
        Aexp * exp(-Sdg*(wave-400))

    Free parameters:
        Aexp

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
    pnames = ['Anw', 'Snw']  # log10, linear
    log_params = [True, False]
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


class aNWBricaud(aNWModel):
    """
    Bricaud aph for non-water absorption
        aph = a_ph(440) * A_B * chlA**B_B

    Attributes:

    """
    name = 'Bricaud'
    nparam = 1
    pnames = ['Aph']
    pivot = 400.
    uses_Chl = True
    fix_Chl = False

    L23_A:np.ndarray = None
    """
    Pre-evaluation of Bricaud parameters at model wavelengths
    """

    L23_E:np.ndarray = None
    """
    Pre-evaluation of Bricaud parameters at model wavelengths
    """

    L23_A_440:np.ndarray = None
    """
    Pre-evaluation of Bricaud parameter at 440nm
    """

    L23_E_440:np.ndarray = None
    """
    Pre-evaluation of Bricaud parameter at 440nm
    """

    L23_A_ex:np.ndarray = None
    """
    Pre-evaluation of Bricaud parameters at excitation wavelengths (Raman)
    """

    L23_E_ex:np.ndarray = None
    """
    Pre-evaluation of Bricaud parameters at excitation wavelengths (Raman)
    """


    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWModel.__init__(self, wave, prior_dicts)

        # Save parameterization
        self.L23_A = f_b1998_A(self.wave)
        self.L23_E = f_b1998_E(self.wave)
        self.L23_A_440 = f_b1998_A(440.)
        self.L23_E_440 = f_b1998_E(440.)
        self.L23_A_ex = f_b1998_A(self.wave_ex)
        self.L23_E_ex = f_b1998_E(self.wave_ex)


    def set_aph(self, Chla, wave:np.ndarray=None):
        """
        Set the phytoplankton absorption spectrum using Bricaud (1995) parameterization.

        Computes normalized phytoplankton absorption a*_ph(λ) such that::

            a_ph(λ) = Aph × a*_ph(λ)

        where a*_ph is normalized to have value 1.0 at 440 nm. The shape varies
        with chlorophyll concentration following Bricaud et al. (1995)::

            a_ph(λ) = A(λ) × Chl^E(λ)

        Parameters
        ----------
        Chla : float or np.ndarray
            Chlorophyll-a concentration in mg m^-3. Can be a single value or
            an array for batch processing (e.g., MCMC chains).
        wave : np.ndarray, optional
            Wavelengths for evaluation. If None, uses self.wave.
            Can also be self.wave_ex for Raman excitation wavelengths.

        Notes
        -----
        - The result is stored in self.a_ph as a normalized spectrum
        - For wavelengths < 400 nm, linear extrapolation is applied
        - Pre-computed coefficients (L23_A, L23_E) are used when possible
          for efficiency

        See Also
        --------
        aNWExpBricaud : Model combining exponential a_dg with Bricaud a_ph
        """
        # Bricaud

        if wave is None:
            wave = self.wave  # Model values

        # Load up the coefficients
        if np.all(np.isclose(wave, self.wave)):
            L23_A = self.L23_A
            L23_E = self.L23_E
        elif np.all(np.isclose(wave, self.wave_ex)):
            L23_A = self.L23_A_ex
            L23_E = self.L23_E_ex
        else:
            L23_A = f_b1998_A(wave)
            L23_E = f_b1998_E(wave)

        # Calculate
        if len(Chla.shape) == 2:
            Chla_array = np.outer(Chla, np.ones(L23_E.size))
            self.a_ph = L23_A * Chla_array**L23_E
            # Normalize
            aph_440 = self.L23_A_440 * Chla[:,0]**self.L23_E_440
            norm = np.outer(aph_440, np.ones(self.a_ph.shape[1]))
            self.a_ph /= norm
        else:
            self.a_ph = L23_A * Chla**L23_E
            aph_440 = self.L23_A_440 * Chla**self.L23_E_440
            self.a_ph /= aph_440

        #embed(header='498 of anw.py')

        # Extrapolate to <400nm, as necessary
        if wave.min() < 400:
            iwave = np.argmin(np.abs(wave-400))
            wv_ext = wave < 400.
            if len(Chla.shape) == 2:
                a400 = np.outer(self.a_ph[:,iwave], np.ones(np.sum(wv_ext)))
            else:
                a400 = self.a_ph[iwave]
            scl_400 = 2./3
            # 
            if len(Chla.shape) == 2:
                self.a_ph[:,wv_ext] = scl_400*a400 + (
                    np.outer(np.ones(a400.shape[0]), wave[wv_ext]-350) * a400 *
                    (1-scl_400) / 50.)
                    #self.wave[wv_ext]-350) * a400 * (1-scl_400) / 50.
            else:
                self.a_ph[wv_ext] = scl_400*a400 + (
                    wave[wv_ext]-350) * a400 * (1-scl_400) / 50.

    def init_guess(self, a_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        i400 = np.argmin(np.abs(self.wave-400))
        p0_a = np.array([a_nw[i400]/2.])
        assert p0_a.size == self.nparam
        # Return
        return p0_a

class aNWExpBricaud(aNWBricaud):
    """
    Exponential CDOM/detrital + Bricaud phytoplankton absorption model.

    This is the most commonly used absorption model in BING, combining:
    - Exponential decay for dissolved and detrital matter (a_dg)
    - Bricaud et al. (1995) parameterization for phytoplankton (a_ph)

    Model equations:
        a_dg(λ) = Adg × exp(-Sdg × (λ - 400))
        a_ph(λ) = Aph × a*_ph(λ, Chl)
        a_nw(λ) = a_dg(λ) + a_ph(λ)

    where a*_ph is the Bricaud spectral shape normalized at 440 nm.

    Parameters (in fitting space)
    -----------------------------
    Adg : float (log10)
        CDOM + detrital absorption amplitude at 400 nm [m^-1]
    Sdg : float (linear)
        Spectral slope of a_dg, typically 0.010-0.020 [nm^-1]
    Aph : float (log10)
        Phytoplankton absorption amplitude at 440 nm [m^-1]

    Attributes
    ----------
    name : str
        'ExpBricaud'
    nparam : int
        3 (Adg, Sdg, Aph)
    pnames : list
        ['Adg', 'Sdg', 'Aph']
    pivot : float
        Reference wavelength = 400 nm
    uses_Chl : bool
        True - requires Chl for Bricaud shape
    fix_Chl : bool
        False - Chl is derived from fitted Aph

    Notes
    -----
    Chlorophyll is derived from the fitted Aph using:
        Chl = 10^Aph / 0.05582

    where 0.05582 is the Bricaud coefficient at 440 nm for Chl = 1 mg/m³.

    Examples
    --------
    >>> model = aNWExpBricaud(wave)
    >>> model.set_aph(Chl=1.0)  # Initialize Bricaud shape
    >>> params = np.array([-1.5, 0.017, -1.3])  # log10(Adg), Sdg, log10(Aph)
    >>> a_nw = model.eval_anw(params)
    >>> a_dg, a_ph = model.eval_anw(params, retsub_comps=True)
    """
    name = 'ExpBricaud'
    nparam = 3
    pnames = ['Adg', 'Sdg', 'Aph']
    log_params = [True, False, True]
    pivot = 400.
    uses_Chl = True
    fix_Chl = False

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWBricaud.__init__(self, wave, prior_dicts)

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

class aNWExpBricaudFix(aNWExpBricaud):
    """
    Exponential model + Bricaud aph for non-water absorption
        adg = Adg * exp(-Sdg*(wave-400))
        aph = A_B * chlA**B_B

    Here, the Chl is fixed to its provided value, 
        estimated in some other way
        e.g. like GIOP

    Attributes:

    """
    name = 'ExpBricaudFix'
    nparam = 3
    pnames = ['Adg', 'Sdg', 'Aph']
    log_params = [True, False, True]
    pivot = 400.
    uses_Chl = True
    fix_Chl = True

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWExpBricaud.__init__(self, wave, prior_dicts)

class aNWExpBricaudFree(aNWExpBricaud):
    """
    Exponential model + Bricaud aph for non-water absorption
        adg = Adg * exp(-Sdg*(wave-400))
        aph = A_B * chlA**B_B

    Here, the Chl value used to set the shape is a free parameter

    Attributes:

    """
    name = 'ExpBricaudFree'
    nparam = 4
    pnames = ['Adg', 'Sdg', 'Chl', 'Aph'] # Keep Aph last
    log_params = [True, False, True, True]
    pivot = 400.
    uses_Chl = True
    fix_Chl = False

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWExpBricaud.__init__(self, wave, prior_dicts)

    def init_guess(self, a_nw:np.ndarray):
        """
        Initialize the model with a guess

        Parameters:
            a_nw (np.ndarray): The non-water absorption coefficient

        Returns:
            np.ndarray: The initial guess for the parameters
        """
        i400 = np.argmin(np.abs(self.wave-400))
        p0_a = np.array([a_nw[i400]/2., 0.017, a_nw[i400]/2., a_nw[i400]/2.])
        assert p0_a.size == self.nparam
        # Return
        return p0_a

    def set_aph(self, Chla):
        """
        Set the phytoplankton absorption coefficient (a_ph) based on chlorophyll-a concentration (Chla).

        Parameters:
        -----------
        Chla : float or numpy.ndarray
            Chlorophyll-a concentration. Can be a single value (float) or a 1D numpy array.
        

        Attributes Modified:
        --------------------
        self.a_ph : numpy.ndarray
            The phytoplankton absorption coefficient calculated using the Bricaud model.
            If `Chla` is a single value, `self.a_ph` is a 1D array normalized at 440 nm.
            If `Chla` is an array, `self.a_ph` is a 2D array where each row
            corresponds to the absorption spectrum for a specific
            chlorophyll-a concentration. It too is normalized at 440 nm.

        Raises:
        -------
        NotImplementedError
            If extrapolation for multi-dimensional `Chla` is attempted when wavelengths are < 400 nm.

        Notes:
        ------
        - The Bricaud model is used to calculate the absorption coefficient.
        - The absorption coefficient is normalized at 440 nm.
        - For wavelengths < 400 nm, extrapolation is performed using a scaling factor of 2/3
          and a linear adjustment between 350 nm and 400 nm.
        """

        # Bricaud
        if not isinstance(Chla, np.ndarray) or Chla.size == 1:
            self.a_ph = self.L23_A * Chla**self.L23_E
        else:
            #embed(header='aNWExpBricaudFree.set_aph 532')
            # Take an array to an array
            self.a_ph = np.empty((Chla.shape[0], self.wave.size))
            for ss in range(Chla.shape[0]):
                self.a_ph[ss] = self.L23_A * Chla[ss]**self.L23_E

        # Normalize
        if self.a_ph.ndim == 2:
            norm = np.outer(self.a_ph[:,self.i440], np.ones(self.a_ph.shape[1]))
            self.a_ph /= norm
        else:
            self.a_ph /= self.a_ph[self.i440]

        # Extrapolate to <400nm, as necessary
        if self.wave.min() < 400:
            if not oned:
                raise NotImplementedError("Extrapolation for multi-dimensional Chla not implemented")
            iwave = np.argmin(np.abs(self.wave-400))
            a400 = self.a_ph[iwave]
            scl_400 = 2./3
            # 
            wv_ext = self.wave < 400.
            self.a_ph[wv_ext] = scl_400*a400 + (
                self.wave[wv_ext]-350) * a400 * (1-scl_400) / 50.


class aNWGIOP(aNWModel):
    """
    Generalized Inherent Optical Properties (GIOP) absorption model.

    Implements the GIOP algorithm from Werdell et al. (2013). Uses a fixed
    spectral slope for the exponential term to reduce parameter degeneracy.

    Model equations:
        a_dg(λ) = Aexp × exp(-0.018 × (λ - 400))
        a_ph(λ) = Aph × a*_ph(λ, Chl)
        a_nw(λ) = a_dg(λ) + a_ph(λ)

    The spectral slope Sdg = 0.018 nm^-1 is fixed to the global average.

    Parameters (in fitting space)
    -----------------------------
    Aexp : float (log10)
        CDOM + detrital absorption amplitude at 400 nm [m^-1]
    Aph : float (log10)
        Phytoplankton absorption amplitude at 440 nm [m^-1]

    Attributes
    ----------
    name : str
        'GIOP'
    nparam : int
        2 (Aexp, Aph)
    Sdg : float
        Fixed spectral slope = 0.018 nm^-1

    References
    ----------
    Werdell, P.J. et al. (2013). "Generalized ocean color inversion model
    for retrieving marine inherent optical properties," Appl. Opt. 52, 2019-2037.
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

    def set_aph(self, Chla, wave:np.ndarray=None):

        if wave is None:
            wave = self.wave  # Model values
        # ##################################
        # Bricaud
        b1998 = ph_absorption.load_bricaud1998()

        # Interpolate
        f_b1998_A = interp1d(b1998['lambda'], b1998.Aphi, bounds_error=False, fill_value=0.)
        f_b1998_E = interp1d(b1998['lambda'], b1998.Ephi, bounds_error=False, fill_value=0.)

        # Apply
        L23_A = f_b1998_A(wave)
        L23_E = f_b1998_E(wave)

        self.a_ph = L23_A * Chla**L23_E

        # Normalize at 440
        L23_A = f_b1998_A(440.)
        L23_E = f_b1998_E(440.)
        a_ph_440 = L23_A * Chla**L23_E
        self.a_ph /= a_ph_440

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
        aexp = Aexp * exp(-Sdg*(wave-400))
        aph = H1*W1 + H2*W2

    """
    name = 'ExpNMF'
    nparam = 4
    pnames = ['Aexp', 'Sdg', 'H1', 'H2']
    log_params = [True, False, True, True]
    pivot = 400.

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWModel.__init__(self, wave, prior_dicts)

        # Set the basis functions
        self.set_w1w2()

    def set_w1w2(self):

        # Hiding this import here to avoid making
        #  CNMF a requirement for the package
        from cnmf import io as cnmf_io

        # ##################################
        # NMF for aph
        # Load the decomposition of aph
        aph_file = cnmf_io.pcanmf_filename('L23', 'NMF', 2, 'aph')
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

    Exponential model with Sdg fixed + Bricaud aph for non-water
    absorption::

        adg = Adg * exp(-Sdg*(wave-400))    # Sdg = 0.0206
        aph = Chl * a_ph*                   # a_ph* interpolated from
                                            # Maritorena+2002 values

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

    def set_aph(self, Chla, wave:np.ndarray=None, version:str='Maritorena2002'):

        if wave is None:
            wave = self.wave  # Model values

        if version == 'Maritorena2002':
            # ##################################
            # Maritorena+2002
            interp_wv = [412., 443., 490., 510., 555.]
            interp_aph_star = [0.00665, 0.05582, 0.02055, 0.01910, 0.01015]

            # Interpolate
            f = interp1d(interp_wv, interp_aph_star, kind='linear', fill_value='extrapolate')
        else:
            raise ValueError(f"Unknown aph* version: {version}")

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

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWModel.__init__(self, wave, prior_dicts)

        # Priors
        self.ngauss = 8
        self.init_priors()

        # Set
        self.prior_dicts = prior_dicts

    def init_priors(self):
        self.priors = bing_priors.Priors(self.prior_dicts)

    def eval_adg(self, params:np.ndarray):
        """
        Evaluate the CDOM and NAP components

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
        return aNAP + aCDOM

    def eval_anw(self, params:np.ndarray):
        """
        Evaluate the non-water absorption coefficient

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The non-water absorption coefficient
                This is always a multi-dimensional array
        """
        # adg
        atot = self.eval_adg(params)

        # Aph
        aphs = params[...,4:12]
        sigs = params[...,12:20]
        cens = params[...,20:]

        for igaus in range(self.ngauss):
            # Repackage
            params = np.array([aphs[...,igaus], sigs[...,igaus], cens[...,igaus]]).T
            #embed(header='gaus 626 anw')
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

class aNWChaseMini(aNWChase):
    """
    Chase+2017 with fixed cen and sigma 

    Exponential model with Sdg fixed + Bricaud aph for non-water absorption
        aNAP = CNAP * exp(-SNAP*(wave-400))
        aCDOM = CCDOM * exp(-SCDOM*(wave-400))

        aph = Sum aph_i * exp((wave-wave_i)**2/2/sigma_i**2)
            8 Gaussians
                wave_i, sigma_i fixed

    Attributes:

    """
    name = 'Chase2017Mini'
    nparam = 12
    pnames = ['CNAP', 'SNAP', 'CCDOM', 'SCDOM', 
              'aph384', 'aph413', 'aph435', 'aph461', 
              'aph464', 'aph490', 'aph532', 'aph583'] 
    pivot = 400.

    def __init__(self, wave:np.ndarray, prior_dicts:list=None):
        aNWChase.__init__(self, wave, prior_dicts)

        # Priors
        self.ngauss = 8
        self.init_priors()

    def init_priors(self):
        prior_dicts = super().prior_dicts
        # Subset
        self.prior_dicts = prior_dicts[0:self.nparam]
        # Set
        self.priors = bing_priors.Priors(self.prior_dicts)

    def eval_anw(self, params:np.ndarray):
        """
        Evaluate the non-water absorption coefficient

        Parameters:
            params (np.ndarray): The parameters for the model

        Returns:
            np.ndarray: The non-water absorption coefficient
                This is always a multi-dimensional array
        """
        # adg
        atot = self.eval_adg(params)

        # Aph
        aphs = params[...,4:12]

        # Set the fixed values
        prior_dicts = super().prior_dicts
        def calc_mid(pdict):
            return (pdict['pmin'] + pdict['pmax'])/2.
        sigs = np.array([calc_mid(prior_dicts[12+ii]) for ii in range(8)])
        cens = np.array([calc_mid(prior_dicts[20+ii]) for ii in range(8)])
        if params.ndim == 1:
            pass
        else: # Chains
            sigs = np.outer(np.ones(params.shape[0]), sigs)
            cens = np.outer(np.ones(params.shape[0]), cens)

        # Do it
        for igaus in range(self.ngauss):
            # Repackage
            params = np.array([aphs[...,igaus], sigs[...,igaus], cens[...,igaus]]).T
            #embed(header='gaus 626 anw')
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
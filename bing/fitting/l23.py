
from collections import namedtuple
import os

import numpy as np
import pandas

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ocpy.satellites import modis as sat_modis
from ocpy.satellites import seawifs as sat_seawifs
from ocpy.satellites import pace as sat_pace
from ocpy.hydrolight import loisel23

from bing import rt as bing_rt
from bing.models import utils as model_utils
from bing.models import functions
from bing.priors import priors as bing_priors
from bing.fitting import inference as bing_inf

from bing.preproc import convert_to_satwave
from bing.noise import scale_noise, add_noise

#import anly_utils_20


def load_one_l23(idx:int, step:int=1, 
                  ds=None, 
                  wv_max:float=None, 
                  wv_min:float=None):
    """
    Load and process data for a single index from the Loisel 2023 dataset.

    This function extracts and processes various oceanographic parameters 
    such as remote sensing reflectance (Rrs), absorption coefficients, 
    backscattering coefficients, and other related properties for a given 
    index. It also computes derived quantities like chlorophyll concentration 
    and spectral slope of dissolved and detrital absorption (Sdg).

    Parameters:
        idx (int): Index of the dataset to process.
        step (int, optional): Step size for downsampling the wavelength bands. 
                                Defaults to 1 (no downsampling).
        ds (xarray.Dataset, optional): Dataset to load the data from. If None, 
                                        the function will load the default 
                                        Loisel 2023 dataset. Defaults to None.
        wv_max (float, optional): Maximum wavelength to include in the 
                                    processing. Defaults to None.
        wv_min (float, optional): Minimum wavelength to include in the 
                                    processing. Defaults to None.

    Returns:
        dict: A dictionary containing the following keys:
            - wave (numpy.ndarray): Processed wavelengths after downsampling.
            - Rrs (numpy.ndarray): Remote sensing reflectance values.
            - a (numpy.ndarray): Absorption coefficients.
            - bb (numpy.ndarray): Backscattering coefficients.
            - true_wave (numpy.ndarray): Original wavelengths before processing.
            - true_Rrs (numpy.ndarray): Original Rrs values before processing.
            - gordon_Rrs (numpy.ndarray): Rrs values computed using Gordon's model.
            - bbw (numpy.ndarray): Backscattering coefficients of water.
            - bbnw (numpy.ndarray): Backscattering coefficients of non-water components.
            - aw (numpy.ndarray): Absorption coefficients of water.
            - anw (numpy.ndarray): Absorption coefficients of non-water components.
            - adg (numpy.ndarray): Combined absorption of dissolved and detrital matter.
            - ag (numpy.ndarray): Absorption by dissolved material
            - aph (numpy.ndarray): Phytoplankton absorption coefficients.
            - Sdg (float): Spectral slope of dissolved and detrital absorption.
            - Y (float): Spectral slope parameter for backscattering.
            - Chl (float): Chlorophyll concentration.

    Notes:
        - The function uses the Lee et al. (2002) prescription for computing 
            the spectral slope parameter (Y).
        - Chlorophyll concentration is estimated using the absorption at 
            440 nm and a predefined coefficient.
        - The function fits the spectral slope of dissolved and detrital 
            absorption (Sdg) using a custom fitting function.
    """

    # Load
    if ds is None:
        ds = loisel23.load_ds(4,0)

    # Wavelengths
    wave = ds.Lambda.data

    gd_wave = np.ones_like(ds.Lambda.data, dtype=bool)
    if wv_max is not None:
        gd_wave &= ds.Lambda.data <= wv_max
    if wv_min is not None:
        gd_wave &= ds.Lambda.data >= wv_min
    iwave = np.where(gd_wave)[0]

    # Grab
    Rrs = ds.Rrs.data[idx,iwave]
    wave = wave[iwave]
    true_Rrs = Rrs.copy()
    true_wave = wave.copy()
    a = ds.a.data[idx,iwave]
    bb = ds.bb.data[idx,iwave]
    adg = ds.ag.data[idx,iwave] + ds.ad.data[idx,iwave]
    aph = ds.aph.data[idx,iwave]
    ag = ds.ag.data[idx,iwave] 

    # For bp: Lee+2002 prescription
    rrs = Rrs / (bing_rt.A_Rrs + bing_rt.B_Rrs*Rrs)
    i440 = np.argmin(np.abs(true_wave-440))
    i555 = np.argmin(np.abs(true_wave-555))
    Y = 2.2 * (1 - 1.2 * np.exp(-0.9 * rrs[i440]/rrs[i555]))

    # For aph
    aph = ds.aph.data[idx,iwave]
    Chl = aph[i440] / 0.05582

    # For adg
    ans, _ = functions.fit_Sdg(wave, adg,
                                 wv_min=wv_min)

    # Cut down to 40 bands
    Rrs = Rrs[::step]
    wave = wave[::step]

    # Gordon
    gordon_Rrs = bing_rt.calc_Rrs(a, bb)

    # Error
    #varRrs = (scl_noise * Rrs)**2

    # Dict me
    odict = dict(wave=wave, Rrs=Rrs, a=a, bb=bb, 
                 true_wave=true_wave, true_Rrs=true_Rrs,
                 gordon_Rrs=gordon_Rrs,
                 bbw=ds.bb.data[idx,iwave]-ds.bbnw.data[idx,iwave],
                 bbnw=ds.bbnw.data[idx,iwave],
                 aw=ds.a.data[idx,iwave]-ds.anw.data[idx,iwave],
                 anw=ds.anw.data[idx,iwave],
                 adg=adg, aph=aph, Sdg=float(ans[1]),
                 ag=ag,
                 Y=Y, Chl=Chl)

    return odict

def prep_one_l23(p, idx, chk:bool=False):
    """
    Prepare data and models for L23 fitting.
    This function initializes the necessary data, models, priors, and MCMC 
    parameters for fitting L23 data. It also handles wavelength conversions, 
    noise scaling, and initial guesses for the fitting process.
    Args:
        p (object): Parameter object containing configuration settings such as 
            wavelength range (`wv_min`, `wv_max`), satellite type (`satellite`), 
            model names (`model_names`), number of MCMC steps (`nsteps`), 
            burn-in steps (`nburn`), and other prior settings.
        idx (int): Index used to load specific L23 data.
        chk (bool, optional): If True, checks the initial guess for Rrs and 
            prints the relative difference. Defaults to False.
    Returns:
        dict: A dictionary containing the following keys:
            - 'odict': Dictionary with loaded L23 data.
            - 'model_Rrs': Modeled remote sensing reflectance (Rrs).
            - 'model_varRrs': Variance of the modeled Rrs after scaling noise.
            - 'p0': Initial guess for model parameters.
            - 'pdict': Dictionary containing MCMC initialization parameters.
            - 'models': List of initialized models for fitting.
    Raises:
        ValueError: If the satellite type specified in `p.satellite` is unknown.
    """
    odict = load_one_l23(idx, wv_min=p.wv_min, wv_max=p.wv_max)

    # Set power-law
    if p.beta is not None:
        odict['Y'] = p.beta

    # Unpack
    wave = odict['wave']
    l23_wave = odict['true_wave']

    # Wavelenegths
    if p.satellite == 'MODIS':
        model_wave = sat_modis.modis_wave
    elif p.satellite in ['PACE', 'SBG']:
        model_wave = sat_pace.wave(wv_min=p.wv_min,
                                   wv_max=p.wv_max)
    elif p.satellite == 'SeaWiFS':
        model_wave = sat_seawifs.seawifs_wave
    elif p.satellite == 'L23':
        model_wave = wave
    else:
        raise ValueError(f"Unknown option: {p.satellite}")

    # Priors
    if p.model_names[0] == 'ExpB':
        use_model_names = ['Exp', p.model_names[1]]
    else:
        use_model_names = p.model_names.copy()

    # Models
    models = model_utils.init(use_model_names, model_wave)

    # Set priors
    bing_priors.set_standard_priors(models, p)

    # Extra priors?
    if p.othera_priors is not None:
        for prior_dict in p.othera_priors:
            # Append
            models[0].priors.add_prior(prior_dict)

    # Initialize the MCMC
    pdict = bing_inf.init_mcmc(models, nsteps=p.nsteps, nburn=p.nburn)
    
    # Gordon Rrs
    gordon_Rrs = bing_rt.calc_Rrs(odict['a'], odict['bb'])

    model_Rrs = convert_to_satwave(l23_wave, gordon_Rrs, model_wave)
    model_anw = convert_to_satwave(l23_wave, odict['anw'], model_wave)
    model_bbnw = convert_to_satwave(l23_wave, odict['bbnw'], model_wave)
    model_varRrs = scale_noise(p.scl_noise, model_Rrs, model_wave)

    orig_model_Rrs = model_Rrs.copy()
    if p.add_noise:
        model_Rrs = add_noise(
                orig_model_Rrs, abs_sig=np.sqrt(model_varRrs))

    # Internals (some of which depend on Rrs)
    _ = model_utils.init_other_bits(
        models, Chl=odict['Chl'], Y=odict['Y'],
        update_dict=odict, Rrs=model_Rrs)

    # Initial guess
    p0_a = models[0].init_guess(model_anw)
    p0_b = models[1].init_guess(model_bbnw)
    p0 = np.concatenate((np.atleast_1d(p0_a), 
                         np.atleast_1d(p0_b)))

    # Log 10
    cnt = 0
    for ss in [0,1]:
        for prior in models[ss].priors.priors:
            if prior.flavor[0:3] == 'log':
                p0[cnt] = np.log10(p0[cnt])
            cnt += 1

    # Chk initial guess
    if chk:
        ca = models[0].eval_a(p0[0:models[0].nparam])
        cbb = models[1].eval_bb(p0[models[0].nparam:])
        pRrs = bing_rt.calc_Rrs(ca, cbb)
        print(f'Initial Rrs guess: {np.mean((model_Rrs-pRrs)/model_Rrs)}')

    # Return a dictionary
    ret_dict = {}
    ret_dict['odict'] = odict
    ret_dict['model_Rrs'] = model_Rrs
    ret_dict['model_varRrs'] = model_varRrs
    ret_dict['p0'] = p0
    ret_dict['pdict'] = pdict
    ret_dict['models'] = models

    return ret_dict
    

def fit_one(p:namedtuple, idx:int,
        debug:bool=False):
    """
    Fits a model to the data for a given index.

    Args:
        model_names (list): List of model names.
        idx (int): Index of the data.
    Returns:
        tuple: Tuple containing the fitted parameters,
            models, the index, and additional information.
    """
    # Prep and unpack
    prep_dict = prep_one_l23(p, idx)
    odict = prep_dict['odict']
    pdict = prep_dict['pdict']
    models = prep_dict['models']
    model_Rrs = prep_dict['model_Rrs']
    model_varRrs = prep_dict['model_varRrs']
    model_wave = models[0].wave
    p0 = prep_dict['p0']
    l23_wave = odict['true_wave']

    # pdict -- this is a hack for a single run
    pdict['Chl'] = np.zeros(idx+1)
    pdict['Chl'][idx] = odict['Chl']
    pdict['Y'] = np.zeros(idx+1)
    pdict['Y'][idx] = odict['Y']

    # Set the items
    #p0 -= 1
    items = [(model_Rrs, model_varRrs, p0, idx)]


    # Fit
    chains, idx = bing_inf.fit_one(
            items[0], models=models, pdict=pdict, chains_only=True)

    # Show?
    if False:
        plt.show()

        if p.burn > chains.shape[0]:
            embed(header='210 of dev_fits')
        thin = 1
        coeff = chains[p.burn::thin, :, :].reshape(-1, chains.shape[-1])

        # Corner plot
        # Labels
        clbls = models[0].pnames + models[1].pnames
        # Add log 10
        clbls = [r'$\log_{10}('+f'{clbl}'+r'$)' for clbl in clbls]
        fig = corner.corner(
            coeff, labels=clbls,
            label_kwargs={'fontsize':17},
            color='k',
            #axes_scale='log',
            truths=None, #truths,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            )
        # Add 90%
        ss = 0
        for ax in fig.get_axes():
            if len(ax.get_title()) > 0:
                # Calculate the percntile
                p_5, p_95 = np.percentile(coeff[:,ss], [5, 95], axis=0)
                # Plot a vertical line
                ax.axvline(p_5, color='b', linestyle=':')
                ax.axvline(p_95, color='b', linestyle=':')
                ss += 1
        plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
        plt.show()

        # a_nw
        bing_plot.show_anw_fits(
            models, coeff,
            anw_true=dict(
                wave=l23_wave, a_dg=odict['adg'],
                a_ph=odict['aph']),
            perc=(16, 84))

        if debug:
            embed(header='268 of dev')

    # Extras``
    extras=dict(wave=model_wave,
                obs_Rrs=model_Rrs,
                varRrs=model_varRrs,
                Chl=odict['Chl'],
                Y=odict['Y'])

    # Return
    return chains, models, prep_dict, idx, extras



def batch_fit(p, n_batch:int=5, n_cores:int=15, debug:bool=False,
        seed:bool=None, out_dir:str='../Analysis/Fits/'): 
    """
    Fits the data with or without considering any errors.

    Args:
        p (namedtuple): Parameters for the fit.
            Required attributes:
                - model_names (list): List of model names.
                - satellite (str): Satellite name.
                - add_noise (bool): Whether to add noise.
                - scl_noise (float): Scale of the noise.
                - wv_min (int or None): Minimum wavelength.
                - set_Sdg (bool): Whether to set Sdg.
                - sSdg (float): Sdg value.
                - beta (float or None): Beta value.
                - nMC (int or None): Number of Monte Carlo simulations.
        n_batch (int): Number of batches to process in parallel.
        seed (int): Random seed for reproducibility.
        debug (bool): Whether to run in debug mode. Default is False.
        n_cores (int): The number of CPU cores to use for parallel processing. Default is 1.
        out_dir (str): The directory to save the output files. Default is '../Analysis/Fits/'.

    """
    if seed is not None:
        np.random.seed(seed)

    # Load L23
    ds = loisel23.load_ds(4,0)
    # Prep
    all_idx = np.arange(ds.Rrs.shape[0]).tolist()
    if debug:
        #idx = idx[0:2]
        all_idx = [170, 180, 200, 250]
        #idx = [2706]

    Rrs = []
    varRrs = []
    params = []
    Chls = []
    Ys = []
    for idx in all_idx:
        if idx % 50 == 0:
            print("Working on {:d}".format(idx))
        prep_dict = prep_one_l23(p, idx)

        # Others
        varRrs.append(prep_dict['model_varRrs'])
        Rrs.append(prep_dict['model_Rrs'])
        Chls.append(prep_dict['odict']['Chl'])
        Ys.append(prep_dict['odict']['Y'])
        params.append(prep_dict['p0'])

    # Arrays
    Rrs = np.array(Rrs)
    params = np.array(params)
    varRrs = np.array(varRrs)

    # Add to pdict
    pdict = prep_dict['pdict']
    max_idx = np.max(all_idx)
    # Brutal kludge for multi-processing
    pdict['Chl'] = np.zeros(max_idx+1)
    pdict['Y'] = np.zeros(max_idx+1)
    for ss, idx in enumerate(all_idx):
        pdict['Chl'][idx] = Chls[ss]
        pdict['Y'][idx] = Ys[ss]

    # Models
    models = prep_dict['models']

    i0 = 0
    while i0 < len(all_idx):
        # Batch
        i1 = i0 + n_batch*n_cores
        i1 = min(i1, len(all_idx))
        print("Working on {:d} to {:d}".format(i0, i1))

        # Build the items
        items = []
        for idx in all_idx[i0:i1]:
            ss = all_idx.index(idx)
            item = (Rrs[ss], varRrs[ss], params[ss], idx)
            items.append(item)

        # Fit
        all_samples, sub_idx = bing_inf.fit_batch(
            models, pdict, items, n_cores=n_cores)

        # Check
        assert np.all([item[3] for item in items] == sub_idx)

        # Output
        for ss, item in enumerate(items):
            # Unpack
            iRrs, ivarRrs, iparams, idx = item
            chains = all_samples[ss]
            outfile = chain_filename(p, idx=idx, path=out_dir)
            save_chains(chains, idx, outfile, 
                            extras=dict(wave=models[0].wave, 
                                        obs_Rrs=iRrs, 
                                        varRrs=ivarRrs, 
                                        Chl=pdict['Chl'][idx],
                                        Y=pdict['Y'][idx]))

        # Update i0
        i0 = i1

    print("Done!")


def process_one(idx, pdict=None, perc=(16, 84), burn:int=7000, thin:int=1,
                verbose:bool=False):
    """
    Processes a single dataset by reconstructing model parameters and computing statistics.

    Args:
        idx (int): Index of the dataset to process.
        pdict (dict, optional): Dictionary of parameters used for processing. Keys should match 
            the expected attributes of the named tuple `BING20_tuple`.
        perc (tuple, optional): Percentile range for uncertainty estimation (default is (16, 84)).
        burn (int, optional): Number of initial samples to discard from the chains (default is 7000).
        thin (int, optional): Thinning factor for the chains (default is 1).
        verbose (bool, optional): If True, prints additional information during processing (default is False).

    Returns:
        tuple:
            - standard (dict): A dictionary containing standard statistics for the dataset, including:
                - `bbp_440` (float): Backscattering coefficient at 440 nm.
                - `sig_bbp_440` (float): Uncertainty of `bbp_440`.
                - `aph_440` (float): Phytoplankton absorption coefficient at 440 nm.
                - `sig_aph_440` (float): Uncertainty of `aph_440`.
                - `adg_440` (float): Detrital and gelbstoff absorption coefficient at 440 nm.
                - `sig_adg_440` (float): Uncertainty of `adg_440`.
            - extras (dict): A dictionary containing additional statistics, if available, including:
                - `Sdg` (float): Median value of the Sdg parameter (if present in the model).
                - `sig_Sdg` (float): Uncertainty of `Sdg`.
                - `beta` (float): Median value of the beta parameter (if present in the model).
                - `sig_beta` (float): Uncertainty of `beta`.

    Notes:
        - The function loads model chains from a file, reconstructs model parameters, and computes
            statistics such as medians and percentiles for key parameters.
        - The function assumes that the input `pdict` contains all necessary keys for initializing
            the named tuple and models.
        - The function uses the `burn` and `thin` parameters to preprocess the chains before
            computing statistics.
    """

    MyNamedTuple = namedtuple('BING20_tuple', pdict.keys())
    p = MyNamedTuple(**pdict)

    # Load up
    odict = load_one_l23(
        idx, wv_min=p.wv_min, wv_max=p.wv_max)

    model_wave = sat_pace.wave(
        wv_min=p.wv_min, wv_max=p.wv_max)
    models = model_utils.init(p.model_names, model_wave)

    # Load chains
    chain_file = anly_utils_20.chain_filename(p, idx=idx)
    if verbose:
        print(f"Loading chains from {chain_file}")
                                              #path='../../bing_2.0/Analysis/Fits')
    d = np.load(chain_file)
    chains = d['chains']

    # Init the other stuff..
    _ = model_utils.init_other_bits(models, Chl=d['Chl'], Y=d['Y'])


    # Reconstruct
    a_mean, bb_mean, a_low, anw_high, bb_low, bb_high,\
            model_Rrs, sigRs = evaluate.reconstruct_from_chains(
            models, chains, perc=perc)

    # a_ph, a_dg
    prep_chains = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])

    a_dg, a_ph = models[0].eval_anw(prep_chains[..., :models[0].nparam], retsub_comps=True)
    adg_mean = np.median(a_dg, axis=0)
    adg_low, adg_high = np.percentile(a_dg, perc, axis=0)
    aph_mean = np.median(a_ph, axis=0)
    aph_low, aph_high = np.percentile(a_ph, perc, axis=0)

    # Stats
    i440 = np.argmin(np.abs(models[0].wave - 440))

    bbp_440 = bb_mean[i440] - bbw_440
    sig_bbp_440 = 0.5*(bb_high[i440] - bb_low[i440])

    aph_440 = aph_mean[i440]
    sig_aph_440 = 0.5*(aph_high[i440] - aph_low[i440])

    adg_440 = adg_mean[i440]
    sig_adg_440 = 0.5*(adg_high[i440] - adg_low[i440])

    # Generate a simple dict
    standard = dict(bbp_440=bbp_440, sig_bbp_440=sig_bbp_440,
                    aph_440=aph_440, sig_aph_440=sig_aph_440,
                    adg_440=adg_440, sig_adg_440=sig_adg_440)

    # Extras
    extras = {}
    if 'Sdg' in models[0].pnames:
        iSdg = models[0].pnames.index('Sdg')
        extras['Sdg'] = np.median(prep_chains[:, iSdg])
        Sdg_low, Sdg_high = np.percentile(prep_chains[:, iSdg], perc)
        extras['sig_Sdg'] = 0.5*(Sdg_high - Sdg_low)
    if 'beta' in models[1].pnames:
        ibeta = models[1].pnames.index('beta')
        extras['beta'] = np.median(prep_chains[:, models[0].nparam+ibeta])
        beta_low, beta_high = np.percentile(prep_chains[:, models[0].nparam+ibeta], perc)
        extras['sig_beta'] = 0.5*(beta_high - beta_low)

    # Return
    return standard, extras



def process_all(p, outfile:str, n_cores:int=15, debug:bool=False):
    """
    Processes a range of items in parallel, aggregates the results, and saves them to a CSV file.

    Args:
        p: A parameter object containing necessary data for processing. It should have a `_asdict()` method.
        outfile (str): The path to the output CSV file where the results will be saved.
        n_cores (int, optional): The number of CPU cores to use for parallel processing. Defaults to 15.
        debug (bool, optional): If True, processes a smaller range of items for debugging purposes. Defaults to False.

    Returns:
        None

    Notes:
        - The function uses a `ProcessPoolExecutor` to process items in parallel.
        - The `process_one` function is expected to return a tuple containing two dictionaries: 
            `standard` and `extras`. These are aggregated into a single dictionary (`big_dict`).
        - The aggregated results are saved as a CSV file at the specified `outfile` location.
    """

    map_fn = partial(process_one, pdict=p._asdict())

    # Item me
    items = np.arange(0, 3320)
    if debug:
        items = np.arange(0, 30)

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, items,
                                            chunksize=chunksize), total=len(items)))

    # Unpack
    big_dict = {}
    for ss in items:
        # Unpack
        standard, extras = answers[ss]
        # Save
        for key in standard.keys():
            if key not in big_dict.keys():
                big_dict[key] = []
            big_dict[key].append(standard[key])
        for key in extras.keys():
            if key not in big_dict.keys():
                big_dict[key] = []
            big_dict[key].append(extras[key])

    # Save as CSV
    df = pandas.DataFrame(big_dict)
    df.to_csv(outfile, index=False)
    print(f'Saved: {outfile}')



def chain_filename(p:namedtuple, idx:int=None, path:str='../Analysis/Fits/'): 
    """
    Generate a filename for saving chain data based on the parameters provided.

    Args:
        p (namedtuple): A namedtuple containing the following attributes:
            - model_names (list): A list of model names (at least two elements).
            - satellite (str): The satellite name, one of 'MODIS', 'PACE', 'SBG', or 'SeaWiFS'.
            - add_noise (bool): Whether noise is added to the data.
            - scl_noise (str or float): Noise scaling factor or satellite name ('SeaWiFS', 'MODIS_Aqua', 'PACE', 'SBG').
            - wv_min (int or None): Minimum wavelength value for UV fussing.
            - set_Sdg (bool): Whether Sdg is set.
            - sSdg (float): Sdg value (used if set_Sdg is True).
            - beta (float or None): Beta value.
            - nMC (int or None): Number of Monte Carlo simulations (if applicable).
        idx (int, optional): Index to append to the filename. Defaults to None.
        path (str, optional): Base directory path for the file. Defaults to '../Analysis/Fits/'.

    Returns:
        str: The generated filename with the appropriate suffixes based on the input parameters.

    Notes:
        - The filename is constructed using the model names, satellite type, noise settings, 
          UV fussing, Sdg, beta, and Monte Carlo simulation flag.
        - The file extension is '.npz'.
    """
    outfile = os.path.join(path, f'BING20_{p.model_names[0]}{p.model_names[1]}')

    if idx is not None:
        outfile += f'_{idx}'
        if p.satellite == 'MODIS':
            outfile += '_M'
        elif p.satellite == 'PACE':
            outfile += '_P'
        elif p.satellite == 'SBG':
            outfile += '_B'
        elif p.satellite == 'SeaWiFS':
            outfile += '_S'
    else:
        if p.satellite == 'MODIS':
            outfile += '_M23'
        elif p.satellite == 'PACE':
            outfile += '_P23'
        elif p.satellite == 'SBG':
            outfile += '_B23'
        elif p.satellite == 'SeaWiFS':
            outfile += '_S23'
        else:
            outfile += '_L23'
    # Added?
    if p.add_noise:
        outfile += '_N'
    else:
        outfile += '_n'

    # Value
    if p.scl_noise == 'SeaWiFS':
        outfile += 'S'
    elif p.scl_noise == 'MODIS_Aqua':
        outfile += 'M'
    elif p.scl_noise == 'PACE':
        outfile += 'P'
    elif p.scl_noise == 'SBG':
        outfile += 'B'
    else:
        outfile += f'{int(100*p.scl_noise):02d}'

    # UV fussing
    if p.wv_min is not None:
        outfile += f'_UV{int(p.wv_min)}'

    # Sdg
    if p.set_Sdg: 
        outfile += f'_Sdg{int(1000*p.sSdg)}'
    else:
        outfile += f'_SdgU'

    # beta
    if p.beta is not None:
        outfile += f'_b{p.beta:0.1f}'

    # Monte Carlo?
    if p.nMC is not None:
        outfile += f'_MC'

    outfile += '.npz'
    return outfile



def save_chains(all_samples, all_idx, outfile, 
              extras:dict=None):
    """
    Save the fitting results to a file.

    Parameters:
        all_samples (numpy.ndarray): Array of fitting chains.
        all_idx (numpy.ndarray): Array of indices.
        Rs (numpy.ndarray): Array of Rs values.
        use_Rs (numpy.ndarray): Array of observed Rs values.
        outroot (str): Root name for the output file.
    """  
    # Outdict
    outdict = dict()
    outdict['chains'] = all_samples
    outdict['idx'] = all_idx
    
    # Extras
    if extras is not None:
        for key in extras.keys():
            outdict[key] = extras[key]
    np.savez(outfile, **outdict)
    print(f"Saved: {outfile}")
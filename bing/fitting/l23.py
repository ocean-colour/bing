
from collections import namedtuple

import numpy as np

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
    """ Prepare L23 the data for the fit """

    # Load
    if ds is None:
        ds = loisel23.load_ds(4,0)

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
                 Y=Y, Chl=Chl)

    return odict

def prep_one_l23(p, idx, chk:bool=False):

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
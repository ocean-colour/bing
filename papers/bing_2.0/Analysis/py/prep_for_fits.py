import numpy as np

from ocpy.satellites import modis as sat_modis
from ocpy.satellites import seawifs as sat_seawifs

from bing import rt as bing_rt
from bing.models import utils as model_utils
from bing.priors import priors as bing_priors
from bing import inference as bing_inf

import anly_utils_20

def one_l23(p, idx, chk:bool=False):

    odict = anly_utils_20.prep_l23_data(
        idx, wv_min=p.wv_min, wv_max=p.wv_max)

    # Set power-law
    if p.beta is not None:
        odict['Y'] = p.beta

    # Unpack
    wave = odict['wave']
    l23_wave = odict['true_wave']

    # Wavelenegths
    if p.satellite == 'MODIS':
        model_wave = sat_modis.modis_wave
    elif p.satellite == 'PACE':
        model_wave = anly_utils_20.pace_wave(wv_min=p.wv_min,
                                             wv_max=p.wv_max)
    elif p.satellite == 'SeaWiFS':
        model_wave = sat_seawifs.seawifs_wave
    else:
        model_wave = wave

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

    model_Rrs = anly_utils_20.convert_to_satwave(l23_wave, gordon_Rrs, model_wave)
    model_anw = anly_utils_20.convert_to_satwave(l23_wave, odict['anw'], model_wave)
    model_bbnw = anly_utils_20.convert_to_satwave(l23_wave, odict['bbnw'], model_wave)
    model_varRrs = anly_utils_20.scale_noise(p.scl_noise, model_Rrs, model_wave)

    orig_model_Rrs = model_Rrs.copy()
    if p.add_noise:
        model_Rrs = anly_utils_20.add_noise(
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
    
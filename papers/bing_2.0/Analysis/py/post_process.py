""" Algorithms to post-process the results of the Bing 2.0 analysis. """

from collections import namedtuple
import numpy as np

from ocpy.utils import io as ocio

from bing.models import utils as model_utils

import anly_utils_20
import param

from IPython import embed

def analyze_chains(p:namedtuple, idx:int,
                   debug:bool=False):
    """ Analyze the MCMC chains for the given parameters. """

    # Load L23
    odict = anly_utils_20.prep_l23_data(
        idx, wv_min=p.wv_min, wv_max=p.wv_max) 

    if p.satellite == 'PACE':
        model_wave = anly_utils_20.pace_wave(wv_min=p.wv_min,
                                             wv_max=p.wv_max)
    else:
        raise NotImplementedError('Only PACE is implemented')                                            

    # Models
    models = model_utils.init(p.model_names, model_wave)
    if models[0].uses_Chl:
        models[0].set_aph(odict['Chl'])
    
    # Chain file
    chain_file = anly_utils_20.chain_filename(p, idx=idx)
    # Load
    d = np.load(chain_file)

    burn = 7000
    thin = 1
    #chains = d['chains'][:,burn::thin, :, :].reshape(-1, d['chains'].shape[-1])
    nMC = d['chains'].shape[0]
    if debug:
        nMC = 2
    chains = d['chains'][:nMC,burn::thin, :, :].reshape(-1, d['chains'].shape[-1])
    
    # TESTING
    a_dg, a_ph = models[0].eval_anw(
        chains[..., :models[0].nparam], 
        retsub_comps=True)
    # Reshape
    a_dg = a_dg.reshape(nMC, -1, model_wave.size)
    a_ph = a_ph.reshape(nMC, -1, model_wave.size)

    # Wavelengths
    i400 = np.argmin(np.abs(model_wave-400))
    i440 = np.argmin(np.abs(model_wave-440))

    # Stats
    all_adg_400 = np.median(a_dg[..., i400],axis=1)
    all_aph_440 = np.median(a_ph[..., i440],axis=1)
    adg_400 = np.median(all_adg_400)
    aph_440 = np.median(all_aph_440)
    adg_5, adg_95 = np.percentile(all_adg_400, [16, 84])
    aph_5, aph_95 = np.percentile(all_aph_440, [16, 84])

    # Save
    sdict = dict(adg_400=adg_400, adg_5=adg_5, adg_95=adg_95,
                 aph_440=aph_440, aph_5=aph_5, aph_95=aph_95)

    jdict = ocio.jsonify(sdict)
    outfile = chain_file.replace('.npz', '.json')
    ocio.savejson(outfile, jdict, overwrite=True)
    print(f'Saved: {outfile}')

    # Return
    return odict, sdict

# Command line execution
if __name__ == '__main__':
    p = param.p_ntuple(['ExpBricaud', 'Pow'], 
            set_Sdg=True, sSdg=0.002, beta=1., nMC=100,
            add_noise=True)
    analyze_chains(p, 170)#, debug=True)
""" Algorithms to post-process the results of the Bing 2.0 analysis. """

from collections import namedtuple
import os
import numpy as np

from ocpy.utils import io as ocio

from matplotlib import pyplot as plt
import seaborn as sns

from bing.models import utils as model_utils

import anly_utils_20
import param

from IPython import embed

def analyze_chains(p:namedtuple, idx:int,
                   debug:bool=False,
                   clobber:bool=False):
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
    # Output
    outfile = chain_file.replace('.npz', '.json')
    if not clobber and os.path.isfile(outfile):
        print(f'File exists.  Skipping: {outfile}')
        return
    else:
        print(f'Working on: {outfile}')

    # Right answer
    l23_wave = odict['wave']
    l23_i400 = np.argmin(np.abs(l23_wave-400))
    l23_i440 = np.argmin(np.abs(l23_wave-440))
    l23_adg_400 = odict['adg'][l23_i400]
    l23_aph_440 = odict['aph'][l23_i440]

    # Load
    d = np.load(chain_file)
    print('Loaded chains:', d['chains'].shape)

    burn = 7000
    thin = 1
    #chains = d['chains'][:,burn::thin, :, :].reshape(-1, d['chains'].shape[-1])
    nMC = d['chains'].shape[0]
    if debug:
        nMC = 2
    chains = d['chains'][:nMC,burn::thin, :, :].reshape(-1, d['chains'].shape[-1])
    
    # Generate the ANW components
    a_dg, a_ph = models[0].eval_anw(
        chains[..., :models[0].nparam], 
        retsub_comps=True)
    print('Done with eval_anw')
    # Reshape
    a_dg = a_dg.reshape(nMC, -1, model_wave.size)
    a_ph = a_ph.reshape(nMC, -1, model_wave.size)
    #embed(header='analyze_chains 64 of post_process.py')

    # Wavelengths
    i400 = np.argmin(np.abs(model_wave-400))
    i440 = np.argmin(np.abs(model_wave-440))

    # Stats
    all_adg_400 = a_dg[..., i400]
    all_aph_440 = a_ph[..., i440]
    adg_400 = np.median(all_adg_400)
    aph_440 = np.median(all_aph_440)
    adg_16, adg_84 = np.percentile(all_adg_400.flatten(), [16, 84])
    aph_16, aph_84 = np.percentile(all_aph_440.flatten(), [16, 84])

    # Save
    sdict = dict(adg_400=adg_400, adg_16=adg_16, adg_84=adg_84,
                 aph_440=aph_440, aph_16=aph_16, aph_84=aph_84,
                 l23_adg_400=l23_adg_400, l23_aph_440=l23_aph_440)

    jdict = ocio.jsonify(sdict)
    ocio.savejson(outfile, jdict, overwrite=True)
    print(f'Saved: {outfile}')

    # Return
    return odict, sdict

# Command line execution
if __name__ == '__main__':

    p = param.p_ntuple(['ExpBricaud', 'Pow'], 
            set_Sdg=True, sSdg=0.002, beta=1., nMC=100,
            add_noise=True, wv_min=400.)
    analyze_chains(p, 170, clobber=True)#, debug=True)

    # 350nm
    p350 = param.p_ntuple(['ExpBricaud', 'Pow'], 
            set_Sdg=True, sSdg=0.002, beta=1., nMC=100,
            add_noise=True, wv_min=350.)
    analyze_chains(p350, 170, clobber=True)#, debug=True)
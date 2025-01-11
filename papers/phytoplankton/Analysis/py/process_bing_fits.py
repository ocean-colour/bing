""" Process BING's fits to the data. """
import sys, os
from collections import namedtuple

import numpy as np
import pandas


from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ocpy.hydrolight import loisel23

from bing.models import utils as model_utils
from bing import evaluate

sys.path.append(os.path.abspath("../../bing_2.0/Analysis/py"))
import anly_utils_20
import param as param20

import single_bing_fits

from IPython import embed

# bb water
ds = loisel23.load_ds(4,0)
iwave = np.argmin(np.abs(ds.Lambda.data - 440))
bbw_440=ds.bb.data[0,iwave]-ds.bbnw.data[0,iwave]

def process_one(idx, pdict=None, perc=(16, 84), burn:int=7000, thin:int=1,
                verbose:bool=False):

    MyNamedTuple = namedtuple('BING20_tuple', pdict.keys())
    p = MyNamedTuple(**pdict)

    # Load up
    odict = anly_utils_20.prep_l23_data(
        idx, wv_min=p.wv_min, wv_max=p.wv_max)

    model_wave = anly_utils_20.pace_wave(
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


def main(flg):
    flg = int(flg)

    # Test
    if flg == 1:
        p = single_bing_fits.standard_expb_pow()
        standard, extras = process_one(170, pdict=p._asdict(), verbose=True)
        embed(header='119 of process')

    # Run em all
    if flg == 2:
        p = single_bing_fits.standard_expb_pow()
        process_all(p, 'BING_L23_results_ExpBricaudPow.csv')#, debug=True)

    # Run em all on GIOP
    if flg == 3:
        p = single_bing_fits.standard_giop()
        process_all(p, 'BING_L23_results_GIOPLee.csv')#, debug=True)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg = 1 # -- Testing
        #flg = 2 # -- ExpBricaud Pow
        #flg = 3 # -- GIOP Lee

    else:
        flg = sys.argv[1]

    main(flg)
    

""" Fit the full L23 dataset """
import numpy as np

from oceancolor.hydrolight import loisel23

from big.models import anw as big_anw
from big.models import bbnw as big_bbnw
from big import inference as big_inf
from big import rt as big_rt

import anly_utils 

def fit(model_names:list, 
        Nspec:int=None, abs_sig:float=None,
        prior_approach:str='log',
        nsteps=80000, nburn=8000,
        scl_noise:float=0.02, add_noise:bool=False,
        n_cores:int=20, wstep:int=1, debug:bool=False, 
        max_wv:float=None):
    """
    Fits the data with or without considering any errors.

    Args:
        edict (dict): A dictionary containing the necessary information for fitting.
        Nspec (int): The number of spectra to fit. Default is None = all
        abs_sig (float): The absolute value of the error to consider. Default is None.
            if None, use no error!
        debug (bool): Whether to run in debug mode. Default is False.
        n_cores (int): The number of CPU cores to use for parallel processing. Default is 1.
        max_wv (float): The maximum wavelength to consider. Default is None.
        use_log_ab (bool): Whether to use log(ab) in the priors. Default is False.
        use_NMF_pos (bool): Whether to use positive priors for NMF. Default is False.

    """
    # Load L23
    ds = loisel23.load_ds(4,0)
    wave = ds.Lambda.data[::wstep]

    # Init the models
    anw_model = big_anw.init_model(model_names[0], wave, prior_approach)
    bbnw_model = big_bbnw.init_model(model_names[1], wave, prior_approach)
    models = [anw_model, bbnw_model]
    
    # Initialize the MCMC
    pdict = big_inf.init_mcmc(models, nsteps=nsteps, nburn=nburn)

    # Prep
    if Nspec is None:
        idx = np.arange(ds.Rrs.shape[0])
    else:
        idx = np.arange(Nspec)
    if debug:
        #idx = idx[0:2]
        idx = [170, 180]
        #idx = [2706]

    # Calcualte the Rrs
    Rrs = []
    varRrs = []
    params = []
    for ss in idx:
        odict = anly_utils.prep_l23_data(
            ss, scl_noise=scl_noise, step=wstep, ds=ds)
        # Rrs
        gordon_Rrs = big_rt.calc_Rrs(odict['a'][::wstep], 
                                 odict['bb'][::wstep])
        Rrs.append(gordon_Rrs)
        # varRrs
        ivarRrs = (scl_noise * gordon_Rrs)**2
        varRrs.append(ivarRrs)
        # Params
        p0_a = anw_model.init_guess(odict['anw'][::wstep])
        p0_b = bbnw_model.init_guess(odict['bbnw'][::wstep])
        p0 = np.concatenate((np.log10(np.atleast_1d(p0_a)), 
                         np.log10(np.atleast_1d(p0_b))))
        params.append(p0)
    # Arrays
    Rrs = np.array(Rrs)
    params = np.array(params)
    varRrs = np.array(varRrs)

    # Build the items
    items = [(Rrs[i], varRrs[i], params[i], i) for i in idx]

    # Output file

    all_samples, all_idx = big_inf.fit_batch(
        models, pdict, items, n_cores=n_cores)

    # Save
    outfile = f'BIG_{model_names[0]}{model_names[1]}_L23'
    if add_noise:
        # Add noise to the outfile with padding of 2
        outfile += f'_N{int(100*scl_noise):02d}'
    else:
        outfile += f'_n{int(100*scl_noise):02d}'
    anly_utils.save_fits(all_samples, all_idx, outfile,
                         extras=dict(Rrs=Rrs))


def main(flg):
    flg = int(flg)

    # Testing
    if flg == 1:
        fit(['Exp', 'Pow'], Nspec=50, nsteps=10000, nburn=1000)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Testing
        #flg += 2 ** 1  # 2 -- No priors
        #flg += 2 ** 2  # 4 -- bb_water

    else:
        flg = sys.argv[1]

    main(flg)
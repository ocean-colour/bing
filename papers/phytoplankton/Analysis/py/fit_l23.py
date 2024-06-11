""" Fit the full L23 dataset """
import os
import numpy as np

from oceancolor.hydrolight import loisel23

from big.models import anw as big_anw
from big.models import bbnw as big_bbnw
from big import inference as big_inf
from big import rt as big_rt
from big import chisq_fit

import anly_utils 

from IPython import embed

def fit(model_names:list, 
        Nspec:int=None, 
        prior_approach:str='log',
        nsteps=80000, nburn=8000,
        use_chisq:bool=False,
        max_wave:float=None,
        scl_noise:float=0.02, add_noise:bool=False,
        n_cores:int=20, wstep:int=1, debug:bool=False): 
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
    if max_wave is not None:
        imax = np.argmin(np.abs(ds.Lambda.data - max_wave))
        iwave = np.arange(imax)
    else:
        iwave = np.arange(ds.Lambda.size)

    wave = ds.Lambda.data[iwave][::wstep]

    # Init the models
    if not use_chisq:
        raise ValueError("Add the priors!!")
    anw_model = big_anw.init_model(model_names[0], wave)
    bbnw_model = big_bbnw.init_model(model_names[1], wave)
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
    Chls = []
    for ss in idx:
        odict = anly_utils.prep_l23_data(
            ss, scl_noise=scl_noise, step=wstep, ds=ds,
            max_wave=max_wave)
        # Rrs
        gordon_Rrs = big_rt.calc_Rrs(odict['a'][::wstep], 
                                 odict['bb'][::wstep])
        Rrs.append(gordon_Rrs)
        # varRrs
        ivarRrs = (scl_noise * gordon_Rrs)**2
        varRrs.append(ivarRrs)
        # Params
        if models[0].name in ['ExpBricaud', 'GIOP']:
            models[0].set_aph(odict['Chl'])

        p0_a = anw_model.init_guess(odict['anw'][::wstep])
        p0_b = bbnw_model.init_guess(odict['bbnw'][::wstep])
        p0 = np.concatenate((np.log10(np.atleast_1d(p0_a)), 
                         np.log10(np.atleast_1d(p0_b))))
        params.append(p0)
        # Others
        Chls.append(odict['Chl'])
    # Arrays
    Rrs = np.array(Rrs)
    params = np.array(params)
    varRrs = np.array(varRrs)

    # Build the items
    items = [(Rrs[i], varRrs[i], params[i], i) for i in idx]

    # Output file
    outfile = anly_utils.chain_filename(
        model_names, scl_noise, add_noise)

    # Fit
    if use_chisq:
        all_ans = []
        all_cov = []
        all_idx = []
        # Fit
        for item in items:
            if models[0].name in ['ExpBricaud', 'GIOP']:
                models[0].set_aph(odict['Chl'])
            if models[1].name == 'Lee':
                models[1].set_Y(odict['Y'])
            try:
                ans, cov, idx = chisq_fit.fit(item, models)
            except RuntimeError:
                print("*****************")
                print(f"Failed on {item[3]}")
                print("*****************")
                ans = np.zeros_like(prev_ans)
                cov = np.zeros_like(prev_cov)
            all_ans.append(ans)
            all_cov.append(cov)
            all_idx.append(idx)
            #
            prev_ans = ans
            prev_cov = cov
        # Save
        outfile = outfile.replace('BIG', 'BIG_LM')
        np.savez(outfile, ans=all_ans, cov=all_cov,
              wave=wave, obs_Rrs=Rrs, varRrs=varRrs,
              idx=all_idx, Chl=Chls)
    else:
        embed(header='fit 116; need to deal with Chl')
        all_samples, all_idx = big_inf.fit_batch(
            models, pdict, items, n_cores=n_cores)
        # Save
        anly_utils.save_fits(all_samples, all_idx, outfile,
                         extras=dict(Rrs=Rrs, varRrs=varRrs))
    print(f"Saved: {outfile}")                        

def reconstruct(model_names:list, wstep:int=1,
        prior_approach:str='log',
        scl_noise:float=0.02, add_noise:bool=False):
    # Load L23
    ds = loisel23.load_ds(4,0)
    wave = ds.Lambda.data[::wstep]

    # Init the models
    anw_model = big_anw.init_model(model_names[0], wave, prior_approach)
    bbnw_model = big_bbnw.init_model(model_names[1], wave, prior_approach)
    models = [anw_model, bbnw_model]

    # Load the chains
    chain_file = anly_utils.chain_filename(
        model_names, scl_noise, add_noise)
    d_chains = np.load(chain_file)

    # Loop me
    recon_Rrs = []
    recon_sigRs = []
    recon_a = []
    recon_bb = []
    recon_a_5 = []
    recon_a_95 = []
    recon_bb_5 = []
    recon_bb_95 = []
    # Parallize?
    for ss in range(d_chains['chains'].shape[0]):
        if ss % 100 == 0:
            print(f'Working on {ss}')
        # Reconstruct
        a_mean, bb_mean, a_5, a_95, bb_5, bb_95, Rrs, sigRs =\
            anly_utils.reconstruct(models, d_chains['chains'][ss])
        # Save what we want
        recon_Rrs.append(Rrs)
        recon_sigRs.append(sigRs)
        recon_a.append(a_mean)
        recon_bb.append(bb_mean)
        recon_a_5.append(a_5)
        recon_a_95.append(a_95)
        recon_bb_5.append(bb_5)
        recon_bb_95.append(bb_95)

    # Save
    basename = os.path.basename(chain_file)
    outfile = 'recon_' + basename
    np.savez(outfile, Rrs=recon_Rrs, idx=d_chains['idx'],
             sigRs=recon_sigRs, a=recon_a, bb=recon_bb,
             a_5=recon_a_5, a_95=recon_a_95,
             bb_5=recon_bb_5, bb_95=recon_bb_95)
    print(f'Saved: {outfile}')

def main(flg):
    flg = int(flg)

    # Testing
    if flg == 1:
        fit(['Exp', 'Pow'], Nspec=50, nsteps=10000, nburn=1000)

    # Full L23
    if flg == 2:
        fit(['Exp', 'Pow'], nsteps=50000, nburn=5000)
        reconstruct(['Exp', 'Pow']) 

    # Full L23 with LM; constant relative error
    if flg == 3:
        fit(['Cst', 'Cst'], use_chisq=True, max_wave=700.)
        fit(['Exp', 'Cst'], use_chisq=True, max_wave=700.)
        fit(['Exp', 'Pow'], use_chisq=True, max_wave=700.)
        fit(['ExpBricaud', 'Pow'], use_chisq=True, max_wave=700.)
        fit(['ExpNMF', 'Pow'], use_chisq=True, max_wave=700.)

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
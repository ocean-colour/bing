""" Run BIG one one spectrum """

import os

import numpy as np

from ihop.inference import noise

from big import rt as big_rt
from big.models import anw as big_anw
from big.models import bbnw as big_bbnw
from big import inference as big_inf
from big import rt as big_rt
from big import chisq_fit

from IPython import embed

import anly_utils 

def fit_one(model_names:list, idx:int, n_cores=20, 
            wstep:int=1,
            nsteps:int=10000, nburn:int=1000, 
            scl_noise:float=0.02, use_chisq:bool=False,
            scl:float=None,  # Scaling for the priors
            add_noise:bool=False):

    odict = anly_utils.prep_l23_data(idx, scl_noise=scl_noise, step=wstep)

    # Unpack
    wave = odict['wave']
    Rrs = odict['Rrs']
    varRrs = odict['varRrs']
    a = odict['a']
    bb = odict['bb']
    bbw = odict['bbw']
    aw = odict['aw']

    # Init the models
    anw_model = big_anw.init_model(model_names[0], wave, 'log')
    bbnw_model = big_bbnw.init_model(model_names[1], wave, 'log')
    models = [anw_model, bbnw_model]
    
    # Initialize the MCMC
    pdict = big_inf.init_mcmc(models, nsteps=nsteps, nburn=nburn)
    
    # Gordon Rrs
    gordon_Rrs = big_rt.calc_Rrs(odict['a'][::wstep], 
                                 odict['bb'][::wstep])
    if add_noise:
        gordon_Rrs = noise.add_noise(gordon_Rrs, perc=scl_noise*100)

    # Bricaud?
    if models[0].name == 'ExpBricaud':
        models[0].set_aph(odict['Chl'])

    # Initial guess
    p0_a = anw_model.init_guess(odict['anw'][::wstep])
    p0_b = bbnw_model.init_guess(odict['bbnw'][::wstep])
    p0 = np.concatenate((np.log10(np.atleast_1d(p0_a)), 
                         np.log10(np.atleast_1d(p0_b))))

    # Chk initial guess
    ca = models[0].eval_a(p0[0:models[0].nparam])
    cbb = models[1].eval_bb(p0[models[0].nparam:])
    pRrs = big_rt.calc_Rrs(ca, cbb)
    print(f'Initial Rrs guess: {np.mean((gordon_Rrs-pRrs)/gordon_Rrs)}')
    #embed(header='65 of fit one')
    
    # Set the items
    items = [(gordon_Rrs, varRrs, p0, idx)]

    outfile = anly_utils.chain_filename(model_names, scl_noise, add_noise, idx=idx)
    if not use_chisq:
        # Fit
        chains, idx = big_inf.fit_one(items[0], models=models, pdict=pdict, chains_only=True)

        # Save
        anly_utils.save_fits(chains, idx, outfile,
              extras=dict(wave=wave, obs_Rrs=gordon_Rrs, varRrs=varRrs))
    else:
        # Fit
        ans, cov, idx = chisq_fit.fit(items[0], models)
        # Save
        outfile = outfile.replace('BIG', 'BIG_LM')
        np.savez(outfile, ans=ans, cov=cov,
              wave=wave, obs_Rrs=gordon_Rrs, varRrs=varRrs)
        print(f"Saved: {outfile}")
        #
        return ans, cov


def main(flg):
    flg = int(flg)

    # Testing
    if flg == 1:
        odict = anly_utils.prep_l23_data(170)

    # First one
    if flg == 2:
        fit_one(['Exp', 'Pow'], idx=170, nsteps=10000, nburn=1000) 

    # Fit 170
    if flg == 3:
        idx = 170
        #fit_one(['Cst', 'Cst'], idx=idx, nsteps=80000, nburn=8000) 
        #fit_one(['Exp', 'Cst'], idx=idx, nsteps=80000, nburn=8000) 
        #fit_one(['Exp', 'Pow'], idx=idx, nsteps=10000, nburn=1000) 
        fit_one(['ExpBricaud', 'Pow'], idx=idx, nsteps=10000, nburn=1000) 

    # chisq fits
    if flg == 4:
        #fit_one(['Cst', 'Cst'], idx=170, use_chisq=True)
        #fit_one(['Exp', 'Cst'], idx=170, use_chisq=True)
        #fit_one(['Exp', 'Pow'], idx=170, use_chisq=True)
        fit_one(['ExpBricaud', 'Pow'], idx=170, use_chisq=True)

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
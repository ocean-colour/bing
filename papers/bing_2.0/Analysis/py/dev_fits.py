""" Fits for development of BING 2.0 """

import os

import numpy as np

from matplotlib import pyplot as plt
import corner

from bing import inference as bing_inf
from bing import plotting as bing_plot

#from xqaa.params import XQAAParams
#from xqaa import retrieve

from IPython import embed

import anly_utils_20 
import param
import prep_for_fits


def main(flg):
    flg = int(flg)

    # NMF
    if flg == 1:
        #fit_one(['ExpNMF', 'Pow'], idx=170, use_chisq=True,
        #        show=True)
        #fit_one(['ExpNMF', 'Lee'], idx=170, use_chisq=True,
        #        show=True, add_noise=True, PACE=True,
        #        scl_noise='PACE', show_xqaa=True)

        # Priors
        apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*4
        # Gaussian for Sdg
        #apriors[1] = dict(flavor='gaussian', mean=0.015, sigma=0.001)
        apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

        # Do it
        fit(['ExpNMF', 'Pow'], idx=170, use_chisq=False,
                show=True, add_noise=True, PACE=True,
                scl_noise='PACE', show_xqaa=True,
                apriors=apriors)#, nsteps=50000, nburn=5000)

    # Single fit
    if flg == 2:
        #p = param.p_ntuple(['ExpBricaud', 'Pow'], 
        #    set_Sdg=True, sSdg=0.002, beta=1., 
        #    add_noise=True, wv_min=400.)

        p = param.p_ntuple(['GIOP', 'Lee'], 
            set_Sdg=False, sSdg=0.002, beta=1., 
            add_noise=True, wv_min=400.)
        # Priors
        apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3

        # Uniform for Sdg
        apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

        # Do it
        #fit(p, 170, show=True, apriors=apriors) 
        #fit(p, 2532, show=True, apriors=apriors) 
        fit(p, 2773, show=True, apriors=apriors, nsteps=40000) 
            

    # Bricaud + UV (100 trials)
    #  - 100 trials
    #  - Strict prior on Sdg and beta
    if flg == 3:
        p = param.p_ntuple(['ExpBricaud', 'Pow'], 
            set_Sdg=True, sSdg=0.002, beta=1., nMC=100,
            add_noise=True, wv_min=375.)
            #add_noise=True, wv_min=350.)
            #add_noise=True, wv_min=400.)

        # Do it
        fit(p, 170, show=False, 
            nsteps=20000, nburn=2000,
            debug=False)

    # Bricaud + UV (100 trials)
    #  - 100 trials
    #  - Loos prior on Sdg
    if flg == 4:
        #for wv_min in [375., 400]:
        for wv_min in [350.]:#, 375, 400]:
            p = param.p_ntuple(['ExpBricaud', 'Pow'], 
                set_Sdg=False, beta=1., nMC=100,
                add_noise=True, wv_min=wv_min)
                #add_noise=True, wv_min=350.)
                #add_noise=True, wv_min=400.)

            # Priors
            apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
            # Uniform for Sdg
            apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

            # Do it
            fit(p, 170, show=False, 
                nsteps=20000, nburn=2000,
                apriors=apriors, debug=False)

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

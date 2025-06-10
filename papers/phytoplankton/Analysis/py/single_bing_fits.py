""" Module to run single BING fits for the paper """
import sys, os

from matplotlib import pyplot as plt

from bing import parameters as param20
from bing.parameters import standard 
from bing.fitting import l23
from bing import plotting

# Local
sys.path.append(os.path.abspath("../../bing_2.0/Analysis/py"))
import anly_utils_20
import dev_fits

def show(prep_dict, extras, models, chains):
    odict = prep_dict['odict']
    pdict = prep_dict['pdict']
    plotting.show_fits(
        models, chains, 
        extras['Chl'], extras['Y'],
        Rrs_true=dict(wave=models[0].wave, spec=prep_dict['model_Rrs']),
        anw_true=dict(wave=odict['true_wave'], spec=odict['anw']),
        bbnw_true=dict(wave=odict['true_wave'], spec=odict['bbnw']),
        perc=(16, 84),
        )
    plt.show()


def main(flg):
    flg = int(flg)

    # High Chl + ExpBricaud, Pow
    if flg == 1:

        # Do it
        idx = 2773
        p = standard.expb_pow(scl_noise='PACE', nsteps=40000, add_noise=True)
        # Do it
        chains, models, prep_dict, idx, extras = l23.fit_one(p, idx)
        outfile = l23.chain_filename(p, idx=idx, path='Fits/')
        l23.save_chains(chains, idx, outfile, extras=extras)
        # Show
        show(prep_dict, extras, models, chains)

    # High Chl + GIOP, Lee
    if flg == 2:

        # Do it
        p = standard.giop()
        dev_fits.fit(p, 2773, show=True, seed=54321)

    # High Chl + GSM
    if flg == 3:

        # Priors
        apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2
        bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*1

        p = param20.p_ntuple(['GSM', 'GSM'],
            scl_noise='PACE', nsteps=40000,
            add_noise=True, wv_min=400., wv_max=700.)

        # Do it
        dev_fits.fit(p, 2773, show=True, seed=54321)

    # Low Chl + ExpBricaud, Pow
    if flg == 4:

        idx = 170
        p = standard.expb_pow(scl_noise='PACE', nsteps=40000, add_noise=True)

        # Do it
        chains, models, prep_dict, idx, extras = l23.fit_one(p, idx)
        outfile = l23.chain_filename(p, idx=idx, path='Fits/')
        l23.save_chains(chains, idx, outfile, extras=extras)
        # Show
        show(prep_dict, extras, models, chains)

    # Degenerate solutions
    if flg == 5:
        p = p_every_every(nsteps=40000)
        dev_fits.fit(p, 170, show=True, seed=54321) 
    
    # k=2b
    if flg == 6:
        # Priors
        apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*1
        bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*1

        p = param20.p_ntuple(['Bricaud', 'Cst'], 
            set_Sdg=False, apriors=apriors, bpriors=bpriors,
            scl_noise='PACE', nsteps=40000,
            add_noise=True, wv_min=400., wv_max=700.)

        # Do it
        dev_fits.fit(p, 170, show=True, seed=54321) 

    # k=6
    if flg == 7:

        # Do it
        idx = 2773
        p = standard.expbf_pow(scl_noise='PACE', nsteps=40000, add_noise=True)
        # Do it
        chains, models, prep_dict, idx, extras = l23.fit_one(p, idx)
        outfile = l23.chain_filename(p, idx=idx, path='Fits/')
        l23.save_chains(chains, idx, outfile, extras=extras)
        # Show
        show(prep_dict, extras, models, chains)


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
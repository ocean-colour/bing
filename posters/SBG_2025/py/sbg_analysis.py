import os, sys

from matplotlib import pyplot as plt

from bing.parameters import standard
from bing.fitting import l23
from bing import plotting

# Local
sys.path.append(os.path.abspath("../../papers/phytoplankton/Analysis/py"))
import fit_l23

sys.path.append(os.path.abspath("../../papers/bing_2.0/Analysis/py"))
import anly_utils_20

from IPython import embed


def main(flg):
    flg = int(flg)

    MODIS = False
    PACE = False
    SBG = False
    SeaWiFS = False
    scl_noise = 0.02
    add_noise = False
    reduce_by_in_situ = None

    # Testing
    if flg == 1:
        fit(['Exp', 'Pow'], Nspec=50, nsteps=10000, nburn=1000)

    SBG = True

    if flg == 10:
        add_noise = True

    if flg in [9,10]:
        if MODIS:
            scl_noise = 'MODIS_Aqua'
        elif SeaWiFS:
            scl_noise = 'SeaWiFS'
        elif PACE:
            scl_noise = 'PACE'
        elif SBG:
            scl_noise = 'SBG'

    #embed(header='main 168')
    if flg in [4,5,6,7,8,9,10,11,12]:
        param = dict(use_chisq=True, PACE=PACE, SeaWiFS=SeaWiFS, 
                     MODIS=MODIS, SBG=SBG,
                     scl_noise=scl_noise, add_noise=add_noise,
                     reduce_by_in_situ=reduce_by_in_situ,
                     outroot='Fits/'
                     )
        fit_l23.fit(['Cst', 'Cst'], **param)
        fit_l23.fit(['Exp', 'Cst'], **param)
        fit_l23.fit(['Exp', 'Pow'], **param)
        fit_l23.fit(['ExpBricaud', 'Pow'], **param)
        fit_l23.fit(['ExpBricaudFix', 'Pow'], **param)
        #fit(['ExpNMF', 'Pow'], use_chisq=True, PACE=PACE, SeaWiFS=SeaWiFS, MODIS=MODIS, scl_noise=scl_noise, add_noise=add_noise)
        fit_l23.fit(['GIOP', 'Pow'], **param)
        fit_l23.fit(['GIOP', 'Lee'], **param)
        fit_l23.fit(['GSM', 'GSM'], **param)
        fit_l23.fit(['GSM', 'Pow'], **param)

    if flg == 20:
        # Show
        show = False
        idx = 2773

        # Do it
        p_expb = standard.expb_pow(satellite='SBG', add_noise=True)
        p_giop = standard.giop(satellite='SBG', add_noise=True)
        p_gsm = standard.gsm(satellite='SBG', add_noise=True)

        for p in [p_expb, p_giop, p_gsm]:
            outfile = anly_utils_20.chain_filename(p, idx=idx, path='Fits/')
            chains, models, prep_dict, idx, extras = l23.fit_one(p, idx)
            
            if show:
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

            # Save
            anly_utils_20.save_chains(chains, idx, outfile,
                                    extras=extras)

    # Run em all on GIOP
    if flg == 30:
        p_giop = standard.giop(satellite='SBG', add_noise=True)
        # Fit
        #l23.batch_fit(p_giop, seed=54321, out_dir='Fits/')
                      
        # Process
        l23.process_all(p_giop, 'BING_L23_SBG_results_GIOPLee.csv')#, debug=True)


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


# python py/sbg_analysis.py 9
# python py/sbg_analysis.py 10
import os, sys


# Local
sys.path.append(os.path.abspath("../papers/phytoplankton/Analysis/py"))
import fit_l23



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

    # Full L23
    if flg == 2:
        fit(['Exp', 'Pow'], nsteps=50000, nburn=5000)

    # Full L23 with LM; constant relative error
    if flg == 3:
        fit(['Cst', 'Cst'], use_chisq=True, max_wave=700., min_wave=400.)
        fit(['Exp', 'Cst'], use_chisq=True, max_wave=700., min_wave=400.)
        fit(['Exp', 'Pow'], use_chisq=True, max_wave=700., min_wave=400.)
        fit(['ExpBricaud', 'Pow'], use_chisq=True, max_wave=700., min_wave=400.)
        #fit(['ExpNMF', 'Pow'], use_chisq=True, max_wave=700., min_wave=400.)
        fit(['GIOP', 'Lee'], use_chisq=True, max_wave=700., min_wave=400.)
        fit(['GSM', 'GSM'], use_chisq=True, max_wave=700., min_wave=400.)


    SBG = True

    if flg in [10,11,12]:
        add_noise = True

    if flg in [7,8,9,10,11,12]:
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
                     reduce_by_in_situ=reduce_by_in_situ)
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
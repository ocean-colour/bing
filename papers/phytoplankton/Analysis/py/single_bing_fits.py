""" Module to run single BING fits for the paper """
import sys, os


from bing import parameters as param20
from bing.priors import standard 

# Local
sys.path.append(os.path.abspath("../../bing_2.0/Analysis/py"))
import anly_utils_20
import dev_fits

def main(flg):
    flg = int(flg)

    # High Chl + ExpBricaud, Pow
    if flg == 1:

        # Do it
        p = standard.expb_pow()
        dev_fits.fit(p, 2773, show=True, seed=54321) 

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
        # Priors
        apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
        bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2

        # Uniform for Sdg from 0.01 - 0.02
        apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

        # Uniform for beta from 0. - 2. (positive here means negative slope)
        bpriors[1]=dict(flavor='uniform', pmin=0., pmax=2.)

        p = param20.p_ntuple(['ExpBricaud', 'Pow'], 
            set_Sdg=False, sSdg=0.002, apriors=apriors, bpriors=bpriors,
            scl_noise='PACE', nsamps=40000,
            add_noise=True, wv_min=400., wv_max=700.)

        # Do it
        dev_fits.fit(p, 170, show=True, seed=54321) 

    # Degenerate solutions
    if flg == 5:
        p = p_every_every(nsteps=40000)
        dev_fits.fit(p, 170, show=True, seed=54321) 
    

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
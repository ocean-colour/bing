""" Module to run single BING fits for the paper """
import sys, os


# Local
sys.path.append(os.path.abspath("../../bing_2.0/Analysis/py"))
import anly_utils_20
import param as param20
import dev_fits

def standard_expb_pow():
    # Priors
    apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2

    # Uniform for Sdg from 0.01 - 0.02
    apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

    # Uniform for beta from 0. - 2. (positive here means negative slope)
    bpriors[1]=dict(flavor='uniform', pmin=0., pmax=2.)

    p = param20.p_ntuple(['ExpBricaud', 'Pow'], 
        set_Sdg=False, sSdg=0.002, apriors=apriors, bpriors=bpriors,
        scl_noise='PACE', nsteps=40000,
        add_noise=True, wv_min=400., wv_max=700.)

    return p

def standard_giop():
    # Priors
    apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*1

    p = param20.p_ntuple(['GIOP', 'Lee'], 
        set_Sdg=False, sSdg=0.002, apriors=apriors, bpriors=bpriors,
        scl_noise='PACE', nsteps=40000,
        add_noise=True, wv_min=400., wv_max=700.)

    return p

def p_every_every(nsteps:int=500000):

    p = param20.p_ntuple(['Every', 'Every'],
        set_Sdg=False, sSdg=0.002, apriors=None, bpriors=None,
        scl_noise=0.02, nsteps=nsteps,
        add_noise=False, wv_min=400., wv_max=700.)

    return p

def p_k2b(nsteps:int=500000):
    p = param20.p_ntuple(['Bricaud', 'Cst'],
        set_Sdg=False, apriors=None, bpriors=None,
        scl_noise=0.02, nsteps=nsteps,
        add_noise=False, wv_min=400., wv_max=700.)
    return p


def main(flg):
    flg = int(flg)

    # High Chl + ExpBricaud, Pow
    if flg == 1:

        # Do it
        p = standard_expb_pow()
        dev_fits.fit(p, 2773, show=True, seed=54321) 

    # High Chl + GIOP, Lee
    if flg == 2:

        # Do it
        p = standard_giop()
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
            scl_noise='PACE', nsteps=40000,
            add_noise=True, wv_min=400., wv_max=700.)

        # Do it
        dev_fits.fit(p, 170, show=True, seed=54321) 

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
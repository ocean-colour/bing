
import os

import numpy as np

from oceancolor.hydrolight import loisel23

from big import rt as big_rt

from IPython import embed


def prep_l23_data(idx:int, step:int=1, scl_noise:float=0.02,
                  ds=None):
    """ Prepare L23 the data for the fit """

    # Load
    if ds is None:
        ds = loisel23.load_ds(4,0)

    # Grab
    Rrs = ds.Rrs.data[idx,:]
    true_Rrs = Rrs.copy()
    wave = ds.Lambda.data
    true_wave = ds.Lambda.data.copy()
    a = ds.a.data[idx,:]
    bb = ds.bb.data[idx,:]
    adg = ds.ag.data[idx,:] + ds.ad.data[idx,:]
    aph = ds.aph.data[idx,:]

    # For bp
    rrs = Rrs / (big_rt.A_Rrs + big_rt.B_Rrs*Rrs)
    i440 = np.argmin(np.abs(true_wave-440))
    i555 = np.argmin(np.abs(true_wave-555))
    Y = 2.2 * (1 - 1.2 * np.exp(-0.9 * rrs[i440]/rrs[i555]))

    # For aph
    aph = ds.aph.data[idx,:]
    Chl = aph[i440] / 0.05582

    # Cut down to 40 bands
    Rrs = Rrs[::step]
    wave = wave[::step]

    # Error
    varRrs = (scl_noise * Rrs)**2

    # Dict me
    odict = dict(wave=wave, Rrs=Rrs, varRrs=varRrs, a=a, bb=bb, 
                 true_wave=true_wave, true_Rrs=true_Rrs,
                 bbw=ds.bb.data[idx,:]-ds.bbnw.data[idx,:],
                 bbnw=ds.bbnw.data[idx,:],
                 aw=ds.a.data[idx,:]-ds.anw.data[idx,:],
                 anw=ds.anw.data[idx,:],
                 adg=adg, aph=aph,
                 Y=Y, Chl=Chl)

    return odict

def reconstruct(models:list, chains, burn=7000, thin=1):
    # Burn the chains
    chains = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])
    # Calc
    a = models[0].eval_a(chains[..., :models[0].nparam])
    bb = models[1].eval_bb(chains[..., models[0].nparam:])
    del chains

    # Calculate the mean and standard deviation
    a_mean = np.median(a, axis=0)
    a_5, a_95 = np.percentile(a, [5, 95], axis=0)
    #a_std = np.std(a, axis=0)
    bb_mean = np.median(bb, axis=0)
    bb_5, bb_95 = np.percentile(bb, [5, 95], axis=0)
    #bb_std = np.std(bb, axis=0)

    # Calculate the model Rrs
    Rrs = big_rt.calc_Rrs(a, bb)

    # Stats
    sigRs = np.std(Rrs, axis=0)
    Rrs = np.median(Rrs, axis=0)

    # Return
    return a_mean, bb_mean, a_5, a_95, bb_5, bb_95, Rrs, sigRs 


def save_fits(all_samples, all_idx, outroot, extras:dict=None):
    """
    Save the fitting results to a file.

    Parameters:
        all_samples (numpy.ndarray): Array of fitting chains.
        all_idx (numpy.ndarray): Array of indices.
        Rs (numpy.ndarray): Array of Rs values.
        use_Rs (numpy.ndarray): Array of observed Rs values.
        outroot (str): Root name for the output file.
    """  
    # Save
    outdir = 'Fits/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, outroot)
    # Outdict
    outdict = dict()
    outdict['chains'] = all_samples
    outdict['idx'] = all_idx
    
    # Extras
    if extras is not None:
        for key in extras.keys():
            outdict[key] = extras[key]
    np.savez(outfile, **outdict)
    print(f"Saved: {outfile}")
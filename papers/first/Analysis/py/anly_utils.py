
import os

import numpy as np

from oceancolor.hydrolight import loisel23

from big import rt as big_rt

from IPython import embed

def chain_filename(model_names:list, scl_noise, add_noise,
                       idx:int=None, MODIS:bool=False): 
    outfile = f'../Analysis/Fits/BIG_{model_names[0]}{model_names[1]}'

    if idx is not None:
        outfile += f'_{idx}'
    else:
        if MODIS:
            outfile += '_M23'
        else:
            outfile += '_L23'
    if add_noise:
        outfile += f'_N{int(100*scl_noise):02d}'
    else:
        outfile += f'_n{int(100*scl_noise):02d}'
    outfile += '.npz'
    return outfile

def prep_l23_data(idx:int, step:int=1, scl_noise:float=0.02,
                  ds=None, max_wave:float=None):
    """ Prepare L23 the data for the fit """

    # Load
    if ds is None:
        ds = loisel23.load_ds(4,0)

    wave = ds.Lambda.data

    if max_wave is not None:
        imax = np.argmin(np.abs(ds.Lambda.data - max_wave))
        iwave = np.arange(imax)
    else:
        iwave = np.arange(ds.Lambda.size)


    # Grab
    Rrs = ds.Rrs.data[idx,iwave]
    wave = wave[iwave]
    true_Rrs = Rrs.copy()
    true_wave = wave.copy()
    a = ds.a.data[idx,iwave]
    bb = ds.bb.data[idx,iwave]
    adg = ds.ag.data[idx,iwave] + ds.ad.data[idx,iwave]
    aph = ds.aph.data[idx,iwave]

    # For bp
    rrs = Rrs / (big_rt.A_Rrs + big_rt.B_Rrs*Rrs)
    i440 = np.argmin(np.abs(true_wave-440))
    i555 = np.argmin(np.abs(true_wave-555))
    Y = 2.2 * (1 - 1.2 * np.exp(-0.9 * rrs[i440]/rrs[i555]))

    # For aph
    aph = ds.aph.data[idx,iwave]
    Chl = aph[i440] / 0.05582

    # Cut down to 40 bands
    Rrs = Rrs[::step]
    wave = wave[::step]

    # Error
    varRrs = (scl_noise * Rrs)**2

    # Dict me
    odict = dict(wave=wave, Rrs=Rrs, varRrs=varRrs, a=a, bb=bb, 
                 true_wave=true_wave, true_Rrs=true_Rrs,
                 bbw=ds.bb.data[idx,iwave]-ds.bbnw.data[idx,iwave],
                 bbnw=ds.bbnw.data[idx,iwave],
                 aw=ds.a.data[idx,iwave]-ds.anw.data[idx,iwave],
                 anw=ds.anw.data[idx,iwave],
                 adg=adg, aph=aph,
                 Y=Y, Chl=Chl)

    return odict

def save_fits(all_samples, all_idx, outfile, 
              extras:dict=None):
    """
    Save the fitting results to a file.

    Parameters:
        all_samples (numpy.ndarray): Array of fitting chains.
        all_idx (numpy.ndarray): Array of indices.
        Rs (numpy.ndarray): Array of Rs values.
        use_Rs (numpy.ndarray): Array of observed Rs values.
        outroot (str): Root name for the output file.
    """  
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

import os

import numpy as np

from oceancolor.hydrolight import loisel23

from boring import rt as boring_rt
from boring.models import anw as boring_anw
from boring.models import bbnw as boring_bbnw
from boring.models import utils as model_utils
from boring import stats as boring_stats
from boring import chisq_fit
from boring.satellites import pace as boring_pace
from boring.satellites import modis as boring_modis

from IPython import embed

def chain_filename(model_names:list, scl_noise, add_noise,
                       idx:int=None, MODIS:bool=False, PACE:bool=False): 
    outfile = f'../Analysis/Fits/BORING_{model_names[0]}{model_names[1]}'

    if idx is not None:
        outfile += f'_{idx}'
    else:
        if MODIS:
            outfile += '_M23'
        elif PACE:
            outfile += '_P23'
        else:
            outfile += '_L23'
    if add_noise:
        outfile += f'_N{int(100*scl_noise):02d}'
    else:
        outfile += f'_n{int(100*scl_noise):02d}'
    outfile += '.npz'
    return outfile

def get_chain_file(model_names, scl_noise, add_noise, idx,
                   use_LM=False, full_LM=True, MODIS:bool=False,
                   PACE:bool=False):
    scl_noise = 0.02 if scl_noise is None else scl_noise
    noises = f'{int(100*scl_noise):02d}'
    noise_lbl = 'N' if add_noise else 'n'

    if full_LM:
        if MODIS:
            cidx = 'M23'
        elif PACE:
            cidx = 'P23'
        else:
            cidx = 'L23'
    else:
        cidx = str(idx)

    chain_file = f'../Analysis/Fits/BORING_{model_names[0]}{model_names[1]}_{cidx}_{noise_lbl}{noises}.npz'
    # LM
    if use_LM:
        chain_file = chain_file.replace('BORING', 'BORING_LM')
    return chain_file, noises, noise_lbl

def calc_ICs(ks:list, s2ns:list, use_LM:bool=False,
             MODIS:bool=False, PACE:bool=False):

    Bdict = dict()
    Adict = dict()
    for k in ks:
        Adict[k] = []
        Bdict[k] = []

        # Model names
        if k == 2:
            model_names = ['Cst', 'Cst']
        elif k == 3:
            model_names = ['Exp', 'Cst']
        elif k == 4:
            model_names = ['Exp', 'Pow']
        elif k == 5:
            model_names = ['ExpBricaud', 'Pow']
        elif k == 6:
            model_names = ['ExpNMF', 'Pow']
        elif k == 9: # GIOP
            model_names = ['GIOP', 'Lee']
        else:
            raise ValueError("Bad k")

        chain_file, noises, noise_lbl = get_chain_file(
            model_names, 0.02, False, 'L23', use_LM=use_LM,
            MODIS=MODIS, PACE=PACE)
        d_chains = np.load(chain_file)
        print(f'Loaded: {chain_file}')
        wave = d_chains['wave']

        # Init the models
        models = model_utils.init(model_names, wave)

        # Loop on S/N
        if k == 3:
            sv_s2n = []
            sv_idx = []
        for s2n in s2ns:
            if PACE and (s2n == 'PACE'):
                noise_vector = boring_pace.gen_noise_vector(anw_model.wave)
            else:
                noise_vector = None
            # Calculate BIC
            AICs, BICs = boring_stats.calc_ICs(
                d_chains['obs_Rrs'], models, d_chains['ans'],
                            s2n, use_LM=use_LM, debug=False,
                            Chl=d_chains['Chl'],
                            bb_basis_params=d_chains['Y'], # Lee
                            noise_vector=noise_vector)
            Adict[k].append(AICs)
            Bdict[k].append(BICs)
            # 
            if k == 3:
                sv_s2n += [s2n]*BICs.size
                sv_idx += d_chains['idx'].tolist()
        #embed(header='678 of fig_all_bic')
        # Concatenate
        Bdict[k] = np.array(Bdict[k])
        Adict[k] = np.array(Adict[k])

    # Return
    return Adict, Bdict
        

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
    rrs = Rrs / (boring_rt.A_Rrs + boring_rt.B_Rrs*Rrs)
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

# #############################################################################
def recon_one(model_names:list, idx:int, max_wave:float=None,
              scl_noise:float=None, add_noise:bool=False, use_LM:bool=False,
              full_LM:bool=False, MODIS=False, PACE=False):

    # Load up the chains or parameters
    chain_file, noises, noise_lbl = get_chain_file(
        model_names, scl_noise, add_noise, idx, use_LM=use_LM,
        full_LM=full_LM, MODIS=MODIS)
    print(f'Loading: {chain_file}')
    d_chains = np.load(chain_file)

    # Load the data
    odict = prep_l23_data(idx, max_wave=max_wave)
    model_wave = odict['wave']
    Rrs = odict['Rrs']
    varRrs = odict['varRrs']
    a_true = odict['a']
    bb_true = odict['bb']
    aw = odict['aw']
    adg = odict['adg']
    aph = odict['aph']
    bbw = odict['bbw']
    bbnw = bb_true - bbw
    wave_true = odict['true_wave']
    Rrs_true = odict['true_Rrs']

    gordon_Rrs = boring_rt.calc_Rrs(odict['a'], odict['bb'])

    # MODIS?
    if MODIS:
        model_wave = boring_modis.modis_wave
        model_Rrs = boring_modis.convert_to_modis(wave_true, gordon_Rrs)

    # Init the models

    # Extras?
    if models[0].uses_Chl:
        models[0].set_aph(odict['Chl'])
    if models[1].uses_basis_params:  # Lee
        models[1].set_basis_func(odict['Y'])

    # Interpolate
    aw_interp = np.interp(model_wave, wave_true, aw)

    #embed(header='figs 167')

    # Reconstruct
    if use_LM:
        if full_LM:
            params = d_chains['ans'][idx]
        else:
            params = d_chains['ans']
        model_Rrs, a_mean, bb_mean = chisq_fit.fit_func(
            model_wave, *params, models=models, return_full=True)
    else:
        raise ValueError("Need to implement")
        #a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
        #    model_Rrs, sigRs = anly_utils.reconstruct(
        #    models, d_chains['chains']) 

    # Return as a dict
    rdict = dict(wave=model_wave, Rrs=Rrs, varRrs=varRrs, noise_lbl=noise_lbl,
                 noises=noises, idx=idx,
                 a_true=a_true, bb_true=bb_true,
                 aw=aw, adg=adg, aph=aph,
                 anw_model=anw_model, bbnw_model=bbnw_model,
                 aw_interp=aw_interp, 
                 bbw=bbw, bbnw=bbnw,
                 wave_true=wave_true, Rrs_true=Rrs_true,
                 gordon_Rrs=gordon_Rrs,
                 model_Rrs=model_Rrs, a_mean=a_mean, bb_mean=bb_mean)
    # Return
    return rdict
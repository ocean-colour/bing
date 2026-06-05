# Assess PACE sensitivity to bbp by fitting synthetic L23 spectra
# starting from the Hydrolight model with the smallest particulate
# backscattering.

import os, sys
import numpy as np

from matplotlib import pyplot as plt

from ocpy.utils import plotting
from ocpy.hydrolight import loisel23
from ocpy.satellites import pace as sat_pace

from correct_atmosphere import downwelling

from bing.fitting import l23 as bing_l23
from bing.preproc import convert_to_satwave
from bing.noise import scale_noise, add_noise
from bing import io as bing_io
from bing.models import utils as model_utils
from bing import rt as bing_rt
from bing.rt import chl_fl
from bing.parameters import standard

# Locals
import fitting

from IPython import embed

WV_REF = 700.
LOWBBP_DIR = 'Low_bbp'

def load_l23(rank:int=1, use_elastic:bool=False, wv_ref:float=WV_REF, 
             wv_min:float=400., wv_max:float=700., idx:int=None):
    # Loisel 2023 dataset (cached after the first call)
    if use_elastic:
        ds = loisel23.load_ds(1, 0)
    else:   
        ds = loisel23.load_ds(4, 0)

    # Pick the spectrum with the smallest bbp at wv_ref
    if idx is None:
        idx, bbp_value = find_lowest_bbp_idx(rank=rank, ds=ds, wv_ref=wv_ref)
    else:
        bbp_value = 0.
    print(f"Lowest bbp spectrum: idx={idx}, "
          f"bbp({wv_ref:.0f})={bbp_value:.3e} m^-1")

    # Pull the L23 spectrum (uses the same loader as bing.fitting.l23)
    odict = bing_l23.load_one_l23(idx, ds=ds, wv_min=wv_min, wv_max=wv_max)

    # Return 
    return ds, idx, bbp_value, odict

def find_lowest_bbp_idx(rank:int=1,ds=None, wv_ref:float=WV_REF):
    """Locate the L23 spectrum with the smallest bbp at a reference wavelength.

    In the Loisel et al. (2023) Hydrolight dataset the particulate
    backscattering is stored as ``bbnw`` (non-water backscattering).  We
    use that as the proxy for ``bbp`` since the two are equivalent in
    these clear-water simulations.

    Parameters
    ----------
    rank : int
        Rank of the spectrum to select.  Default is 1 (smallest).
    ds : xarray.Dataset, optional
        Pre-loaded L23 dataset.  Loaded if not provided.
    wv_ref : float, optional
        Reference wavelength (nm) at which to evaluate bbp.

    Returns
    -------
    idx : int
        Row index in the L23 dataset with the smallest bbp(wv_ref).
    bbp_value : float
        bbp value at ``wv_ref`` for that spectrum.
    """
    if ds is None:
        ds = loisel23.load_ds(4, 0)

    # Locate the closest wavelength index
    wave = ds.Lambda.data
    iwv = int(np.argmin(np.abs(wave - wv_ref)))

    # Sort the bbnw values at the reference wavelength
    bbnw_ref = ds.bbnw.data[:, iwv]
    sorted_idx = np.argsort(bbnw_ref)
    idx = sorted_idx[rank-1]

    # bbnw across all spectra at the reference wavelength
    return idx, float(bbnw_ref[idx])


def generate_pace_spectrum(rank:int=1, wv_ref:float=WV_REF, 
                           wv_min:float=400.,
                           wv_max:float=700., scl_noise:str='PACE',
                           add_satellite_noise:bool=True,
                           use_elastic:bool=False, idx:int=None,
                           seed:int=1234, use_Gordon:bool=False):
    """Build a synthetic PACE Rrs spectrum from the lowest-bbp L23 model.

    Loads the chosen Hydrolight spectrum, interpolates the true
    Rrs onto the PACE wavelength grid, applies the PACE
    noise vector, and optionally adds a random noise realisation.

    Parameters
    ----------
    rank : int
        Rank of the spectrum to select.  Default is 1 (smallest).
    wv_ref : float
        Wavelength used to select the lowest-bbp spectrum.
    wv_min, wv_max : float
        Wavelength bounds (nm) for the PACE grid.
    scl_noise : str or float
        Noise specification accepted by :func:`bing.noise.scale_noise`.
    add_satellite_noise : bool
        If True, draw a random realisation of the satellite noise.
    seed : int or None
        Seed for reproducibility when ``add_satellite_noise`` is True.
    use_Gordon : bool
        If True, use the Gordon Rrs model instead of the true Rrs.
        This is the variable Gordon model.
    use_elastic : bool
        If True, use the elastic Rrs model instead of the inelastic Rrs model.

    Returns
    -------
    out : dict
        ``pace_wave``, ``Rrs`` (noisy if requested), ``Rrs_true`` (noise
        free), ``sigRrs``, ``bbp_value``, ``bbp_wave`` (wavelength used
        for the bbp reference), ``idx``, and ``odict`` (raw L23 data).
    """
    _, idx, bbp_value, odict = load_l23(rank=rank, use_elastic=use_elastic,
                                         wv_ref=wv_ref, wv_min=wv_min, wv_max=wv_max, idx=idx)
    '''
    # Loisel 2023 dataset (cached after the first call)
    if use_elastic:
        ds = loisel23.load_ds(1, 0)
    else:   
        ds = loisel23.load_ds(4, 0)

    # Pick the spectrum with the smallest bbp at wv_ref
    idx, bbp_value = find_lowest_bbp_idx(rank=rank, ds=ds, wv_ref=wv_ref)
    print(f"Lowest bbp spectrum: idx={idx}, "
          f"bbp({wv_ref:.0f})={bbp_value:.3e} m^-1")

    # Pull the L23 spectrum (uses the same loader as bing.fitting.l23)
    odict = bing_l23.load_one_l23(idx, ds=ds, wv_min=wv_min, wv_max=wv_max)
    '''

    # Grab aph at 440nm
    aph_440 = odict['aph'][np.argmin(np.abs(odict['true_wave']-440))]

    # PACE wavelength grid
    pace_wave = sat_pace.wave(wv_min=wv_min, wv_max=wv_max)

    # Parse Rrs
    if use_Gordon:  # This is variable Gordon
        models = model_utils.init(['ExpBricaud', 'Pow'], 
                                  odict['true_wave'])
        models[0].init_var_gordon()
        a_ex = odict['f_a'](models[0].wave_ex)
        bb_ex = odict['f_bb'](models[1].wave_ex)
        Rrs_true = bing_rt.calc_Rrs(odict['a'], odict['bb'],
            in_G1=models[0].G1, in_G2=models[0].G2, 
            a_ex = a_ex, bb_ex=bb_ex,
            bb_R=models[1].bb_R)
    else:
        Rrs_true = odict['true_Rrs']

    # Interpolate the noise-free Rrs onto PACE bands
    Rrs_true = convert_to_satwave(odict['true_wave'],
                                  Rrs_true, pace_wave)

    # Per-band Rrs variance from the PACE noise model
    varRrs = scale_noise(scl_noise, Rrs_true, pace_wave)
    sigRrs = np.sqrt(varRrs)

    # Add a random noise realisation if requested
    if add_satellite_noise:
        if seed is not None:
            np.random.seed(seed)
        Rrs = add_noise(Rrs_true, abs_sig=sigRrs)
    else:
        Rrs = Rrs_true.copy()

    return dict(
        pace_wave=pace_wave, Rrs=Rrs, Rrs_true=Rrs_true,
                sigRrs=sigRrs, bbp_value=bbp_value, 
                bbp_wave=wv_ref,
                aph_440=aph_440,
                idx=idx, odict=odict)


def plot_spectrum(spec:dict, outfile:str='Low_bbp/lowest_bbp_spectrum.png',
                  show:bool=True, log10:bool=False):
    """Plot the synthetic PACE Rrs spectrum with error bars.

    Parameters
    ----------
    spec : dict
        Output of :func:`generate_pace_spectrum`.
    outfile : str
        Path for the saved PNG.
    show : bool
        If True, call ``plt.show()`` after saving.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # If log10, suppress negative values
    if log10:
        good = spec['Rrs'] > 0
    else:
        good = np.ones(len(spec['Rrs']), dtype=bool)

    # Noisy realisation with PACE error bars
    ax.errorbar(spec['pace_wave'][good], spec['Rrs'][good], 
                yerr=spec['sigRrs'][good],
                color='k', fmt='o', capsize=3, markersize=4,
                label='PACE Rrs (with noise)')

    # Reference noise-free curve for context
    ax.plot(spec['pace_wave'], spec['Rrs_true'], color='steelblue',
            ls='-', lw=1.5, label='True Rrs')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$R_{rs}$ (sr$^{-1}$)')

    # Annotate bbp value and L23 index
    txt = (f"L23 idx = {spec['idx']}\n"
           r"$b_{bp}(" + f"{spec['bbp_wave']:.0f}" + r")$ = "
           f"{spec['bbp_value']:.3e} m$^{{-1}}$\n"
           r"$a_{ph}$ = "
           f"{spec['aph_440']:.3e} m$^{-1}$")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=13,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.legend(loc='lower left', fontsize=12)
    plotting.set_fontsize(ax, 14)

    # Zero line
    ax.axhline(0., color='g', ls='--', lw=1.)

    # Log scale y-axis
    if log10:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
    if show:
        plt.show()
    plt.close()

def examine_fit(rank:int, fit_file:str, outfile:str, use_elastic:bool=False): 

    # Load L23
    _, idx, bbp_value, odict = load_l23(rank=rank, use_elastic=use_elastic)

    # Load the fit
    result = bing_io.load_fit(fit_file)

    # Prep
    #embed(header='examine_fit 262')
    Rrs_obs = dict(wave=result['wave'], spec=result['Rrs'], 
                   var=result['varRrs'])

    # Add Rrs with Gordon
    Rrs_OrigG = bing_rt.calc_Rrs(odict['a'], odict['bb'])

    # Plot it
    fitting.plot_fit(result['models'], result['chains'], 
                     Rrs_obs, "examine", result['rt_dict'],
                     stats=result['stats'], show_Rsig=True, 
                     add_Rrs=Rrs_OrigG,
                     true_bbp=odict['bbnw'],
                     true_anw=odict['anw'],
                     outfile=outfile)
    
def compare_Rrs(rank:int=1, outroot:str='Low_bbp/compare_Rrs',
                  wv_min=400, show:bool=True, log10:bool=False,
                  use_var:bool=False):
    """Compare various Rrs spectra for a given, low bbp example.

    Parameters
    ----------
    rank : int
        Rank of the spectrum to select.  Default is 1 (smallest).
    outroot : str
        Root for the saved PNGs.
    show : bool
        If True, call ``plt.show()`` after saving.
    """
    ds_inelastic = loisel23.load_ds(4, 0)
    ds_elastic = loisel23.load_ds(1, 0)

    idx, bbp_value = find_lowest_bbp_idx(rank=rank, ds=ds_inelastic)

    # Grab the spectra
    odict_inelastic = bing_l23.load_one_l23(idx, ds=ds_inelastic, wv_min=wv_min)
    odict_elastic = bing_l23.load_one_l23(idx, ds=ds_elastic, wv_min=wv_min)

    # Gordon me
    models = model_utils.init(['ExpBricaud', 'Pow'], 
                                odict_elastic['true_wave'])
    models[0].init_var_gordon()

    a_ex = odict_elastic['f_a'](models[0].wave_ex)
    bb_ex = odict_elastic['f_bb'](models[1].wave_ex)

    # Gordon models
    Rrs_GordonS = bing_rt.calc_Rrs(odict_elastic['a'], odict_elastic['bb'])
    Rrs_GordonV = bing_rt.calc_Rrs(odict_elastic['a'], odict_elastic['bb'],
        in_G1=models[0].G1, in_G2=models[0].G2) 
    Rrs_GordonR = bing_rt.calc_Rrs(odict_elastic['a'], odict_elastic['bb'],
        a_ex = a_ex, bb_ex=bb_ex,
        bb_R=models[1].bb_R)
    Rrs_GordonRV = bing_rt.calc_Rrs(odict_elastic['a'], odict_elastic['bb'],
        in_G1=models[0].G1, in_G2=models[0].G2,
        a_ex = a_ex, bb_ex=bb_ex,
        bb_R=models[1].bb_R)
    Rrs_GordonRV = bing_rt.calc_Rrs(odict_elastic['a'], odict_elastic['bb'],
        in_G1=models[0].G1, in_G2=models[0].G2,
        a_ex = a_ex, bb_ex=bb_ex,
        bb_R=models[1].bb_R)

    # G0
    models = model_utils.init(['ExpBricaud', 'Pow'], 
                                odict_elastic['true_wave'])
    models[0].init_var_gordon(include_G0=True)
    Rrs_GordonV0 = bing_rt.calc_Rrs(odict_elastic['a'], odict_elastic['bb'],
        in_G1=models[0].G1, in_G2=models[0].G2, in_G0=models[0].G0)

    # Add Chl fluorescence
    Ed = downwelling.downwelling_irradiance(models[0].wave, 0.)
    Ed_em = downwelling.downwelling_irradiance(chl_fl.LAMBDA_FL_PRIMARY, 0.)
    models[0].init_Chl_fluorescence(Ed=Ed, Ed_em=Ed_em)
    Rrs_fl = bing_rt.rrs.calc_Rrs_fluorescence(
        odict_elastic['true_wave'], odict_elastic['a'], odict_elastic['bb'],
        odict_elastic['a'][models[0].i_Chl_ex],
        odict_elastic['bb'][models[0].i_Chl_ex],
        odict_elastic['aph'][models[0].i_Chl_ex],
        models[0].wave[models[0].i_Chl_ex],
        models[0].Ed_ex,
        models[0].Ed_em,
        phi_C=0.02,
        double_gaussian=False)
    Rrs_GordonRVCF = Rrs_GordonRV + Rrs_fl

    # Correct L23 elastic Rrs to account for Raman scattering
    corr = bing_rt.rrs.calc_raman_correction_factor(
        odict_elastic['a'], odict_elastic['bb'], a_ex, bb_ex, models[1].bb_R)
    Rrs_L23R = odict_elastic['true_Rrs'] * corr
    Rrs_L23RCF = Rrs_L23R + Rrs_fl


    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for lbl, spec in zip(
        ['Inelastic', 'BING_S', 'BING_V', 'BING_V0',
         'BING_R', 'BING_RV', 'BING_RVCF', 
         'L23R', 'L23RCF'], 
        [odict_inelastic['true_Rrs'], Rrs_GordonS, Rrs_GordonV, Rrs_GordonV0,
         Rrs_GordonR, Rrs_GordonRV, Rrs_GordonRVCF, Rrs_L23R, Rrs_L23RCF]):

        # If log10, suppress negative values
        #if log10:
        #    good = spec['Rrs'] > 0
        #else:
        #    good = np.ones(len(spec['Rrs']), dtype=bool)

        # Reference noise-free curve for context
        ax.plot(odict_elastic['true_wave'], spec/odict_elastic['true_Rrs'],
                ls='-', lw=1.5, label=lbl)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$R_{rs}$ relative to Elastic')

    # Annotate bbp value and L23 index
    txt = (f"L23 idx = {idx}\n"
           r"$b_{bp}(" + f"{WV_REF:.0f}" + r")$ = "
           f"{bbp_value:.3e} m$^{{-1}}$")
    ax.text(0.97, 0.15, txt, transform=ax.transAxes, fontsize=13,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.legend(fontsize=12)
    plotting.set_fontsize(ax, 17)

    # Zero line
    ax.axhline(1., color='k', ls='--', lw=1.)
    ax.set_ylim(0.0, None)

    # Minor tick marks
    ax.minorticks_on()
    ax.grid(True, which='major', alpha=0.5)
    ax.grid(True, which='minor', alpha=0.2)

    # Log scale y-axis
    if log10:
        ax.set_yscale('log')

    plt.tight_layout()
    outfile = f'{outroot}_L23_{idx}.png'
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
    if show:
        plt.show()
    plt.close()

def fit_lowest_bbp(outdir:str=LOWBBP_DIR, wv_ref:float=WV_REF,
                   wv_min:float=400., wv_max:float=700.,
                   scl_noise:str='PACE', add_satellite_noise:bool=True,
                   use_Gordon_G0:bool=False,
                   use_elastic:bool=False,
                   correct_RT:bool=False,
                   variable_Gordon:bool=True,
                   seed:int=1234, use_Gordon:bool=False,
                   show:bool=False, spec:dict=None):
    """Fit the lowest-bbp synthetic PACE spectrum with BING.

    Mirrors the pipeline in :func:`jr_analysis.fit_jr_rrs`: generate the
    synthetic spectrum, run the standard LM + MCMC pipeline via
    :func:`fitting.fit_me`, persist with :func:`bing.io.save_fit`, and
    render the diagnostic figure with :func:`fitting.plot_fit`.

    Parameters
    ----------
    outdir : str
        Folder for the saved fit and figure.  Created if absent.
    wv_ref, wv_min, wv_max : float
        Reference wavelength for bbp lookup and PACE grid bounds.
    scl_noise, add_satellite_noise, seed, use_Gordon : passed through to
        :func:`generate_pace_spectrum`.
    show : bool
        If True, leave the figure open after writing it.
    spec : dict, optional
        Pre-built spectrum dict from :func:`generate_pace_spectrum` to
        avoid re-loading the Loisel dataset.
    use_Gordon_G0 : bool
        If True, use the Gordon Rrs model with G0.

    Returns
    -------
    dict
        ``outroot``, ``fit_file``, ``plot_file``, ``spec`` (input
        spectrum dict), ``stats``, ``ans`` (LM best-fit), ``chains``.
    """
    os.makedirs(outdir, exist_ok=True)

    # Build the synthetic PACE spectrum if the caller did not pass one
    if spec is None:
        spec = generate_pace_spectrum(
            wv_ref=wv_ref, wv_min=wv_min, wv_max=wv_max,
            scl_noise=scl_noise, add_satellite_noise=add_satellite_noise,
            seed=seed, use_Gordon=use_Gordon, use_elastic=use_elastic)

    iwave = spec['pace_wave'].astype(float)
    ispec = spec['Rrs'].astype(float)
    isig = spec['sigRrs'].astype(float)

    # File naming follows the L23 index for traceability
    base = f"Lowbbp_L23_{spec['idx']:04d}_fits"
    if use_Gordon_G0:
        base = base.replace('Lowbbp_', 'Lowbbp_G0_')
    if use_elastic:
        base = base.replace('Lowbbp_', 'Lowbbp_E_')
        if correct_RT:
            base = base.replace('Lowbbp_E', 'Lowbbp_ECRT_')
    if use_Gordon:
        base = base.replace('L23', 'Gordon')
    if not variable_Gordon:
        base = base.replace('Lowbbp_', 'Lowbbp_Orig_')
    outroot = os.path.join(outdir, base)
    fit_file = outroot + '.npz'

    # Parameters
    if use_elastic:
        Raman = False
        Chl_fl = False
    else:
        Raman = True
        Chl_fl = True

    p = standard.expb_pow(satellite='PACE', add_noise=False,
        variable_Gordon=variable_Gordon, include_Raman=Raman,
        variable_Gordon_G0=use_Gordon_G0,
        include_Chl_fl=Chl_fl, phi_C=0.02, double_gaussian=True)

    # Correct RT
    if correct_RT:
        if not use_elastic:
            raise ValueError("Correct RT only works with elastic Rrs")
        Rrs_GordonE = bing_rt.calc_Rrs(spec['odict']['a'], 
                                       spec['odict']['bb'])
        RT_correction = spec['Rrs_true'] / Rrs_GordonE
    else:
        RT_correction = None

    # LM + MCMC via the standard pipeline used by fit_with_argo / jr_analysis
    print("=" * 80)
    print(f"Fitting lowest-bbp L23 idx={spec['idx']} "
          f"(bbp({spec['bbp_wave']:.0f})={spec['bbp_value']:.3e} m^-1)")
    print("=" * 80)
    models, chains, ans, stats, rt_dict, pdict, p = fitting.fit_me(
        [iwave, ispec, isig], in_p=p, RT_correction=RT_correction)

    # Persist the fit (writes .npz + .json)
    bing_io.save_fit(outroot, p, models, chains, ans, ispec, isig**2)
    print(f"Saved fit: {fit_file}")

    # Diagnostic plot in the style of jr_analysis.fit_jr_rrs
    title = (f"L23 idx={spec['idx']}, "
             r"$b_{bp}(" + f"{spec['bbp_wave']:.0f}" + r")$ = "
             f"{spec['bbp_value']:.3e} m$^{{-1}}$, "
             r"$a_{ph}(440)$ = "+f"{spec['aph_440']:.3e}"+r" m$^{-1}$")
    if use_Gordon:
        title = title.replace('L23', 'L23 Gordon')
    Rrs_obs = dict(wave=iwave, spec=ispec, var=isig**2)
    plot_file = outroot + '.png'
    fitting.plot_fit(models, chains, Rrs_obs, title, rt_dict,
                     stats=stats, show_Rsig=True, outfile=plot_file)
    if show:
        plt.show()
    plt.close('all')

    return dict(outroot=outroot, fit_file=fit_file, plot_file=plot_file,
                spec=spec, stats=stats, ans=ans, chains=chains)


def main(flg):
    flg = int(flg)

    # Generate the synthetic PACE spectrum + figure for the lowest-bbp model
    if flg == 1:
        spec = generate_pace_spectrum()
        plot_spectrum(spec, log10=True)

    # Fit the lowest-bbp synthetic PACE spectrum with BING
    if flg == 2:
        #fit_lowest_bbp()
        #fit_lowest_bbp(use_Gordon_G0=True)
        #fit_lowest_bbp(use_Gordon=True)
        #fit_lowest_bbp(use_elastic=True)
        fit_lowest_bbp(use_elastic=True, variable_Gordon=False)
                       #seed=2522) # Orig

        # Corrected RT
        #fit_lowest_bbp(use_elastic=True, correct_RT=True)

    # Plot various Rrs spectra for a given, low bbp example
    if flg == 3:
        compare_Rrs(rank=1) # 3003


    # Advanced figures
    if flg == 11:
        examine_fit(rank=1, 
                    fit_file='Low_bbp/Lowbbp_Orig_E_L23_3003_fits.npz',
                    outfile='Low_bbp/Examine_L23_Elastic_3003.png',
                    use_elastic=True)


# Command line
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 1
    else:
        flg = sys.argv[1]

    main(flg)

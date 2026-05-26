# Assess PACE sensitivity to bbp by fitting synthetic L23 spectra
# starting from the Hydrolight model with the smallest particulate
# backscattering.

import os, sys
import numpy as np

from matplotlib import pyplot as plt

from ocpy.utils import plotting
from ocpy.hydrolight import loisel23
from ocpy.satellites import pace as sat_pace

from bing.fitting import l23 as bing_l23
from bing.preproc import convert_to_satwave
from bing.noise import scale_noise, add_noise
from bing import io as bing_io

# Locals
import fitting

from IPython import embed

WV_REF = 700.
LOWBBP_DIR = 'Low_bbp'

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


def generate_pace_spectrum(wv_ref:float=WV_REF, wv_min:float=400.,
                           wv_max:float=700., scl_noise:str='PACE',
                           add_satellite_noise:bool=True,
                           seed:int=1234, use_Gordon:bool=False):
    """Build a synthetic PACE Rrs spectrum from the lowest-bbp L23 model.

    Loads the chosen Hydrolight spectrum, interpolates the true
    Rrs onto the PACE wavelength grid, applies the PACE
    noise vector, and optionally adds a random noise realisation.

    Parameters
    ----------
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

    Returns
    -------
    out : dict
        ``pace_wave``, ``Rrs`` (noisy if requested), ``Rrs_true`` (noise
        free), ``sigRrs``, ``bbp_value``, ``bbp_wave`` (wavelength used
        for the bbp reference), ``idx``, and ``odict`` (raw L23 data).
    """
    # Loisel 2023 dataset (cached after the first call)
    ds = loisel23.load_ds(4, 0)

    # Pick the spectrum with the smallest bbp at wv_ref
    idx, bbp_value = find_lowest_bbp_idx(ds=ds, wv_ref=wv_ref)
    print(f"Lowest bbp spectrum: idx={idx}, "
          f"bbp({wv_ref:.0f})={bbp_value:.3e} m^-1")

    # Pull the L23 spectrum (uses the same loader as bing.fitting.l23)
    odict = bing_l23.load_one_l23(idx, ds=ds, wv_min=wv_min, wv_max=wv_max)

    # PACE wavelength grid
    pace_wave = sat_pace.wave(wv_min=wv_min, wv_max=wv_max)

    # Parse Rrs
    if use_Gordon:
        Rrs_true = odict['gordon_Rrs']
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
                sigRrs=sigRrs, bbp_value=bbp_value, bbp_wave=wv_ref,
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
           f"{spec['bbp_value']:.3e} m$^{{-1}}$")
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


def fit_lowest_bbp(outdir:str=LOWBBP_DIR, wv_ref:float=WV_REF,
                   wv_min:float=400., wv_max:float=700.,
                   scl_noise:str='PACE', add_satellite_noise:bool=True,
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
            seed=seed, use_Gordon=use_Gordon)

    iwave = spec['pace_wave'].astype(float)
    ispec = spec['Rrs'].astype(float)
    isig = spec['sigRrs'].astype(float)

    # File naming follows the L23 index for traceability
    base = f"Lowbbp_L23_{spec['idx']:04d}_fits"
    outroot = os.path.join(outdir, base)
    fit_file = outroot + '.npz'

    # LM + MCMC via the standard pipeline used by fit_with_argo / jr_analysis
    print("=" * 80)
    print(f"Fitting lowest-bbp L23 idx={spec['idx']} "
          f"(bbp({spec['bbp_wave']:.0f})={spec['bbp_value']:.3e} m^-1)")
    print("=" * 80)
    models, chains, ans, stats, rt_dict, pdict, p = fitting.fit_me(
        [iwave, ispec, isig])

    # Persist the fit (writes .npz + .json)
    bing_io.save_fit(outroot, p, models, chains, ans, ispec, isig**2)
    print(f"Saved fit: {fit_file}")

    # Diagnostic plot in the style of jr_analysis.fit_jr_rrs
    title = (f"L23 idx={spec['idx']}, "
             r"$b_{bp}(" + f"{spec['bbp_wave']:.0f}" + r")$ = "
             f"{spec['bbp_value']:.3e} m$^{{-1}}$")
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
        fit_lowest_bbp()


# Command line
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 1
    else:
        flg = sys.argv[1]

    main(flg)

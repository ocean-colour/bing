"""
PACE OCI Rrs Analysis Module

This module provides functions for examining PACE OCI remote sensing reflectance (Rrs) data,
loading matched Argo BGC profiles, and analyzing fitting results.

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - xarray
    - ocpy (pace.io, utils.plotting, utils.coords, water.scattering)
    - bing (evaluate, parameters.standard, models.utils, priors, fitting)
    - grab_pace_granules (local)
    - fitting (local, as m_fitting)
"""

import os
import numpy as np
import pandas
from matplotlib import pyplot as plt

# OCPY imports
from ocpy.utils import plotting

# BING imports
from bing.parameters import standard
from bing.models import utils as model_utils

# Local imports
from grab_pace_granules import load_from_json
import fitting as m_fitting


def load_matched_data(match_file='matched_argo_bgc_profiles_bbp.csv', 
                      pace_json='PACE_50clouds.json'):
    """
    Load matched Argo BGC profiles and PACE granules.
    
    Parameters
    ----------
    match_file : str, optional
        Path to CSV file containing matched Argo profiles (default: 'matched_argo_bgc_profiles_bbp.csv')
    pace_json : str, optional
        Path to JSON file containing PACE granule information (default: 'PACE_50clouds.json')
    
    Returns
    -------
    matched : pandas.DataFrame
        DataFrame containing matched Argo profile data
    granules : list
        List of PACE granules
    pace : dict or object
        PACE data structure from JSON
    """
    # Load up Argo profiles, already matched to PACE
    matched = pandas.read_csv(match_file)
    
    # Load up PACE granules
    granules, pace = load_from_json(pace_json)
    
    return matched, granules, pace


def get_fit_file_path(matched_profile, base_dir=None):
    """
    Construct the path to a fit file for a given matched profile.
    
    Parameters
    ----------
    matched_profile : pandas.Series
        A row from the matched profiles DataFrame containing 'cruise' and 'profile' fields
    base_dir : str, optional
        Base directory for fits. If None, uses OS_COLOR environment variable
    
    Returns
    -------
    str
        Full path to the fit file
    
    Raises
    ------
    AssertionError
        If the fit file does not exist
    """
    if base_dir is None:
        base_dir = os.getenv('OS_COLOR')
    
    fit_file = os.path.join(base_dir, 'Biomass', 'Fits',
                            f'Argo_{matched_profile.cruise}_{matched_profile.profile:03d}_fits.npz')
    assert os.path.isfile(fit_file), f"Fit file not found: {fit_file}"
    
    return fit_file


def load_fit_data(fit_file):
    """
    Load fit data from an NPZ file.
    
    Parameters
    ----------
    fit_file : str
        Path to the fit file (.npz format)
    
    Returns
    -------
    numpy.lib.npyio.NpzFile
        Loaded fit data containing arrays for 'Rrs', 'Rrs_sig', 'wave', 'chains', 'model_names', etc.
    """
    return np.load(fit_file)


def plot_rrs_spectrum(fit_data, output_file='Rrs_example.png', figsize=(12, 7), 
                      color='b', fontsize=18.0, dpi=300):
    """
    Plot the Rrs spectrum with error bars.
    
    Parameters
    ----------
    fit_data : numpy.lib.npyio.NpzFile
        Fit data containing 'wave', 'Rrs', and 'Rrs_sig' arrays
    output_file : str, optional
        Path to save the output figure (default: 'Rrs_example.png')
    figsize : tuple, optional
        Figure size in inches (default: (12, 7))
    color : str, optional
        Color for the plot (default: 'b')
    fontsize : float, optional
        Font size for axis labels (default: 18.0)
    dpi : int, optional
        DPI for saved figure (default: 300)
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    matplotlib.axes.Axes
        The axes object
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Plot Rrs with error bars
    ax.errorbar(fit_data['wave'][0], fit_data['Rrs'][0], 
                yerr=fit_data['Rrs_sig'][0],
                color=color, fmt='o', capsize=5)
    
    # Set labels
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$R_{\rm rs}$')
    
    # Set font size
    plotting.set_fontsize(ax, fontsize)
    
    # Add zero line
    ax.axhline(0., color='r', ls='--')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    plt.show()
    
    return fig, ax


def plot_model_fit(fit_data, matched_profile, show_Rsig=True, outfile=None):
    """
    Plot the model fit to Rrs observations.
    
    Parameters
    ----------
    fit_data : numpy.lib.npyio.NpzFile
        Fit data containing 'wave', 'Rrs', 'Rrs_sig', 'chains', and 'model_names'
    matched_profile : pandas.Series
        Matched profile information containing 'cruise' and 'profile' for labeling
    show_Rsig : bool, optional
        Whether to show Rrs uncertainties (default: True)
    outfile : str, optional
        Path to save the output file (default: None, no file saved)
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Initialize models
    p = standard.expb_pow()
    models = model_utils.init(fit_data['model_names'], fit_data['wave'][0])
    
    # Prepare observation dictionary
    Rrs_obs = dict(
        wave=fit_data['wave'][0],
        spec=fit_data['Rrs'][0],
        var=fit_data['Rrs_sig'][0]**2
    )
    
    # Create label
    label = f'{matched_profile.cruise}_{matched_profile.profile:03d}'
    
    # Plot fit
    fig = m_fitting.plot_fit(models, fit_data['chains'], Rrs_obs, label,
                             show_Rsig=show_Rsig, outfile=outfile)
    
    return fig


def analyze_profile(match_index, matched_df, base_dir=None, 
                   plot_spectrum=True, plot_fit=True,
                   spectrum_output='Rrs_example.png'):
    """
    Complete analysis workflow for a single matched profile.
    
    Parameters
    ----------
    match_index : int
        Index of the profile in the matched DataFrame
    matched_df : pandas.DataFrame
        DataFrame containing matched profile information
    base_dir : str, optional
        Base directory for fit files (default: None, uses OS_COLOR env var)
    plot_spectrum : bool, optional
        Whether to plot the Rrs spectrum (default: True)
    plot_fit : bool, optional
        Whether to plot the model fit (default: True)
    spectrum_output : str, optional
        Output filename for spectrum plot (default: 'Rrs_example.png')
    
    Returns
    -------
    dict
        Dictionary containing:
            - 'matched_profile': The selected profile data
            - 'fit_data': Loaded fit data
            - 'fit_file': Path to fit file
    """
    # Get matched profile
    matched_profile = matched_df.iloc[match_index]
    
    # Get fit file path
    fit_file = get_fit_file_path(matched_profile, base_dir=base_dir)
    
    # Load fit data
    fit_data = load_fit_data(fit_file)
    
    # Plot spectrum if requested
    if plot_spectrum:
        plot_rrs_spectrum(fit_data, output_file=spectrum_output)
    
    # Plot model fit if requested
    if plot_fit:
        plot_model_fit(fit_data, matched_profile, show_Rsig=True,
                       outfile='Rrs_fit_example.png')
    
    return {
        'matched_profile': matched_profile,
        'fit_data': fit_data,
        'fit_file': fit_file
    }


# Example usage when run as script
if __name__ == '__main__':
    # Load data
    matched, granules, pace = load_matched_data()
    
    print(f"Loaded {len(matched)} matched profiles")
    print("\nFirst few profiles:")
    print(matched.head())
    
    # Analyze first profile
    print("\nAnalyzing first profile...")
    result = analyze_profile(0, matched, plot_spectrum=False, plot_fit=True)
    
    print(f"\nAnalyzed profile:")
    print(result['matched_profile'])

""" Methods related to priors for a_dg """
import warnings
import os

from importlib import resources
import numpy as np
import pandas

from scipy.io import matlab
from scipy.optimize import curve_fit

def exp_func(wave, A, S, pivot=440.):
    return A * np.exp(-S*(wave-pivot))

def fit_exp_to_adg(ag, ad, wvmin:float=400., wvmax:float=525.):
    """
    Fit the adg data from Kehrli2024.
    """

    # Fit the data
    p0 = [0.2, 0.015] # Amplitude, shape
    ag_fits = []
    ad_fits = []
    adg_fits = []
    adg_idx = []

    # Loop on the datasets
    for ss, adict in enumerate([ag, ad]):
        cut = (adict['wave'] > wvmin) & (adict['wave'] < wvmax)
        for iadg in range(adict['spec'].shape[0]):
            ans, cov =  curve_fit(exp_func, adict['wave'][cut], 
                                adict['spec'][iadg, cut],
                                p0=p0, #sigma=np.sqrt(varRrs),
                                full_output=False)
            if ss == 0: # ag
                ag_fits.append(ans)
            else:  # ad
                ad_fits.append(ans)
            # Combined (adg)
            if ss == 1:
                mt = np.where((adict['date'][iadg] == ag['date']) &
                              (np.abs(adict['lat'][iadg] - ag['lat']) < 0.01) &
                              (np.abs(adict['lon'][iadg] - ag['lon']) < 0.01) 
                              )[0]
                if len(mt) == 0:
                    pass
                elif len(mt) == 1:
                    adg = adict['spec'][iadg] + ag['spec'][mt[0]]
                    ans, cov =  curve_fit(exp_func, adict['wave'][cut], 
                                adg[cut],
                                p0=p0, #sigma=np.sqrt(varRrs),
                                full_output=False)
                    adg_fits.append(ans)
                    adg_idx.append(iadg)

                else:
                    warnings.warn("Multiple matches")

    # Generate tables
    ag_tbl = pandas.DataFrame(dict(Ag=[ag_fit[0] for ag_fit in ag_fits], 
                              Sg=[ag_fit[1] for ag_fit in ag_fits])) 
    ad_tbl = pandas.DataFrame(dict(Ad=[ad_fit[0] for ad_fit in ad_fits], 
                              Sd=[ad_fit[1] for ad_fit in ad_fits])) 
    adg_tbl = pandas.DataFrame(dict(Adg=[adg_fit[0] for adg_fit in adg_fits], 
                              Sdg=[adg_fit[1] for adg_fit in adg_fits],
                              ad_idx = adg_idx)) 

    # Return
    return ag_tbl, ad_tbl, adg_tbl


def load_kehrli2024():
    """
    Load the Kehrli2024 data for absorption and attenuation coefficients.
        Kindly provided by Matt Kehrli.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing two dictionaries.
            The first dictionary contains the absorption coefficient data with keys:
                - 'wave': array of wavelengths
                - 'date': array of dates
                - 'lat': latitude
                - 'lon': longitude
                - 'spec': absorption coefficient spectrum
            The second dictionary contains the attenuation coefficient data with keys:
                - 'wave': array of wavelengths
                - 'date': array of dates
                - 'lat': latitude
                - 'lon': longitude
                - 'spec': attenuation coefficient spectrum
    """
    dfile = resources.files('bing').joinpath(os.path.join('data', 'adg', 'ADG_part_data_fig2_spec.mat'))
    d = matlab.loadmat(dfile)

    # ag
    ag_wave =  d['ag']['Lambda'][0][0][0].astype(float)
    ag_date =  d['ag']['DateNum'][0][0]
    ag_lat =  d['ag']['LatN'][0][0].flatten()
    ag_lon =  d['ag']['LonE'][0][0].flatten()
    ag =  d['ag']['ag'][0][0]

    # ad
    ad_wave =  d['ad']['Lambda'][0][0][0].astype(float)
    ad_date =  d['ad']['DateNum'][0][0]
    ad_lat =  d['ad']['LatN'][0][0].flatten()
    ad_lon =  d['ad']['LonE'][0][0].flatten()
    ad =  d['ad']['ad'][0][0]

    # Turn into dicts
    ag_dict = dict(wave=ag_wave, date=ag_date, lat=ag_lat, lon=ag_lon, spec=ag)
    ad_dict = dict(wave=ad_wave, date=ad_date, lat=ad_lat, lon=ad_lon, spec=ad)

    # Return
    return ag_dict, ad_dict



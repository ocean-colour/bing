
import numpy as np
import pandas

import fitting
from grab_pace_granules import load_from_json

from shapely.vectorized import contains

from IPython import embed

def pace_for_allie(lat:float=24.797, lon:float=-93.388,
                   time='2025-08-21T17:06:12.002000128'):

    # Load up PACE granules
    granules, pace = load_from_json('PACE_50clouds.json')
    ptime = pandas.Timestamp(time, tz='UTC')

    for ss in range(len(pace)):
        inside = contains(pace.polygon.values[ss], lon, lat)
        dt = pandas.Timestamp(pace.iloc[ss].time) - ptime
        if np.abs(dt) < pandas.Timedelta('1 days') and inside:
            print(f'Found granule {pace.id.values[ss]} for {time} at {lat},{lon}')

# Command line
if __name__ == '__main__':

    fit_allie = False
    pace_me = True

    if pace_me:
        pace_for_allie()

    if fit_allie:
        # Load
        df = pandas.read_csv('allie_rrs_spectrum.csv')
        wave = df['Wavelength'].values
        Rrs = df['Rrs'].values
        gd_wave = (wave >= 400.) &  (wave <= 700.) 

        iwave = wave[gd_wave]
        ispec = Rrs[gd_wave]
        isig = np.ones_like(ispec) * 0.0005

        # Fit
        models, chains, ans, stats = fitting.fit_me(iwave, ispec, isig)
        Rrs_obs=dict(wave=models[0].wave, spec=ispec, var=isig**2)

        fitting.plot_fit(models, chains, Rrs_obs, "Allie's float", show_Rsig=True,
                   outfile='allie_fit.png')
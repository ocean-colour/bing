""" Simple script to generate Rrs data for testing purposes. """

import os
import numpy as np

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import pandas
import seaborn as sns
from matplotlib import pyplot as plt

from ocpy.hydrolight import loisel23
from ocpy.utils import plotting as oc_plotting
from ocpy.satellites import pace as sat_pace
from ocpy.water import absorption

from ocpy.satellites import modis as sat_modis
from ocpy.satellites import pace as sat_pace
from ocpy.satellites import seawifs as sat_seawifs

from bing.models import anw as boring_anw
from bing.models import bbnw as boring_bbnw
from bing.models import utils as model_utils
from bing.models import functions
from bing import inference as big_inf
from bing import rt as bing_rt
from bing import chisq_fit
from bing import plotting as bing_plot

# Load up ZP data
data_file = '../../../papers/phytoplankton/Analysis/ZP_Lee/aph_samples.csv'
df_lee = pandas.read_csv(data_file)

# Loisel
ds = loisel23.load_ds(4,0)
l23_wave = ds.Lambda.data
l23_Rrs = ds.Rrs.data
all_a = ds.a.data
all_bb = ds.bb.data
all_bbnw = ds.bbnw.data
all_adg = ds.ag.data + ds.ad.data
all_ad = ds.ad.data
all_ag = ds.ag.data
all_aph = ds.aph.data
all_anw = ds.anw.data

# aw
aw = all_a[0] - all_anw[0]
# bbw
l23_bbw = all_bb[0] - all_bbnw[0]

# Grab an example from L23
i600 = np.argmin(np.abs(l23_wave-600.))
i_bb = np.argmin(np.abs(all_bbnw[:,i600]-0.01))

# adg
f_dg = interp1d(l23_wave, all_adg[i_bb])
lee_adg = f_dg(df_lee.wave)
# Lee aw
lee_aw = absorption.a_water(df_lee.wave)
# Lee bbnw
f_bb = interp1d(l23_wave, all_bbnw[i_bb])
lee_bbnw = f_bb(df_lee.wave)
# Lee bbw
f_bbw = interp1d(l23_wave, l23_bbw)
lee_bbw = f_bbw(df_lee.wave)

# Rrs
a = lee_aw + lee_adg + df_lee.aph_1
bb = lee_bbw + lee_bbnw

Rrs = bing_rt.calc_Rrs(a, bb)


# Output
output_file = 'fit_Rrs_data.csv'
data = {}
data['wave'] = df_lee.wave
data['Rrs'] = Rrs
data['sigRrs'] = 5e-4

data['anw'] = lee_adg + df_lee.aph_1
data['bbnw'] = lee_bbnw


df = pandas.DataFrame(data)
df.to_csv(output_file, index=False)
print(f'Output saved to {output_file}')
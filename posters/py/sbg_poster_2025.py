""" Figs for SBG Poster (2025) """
import os, sys
from importlib.resources import files

import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import sigmaclip
from scipy.interpolate import interp1d
import pandas


from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.rcParams['font.family'] = 'stixgeneral'

import seaborn as sns

import corner

from ocpy.water import absorption
from ocpy.utils import plotting 
from ocpy.hydrolight import loisel23
from ocpy.satellites import pace as sat_pace
from ocpy.satellites import seawifs as sat_seawifs
from ocpy.satellites import modis as sat_modis
from ocpy.water import absorption

from bing import plotting as bing_plot
from bing.models import utils as model_utils
from bing.models import functions
from bing import evaluate

#from bing.models import anw as bing_anw
#from bing.models import bbnw as bing_bbnw
#from bing import chisq_fit
#from bing import stats as bing_stats

# Local
sys.path.append(os.path.abspath("../papers/bing_2.0/Analysis/py"))
import anly_utils_20
import param as param20

sys.path.append(os.path.abspath("../Analysis/py"))
import anly_utils

from IPython import embed


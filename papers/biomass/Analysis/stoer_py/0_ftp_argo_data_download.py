#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:11:29 2023

@author: adamstoer

This program is associated with the article:
    
    Stoer, A., & Fennel K. 2024. Carbon-centric dynamics of Earth’s 
    marine phytoplankton. PNAS.

Software Summary:
    This program downloads Sprof files from BGC-Argo GDAC FTP. 
    
    The programs reads the index files from the GDAC, then looks for and 
    downloads the associated file locally.

"""

# Import Packages
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import xarray as xr
import datetime
import warnings
warnings.simplefilter("ignore", UserWarning)
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.gridspec as gridspec
import geopandas as gpd
from shapely import geometry
from scipy import stats
from scipy.ndimage import gaussian_filter
import proplot as pplt
from sklearn.linear_model import HuberRegressor, Ridge
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
import geopandas as gpd
import urllib
import shutil
import urllib.request as request
from contextlib import closing
from itertools import groupby

# Paths for data downloads and projects
root_dir = '/Users/adamstoer/Synced Documents/data'
dep_dir = '/Users/adamstoer/Synced Documents/projects'

# Custom Fuction for grabbing files
def filegrab(root,find,start): # custom function to find all the files in a folder
    filelst = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(find):
                if filepath.startswith(start):
                    filelst.append(filepath)
    return filelst

# List of traj files available
traj_lst_df = pd.read_csv('https://data-argo.ifremer.fr/ar_index_global_traj.txt', skiprows =  8)
traj_lst_df['wmo'] = [int(flle_loc.split('/')[1]) for flle_loc in traj_lst_df.file.tolist()]

# List of Metadata Files
meta_lst_df = pd.read_csv('https://data-argo.ifremer.fr/ar_index_global_meta.txt', skiprows =  8)
meta_lst_df['wmo'] = [int(flle_loc.split('/')[1]) for flle_loc in meta_lst_df.file.tolist()]

# List of Sprof files available
data_lst_df = pd.read_csv('https://data-argo.ifremer.fr/etc/argo_sprof_index.txt', skiprows =  8)
data_lst_df['wmo'] = [int(flle_loc.split('/')[1]) for flle_loc in data_lst_df.file.tolist()]

# Look for and download each float in the Sprof list (by WMO number)
for float_in_use in data_lst_df['wmo'].tolist()[:]:
    # Check if there are BGC parameters    
    print('Processing Float #' + str(float_in_use) + ' ('\
          + str(data_lst_df['wmo'].tolist().index(float_in_use)+1) + '/' + str(len(data_lst_df)) + ')')

    # Check for metadata file and download if not available                
    float_folder = root_dir + '/bgc-argo database/bgc-argo program/' + str(float_in_use) + '/'
    if os.path.exists(float_folder)== True:
        print('File is downloaded') #sys.exit()
    #    print(float_folder)
        
    # Check if folder exists
    if os.path.exists(float_folder) == False:
        
        # Create float folder with WMO# as name
        if not os.path.exists(float_folder):
            os.makedirs(float_folder)
        
        # Set as float folder as working directory where downloads will be
        os.chdir(float_folder) # deposit metadata download here
        
        # Look for and download a metadata file
        meta_ftp_file = meta_lst_df.loc[meta_lst_df['wmo']==float_in_use,'file'].tolist()
        if len(meta_ftp_file)!=0:
            meta_ftp_file = 'ftp://ftp.ifremer.fr/ifremer/argo/dac/' + meta_ftp_file[0]
            meta_loc_file = float_folder + str(float_in_use) + '_meta.nc'
            
            if (os.path.isfile(meta_loc_file) == False) or (os.path.getsize(meta_loc_file)==0):
                print('Downloading Metadata File')                    
                with closing(request.urlopen(meta_ftp_file)) as r:
                    with open(meta_ftp_file.split('/')[-1], 'wb') as f:
                        shutil.copyfileobj(r, f)
        
        # Look for and download a Sprof file
        sprof_ftp_file = data_lst_df.loc[data_lst_df['wmo']==float_in_use,'file'].tolist()
        if len(sprof_ftp_file)!=0:
            sprof_ftp_file = 'ftp://ftp.ifremer.fr/ifremer/argo/dac/' + sprof_ftp_file[0]
            sprof_loc_file = float_folder + str(float_in_use) + '_Sprof.nc'
            
            if (os.path.isfile(sprof_loc_file) == False) or (os.path.getsize(sprof_loc_file)==0):
                print('Downloading Sprof File')
                with closing(request.urlopen(sprof_ftp_file)) as r:
                    with open(sprof_ftp_file.split('/')[-1], 'wb') as f:
                        shutil.copyfileobj(r, f)
        
        # Look for and download a trajectory file
        '''
        traj_ftp_file = traj_lst_df.loc[traj_lst_df['wmo']==float_in_use,'file'].tolist()
        if len(traj_ftp_file)!=0:
            traj_ftp_file = 'ftp://ftp.ifremer.fr/ifremer/argo/dac/' + traj_ftp_file[0]
            traj_loc_file = float_folder + str(float_in_use) + '_Rtraj.nc'
            
            if (os.path.isfile(traj_loc_file) == False) or (os.path.getsize(traj_loc_file)==0):
                print('Downloading Trajectory File')                    
                with closing(request.urlopen(traj_ftp_file)) as r:
                    with open(traj_ftp_file.split('/')[-1], 'wb') as f:
                        shutil.copyfileobj(r, f)
        '''

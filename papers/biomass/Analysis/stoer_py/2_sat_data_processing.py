#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:44:05 2023

@author: adamstoer

This program is associated with the article:
    
    Stoer, A., & Fennel K. 2024. Carbon-centric dynamics of Earth’s 
    marine phytoplankton. PNAS.
    
Software Summary: 
    This program processes the downloaded satellite imagery that creates a 
    time series of average chlorophyll-a for each 10 deg latitude and week
    of the year.
    
    This data is processed in 3_stock_bloom_calc.py to create an average
    weekly climatology for surface chlorophyll-a.

"""

# Import libaries below
import pandas as pd
import os
import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import geometry
import proplot as pplt
from shapely.geometry import Polygon
import geopandas as gpd
import scipy.io
import sys
from matplotlib.patches import Patch
#sys.path.append('/Users/adamstoer/Synced Documents/Custom Functions')
#import ocean_toolbox as ot
import statsmodels.api as sm
# Important Variables
res = 10
lat_bins = np.arange(-80,80+res,res)

root_dir = '/Users/adamstoer/Synced Documents/data'
dep_dir = '/Users/adamstoer/Synced Documents/projects'

# Custom Fuction for grabbing files from a folder
def filegrab(root,find): # custom function to find all the files in a folder
    filelst = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(find):
                filelst.append(filepath)
    return filelst

sat_files = filegrab(root_dir + '/MODIS L3 Mapped/2012-2022 9km Chla/','.nc')

chl_sat_df_lst = []
for file in sat_files:

    # Open Xarray and Mask with Global Ocean
    chl_sat_ds = xr.open_dataset(file).drop_dims('rgb')
    chl_sat_ds = chl_sat_ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True) # set spatial dims
    chl_sat_ds = chl_sat_ds.rio.write_crs("EPSG:4326", inplace=True) # set CRS
    chl_sat_ds['nocean_mask'] = chl_sat_ds['chlor_a'].where(chl_sat_ds['chlor_a'] == 1, other=1)
    chl_sat_ds = chl_sat_ds.rio.clip(glob_ocean_clipped.geometry.values,glob_ocean_clipped.crs, drop = False)
    
    # Create DataFrame
    chl_sat_df = chl_sat_ds.to_dataframe().reset_index() 
    chl_sat_df = chl_sat_df.groupby(pd.cut(chl_sat_df['lat'], np.arange(-90,100,10))).agg(chlor_a = ('chlor_a', 'mean'),
                                                                                          chlor_a_count = ('chlor_a', 'count'),
                                                                                          mask_count = ('nocean_mask', 'count'))
    chl_sat_df = chl_sat_df.reset_index()
    chl_sat_df['lat'] = chl_sat_df['lat'].apply(lambda x: x.mid).tolist()
    chl_sat_df['end_time'] = datetime.datetime.strptime(chl_sat_ds.time_coverage_start[:10], '%Y-%m-%d').date()
    chl_sat_df['start_time'] = datetime.datetime.strptime(chl_sat_ds.time_coverage_end[:10], '%Y-%m-%d').date()
    
    chl_sat_df_lst.append(chl_sat_df)
    print(sat_files.index(file))
    
chl_sat_df_merged = pd.concat(chl_sat_df_lst)
chl_sat_df_merged['mid_time'] = pd.to_datetime(chl_sat_df_merged['end_time'])+datetime.timedelta(days= 4)
chl_sat_df_merged['woy'] = chl_sat_df_merged['mid_time'].dt.isocalendar().week.astype(int)

chl_sat_df_merged.loc[chl_sat_df_merged.chlor_a_count.div(chl_sat_df_merged.mask_count)<0.7,'chlor_a'] = np.nan
chl_sat_df_merged.to_csv(dep_dir + '/Data/Processed Satellite Climatology/chl_sat_df_merged.csv')
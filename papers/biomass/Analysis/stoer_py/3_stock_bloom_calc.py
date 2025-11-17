#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: adamstoer

This program is associated with the article:
    
    Stoer, A., & Fennel K. 2024. Carbon-centric dynamics of Earth’s 
    marine phytoplankton. PNAS.

Software Summary:
    This program estimates phytoplankton carbon and chlorophyll-a from profiles 
    of particle backscattering and chlorophyll-a fluorescence, respectively.
    These profiles are collected from the biogeochemical-Argo floats.

"""

#############################################################################
# Libraries and Global Variables
#############################################################################
# Import libaries below
import pandas as pd
import os
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import datetime
import matplotlib as mpl
import matplotlib.path as mpath
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import geopandas as gpd
from shapely import geometry
from scipy import stats
import proplot as pplt
import statsmodels.formula.api as smf
import matplotlib.patches as mpatches
from shapely.geometry import Polygon, shape
import geopandas as gpd
import scipy.io
from matplotlib.patches import Patch
import ocean_toolbox as ot # you will need to run the oceano_tool.py program in a second tab
import statsmodels.api as sm
from scipy import ndimage
from pygam import GAM
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from sklearn.metrics import r2_score
import statsmodels.api as sm
import rasterio
from rasterio.enums import Resampling
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore")

# Useful Variables
res = 10
lat_bins = np.arange(-80,80+res,res)

lat_formatter = LatitudeFormatter()
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',
                                        facecolor = cfeature.COLORS['land'])

root_dir = '/Users/adamstoer/Synced Documents/data'
dep_dir = '/Users/adamstoer/Synced Documents/projects/2021/project 2103'
alpha_lst = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n']

# Set plot settings
font_bold_small = {'family' : 'TeX Gyre Heros',
             'weight' : 'bold',
             'size'   : 10}
font = {'family' : 'TeX Gyre Heros',
        'weight' : 'bold',
        'size'   : 10}
font_small = {'family' : 'TeX Gyre Heros',
              'weight' : 'normal',
              'size'   : 10} 
pplt.rc.metacolor = 'black'
pplt.rc['figure.facecolor'] = 'white'
pplt.rc.axesfacecolor = 'white'
pplt.rc['meta.width'] = 1.
pplt.rc['font.sans-serif'] = 'TeX Gyre Heros'
pplt.rc['font.size'] = 10
pplt.rc['tick.len'] = 5
mpl.rcParams.update({'font.size': 10})


def filegrab(root,find,start): #grabs files by file extensions and location
    filelst = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(find):
                if filepath.startswith(start):
                    filelst.append(filepath)
    return filelst

# Import auxillary data
glob_ocean = gpd.read_file(dep_dir + '/02 processed data/global region data/glob_ocean.shp')  # global ocean map

g15_df = pd.read_csv(dep_dir + '/02 processed data/graff et al 2015 data/graff15_data.csv') # digitized data from Graff et al. (2015)
lr_graff = stats.linregress(g15_df.bbp470,g15_df.cphyto) # run linear regression to get slope
corg_mape = g15_df.bbp470.mul(lr_graff.slope).add(lr_graff.intercept)\
    .sub(g15_df.cphyto).abs().div(g15_df.cphyto).mean() # calculate MAPE for C_phy

#############################################################################
# Gather Biogeocheical Data from Floats
#############################################################################
# Access the processed float data
argo_file_grab_lst = filegrab(dep_dir + '/02 processed data/processed float data',
                              "binned.csv",
                              dep_dir + '/02 processed data/processed float data/')

df_floats = []
for file in argo_file_grab_lst:
    wmo = file.split('/')[-1].split('_')[0]
    good_bbp_float = True
    chk_bad_bbp_float_lst = ['2902158','2902160','7901106'] # floats to discard completely
    for chk_bad_float in chk_bad_bbp_float_lst:
        if chk_bad_float in file:
            good_bbp_float = False
            print('Not using data from ' + wmo)
    
    if good_bbp_float == True:
        print('Adding data from ' + wmo)
        df = pd.read_csv(file, usecols = ['local_time','time', 'profile_index',
                                          'latitude','longitude',
                                          'depth',
                                          'profile_index','temperature','chla',
                                          'depth', 'bbp700', 'mld', 'irrad490','par'])
        df['wmo'] = file.split('/')[-1].split('_')[0]
        df = df[(df['depth']<500)]

        df = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.longitude, df.latitude), 
                              crs=glob_ocean.crs)
        df_floats.append(df)

df_main = pd.concat(df_floats, ignore_index=True, sort=False)
df_main['id'] = df_main.wmo.astype(str) + '_' + df_main.profile_index.astype(str)
df_main['local_time'] = pd.to_datetime(df_main['local_time'], format = 'mixed')
df_main['time'] = pd.to_datetime(df_main['time'], format = 'mixed')
df_main['month'] = df_main['local_time'].dt.month
df_main['year'] = df_main['local_time'].dt.year
df_main.loc[:,'bbp470'] = df_main.loc[:,'bbp700'].mul((700/470)**0.73)
df_main['lat_bin'] = pd.cut(df_main['latitude'], lat_bins).apply(lambda x: x.mid).tolist()
df_main = df_main[df_main['year']<2024] # Only data from start to 2022

df_main.loc[(df_main['wmo']=='6901473')&(df_main['local_time']>=datetime.datetime(2015,1,1)),'chla'] = np.nan
df_main.loc[(df_main['wmo']=='6901473')&(df_main['local_time']>=datetime.datetime(2015,1,1)),'bbp470'] = np.nan

df_main.loc[(df_main['wmo']=='6901650')&(df_main['local_time']>=datetime.datetime(2015,6,1)),'chla'] = np.nan
df_main.loc[(df_main['wmo']=='6901650')&(df_main['local_time']>=datetime.datetime(2015,6,1)),'bbp470'] = np.nan

df_main.loc[(df_main['wmo']=='6901687')&(df_main['local_time']>=datetime.datetime(2017,8,1)),'chla'] = np.nan
df_main.loc[(df_main['wmo']=='6901687')&(df_main['local_time']>=datetime.datetime(2017,8,1)),'bbp470'] = np.nan

df_main.loc[(df_main['wmo']=='5904218')&(df_main['local_time']>=datetime.datetime(2012,11,1)),'chla'] = np.nan
df_main.loc[(df_main['wmo']=='5904218')&(df_main['local_time']>=datetime.datetime(2012,11,1)),'bbp470'] = np.nan

df_main.loc[(df_main['wmo']=='5906207')&(df_main['local_time']>=datetime.datetime(2022,1,1)),'chla'] = np.nan
df_main.loc[(df_main['wmo']=='5906207')&(df_main['local_time']>=datetime.datetime(2022,1,1)),'bbp470'] = np.nan

df_main.loc[(df_main['wmo']=='2902209')&(df_main['local_time']>=datetime.datetime(2020,1,1)),'chla'] = np.nan
df_main.loc[(df_main['wmo']=='2902209')&(df_main['local_time']>=datetime.datetime(2020,1,1)),'bbp470'] = np.nan

df_main = df_main[(df_main['bbp470'].notna())&(df_main['chla'].notna())].copy()

df_main_irrad = df_main[df_main.irrad490.notna()].copy()

# Sample sizes
num_of_prof = df_main['id'].nunique()
print(f'Profiles with CHLA/BBP Data: {num_of_prof}')

num_of_floats = df_main['wmo'].nunique()
print(f'Floats with CHLA/BBP Data: {num_of_floats}')


num_of_irrad_prof = df_main[df_main.par.notna()]['id'].nunique()
print(f'Profiles with Raw Irradiance Data: {num_of_irrad_prof:.1f}')

#############################################################################
# Land & Ocean Masks and Surface Area
#############################################################################
# Below lowers the resolution of the topographic data - which speeds up processing
# This code is commented out but data can be obtained from:
# https://www.ncei.noaa.gov/products/etopo-global-relief-model 

'''
upscale_factor = 0.2
with rasterio.open(root_dir + '/ETOP05 Data/ETOPO_2022_v1_60s_N90W180_surface.tiff') as dataset:
    profile = dataset.profile.copy()
    
    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * upscale_factor),
            int(dataset.width * upscale_factor)
        ),
        resampling=Resampling.bilinear
    )

    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]),
        (dataset.height / data.shape[-2])
    )
    profile.update({"height": data.shape[-2],
                "width": data.shape[-1],
               "transform": transform})
    
with rasterio.open(root_dir + '/ETOP05 Data/ETOPO_2022_v1_60s_N90W180_surface_mod.tiff', "w", **profile) as dataset:
    dataset.write(data) # save this lower res data
'''
# Use the low res topograph to 
topo_image = rasterio.open(dep_dir + '/02 processed data/processed topography data/ETOPO_2022_v1_60s_N90W180_surface_mod.tiff').read(1)

glob_ocean_clipped_lst = []
depth_max = 500
cs = plt.contourf(np.where(topo_image<(-depth_max),1,np.nan),levels=1, extent = (-180,180,90,-90))
plt.close() #dont show this plot

# Finding/combining all the polygons takes awhile
polylst = []
for col in cs.collections:
    for contour_path in col.get_paths(): 
        for ncp,cp in enumerate(contour_path.to_polygons()):
            x = cp[:,0]
            y = cp[:,1]
            new_shape = geometry.Polygon([(i[0], i[1]) for i in zip(x,y)])
            if ncp == 0:
                poly = new_shape
            else:
                poly = poly.difference(new_shape)
        polylst.append(poly)

mask = gpd.GeoDataFrame(geometry=polylst, crs="EPSG:4326")  # this is the mask for waters >2000m

mask['combine'] = 'combine'
mask = mask.dissolve(by = 'combine')
print('Mask made')

glob_ocean_clipped = glob_ocean.clip(mask) # 500 m depth mask applied to global_ocean
glob_ocean_clipped['area'] = glob_ocean_clipped.to_crs(epsg=6933).area # Calculate area (m^2)

df_area = glob_ocean_clipped.copy()
df_area = df_area[['area','lat_bin','basin']]
df_area['volume'] = df_area['area'].mul(5) #meters cubed or m^3

cs = plt.contourf(np.where(topo_image<(-depth_max),1,np.nan),levels=1, extent = (-180,180,90,-90))
plt.close() #dont show this plot

# Finding/combining all the polygons takes awhile
polylst = []
for col in cs.collections:
    for contour_path in col.get_paths(): 
        for ncp,cp in enumerate(contour_path.to_polygons()):
            x = cp[:,0]
            y = cp[:,1]
            new_shape = geometry.Polygon([(i[0], i[1]) for i in zip(x,y)])
            if ncp == 0:
                poly = new_shape
            else:
                poly = poly.difference(new_shape)
        polylst.append(poly)

mask = gpd.GeoDataFrame(geometry=polylst, crs="EPSG:4326")  # this is the mask for waters >2000m

mask['combine'] = 'combine'
mask = mask.dissolve(by = 'combine')
print('Mask made')

glob_ocean = glob_ocean.clip(mask)

# Create a land mask for maps
cs = plt.contourf(np.where(topo_image>(-0),1,np.nan),levels=1, extent = (-180,180,90,-90))
plt.close() #dont show this plot

# Finding/combining all the polygons takes awhile
polylst = []
for col in cs.collections:
    for contour_path in col.get_paths(): 
        for ncp,cp in enumerate(contour_path.to_polygons()):
            x = cp[:,0]
            y = cp[:,1]
            new_shape = geometry.Polygon([(i[0], i[1]) for i in zip(x,y)])
            if ncp == 0:
                poly = new_shape
            else:
                poly = poly.difference(new_shape)
        polylst.append(poly)

land_plot = gpd.GeoDataFrame(geometry=polylst, crs="EPSG:4326")  # this is the mask for waters >2000m

land_plot['combine'] = 'combine'
land_plot = land_plot.dissolve(by = 'combine')
print('Mask made')

#############################################################################
# Irradiance QC and Slope Factor Calculations
#############################################################################
# Below is commented out to save time and instead skips to using the
# processing from 'chla_cal_data.csv'
'''
chla_cal_data = df_main_irrad.groupby('id').first().reset_index()

df_main_irrad['irrad490_qc'] = np.nan
df_main_irrad['par_qc'] = np.nan

df_main_irrad.loc[(df_main_irrad['irrad490']<-0.001)|(df_main_irrad['irrad490']>3.4),'irrad490_qc'] = 3
df_main_irrad.loc[(df_main_irrad['irrad490']>-0.001)&(df_main_irrad['irrad490']<3.4),'irrad490_qc'] = 1
df_main_irrad = df_main_irrad[df_main_irrad['irrad490_qc']==1]

df_main_irrad.loc[(df_main_irrad['par']<-1)|(df_main_irrad['par']>4672),'par_qc'] = 3
df_main_irrad.loc[(df_main_irrad['par']>-1)&(df_main_irrad['par']<4672),'par_qc'] = 1
df_main_irrad = df_main_irrad[df_main_irrad['par_qc']==1]

# df_main[df_main.irrad490.notna()]['id'].unique()
irrad_prof_id = list(df_main_irrad[df_main_irrad.irrad490.notna()]['id'].unique())

# Fourth order polynomial function for curve fit
def poly_fit(x, a4, a3, a2, a1, a0):
    return (a4 * x**4) + (a3 * x**3) + (a2 * x**2) + (a1 * x) + a0

# Processing the irradiance data/ can take a couple hours
# This data is saved if you prefer to skip the loop below (see 'chla_cal_data.csv')

counter = 0
for pid in irrad_prof_id:
    counter = counter + 1
    
    #print(counter)
    #print(np.round(irrad_prof_id.index(pid)/len(irrad_prof_id),3)*100)
    df_irrad = df_main_irrad[df_main_irrad['id']==pid][['chla','depth','irrad490','irrad490_qc',
                                                        'par','par_qc',
                                                        'latitude','longitude', 'time','mld']].copy()    
    for var in ['irrad490','par']:
        if len(df_irrad[df_irrad[var + '_qc']==1])<5:
            df_irrad[var + '_qc'] = 3
            
        if len(df_irrad[df_irrad[var + '_qc']==1])>5:
            kd490 = np.nan
            meanchla = np.nan
            
            loc = coord.EarthLocation(lon = df_irrad.longitude.values[0] * u.deg,
                                      lat = df_irrad.latitude.values[0] * u.deg)
            
            now = Time(df_irrad.time.values[0], format = 'datetime64')
            
            altaz = coord.AltAz(location=loc, obstime=now)
            sun = coord.get_sun(now)
            sun_elev = float((sun.transform_to(altaz).alt) / u.deg)
        
            if sun_elev < 2:
                df_irrad[var + '_qc'] = 3
        
            if sun_elev > 2:
                p, n = 0, 0
                dark_data = df_irrad[var].dropna().values    
                while (p < 0.01) and (len(dark_data)>4): 
                    _,p = sm.stats.diagnostic.lilliefors(dark_data, dist='norm', pvalmethod='table')         
                    if (p < 0.01):
                        dark_data = dark_data[1:]
                        n = n + 1
                    if (p > 0.01):
                        break
                
                dark_value = dark_data[0]
    
                df_irrad.loc[:, var] = df_irrad.loc[:, var].sub(dark_value)                
    
                df_irrad.loc[df_irrad[var] <= 0, var + '_qc'] = 2
                df_irrad.loc[df_irrad[var] <= 0, var] = 0
                
                if len(df_irrad[df_irrad[var + '_qc']==1][:n])<=5:
                    df_irrad[var + '_qc'] = 3
                    
                if len(df_irrad[df_irrad[var + '_qc']==1][:n])>5:
                    
                                
                    x = df_irrad[df_irrad[var + '_qc']==1]['depth'].values
                    y = df_irrad[df_irrad[var + '_qc']==1][var].apply(np.log).values
                        
                    popt, pcov = scipy.optimize.curve_fit(poly_fit,  x,  y,  
                                                          p0=(0.1,0.1,0.1,0.1,0.1), 
                                                          maxfev=100000)
                    
                    r2 = r2_score(df_irrad[df_irrad[var + '_qc']==1][var].apply(np.log), 
                                  poly_fit(df_irrad[df_irrad[var + '_qc']==1]['depth'], *popt))
                    x_pred = np.linspace(df_irrad[df_irrad[var + '_qc']==1]['depth'].min(),
                                         df_irrad[df_irrad[var + '_qc']==1]['depth'].max(),100)
                    
                    # Residual fit  
                    res = (df_irrad[df_irrad[var + '_qc']==1][var].apply(np.log)-poly_fit(df_irrad[df_irrad[var + '_qc']==1]['depth'], *popt))
                    mean_res, std_res = np.mean(res), np.std(res)
                    
                    df_irrad.loc[(df_irrad[var + '_qc']==1)&(res >= (mean_res + 2*std_res)), var + '_qc'] = 3
                    df_irrad.loc[(df_irrad[var + '_qc']==1)&(res <= (mean_res - 2*std_res)), var + '_qc'] = 3
                    
                    
                    if (r2 <= 0.995) or (len(df_irrad[df_irrad[var + '_qc']==1])<=5):
                        df_irrad[var + '_qc'] = 3
                        
                    if (r2 > 0.995) and (len(df_irrad[df_irrad[var + '_qc']==1])>5):
                        # Second poly fit      
                        x = df_irrad[df_irrad[var + '_qc']==1]['depth'].values
                        y = df_irrad[df_irrad[var + '_qc']==1][var].apply(np.log).values
                            
                        popt, pcov = scipy.optimize.curve_fit(poly_fit,  x,  y,  
                                                              p0=(0.1,0.1,0.1,0.1,0.1), 
                                                              maxfev=100000)
                        
                        r2 = r2_score(df_irrad[df_irrad[var + '_qc']==1][var].apply(np.log), 
                                      poly_fit(df_irrad[df_irrad[var + '_qc']==1]['depth'], *popt))
    
                        # Residual fit
                        res = (df_irrad[df_irrad[var + '_qc']==1][var].apply(np.log)-poly_fit(df_irrad[df_irrad[var + '_qc']==1]['depth'], *popt))                        
                        mean_res, std_res = np.mean(res), np.std(res)
                        
                        df_irrad.loc[(df_irrad[var + '_qc']==1)&(res >= (mean_res + 2*std_res)), var + '_qc'] = 3
                        df_irrad.loc[(df_irrad[var + '_qc']==1)&(res <= (mean_res - 2*std_res)), var + '_qc'] = 3
                        
                        if (r2 <= 0.996) or (len(df_irrad[df_irrad[var + '_qc']==1])<=5):
                            df_irrad[var + '_qc'] = 3
                            
                        if (r2 > 0.996) and (len(df_irrad[df_irrad[var + '_qc']==1])>5):
                            if ((df_irrad[df_irrad[var + '_qc']==1]['depth'].min()<=12.5)):
                                    x = df_irrad[df_irrad[var + '_qc']==1]['depth'].values
                                    y = df_irrad[df_irrad[var + '_qc']==1][var].apply(np.log).values
                                        
                                    popt, pcov = scipy.optimize.curve_fit(poly_fit,  x,  y,  
                                                                          p0=(0.1,0.1,0.1,0.1,0.1), 
                                                                          maxfev=100000)
                                    
                                    # Interpolate missing data                         
                                    df_irrad.loc[(df_irrad[var + '_qc']!=1)&(df_irrad[var] > dark_value),var] =\
                                        (np.e**poly_fit(df_irrad.loc[(df_irrad[var + '_qc']!=1)&(df_irrad[var] > dark_value)].depth,*popt))
                                    df_irrad.loc[(df_irrad[var + '_qc']!=1)&(df_irrad[var] > dark_value),var + '_qc'] = 5
                                    
                                    depth_max = df_irrad[df_irrad[var + '_qc']==1]['depth'].max()
                                    modelled_z = np.linspace(0, depth_max, int((depth_max/0.5) + 1))
                                    modelled_ed = np.e**poly_fit(modelled_z,*popt)
                                    
                                    modelled_ed_df = pd.DataFrame({'modelled_z': modelled_z,
                                                                   'modelled_ed': modelled_ed})
                                    
                                    # Based on the 1% light threshold
                                    ez_irrad_thres = (np.e**poly_fit(0,*popt))*(0.01)
                                    ez_data = df_irrad[(df_irrad[var + '_qc'].isin([1,5]))&(df_irrad[var]>ez_irrad_thres)].copy()
                                    if len(ez_data)>=2:
                                        lr_ez = stats.linregress(ez_data['depth'].values, ez_data[var].apply(np.log).values)
        
                                        avg_chlaf = ez_data['chla'].mean()
                                        kd = lr_ez.slope*-1
                                        
                                        # Add these variables to dataset
                                        chla_cal_data.loc[chla_cal_data['id']==pid,'kd_' + var] = kd
                                        chla_cal_data.loc[chla_cal_data['id']==pid,'lr_r2_' + var] = lr_ez.rvalue**2
                                        chla_cal_data.loc[chla_cal_data['id']==pid,'ed_0_poly_' + var] = (np.e**poly_fit(0,*popt))
                                        chla_cal_data.loc[chla_cal_data['id']==pid,'r2_poly' + var] = r2
                                        chla_cal_data.loc[chla_cal_data['id']==pid,'kd_n' + var] = len(df_irrad[df_irrad[var + '_qc'].isin([1,5])])
                                        
                                        if var == 'irrad490':
                                            chla_cal_data.loc[chla_cal_data['id']==pid,'avg_chlaf_' + var] = avg_chlaf
                                            
                                        print('added')
'''
chla_cal_data = pd.read_csv(dep_dir + '/02 processed data/processed chla-irrad data/chla_cal_data.csv')

# Prepare cal data for chla flor
cal_data = chla_cal_data.copy()

cal_data['chla_kd'] = ((cal_data['kd_irrad490'].sub(0.01660)).div(0.077298)).pow(1/0.67155) # Merged LOV + NOMAD Dataset
#cal_data['chla_kd'] = ((cal_data['kd_irrad490'].sub(0.01660)).div(0.082530)).pow(1/0.62588) # Merged LOV Dataset

cal_data['sf'] = cal_data['avg_chlaf_irrad490'].div(cal_data['chla_kd'])
cal_data['z90'] = cal_data['kd_par'].pow(-1)

cal_data.loc[(cal_data['kd_irrad490']<0.01660)|(cal_data['kd_irrad490']>0.5)|\
             (cal_data['sf']<0)|(cal_data['sf']>30)|(cal_data['lr_r2_irrad490']<0.9), 'chla_kd'] = np.nan
cal_data.loc[(cal_data['kd_irrad490']<0.01660)|(cal_data['kd_irrad490']>0.5)|\
             (cal_data['sf']<0)|(cal_data['sf']>30)|(cal_data['lr_r2_irrad490']<0.9), 'sf'] = np.nan
    

cal_data.loc[(cal_data['lr_r2_par']<0.9),'z90'] = np.nan
cal_data.loc[(cal_data['lr_r2_par']<0.9),'kd_par'] = np.nan
del cal_data['lat_bin']

cal_data = gpd.GeoDataFrame(cal_data, geometry = gpd.points_from_xy(cal_data.longitude, cal_data.latitude), 
                            crs=glob_ocean.crs)
cal_data = gpd.sjoin(cal_data, glob_ocean, predicate ='within')

cal_data.loc[:,'sf'] = (cal_data.loc[:,'sf']).add(1).apply(np.log)
cal_data.loc[:,'z90'] = (cal_data.loc[:,'z90']).add(1).apply(np.log)

cal_data_group = cal_data.groupby(['basin','lat_bin','month']).mean(numeric_only= True).reset_index()
cal_data_group = cal_data_group.groupby(['basin','lat_bin']).mean(numeric_only= True).reset_index()

cal_data_group.loc[:,'sf'] = (np.e**(cal_data_group.loc[:,'sf'])).sub(1)
cal_data_group.loc[:,'z90'] = (np.e**(cal_data_group.loc[:,'z90'])).sub(1)
cal_data.loc[:,'sf'] = (np.e**(cal_data.loc[:,'sf'])).sub(1)
cal_data.loc[:,'z90'] = (np.e**(cal_data.loc[:,'z90'])).sub(1)


for basin in cal_data_group['basin'].unique():
    
    for lat_bin in cal_data_group['lat_bin'].unique():        
        if len(cal_data_group[(cal_data_group['basin']==basin)&(cal_data_group['lat_bin']==lat_bin)])!=0:
            sf = cal_data_group[(cal_data_group['basin']==basin)&(cal_data_group['lat_bin']==lat_bin)].sf.values[0]
            data_sub = cal_data[(cal_data['basin']==basin)&(cal_data['lat_bin']==lat_bin)].copy()
                        
            chla_pre = data_sub.avg_chlaf_irrad490.div(sf)
            chla_act = data_sub.chla_kd
            
            abs_err = chla_pre.sub(chla_act).abs().div(chla_act)
            mape = abs_err.mean() # MApE in depth avg chlorophyll-a 
            cal_data_group.loc[(cal_data_group['basin']==basin)&(cal_data_group['lat_bin']==lat_bin), 'chla_err'] = mape

            # Scale Factor Error
            sf_pre = sf
            sf_act = data_sub.sf
            
            abs_err = sf_act.sub(sf_pre).abs()
            mape = data_sub.sf.std() # MAE in depth avg chlorophyll-a
            cal_data_group.loc[(cal_data_group['basin']==basin)&(cal_data_group['lat_bin']==lat_bin), 'sf_err'] = mape

cal_map = glob_ocean.merge(cal_data_group, on = ['basin','lat_bin'], how = 'left')
#cal_map.plot(column = cal_map.sf, cmap = 'jet')
#cal_data.geometry.plot(markersize = 1, color = 'black')

for basin in ['pacific','atlantic','indian']:
    cal_map.loc[cal_map.basin == basin, 'sf'] = \
        cal_map[cal_map.basin == basin]['sf']\
            .interpolate(method = 'linear', limit_direction = 'both', limit = 5)\
            .rolling(window = 3, min_periods = 1, center=True).mean()

    cal_map.loc[cal_map.basin == basin, 'sf_err'] = \
        cal_map[cal_map.basin == basin]['sf_err']\
            .interpolate(method = 'linear', limit_direction = 'both', limit = 5)\
            .rolling(window = 3, min_periods = 1, center=True).mean()
        
    cal_map.loc[cal_map.basin == basin, 'chla_err'] = \
        cal_map[cal_map.basin == basin]['chla_err']\
            .interpolate(method = 'linear', limit_direction = 'both', limit = 5)\
            .rolling(window = 3, min_periods = 1, center=True).mean()
            
    cal_map.loc[cal_map.basin == basin, 'z90'] = \
        cal_map[cal_map.basin == basin]['z90']\
            .interpolate(method = 'linear', limit_direction = 'both', limit = 5)\
            .rolling(window = 3, min_periods = 1, center=True).mean()
                
cal_map = cal_map.sort_values(by = ['basin','lat_bin'])

###############################################################
# Fig. S2: Map of Light Profiles and Slope Factor Distribution
###############################################################
array = [[0,1,1,0],
         [2,2,3,3],
         [0,4,4,0]]

fig, ax = pplt.subplots(array, width = 6, height = 6, proj = {1:'moll'}, 
                        proj_kw={'central_longitude': -60}, 
                        hratios = (3,2,2), abc = 'A', tight = True)

ax[0].format(grid=False)
ax[0].set_global()
ax[0].set_facecolor('gray6')
legend_elements = [Patch(facecolor='blue3', edgecolor='blue3', label='Indian Ocean'),
                   Patch(facecolor='blue2', edgecolor='blue2', label='Atlantic Ocean'),
                   Patch(facecolor='blue1', edgecolor='blue1', label='Pacific Ocean')]
ax[0].legend(handles=legend_elements, ncol = 1, bbox_to_anchor = [0.0, 0.5], fontsize = 12, frameon = False)

ax[0].add_feature(cfeature.LAKES, edgecolor='black', facecolor = 'gray6', zorder = 59, lw = 0.5)
ax[0].add_geometries(land_plot.geometry.values, crs=ccrs.PlateCarree(), facecolor = 'gray4', 
                     edgecolor='black', zorder = 35, lw = 0.5, label = 'Land')

ax[0].add_geometries(glob_ocean[glob_ocean['basin'] == 'pacific'].geometry.values, crs=ccrs.PlateCarree(), facecolor = 'blue1', 
                     edgecolor='blue1', zorder = 10, lw = 1, label = 'Pacific Ocean')
ax[0].add_geometries(glob_ocean[glob_ocean['basin'] == 'atlantic'].geometry.values, crs=ccrs.PlateCarree(), facecolor = 'blue2', 
                     edgecolor='blue2', zorder = 10, lw = 1, label = 'Atlantic Ocean')
ax[0].add_geometries(glob_ocean[glob_ocean['basin'] == 'indian'].geometry.values, crs=ccrs.PlateCarree(), facecolor = 'blue3', 
                     edgecolor='blue3', zorder = 10, lw = 1, label = 'Indian Ocean')

ax[0].plot(np.repeat(146.916714, 50), np.linspace(-90, -35, 50), color = 'black', 
        lw = 2, transform = ccrs.PlateCarree(), zorder = 28, label = '')
ax[0].plot(np.repeat(-67.25, 50), np.linspace(-65, -55, 50), color = 'black', 
        lw = 2, transform = ccrs.PlateCarree(), zorder = 28, label = '')
ax[0].plot(np.repeat(20, 50), np.linspace(-90, -35, 50), color = 'black', 
        lw = 2, transform = ccrs.PlateCarree(), zorder = 28, label = '')
ax[0].scatter(cal_data[cal_data['sf'].notna()].longitude,cal_data[cal_data['sf'].notna()].latitude, s = 0.5, zorder = 25, marker = "o",
              color = 'black', alpha = 1, transform = ccrs.PlateCarree(), label = '')

# Plot Slope Factor Transects
ax[1].set_ylabel('Slope Factor (unitless)', labelpad = 8)
ax[3].set_ylabel('Slope Factor (unitless)', labelpad = 8)
ax[3].set_xlabel('Latitude', labelpad = 8)

color_lst = ['blue3','blue2','blue1']
basin_lst = ['indian','atlantic','pacific']
label_lst = ['Indian Ocean','Atlantic Ocean','Pacific Ocean']
marker_lst = ['o','^','s']
adder = 0
for basin in basin_lst:
    ind = basin_lst.index(basin)
    ax[ind+1].format(grid=False, title = label_lst[ind])
    ax[ind+1].set_ylim(0,6)
    ax[ind+1].set_yticks(np.linspace(0,6,4))        
    ax[ind+1].axhline(1, color = 'black', ls = 'dashed')
    ax[ind+1].xaxis.set_major_formatter(lat_formatter)
    ax[ind+1].set_xticks(np.arange(-90,100,30))
    ax[ind+1].xaxis.set_minor_locator(MultipleLocator(10))

    ax[ind+1].set_xlim(-90,90)
    ind = basin_lst.index(basin)
    ax[ind+1].errorbar(cal_map[cal_map.basin == basin].lat_bin.values, cal_map[cal_map.basin == basin].sf.values,
                       label = label_lst[ind], markerfacecolor = color_lst[ind], markersize = 6, marker = 'o',
                       capsize = 0, elinewidth = 1, color = 'black',
                       yerr = cal_map[cal_map.basin == basin]['sf_err'].values)
    adder = adder + 10/3


#plt.savefig(dep_dir + '/03 figures/publish quality/fig s2.jpg', dpi = 300)
plt.show()

#############################################################################
# Global and Ocean Basin Cphyto & Chla Calculations
#############################################################################
# Seperate data into each basin
df_profiles = df_main[['id','longitude','latitude']].groupby('id').first().reset_index()
df_profiles = gpd.GeoDataFrame(df_profiles, geometry = gpd.points_from_xy(df_profiles.longitude, df_profiles.latitude), 
                               crs=glob_ocean.crs)

reg_clim_lst = []    
for basin in glob_ocean['basin'].unique():
    #print(basin)
    for lat in glob_ocean[glob_ocean['basin'] == basin]['lat_bin'].unique(): 
        #print(lat)
        ocean_poly = glob_ocean[(glob_ocean['basin'] == basin)&(glob_ocean['lat_bin'] == lat)]
        minlon , minlat , maxlon , maxlat = ocean_poly.total_bounds # Faster processing wtih dropping float data out of bounds
        
        df_profiles_present = df_profiles[(df_profiles['longitude']>=minlon)&(df_profiles['longitude']<=maxlon)&\
                                          (df_profiles['latitude']>=minlat)&(df_profiles['latitude']<=maxlat)]
        ava_profiles = gpd.sjoin(df_profiles_present, ocean_poly, predicate ='within')
        
        if len(ava_profiles)!=0:
                
            df_main.loc[df_main['id'].isin(ava_profiles.id.tolist()),'basin'] = basin
            
            reg_sub = df_main[(df_main['id'].isin(ava_profiles.id.tolist()))]
            
            t_res = 'woy'
            reg_sub['woy'] = reg_sub['local_time'].dt.isocalendar().week.astype(int)
            reg_sub = reg_sub[reg_sub['woy']!=53]
            
            sf = cal_map[(cal_map['basin']==basin)&(cal_map['lat_bin']==lat)].sf.values[0]
            
            reg_sub.loc[:,'chla'] = reg_sub.loc[:,'chla'].div(sf)
            
            reg_sub.loc[:,'bbp470'] = reg_sub.loc[:,'bbp470'].add(1).apply(np.log)
            reg_sub.loc[:,'chla'] = reg_sub.loc[:,'chla'].add(1).apply(np.log)
                        
            reg_sub = reg_sub.groupby([t_res,'depth'], dropna = False).agg({'bbp470': 'mean',
                                                                            'chla': 'mean',
                                                                            'mld':'mean',
                                                                            'temperature':'mean',
                                                                            'id': 'nunique'})
            reg_sub = reg_sub.reset_index()
            reg_sub.loc[:,'bbp470'] = (np.e**(reg_sub.loc[:,'bbp470'])).sub(1)
            reg_sub.loc[:,'chla'] = (np.e**(reg_sub.loc[:,'chla'])).sub(1)
            
                 
            
            for t_ind in reg_sub.woy.unique():
                reg_sub.loc[reg_sub[t_res]==t_ind,'bbpphy'], z_lim = ot.sep_bbp(reg_sub[reg_sub[t_res]==t_ind],'depth','chla','bbp470')
                reg_sub.loc[reg_sub[t_res]==t_ind,'cphy'] = ot.bbp_to_cphy(reg_sub.loc[reg_sub[t_res]==t_ind,'bbpphy'], lr_graff.slope)
                reg_sub.loc[reg_sub[t_res]==t_ind,'z_lim'] = z_lim
                
            '''
            ###################################################
            # Fig. S3: CPHYTO CALCULATION ILLUSTRATION
            ###################################################
            array = [[1,2]]
            fig, axes = pplt.subplots(array, width = 4, height = 4, ylabel = 'Depth (m)', abc = 'A',
                                      ylim = (500,0), sharex = False, sharey = True, grid = False)
            
            w = 40
            axes[0].plot(reg_sub.loc[(reg_sub['woy']==w),'bbp470'].values,reg_sub.loc[(reg_sub['woy']==w),'depth'].values,
                         color = 'orange', lw = 2, label = 'bbp')
            
            data = reg_sub[reg_sub[t_res]==w]

            name_chla = 'chla'
            name_z = 'depth'
            name_bbp = 'bbp470'
            dcm = data[data.loc[:,name_chla]==data.loc[:,name_chla].max()][name_z].values[0]        # Find depth of deep chla maximum
            part_prof = data[(data.loc[:,name_bbp]<np.median(data.loc[:,name_bbp]))]                # find median bbp of profile
            
            #part_prof.loc[:,'bbp470_nom'] = unp.nominal_values(part_prof.loc[:,'bbp470'])
            
            mod = smf.quantreg('bbp470 ~ ' + str(name_z), 
                               part_prof).fit(q=0.01)                                   # Find model to 1 percentile
            y_pred = mod.predict(part_prof.loc[:,name_z])                                     # Create predicted bbp_nap
            
            part_prof.loc[:,'bbp_back'] = y_pred.values                                 # Predicted bbp NAP from linear trend
            z_lim = part_prof.loc[(part_prof.loc[:,'bbp_back'].div(part_prof.loc[:,name_bbp])>=1), name_z].min()                                         
                    
            #lr = stats.linregress([0,z_lim],[0,data.loc[data[name_z]==z_lim, name_bbp].values[0]])
            
                        
            axes[0].plot(part_prof.loc[(part_prof['woy']==w),'bbp_back'].values,part_prof.loc[(part_prof['woy']==w),'depth'].values,
                         color = 'black', ls = 'dashed', lw = 2, label = 'Regression', zorder = 30)       
        
            # Find depth where bbp NAP and bbp intersect
            data.loc[data[name_z]>=z_lim, 'bbp_back'] = data.loc[data[name_z]>=z_lim, name_bbp].tolist()
            data.loc[data[name_z]<z_lim,'bbp_back'] = data.loc[data[name_z]==z_lim, name_bbp].values[0] #data.loc[data[name_z]<z_lim, name_z].mul(lr.slope).add(lr.intercept)

            data.loc[:,'bbpphy'] = data.loc[:, name_bbp].sub(data.loc[:,'bbp_back'])      # Subtract bbp NAP from bbp for bbp from phytoplankton
            data.loc[(data['bbpphy']<0)|(data['depth']>z_lim),'bbpphy'] = 0     # Subtract bbp NAP from bbp for bbp from phytoplankton

            axes[0].axhline(z_lim, lw = 1, color = 'black')
            axes[0].plot(data.loc[(data['woy']==w),'bbp_back'].values,data.loc[(data['woy']==w),'depth'].values,
                         color = 'yellow brown', lw = 2, label = 'bbp$_{NAP}$')       
            axes[0].plot(reg_sub.loc[(reg_sub['woy']==w),'bbpphy'].values,reg_sub.loc[(reg_sub['woy']==w),'depth'].values,
                         color = 'brown', lw = 2, label = 'bbp$_{phy}$')       
            axes[0].set_xlabel('bbp (m$^{-1}$)')
            axes[0].set_xlim(0,0.0012)

            axes[0].legend(loc = 'lower right', ncol = 1, prop={'size': 7})
            axes[1].plot(reg_sub.loc[(reg_sub['woy']==w),'chla'].values,reg_sub.loc[(reg_sub['woy']==w),'depth'].values,
                         color = 'green', lw = 2)                   
            axes[1].set_xlabel('Chla (mg m$^{-3}$)')
            axes[1].set_xlim(0,0.4)
            plt.savefig(dep_dir + '/03 figures/publish quality/fig_s3.jpg', dpi = 300)
            plt.show()
            '''
            
            reg_sub_before = reg_sub.copy()
            reg_sub_before['woy'] = reg_sub_before['woy'].sub(52)
            reg_sub_after = reg_sub.copy()
            reg_sub_after['woy'] = reg_sub_after['woy'].add(52)
    
            reg_clim = pd.concat([reg_sub_before,reg_sub,reg_sub_after]).reset_index(drop=False)
            
            for w in np.arange(1-52,53+52,1):
                if w not in reg_clim.woy.unique():                    
                    empty_df = reg_clim.loc[reg_clim['woy']==1, :]
                    empty_df.loc[empty_df['woy']==1, 'woy'] = w
                    empty_df.loc[empty_df['woy']==w, 'id_nunique'] = 0
                    empty_df.loc[empty_df['woy']==w, ['bbp470', 'chla','mld','temperature']] = np.nan

                    reg_clim = pd.concat([reg_clim,empty_df])
                    
            reg_clim = reg_clim.sort_values(by = ['woy','depth'])
            
            for d in reg_clim['depth'].unique():
                for var in ['bbp470', 'chla','temperature']:
                    reg_clim.loc[reg_clim['depth']==d,var] = reg_clim.loc[reg_clim['depth']==d,var].interpolate(method = 'linear')
                    
            win = 9
            for d in reg_clim['depth'].unique():
                for var in ['cphy','chla']:
                    reg_clim.loc[reg_clim['depth']==d,var] = reg_clim.loc[reg_clim['depth']==d,var]\
                        .rolling(window = win, min_periods = 1, center=True, closed = 'right').median().rolling(window = win, min_periods = 1, center=True, closed = 'right').mean()

            smooth_mld = reg_clim.groupby('woy')['mld'].first().reset_index()
            smooth_mld.loc[:,'mld'] = ndimage.uniform_filter1d(ndimage.median_filter(smooth_mld.loc[:,'mld'].interpolate(method = 'linear'), size = win,
                                                                                            mode = 'wrap'), size = win, mode = 'wrap')
            for w in smooth_mld['woy'].unique():
                reg_clim.loc[reg_clim['woy']==w,'mld'] = smooth_mld.loc[smooth_mld['woy']==w,'mld'].values[0]
            
            reg_clim = reg_clim[(reg_clim['woy']>=1)&(reg_clim['woy']<=52)]
            
            
            poly_vol = df_area[(df_area.basin == basin)&(df_area.lat_bin==lat)].reset_index() # in m^2
            
            reg_clim.loc[:,'volume'] = poly_vol.loc[:,'volume'].values[0]
            reg_clim.loc[:,'area'] = poly_vol.loc[:,'area'].values[0]
                
            reg_clim['basin'] = basin
            reg_clim['lat'] = lat # deg N
            reg_clim_lst.append(reg_clim)
        
glob_df = pd.concat(reg_clim_lst).reset_index(drop=True)
  
for basin in glob_ocean['basin'].unique():
    print(basin)
    #basin = 'atlantic'
    for lat in glob_ocean[glob_ocean['basin'] == basin]['lat_bin'].unique():
        data_ava = glob_df[(glob_df['basin']==basin)&(glob_df['lat']==lat)]
        
        if len(data_ava)==0:
            print(lat)
            lat_lst = glob_df[(glob_df['basin']==basin)].lat.unique()
            closest_lat = min(lat_lst, key=lambda x:abs(x-lat))
            
            reg_clim = glob_df[(glob_df['basin']==basin)&(glob_df['lat']==closest_lat)]
            
            poly_vol = df_area[(df_area['basin'] == basin)&(df_area['lat_bin']==lat)].reset_index() # in m^2
            
            reg_clim.loc[:,'volume'] = poly_vol.loc[:,'volume'].values[0]
            reg_clim.loc[:,'area'] = poly_vol.loc[:,'area'].values[0]
                
            reg_clim['basin'] = basin
            reg_clim['lat'] = lat #deg N
            reg_clim_lst.append(reg_clim)

glob_df = pd.concat(reg_clim_lst).reset_index(drop=True)
cal_map['lat'] = cal_map['lat_bin']
glob_df = glob_df.merge(cal_map[['chla_err','basin','lat','sf','z90']],on= ['basin','lat'])

glob_df['chla_l'] = glob_df['chla'].sub(glob_df['chla'].mul(glob_df['chla_err']))
glob_df['chla_u'] = glob_df['chla'].add(glob_df['chla'].mul(glob_df['chla_err']))

glob_df['cphy_l'] = glob_df['cphy'].sub(glob_df['cphy'].mul(corg_mape))
glob_df['cphy_u'] = glob_df['cphy'].add(glob_df['cphy'].mul(corg_mape))

# Define a lambda function to compute the weighted mean:
integral = lambda x: np.sum(x)*5 #mean
add = lambda x: np.sum(x) #mean
wm = lambda x: np.average(x, weights=glob_df.loc[x.index, 'area']) #weighted mean

##############################################################################
### CALCULATE THE GLOBAL WEEKLY DEPTH-RESOLVED DATA
### VALUES ARE AVERAGES WEIGHTED BY SURFACE AREA

### Apply log-transform for calculating geometric mean
for var in ['chla','chla_u','chla_l','cphy','cphy_l','cphy_u']:
    glob_df.loc[:,var] = glob_df.loc[:,var].add(1).apply(np.log)

# Add data from each basin by lat, week, and depth
glob_dclim = glob_df.groupby(['lat','woy','depth']).agg(area = ('area', 'sum'),
                                                        cphy = ('cphy', wm),
                                                        cphy_l = ('cphy_l', wm),
                                                        cphy_u = ('cphy_u', wm),
                                                        bbpphy = ('bbpphy', wm),
                                                        bbp470 = ('bbp470', wm),
                                                        chla = ('chla', wm),
                                                        chla_u = ('chla_u', wm),
                                                        chla_l = ('chla_l', wm),
                                                        mld = ('mld', wm),
                                                        z90 = ('z90', wm)).reset_index()    

### Reverse log-transform for calculating geometric mean
for var in ['chla','chla_u','chla_l','cphy','cphy_l','cphy_u']:
    glob_dclim.loc[:,var] = (np.e**(glob_dclim.loc[:,var])).sub(1)

### Calculate surface averages
glob_dclim.loc[:, 'schla'] = glob_dclim.loc[glob_dclim['depth']<glob_dclim['mld'], 'chla']
glob_dclim.loc[:, 'scphy'] = glob_dclim.loc[glob_dclim['depth']<glob_dclim['mld'], 'cphy']

###  CALCULATE THE GLOBAL WEEKLY INTEGRATED DATA
glob_woy_dclim = glob_dclim.groupby(['lat','woy']).agg(area=('area', 'first'),  
                                                       mld = ('mld', 'first'),
                                                       z90 = ('z90', 'first'),
                                                       cphy = ('cphy', integral),
                                                       cphy_u = ('cphy_u', integral),
                                                       cphy_l = ('cphy_l', integral),
                                                       chla = ('chla', integral),
                                                       chla_u = ('chla_u', integral),
                                                       chla_l = ('chla_l', integral),
                                                       schla = ('schla', 'mean'),
                                                       scphy = ('scphy', 'mean')).reset_index()

summer_solstice_week = (datetime.datetime(int(df_main.year.median()),6,21)).isocalendar().week + 0.5
winter_solstice_week = (datetime.datetime(int(df_main.year.median()),12,21)).isocalendar().week + 0.5

glob_woy_dclim.loc[(glob_woy_dclim['lat']>0), 'rwoy'] = glob_woy_dclim.loc[(glob_woy_dclim['lat']>0), 'woy'] - 25.5
glob_woy_dclim.loc[(glob_woy_dclim['woy']>winter_solstice_week)&(glob_woy_dclim['lat']>0), 'rwoy'] = glob_woy_dclim.loc[(glob_woy_dclim['woy']>winter_solstice_week)&(glob_woy_dclim['lat']>0), 'woy'] - 52 - 25.5

glob_woy_dclim.loc[(glob_woy_dclim['lat']<0), 'rwoy'] = glob_woy_dclim.loc[(glob_woy_dclim['lat']<0), 'woy'] + 0.5
glob_woy_dclim.loc[(glob_woy_dclim['woy']>summer_solstice_week)&(glob_woy_dclim['lat']<0), 'rwoy'] = glob_woy_dclim.loc[(glob_woy_dclim['woy']>summer_solstice_week)&(glob_woy_dclim['lat']<0), 'woy'] - winter_solstice_week


glob_woy_dclim.loc[(glob_woy_dclim['lat']>0), 'wwoy'] = glob_woy_dclim.loc[(glob_woy_dclim['lat']>0), 'woy'] - winter_solstice_week
glob_woy_dclim.loc[(glob_woy_dclim['woy']<summer_solstice_week)&(glob_woy_dclim['lat']>0), 'wwoy'] = glob_woy_dclim.loc[(glob_woy_dclim['woy']<summer_solstice_week)&(glob_woy_dclim['lat']>0), 'woy'] + 0.5

glob_woy_dclim.loc[(glob_woy_dclim['lat']<0), 'wwoy'] = glob_woy_dclim.loc[(glob_woy_dclim['lat']<0), 'woy'] - 25.5
glob_woy_dclim.loc[(glob_woy_dclim['woy']>winter_solstice_week)&(glob_woy_dclim['lat']<0), 'wwoy'] = glob_woy_dclim.loc[(glob_woy_dclim['woy']>winter_solstice_week)&(glob_woy_dclim['lat']<0), 'woy'] -winter_solstice_week-26


### Apply log-transform for calculating geometric mean
for var in ['chla','chla_l','chla_u','cphy','cphy_l','cphy_u']:
    glob_dclim.loc[:,var] = glob_dclim.loc[:,var].add(1).apply(np.log)
    
###  CALCULATE THE GLOBAL ANNUAL MEAN DEPTH-RESOLVED DATA
glob_ann_dclim = glob_dclim.groupby(['lat','depth']).agg(area = ('area', 'mean'),
                                                         mld = ('mld', 'mean'),
                                                         z90 = ('z90', 'mean'),
                                                         cphy = ('cphy', 'mean'),
                                                         cphy_u = ('cphy_u', 'mean'),
                                                         cphy_l = ('cphy_l', 'mean'),
                                                         bbpphy = ('bbpphy', 'mean'),
                                                         bbp470 = ('bbp470', 'mean'),
                                                         chla = ('chla', 'mean'),
                                                         chla_u = ('chla_u', 'mean'),
                                                         chla_l = ('chla_l', 'mean')).reset_index()    

### Reverse log-transform for calculating geometric mean
for var in ['chla','chla_l','chla_u','cphy','cphy_l','cphy_u']:
    glob_ann_dclim.loc[:,var] = (np.e**(glob_ann_dclim.loc[:,var])).sub(1)
    
    
glob_ann_iclim = glob_ann_dclim.groupby(['lat',]).agg(area=('area', 'first'), 
                                                      mld = ('mld', 'first'),
                                                      z90 = ('z90', 'first'),
                                                      cphy = ('cphy', integral),
                                                      cphy_u = ('cphy_u', integral),
                                                      cphy_l = ('cphy_l', integral),
                                                      chla = ('chla', integral),
                                                      chla_u = ('chla_u', integral),
                                                      chla_l = ('chla_l', integral)).reset_index()

glob_ann_iclim['cphy_stock'] = glob_ann_iclim.cphy.mul(glob_ann_iclim.area)
glob_ann_iclim['cphy_l_stock'] = glob_ann_iclim.cphy_l.mul(glob_ann_iclim.area)
glob_ann_iclim['cphy_u_stock'] = glob_ann_iclim.cphy_u.mul(glob_ann_iclim.area)

glob_ann_iclim['chla_stock'] = glob_ann_iclim.chla.mul(glob_ann_iclim.area)
glob_ann_iclim['chla_l_stock'] = glob_ann_iclim.chla_l.mul(glob_ann_iclim.area)
glob_ann_iclim['chla_u_stock'] = glob_ann_iclim.chla_u.mul(glob_ann_iclim.area)

glob_ann_dclim['cphy_stock'] = glob_ann_dclim.cphy.mul(glob_ann_dclim.area).mul(5)
glob_ann_dclim['cphy_l_stock'] = glob_ann_dclim.cphy_l.mul(glob_ann_dclim.area).mul(5)
glob_ann_dclim['cphy_u_stock'] = glob_ann_dclim.cphy_u.mul(glob_ann_dclim.area).mul(5)

glob_ann_dclim['chla_stock'] = glob_ann_dclim.chla.mul(glob_ann_dclim.area).mul(5)
glob_ann_dclim['chla_l_stock'] = glob_ann_dclim.chla_l.mul(glob_ann_dclim.area).mul(5)
glob_ann_dclim['chla_u_stock'] = glob_ann_dclim.chla_u.mul(glob_ann_dclim.area).mul(5)

'''
np.round(glob_ann_iclim['chla_stock'].sum()*1e-15,1)
np.round(glob_ann_iclim['chla_l_stock'].sum()*1e-15,1)
np.round(glob_ann_iclim['chla_u_stock'].sum()*1e-15,1)

np.round(glob_ann_iclim['cphy_stock'].sum()*1e-15,0)
np.round(glob_ann_iclim['cphy_u_stock'].sum()*1e-15,0)
np.round(glob_ann_iclim['cphy_l_stock'].sum()*1e-15,0)
'''

### CALCULATE THE BASIN SPECIFIC ANNUAL DEPTH-RESOLVED DATA
### VALUES ARE AVERAGES WEIGHTED BY SURFACE AREA
### Calculate surface averages
bas_ann_dclim = glob_df.groupby(['basin','lat','depth']).agg(area=('area', 'first'),  
                                                             mld = ('mld', 'first'),
                                                             z90 = ('z90', 'first'),
                                                             cphy = ('cphy', 'mean'),
                                                             cphy_u = ('cphy_u', 'mean'),
                                                             cphy_l = ('cphy_l', 'mean'),
                                                             chla = ('chla', 'mean'),
                                                             chla_u = ('chla_u', 'mean'),
                                                             chla_l = ('chla_l', 'mean')).reset_index()

### Reverse log-transform for calculating geometric mean
for var in ['chla','chla_u','chla_l','cphy']:
    bas_ann_dclim.loc[:,var] = (np.e**(bas_ann_dclim.loc[:,var])).sub(1)
    
for basin in ['pacific','atlantic','indian']:
    print(basin)
    basin_stock = (bas_ann_dclim[bas_ann_dclim.basin==basin].cphy.mul(bas_ann_dclim[bas_ann_dclim.basin==basin].area).sum()*1e-15*5)
    global_stock = (bas_ann_dclim.cphy.mul(bas_ann_dclim.area).sum()*1e-15*5)
    print(np.round(basin_stock/global_stock,2))

##############################################################################
### Bloom Metric calculations/Satellite Data Processing
##############################################################################

chl_sat_df_merged = pd.read_csv(dep_dir + '/02 processed data/Processed Satellite Climatology/chl_sat_df_merged.csv')
# Average 
chla_sat_mean = chl_sat_df_merged[['lat','chlor_a','woy']].groupby(['woy','lat']).mean().reset_index()

# Recenter data around hemisphere's solstice
chla_sat_mean.loc[(chla_sat_mean['lat']>0), 'rwoy'] = chla_sat_mean.loc[(chla_sat_mean['lat']>0), 'woy'] - 25.5
chla_sat_mean.loc[(chla_sat_mean['woy']>winter_solstice_week)&(chla_sat_mean['lat']>0), 'rwoy'] = chla_sat_mean.loc[(chla_sat_mean['woy']>winter_solstice_week)&(chla_sat_mean['lat']>0), 'woy'] - 52 - 25.5

chla_sat_mean.loc[(chla_sat_mean['lat']<0), 'rwoy'] = chla_sat_mean.loc[(chla_sat_mean['lat']<0), 'woy'] + 0.5
chla_sat_mean.loc[(chla_sat_mean['woy']>summer_solstice_week)&(chla_sat_mean['lat']<0), 'rwoy'] = chla_sat_mean.loc[(chla_sat_mean['woy']>summer_solstice_week)&(chla_sat_mean['lat']<0), 'woy'] - winter_solstice_week

chla_sat_mean = chla_sat_mean[chla_sat_mean.woy !=53].copy()

win= 3
for lat in chla_sat_mean.lat.unique():    
    ireg_clim = chla_sat_mean[chla_sat_mean['lat']==lat].copy()
    ireg_clim = pd.concat([ireg_clim, ireg_clim, ireg_clim]).reset_index(drop= True).reset_index()
    
    ireg_clim.loc[ireg_clim.lat==lat,'chlor_a'] = ireg_clim.loc[ireg_clim.lat==lat,'chlor_a'].interpolate(method = 'linear',limit = 2).rolling(window = win, center = True, min_periods = 1, closed = 'right')\
        .median().rolling(window = win, center=True, min_periods = 1, closed = 'right').mean()

    ireg_clim = ireg_clim[(ireg_clim['index']>=52)&(ireg_clim['index']<=103)].reset_index()
    del ireg_clim['index'], ireg_clim['level_0']
    
    chla_sat_mean.loc[chla_sat_mean.lat==lat,'chlor_a_zs'] = ((ireg_clim.loc[ireg_clim.lat==lat,'chlor_a']-ireg_clim.loc[ireg_clim.lat==lat,'chlor_a'].mean())\
                                                                                     /ireg_clim.loc[ireg_clim.lat==lat,'chlor_a'].std()).values
    chla_sat_mean.loc[chla_sat_mean.lat==lat,'chlor_a'] = ireg_clim.loc[ireg_clim.lat==lat,'chlor_a'].values

ireg_df = []
# Take the WEEKLY climatogies metrics and assign to annual average
for lat in glob_woy_dclim['lat'].unique(): 
    
    ireg_clim = glob_woy_dclim[glob_woy_dclim['lat']==lat].copy()
    ireg_pro = glob_ann_dclim[glob_ann_dclim['lat']==lat].copy().reset_index()
    dcm = ireg_pro[ireg_pro.index==ireg_pro.chla.argmax()].depth.mean()
    dbm = ireg_pro[ireg_pro.index==ireg_pro.cphy.argmax()].depth.mean()

    ireg_clim = pd.concat([ireg_clim, ireg_clim, ireg_clim]).reset_index(drop= True).reset_index()

    ireg_clim.loc[:,'cphy_r'] = np.log(ireg_clim.loc[:,'cphy'].div(ireg_clim.loc[:,'cphy'].shift(1))).div(7)
    ireg_clim.loc[:,'chla_r'] = np.log(ireg_clim.loc[:,'schla'].div(ireg_clim.loc[:,'schla'].shift(1))).div(7)
    
    win = 6
    for var in ['schla','cphy']:
        ireg_clim.loc[:,var] = ireg_clim.loc[:,var].rolling(window = win, center = True, min_periods = 1, closed = 'right')\
            .median().rolling(window = win, center=True, min_periods = 1, closed = 'right').mean()
    
    ireg_clim['cphy_r'] = np.log(ireg_clim['cphy']/ireg_clim['cphy'].shift(1))
    ireg_clim['schla_r'] = np.log(ireg_clim['schla']/ireg_clim['schla'].shift(1))
    
    '''
    win = 6
    for var in ['schla_r','cphy_r']:
        ireg_clim.loc[:,var] = ireg_clim.loc[:,var].rolling(window = win, center = True, min_periods = 1, closed = 'right')\
            .median().rolling(window = win, center=True, min_periods = 1, closed = 'right').mean()
    '''
    ireg_clim = ireg_clim[(ireg_clim['index']>=52)&(ireg_clim['index']<=103)].reset_index()
    del ireg_clim['index'], ireg_clim['level_0']
    
    sat_ireg_clim = chla_sat_mean.loc[chla_sat_mean['lat']==lat].sort_values(by='woy')

    # Obtain z-scores
    ireg_clim['cphy_zs'] = (ireg_clim['cphy'] - ireg_clim['cphy'].mean()) / (ireg_clim['cphy'].std())
    ireg_clim['schla_zs'] = (ireg_clim['schla'] - ireg_clim['schla'].mean()) / (ireg_clim['schla'].std())
    
    ireg_clim['sat_chla_zs'] = sat_ireg_clim['chlor_a_zs'].values
    ireg_clim['sat_chla'] = sat_ireg_clim['chlor_a'].values
    ireg_clim.loc[:,'satchla_r'] = np.log(ireg_clim.loc[:,'sat_chla'].div(ireg_clim.loc[:,'sat_chla'].shift(1))).div(7)

    ireg_df.append(ireg_clim)
    
    '''
    Metrics for Main Analysis
    '''
    # TIMING OF PEAK BLOOM
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'peak_cphy'] = ireg_clim.loc[ireg_clim['cphy'] == ireg_clim['cphy'].max(),'rwoy'].values[0]
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'peak_schla'] = ireg_clim.loc[ireg_clim['schla'] == ireg_clim['schla'].max(),'rwoy'].values[0]
    if len(ireg_clim.sat_chla.dropna())>0:
        glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'chla_sat_peak'] = ireg_clim.loc[ireg_clim['sat_chla'] == ireg_clim['sat_chla'].max(),'rwoy'].values[0]

    #
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'schla_dur'] =  ireg_clim[ireg_clim['schla_r']>0].woy.nunique()
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'cphy_dur'] =  ireg_clim[ireg_clim['cphy_r']>0].woy.nunique()
    
    if len(ireg_clim.sat_chla_zs.dropna())>=47:
        glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'chla_sat_dur'] = ireg_clim[ireg_clim['satchla_r']>0].woy.nunique()

    # SEASONAL RANGE (NORMALIZED)
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'schla_cv'] = (ireg_clim['schla'].max()-ireg_clim['schla'].min())/ireg_clim['schla'].mean()
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'cphy_cv'] = (ireg_clim['cphy'].max()-ireg_clim['cphy'].min())/ireg_clim['cphy'].mean()
    if len(ireg_clim.sat_chla_zs.dropna())>=47:
        glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'chla_sat_cv'] = (ireg_clim['sat_chla'].max()-ireg_clim['sat_chla'].min())/ireg_clim['sat_chla'].mean()
    
    '''
    Metrics for Supplementary Analysis
    '''
    
    # TIMING OF R MAXIMUM
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'schla_r_max'] = ireg_clim.loc[ireg_clim['schla_r'] == ireg_clim['schla_r'].max(),'wwoy'].values[0]
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'cphy_r_max'] = ireg_clim.loc[ireg_clim['cphy_r'] == ireg_clim['cphy_r'].max(),'wwoy'].values[0]
    
    # TIMING OF R MINIMUM
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'schla_r_min'] = ireg_clim.loc[ireg_clim['schla_r'] == ireg_clim['schla_r'].min(),'rwoy'].values[0]
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'cphy_r_min'] = ireg_clim.loc[ireg_clim['cphy_r'] == ireg_clim['cphy_r'].min(),'rwoy'].values[0]
    
    # TIMING OF SEASONAL MINIMUM
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'min_schla'] = ireg_clim.loc[ireg_clim['schla'] == ireg_clim['schla'].min(),'wwoy'].values[0]
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'min_cphy'] = ireg_clim.loc[ireg_clim['cphy'] == ireg_clim['cphy'].min(),'wwoy'].values[0]
    
    # CORRELATION COEFFICIENT
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'satchla_r2'] = np.nan
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'chla_r2'] = ireg_clim[['cphy','schla']].apply(np.log).corr().values[1][0]
    if len(ireg_clim.sat_chla_zs.dropna())>=47:
        glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'satchla_r2'] = ireg_clim[['sat_chla','schla']].apply(np.log).corr().values[1][0] #np.median(ireg_clim['sat_chla']/ireg_clim['schla'])
        
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'dcm'] = dcm
    glob_ann_iclim.loc[glob_ann_iclim['lat']==lat,'dbm'] = dbm

ireg_df = pd.concat(ireg_df)
glob_ann_iclim['schla_dur'] = glob_ann_iclim['schla_dur'].rolling(window = 3, center= True, min_periods = 1).mean().round(0).values
glob_ann_iclim['cphy_dur'] = glob_ann_iclim['cphy_dur'].rolling(window = 3, center= True, min_periods = 1).mean().round(0).values


'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN MANUSCRIPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

###################################################
# Results in the Manuscript Text
###################################################
# Horizontal and Vertical Distributions of Chla and Cphyto

global_corg_stock = glob_ann_iclim.cphy_stock.sum()*1e-15
global_corg_l_stock = glob_ann_iclim.cphy_l_stock.sum()*1e-15
global_corg_u_stock = glob_ann_iclim.cphy_u_stock.sum()*1e-15
    
avg_cphy_integral = glob_ann_iclim.cphy_stock.sum()/glob_ann_iclim.area.sum()
avg_cphy_u_integral = glob_ann_iclim.cphy_u_stock.sum()/glob_ann_iclim.area.sum()
avg_cphy_l_integral = glob_ann_iclim.cphy_l_stock.sum()/glob_ann_iclim.area.sum()

print(f'The global stock of phytoplankton carbon equals {global_corg_stock:.0f} Tg ({global_corg_l_stock:.0f} - {global_corg_u_stock:.0f} Tg). ' +\
      f'The average depth-integral of carbon equals {avg_cphy_integral:.0f} mg/m^2 ({avg_cphy_l_integral:.0f} - {avg_cphy_u_integral:.0f} mg/m^2)\n')

global_chla_stock = glob_ann_iclim.chla_stock.sum()*1e-15
global_chla_l_stock = glob_ann_iclim.chla_l_stock.sum()*1e-15
global_chla_u_stock = glob_ann_iclim.chla_u_stock.sum()*1e-15
    
avg_chla_integral = glob_ann_iclim.chla_stock.sum()/glob_ann_iclim.area.sum()
avg_chla_l_integral = glob_ann_iclim.chla_l_stock.sum()/glob_ann_iclim.area.sum()
avg_chl_u_integral = glob_ann_iclim.chla_u_stock.sum()/glob_ann_iclim.area.sum()

print(f'The global stock of phytoplankton chlorophyll-a equals {global_chla_stock:.1f} Tg ({global_chla_l_stock:.1f} - {global_chla_u_stock:.1f} Tg). ' +\
      f'The average depth-integral of chlorophyll-a equals {avg_chla_integral:.1f} mg/m^2 ({avg_chla_l_integral:.1f} - {avg_chl_u_integral:.1f} mg/m^2)\n')

sh_per_vol = glob_ann_iclim[glob_ann_iclim['lat']<-0]['area'].sum()/glob_ann_iclim['area'].sum()*100
sh_per_chla = glob_ann_iclim[glob_ann_iclim['lat']<-0]['chla_stock'].sum()/glob_ann_iclim['chla_stock'].sum()*100
sh_per_cphy = glob_ann_iclim[glob_ann_iclim['lat']<-0]['cphy_stock'].sum()/glob_ann_iclim['cphy_stock'].sum()*100

print(f'The Southern Hemisphere ({sh_per_vol:.0f}% of ocean area) ' + 
      f'contains {sh_per_chla:.0f}% of global phytoplankton chlorophyll-a and {sh_per_cphy:.0f}% of global phytoplankton carbon.\n')

so30_per_vol = glob_ann_iclim[glob_ann_iclim['lat']<-30]['area'].sum()/glob_ann_iclim['area'].sum()*100
so30_per_chla = glob_ann_iclim[glob_ann_iclim['lat']<-30]['chla_stock'].sum()/glob_ann_iclim['chla_stock'].sum()*100
so30_per_cphy = glob_ann_iclim[glob_ann_iclim['lat']<-30]['cphy_stock'].sum()/glob_ann_iclim['cphy_stock'].sum()*100

print(f'The Southern Ocean and its subtropical boundaries ({so30_per_vol:.0f}% of ocean area) ' + 
      f'contains {so30_per_chla:.0f}% of global phytoplankton chlorophyll-a and {so30_per_cphy:.0f}% of global phytoplankton carbon.\n')
    
cphy_below_mld = glob_ann_dclim[glob_ann_dclim['depth']>glob_ann_dclim['mld']]['cphy_stock'].sum()/glob_ann_dclim['cphy_stock'].sum()*100
chla_below_mld = glob_ann_dclim[glob_ann_dclim['depth']>glob_ann_dclim['mld']]['chla_stock'].sum()/glob_ann_dclim['chla_stock'].sum()*100

print(f'The subsurface layer (below the mixed layer) contains {cphy_below_mld:.0f}% of phytoplankton carbon and '+\
      f'{chla_below_mld:.0f}% of phytoplankton chlorophyll-a\n')

cphy_below_z90 = glob_ann_dclim[glob_ann_dclim['depth']>glob_ann_dclim['z90']]['cphy_stock'].sum()/glob_ann_dclim['cphy_stock'].sum()*100
chla_below_z90 = glob_ann_dclim[glob_ann_dclim['depth']>glob_ann_dclim['z90']]['chla_stock'].sum()/glob_ann_dclim['chla_stock'].sum()*100

print(f'The subsurface layer (below the mixed layer) contains {cphy_below_z90:.0f}% of phytoplankton carbon and '+\
      f'{chla_below_z90:.0f}% of phytoplankton chlorophyll-a\n')
    
dcm_offset = glob_ann_iclim[glob_ann_iclim['dbm'].sub(glob_ann_iclim['dcm']).abs()>10].area.sum()/glob_ann_iclim.area.sum()*100
print(f'The DCM is offset by the DBM > 10 m in  {dcm_offset:.0f}% of the ocean\n')
    
cphy_below_300 = glob_ann_dclim[glob_ann_dclim['depth']>300]['cphy_stock'].sum()/glob_ann_dclim['cphy_stock'].sum()*100
chla_below_300 = glob_ann_dclim[glob_ann_dclim['depth']>300]['chla_stock'].sum()/glob_ann_dclim['chla_stock'].sum()*100

print(f'{cphy_below_300:.1f}% and {chla_below_300:.1f}% of global carbon and chlorophyll-a stocks are present below 300 m depth.\n')

# Seasonal Distributions of Chla and Cphyto
# Bloom peak results
peak_bloom_dif = abs(np.round(np.average(glob_ann_iclim['peak_schla'].sub(glob_ann_iclim['peak_cphy']), weights = glob_ann_iclim.area.values),0))

print(f'For the global ocean, we found that the average timing of the bloom peak, ' + \
      f'weighted by surface area, is ~{peak_bloom_dif:.0f} weeks after surface Chla has reached its' +  \
      f'annual maximum\n')

peak_bloom_dif_30 = abs(np.round(np.average(glob_ann_iclim[glob_ann_iclim['lat'].abs()<30]['peak_schla'].sub(glob_ann_iclim[glob_ann_iclim['lat'].abs()<30]['peak_cphy']).abs(), 
                                         weights = glob_ann_iclim[glob_ann_iclim['lat'].abs()<30]['area'].values),0))

print(f'In the tropics (<30deg latitude), this difference increases to ~{peak_bloom_dif_30:.0f} weeks on average.\n')


global_per_dis = np.round(glob_ann_iclim[glob_ann_iclim['peak_schla'].sub(glob_ann_iclim['peak_cphy']).abs()>4]['area'].sum()/glob_ann_iclim['area'].sum()*100,0)

global_per_dis_stock = np.round(glob_ann_iclim[glob_ann_iclim['peak_schla'].sub(glob_ann_iclim['peak_cphy']).abs()>4]['cphy_stock'].sum()/glob_ann_iclim['cphy_stock'].sum()*100,0)
 
print(f'In about {global_per_dis:.0f}% of the ocean, the timing of the peak phytoplankton bloom is' + \
      f'more than 4 weeks off when surface Chla reaches its annual peak. The area' + \
      f'where this discrepancy occurs, mainly equatorial and temperate latitudes, ' + \
      f'contains close to half of global phytoplankton biomass ({global_per_dis_stock:.0f}%), meaning that blooms ' + \
      f'are misidentified by surface Chla for about half of Earth’s phytoplankton.')

# Bloom duration results
dur_dif = np.round(np.average(glob_ann_iclim['schla_dur'].sub(glob_ann_iclim['cphy_dur']), 
                     weights = glob_ann_iclim['area'].values),0)

schla_dur_max = glob_ann_iclim['schla_dur'].max()
schla_dur_min = glob_ann_iclim['schla_dur'].min()
cphy_dur_max = glob_ann_iclim['cphy_dur'].max()
cphy_dur_min = glob_ann_iclim['cphy_dur'].min()

print(f'The blooming period, defined as the number of weeks where the rate of change in' + \
      f'intCorg is greater than 0 d-1, follows a sinusoidal pattern ranging from ' + \
          f'approximately {cphy_dur_min:.0f} to {cphy_dur_max:.0f} weeks, with the half-period centered near the equator ' + \
              f'(Figure 3D). If the metric of surface Chla is used in place of intCorg, the ' + \
                  f'latitudinal pattern in the blooming period is substantially different. ' + \
                      f'On average, the blooming period based on surface Chla appears ~{dur_dif:.0f} weeks ' + \
                          f'longer ranging from {schla_dur_min:.0f} to {schla_dur_max:.0f} weeks.')

dur_dif_25N = glob_ann_iclim[glob_ann_iclim['lat']==25]['schla_dur'].sub(glob_ann_iclim[glob_ann_iclim['lat']==25]['cphy_dur']).values[0]
print(f'For example, blooming appears to last ~{dur_dif_25N:.0f} weeks longer based on surface Chla than ΣCorg in the region 20-30N')

# Polar differences in surface Chla and depth-integrated carbon
dur_dif_SO_50S = glob_ann_iclim[glob_ann_iclim['lat']<-50]['schla_dur'].sub(glob_ann_iclim[glob_ann_iclim['lat']<-50]['cphy_dur'])
dur_dif_AO_50N = glob_ann_iclim[glob_ann_iclim['lat']>50]['schla_dur'].sub(glob_ann_iclim[glob_ann_iclim['lat']>50]['cphy_dur'])

# Bloom amplitude results
# roughly equal to 2
bloom_amp_avg = np.average(glob_ann_iclim.schla_cv.div(glob_ann_iclim.cphy_cv), weights = glob_ann_iclim.area.values)

# Seasonal co-variability results
var_b05 = np.round(glob_ann_iclim[glob_ann_iclim['chla_r2']<0.5]['area'].sum()/glob_ann_iclim['area'].sum(),2)*100
var_b0 = np.round(glob_ann_iclim[glob_ann_iclim['chla_r2']<0]['area'].sum()/glob_ann_iclim['area'].sum(),2)*100

print(f'These discrepancies are well exemplified by the fact that the Pearson correlation coefficient between surface Chla' + \
      f' and depth-integrated ΣCphy is less than 0.5 in ~{var_b05:.0f}% of the ocean and less than 0 for ~{var_b0:.0f}%% of' + \
      f'the ocean (by surface area; SI Appendix, Fig. S6 )')

###################################################
# Fig. 1: Map of Float Profiles and Basin Divisions
###################################################
atl_bar = df_main[df_main['basin']=='atlantic'][['id','lat_bin']].groupby('id').first().reset_index().groupby('lat_bin').count().id.reset_index().fillna(0)
pac_bar = df_main[df_main['basin']=='pacific'][['id','lat_bin']].groupby('id').first().reset_index().groupby('lat_bin').count().id.reset_index().fillna(0)
ind_bar = df_main[df_main['basin']=='indian'][['id','lat_bin']].groupby('id').first().reset_index().groupby('lat_bin').count().id.reset_index().fillna(0)

prof_bar = atl_bar.merge(pac_bar, on = 'lat_bin', how = 'left').merge(ind_bar, on = 'lat_bin', how = 'left').fillna(0)
argo_img = dep_dir + '/03 figures/assets/argo_icon.jpg'
im = plt.imread(argo_img)

float_profile_n = df_main.id.nunique()

array = [[1,0],
         [1,2],
         [1,2],
         [1,0],
         [1,0],
         [1,0]]
fig, ax = pplt.subplots(array, width = 8.5, proj = {1:'moll'}, proj_kw={'central_longitude': -60},wratios = (5,2), hratios = (2,4,4,2,2,2),
                        abc = 'A', wspace = 7)
ax[0].format(grid=False)
ax[0].set_global()
ax[0].set_facecolor('gray6')

ax[0].add_feature(cfeature.LAKES, edgecolor='black', facecolor = 'gray6', zorder = 59, lw = 0.5)
ax[0].add_geometries(land_plot.geometry.values, crs=ccrs.PlateCarree(), facecolor = 'gray4', 
                     edgecolor='black', zorder = 35, lw = 0.5, label = 'Land')

ax[0].add_geometries(glob_ocean[glob_ocean['basin'] == 'pacific'].geometry.values, crs=ccrs.PlateCarree(), facecolor = 'blue1', 
                     edgecolor='blue1', zorder = 10, lw = 1, label = 'Pacific Ocean')
ax[0].add_geometries(glob_ocean[glob_ocean['basin'] == 'atlantic'].geometry.values, crs=ccrs.PlateCarree(), facecolor = 'blue2', 
                     edgecolor='blue2', zorder = 10, lw = 1, label = 'Atlantic Ocean')
ax[0].add_geometries(glob_ocean[glob_ocean['basin'] == 'indian'].geometry.values, crs=ccrs.PlateCarree(), facecolor = 'blue3', 
                     edgecolor='blue3', zorder = 10, lw = 1, label = 'Indian Ocean')

legend_elements = [Patch(facecolor='blue3', edgecolor='blue3', label='Indian Ocean'),
                   Patch(facecolor='blue2', edgecolor='blue2', label='Atlantic Ocean'),
                   Patch(facecolor='blue1', edgecolor='blue1', label='Pacific Ocean')]
ax[0].legend(handles=legend_elements, ncol = 1, bbox_to_anchor = [1.55, 0.15], fontsize = 12, frameon = False)

ax[0].plot(np.repeat(146.916714, 50), np.linspace(-90, -35, 50), color = 'black', 
        lw = 2, transform = ccrs.PlateCarree(), zorder = 28, label = '')
ax[0].plot(np.repeat(-67.25, 50), np.linspace(-65, -55, 50), color = 'black', 
        lw = 2, transform = ccrs.PlateCarree(), zorder = 28, label = '')
ax[0].plot(np.repeat(20, 50), np.linspace(-90, -35, 50), color = 'black', 
        lw = 2, transform = ccrs.PlateCarree(), zorder = 28, label = '')
ax[0].scatter(df_profiles.longitude,df_profiles.latitude, s = 0.1, zorder = 25, marker = "o",
              color = 'black', alpha = 1, transform = ccrs.PlateCarree(), label = '')

newax = fig.add_axes([0.4, 0.04, 0.45, 0.3], xmargin = 0, ymargin = 0)
newax.imshow(im)
newax.set_xlim(-2000,524)
newax.scatter([-1550],[800], color = 'black', s = 70)
newax.text(-1250,800,'Float Profiles\n(n = ' + str(float_profile_n) + ')', ha = 'left', va = 'center')
newax.axis('off')

ax2 = ax[1]
ax2.spines[['right', 'top']].set_visible(False)
ax2.grid(False)
b1 = ax2.bar(prof_bar['lat_bin'],prof_bar['id_y'], color = 'blue1', edgecolor = 'none', width = 0.9)
b2 = ax2.bar(prof_bar['lat_bin'],prof_bar['id_x'], color = 'blue2', edgecolor = 'none', width = 0.9, bottom = prof_bar['id_y'])
ax2.bar(prof_bar['lat_bin'],prof_bar['id'], color = 'blue3', edgecolor = 'none', width = 0.9, 
        bottom = prof_bar['id_y'].add(prof_bar['id_x']))

ax2.set_ylabel('Number of Profiles')
ax2.set_xlabel('Latitude')
ax2.set_ylim(0,12000)
ax2.set_yticks(np.arange(0,16000,4000))
ax2.xaxis.set_major_formatter(lat_formatter)
ax2.set_xticks(np.arange(-90 ,100,30))
ax2.xaxis.set_minor_locator(MultipleLocator(10))

#plt.savefig(dep_dir + '/03 figures/publish quality/fig 1.jpg', dpi = 300, tight_layout = True)
plt.show()

###################################################
# Fig. 2: Map of Float Profiles and Basin Divisions
###################################################
array = [[1,2],[3,4]]
fig, axes = pplt.subplots(array, width = 7, height = 4.5, sharey = False, sharex = True, spanx = False,
                          xlabel = 'Latitude', ylabel = 'Depth (m)', abc = 'A',
                          grid = False, xlim = (-90,90), hratios = (2,3))
[ax.xaxis.set_major_formatter(lat_formatter) for ax in axes]
[ax.set_xticks(np.arange(-90,100,30)) for ax in axes]
[ax.xaxis.set_minor_locator(MultipleLocator(10)) for ax in axes]
    
    
axes[0].bar(glob_ann_iclim['lat'], glob_ann_iclim['cphy_stock'].mul(1e-15), color = 'brown', 
            zorder = 25, width = 0.9, lw = 0, edgecolor = 'none', alpha = 1,
            yerr = [glob_ann_iclim['cphy_stock'].sub(glob_ann_iclim['cphy_l_stock']).mul(1e-15).abs(),
                    glob_ann_iclim['cphy_stock'].sub(glob_ann_iclim['cphy_u_stock']).mul(1e-15).abs()], error_kw = {'capsize': 1.5, 'elinewidth': 0.5, 'color': 'midnight'})
axes[0].set_ylim(0,90)
axes[0].set_yticks(np.arange(0,110,20))
axes[0].set_ylabel('C$_{phy}$ Stock (Tg)', labelpad = 8)

c_stock = glob_ann_iclim['cphy_stock'].mul(1e-15).sum()
c_stock_err = glob_ann_iclim['cphy_stock'].sub(glob_ann_iclim['cphy_l_stock']).mul(1e-15).abs().sum()
axes[0].annotate('Global C$_{phy}$ = ' + str(int(np.round(c_stock,0))) + ' ± ' + str(int(np.round(c_stock_err,0))) + ' Tg', 
                 xy=(0.95, 0.92), xycoords='axes fraction', fontsize=10, ha='right', va='top')

axes[1].bar(glob_ann_iclim['lat'], glob_ann_iclim['chla_stock'].mul(1e-15), color = 'green9', 
            zorder = 25, width = 0.9, lw = 0, edgecolor = 'none', alpha = 1,
            yerr = [glob_ann_iclim['chla_stock'].sub(glob_ann_iclim['chla_l_stock']).mul(1e-15).abs(),
                    glob_ann_iclim['chla_stock'].sub(glob_ann_iclim['chla_u_stock']).mul(1e-15).abs()], error_kw = {'capsize': 1.5, 'elinewidth': 0.5, 'color': 'midnight'})
axes[1].set_ylim(0,3)
axes[1].set_yticks(np.arange(0,4,1))
axes[1].set_ylabel('Chla Stock (Tg)', labelpad = 8)

chla_stock = glob_ann_iclim['chla_stock'].mul(1e-15).sum()
chla_stock_err = glob_ann_iclim['chla_stock'].sub(glob_ann_iclim['chla_u_stock']).mul(1e-15).abs().sum()

axes[1].annotate('Global Chla = ' + str((np.round(chla_stock,1))) + ' ± ' + str((np.round(chla_stock_err,1))) + ' Tg', 
                 xy=(0.95, 0.92), xycoords='axes fraction', fontsize=10, ha='right', va='top')

axes[2:4].set_ylim(300,0)
axes[2:4].set_yticks([300,200,100,0])
cf = axes[2].contourf(glob_ann_dclim.pivot(columns = 'lat',index = 'depth', values = 'cphy_stock')/1e15, 
                      cmap='Browns1', levels = np.linspace(0,2.5,21))
divider = make_axes_locatable(axes[2])
cax = divider.append_axes('bottom', size='15%', pad=0.55)
fig.colorbar(cf, cax = cax, orientation='horizontal', extend = 'both', ticks = np.arange(0,3,0.5),
             label='C$_{phy}$ Stock (Tg)')
axes[2].plot(glob_ann_iclim['lat'],glob_ann_iclim['mld'], color = 'black', lw = 1)
axes[2].plot(glob_ann_iclim['lat'],glob_ann_iclim['z90'], color = 'black', lw = 1, ls = 'dashed')

cf = axes[3].contourf(glob_ann_dclim.pivot(columns = 'lat',index = 'depth', values = 'chla_stock')/1e12, 
                      cmap='Greens1', levels = np.linspace(0,80,21))
divider = make_axes_locatable(axes[3])
cax = divider.append_axes('bottom', size='15%', pad=0.55)
fig.colorbar(cf, cax = cax, orientation='horizontal', extend = 'both', ticks = np.arange(0,90,10),
             label='Chla Stock (Gg)')
axes[3].plot(glob_ann_iclim['lat'],glob_ann_iclim['mld'], color = 'black', lw = 1)
axes[3].plot(glob_ann_iclim['lat'],glob_ann_iclim['z90'], color = 'black', lw = 1, ls = 'dashed')

#plt.savefig(dep_dir + '/03 figures/publish quality/fig 2.jpg', dpi = 300)
plt.show()

###################################################
# Fig. 3: BLOOM TIMING, DURATION, AND AMPLITUDE
###################################################

array = [[1,1,1,2,2,2],
         [3,3,4,4,5,5]]
fig, axes = pplt.subplots(array, width = 5, height = 6, sharey = False, sharex = True, spanx = False,
                          grid = False, hratios = (3,3), abc = 'A', wspace = 3, ylim = (-90,90), hspace = 5)
[ax.yaxis.set_major_formatter(lat_formatter) for ax in axes]
[ax.set_yticks(np.arange(-90,100,30)) for ax in axes]
axes[2].yaxis.set_minor_locator(MultipleLocator(10))
    

splines = 15 # number of splines we will use
im = axes[0].pcolor(ireg_df.pivot(index = 'lat', columns = 'rwoy', values = 'schla_zs'),
                    cmap = 'Greens1', vmax = 1.5, vmin = -1, levels = 100,)
divider = make_axes_locatable(axes[0])
cax = divider.append_axes('bottom', size='5%', pad=0.3)
fig.colorbar(im, cax = cax, orientation='horizontal', extend = 'both', ticks = np.arange(-1,2,0.5),
             label='Surface Chla (z-score)')
axes[0].set_xlim(-26.5,26.5)
axes[0].set_xticks(np.arange(-26,26+13,13))
axes[0].axvline(0, color = 'black', lw = 1, zorder = 1, ls = 'dashed')
axes[0].set_ylabel('Latitude', labelpad = 8)
axes[0].set_xlabel('')

im = axes[1].pcolor(ireg_df.pivot(index = 'lat', columns = 'rwoy', 
                         values = 'cphy_zs'), 
               cmap = 'Browns1', vmax = 1.5, vmin = -1, levels = 100)
divider = make_axes_locatable(axes[1])
cax = divider.append_axes('bottom', size='5%', pad=0.3)
fig.colorbar(im, cax = cax, orientation='horizontal', extend = 'both', ticks = np.arange(-1,2,0.5),
             label='Depth-integrated C$_{phy}$ (z-score)')
axes[1].set_xlim(-26,26)
axes[1].set_xticks(np.arange(-26,26+13,13))
axes[1].axvline(0, color = 'black', lw = 1, zorder = 1, ls = 'dashed')
axes[1].set_ylabel('', labelpad = 8)
axes[1].set_yticklabels([])
axes[1].set_xlabel('')

X = glob_ann_iclim[['lat']].values
chla_var = ['peak_schla', 'schla_dur', 'schla_cv']
for ax_ind in [2,3,4]:
    y = glob_ann_iclim[chla_var[ax_ind-2]].values
    gam = GAM(n_splines = splines, max_iter = 1000).gridsearch(X, y)
    XX = gam.generate_X_grid(term = 0)
    axes[ax_ind].plot(gam.predict(XX), XX, color = 'green9', lw= 2, clip_on = False)
    axes[ax_ind].fill_betweenx(XX, gam.confidence_intervals(XX, width=.95)[:,1],
                                   gam.confidence_intervals(XX, width=.95)[:,0], XX, 
                                   color='green9', alpha = 0.3, clip_on = False)
    axes[ax_ind].scatter(y, X, color = 'green9', s= 15, clip_on = False)

cphy_var = ['peak_cphy', 'cphy_dur', 'cphy_cv']
for ax_ind in [2,3,4]:
    y = glob_ann_iclim[cphy_var[ax_ind-2]].values
    gam = GAM(n_splines = splines, max_iter = 1000).gridsearch(X, y)
    XX = gam.generate_X_grid(term = 0)
    axes[ax_ind].plot(gam.predict(XX), XX, color = 'brown', lw= 2, clip_on = False)
    axes[ax_ind].fill_betweenx(XX, gam.confidence_intervals(XX, width=.95)[:,1],
                                   gam.confidence_intervals(XX, width=.95)[:,0], XX, 
                                   color='brown', alpha = 0.3, clip_on = False)
    axes[ax_ind].scatter(y, X, color = 'brown', s= 15, clip_on = False)
    
axes[2].set_xlim(-30,30)
axes[2].set_xticks(np.arange(-30,30+15,15))
axes[2].set_xlabel('Peak Timing\n(Weeks from Midsummer)', labelpad = 8)
axes[2].set_ylabel('Latitude', labelpad = 8)
axes[2].spines[['top', 'right']].set_visible(False)
axes[2].axvline(0, color = 'black', lw = 1, zorder = 1, ls = 'dashed')

axes[3].set_xlim(15,35)
axes[3].set_xticks(np.arange(15,35+5,5))
axes[3].set_xlabel('Blooming Period \n(Weeks)', labelpad = 8)
axes[3].spines[['top', 'right', 'left']].set_visible(False)
axes[3].set_yticks([])
axes[3].set_ylabel('')

axes[4].set_xlim(0,3)
axes[4].set_xticks(np.arange(0, 3+1,1))
axes[4].set_xlabel('Bloom Amplitude\n(normalized to mean)', labelpad = 8)
axes[4].spines[['top', 'right', 'left']].set_visible(False)
axes[4].set_yticks([])
axes[4].set_ylabel('')

#plt.savefig(dep_dir + '/03 figures/publish quality/fig 3.jpg', dpi = 300)
plt.show()

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~ SUPPLEMENTARY INFORMATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

###################################################
# Fig. S5: CPHYTO AND CHLA (Z-INTRGRATED/VOLUMETRIC)
###################################################
array = [[1,2],[3,4]]
fig, axes = pplt.subplots(array, width = 6, height = 4, sharey = False, sharex = True, spanx = False,
                        xlabel = 'Latitude', abc = 'A',
                        grid = False, xlim = (-90,90), hratios = (2,3))
[ax.xaxis.set_major_formatter(lat_formatter) for ax in axes]
[ax.set_xticks(np.arange(-90,100,45)) for ax in axes]

axes[0].bar(glob_ann_iclim['lat'], glob_ann_iclim['cphy'], color = 'brown', 
            zorder = 25, width = 0.9, lw = 0, edgecolor = 'none', alpha = 1,
            yerr = [glob_ann_iclim['cphy'].sub(glob_ann_iclim['cphy_l']).abs(),
                    glob_ann_iclim['cphy'].sub(glob_ann_iclim['cphy_u']).abs()], 
            error_kw = {'capsize': 1.5, 'elinewidth': 0.5, 'color': 'midnight'})
axes[0].set_ylim(0,3000)
axes[0].set_yticks(np.linspace(0,3000,6))
axes[0].set_ylabel('\u03A3C$_{phy}$ (mg m$^{-2}$)', labelpad = 5)

axes[1].bar(glob_ann_iclim['lat'], glob_ann_iclim['chla'], color = 'green9', 
            zorder = 25, width = 0.9, lw = 0, edgecolor = 'none', alpha = 1,
            yerr = [glob_ann_iclim['chla'].sub(glob_ann_iclim['chla_l']).abs(),
                    glob_ann_iclim['chla'].sub(glob_ann_iclim['chla_u']).abs()], 
            error_kw = {'capsize': 1.5, 'elinewidth': 0.5, 'color': 'midnight'})
axes[1].set_ylim(0,100)
axes[1].set_yticks(np.linspace(0,100,6))
axes[1].set_ylabel('\u03A3Chla (mg m$^{-2}$)', labelpad = 5)

axes[2:4].set_ylim(300,0)
axes[2:4].set_yticks([300,200,100,0])
cf = axes[2].contourf(glob_ann_dclim.pivot(columns = 'lat',index = 'depth', values = 'cphy'), 
                      cmap='Browns1', levels = np.linspace(0,20,21))
divider = make_axes_locatable(axes[2])
cax = divider.append_axes('bottom', size='15%', pad=0.55)
fig.colorbar(cf, cax = cax, orientation='horizontal', extend = 'both', ticks = np.linspace(0,20,5), label = 'C$_{phy}$ (mg m$^{-3}$)')
axes[2].plot(glob_ann_iclim['lat'],glob_ann_iclim['mld'], color = 'black', lw = 1)
axes[2].plot(glob_ann_iclim['lat'],glob_ann_iclim['z90'], color = 'black', lw = 1, ls = 'dashed')

cf = axes[3].contourf(glob_ann_dclim.pivot(columns = 'lat',index = 'depth', values = 'chla'), 
                      cmap='Greens1', levels = np.linspace(0,0.8,21))
divider = make_axes_locatable(axes[3])
cax = divider.append_axes('bottom', size='15%', pad=0.55)
fig.colorbar(cf, cax = cax, orientation='horizontal', extend = 'both', ticks = np.linspace(0,0.8,5), label='Chla (mg m$^{-3}$)')
axes[3].plot(glob_ann_iclim['lat'],glob_ann_iclim['mld'], color = 'black', lw = 1)
axes[3].plot(glob_ann_iclim['lat'],glob_ann_iclim['z90'], color = 'black', lw = 1, ls = 'dashed')

axes[2].set_ylabel('Depth (m)', labelpad = 5)
axes[3].set_ylabel('Depth (m)', labelpad = 5)

#plt.savefig(dep_dir + '/03 figures/publish quality/fig s5.jpg', dpi = 300)
plt.show()

###################################################
# Fig. S8: ARGO/MODIS-AQUA CHLA BLOOM METRICS
###################################################
array = [[1,1,1,2,2,2],
         [3,3,4,4,5,5]]
fig, axes = pplt.subplots(array, width = 5, height = 6, sharey = False, sharex = True, spanx = False,
                          grid = False, hratios = (3,3), abc = 'A', wspace = 3, ylim = (-90,90), hspace = 5)
[ax.yaxis.set_major_formatter(lat_formatter) for ax in axes]
[ax.set_yticks(np.arange(-90,100,45)) for ax in axes]

splines = 15 # number of splines we will use
cgrid = ireg_df.pivot(index = 'lat', columns = 'rwoy', values = 'schla_zs')
im = axes[0].pcolor(cgrid.columns,cgrid.index, cgrid.values,
               cmap = 'Greens1', vmax = 1.5, vmin = -1, levels = 100)
divider = make_axes_locatable(axes[0])
cax = divider.append_axes('bottom', size='5%', pad=0.3)
fig.colorbar(im, cax = cax, orientation='horizontal', extend = 'both', ticks = np.arange(-1,2,0.5),
             label='Surface Chla (z-score)')
axes[0].set_xlim(-26.5,26.5)
axes[0].set_xticks(np.arange(-26,26+13,13))
axes[0].axvline(0, color = 'black', lw = 1, zorder = 1, ls = 'dashed')
axes[0].set_ylabel('Latitude', labelpad = 8)

cgrid = ireg_df.pivot(index = 'lat', columns = 'rwoy', values = 'sat_chla_zs')
im = axes[1].pcolor(cgrid.columns,cgrid.index, cgrid.values,
                    cmap = 'Greens1', vmax = 1.5, vmin = -1, levels = 100)
divider = make_axes_locatable(axes[1])
cax = divider.append_axes('bottom', size='5%', pad=0.3)
fig.colorbar(im, cax = cax, orientation='horizontal', extend = 'both', ticks = np.arange(-1,2,0.5),
             label='Satellite Chla (z-score)')
axes[1].set_xlim(-26.5,26.5)
axes[1].set_xticks(np.arange(-26,26+13,13))
axes[1].axvline(0, color = 'black', lw = 1, zorder = 1, ls = 'dashed')
axes[1].set_ylabel('', labelpad = 8)
axes[1].set_yticklabels([])
X = glob_ann_iclim[['lat']].values
chla_var = ['peak_schla', 'schla_dur', 'schla_cv']

for ax_ind in [2,3,4]:
    y = glob_ann_iclim[chla_var[ax_ind-2]].values
    gam = GAM(n_splines = splines).gridsearch(X, y)
    XX = gam.generate_X_grid(term = 0)
    axes[ax_ind].plot(gam.predict(XX), XX, color = 'green9', lw= 2, clip_on = False)
    axes[ax_ind].fill_betweenx(XX, gam.confidence_intervals(XX, width=.95)[:,1],
                                   gam.confidence_intervals(XX, width=.95)[:,0], XX, 
                                   color='green9', alpha = 0.3, clip_on = False)
    axes[ax_ind].scatter(y, X, color = 'green9', s= 15, clip_on = False)

cphy_var = ['chla_sat_peak', 'chla_sat_dur', 'chla_sat_cv']
for ax_ind in [2,3,4]:
    X = glob_ann_iclim[glob_ann_iclim[cphy_var[ax_ind-2]].notna()][['lat']].values
    y = glob_ann_iclim[cphy_var[ax_ind-2]].dropna().values
    axes[ax_ind].scatter(y, X, color = 'black', s= 30, clip_on = False, marker = 'x')
    
axes[2].set_xlim(-30,30)
axes[2].set_xticks(np.arange(-30,30+15,15))
axes[2].set_xlabel('Bloom Peak\n(Weeks from Solstice)', labelpad = 8)
axes[2].set_ylabel('Latitude', labelpad = 8)
axes[2].spines[['top', 'right']].set_visible(False)
axes[2].axvline(0, color = 'black', lw = 1, zorder = 1, ls = 'dashed')

axes[3].set_xlim(15,35)
axes[3].set_xticks(np.arange(15,35+5,5))
axes[3].set_xlabel('Blooming Period\n(Weeks)', labelpad = 8)
axes[3].spines[['top', 'right', 'left']].set_visible(False)
axes[3].set_yticks([])
axes[3].set_ylabel('')

axes[4].set_xlim(0,3)
axes[4].set_xticks(np.arange(0, 3+1,1))
axes[4].set_xlabel('Bloom Amplitude\n(normalized to mean)', labelpad = 8)
axes[4].spines[['top', 'right', 'left']].set_visible(False)
axes[4].set_yticks([])
axes[4].set_ylabel('')

#plt.savefig(dep_dir + '/03 figures/publish quality/fig s8.jpg', dpi = 300)
plt.show()

###################################################
# Fig. S6: CORRELATION COEFFICEINTS FOR CHLA VARS
###################################################
array = [[1,2]]
fig, axes = pplt.subplots(array, width = 5, height = 3, sharey = True, sharex = True, spanx = True,
                          grid = False, abc = 'A', ylim = (-90,90), xlim = (-1,1), ylabel = 'Latitude',
                          xlabel = 'Pearson Coefficient')
[ax.yaxis.set_major_formatter(lat_formatter) for ax in axes]
[ax.set_yticks(np.arange(-90,100,45)) for ax in axes]

axes[0].barh(glob_ann_iclim.lat, glob_ann_iclim.chla_r2, color = 'black')
axes[1].barh(glob_ann_iclim.lat, glob_ann_iclim.satchla_r2, color = 'black')

axes[0].axvline(0, color = 'black', lw = 1, zorder = 1, ls = 'dashed')
axes[1].axvline(0, color = 'black', lw = 1, zorder = 1, ls = 'dashed')

axes[0].annotate('Surface Chla v. \u03A3C$_{phy}$', 
                 xy=(0.5, 1.08), xycoords='axes fraction', fontsize=10, ha='center', va='top')
axes[1].annotate('Float v. Satellite Chla',
                 xy=(0.5, 1.08), xycoords='axes fraction', fontsize=10, ha='center', va='top')

#plt.savefig(dep_dir + '/03 figures/publish quality/fig s6.jpg', dpi = 300)
plt.show()


###################################################
# Fig. S7: ADDITIONAL BLOOM METRICS
###################################################
glob_ann_iclim['min_schla'] = glob_ann_iclim['min_schla'].rolling(window = 3, center= True, min_periods = 1).mean().round(0).values

array = [[1,2,3]]
X = glob_ann_iclim[['lat']].values
splines = 15
fig, axes = pplt.subplots(array, width = 5, height = 3, sharey = False, sharex = True, spanx = False,
                          grid = False, abc = 'A', wspace = 3, ylim = (-90,90), hspace = 5)
[ax.yaxis.set_major_formatter(lat_formatter) for ax in axes]
[ax.set_yticks(np.arange(-90,100,45)) for ax in axes]

cphy_var = ['min_schla','schla_r_max', 'schla_r_min']
for ax_ind in [0,1,2]:
    y = glob_ann_iclim[cphy_var[ax_ind]].values
    gam = GAM(n_splines = splines).gridsearch(X, y)
    XX = gam.generate_X_grid(term = 0)
    axes[ax_ind].plot(gam.predict(XX), XX, color = 'green9', lw= 2, clip_on = False)
    axes[ax_ind].fill_betweenx(XX, gam.confidence_intervals(XX, width=.95)[:,1],
                                   gam.confidence_intervals(XX, width=.95)[:,0], XX, 
                                   color='green9', alpha = 0.3, clip_on = False)
    axes[ax_ind].scatter(y, X, color = 'green9', s= 15, clip_on = False)

cphy_var = ['min_cphy','cphy_r_max', 'cphy_r_min']
for ax_ind in [0,1,2]:
    y = glob_ann_iclim[cphy_var[ax_ind]].values
    gam = GAM(n_splines = splines).gridsearch(X, y)
    XX = gam.generate_X_grid(term = 0)
    axes[ax_ind].plot(gam.predict(XX), XX, color = 'brown', lw= 2, clip_on = False)
    axes[ax_ind].fill_betweenx(XX, gam.confidence_intervals(XX, width=.95)[:,1],
                                   gam.confidence_intervals(XX, width=.95)[:,0], XX, 
                                   color='brown', alpha = 0.3, clip_on = False)
    axes[ax_ind].scatter(y, X, color = 'brown', s= 15, clip_on = False)
    
    axes[ax_ind].set_xlim(-30,30)
    axes[ax_ind].set_xticks(np.arange(-30,30+15,15))
    axes[ax_ind].axvline(0, color = 'black', lw = 1, zorder = 1, ls = 'dashed')

axes[0].set_xlabel('Timing of Minimum\n(Weeks from Midwinter)', labelpad = 8, fontsize = 9)
axes[1].set_xlabel('Maximum $\it{r}$\n(Weeks from Midwinter)', labelpad = 8, fontsize = 9)
axes[2].set_xlabel('Minimum $\it{r}$\n(Weeks from Midsummer)', labelpad = 8, fontsize = 9)

axes[0].set_ylabel('Latitude', labelpad = 8, fontsize = 9)
axes[0].spines[['top', 'right']].set_visible(False)
axes[1].spines[['top', 'right', 'left']].set_visible(False)
axes[2].spines[['top', 'right', 'left']].set_visible(False)
axes[1].set_yticks([])
axes[1].set_ylabel('')
axes[2].set_yticks([])
axes[2].set_ylabel('')
#plt.savefig(dep_dir + '/03 figures/publish quality/fig s7.jpg', dpi = 300)
plt.show()

###################################################
# Fig. S4: ABUNDANCE-BASED CPHYTO COMPARISON
###################################################
p_orig = pd.read_csv('/Users/adamstoer/Downloads/Global Picophytoplankton Data/picophyto111130.csv').rename(columns = {'picophyto [ug C/L]':'picobio'})
gdf = gpd.GeoDataFrame(p_orig, geometry=gpd.points_from_xy(p_orig.Long, p_orig.Lat))
p_orig = gdf.clip(mask)

sites = [[-60,-40,48,65],
         [0,36,30,45],
         [-72,-55,27,35],         
         [-170,-135,18,28],
         [160,-135,-5,5],
         [54,73,7,25],
         [-152,-130,47,54],
         [150,-148,-70,-50]]
site_name = ['Labrador Sea',
             'Mediterranean Sea', 
             'Sargasso Sea',
             'North Pacific Gyre',
             'Equatorial Pacific',
             'Arabian Sea',
             'Northeast Pacific',
             'Southern Ocean']
col_lst = ['grape5', 'red7','teal8','orange6','pink5','lime6', 'cyan5', 'yellow5']

# Create validation plot
array = [[1, 1, 1, 1],
         [2, 3, 4, 5], 
         [6, 7, 8, 9,]]
fig, ax = pplt.subplots(array, width = 5, height = 6, proj = {1:'moll'}, proj_kw={'central_longitude': -60}, hratios = (6,4,4),
                        abc = 'A', sharey = True, spany= False)
ax[0].format(grid=False)
ax[0].set_global()
ax[0].set_facecolor('gray6')

ax[0].add_feature(cfeature.LAKES, edgecolor='black', facecolor = 'gray6', zorder = 59, lw = 0.5)
ax[0].add_geometries(land_plot.geometry.values, crs=ccrs.PlateCarree(), facecolor = 'gray4', 
                     edgecolor='black', zorder = 35, lw = 0.5, label = 'Land')

ax[0].add_geometries(glob_ocean[glob_ocean['basin'] == 'pacific'].geometry.values, crs=ccrs.PlateCarree(), facecolor = 'blue1', 
                     edgecolor='blue1', zorder = 10, lw = 1, label = 'Pacific Ocean')
ax[0].add_geometries(glob_ocean[glob_ocean['basin'] == 'atlantic'].geometry.values, crs=ccrs.PlateCarree(), facecolor = 'blue2', 
                     edgecolor='blue2', zorder = 10, lw = 1, label = 'Atlantic Ocean')
ax[0].add_geometries(glob_ocean[glob_ocean['basin'] == 'indian'].geometry.values, crs=ccrs.PlateCarree(), facecolor = 'blue3', 
                     edgecolor='blue3', zorder = 10, lw = 1, label = 'Indian Ocean')

legend_elements = [Patch(facecolor='blue3', edgecolor='blue3', label='Indian Ocean'),
                   Patch(facecolor='blue2', edgecolor='blue2', label='Atlantic Ocean'),
                   Patch(facecolor='blue1', edgecolor='blue1', label='Pacific Ocean')]
#ax[0].legend(handles=legend_elements, ncol = 1, bbox_to_anchor = [1.55, 0.15], fontsize = 12, frameon = False)

ax[0].plot(np.repeat(146.916714, 50), np.linspace(-90, -35, 50), color = 'black', 
        lw = 2, transform = ccrs.PlateCarree(), zorder = 28, label = '')
ax[0].plot(np.repeat(-67.25, 50), np.linspace(-65, -55, 50), color = 'black', 
        lw = 2, transform = ccrs.PlateCarree(), zorder = 28, label = '')
ax[0].plot(np.repeat(20, 50), np.linspace(-90, -35, 50), color = 'black', 
        lw = 2, transform = ccrs.PlateCarree(), zorder = 28, label = '')
ax[0].scatter(df_profiles.longitude,df_profiles.latitude, s = 0.1, zorder = 25, marker = "o",
              color = 'black', alpha = 1, transform = ccrs.PlateCarree(), label = '')
ax[0].scatter(p_orig.Long, p_orig.Lat, s = 1, marker = 'o', color = 'red8',
              zorder = 26, transform = ccrs.PlateCarree())

for site in sites:
    count = sites.index(site)
    if site[0]<site[1]:
        ax[0].add_patch(mpatches.Rectangle(xy=[site[0], site[2]], width=site[1]-site[0], 
                                         height=site[3]-site[2], edgecolor = col_lst[count], 
                                         lw = 1.5, zorder = 50,facecolor='none',
                                         transform=ccrs.PlateCarree()))
    if site[0]>site[1]:
        ax[0].plot([-180,site[1]],[site[2],site[2]], lw = 1.5, color = col_lst[count], zorder = 80)
        ax[0].plot([-180,site[1]],[site[3],site[3]], lw = 1.5, color = col_lst[count], zorder = 80)
        ax[0].plot([180,site[0]],[site[2],site[2]], lw = 1.5, color = col_lst[count], zorder = 80)
        ax[0].plot([180,site[0]],[site[3],site[3]], lw = 1.5, color = col_lst[count], zorder = 80)
        ax[0].plot([site[1],site[1]],[site[2],site[3]], lw = 1.5, color = col_lst[count], zorder = 80)
        ax[0].plot([site[0],site[0]],[site[2],site[3]], lw = 1.5, color = col_lst[count], zorder = 80)

count = 0
n1, n2 = [],[]
for ind in np.arange(1,len(sites)+1,1):
    col = col_lst[ind-1]
        
    ax[ind].grid(False)
    ax[ind].patch.set_alpha(1.0)  

    site = sites[ind-1]      

    if site[0]<site[1]:
        p_site = p_orig[(p_orig.Long>site[0])&(p_orig.Long<site[1])&\
                        (p_orig.Lat>site[2])&(p_orig.Lat<site[3])]
    if site[0]>site[1]:
        p_site = p_orig[((p_orig.Long>site[0])|(p_orig.Long<site[1]))&\
                        (p_orig.Lat>site[2])&(p_orig.Lat<site[3])]
    
    p_site['picobio'] = p_site['picobio'].mul(0.8)
    if site_name[ind-1] == 'Labrador Sea':
        p_site = p_site[(p_site.month<=7)&(p_site.month>=5)]
    if site_name[ind-1] == 'Mediterranean Sea':
        p_site = p_site[(p_site.month>=5)&(p_site.month<=7)]  
    if site_name[ind-1] == 'Southern Ocean':
        p_site = p_site[(p_site.month>=10)|(p_site.month<=3)]         
    if site_name[ind-1] == 'Northeast Pacific':
        p_site = p_site[(p_site.month>=6)&(p_site.month<=8)]     
        
    n1.append(len(p_site.groupby(['Lat','Long','year','month','day']).mean(numeric_only = True)))
    bins = np.arange(0,240+20,20)
    p_site = p_site.groupby(['month',pd.cut(p_site.Depth, bins)], dropna = False).agg({'picobio':['mean','std']}).reset_index()
    p_site.columns = list(map('_'.join, p_site.columns.values))
    p_site.loc[:,'picobio_std'] = p_site.loc[:,'picobio_std'].pow(2) #square each
    p_site.loc[p_site.picobio_mean.notna(),'counter'] = 1

    p_site = p_site.reset_index().groupby('Depth_').sum().reset_index()
    p_site.loc[:,'picobio_std'] = p_site.loc[:,'picobio_std'].div(p_site.loc[:,'counter']).pow(0.5) #return propagated error
    p_site.loc[:,'picobio_mean'] = p_site.loc[:,'picobio_mean'].div(p_site.loc[:,'counter']) #square each
    p_site['depth'] = p_site['Depth_'].apply(lambda x: x.mid).tolist()
        
    ax[ind].errorbar(p_site.picobio_mean.div(p_site.picobio_mean.max()).values, 
                     p_site.depth.values, xerr = p_site.picobio_std.div(p_site.picobio_mean.max()).values,
                capsize = 0, markersize = 5, elinewidth = 2, 
                color = col, lw = 2.5, marker = 'o', zorder = 4)
    if site[0]<site[1]:
        p_subset = df_main[(df_main.longitude>site[0])&(df_main.longitude<site[1])&\
                           (df_main.latitude>site[2])&(df_main.latitude<site[3])]

    if site[0]>site[1]:
        p_subset = df_main[((df_main.longitude>site[0])|(df_main.longitude<site[1]))&\
                             (df_main.latitude>site[2])&(df_main.latitude<site[3])]
    n2.append(len(p_subset.id.unique()))
    
    
    if site_name[ind-1] == 'Labrador Sea':
        p_subset = p_subset[(p_subset.month<=7)&(p_subset.month>=5)]
        ax[ind].text(1.1,280,'Labrador\nSea', ha = 'right', fontsize = 8)
    if site_name[ind-1] == 'Mediterranean Sea':
        p_subset = p_subset[(p_subset.month>=5)&(p_subset.month<=7)]  
        p_subset = p_subset[(p_subset.latitude<40)|(p_subset.longitude<30)]  
        ax[ind].text(1.1,280,'Mediterranean\nSea', ha = 'right', fontsize = 8)
    if site_name[ind-1] == 'Sargasso Sea':
        ax[ind].text(1.1,280,'Sargasso\nSea', ha = 'right', fontsize = 8)
    if site_name[ind-1] == 'North Pacific Gyre':
        ax[ind].text(1.1,280,'North Pacific\nGyre', ha = 'right', fontsize = 8)
        
    if site_name[ind-1] == 'Equatorial Pacific':
        ax[ind].text(1.1,280,'Equatorial\nPacific', ha = 'right', fontsize = 8)
    if site_name[ind-1] == 'Arabian Sea':
        ax[ind].text(1.1,280,'Arabian Sea\nSea', ha = 'right', fontsize = 8)
    if site_name[ind-1] == 'Northeast Pacific':
        ax[ind].text(1.1,280,'Northeast\nPacific', ha = 'right', fontsize = 8)
        p_subset = p_subset[(p_subset.month>=6)&(p_subset.month<=8)]      
    if site_name[ind-1] == 'Southern Ocean':
        ax[ind].text(1.1,280,'Southern\nOcean', ha = 'right', fontsize = 8)
        p_subset = p_subset[(p_subset.month>=10)|(p_subset.month<=3)]      
        
    reg_sub = p_subset.copy()
    
    t_res = 'month'
    reg_sub['woy'] = reg_sub['local_time'].dt.isocalendar().week.astype(int)
    reg_sub = reg_sub[reg_sub['woy']!=53]
        
    reg_sub.loc[:,'chla'] = reg_sub.loc[:,'chla'].div(1)
    
    reg_sub.loc[:,'bbp470'] = reg_sub.loc[:,'bbp470'].add(1).apply(np.log)
    reg_sub.loc[:,'chla'] = reg_sub.loc[:,'chla'].add(1).apply(np.log)
                
    reg_sub = reg_sub.groupby([t_res,'depth'], dropna = False).agg({'bbp470': 'mean',
                                                                    'chla': 'mean',
                                                                    'mld':'mean',
                                                                    'temperature':'mean',
                                                                    'id': 'nunique'})
    
    reg_sub = reg_sub.reset_index()
    reg_sub.loc[:,'bbp470'] = (np.e**(reg_sub.loc[:,'bbp470'])).sub(1)
    reg_sub.loc[:,'chla'] = (np.e**(reg_sub.loc[:,'chla'])).sub(1)
    
    for t_ind in reg_sub[t_res].unique():
        reg_sub.loc[reg_sub[t_res]==t_ind,'bbpphy'], z_lim = ot.sep_bbp(reg_sub[reg_sub[t_res]==t_ind],'depth','chla','bbp470')
        reg_sub.loc[reg_sub[t_res]==t_ind,'cphy'] = ot.bbp_to_cphy(reg_sub.loc[reg_sub[t_res]==t_ind,'bbpphy'], lr_graff.slope)
        reg_sub.loc[reg_sub[t_res]==t_ind,'z_lim'] = z_lim
        
    # geometric mean annual profiles
    reg_sub.loc[:,'cphy'] = reg_sub.loc[:,'cphy'].add(1).apply(np.log)
    reg_sub.loc[:,'chla'] = reg_sub.loc[:,'chla'].add(1).apply(np.log)
                
    avg_profile = reg_sub.groupby('depth').mean(numeric_only=True).reset_index()
    
    avg_profile.loc[:,'cphy'] = (np.e**(avg_profile.loc[:,'cphy'])).sub(1)
    avg_profile.loc[:,'chla'] = (np.e**(avg_profile.loc[:,'chla'])).sub(1)
    
    ax[ind].plot(avg_profile['cphy'].div(avg_profile['cphy'].max()).values, 
                 avg_profile['depth'].values, lw = 2.5, color = 'black', zorder = 8)
    ax[ind].set_xlim(0,1.2)
    ax[ind].set_xticks([0,0.5,1])
    ax[ind].set_ylim(300,0)
    
    ax[ind].set_ylabel('Depth (m)')
    ax[ind].set_xlabel('C$_{phy}$ (relative)')
    
    
    bins = np.arange(0,240+20,20)
    avg_profile_20m = avg_profile.groupby([pd.cut(avg_profile.depth, bins)], dropna = False).mean().reset_index(drop=True)
    merged_df = p_site[['depth','picobio_mean']].merge(avg_profile_20m[['depth','cphy']], on = 'depth', how = 'right').dropna()
    pc = stats.linregress(merged_df['cphy'], merged_df['picobio_mean'])
    
    #print(np.round(pc.rvalue**2,2))
    #print(np.round(pc.pvalue))

#plt.savefig(dep_dir + '/03 figures/publish quality/fig s4.jpg', dpi = 300)
plt.show()
 



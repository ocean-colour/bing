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

root_dir = '/Volumes/T7/data/'
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
lbp_ocean = gpd.read_file(root_dir + '/marine regions/longhurst_v4_2010/Longhurst_world_v4_2010.shp')  

g15_df = pd.read_csv(dep_dir + '/02 processed data/graff et al 2015 data/graff15_data.csv') # digitized data from Graff et al. (2015)
lr_graff = stats.linregress(g15_df.bbp470,g15_df.cphyto) # run linear regression to get slope
corg_mape = g15_df.bbp470.mul(lr_graff.slope).add(lr_graff.intercept)\
    .sub(g15_df.cphyto).abs().div(g15_df.cphyto).mean() # calculate MAPE for C_phy

#############################################################################
# Gather Biogeocheical Data from Floats
#############################################################################
# Access the processed float data
argo_file_grab_lst = filegrab(dep_dir + '/02 processed data/processed float data',
                              'binned.csv',
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
                              crs=lbp_ocean.crs)
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
topo_image = rasterio.open(root_dir + '/ETOP05 Data/ETOPO_2022_v1_60s_N90W180_surface_mod.tiff').read(1)

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

lbp_ocean_clipped = lbp_ocean.clip(mask)
lbp_ocean_clipped['area'] = lbp_ocean_clipped.to_crs(epsg=6933).area # Calculate area (m^2)

df_area = lbp_ocean_clipped.copy()
df_area = df_area[['area','ProvCode']]
df_area['volume'] = df_area['area'].mul(5) #meters cubed or m^3

#############################################################################
# Irradiance QC and Slope Factor Calculations
#############################################################################
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
    

# par stuff
cal_data.loc[(cal_data['lr_r2_par']<0.9),'z90'] = np.nan
cal_data.loc[(cal_data['lr_r2_par']<0.9),'kd_par'] = np.nan
del cal_data['lat_bin']
cal_data = gpd.GeoDataFrame(cal_data, geometry = gpd.points_from_xy(cal_data['longitude'], cal_data['latitude']), 
                               crs=lbp_ocean.crs)
cal_data = gpd.sjoin(cal_data, lbp_ocean, predicate ='within')

cal_data.loc[:,'sf'] = (cal_data.loc[:,'sf']).add(1).apply(np.log)
cal_data.loc[:,'z90'] = (cal_data.loc[:,'z90']).add(1).apply(np.log)

cal_data_group = cal_data.groupby(['ProvCode','month']).mean(numeric_only= True).reset_index()
cal_data_group = cal_data_group.groupby(['ProvCode']).mean(numeric_only= True).reset_index()

cal_data_group.loc[:,'sf'] = (np.e**(cal_data_group.loc[:,'sf'])).sub(1)
cal_data_group.loc[:,'z90'] = (np.e**(cal_data_group.loc[:,'z90'])).sub(1)
cal_data.loc[:,'sf'] = (np.e**(cal_data.loc[:,'sf'])).sub(1)
cal_data.loc[:,'z90'] = (np.e**(cal_data.loc[:,'z90'])).sub(1)

for basin in cal_data_group['ProvCode'].unique():
            
        if len(cal_data_group[(cal_data_group['ProvCode']==basin)])!=0:
            sf = cal_data_group[(cal_data_group['ProvCode']==basin)].sf.values[0]
            data_sub = cal_data[(cal_data['ProvCode']==basin)].copy()
                        
            chla_pre = data_sub.avg_chlaf_irrad490.div(sf)
            chla_act = data_sub.chla_kd
            
            abs_err = chla_pre.sub(chla_act).div(chla_act).abs()
            mape = abs_err.mean() # MAE in depth avg chlorophyll-a 
            cal_data_group.loc[(cal_data_group['ProvCode']==basin), 'chla_err'] = mape

            # Scale Factor Error
            sf_pre = sf
            sf_act = data_sub.sf
            
            abs_err = sf_act.sub(sf_pre).abs()
            mape = data_sub.sf.std() # MAE in depth avg chlorophyll-a
            cal_data_group.loc[(cal_data_group['ProvCode']==basin), 'sf_err'] = mape
        
        
cal_map = lbp_ocean.merge(cal_data_group, on = ['ProvCode'], how = 'left')
#cal_map.plot(column = cal_map.sf, cmap = 'jet')
#cal_data.geometry.plot(markersize = 1, color = 'black')

cal_map.loc[cal_map['ProvCode']=='NEWZ', 'sf'] = cal_map.loc[cal_map['ProvCode']=='SANT', 'sf'].values[0]
cal_map.loc[cal_map['ProvCode']=='BERS', 'sf'] = cal_map.loc[cal_map['ProvCode']=='PSAW', 'sf'].values[0]
cal_map.loc[cal_map['ProvCode']=='ALSK', 'sf'] = cal_map.loc[cal_map['ProvCode']=='PSAW', 'sf'].values[0]
cal_map.loc[cal_map['ProvCode']=='PSAE', 'sf'] = cal_map.loc[cal_map['ProvCode']=='PSAW', 'sf'].values[0]
cal_map.loc[cal_map['ProvCode']=='NPTG', 'sf'] = cal_map.loc[cal_map['ProvCode']=='NPSW', 'sf'].values[0]
cal_map.loc[cal_map['ProvCode']=='NECS', 'sf'] = cal_map.loc[cal_map['ProvCode']=='NADR', 'sf'].values[0]
cal_map.loc[cal_map['ProvCode']=='INDE', 'sf'] = cal_map.loc[cal_map['ProvCode']=='MONS', 'sf'].values[0]
cal_map.loc[cal_map['ProvCode']=='INDW', 'sf'] = cal_map.loc[cal_map['ProvCode']=='MONS', 'sf'].values[0]
cal_map.loc[cal_map['ProvCode']=='CHIN', 'sf'] = cal_map.loc[cal_map['ProvCode']=='SUND', 'sf'].values[0]
cal_map.loc[cal_map['ProvCode']=='CARB', 'sf'] = cal_map.loc[cal_map['ProvCode']=='NATR', 'sf'].values[0]
cal_map.loc[cal_map['ProvCode']=='GUIA', 'sf'] = cal_map.loc[cal_map['ProvCode']=='WTRA', 'sf'].values[0]


cal_map.loc[cal_map['ProvCode']=='NEWZ', 'chla_err'] = cal_map.loc[cal_map['ProvCode']=='SANT', 'chla_err'].values[0]
cal_map.loc[cal_map['ProvCode']=='BERS', 'chla_err'] = cal_map.loc[cal_map['ProvCode']=='PSAW', 'chla_err'].values[0]
cal_map.loc[cal_map['ProvCode']=='ALSK', 'chla_err'] = cal_map.loc[cal_map['ProvCode']=='PSAW', 'chla_err'].values[0]
cal_map.loc[cal_map['ProvCode']=='PSAE', 'chla_err'] = cal_map.loc[cal_map['ProvCode']=='PSAW', 'chla_err'].values[0]
cal_map.loc[cal_map['ProvCode']=='NPTG', 'chla_err'] = cal_map.loc[cal_map['ProvCode']=='NPSW', 'chla_err'].values[0]
cal_map.loc[cal_map['ProvCode']=='NECS', 'chla_err'] = cal_map.loc[cal_map['ProvCode']=='NADR', 'chla_err'].values[0]
cal_map.loc[cal_map['ProvCode']=='INDE', 'chla_err'] = cal_map.loc[cal_map['ProvCode']=='MONS', 'chla_err'].values[0]
cal_map.loc[cal_map['ProvCode']=='INDW', 'chla_err'] = cal_map.loc[cal_map['ProvCode']=='MONS', 'chla_err'].values[0]
cal_map.loc[cal_map['ProvCode']=='CHIN', 'chla_err'] = cal_map.loc[cal_map['ProvCode']=='SUND', 'chla_err'].values[0]
cal_map.loc[cal_map['ProvCode']=='CARB', 'chla_err'] = cal_map.loc[cal_map['ProvCode']=='NATR', 'chla_err'].values[0]
cal_map.loc[cal_map['ProvCode']=='GUIA', 'chla_err'] = cal_map.loc[cal_map['ProvCode']=='WTRA', 'chla_err'].values[0]


cal_map.plot(column = cal_map.sf, cmap = 'jet')


# Seperate data into each basin
df_profiles = df_main[['id','longitude','latitude']].groupby('id').first().reset_index()
df_profiles = gpd.GeoDataFrame(df_profiles, geometry = gpd.points_from_xy(df_profiles.longitude, df_profiles.latitude), 
                               crs=lbp_ocean.crs)

reg_clim_lst = []    
for basin in lbp_ocean['ProvCode'].unique():
    
    ocean_poly = lbp_ocean[(lbp_ocean['ProvCode'] == basin)]
    minlon , minlat , maxlon , maxlat = ocean_poly.total_bounds # Faster processing wtih dropping float data out of bounds
    
    df_profiles_present = df_profiles[(df_profiles['longitude']>=minlon)&(df_profiles['longitude']<=maxlon)&\
                                      (df_profiles['latitude']>=minlat)&(df_profiles['latitude']<=maxlat)]
    ava_profiles = gpd.sjoin(df_profiles_present, ocean_poly, predicate ='within')

    sf = cal_map[(cal_map['ProvCode']==basin)]['sf'].values[0]
    if (len(ava_profiles)!=0):
            
        df_main.loc[df_main['id'].isin(ava_profiles.id.tolist()),'ProvCode'] = basin
        

        reg_sub = df_main[(df_main['id'].isin(ava_profiles.id.tolist()))]
        
        t_res = 'woy'
        reg_sub['woy'] = reg_sub['local_time'].dt.isocalendar().week.astype(int)
        reg_sub = reg_sub[reg_sub['woy']!=53]
        
        
        
        sf = cal_map[(cal_map['ProvCode']==basin)].sf.values[0]
        
        reg_sub.loc[:,'chla'] = reg_sub.loc[:,'chla'].div(sf)
        
        reg_sub.loc[:,'bbp470'] = reg_sub.loc[:,'bbp470'].add(1).apply(np.log)
        reg_sub.loc[:,'chla'] = reg_sub.loc[:,'chla'].add(1).apply(np.log)
                    
        reg_sub = reg_sub.groupby([t_res,'depth'], dropna = False).agg({'bbp470': 'mean',
                                                                        'chla': 'mean',
                                                                        'mld':'mean',
                                                                        'temperature':'mean',
                                                                        'id': 'nunique'})
        
        #reg_sub.columns = list(map('_'.join, reg_sub.columns.values))
        reg_sub = reg_sub.reset_index()
                    
        reg_sub.loc[:,'bbp470'] = (np.e**(reg_sub.loc[:,'bbp470'])).sub(1)
        reg_sub.loc[:,'chla'] = (np.e**(reg_sub.loc[:,'chla'])).sub(1)
        
             
        
        for t_ind in reg_sub.woy.unique():
            reg_sub.loc[reg_sub[t_res]==t_ind,'bbpphy'], z_lim = ot.sep_bbp(reg_sub[reg_sub[t_res]==t_ind],'depth','chla','bbp470')
            reg_sub.loc[reg_sub[t_res]==t_ind,'cphy'] = ot.bbp_to_cphy(reg_sub.loc[reg_sub[t_res]==t_ind,'bbpphy'], lr_graff.slope)
            reg_sub.loc[reg_sub[t_res]==t_ind,'z_lim'] = z_lim

        reg_sub_before = reg_sub.copy()
        reg_sub_before['woy'] = reg_sub_before['woy'].sub(52)
        reg_sub_after = reg_sub.copy()
        reg_sub_after['woy'] = reg_sub_after['woy'].add(52)

        reg_clim = pd.concat([reg_sub_before,reg_sub,reg_sub_after]).reset_index(drop=False)
        
        for w in np.arange(1-52,53+52,1):
            if w not in reg_clim.woy.unique():                    
                empty_df = reg_clim.loc[reg_clim['woy']==reg_clim['woy'].unique()[0], :]
                empty_df.loc[empty_df['woy']==reg_clim['woy'].unique()[0], 'woy'] = w
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
        
        
        poly_vol = df_area[(df_area.ProvCode == basin)].reset_index() # in m^2
        
        reg_clim.loc[:,'volume'] = poly_vol.loc[:,'volume'].values[0]
        reg_clim.loc[:,'area'] = poly_vol.loc[:,'area'].values[0]
            
        reg_clim['ProvCode'] = basin # deg N
        reg_clim_lst.append(reg_clim)
    
glob_df = pd.concat(reg_clim_lst).reset_index(drop=True)

#here
missing_lbp_codes = lbp_ocean[~lbp_ocean.ProvCode.isin(glob_df.ProvCode.unique())].ProvCode.values

reg_clim = glob_df[(glob_df['ProvCode']=='SUND')]
reg_clim['ProvCode'] = 'CHIN'

poly_vol = df_area[(df_area.ProvCode == 'CHIN')].reset_index() # in m^2
reg_clim.loc[:,'volume'] = poly_vol.loc[:,'volume'].values[0]

reg_clim['cphy_stock'] = reg_clim['cphy'].mul(reg_clim['volume']) #mg m^{-3} to mg
reg_clim['chla_stock'] = reg_clim['chla'].mul(reg_clim['volume']) #mg m^{-3} to mg  
  
reg_clim_lst.append(reg_clim)

reg_clim = glob_df[(glob_df['ProvCode']=='NADR')]
reg_clim['ProvCode'] = 'NECS'

poly_vol = df_area[(df_area.ProvCode == 'NECS')].reset_index() # in m^2
reg_clim.loc[:,'volume'] = poly_vol.loc[:,'volume'].values[0]

reg_clim['cphy_stock'] = reg_clim['cphy'].mul(reg_clim['volume']) #mg m^{-3} to mg
reg_clim['chla_stock'] = reg_clim['chla'].mul(reg_clim['volume']) #mg m^{-3} to mg  

reg_clim_lst.append(reg_clim)

glob_df = pd.concat(reg_clim_lst).reset_index(drop=True)
glob_df = glob_df.merge(cal_map[['chla_err','ProvCode']],on= ['ProvCode'])


glob_df['chla_l'] = glob_df['chla'].sub(glob_df['chla'].mul(glob_df['chla_err']))
glob_df['chla_u'] = glob_df['chla'].add(glob_df['chla'].mul(glob_df['chla_err']))

glob_df['cphy_l'] = glob_df['cphy'].sub(glob_df['cphy'].mul(corg_mape))
glob_df['cphy_u'] = glob_df['cphy'].add(glob_df['cphy'].mul(corg_mape))


# Define a lambda function to compute the weighted mean:
integral = lambda x: np.sum(x)*5 #mean
add = lambda x: np.sum(x) #mean
##############################################################################
### CALCULATE THE GLOBAL WEEKLY DEPTH-RESOLVED DATA
### VALUES ARE AVERAGES WEIGHTED BY SURFACE AREA


### Calculate surface averages
glob_df.loc[:, 'schla'] = glob_df.loc[glob_df['depth']<glob_df['mld'], 'chla']
glob_df.loc[:, 'scphy'] = glob_df.loc[glob_df['depth']<glob_df['mld'], 'cphy']

###  CALCULATE THE GLOBAL WEEKLY INTEGRATED DATA
glob_woy_dclim = glob_df.groupby(['ProvCode','woy']).agg(area=('area', 'first'),  
                                                       mld = ('mld', 'first'),
                                                       cphy = ('cphy', integral),
                                                       cphy_u = ('cphy_u', integral),
                                                       cphy_l = ('cphy_l', integral),
                                                       chla = ('chla', integral),
                                                       chla_u = ('chla_u', integral),
                                                       chla_l = ('chla_l', integral),
                                                       schla = ('schla', 'mean'),
                                                       scphy = ('scphy', 'mean')).reset_index()

for var in ['chla','chla_l','chla_u','cphy', 'cphy_u', 'cphy_l']:
    glob_woy_dclim.loc[:,var] = glob_woy_dclim.loc[:,var].add(1).apply(np.log)
    
glob_ann_iclim = glob_woy_dclim.groupby(['ProvCode']).agg(area=('area', 'first'), 
                                                          mld = ('mld', 'mean'),
                                                         cphy = ('cphy', 'mean'),
                                                         cphy_u = ('cphy_u', 'mean'),
                                                         cphy_l = ('cphy_l', 'mean'),
                                                         chla = ('chla', 'mean'),
                                                         chla_u = ('chla_u', 'mean'),
                                                         chla_l = ('chla_l', 'mean'),).reset_index()    


for var in ['chla','chla_l','chla_u','cphy', 'cphy_u', 'cphy_l']:
    glob_ann_iclim.loc[:,var] = (np.e**(glob_ann_iclim.loc[:,var])).sub(1)

glob_ann_iclim['cphy_stock'] = glob_ann_iclim.cphy.mul(glob_ann_iclim.area)
glob_ann_iclim['cphy_l_stock'] = glob_ann_iclim.cphy_l.mul(glob_ann_iclim.area)
glob_ann_iclim['cphy_u_stock'] = glob_ann_iclim.cphy_u.mul(glob_ann_iclim.area)

glob_ann_iclim['chla_stock'] = glob_ann_iclim.chla.mul(glob_ann_iclim.area)
glob_ann_iclim['chla_l_stock'] = glob_ann_iclim.chla_l.mul(glob_ann_iclim.area)
glob_ann_iclim['chla_u_stock'] = glob_ann_iclim.chla_u.mul(glob_ann_iclim.area)


print('Total: ' + str((glob_ann_iclim.cphy_stock.sum())*1e-15))  
print('Total: ' + str((glob_ann_iclim.cphy_l_stock.sum()*1e-15)))
print('Total: ' + str((glob_ann_iclim.cphy_u_stock.sum()*1e-15)))

print('Total: ' + str((glob_ann_iclim.chla_stock.sum())*1e-15))   
print('Total: ' + str((glob_ann_iclim.chla_l_stock.sum())*1e-15))   
print('Total: ' + str((glob_ann_iclim.chla_u_stock.sum())*1e-15))
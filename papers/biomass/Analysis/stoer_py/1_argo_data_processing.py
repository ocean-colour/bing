#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This program is associated with the article:
    
    Stoer, A., & Fennel K. 2024. Carbon-centric dynamics of Earth’s 
    marine phytoplankton. PNAS.

Software Summary:
    This program processes the Sprof files from BGC-Argo floats. This includes
    quality-control of the particle backscatter and chlorophyll-a fluorescence
    data profiles. The quality-control for temperature, salinity, and pressure
    are based on the Argo quality-control flags if available. 
    
    The processed files are binned and averaged as csv files.

"""

# Import packages
from itertools import compress,groupby
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import xarray as xr
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec
import datetime
import gsw
import matlab.engine #need for accessing matlab code from Python
eng = matlab.engine.start_matlab()
from scipy import stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import sys

# Variable List/ needed for efficient plotting
vlst = ['temperature','salinity','chla','bbp700'] 

# Labels for Figures/ needed for plotting
tlab = 'Temperature [°C]'
slab = 'Salinity [PSU]'
clab = 'Chlorophyll-a Fluorescence [mg m$^{-3}$]'
blab = 'Particulate Backscattering [700 nm; m$^{-1}$]'

lablst = [tlab,slab,clab,blab]

#Set depth limits and resolution    
min_depth, bin_size, max_depth  = 0, 5, 2000
    
# Colormaps for Data Summary
colormap = plt.get_cmap('turbo')

#Set Figures Style
size = 12
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : size}
font_small = {'family' : 'Arial',
              'weight' : 'normal',
              'size'   : 9}
plt.rc('font', **font)
axeswidth = 1.2
length = 8
linewidth = 2
plt.rcParams['axes.labelsize'] = size
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.linewidth'] = axeswidth
plt.rcParams['xtick.labelsize'] = size
plt.rcParams['ytick.labelsize'] = size
plt.rcParams["figure.titleweight"] = "normal"
plt.rcParams["ytick.major.width"] = axeswidth
plt.rcParams["xtick.major.width"] = axeswidth
plt.rcParams["ytick.major.size"] = length
plt.rcParams["xtick.major.size"] = length
plt.rcParams["ytick.direction"] = 'out'
plt.rcParams["xtick.direction"] = 'out'
plt.rcParams['lines.linewidth'] = linewidth
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.grid'] = True
plt.rc('grid', ls='--', lw=axeswidth)
plt.rc('font', **font)

# Data Directories
root_dir = '/Users/adamstoer/Synced Documents/data/'

# Custom Functions
def filegrab(root,find,start): # custom function to find all the files in a folder
    filelst = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(find):
                if filepath.startswith(start):
                    filelst.append(filepath)
    return filelst

def add_colorbar(fig,sc,a): # custom function to add a nice colorbar
    divider = make_axes_locatable(a)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax, orientation='vertical')
    
    return

def plot_data_summary(df,file_name): 
    # plot the float data/ this is useful for quality assurance
    df_good = df.copy()
    
    d_type = 'depth'        
    plt.close('all')
    fig = plt.figure(figsize = (20,10))    
    gs1 = gridspec.GridSpec(2, 2)

    axlst  = []
    for gs in gs1:
        axlst.append(fig.add_subplot(gs))
       
    gs1.tight_layout(fig, rect=[0.01, 0.04, 0.5, 0.98], w_pad = 2)
    
    gs2 = gridspec.GridSpec(1, 1)
    ax_map = fig.add_subplot(gs2[0],projection=ccrs.PlateCarree())
    gs2.tight_layout(fig, rect=[0.51, 0.6, 1, 0.95], h_pad = 0.5)
    
    gs3 = gridspec.GridSpec(2, 2)

    axhlst  = []
    for gs in gs3:
        axhlst.append(fig.add_subplot(gs))
    gs3.tight_layout(fig, rect=[0.54, 0.04, 0.95, 0.55], h_pad = 2, w_pad = 1)

    profile_df_mean = df.groupby(['time']).mean()
    profile_df_mean = profile_df_mean.reset_index()
    for ax in axlst:
        i = axlst.index(ax)
        if vlst[i] in df_good.columns and len(df_good[vlst[i]].dropna()) != 0:
            ran = (df_good[vlst[i]].quantile(0.05),df_good[vlst[i]].quantile(0.95))
            if vlst[i] in ['chla']:
                ran = (0,df_good[vlst[i]].quantile(0.99))
            if vlst[i] in ['bbp700']:
                ran = (0.0001,df_good[vlst[i]].quantile(0.99))
            sc = ax.scatter(df_good.time,df_good[d_type], 
                            c = df_good[vlst[i]], s = 20,
                            vmin = ran[0], vmax = ran[1], 
                            cmap = colormap)
            add_colorbar(fig,sc,ax)
            ax.grid(False)
    
        ax.set_xlim(profile_df_mean.time.min().date() - datetime.timedelta(days=1),
                    profile_df_mean.time.max().date() + datetime.timedelta(days=1))
        
        ax.set_title(lablst[i], loc = 'left')
        ax.plot(df_good.time.to_numpy(),df_good.mld.to_numpy(), color = 'red')
            
        ax.set_ylim(df_good.depth.round(-1).max(),0)
        ax.set_yticks(np.linspace(df_good.depth.round(-1).max(),0,6))    
        if i in [1,3]:
            plt.setp(ax.get_yticklabels(), visible=False)

        if i in [0,1,]:
            plt.setp(ax.get_xticklabels(), visible=False)

        if i in [0,2]:
            ax.set_ylabel('Depth [m]')   
        
        if i in [2,3]:
            ax.set_xlabel('Date (yyyy-mm-dd)')
            ax.tick_params(axis='x', rotation=30)
  
    ax_map.plot(df.sort_values(by=['time']).longitude.to_numpy(),
                df.sort_values(by=['time']).latitude.to_numpy(), color = 'black',
                transform = ccrs.PlateCarree(), marker = 'o', markersize = 2)
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='olive',
                                            facecolor='olive')
    ax_map.add_feature(land_50m)
    ax_map.add_feature(cfeature.LAKES, e dgecolor='black', facecolor='white')
 
                    width=(bins[1]-bins[0]), color='black')
            axh.set_xlim(ran)
        axh.set_title(lablst[i], loc = 'left')
        axh.set_ylim(0,1)

    plt.tight_layout()
    plt.savefig('/Users/adamstoer/Synced Documents/projects/2021/project 2103/'\
                         + '03 figures/Quick View Figures/Floats/' + \
                file_name.split('/')[-1].split('.')[0] + '_summary' + '.png', 
                dpi=300)
    plt.close()
    
    del df_good

    return

def prep_df(df, file_grabbed):
    df_binned = np.nan
    
    # Calculate depth in m from pressure in dbar
    df['depth'] = gsw.z_from_p(df['pressure'].values, df['latitude'].mean())*-1
    df = df.sort_values(['time','depth'])
        
    print('Found ' + str(max(df['profile_index'])) + ' profiles')    
    
    df = df[(df['depth']<max_depth)&(df['depth']>0.0)]# Trim the profile to 2000
    df = df[(df.chla.notna())&(df.bbp700.notna())&\
            (df.temperature.notna())&(df.salinity.notna())].copy() 
            
    if 'NOAA APEX' in file_grabbed:
        print('recalc bbp')
        df['betasw'] = [eng.betasw_ZHH2009(float(700),t,float(142), s, nargout = 1) for t,s in zip(df['temperature'].tolist(),df['salinity'].tolist())]        
        df['bbp700'] = (df['bbp700'].sub(df['betasw']).mul(2*np.pi*1.097))

    # QC Steps Per Profiles
    print('\nProcessing Profiles')
    malfunc_biofoul_lst = []
    malfunc_biofoul_time_lst = []
    profile_lst = []
    pr_lst = []
    chla_res_lst = []
    prof_counter = 0
    chl_offset_old = np.nan
    profile_reindex = 0
    
    for pindex in np.sort(list(set(df.profile_index))):
        profile = df[(df.profile_index == pindex)].copy() # Runs faster than directly indexing
        profile['profile_index'] = profile_reindex # Re-index the profile
        profile_reindex = profile_reindex + 1
        
        ### GAP TEST
        zgap = np.nan
        zgap1 = profile[(profile.depth<300)].depth.sub(profile[(profile.depth<300)].depth.shift(1)).max()
        zgap2 = 300-profile[(profile.depth<300)].depth.max()
        
        if zgap1 > zgap2:
            zgap = zgap1
        if zgap2 > zgap1:
            zgap = zgap2
        
        zgap_test = zgap<100
        
        ### DEPTH RANGE TEST
        min_depth_test = (profile['depth'].min()<15)
        max_depth_test = (profile['depth'].max()>300)
        
        ### SEA ICE AVOIDANCE TEST
        ice_detect = profile[(profile.depth<50)].temperature.median()<-1
        
        if ice_detect and (min_depth_test == False): # float didnt reach the surface for some reason
            min_depth_test = (profile['depth'].min()<25)
        
        ### PROCEED WITH GAP TEST AND RANGE CHECK
        if zgap_test and max_depth_test and min_depth_test:
            
            ### Calculate the MLD
            profile['abs_salinity'] = gsw.SA_from_SP(profile['salinity'], profile['pressure'], 
                                                     profile['longitude'], profile['latitude']) # Calculate Absolute Salinity
            profile['con_temperature'] = gsw.CT_from_t(profile['abs_salinity'], profile['temperature'], 
                                                       profile['pressure']) # Calculate Conservative Temperature
            profile['density'] = gsw.sigma0(profile['abs_salinity'], profile['con_temperature']) # Calculate Potential Density
            
            if ice_detect == False: 
                ref_depth = 15
                
                up_z = min(profile[profile.depth<ref_depth].depth, key=lambda x:abs(x-ref_depth))
                low_z = min(profile[profile.depth>ref_depth].depth, key=lambda x:abs(x-ref_depth))
                
                df_id = profile[profile.depth.isin([up_z,low_z])][['density','depth']]
                lr = stats.linregress(df_id.depth,df_id.density)
                ref_dens = lr.slope*ref_depth+lr.intercept # find reference density at depth z
                
            if ice_detect:
                ref_depth = profile.depth == profile.depth.min()
                
                ref_dens = profile[profile.depth == profile.depth.min()].density.values[0]
            
            mld = profile.loc[(profile['density'] > (ref_dens+0.03))&(profile.depth>ref_depth),'depth'].min() #calculate mld with reference density
            
            if (profile.loc[(profile['depth']>ref_depth),'density'].max() < (ref_dens+0.03)):
                mld = profile['depth'].max()
            
            profile['mld'] = mld
                        
            ### BBP PROCESSING ###
            # 1) Gross Filter Test
            high_bbp = len(profile.loc[(profile['bbp700'].notna())&((profile['bbp700']>0.03)|(profile['bbp700']<0)),'bbp700'])
            all_bbp = len(profile.loc[(profile['bbp700'].notna()),'bbp700'])
            filter_test = high_bbp>all_bbp*0.1
            if filter_test:
                profile['bbp700'] = np.nan
                if profile_reindex not in malfunc_biofoul_lst:
                    malfunc_biofoul_lst.append(profile_reindex)
                    malfunc_biofoul_time_lst.append(pd.to_datetime(profile.time.values[0]))
                                
            # 2) PARKING BIAS TEST
            if 'park_depth' in df.columns:
                parking_pres = profile.park_depth.values[0]
                max_profile_pres = profile.pressure.max()
                close_test = abs(max_profile_pres-parking_pres)<20 # within 20 dbar of parking depth
                
                if close_test:
                    bbp_limit = profile[profile.pressure>max_profile_pres-100].bbp700.median() + 0.0001
                    profile.loc[(profile.bbp700>bbp_limit)&(profile.pressure>max_profile_pres-20),'bbp700'] = np.nan
            
            # 3) SPIKE AND NOISE TEST
            if profile['depth'].max() > 200:
                res = profile.loc[(profile['depth']>200)&(profile['bbp700'].notna()),'bbp700']\
                    .sub(profile.loc[(profile['depth']>200)&(profile['bbp700'].notna()),'bbp700']\
                        .rolling(window = 7, center = True,  min_periods = 1).median())
                res_thres_test = res.abs()>0.0005
                
                spike_bbp = len(profile.loc[(profile['depth']>200)&(profile['bbp700'].notna()),'bbp700'][res_thres_test])
                all_bbp = len(profile.loc[(profile['depth']>200)&(profile['bbp700'].notna()),'bbp700'])
                if spike_bbp > (all_bbp*0.1):
                    profile['bbp700'] = np.nan
                    if profile_reindex not in malfunc_biofoul_lst:
                        malfunc_biofoul_lst.append(profile_reindex)
                        malfunc_biofoul_time_lst.append(pd.to_datetime(profile.time.values[0]))
                        
                profile.loc[(profile['depth']>200)&((res>0.0005)|(res<-0.0001))&(profile['bbp700'].notna()),'bbp700'] = np.nan # remove big spikes
            
            # 4) DEEP BBP TEST
            if profile['depth'].max()>500:
                deep_bbp_min_test = profile[(profile['depth']>500)].bbp700.min()>0.0006
                
                lower_deep_count = profile[(profile['depth']>500)&(profile['bbp700']<0)].bbp700.count()
                deep_count = profile[(profile['depth']>500)].bbp700.count()
                
                neg_bbp_test = (lower_deep_count>(deep_count*0.2))
                if neg_bbp_test or deep_bbp_min_test:
                    profile['bbp700'] = np.nan
                    if profile_reindex not in malfunc_biofoul_lst:
                        malfunc_biofoul_lst.append(profile_reindex)
                        malfunc_biofoul_time_lst.append(pd.to_datetime(profile.time.values[0]))
            
            # 5) APPLY GENERAL FILTER TO PROFILE: filter data for any remaining bad points
            profile.loc[(profile['bbp700']>0.03)|(profile['bbp700']<0),'bbp700'] = np.nan
            
            # 6) ZGAP AND SMOOTH
            profile.loc[(profile['bbp700'].notna()),'bbp700'] = profile.loc[(profile['bbp700'].notna()),'bbp700']\
                .rolling(window = 5, center = True,  min_periods = 1).median()\
                .rolling(window = 7, center = True, min_periods = 1).mean()
                
            
            ### CHLA PROCESSING ###
            # 1) Gross Filter Test: remove profile if too many value out are out of range
            high_chla = len(profile.loc[(profile['chla'].notna())&((profile['chla']>20)|(profile['chla']<-0.5)),'chla'])
            all_chla = len(profile.loc[(profile['chla'].notna()),'chla'])
            filter_test = high_chla>all_chla*0.1
            if filter_test:
                profile['chla'] = np.nan
                if profile_reindex not in malfunc_biofoul_lst:
                    malfunc_biofoul_lst.append(profile_reindex)
                    malfunc_biofoul_time_lst.append(pd.to_datetime(profile.time.values[0]))
            
            # 2) RESPONSE TEST: check the variability of the float, and ensure the surface is more more variable than at depth
            if profile.latitude.abs().mean() < 50:
                chla_mean = profile.loc[profile.depth<100,'chla'].median()
                chla_res = profile.loc[profile.depth<200,'chla'].quantile(0.9)-profile.loc[profile.depth<200,'chla'].quantile(0.1)
                chla_deep_res = profile.loc[profile.depth>300,'chla'].quantile(0.9)-profile.loc[profile.depth>300,'chla'].quantile(0.1)
                
                chla_res_test = (chla_res/chla_deep_res)<1.1
                if (chla_res_test) and (chla_mean < 0.05):
                    profile['chla'] = np.nan
                    if profile_reindex not in malfunc_biofoul_lst:
                        malfunc_biofoul_lst.append(profile_reindex)
                        malfunc_biofoul_time_lst.append(pd.to_datetime(profile.time.values[0]))

            # 3) SPIKE AND NOISE TEST: Test for spikes/noise; remove any spikes or noise; remove whole profile is it is too noisy/spikey.
            if profile['depth'].max() > 200:
                res = profile.loc[(profile['depth']>200)&(profile['chla'].notna()),'chla']\
                    .sub(profile.loc[(profile['depth']>200)&(profile['chla'].notna()),'chla']\
                        .rolling(window = 7, center = True,  min_periods = 1).median())
                res_thres_test = res.abs()>0.1
                
                spike_chla = len(profile.loc[(profile['depth']>200)&(profile['chla'].notna()),'chla'][res_thres_test])
                all_chla = len(profile.loc[(profile['depth']>200)&(profile['chla'].notna()),'chla'])
                if spike_chla > (all_chla*0.1):
                    profile['chla'] = np.nan
                    if profile_reindex not in malfunc_biofoul_lst:
                        malfunc_biofoul_lst.append(profile_reindex)
                        malfunc_biofoul_time_lst.append(pd.to_datetime(profile.time.values[0]))
                
                profile.loc[(profile['depth']>200)&((res>0.1)|(res<-0.1))&(profile['chla'].notna()),'chla'] = np.nan # remove big spikes
            
            # 4) DARK OFFSET CORRECTION: offset profile with dark values chlorphyll-a fluorescence
            if len(profile.chla.dropna())!=0:
                chl_offset = np.nan
                cm_ind = profile.loc[(profile['chla'].notna()),'chla'].\
                            rolling(window = 5, center = True,  min_periods = 1).median().\
                                rolling(window = 7, center = True, min_periods = 1).mean().argmax()
                cmd = profile['depth'].tolist()[cm_ind]
                
                if mld > cmd:
                    offset_ref_depth = mld
                if cmd > mld:
                    offset_ref_depth = cmd
                if cmd == mld:
                    offset_ref_depth = mld
                
                if (offset_ref_depth < profile['depth'].max()):
                    # Find Dark Offset (avoiding potential spikes)
                    chl_offset = profile.loc[(profile['depth'] > offset_ref_depth),'chla'].\
                                rolling(window = 7, center = True,  min_periods = 1).median().min()
                    
                    chl_offset_idx = profile.loc[(profile['depth'] > offset_ref_depth),'chla'].idxmin()
                    chl_offset_z = profile[profile.index==chl_offset_idx]['depth'].values[0]
                    
                    profile['chla'] = profile['chla'].sub(chl_offset) # Set offset and values below to zero by subtraction
                    profile.loc[(profile['chla'] <= 0)&(profile['depth'] >= chl_offset_z),'chla'] = 0
                    profile.loc[profile['depth'] >= chl_offset_z,'chla'] = 0
                    
                    chl_offset_old = chl_offset
                    
                # Offset couldnt be found; set to old one
                if (offset_ref_depth == profile['depth'].max()) and (pindex != 0):
                    if pd.isna(chl_offset_old) == False: 
                        chl_offset = chl_offset_old
                        profile['chla'] = profile['chla'].sub(chl_offset) # Set offset and values below to zero by subtraction
                    if pd.isna(chl_offset_old) == True: 
                        profile['chla'] = np.nan
                        
                # Any remaining negative values are set to 0 mg m-3
                profile.loc[(profile['chla'] < 0),'chla'] = 0
                
            # 5) MALFUNCTION OR BIO-FOUL CHLA TEST: if the offset is out of range, assume bio-fouling and discard profile
            if len(profile.chla.dropna())!=0:
                if (chl_offset > 0.1) or (chl_offset < -0.1):
                    profile['chla'] = np.nan
                    if profile_reindex not in malfunc_biofoul_lst:
                        malfunc_biofoul_lst.append(profile_reindex)
                        malfunc_biofoul_time_lst.append(pd.to_datetime(profile.time.values[0]))
            
            # 6) APPLY GENERAL FILTER TO CORRECTED PROFILE: filter data for any remaining bad points
            #profile.loc[(profile['depth']<10)&(profile['chla']<0),'chla'] = np.nan
            profile.loc[((profile['chla']>20)|(profile['chla']<0)),'chla'] = np.nan
            
            # 7) SMOOTH
            profile.loc[(profile['chla'].notna()),'chla'] = profile.loc[(profile['chla'].notna()),'chla']\
                .rolling(window = 5, center = True,  min_periods = 1).median()\
                .rolling(window = 7, center = True, min_periods = 1).mean()
            
            # 8) NPQ CORRECTION
            timez = eng.timezone(float(profile['longitude'].mean())) # Find timezone
            profile['local_time']\
                = profile['time'] - datetime.timedelta(hours=timez) # Convert from UTC to local time
            
            # Sunset and sunrise 
            sr, ss = eng.suncycle(float(profile['latitude'].round(2).mean()), 
                                  float(profile['longitude'].round(2).mean()), 
                                  eng.datenum(datetime.datetime.now().strftime("%d-%b-%Y")))[0] # Find sunset and  sunrise in UTC
            
            polarday = False
            if (sr == 0) and (ss == 24): # Polar day
                polarday = True
            if (sr == 24) and (ss == 0): # Polar night
                polarday = False
                
            daytime = False
            if (ss not in [0,24]) and (sr not in [0,24]): # Normal sunset and sunrise hours in utc/ ignore polar extremes
                if sr > 24:
                    sr = sr - 24
                
                sunrise_local = datetime.datetime(profile['time'].dt.year.tolist()[0],
                                                       profile['time'].dt.month.tolist()[0],
                                                       profile['time'].dt.day.tolist()[0],
                                                       int(sr),int((sr*60) % 60),int((sr*3600) % 60)) - datetime.timedelta(hours=timez)
                
                sunset_local = datetime.datetime(profile['time'].dt.year.tolist()[0],
                                                      profile['time'].dt.month.tolist()[0],
                                                      profile['time'].dt.day.tolist()[0],
                                                      int(ss),int((ss*60) % 60),int((ss*3600) % 60)) - datetime.timedelta(hours=timez)
                sr_dh = sunrise_local.hour + sunrise_local.minute/ 60
                ss_dh = sunset_local.hour + sunset_local.minute/ 60
                
                pr_dh = pd.to_datetime(profile['local_time'].values[0]).hour + pd.to_datetime(profile['local_time'].values[0]).minute / 60
                
                
                profile_hour = pd.to_datetime(profile['local_time'].values[0]).hour
                daytime = (sr_dh<pr_dh<ss_dh) # True if it is day time
                
            if (daytime == True) or (polarday == True):
                if len(profile.loc[(profile['depth'] < mld),'chla'].dropna()) != 0:
                    chl_max_mld = profile.loc[(profile['depth'] < mld),'chla'].max()
                    chl_max_mld_idx = profile.loc[(profile['depth'] < mld),'chla'].idxmax() # find max chl in mld
                    chl_max_mld_z = profile[profile.index==chl_max_mld_idx]['depth'].values[0]
                    
                    if chl_max_mld<=0:
                        profile['chla'] = np.nan
                    if chl_max_mld>0:
                        profile.loc[profile['depth'] <= chl_max_mld_z,'chla'] = chl_max_mld
                    
                if len(profile.loc[(profile['depth'] < mld),'chla'].dropna()) == 0:
                    profile['chla'] = np.nan
                   
            check_chla_model = ('ECO' in df['chla_model'].values[0]) or ('WETLABS' in df['chla_model'].values[0])\
                                or ('MCOMS' in df['chla_model'].values[0]) or ('FLBB' in df['chla_model'].values[0])
            if check_chla_model:
                profile['chla'] = profile['chla'].div(1) # no change
            if check_chla_model == False:
                sys.exit("Unknown Chla Model")
            ###
                        
            ### MEAN BINNING AND INTERPOLATION
            profile = profile[['profile_index','depth','pressure','temperature',
                               'longitude','latitude','salinity','mld','local_time','time','chla','bbp700','irrad490','par']]
            profile['depth_bins'] = profile['depth']
            bins = np.arange(min_depth,max_depth+bin_size,bin_size)
            profile_bin = profile.groupby([pd.cut(profile.depth_bins, bins)], 
                                          dropna = False).agg({'profile_index':'mean',
                                                               'pressure': 'mean',
                                                               'temperature': 'mean',
                                                               'salinity': 'mean',
                                                               'mld': 'first',
                                                               'chla': 'mean',
                                                               'bbp700': 'mean',
                                                               'irrad490': 'mean',
                                                               'par': 'mean',
                                                               'local_time': 'first',
                                                               'time': 'first',
                                                               'latitude': 'first',
                                                               'longitude': 'first'}).reset_index()
                                                                                       
            profile_bin['depth'] = profile_bin['depth_bins'].apply(lambda x: x.mid).tolist()

            for var in ['profile_index','latitude','longitude','time','local_time','mld']:
                profile_bin[var] = profile_bin[var].dropna().values[0]
            
            chla_bbp_chck = profile_bin[profile_bin.chla.notna()&profile_bin.bbp700.notna()]
            
            # Final ZGAP CHECK (There's interpolation, so there shouldnt be gaps at this point)
            zgap = np.nan
            zgap1 = chla_bbp_chck[(chla_bbp_chck.depth<300)].depth.sub(chla_bbp_chck[(chla_bbp_chck.depth<300)].depth.shift(1)).max()
            zgap2 = 300-chla_bbp_chck[(chla_bbp_chck.depth<300)].depth.max()
            
            if zgap1 > zgap2:
                zgap = zgap1
            if zgap2 > zgap1:
                zgap = zgap2
            
            zgap_test = zgap<100
                        
            if zgap_test:
                # Interpolate missing data with the except of time 
                for var in ['temperature','salinity','bbp700','chla']:
                    profile_bin[var] = profile_bin[var].interpolate(method='linear', limit_area = 'inside')
                    
                    if profile_bin['depth'].min() < mld:
                        profile_bin[var] = profile_bin[var].interpolate(method='linear', limit_area = 'outside',
                                                                        limit_direction = 'backward')
                profile_lst.append(profile_bin) #add profile to processing list if needed
    
        
            if profile_bin.chla.min()<0:
                print(pindex)
                sys.exit()
                
    if len(profile_lst)>0:
        df_binned = pd.concat(profile_lst)
        
        ### BIOFOULING REMOVAL
        if len(malfunc_biofoul_lst) != 0:
            lst = (np.roll(malfunc_biofoul_lst, -1) - malfunc_biofoul_lst)[:] #dont use last value
            res = []
            for k, g in groupby(lst):
                len_ = len(list(g))
                res += [len_ >= 5 and k == 1] * len_
            if True in res:
                first_consistent_bad_profile = list(compress(malfunc_biofoul_time_lst,res))[0]
                possibly_bad_start_date = first_consistent_bad_profile - datetime.timedelta(days=100)
                df_binned = df_binned[df_binned['time']<possibly_bad_start_date]
                
        #Only retain these variables
        var_potential = ['profile_index','depth','pressure','temperature','longitude',
                         'latitude','salinity','mld','local_time','time','chla','bbp700','irrad490','par']
        var_available = list(set(var_potential).intersection(df_binned.columns.tolist()))
        df_binned = df_binned[var_available]

        
    return df_binned

rename_dict = {'N_PROF': 'profile_index',
               'CYCLE_NUMBER': 'cycle_index',
               'JULD': 'time', 'datetime': 'time',
               'DIRECTION': 'direct',
               'LONGITUDE': 'longitude', 'lon': 'longitude', 
               'LATITUDE': 'latitude', 'lat': 'latitude',
               'p': 'pressure',
               't': 'temperature',
               's': 'salinity',
               'PRES': 'pressure',
               'TEMP': 'temperature',
               'PSAL': 'salinity',
               'PRES_ADJUSTED': 'pressure_adj',
               'TEMP_ADJUSTED': 'temperature_adj',
               'PSAL_ADJUSTED': 'salinity_adj',
               
               'BBP700': 'bbp700','BBP_700nm': 'bbp700','bbp': 'bbp700',
               'CHLA': 'chla','Chl': 'chla','fchl': 'chla',
               
               'DOWN_IRRADIANCE490': 'irrad490',
               'DOWNWELLING_PAR': 'par',
               
               'JULD_QC': 'time_qc',
               'POSITION_QC': 'latitude_qc',
               'PRES_QC': 'pressure_qc',
               'TEMP_QC': 'temperature_qc',
               'PSAL_QC': 'salinity_qc',
               'PRES_ADJUSTED_QC': 'pressure_adj_qc',
               'TEMP_ADJUSTED_QC': 'temperature_adj_qc',
               'PSAL_ADJUSTED_QC': 'salinity_adj_qc'}

#############################################################################
# Collect Float Data Files
#############################################################################
print('-----------------------------------------------------------------------')
print('Accessing Datasets')
print('-----------------------------------------------------------------------')

bgc_argo_filelst = filegrab(root_dir + 'bgc-argo database/bgc-argo program (feb 3 2024)',"_Sprof.nc",
                            root_dir)
griidc_argo_filelst = filegrab(root_dir  + '/bgc-argo database/GRIIDC APEX',"_Sprof.nc",
                               root_dir + '/bgc-argo database/GRIIDC APEX')
noaa_argo_filelst = filegrab(root_dir + '/bgc-argo database/NOAA APEX/',"_Sprof.nc",
                             root_dir + '/bgc-argo database/NOAA APEX')
naames_argo_filelst = filegrab(root_dir + '/bgc-argo database/NAAMES MISC',"_Sprof.nc",
                               root_dir + '/bgc-argo database/NAAMES MISC')

file_grab_lst = [] # Where are the files are located
file_grab_lst.extend(bgc_argo_filelst) # Add all files to single list
file_grab_lst.extend(griidc_argo_filelst) # Add all files to single list
file_grab_lst.extend(noaa_argo_filelst) # Add all files to single list
file_grab_lst.extend(naames_argo_filelst) # Add all files to single list

count_bio_optics = 0
for file_grabbed in file_grab_lst[:]:
    datafilename = file_grabbed.split('/')[-1].split('.')[0] + '_' + str(bin_size) + 'm_binned.csv'
    processed_file = '/Users/adamstoer/Synced Documents/projects/2021/project 2103/02 processed data/processed float data/' + datafilename
    
    print('\n\nFile (' + str(file_grab_lst.index(file_grabbed)+1) +\
          '/' + str(len(file_grab_lst)) + '):')
    print('-----------------------------------------------------------------------')
    
    # Check to make sure the file had not already been processed
    #if os.path.isfile(processed_file) is True:
    #    print('File ' + file_grabbed.split('/')[-1] + ' has been processed')
    # If the file has not already been processed, process it
    
    if 'Sprof' in file_grabbed:
        print('Processing ' + file_grabbed.split('/')[-1])
        ds_orig = xr.open_dataset(file_grabbed, decode_times=False) 
        pot_var = list(rename_dict.values()) + list(rename_dict.keys())
        var_available = list(ds_orig.variables)
        var_found = list(set(pot_var).intersection(var_available))
        ds = ds_orig[var_found]
        print("Variables Collected:")
        print(var_found)
        
        ds = ds.rename({k:rename_dict[k] for k in list(ds.variables) if k in rename_dict}) # Rename the variables
        print("Variables Renamed:") # Rename the variables
        print(list(ds.variables))
        
        if ('bbp700' in list(ds.variables)) and ('chla' in list(ds.variables)):
            count_bio_optics = count_bio_optics + 1
            print(count_bio_optics)

        print('Accessing Metadata')
        if 'bgc-argo program' in file_grabbed:
            meta_file = file_grabbed.split('Sprof.nc')[0] + 'meta.nc'
            meta_ds =  xr.open_dataset(meta_file, decode_times=False)
            sensor_mak_var = [s.strip().decode('utf-8') for s in meta_ds.SENSOR_MODEL.values]
            sensor_var = [s.strip().decode('utf-8') for s in meta_ds.SENSOR.values]
            
            if 'chla' in list(ds):
                if ('FLUOROMETER_CHLA' in sensor_var):
                    index_chla = sensor_var.index('FLUOROMETER_CHLA')
                    chla_model = meta_ds.SENSOR_MAKER.values[index_chla].strip().decode('utf-8') + '_' + meta_ds.SENSOR_MODEL.values[index_chla].strip().decode('utf-8')
            
            if 'bbp700' in list(ds):
                if ('BACKSCATTERINGMETER_BBP700' in sensor_var):
                    index_bbp700 = sensor_var.index('BACKSCATTERINGMETER_BBP700')
                    bbp700_model = meta_ds.SENSOR_MAKER.values[index_bbp700].strip().decode('utf-8') + '_' + meta_ds.SENSOR_MODEL.values[index_bbp700].strip().decode('utf-8')
                if ('SCATTEROMETER_BBP' in sensor_var):
                    index_bbp700 = sensor_var.index('SCATTEROMETER_BBP')
                    bbp700_model = meta_ds.SENSOR_MAKER.values[index_bbp700].strip().decode('utf-8') + '_' + meta_ds.SENSOR_MODEL.values[index_bbp700].strip().decode('utf-8')
            # 
            traj_file = file_grabbed.split('Sprof.nc')[0] + 'Rtraj.nc'
            if (os.path.isfile(traj_file) == True):
                traj_ds =  xr.open_dataset(traj_file, decode_times=False)

        # Is there any BGC data in this file?
        if (('bbp700' in ds.variables) and ('chla' in ds.variables)):   #only process profiles with bbp or chl   
            # Convert to pandas dataframe and reset index()
            df = ds.to_dataframe() 
            df = df.reset_index().rename(columns = {'N_PROF':'profile_index'})
            
            if 'irrad490' not in df.columns:
                df['irrad490'] = np.nan
            if 'par' not in df.columns:
                df['par'] = np.nan
            # Decode the quality flags (they have b' infront of them)
            if 'bgc-argo program' in file_grabbed:
                df['direct'] = [p.decode('utf-8') for p in df['direct'].tolist()]
                #df = df[df['direct']=='A']
                for pindex in df['profile_index'].unique():
                    par_lst = [d.decode('utf-8').strip() for d in ds_orig.PARAMETER.values[pindex][0]]
                    for var in ['TEMP','PSAL','PRES']:
                        if (var in list(ds_orig)) and (var in par_lst):
                            var2 = rename_dict[var].split('_')[0]
                            ind = par_lst.index(var)
                            data_mode = ds_orig.PARAMETER_DATA_MODE.values[pindex][ind].decode('utf-8')
                            df.loc[df['profile_index']==pindex, var2 + '_mode'] = data_mode
                for var in df.columns:
                    if var + '_qc' in df.columns: #only apply the qc data
                        qclst = []
                        for qc in df[var + '_qc']:
                            if pd.isnull(qc) == False:
                                qclst.append(int(qc.decode('utf-8')))
                            if pd.isnull(qc) == True:
                                qclst.append(qc)
                        df[var + '_qc'] = qclst
                        
                # Preferentially use adjusted data
                df.loc[df['pressure_mode'].isin(['A','D']),'pressure'] = df.loc[df['pressure_mode'].isin(['A','D']),'pressure_adj']
                df.loc[df['pressure_mode'].isin(['A','D']),'pressure_qc'] = df.loc[df['pressure_mode'].isin(['A','D']),'pressure_adj_qc']
                
                df.loc[df['temperature_mode'].isin(['A','D']),'temperature'] = df.loc[df['temperature_mode'].isin(['A','D']),'temperature_adj']
                df.loc[df['temperature_mode'].isin(['A','D']),'temperature_qc'] = df.loc[df['temperature_mode'].isin(['A','D']),'temperature_adj_qc']
                
                df.loc[df['salinity_mode'].isin(['A','D']),'salinity'] = df.loc[df['salinity_mode'].isin(['A','D']),'salinity']
                df.loc[df['salinity_mode'].isin(['A','D']),'salinity_qc'] = df.loc[df['salinity_mode'].isin(['A','D']),'salinity_adj_qc']
                
                df['chla_model'] = chla_model
                df['bbp700_model'] = bbp700_model
            
                if (os.path.isfile(traj_file) == True):
                    # RECORD PARKING DEPTH FOR BBP CORRECTION
                    ind_park_depth = [s.strip().decode('utf-8') for s in meta_ds.CONFIG_PARAMETER_NAME.values]
                    if 'CONFIG_ParkPressure_dbar' in ind_park_depth:
                        ind_park_depth = ind_park_depth.index('CONFIG_ParkPressure_dbar')
                        
                        park_depth_mnum = meta_ds.CONFIG_PARAMETER_VALUE.T[ind_park_depth].to_dataframe().reset_index()
                        
                        mnum_cyc = traj_ds.CONFIG_MISSION_NUMBER.to_dataframe().reset_index() # The mission numbers of each cycle
                        for cindex in df['cycle_index'].unique():
                            mnum = mnum_cyc.loc[mnum_cyc['N_CYCLE']==cindex,'CONFIG_MISSION_NUMBER']
                            if len(mnum)!=0:
                                mnum = mnum.values[0]
                                park_depth = park_depth_mnum.loc[park_depth_mnum['N_MISSIONS']==mnum,'CONFIG_PARAMETER_VALUE']
                                if len(park_depth)!=0:
                                    park_depth = park_depth.values[0]
                                    df.loc[df['cycle_index']==int(cindex),'park_depth'] = park_depth
            
            # Only use data with QC Flags of 1,2,5 or 8
            if ('bgc-argo program' in file_grabbed): #quality flags for these only available in main Argo DAC
                df['time'] = df['time'].where((df['time_qc'].isin([1,2,5,8])))
                df['time_qc'] = df['time_qc'].where((df['time_qc'].isin([1,2,5,8])))
                df['latitude'] = df['latitude'].where((df['latitude_qc'].isin([1,2,5,8])))
                df['longitude'] = df['longitude'].where((df['latitude_qc'].isin([1,2,5,8])))
                df['latitude_qc'] = df['latitude_qc'].where((df['latitude_qc'].isin([1,2,5,8])))
            if ('GRIIDC' in file_grabbed) or ('bgc-argo program' in file_grabbed):
                df['pressure'] = df['pressure'].where((df['pressure_qc'].isin([1,2,5,8])))
                df['pressure_qc'] = df['pressure_qc'].where((df['pressure_qc'].isin([1,2,5,8])))
                df['salinity'] = df['salinity'].where((df['salinity_qc'].isin([1,2,5,8])))
                df['salinity_qc'] = df['salinity_qc'].where((df['salinity_qc'].isin([1,2,5,8])))
                df['temperature'] = df['temperature'].where((df['temperature_qc'].isin([1,2,5,8])))
                df['temperature_qc'] = df['temperature_qc'].where((df['temperature_qc'].isin([1,2,5,8])))
        
            if ('bgc-argo program' not in file_grabbed):
                df['chla_model'] = 'ECO'
                df['bbp700_model'] = 'ECO'
                
            # Remove any data points where longitude, latitude, or time is unavailable.
            df = df[(df['time'].notna())&(df['longitude'].notna())&(df['latitude'].notna())]
            if 'NAAMES' not in file_grabbed:
                # Convert time variables to datetime
                t_unit = ds['time'].units
                print("Time unit conversion needed: (" + t_unit + ')')
                start = datetime.datetime(1950,1,1,0,0,0) # This is the "days since" part
                offset = [(start + datetime.timedelta(da)) for da in df['time']]
                df['time'] = offset
            if 'NAAMES' in file_grabbed:
                df['time'] = pd.to_datetime(df['time'])
                            
            # Is there any BGC data in this file?
            if ('latitude' in df.columns) and ('longitude' in df.columns) and\
                    ('time' in df.columns) and ('pressure' in df.columns):  
                if ('bbp700' in df.columns) and ('chla' in df.columns) and (len(df) != 0):  
                    
                    df = prep_df(df.reset_index(drop=True), file_grabbed) # Process the float data
                    
                    if (df is not np.nan):
                        if (len(df)!=0):
                            print('\nCreating Data Summary')
                            plot_data_summary(df, file_grabbed)
                            print('\nSaving Quality-Controlled Data')
                            df.to_csv(processed_file)
                            print('\nFile Processing Complete\n')
                        
                    
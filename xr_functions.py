# import packages
import os
import shutil
import glob
import json
import traceback
import gzip
import itertools
import math
from datetime import datetime, date
from pathlib import Path
import netCDF4 as nc
import xarray as xr
import cftime as cft
import numpy as np
import pandas as pd
from pandas import option_context
from reportlab.lib import utils
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib as matplotlib
import docx
import textwrap
import dask.config
from numcodecs import Blosc, Zlib, Zstd
from functools import partial
from cycler import cycler
import zipfile
from scipy.optimize import curve_fit
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_line,
    geom_errorbar,
    geom_smooth,
    geom_ribbon,
    stat_smooth,
    element_rect,
    xlim,
    ylim,
    scale_fill_manual,
    scale_color_manual,
    scale_color_identity,
    scale_x_continuous,
    scale_y_continuous,
    scale_x_datetime,
    labs,
    guides,
    guide_legend,
    theme,
    theme_bw,
    element_text,
    element_line,
    element_blank
)
from mizani.breaks import date_breaks
from mizani.formatters import date_format

###########################################################################################################################
# General
###########################################################################################################################

# check directory paths exist, if not error out
def dir_check(string):
    if Path(string).is_dir():
        return string
    else:
        raise NotADirectoryError(string)

# remove file if it already exists, otherwise skip
def rmv_file(string):
    try:
        Path(string).unlink()
    except OSError:
        pass

# remove entire directory
def rmv_dir(string):
    try:
        if Path(string).exists():
            if os.path.islink(string):
                os.unlink(string)
            else:
                shutil.rmtree(string) #ignore_errors=True)
    except Exception as error:
        print(error)
        pass

# read config file from sys.arg
def read_config(sys_argv_file):
    with open(sys_argv_file) as f:
        config = json.load(f)
    return config

# function to extract first position of all sublists within a list
def extract_sublist(lst):
    return [item[0] for item in lst]

# convert RH to SH; RH (%; 0-100), tair (kelvin), pressure (Pa)
def specific_humidity(rh, tair, pres):
    # constants
    Lv = 2.5*10**6  # latent heat of vaporization
    Rv = 461.0      # gas constant for water vapor
    T0 = 273.15     # Temperature offset for Kelvin scale
    es0 = 6.112     # saturation vapor pressure at 0 deg C
    # calculations borrowed from Jing
    pres_hPa = pres*0.01
    x = (Lv/Rv)*((1.0/T0)-1.0/tair)
    es = es0**x
    sh = 0.622*(rh/100.0)*es/pres_hPa
    return sh

###########################################################################################################################
# subset/resamp/concat output files
###########################################################################################################################

# collect netcdf file names in each given input directory,
# create raw and subset file names based on config.json
def xr_files(config):
    raw_file_info = []
    cat_file_info = []
    for f_dir in config['dir']['in']:
        # create list of netcdfs in directory
        files = sorted(glob.glob("{}*{}*.nc".format(f_dir,config['filename_glob_segment'])))
        # create output directory from input basename and output directory
        base_name = Path(f_dir).parents[config['parent_dir_num']].name
        # create directory for subset/compressed files
        Path(config['dir']['out'],base_name).mkdir(parents=True, exist_ok=True)
        # create list of all files
        files_for_cat = []
        # debug print - print file names to file
        with open(Path(config['dir']['out'], base_name,'debug_output.txt'),"a") as printfile:
            for line in files:
                printfile.write(line)                
        #loop through individual files
        for f in files:
            # create raw file info for subset/resample
            sub_file = Path(config['dir']['out'],'tmp', Path(f).stem + '_tempsub.nc')
            out_file = Path(config['dir']['out'], base_name, Path(f).name) 
            raw_file_info.append([f, sub_file, out_file])
            # create list of subset/resampled filenames (not paths) for xr_mfopendata concatenation
            cat_file_name = config['dir']['out'] + base_name + "/" + Path(f).name 
            files_for_cat.append(cat_file_name)
        # make cat file names
        cat_name = [Path(f_dir).parents[config['parent_dir_num']].name.split(config['filename_splitby_character'])[i] for i in config['filename_segement_select']]
        cat_file = Path(config['dir']['out'],'concat_files','_'.join(cat_name)+"_concat.nc")
        # add each line to subfile list
        cat_file_info.append([files_for_cat, cat_file])
    # return file info list
    return [raw_file_info, cat_file_info]  

# identify grids of certain parameters
def grid_info(config):
    # open file of 1901 simulation with PCT info
    init_file = Path(config['dir']['in'][0],config['initfile'])
    csv_dir = config['dir']['out']+'concat_files'
    # open netcdf file
    with xr.open_dataset(init_file, engine=config['nc_read']['engine']) as ds_tmp:
        ds = ds_tmp.load()
    # select grid cells - here we subset by ALTMAX less than 5 meters
    ds1 = ds[["ALTMAX","lat"]].sel(time=config['sim_start'])
    ds2 = ds1.to_dataframe()
    grids = ds2[(ds2.ALTMAX < 5) & (ds2.lat > 50)].index.get_level_values('lndgrid').unique()
    df = pd.DataFrame(grids)
    df.to_csv(Path(csv_dir,'grid_subset.csv'))
    # output lat lon by lndgrid
    sub_list = ["lon","lat"]
    ds[sub_list].to_dataframe().to_csv(Path(csv_dir,'grid_gps.csv'))
    sub_list = ["ALTMAX"]
    ds[sub_list].sel(time=config['sim_start']).to_dataframe().to_csv(Path(csv_dir,'grid_init_altmax.csv'))
    ds.close()
    # open surfdata for sand,silt,organic
    ds_sf =  xr.open_dataset(Path(config['surfdata']),\
                           engine=config['nc_read']['engine'])
    sub_list = ["PCT_SAND","PCT_CLAY"]
    ds_sf[sub_list].to_dataframe().to_csv(Path(csv_dir,'grid_texture.csv'))
    sub_list = ["ORGANIC"]
    ds_sf[sub_list].to_dataframe().to_csv(Path(csv_dir,'grid_organic.csv'))
    sub_list = ["PCT_NAT_PFT"]
    ds_sf[sub_list].to_dataframe().to_csv(Path(csv_dir,'grid_pfts.csv'))
    return grids

# copy file from slow disk archive to scratch for processing
def copy_subset_resamp(raw_f, config):
    # start subsetting netcdfs
    xr_subset(raw_f,config)

# open, subset, save new netcdf files with/without compression
def xr_subset(raw_f, config):
    src_file = raw_f[0]
    sub_file = raw_f[1]
    # remove subset netcdf output file if it already exists
    rmv_file(sub_file)
    # call xarray to subset/compress individual file
    with xr.open_dataset(src_file, engine=config['nc_read']['engine']) as ds_tmp:
        ds = ds_tmp.load()
    # set netcdf write characteristics for xarray.to_netcdf()
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=config['nc_write']['fillvalue'])
    # encoding
    encoding = {var: comp for var in ds[config['var_sub']].data_vars}
    # write netcdf
    ds[config['var_sub']].to_netcdf(sub_file, mode="w", encoding=encoding, \
                                      format=config['nc_write']['format'],\
                                      engine=config['nc_write']['engine'])
    # remove temp file after copying
    rmv_file(tmp_file)
    # call xr_resample when subset complete
    xr_resample(raw_f, config)

# open, resample, save new netcdf files with/without compression
def xr_resample(raw_f, config):
    sub_file = raw_f[1]
    out_file = raw_f[2]
    # remove subset netcdf output file if it already exists
    rmv_file(out_file)
    # call xarray to subset/compress individual file
    with xr.open_dataset(sub_file, engine=config['nc_read']['engine']) as ds_tmp:
        ds = ds_tmp.load()
    # set netcdf write characteristics for xarray.to_netcdf()
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=config['nc_write']['fillvalue'])
    # encoding
    encoding = {var: comp for var in ds.data_vars}
    # write netcdf
    ds.resample(time='1D').mean().to_netcdf(out_file, mode="w", encoding=encoding, \
                    format=config['nc_write']['format'],\
                    engine=config['nc_write']['engine'])
    # remove temp file after copying
    rmv_file(sub_file)

# open, concat, save new netcdf files with/without compression
def xr_concat(sub_f ,config):
    file_list = sub_f[0]
    cat_file = sub_f[1]
    # remove subset netcdf output file if it already exists
    rmv_file(cat_file)
    # call xarray to subset/compress individual file
    with xr.open_mfdataset(file_list, parallel=True, engine=config['nc_read']['engine']) as ds_tmp:
        ds = ds_tmp.load()
    # set netcdf write characteristics for xarray.to_netcdf()
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=config['nc_write']['fillvalue'])
    # encoding
    encoding = {var: comp for var in ds.data_vars}
    # write netcdf
    ds.to_netcdf(cat_file, mode="w", encoding=encoding, \
                    format=config['nc_write']['format'],\
                    engine=config['nc_write']['engine'])

###########################################################################################################################
# Adjust surface datasets
###########################################################################################################################

# functions to be applied using xarray.apply_ufunc in xr_adjust_surf()
# the apply_ufunc accesses each pft vector along lsmlat/lsmlon (each cell)
def pft_notree_adjust(da): 
    # copy 1D array of length 15 for each pft (ds is read-only when passed in as a numpy.ndarray) 
    da_tmp = da.copy()
    # define vectors for subsetting natpft vector
    tree = [1,2,3,4,5,6,7,8]
    bsg = [0,11,12]
    # calculate tree percentage sum, bareground/actic shrub/arctic grass (bsg) percentage sums 
    tree_sum = da[tree].sum()
    bsg_sum = da[bsg].sum()
    # divide tree sum by three, add to bsg, round to 1 decimal place
    third = tree_sum // 3
    da_tmp[0]  = round(da[0]  + third,1)
    da_tmp[11] = round(da[11] + third,1)
    da_tmp[12] = round(da[12] + third,1)
    # set all tree values to zero
    da_tmp[tree] = 0.0
    # check for 100 percent, if not subtract difference from bare ground
    if (da_tmp.sum() == 100):
        return da_tmp
    elif (da_tmp.sum() > 100.0):
        da_tmp[0] = da_tmp[0] - abs(da_tmp.sum() - 100.0)
        return da_tmp
    elif (da_tmp.sum() < 100.0):
        da_tmp[0] = da_tmp[0] + abs(da_tmp.sum() - 100.0)
        return da_tmp

def pft_c3arcticgrass_adjust(da): 
    # copy 1D array of length 15 for CLM5 pfts (ds is read-only when passed in as a numpy.ndarray) 
    da_tmp = da.copy()
    # list of PFT positions that are not C3 arctic grass to make zero
    other_pfts = [0,1,2,3,4,5,6,7,8,9,10,11,13,14]
    # set all other pfts to zero                
    da_tmp[other_pfts] = 0.0
    # set to all arctic grass PFT
    da_tmp[12] = 100.0
    # return adjusted PFT array
    return da_tmp

def natveg_adjust(da):
    # change all netveg cell percentages to 100%
    da = 100.0 # when no other dimension present ds is simply numpy.float64 reassigned without brackets 
    return da

def urban_adjust(da):
    # copy 1D array of length three containing percentage for each urban level
    da_tmp = da.copy()
    # create list of position in array to assign zeros
    urban_levels = [0,1,2]
    # change all urban percentages to zero
    da_tmp[urban_levels] = 0.0
    return da_tmp

def zero_adjust(da):
    # change all cell values to 0% for other PCT_
    da = 0.0 # when no other dimension present ds is simple numpy.float64 reassigned without brackets
    return da

# adjust surface data using xarray.apply_ufunc
def xr_adjust_surf(f, config):
    # parse model name and file name from xr_surf.py dask client.submit call
    model_name = f[0]
    file_name = f[1]
    # create file paths from json surfdata
    in_file = Path(config['dir']['in'], file_name)
    file_stem = Path(file_name).stem
    out_file = Path(config['dir']['out'], file_stem + config['file_ending'] + ".nc")
    # open netcdf surface dataset file
    with xr.open_dataset(in_file, engine=config['nc_read']['engine']) as ds_tmp:
        ds = ds_tmp.load()
    # use xarray.apply_ufunc to acces/edit each cell value (or cell array if another dimension like pft is present)
    try:                                        # try statement to capture errors during application of xarray.apply_ufunc
        for i in config['vars'][model_name]:    # loop through adjusted vars from json config file
            var = i[0]                          # first position from json file is variable name
            adj_func = globals().get(i[1])      # second position from json file is the function name to grab from globals
            core_dim = i[2]                     # third position from json file is a list of core dimensions
            var_attr = ds[var].attrs            # make copy of dataarray's attributes
            # apply adj_func using xarray.apply_ufunc
            ds[var] = xr.apply_ufunc(           # replace netcdf variable with adjusted variable
                adj_func,                       # function to apply across cells
                ds[var],                        # dataset to vectorize
                input_core_dims=[core_dim],     # core dim of analysis, empty if no 3rd dimenion of data within cell
                output_core_dims=[core_dim],    # returned dim is same as input
                vectorize=True,                 # loop over other dimensions of multi-dimensional array (here lat/lon)
            )
            if core_dim:
                ds[var] = ds[var].transpose(core_dim[0], ...) # return order if core_dim was used and subsequently placed first
            for i in var_attr:
                ds[var].attrs[i] = var_attr[i]  # restore lost attributes from apply_ufunc
    except Exception:                           # write errors to output text file if they occur
        with open(Path(config['dir']['out'], file_stem + config['file_ending'] + '_output.txt'), 'a') as f:
            traceback.print_exc(file=f)
    # set netcdf write characteristics for xarray.to_netcdf()
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'], _FillValue=None) #config['nc_write']['fillvalue'])
    # set encoding for each variable
    encoding = {var: comp for var in ds.data_vars}
    # write netcdf
    ds.to_netcdf(out_file, mode="w", encoding=encoding, \
                    format=config['nc_write']['format'],\
                    engine=config['nc_write']['engine'])

###########################################################################################################################
# unzip .nc.gz and .zip files 
###########################################################################################################################

# create list of files to unzip that is passed to dask cluster
def gunzip_file_list(config):
    # define empty list to hold file paths
    file_info = []
    # loop through all directories listed in config_unzip
    for f_dir in config['dir']['in']:
        # create list of netcdfs in each directory with .gz ending
        files = sorted(glob.glob("{}*.gz".format(f_dir)))
        # find name of directory
        base_name = Path(f_dir).name
        # create directory for unzipped files
        Path(config['dir']['out'],base_name).mkdir(parents=True, exist_ok=True)
        #loop through individual files
        for f in files:
            # create tmp file name
            tmp_file = Path(config['dir']['out'],'temporary_files', Path(f).name)
            # create unziped file name
            unzipped_file = Path(config['dir']['out'], base_name, Path(f).stem) 
            # update unzipped file name to remove odd naming from CEDA for 2010-2021 in all temp folders
            unzipped_file = Path(str(unzipped_file).replace('crujra.v2.3.1.','crujra.v2.3.'))
            # update new dswrf to correct file name
            unzipped_file = Path(str(unzipped_file).replace('crujra.v2.4.','crujra.v2.3.'))
            # append file into
            file_info.append([f, tmp_file, unzipped_file])
    # return file info list
    return file_info  

# create list of files to unzip that is passed to dask cluster
def gunzip_list(config):
    # define empty list to hold file paths
    file_info = []
    # loop through all directories listed in config_unzip
    for f_dir in config['dir']['in']:
        # create list of netcdfs in each directory with .gz ending
        files = sorted(glob.glob("{}*.gz".format(f_dir)))
        #loop through individual files
        for f in files: 
            # append file into
            file_info.append([f, config['dir']['out']])
    # return file info list
    return file_info  

# copy to tmp from project folder and unzip to scratch
def ungunzip_file(f, config):
    # pull file, tmp file, and unzipped file paths from input
    src_file = f[0]
    unzip_dir = f[1]
    unzip_file = Path(unzip_dir+str(Path(src_file).stem))
    # unzip temp file to final scratch destination
    with gzip.open(src_file, 'rb') as f_in:
        with open(unzip_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# copy to tmp from project folder and unzip to scratch
def copy_gunzip(f, config):
    # pull file, tmp file, and unzipped file paths from input
    src_file = f[0]
    tmp_file = f[1]
    unzip_file = f[2]
    # copy from slow project disk to scratch
    shutil.copy(src_file, tmp_file)
    # unzip temp file to final scratch destination
    with gzip.open(tmp_file, 'rb') as f_in:
        with open(unzip_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
        for line in file_info:
            f.write(line + '\n')

def unzip_dir_list(config):
    # define empty list to hold file paths
    file_info = []
    # loop through all directories listed in config_unzip
    for f_dir in config['dir']['in']:
        # create list of netcdfs in each directory with .gz ending
        files = sorted(glob.glob("{}*.zip".format(f_dir)))
        #loop through individual files
        for f in files:
            # check pull filename from globbed path to use as directory for unzipped files
            if '1a' in f:
                dir_name = 'Baseline_1901-2000'
            elif '1b' in f:
                dir_name = 'Baseline_2000-2021'
            elif '1c' in f:
                dir_name = 'OTC'
            elif '1d' in f:
                dir_name = 'Snow_fence'
            else:
                dir_name = Path(f).stem
            # create unziped file location
            unzipped_dir = Path(config['dir']['out']+dir_name) 
            # append to list of zip files to unzip
            file_info.append([f, unzipped_dir])
    # return file info list
    return file_info

def unzip_dir(f, config):
    # pull zipped file and destination file paths from input
    src_zip = f[0]
    unzip_dir = f[1]
    # make directory for output files
    Path(unzip_dir).mkdir(parents=True, exist_ok=True)
    # process each individual file within zipped namelist
    with zipfile.ZipFile(src_zip) as zipped_file:
        for member in zipped_file.namelist():
            filename = os.path.basename(member)
            # skip directories
            if not filename:
                continue
            # copy file (taken from zipfile's extract)
            source = zipped_file.open(member)
            target = open(os.path.join(unzip_dir, filename), "wb")
            with source, target:
                shutil.copyfileobj(source, target)
 
###########################################################################################################################
# Climate Biascorrection
###########################################################################################################################

# create full list of crujra reanalysis files (clm interpolated during transient simulation, all variables, by year)
def full_reanalysis_list(config_file):
    # load site's config information
    config = read_config(config_file)
    # remove previous copy of crujra folder
    rmv_dir(config['site_dir'])
    # remake directory for subset files
    Path(config['site_dir']+"sub/").mkdir(parents=True, exist_ok=True)
    # debug print the config file info
    with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
        for line in config:
            f.write(line + ': ' + str(config[line]) + '\n')
    # read all CRUJRA input file names from reanalysis directory
    reanalysis_files =  sorted(glob.glob("{}*.nc".format(config['reanalysis_dir'])))
    # loop through files in reanalysis archive linking config, in, out files
    file_info = []
    for line in reanalysis_files: 
        in_file = line 
        sub_file = config['site_dir'] + "sub/" + str(Path(line).name)
        file_info.append([config_file, in_file, sub_file])
    return file_info

# subset original dswrf CRUJRAv2.3_updated files (to check on diurnal phase issue)
# CRUJRA uses UTC 0 time (though not labeled or described anywhere I can find)
# this means the phase of the diurnal cycle is off by the GMT offset from the prime meridian
def raw_dswrf_list(config_file):
    # load site's config information
    config = read_config(config_file)
    # remake directory for subset files
    Path(config['site_dir']+"raw_dswrf_sub/").mkdir(parents=True, exist_ok=True)
    # read all CRUJRA input file names from reanalysis directory
    reanalysis_files =  sorted(glob.glob("{}*.nc".format('/scratch/jw2636/wrpmip/CRUJRAv2.3_unzipped/data/dswrf_updated/')))
    # loop through files in reanalysis archive
    file_info = []
    for line in reanalysis_files: 
        in_file = line 
        sub_file = config['site_dir'] + "raw_dswrf_sub/" + str(Path(line).name)
        file_info.append([config_file, in_file, sub_file])
    return file_info

# read in raw dswrf data, scale, apply mask, subsample to check on time issue 
def subset_raw_dswrf(f):
    # parse input strings
    config = read_config(f[0])
    f_in = Path(f[1])
    f_sub = Path(f[2])
    # open half degree CLM domain file for the landmask
    with xr.open_dataset(Path(config['clm_landmask']), engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_mask = ds_tmp.load()
    # rename dimensions
    ds_mask = ds_mask.rename_dims({'nj': 'lat', 'ni': 'lon'})
    # move and rotate lon data from 'xc' dataarray into dataset coordinate
    ds_mask = ds_mask.assign_coords({'lon': (((ds_mask.xc.values[0] + 180) % 360) - 180)})
    # move and subset lat data from 'yc' dataarray into dataset coordinate
    ds_mask = ds_mask.assign_coords({'lat': extract_sublist(ds_mask.yc.values)})
    # sort the lon coornidate into acending order for xarray to work
    ds_mask = ds_mask.sortby('lon')
    # change land mask of 1/0s into boolean for subsetting
    ds_mask['mask'] = ds_mask['mask'].astype(bool)
    # open crujra_dswrf file to apply mask 
    with xr.open_dataset(f_in, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds = ds_tmp.load() 
    # convert dswrf from j/m2 to w/m2 by dividing by the seconds in 6hrs
    ds['dswrf'] = ds['dswrf']/21600
    # use where and drop=True to remove masked (ocean) cells 
    ds = ds.where(ds_mask['mask'], drop=True)
    # subset nearest point based on lat/lon coordinate index
    ds = ds.sel(lon=config['lon'], lat=config['lat'], method='nearest')
    ## set encoding for netcdfs
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'], _FillValue=None) #config['nc_write']['fillvalue'])
    # set encoding for each variable
    encoding = {var: comp for var in ds.data_vars}
    # save netcdf to file
    ds.to_netcdf(f_sub, mode="w", encoding=encoding, \
                    format=config['nc_write']['format'],\
                    engine=config['nc_write']['engine'])

# glob subset raw dswrf files into timeseries to plot against obs and clm output crurja data
# this showed that both the clm output and raw dswrf files were offset by GMT hours from UTC 0
def concat_raw_dswrf(config_file):
    # read config file
    config = read_config(config_file)
    # create name for full crujra site file
    cru_file = Path(config['site_dir'] + "dswrf_" + config['site_name'] + "_dat.nc")
    # glob all files in raw folder
    f = sorted(glob.glob("{}*.nc".format(config['site_dir'] + 'raw_dswrf_sub/')))
    # read in all files
    with xr.open_mfdataset(f, parallel=True, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds = ds_tmp.load()
    ## set netcdf write characteristics for xarray.to_netcdf()
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
    # encoding
    encoding = {var: comp for var in ds.data_vars}
    # write netcdf
    ds.to_netcdf(cru_file, mode="w", encoding=encoding, \
                    format=config['nc_write']['format'],\
                    engine=config['nc_write']['engine'])

# copy crujra file (clm hourly output), subset to only climate forcing and select site grid cell
# also shift crujra time index to GMT time at site so that 1901-01-01 00:00:00 actually means midnight
def subset_reanalysis(f):
    # parse input strings
    config = read_config(f[0])
    f_in = Path(f[1])
    f_sub = Path(f[2])
    # open netcdf file
    with xr.open_dataset(f_in, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds = ds_tmp.load() 
    # list of variables to keep
    keep = ['FSDS','FLDS','PBOT','RAIN','SNOW','QBOT','TBOT','WIND']
    # pull grid form config file
    grid = config['lndgrid'] - 1
    # subset to variables of interest and grid of interest for each site 
    ds = ds[keep].sel(lndgrid=grid)
    # shift the index by offset from GMT time at location 
    ds.coords['time'] = ds.indexes['time'].round('H').shift(config['cru_GMT_adj'], 'H')
    ## set encoding for netcdfs
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'], _FillValue=None) #config['nc_write']['fillvalue'])
    # set encoding for each variable
    encoding = {var: comp for var in ds[keep].data_vars}
    # write netcdf
    ds.to_netcdf(f_sub, mode="w", encoding=encoding, \
                    format=config['nc_write']['format'],\
                    engine=config['nc_write']['engine'])

# glob files for site file of all years of subsampled/time shifted cru climate
# function is applied while makeing the full file list below that is reduced to only years that align with data
def concat_crujra(f, config):
    # create name for full crujra site file
    cru_file = Path(config['site_dir'] + "CRUJRA_" + config['site_name'] + "_allyears.nc")
    # read in all files
    with xr.open_mfdataset(f, parallel=True, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds = ds_tmp.load()
    ## set netcdf write characteristics for xarray.to_netcdf()
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
    # encoding
    encoding = {var: comp for var in ds.data_vars}
    # write netcdf
    ds.to_netcdf(cru_file, mode="w", encoding=encoding, \
                    format=config['nc_write']['format'],\
                    engine=config['nc_write']['engine'])

# create list of cru climate files/years that align with observational data to compare
def cru_sitesubset_list(config_file):
    # load site's config information
    config = read_config(config_file)
    # list of files in sub folder
    file_list = sorted(glob.glob("{}*.nc".format(config['site_dir']+"sub/")))
    # call function to concat all crujra files subset to site's gridcell
    concat_crujra(file_list, config)
    # create list of years present from file names
    year_list = []
    for item in file_list:
        year_list.append(item.split('.')[-2].split('-')[0])
    # subset year list to years where data exists
    year_bool = [((int(i) >= config['year_start'])&(int(i) <= config['year_end'])) for i in year_list]
    year_list = list(itertools.compress(year_list, year_bool))
    # loop through files in reanalysis folder for data years
    file_info = []
    for year in year_list:
        in_file = glob.glob("{}*{}*.nc".format(config['site_dir']+"sub/", year))[0]
        file_info.append(in_file)
    return [config_file, file_info]

# concatenate cru climate years that align with obs data using list created above
def concat_cru_sitesubset(f):
    # read in config file for site
    config = read_config(f[0])
    cat_file = config['site_dir'] + "CRUJRA_" + config['site_name'] + "_dat.nc"
    # open netcdf file
    with xr.open_mfdataset(f[1], parallel=True, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds = ds_tmp.load()
    ## set netcdf write characteristics for xarray.to_netcdf()
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
    # encoding
    encoding = {var: comp for var in ds.data_vars}
    # write netcdf
    ds.to_netcdf(cat_file, mode="w", encoding=encoding,\
                    format=config['nc_write']['format'],\
                    engine=config['nc_write']['engine'])

# read in observational data, adjust all aspects of obs data as needed per site, output to netcdf
def combine_site_observations(config_file):
    # read in config file for site
    config = read_config(config_file)
    try:
        # read in columns of interest from first file
        obs_data = pd.read_csv(config['obs']['f1']['name'], sep=config['obs']['f1']['sep'], \
            index_col=False, engine='python', skiprows=config['obs']['f1']['skip_rows'], \
            usecols=config['obs']['f1']['cols_old'])
    except:
        # if no data files listed end function
        print_string = 'No observation data; combine_site_observations() skipped for ' + config['site_name']
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print(print_string)
        return 
    # enforce column order from subset procedure from usecols in read_csv
    obs_data = obs_data[config['obs']['f1']['cols_old']]
    # rename data columns to CLM standard
    obs_data = obs_data.rename(columns=config['obs']['f1']['cols_new'])
    # remove rows with no date/time of measurement
    obs_data = obs_data.dropna(subset=config['obs']['f1']['datetime_cols'])
    # print statement code to use when testing/adding new sites 
    with option_context('display.max_rows', 10, 'display.max_columns', 10):
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print(obs_data.head(), file=f)
            print(obs_data.dtypes, file=f)
    # handle site specific idiosyncrasies
    site = config['site_name']
    match site:
        case 'USA-EightMileLake':
            # convert numerical timestamp to string for datetime.strptime
            obs_data['time'] = obs_data['TIMESTAMP_START'].astype(str)
            # change -9999 fill values to NAs
            obs_data = obs_data.replace(-9999, np.NaN)
            # convert TBOT from celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # make all negaive values in solar radiation zero
            obs_data.loc[obs_data['FSDS'] < 0.0, 'FSDS'] = 0.0
            obs_data.loc[obs_data['FLDS'] < 0.0, 'FLDS'] = 0.0
            # Convert from kPa to Pa
            obs_data.loc[:,'PBOT'] = obs_data['PBOT'] * 1000
            # convert RH to SH
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            obs_data = obs_data.drop(columns=['RH'])
            # WS
            # precip
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print(obs_data, file=f)

        case 'USA-Toolik':
            # fix hourly timestep - cannot have 24 as hour value, only 0:23 for datetime.strptime
            obs_data.loc[:,'hour'] = obs_data['hour'] - 100
            obs_data['hour'] = obs_data['hour'].astype(int)
            # combine date and hour columns for timestamp -  need to pad hours with preceeding zeros
            obs_data['time'] = obs_data['date'].astype(str) + " " + obs_data['hour'].astype(str).str.zfill(4)
            with option_context('display.max_rows', 10, 'display.max_columns', 10):
                with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                    print(obs_data.head(), file=f)
                    print(obs_data.dtypes, file=f)
            # convert TBOT from celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # make all negaive values in solar radiation zero
            obs_data.loc[obs_data['FSDS'] < 0.0, 'FSDS'] = 0.0
            obs_data.loc[obs_data['FLDS'] < 0.0, 'FLDS'] = 0.0
            # change mbar -> Pa
            obs_data.loc[:,'PBOT'] = obs_data['PBOT'] * 100
            obs_data.loc[obs_data['PBOT'] < 90000, 'PBOT'] = np.NaN 
            # convert RH to SH
            obs_data['QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT']) 
            obs_data = obs_data.drop(columns=['RH'])
            # WS
            # precip
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print(obs_data, file=f)
        
        case 'SWE-Abisko':
            # abiskos data is very messy and has all kinds of non-numeric character strings which confuses python
            # python then turns all columns into objects (strings) which breaks all the math code
            # to fix this I have to force all columns to numeric which makes all non-numbers into NaNs
            cols_to_num = ['TBOT','FSDS','FLDS','PBOT','RH','WIND']
            for col in cols_to_num:
                obs_data.loc[:,col] = pd.to_numeric(obs_data[col], errors='coerce')
            # convert numerical timestamp to string for datetime.strptime
            obs_data['time'] = obs_data['Timestamp (UTC)'].astype(str)
            # change -6999 fill values to NAs
            obs_data = obs_data.replace(-6999, np.NaN)
            # convert TBOT from celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # make all negaive values in solar radiation zero
            obs_data.loc[obs_data['FSDS'] < 0.0, 'FSDS'] = 0.0
            obs_data.loc[obs_data['FLDS'] < 0.0, 'FLDS'] = 0.0
            obs_data.loc[obs_data['FLDS'] < 50,  'FLDS'] = np.NaN 
            # Convert from mbar to Pa
            obs_data.loc[:,'PBOT'] = obs_data['PBOT'] * 100
            obs_data.loc[obs_data['PBOT'] < 95000, 'PBOT'] = np.NaN 
            # convert RH to SH
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            obs_data = obs_data.drop(columns=['RH'])
            # WS
            # precip:
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print(obs_data, file=f)
        case 'RUS-Seida':
            # make time column from datetime string
            obs_data['time'] = obs_data['datetimeGMT3'].astype(str) # + " " + obs_data['hourGMT3'].astype(str).str.zfill(2)
            # drop old time columns
            obs_data = obs_data.drop(columns=['datetimeGMT3'])
            # set to datetime pandas
            obs_data.loc[:,'time'] = pd.to_datetime(obs_data['time'], format="%Y-%m-%d %H:%M")
            # set index to timestamp
            obs_data = obs_data.set_index('time')
            # resample the sub-hourly data to hourly averages
            obs_data = obs_data.resample('1H').mean()
            # change index to str with correct format
            obs_data.index = obs_data.index.strftime('%Y-%m-%d %H:%M:%S')
            # convert numerical timestamp to string for datetime.strptime to integrate back into set coding below
            obs_data = obs_data.reset_index()
            obs_data['time'] = obs_data['time'].astype(str)
            # convert TBOT from celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # only given par - will scale up based on a par = 0.46 shortwave (shortwave = par/0.46)
            obs_data.loc[:,'FSDS'] = (obs_data['PAR']/4.57)/0.46
            obs_data = obs_data.drop(columns = ['PAR'])
            # convert rh to sh
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            obs_data = obs_data.drop(columns=['RH'])
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print(obs_data, file=f)
            # pressure, windspeed seem to be in same units as clm reprocessed crujra
        case 'CAN-DaringLake':
            # read in second dataset
            obs_data2 = pd.read_csv(config['obs']['f2']['name'], sep=config['obs']['f2']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f2']['skip_rows'], \
                usecols=config['obs']['f2']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            obs_data2 = obs_data2[config['obs']['f2']['cols_old']]
            # rename data columns to CLM standard
            obs_data2 = obs_data2.rename(columns=config['obs']['f2']['cols_new'])
            # remove rows with no date/time of measurement
            obs_data2 = obs_data2.dropna(subset=config['obs']['f2']['datetime_cols'])
            # concat pandas dataframes
            obs_data = pd.concat([obs_data, obs_data2], ignore_index=True)
            # convert numerical timestamp to string for datetime.strptime
            obs_data.loc[:,'Hour'] = obs_data['Hour'].astype(int) - 100
            obs_data.loc[:,'Date'] = obs_data['Year'].astype(str) + "-" + obs_data['Month'].astype(str) + "-" + obs_data['Day'].astype(str)
            obs_data.loc[:,'time'] = obs_data['Date'].astype(str) + " " + obs_data['Hour'].astype(str).str.zfill(4)
            obs_data = obs_data.drop(columns = ['Date'])
            # convert TBOT from celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # make all negaive values in solar radiation zero
            obs_data.loc[obs_data['FSDS'] < 0.0, 'FSDS'] = 0.0
            # No pressure given so I'll calculate air pressure given a rough reference (400m ~ 96357Pa) and plug in sites TBOT
            # into barometric pressure equation to estimate the pressure a few meters up at site elevation (424m)
            # basically adding temperature variability to reference pressure through barometric pressure function
            # I'm only doing this because I need pressure to convert RH to SH
            g0 = 9.80665 # gavitational constat in m/s2
            M0 = 0.0289644 # molar mass of air kg/mol
            R0 = 8.3144598 # universal gas constant - J/(mol K)
            hb = 0 # reference level, here just below sites elevation
            Pb = 101325 # estimated reference pressure at 400 meters and 0 degre C 
            obs_data.loc[:,'PBOT'] = Pb*np.exp((-g0*M0*(424-hb))/(R0*obs_data['TBOT']))
            obs_data.loc[obs_data['PBOT'] < 90000, 'PBOT'] = np.NaN 
            # convert RH to SH
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            obs_data = obs_data.drop(columns=['RH'])
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print(obs_data, file=f)
            # WS
            # precip
        case 'USA-Utqiagvik':
            # subset to BD for Barrow in strSitCom
            obs_data = obs_data.loc[obs_data['SITE'] == 'BD']
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print(obs_data, file=f)
            # read other files and concat to first
            for extra_file in config['obs']['f1']['extended_files']:
                obs_data2 = pd.read_csv(extra_file, index_col=False, engine='python', skiprows=config['obs']['f1']['skip_rows'], \
                    usecols=config['obs']['f1']['cols_old'], sep=config['obs']['f1']['sep'])
                # enforce column order from subset procedure from usecols in read_csv
                obs_data2 = obs_data2[config['obs']['f1']['cols_old']]
                # rename data columns to CLM standard
                obs_data2 = obs_data2.rename(columns=config['obs']['f1']['cols_new'])
                # remove rows with no date/time of measurement
                obs_data2 = obs_data2.dropna(subset=config['obs']['f1']['datetime_cols'])
                # select Barrow site
                obs_data2 = obs_data2.loc[obs_data2['SITE'] == 'BD']
                # concat file
                obs_data = pd.concat([obs_data, obs_data2], ignore_index=True)
            # drop site column
            obs_data = obs_data.drop(columns = ['SITE'])
            # set time from timestamp
            obs_data['time'] = obs_data['strAlaska'].astype(str)
            # change all -999.9 values to NaN
            obs_data = obs_data.replace(-999.9, np.NaN)
            # scale celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # calculate SWIN from PAR
            obs_data.loc[:,'FSDS'] = obs_data['PAR']/2.1#4.57)/0.46
            obs_data = obs_data.drop(columns = ['PAR'])
            #WS
            # precip
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print(obs_data, file=f)
        case 'USA-Atqasuk':
            # subset to BD for Barrow in strSitCom
            obs_data = obs_data.loc[obs_data['SITE'] == 'AD']
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print(obs_data, file=f)
            # read other files and concat to first
            for extra_file in config['obs']['f1']['extended_files']:
                obs_data2 = pd.read_csv(extra_file, index_col=False, engine='python', skiprows=config['obs']['f1']['skip_rows'], \
                    usecols=config['obs']['f1']['cols_old'], sep=config['obs']['f1']['sep'])
                # enforce column order from subset procedure from usecols in read_csv
                obs_data2 = obs_data2[config['obs']['f1']['cols_old']]
                # rename data columns to CLM standard
                obs_data2 = obs_data2.rename(columns=config['obs']['f1']['cols_new'])
                # remove rows with no date/time of measurement
                obs_data2 = obs_data2.dropna(subset=config['obs']['f1']['datetime_cols'])
                # select Barrow site
                obs_data2 = obs_data2.loc[obs_data2['SITE'] == 'AD']
                # concat file
                obs_data = pd.concat([obs_data, obs_data2], ignore_index=True)
            # drop site column
            obs_data = obs_data.drop(columns = ['SITE'])
            # set time from timestamp
            obs_data['time'] = obs_data['strAlaska'].astype(str)
            # change all -999.9 values to NaN
            obs_data = obs_data.replace(-999.9, np.NaN)
            # scale celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # calculate SWIN from PAR
            obs_data.loc[:,'FSDS'] = obs_data['PAR']/2.1#4.57)/0.46
            obs_data = obs_data.drop(columns = ['PAR'])
            #WS
            # precip
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print(obs_data, file=f)
        case 'CAN-CambridgeBay':
            # read other files and concat to first
            for extra_file in config['obs']['f1']['extended_files']:
                obs_data2 = pd.read_csv(extra_file, index_col=False, engine='python', skiprows=config['obs']['f1']['skip_rows'], \
                    usecols=config['obs']['f1']['cols_old'], sep=config['obs']['f1']['sep'])
                # enforce column order from subset procedure from usecols in read_csv
                obs_data2 = obs_data2[config['obs']['f1']['cols_old']]
                # rename data columns to CLM standard
                obs_data2 = obs_data2.rename(columns=config['obs']['f1']['cols_new'])
                # remove rows with no date/time of measurement
                obs_data2 = obs_data2.dropna(subset=config['obs']['f1']['datetime_cols'])
                # concat file
                obs_data = pd.concat([obs_data, obs_data2], ignore_index=True)
            # set time from date string
            obs_data['time'] = obs_data['Date/Time (LST)'].astype(str)
            # bring in karasjok loop
            # merge data streams into single dataframe
            # adjust data/units
            # scale celsius to kelvin
            obs_data['TBOT'] = obs_data['TBOT'] + 273.15
            # scale pressure - miss labeled somehow
            obs_data['PBOT'] = obs_data['PBOT']*1000
            # wind from km/h to m/s
            obs_data['WIND'] = (obs_data['WIND']/3600)*1000
            # calculate SH from RH/TBOT/PBOT
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            obs_data = obs_data.drop(columns=['RH','PRECIP'])
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print(obs_data, file=f)
        case 'SVA-Adventdalen':
            # read other files and concat to first
            for extra_file in config['obs']['f1']['extended_files']:
                obs_data_f1e = pd.read_csv(extra_file, index_col=False, sep=config['obs']['f1']['sep'], \
                                engine='python', skiprows=config['obs']['f1']['skip_rows'])
                # add columns that are missing from cols_old
                for col_name in config['obs']['f1']['cols_old']:
                    if col_name not in list(obs_data_f1e.columns.values):
                        obs_data_f1e[col_name] = np.NaN
                # enforce column order from subset procedure from usecols in read_csv
                obs_data_f1e = obs_data_f1e[config['obs']['f1']['cols_old']]
                # rename data columns to CLM standard
                obs_data_f1e = obs_data_f1e.rename(columns=config['obs']['f1']['cols_new'])
                # remove rows with no date/time of measurement
                obs_data_f1e = obs_data_f1e.dropna(subset=config['obs']['f1']['datetime_cols'])
                # concat file
                obs_data = pd.concat([obs_data, obs_data_f1e], ignore_index=True)
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('file 1 done', file=f)
                print(obs_data, file=f)
                print(obs_data.dtypes, file=f)
            # bring in lufthaven data loop
            obs_data_f2 = pd.read_csv(config['obs']['f2']['name'], index_col=False, engine='python', sep=config['obs']['f2']['sep'], \
                            skiprows=config['obs']['f2']['skip_rows'], usecols=config['obs']['f2']['cols_old'])
            # rename columns from first file
            obs_data_f2 = obs_data_f2.rename(columns=config['obs']['f2']['cols_new'])
            for extra_file in config['obs']['f2']['extended_files']:
                obs_data_f2e = pd.read_csv(extra_file, index_col=False, sep=config['obs']['f2']['sep'], \
                                 engine='python', skiprows=config['obs']['f2']['skip_rows'])
                # add columns that are missing from cols_old
                #for col_name in config['obs']['f2']['cols_old']:
                #    if col_name not in list(obs_data_f2e.columns.values):
                #        obs_data_f2e[col_name] = np.NaN
                # enforce column order from subset procedure from usecols in read_csv
                obs_data_f2e = obs_data_f2e[config['obs']['f2']['cols_old']]
                # rename data columns to CLM standard
                obs_data_f2e = obs_data_f2e.rename(columns=config['obs']['f2']['cols_new'])
                # remove rows with no date/time of measurement
                obs_data_f2e = obs_data_f2e.dropna(subset=config['obs']['f2']['datetime_cols'])
                # concat file
                obs_data_f2 = pd.concat([obs_data_f2, obs_data_f2e], ignore_index=True)
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('file 2 done', file=f)
                print(obs_data_f2, file=f)
                print(obs_data_f2.dtypes, file=f)
            # bring in janssonhaugen data loop
            obs_data_f3 = pd.read_csv(config['obs']['f3']['name'], index_col=False, engine='python', sep=config['obs']['f3']['sep'], \
                            skiprows=config['obs']['f3']['skip_rows'], usecols=config['obs']['f3']['cols_old'])
            # rename columns from first file
            obs_data_f3 = obs_data_f3.rename(columns=config['obs']['f3']['cols_new'])
            for extra_file in config['obs']['f3']['extended_files']:
                obs_data_f3e = pd.read_csv(extra_file, index_col=False, sep=config['obs']['f3']['sep'], \
                                 engine='python', skiprows=config['obs']['f3']['skip_rows'])
                # add columns that are missing from cols_old
                #for col_name in config['obs']['f3']['cols_old']:
                #    if col_name not in list(obs_data_f3e.columns.values):
                #        obs_data_f3e[col_name] = np.NaN
                # enforce column order from subset procedure from usecols in read_csv
                obs_data_f3e = obs_data_f3e[config['obs']['f3']['cols_old']]
                # rename data columns to CLM standard
                obs_data_f3e = obs_data_f3e.rename(columns=config['obs']['f3']['cols_new'])
                # remove rows with no date/time of measurement
                obs_data_f3e = obs_data_f3e.dropna(subset=config['obs']['f3']['datetime_cols'])
                # concat file
                obs_data_f3 = pd.concat([obs_data_f3, obs_data_f3e], ignore_index=True)
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('file 3 done', file=f)
                print(obs_data_f3, file=f)
                print(obs_data_f3.dtypes, file=f)
            # change to datetime to merge data
            obs_data['time'] = obs_data['referenceTime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%f%z'))
            obs_data_f2['time'] = obs_data_f2['referenceTime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%f%z'))
            obs_data_f3['time'] = obs_data_f3['referenceTime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%f%z'))
            # remove old column before merge
            obs_data = obs_data.drop(columns=['referenceTime'])
            obs_data_f2 = obs_data_f2.drop(columns=['referenceTime'])
            obs_data_f3 = obs_data_f3.drop(columns=['referenceTime'])
            # merge data streams into single dataframe
            obs_data = pd.merge(obs_data, obs_data_f2, on='time', how='outer')
            obs_data = pd.merge(obs_data, obs_data_f3, on='time', how='outer')
            # sort dates after merge to restore timeseries order
            obs_data = obs_data.sort_values(by='time')
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('files merged', file=f)
                print(obs_data, file=f)
                print(obs_data.dtypes, file=f)
            # convert to desired text string for date
            obs_data['time'] = obs_data['time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')).astype(str)
            # scale celsius to kelvin
            obs_data['TBOT'] = obs_data['TBOT'] + 273.15
            # scale mbar to kpa
            obs_data['PBOT'] = obs_data['PBOT']*100
            # calculate SH from RH/TBOT/PBOT
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            # remove uneeded columns
            obs_data = obs_data.drop(columns=['RH','PRECIP'])
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('data adjusted', file=f)
                print(obs_data, file=f)
                print(obs_data.head(), file=f)
                print(obs_data.dtypes, file=f)
        case 'NOR-Iskoras':
            # read other files and concat to first
            for extra_file in config['obs']['f1']['extended_files']:
                obs_data_f1e = pd.read_csv(extra_file, index_col=False, sep=config['obs']['f1']['sep'],\
                                 engine='python', skiprows=config['obs']['f1']['skip_rows'])
                # add columns that are missing from cols_old
                #for col_name in config['obs']['f1']['cols_old']:
                #    if col_name not in list(obs_data_f1e.columns.values):
                #        obs_data_f1e[col_name] = np.NaN
                # enforce column order from subset procedure from usecols in read_csv
                obs_data_f1e = obs_data_f1e[config['obs']['f1']['cols_old']]
                # rename data columns to CLM standard
                obs_data_f1e = obs_data_f1e.rename(columns=config['obs']['f1']['cols_new'])
                # remove rows with no date/time of measurement
                obs_data_f1e = obs_data_f1e.dropna(subset=config['obs']['f1']['datetime_cols'])
                # concat file
                obs_data = pd.concat([obs_data, obs_data_f1e], ignore_index=True)
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('file 1 done', file=f)
                print(obs_data, file=f)
                print(obs_data.dtypes, file=f)
            # bring in karasjok data
            obs_data_f2 = pd.read_csv(config['obs']['f2']['name'], index_col=False, engine='python', sep=config['obs']['f2']['sep'], \
                            skiprows=config['obs']['f2']['skip_rows'], usecols=config['obs']['f2']['cols_old'])
            # rename columns from first file
            obs_data_f2 = obs_data_f2.rename(columns=config['obs']['f2']['cols_new'])
            for extra_file in config['obs']['f2']['extended_files']:
                obs_data_f2e = pd.read_csv(extra_file, index_col=False, sep=config['obs']['f2']['sep'], \
                                 engine='python', skiprows=config['obs']['f2']['skip_rows'])
                # add columns that are missing from cols_old
                #for col_name in config['obs']['f2']['cols_old']:
                #    if col_name not in list(obs_data_f2e.columns.values):
                #        obs_data_f2e[col_name] = np.NaN
                # enforce column order from subset procedure from usecols in read_csv
                obs_data_f2e = obs_data_f2e[config['obs']['f2']['cols_old']]
                # rename data columns to CLM standard
                obs_data_f2e = obs_data_f2e.rename(columns=config['obs']['f2']['cols_new'])
                # remove rows with no date/time of measurement
                obs_data_f2e = obs_data_f2e.dropna(subset=config['obs']['f2']['datetime_cols'])
                # concat file
                obs_data_f2 = pd.concat([obs_data_f2, obs_data_f2e], ignore_index=True)
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('file 2 done', file=f)
                print(obs_data_f2, file=f)
                print(obs_data_f2.dtypes, file=f)
            # change to datetime to merge data
            obs_data['time'] = obs_data['referenceTime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%f%z'))
            obs_data_f2['time'] = obs_data_f2['referenceTime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%f%z'))
            # remove old column before merge
            obs_data = obs_data.drop(columns=['referenceTime'])
            obs_data_f2 = obs_data_f2.drop(columns=['referenceTime'])
            # merge data streams into single dataframe
            obs_data = pd.merge(obs_data, obs_data_f2, on='time', how='outer')
            # sort dates after merge to restore timeseries order
            obs_data = obs_data.sort_values(by='time')
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('files merged', file=f)
                print(obs_data, file=f)
                print(obs_data.dtypes, file=f)
            # convert to desired text string for date
            obs_data['time'] = obs_data['time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')).astype(str)
            # scale celsius to kelvin
            obs_data['TBOT'] = obs_data['TBOT'] + 273.15
            # scale mbar to kpa
            obs_data['PBOT'] = obs_data['PBOT']*100
            # calculate SH from RH/TBOT/PBOT
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            # remove uneeded columns
            obs_data = obs_data.drop(columns=['RH','PRECIP'])
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('data adjusted', file=f)
                print(obs_data, file=f)
                print(obs_data.head(), file=f)
                print(obs_data.dtypes, file=f)
        case 'GRE-Zackenburg':
            ##### read in air temp
            f2 = pd.read_csv(config['obs']['f2']['name'], sep=config['obs']['f2']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f2']['skip_rows'], \
                usecols=config['obs']['f2']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f2 = f2[config['obs']['f2']['cols_old']]
            # rename data columns to CLM standard
            f2 = f2.rename(columns=config['obs']['f2']['cols_new'])
            # remove rows with no date/time of measurement
            f2 = f2.dropna(subset=config['obs']['f2']['datetime_cols'])
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('f2 read in', file=f)
                print(f2, file=f)
                print(obs_data.dtypes, file=f)
            ##### read in relative humidity
            f3 = pd.read_csv(config['obs']['f3']['name'], sep=config['obs']['f3']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f3']['skip_rows'], \
                usecols=config['obs']['f3']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f3 = f3[config['obs']['f3']['cols_old']]
            # rename data columns to CLM standard
            f3 = f3.rename(columns=config['obs']['f3']['cols_new'])
            # remove rows with no date/time of measurement
            f3 = f3.dropna(subset=config['obs']['f3']['datetime_cols'])
            ###### read in precipitation
            f4 = pd.read_csv(config['obs']['f4']['name'], sep=config['obs']['f4']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f4']['skip_rows'], \
                usecols=config['obs']['f4']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f4 = f4[config['obs']['f4']['cols_old']]
            # rename data columns to CLM standard
            f4 = f4.rename(columns=config['obs']['f4']['cols_new'])
            # remove rows with no date/time of measurement
            f4 = f4.dropna(subset=config['obs']['f4']['datetime_cols'])
            ##### read in wind speed
            f5 = pd.read_csv(config['obs']['f5']['name'], sep=config['obs']['f5']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f5']['skip_rows'], \
                usecols=config['obs']['f5']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f5 = f5[config['obs']['f5']['cols_old']]
            # rename data columns to CLM standard
            f5 = f5.rename(columns=config['obs']['f5']['cols_new'])
            # remove rows with no date/time of measurement
            f5 = f5.dropna(subset=config['obs']['f5']['datetime_cols'])
            ##### read in SWIN
            f6 = pd.read_csv(config['obs']['f6']['name'], sep=config['obs']['f6']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f6']['skip_rows'], \
                usecols=config['obs']['f6']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f6 = f6[config['obs']['f6']['cols_old']]
            # rename data columns to CLM standard
            f6 = f6.rename(columns=config['obs']['f6']['cols_new'])
            # remove rows with no date/time of measurement
            f6 = f6.dropna(subset=config['obs']['f6']['datetime_cols'])
            ##### read in LWIN 
            f7 = pd.read_csv(config['obs']['f7']['name'], sep=config['obs']['f7']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f7']['skip_rows'], \
                usecols=config['obs']['f7']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f7 = f7[config['obs']['f7']['cols_old']]
            # rename data columns to CLM standard
            f7 = f7.rename(columns=config['obs']['f7']['cols_new'])
            # remove rows with no date/time of measurement
            f7 = f7.dropna(subset=config['obs']['f7']['datetime_cols'])
            # subset each file by quality flag
            quality_flags = ['good']
            obs_data = obs_data[obs_data['Quality Flag'].isin(quality_flags)]
            f2 = f2[f2['Quality Flag'].isin(quality_flags)]
            f3 = f3[f3['Quality Flag'].isin(quality_flags)]
            f4 = f4[f4['Quality Flag'].isin(quality_flags)]
            f5 = f5[f5['Quality Flag'].isin(quality_flags)]
            f6 = f6[f6['Quality Flag'].isin(quality_flags)]
            f7 = f7[f7['Quality Flag'].isin(quality_flags)]
            # make datatime column from date/time columns
            obs_data['time'] = obs_data['Date'].astype(str) + ' ' + obs_data['Time'].astype(str)
            f2['time'] = f2['Date'].astype(str) + ' ' + f2['Time'].astype(str) 
            f3['time'] = f3['Date'].astype(str) + ' ' + f3['Time'].astype(str)
            f4['time'] = f4['Date'].astype(str) + ' ' + f4['Time'].astype(str)
            f5['time'] = f5['Date'].astype(str) + ' ' + f5['Time'].astype(str)
            f6['time'] = f6['Date'].astype(str) + ' ' + f6['Time'].astype(str)
            f7['time'] = f7['Date'].astype(str) + ' ' + f7['Time'].astype(str)
            # remove quality flag, Date, and Time columns
            obs_data = obs_data.drop(columns=['Quality Flag','Date','Time'])
            f2 = f2.drop(columns=['Quality Flag','Date','Time'])
            f3 = f3.drop(columns=['Quality Flag','Date','Time'])
            f4 = f4.drop(columns=['Quality Flag','Date','Time'])
            f5 = f5.drop(columns=['Quality Flag','Date','Time'])
            f6 = f6.drop(columns=['Quality Flag','Date','Time'])
            f7 = f7.drop(columns=['Quality Flag','Date','Time'])
            # merge all datacolumns by date
            obs_data = pd.merge(obs_data, f2, on='time', how='outer')
            obs_data = pd.merge(obs_data, f3, on='time', how='outer')
            obs_data = pd.merge(obs_data, f4, on='time', how='outer')
            obs_data = pd.merge(obs_data, f5, on='time', how='outer')
            obs_data = pd.merge(obs_data, f6, on='time', how='outer')
            obs_data = pd.merge(obs_data, f7, on='time', how='outer')
            # sort dates after merge to restore timeseries order
            obs_data = obs_data.sort_values(by='time')
            # scale celsius to kelvin
            obs_data['TBOT'] = obs_data['TBOT'] + 273.15
            # scale mbar to kpa
            obs_data['PBOT'] = obs_data['PBOT']*100
            # calculate SH from RH/TBOT/PBOT
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            # remove uneeded columns
            obs_data = obs_data.drop(columns=['RH','PRECIP'])
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('data adjusted', file=f)
                print(obs_data, file=f)
                print(obs_data.head(), file=f)
                print(obs_data.dtypes, file=f)
        case 'GRE-Disko':
            ##### read in air temp
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print(config['obs']['f2']['name'], file=f)
            f2 = pd.read_csv(config['obs']['f2']['name'], sep=config['obs']['f2']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f2']['skip_rows'], \
                usecols=config['obs']['f2']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f2 = f2[config['obs']['f2']['cols_old']]
            # rename data columns to CLM standard
            f2 = f2.rename(columns=config['obs']['f2']['cols_new'])
            # remove rows with no date/time of measurement
            f2 = f2.dropna(subset=config['obs']['f2']['datetime_cols'])
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('f2 read in', file=f)
                print(f2, file=f)
                print(obs_data.dtypes, file=f)
            ##### read in relative humidity
            f3 = pd.read_csv(config['obs']['f3']['name'], sep=config['obs']['f3']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f3']['skip_rows'], \
                usecols=config['obs']['f3']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f3 = f3[config['obs']['f3']['cols_old']]
            # rename data columns to CLM standard
            f3 = f3.rename(columns=config['obs']['f3']['cols_new'])
            # remove rows with no date/time of measurement
            f3 = f3.dropna(subset=config['obs']['f3']['datetime_cols'])
            ###### read in precipitation
            f4 = pd.read_csv(config['obs']['f4']['name'], sep=config['obs']['f4']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f4']['skip_rows'], \
                usecols=config['obs']['f4']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f4 = f4[config['obs']['f4']['cols_old']]
            # rename data columns to CLM standard
            f4 = f4.rename(columns=config['obs']['f4']['cols_new'])
            # remove rows with no date/time of measurement
            f4 = f4.dropna(subset=config['obs']['f4']['datetime_cols'])
            ##### read in wind speed
            f5 = pd.read_csv(config['obs']['f5']['name'], sep=config['obs']['f5']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f5']['skip_rows'], \
                usecols=config['obs']['f5']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f5 = f5[config['obs']['f5']['cols_old']]
            # rename data columns to CLM standard
            f5 = f5.rename(columns=config['obs']['f5']['cols_new'])
            # remove rows with no date/time of measurement
            f5 = f5.dropna(subset=config['obs']['f5']['datetime_cols'])
            ##### read in SWIN
            f6 = pd.read_csv(config['obs']['f6']['name'], sep=config['obs']['f6']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f6']['skip_rows'], \
                usecols=config['obs']['f6']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f6 = f6[config['obs']['f6']['cols_old']]
            # rename data columns to CLM standard
            f6 = f6.rename(columns=config['obs']['f6']['cols_new'])
            # remove rows with no date/time of measurement
            f6 = f6.dropna(subset=config['obs']['f6']['datetime_cols'])
            ##### read in LWIN 
            f7 = pd.read_csv(config['obs']['f7']['name'], sep=config['obs']['f7']['sep'], \
                index_col=False, engine='python', skiprows=config['obs']['f7']['skip_rows'], \
                usecols=config['obs']['f7']['cols_old'])
            # enforce column order from subset procedure from usecols in read_csv
            f7 = f7[config['obs']['f7']['cols_old']]
            # rename data columns to CLM standard
            f7 = f7.rename(columns=config['obs']['f7']['cols_new'])
            # remove rows with no date/time of measurement
            f7 = f7.dropna(subset=config['obs']['f7']['datetime_cols'])
            # subset each file by quality flag
            quality_flags = ['good']
            obs_data = obs_data[obs_data['quality flag'].isin(quality_flags)]
            f2 = f2[f2['quality flag'].isin(quality_flags)]
            f3 = f3[f3['quality flag'].isin(quality_flags)]
            f4 = f4[f4['quality flag'].isin(quality_flags)]
            f5 = f5[f5['quality flag'].isin(quality_flags)]
            f6 = f6[f6['quality flag'].isin(quality_flags)]
            f7 = f7[f7['quality flag'].isin(quality_flags)]
            # make datatime column from date/time columns
            obs_data['time'] = obs_data['Date'].astype(str) + ' ' + obs_data['Time'].astype(str)
            f2['time'] = f2['Date'].astype(str) + ' ' + f2['Time'].astype(str) 
            f3['time'] = f3['Date'].astype(str) + ' ' + f3['Time'].astype(str)
            f4['time'] = f4['Date'].astype(str) + ' ' + f4['Time'].astype(str)
            f5['time'] = f5['Date'].astype(str) + ' ' + f5['Time'].astype(str)
            f6['time'] = f6['Date'].astype(str) + ' ' + f6['Time'].astype(str)
            f7['time'] = f7['Date'].astype(str) + ' ' + f7['Time'].astype(str)
            # remove quality flag, Date, and Time columns
            obs_data = obs_data.drop(columns=['quality flag','Date','Time'])
            f2 = f2.drop(columns=['quality flag','Date','Time'])
            f3 = f3.drop(columns=['quality flag','Date','Time'])
            f4 = f4.drop(columns=['quality flag','Date','Time'])
            f5 = f5.drop(columns=['quality flag','Date','Time'])
            f6 = f6.drop(columns=['quality flag','Date','Time'])
            f7 = f7.drop(columns=['quality flag','Date','Time'])
            # merge all datacolumns by date
            obs_data = pd.merge(obs_data, f2, on='time', how='outer')
            obs_data = pd.merge(obs_data, f3, on='time', how='outer')
            obs_data = pd.merge(obs_data, f4, on='time', how='outer')
            obs_data = pd.merge(obs_data, f5, on='time', how='outer')
            obs_data = pd.merge(obs_data, f6, on='time', how='outer')
            obs_data = pd.merge(obs_data, f7, on='time', how='outer')
            # sort dates after merge to restore timeseries order
            obs_data = obs_data.sort_values(by='time')
            # scale celsius to kelvin
            obs_data['TBOT'] = obs_data['TBOT'] + 273.15
            # scale mbar to kpa
            obs_data['PBOT'] = obs_data['PBOT']*100
            # calculate SH from RH/TBOT/PBOT
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            # remove uneeded columns
            obs_data = obs_data.drop(columns=['RH','PRECIP'])
            with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                print('data adjusted', file=f)
                print(obs_data, file=f)
                print(obs_data.head(), file=f)
                print(obs_data.dtypes, file=f)
    try:        
        # drop old date/time columns
        obs_data = obs_data.drop(columns=config['obs']['f1']['datetime_cols'], errors='ignore')    
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print(obs_data['time'], file=f)
            print('past1', file=f)
        # remove duplicate timestamps
        obs_data = obs_data.drop_duplicates(subset='time')
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print('past2', file=f)
        # create datetime values from numerical timestamp after conversion to string
        obs_data.loc[:,'time'] = obs_data['time'].apply(lambda x: datetime.strptime(x, config['obs']['f1']['datetime_format']))
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print('past3', file=f)
        # create new index that can fill missing timesteps
        new_index = pd.date_range(start=obs_data.at[obs_data.index[0],'time'], \
                                  end=obs_data.at[obs_data.index[-1],'time'], freq=config['obs']['f1']['freq'])
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print('past4', file=f)
        # set dateime as index
        obs_data = obs_data.set_index(['time'])
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print('past5', file=f)
        # create xarray dataset
        ds = obs_data.to_xarray()
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print('past6', file=f)
        # use reindex to add the missing timesteps and fill data values with na as default
        ds = ds.reindex({"time": new_index})
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print('past7', file=f)
        # convert to 365_day calendar using dataset.convert_calendar (drops leap days)
        ds = ds.convert_calendar("365_day")
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print('past8', file=f)
        # shift time index by offset from GMT described in observational dataset
        ds.coords['time'] = ds.indexes['time'].shift(config['obs_GMT_adj'], 'H')
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print('past9', file=f)
        ## set netcdf write characteristics for xarray.to_netcdf()
        comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                    complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
        # create encoding
        encoding = {var: comp for var in ds.data_vars}
        # create file output name
        nc_out = Path(config['site_dir'] + "Obs_" + config['site_name'] + "_dat.nc" )
        # send to netcdf
        ds.to_netcdf(nc_out, mode="w", encoding=encoding, \
                        format=config['nc_write']['format'],\
                        engine=config['nc_write']['engine'])
    except Exception as error:
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
            print(error, file=f)
        
# define function to create mapped dictionary for groups
def map_groups(time, avg_win, sub_win, config):
    # creat function to create idx strings from time
    def user_groupby(time, avg_win, sub_win):
        if avg_win == 'month':
            aw = time.month
        elif avg_win == 'dayofyear':
            aw = time.dayofyr
        elif avg_win == 'weekofyear':
            doy = time.dayofyr
            aw = math.ceil(doy/7)
            if aw > 52:
                aw = 52
        if sub_win == 'hour':
            sw = time.hour
            idx_str = 'W'+str(aw).zfill(3)+'S'+str(sw).zfill(3)
        elif sub_win == None:
            idx_str = 'W'+str(aw).zfill(3)
        return idx_str
    # start blank list for dictionary pairs
    avg_sub_idx = []
    # match the case of requested averaging and sub windows
    match avg_win:
        case 'dayofyear':
            match sub_win:
                case None:
                    for dayofyear in range(1,366):
                            avg_sub_idx.append('W'+str(dayofyear).zfill(3))
        case 'weekofyear':
            match sub_win:
                case 'hour':
                    for weekofyear in range(1,53):
                        for hour in range(0,24):
                            avg_sub_idx.append('W'+str(weekofyear).zfill(3)+'S'+str(hour).zfill(3))
                case '':
                    pass
        case 'month':
            match sub_win:
                case 'hour':
                    # loop avg and sub windows to create list
                    for month in range(1,13):
                        for hour in range(0,24):
                            avg_sub_idx.append('W'+str(month).zfill(3)+'S'+str(hour).zfill(3))
                case '':
                    pass
    # zip/map a list of integers to list of strings that represent
    mapped_dict = dict(zip(avg_sub_idx, range(1, len(avg_sub_idx)+1)))
    # apply dictionary to datasets input time (as data array
    new_groups = time.map(lambda x: mapped_dict[user_groupby(x, avg_win, sub_win)])
    return new_groups

# convert simulation and obvervation netcdf to multi-year daily means    
def multiyear_daily_means(f_iter):
    # take values from zip for config and nc_type(i.e. CRUJRA vs Obs)
    config = read_config(f_iter[0])
    nc_type = f_iter[1] 
    # create input file name
    nc_file = Path(config['site_dir'] + nc_type + "_" + config['site_name'] + "_dat.nc")
    # create file output name
    nc_out = Path(config['site_dir'] + nc_type + "_" + config['site_name'] + "_mym.nc")
    # open netcdf file using context manager and xarray
    with xr.open_dataset(nc_file, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds = ds_tmp.load()
    # create user defined groupby values to do non-standard time averages
    # new grouping coordinate must be integer data type and added to time dimenion
    ds = ds.assign_coords(groupvar = ('time', map_groups(ds.indexes['time'], 'weekofyear', 'hour', config)))
    # groupby new coordinate
    ds = ds.groupby('groupvar').mean() 
    ## set netcdf write characteristics for xarray.to_netcdf()
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
    # create encoding
    encoding = {var: comp for var in ds.data_vars}
    # save file
    ds.to_netcdf(nc_out, mode="w", \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])

# calculate additive/multiplicative offset (bias)
def bias_calculation(f_iter):
    # read in config
    config = read_config(f_iter[0])
    bias_type = f_iter[1]
    # create file names for CRUJRA and obs mydm
    f_cru = Path(config['site_dir'] + "CRUJRA" + "_" + config['site_name'] + "_mym.nc")
    f_obs = Path(config['site_dir'] + "Obs" + "_" + config['site_name'] + "_mym.nc")
    bias_file = Path(config['site_dir'] + bias_type + "_" + config['site_name'] + "_mym.nc")
    # read cru
    with xr.open_dataset(f_cru, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_cru = ds_tmp.load()
    # read obs
    with xr.open_dataset(f_obs, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_obs = ds_tmp.load()
    # list of vlimate variables to attempt to adjust
    keep = ['FSDS','FLDS','PBOT','RAIN','SNOW','QBOT','TBOT','WIND']
    # deep copy the cru data to replace with nans and fill with bias calculations
    bc = ds_cru.copy(deep=True)
    # calculate biases for each climate variable in keep
    for var in keep:
        try:
            # replace previous values with NaN
            bc[var].loc[:] = np.NaN
            if bias_type == 'ABias':
                # replace empty values with with obs - cru
                bc[var] = ds_obs[var] - ds_cru[var]
            elif bias_type == 'MBias':
                # scale dataaway from zero/very small numbers that cause numeric explosion towards inf
                ds_obs[var] = (ds_obs[var] * 1000.0)
                ds_cru[var] = (ds_cru[var] * 1000.0)
                # replace empty values with with obs / cru 
                # add one to both sides if there are legitimate zeros in the data to avoid division by zero
                if var in ['FSDS','FLDS']:
                    bc[var] = (ds_obs[var]+1) / (ds_cru[var]+1)
                else:
                    bc[var] = ds_obs[var] / ds_cru[var]
            # fill gaps in bias if data is 75% complete or greater
            if bc[var].isnull().sum() < 0.05*len(bc[var]):
                bc[var] = bc[var].interpolate_na(dim='groupvar', method='nearest')
        except Exception:
            pass
    # set netcdf write characteristics for xarray.to_netcdf()
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
    # create encoding
    encoding = {var: comp for var in bc.data_vars}
    # save netcdf of bias calculateions
    bc.to_netcdf(bias_file, mode="w", \
                    format=config['nc_write']['format'], \
                    engine=config['nc_write']['engine'])   

# define function to apply calculated bias offset to full CRUJRA product
def bias_correction(f_iter):
    # read in config
    config = read_config(f_iter[0])
    bias_type = f_iter[1]
    # create file name for fully adjusted dataset
    cru_file = Path(config['site_dir'] + "CRUJRA_" + config['site_name'] + "_allyears.nc")
    cru_bc_file = Path(config['site_dir'] + bias_type + "_" + config['site_name'] + "_allyears.nc")
    bias_file = Path(config['site_dir'] + bias_type + "_" + config['site_name'] + "_mym.nc")
    # read in bias file
    with xr.open_dataset(bias_file, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_bias = ds_tmp.load()
    # read in concatenated CRUJRA site file
    with xr.open_dataset(cru_file, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_cru = ds_tmp.load()
    # list of vlimate variables to attempt to adjust
    keep = ['FSDS','FLDS','PBOT','RAIN','SNOW','QBOT','TBOT','WIND']
    ds_cru = ds_cru[keep]
    # create user defined groupby values to do non-standard time averages
    # new grouping coordinate must be integer data type and added to time dimenion
    ds_cru = ds_cru.assign_coords(groupvar = ('time', map_groups(ds_cru.indexes['time'], 'weekofyear', 'hour', config)))
    # adjust cru using groupby and daily multi-index to replace each variable data
    for var in keep:
        try:
            # check for nans in the bias calculation and skip correction if so
            if ds_bias[var].isnull().sum() > 0:
                continue
            # check for bias type to add or multiple bias values
            if bias_type == 'ABias':
                ds_cru[var] = ds_cru[var].groupby('groupvar') + ds_bias[var]
            elif bias_type == 'MBias':
                # scale data similarly to bias calculation
                ds_cru[var] = (ds_cru[var] * 1000.0)
                ds_cru[var] = ds_cru[var].groupby('groupvar') * ds_bias[var]
                ds_cru[var] = (ds_cru[var] / 1000.0)
        except Exception as error:
            with open(Path(config['site_dir'] + bias_type + '_debug.txt'), 'a') as f:
                print(error, file=f)
            pass
    ## set netcdf write characteristics for xarray.to_netcdf()
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
    # encoding
    encoding = {var: comp for var in ds_cru.data_vars}
    # write site's bias corrected netcdf
    ds_cru.to_netcdf(cru_bc_file, mode="w", \
                    format=config['nc_write']['format'],\
                    engine=config['nc_write']['engine'])   

# funtion to output pdf report
def plot_site_graphs(config_file):
    # read in config
    config = read_config(config_file)
    # create file names to load obs, cru, A/M bias corrected cru netcdfs, and A/M bias netcdfs
    f_obs = Path(config['site_dir'] + "Obs" + "_" + config['site_name'] + "_dat.nc")
    f_cru = Path(config['site_dir'] + "CRUJRA" + "_" + config['site_name'] + "_allyears.nc")
    f_cru_abc = Path(config['site_dir'] + "ABias" + "_" + config['site_name'] + "_allyears.nc")
    f_cru_mbc = Path(config['site_dir'] + "MBias" + "_" + config['site_name'] + "_allyears.nc")
    f_ab = Path(config['site_dir'] + "ABias" + "_" + config['site_name'] + "_mym.nc")
    f_mb = Path(config['site_dir'] + "MBias" + "_" + config['site_name'] + "_mym.nc")
    f_cru_mym = Path(config['site_dir'] + "CRUJRA" + "_" + config['site_name'] + "_mym.nc")
    f_obs_mym = Path(config['site_dir'] + "Obs" + "_" + config['site_name'] + "_mym.nc")
    
    # read all the files into memory using context manager
    with xr.open_dataset(f_obs, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_obs = ds_tmp.load()
    with xr.open_dataset(f_cru, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_cru = ds_tmp.load()
    with xr.open_dataset(f_cru_abc, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_cru_abc = ds_tmp.load()
    with xr.open_dataset(f_cru_mbc, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_cru_mbc = ds_tmp.load()
    with xr.open_dataset(f_ab, engine=config['nc_read']['engine']) as ds_tmp:
        ds_ab = ds_tmp.load()
    with xr.open_dataset(f_mb, engine=config['nc_read']['engine']) as ds_tmp:
        ds_mb = ds_tmp.load()
    with xr.open_dataset(f_cru_mym, engine=config['nc_read']['engine']) as ds_tmp:
        ds_cru_mym = ds_tmp.load()
    with xr.open_dataset(f_obs_mym, engine=config['nc_read']['engine']) as ds_tmp:
        ds_obs_mym = ds_tmp.load()
   
    # define function to graph
    def graph_time(ds1, ds2, year_start, year_end, xlab, title, title_size, file_dir, bias, scatter,\
                    time_scale, cru_color, obs_color, leg_text, leg_loc, fontsize, dpi, file_end):
        for var in ['FSDS','FLDS','TBOT','PBOT','QBOT','WIND']:
            try:
                # set typical scaling
                add_or_mult = 'none'
                # case match for scaling and yaxis titles
                match var:
                    case 'FSDS':
                        ylab = 'Shortwave (W/$m^{2}$)'
                        file_part = 'shortwave'
                    case 'FLDS':
                        ylab = 'Longwave (W/$m^{2}$)'
                        file_part = 'longwave'
                    case 'TBOT':
                        if bias == False:
                            factor = -273.15
                            add_or_mult = 'add'
                        ylab = 'Air Temperature ($^\circ$C)'
                        file_part = 'air_temperature'
                    case 'PBOT':
                        if bias == False:
                            factor = 0.001
                            add_or_mult = 'multiply'
                        ylab = 'Pressure (kPa)'
                        file_part = 'air_pressure'
                    case 'QBOT':
                        ylab = 'Specific Humidity (kg/kg)'
                        file_part = 'specific_humidity'   
                    case 'WIND':
                        ylab = 'Wind Speed (m/s)'   
                        file_part = 'wind_speed'
                # make a deep copy of the first xarray dataset so changes do not propogate to ther graphs
                ds1_copy = ds1.copy(deep=True)
                # rescale bias
                if bias == 'add':
                    ds1_copy = ds1_copy 
                elif bias == 'multiply':
                    ds1_copy = ds1_copy / 1000
                # set start and end years for timeseries data
                if time_scale == 'years':
                    start = str(year_start) + '-01-01'
                    end = str(year_end) + '-12-31'
                    ds1_copy = ds1_copy.sel(time=slice(start, end))
                # start figure
                plt.figure(dpi=dpi)
                # plot firt dataset (typically cru)
                if scatter == False:
                    # scale/plot cru data
                    if add_or_mult == 'add':
                        ds1_copy[var] = ds1_copy[var] + factor
                    elif add_or_mult == 'multiply':
                        ds1_copy[var] = ds1_copy[var] * factor
                    # plot first dataset
                    ds1_copy[var].plot(color=cru_color)
                    # try to scale/plot obs data if it exists
                    try:
                        # make deep copy to keep changes form affecting other graphs
                        ds2_copy = ds2.copy(deep=True) 
                        # slice dataset if timeseries of multiple years
                        if time_scale == 'years': 
                            ds2_copy = ds2_copy.sel(time=slice(start, end))
                        # scale if necessary
                        if add_or_mult == 'add':
                            ds2_copy[var] = ds2_copy[var] + factor
                        elif add_or_mult == 'multiply':
                            ds2_copy[var] = ds2_copy[var] * factor
                        # plot second dataset
                        ds2_copy[var].plot(color=obs_color, alpha=0.8)
                    except Exception as error:
                        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                            print(error, file=f)
                        pass
                elif scatter == True:
                    # create scatter plot labels
                    xlab = 'Obs ' + ylab
                    ylab = 'Cru ' + ylab
                    # set new var names
                    x_var = 'obs_' + var
                    y_var = 'cru_' + var
                    # scale/plot cru data
                    if add_or_mult == 'add':
                        ds1_copy[x_var] = ds1_copy[x_var] + factor
                        ds1_copy[y_var] = ds1_copy[y_var] + factor
                    elif add_or_mult == 'multiply':
                        ds1_copy[x_var] = ds1_copy[x_var] * factor
                        ds1_copy[y_var] = ds1_copy[y_var] * factor
                    # find min/max values for xlim, ylim
                    ds1_xval = ds1_copy[x_var].values
                    ds1_yval = ds1_copy[y_var].values
                    ds1_min = np.nanmin((ds1_xval[ds1_xval != -np.inf], ds1_yval[ds1_yval != -np.inf]))
                    ds1_max = np.nanmax((ds1_xval[ds1_xval != np.inf], ds1_yval[ds1_yval != np.inf]))
                    # plot first dataset
                    xr.plot.scatter(ds=ds1_copy, x=x_var, y=y_var, color=cru_color)
                    try:
                        # make deep copy to keep changes form affecting other graphs
                        ds2_copy = ds2.copy(deep=True) 
                        # slice timeseries by start/stop of data years
                        if time_scale == 'years': 
                            ds2_copy = ds2_copy.sel(time=slice(start, end))
                        # scale if necessary
                        if add_or_mult == 'add':
                            ds2_copy[x_var] = ds2_copy[x_var] + factor
                            ds2_copy[y_var] = ds2_copy[y_var] + factor
                        elif add_or_mult == 'multiply':
                            ds2_copy[x_var] = ds2_copy[x_var] * factor
                            ds2_copy[y_var] = ds2_copy[y_var] * factor
                        # find min/max values for xlim, ylim
                        ds2_xval = ds2_copy[x_var].values
                        ds2_yval = ds2_copy[y_var].values
                        ds2_min = np.nanmin((ds2_xval[ds2_xval != -np.inf], ds2_yval[ds2_yval != -np.inf]))
                        ds2_max = np.nanmax((ds2_xval[ds2_xval != np.inf], ds2_yval[ds2_yval != np.inf]))
                        xr.plot.scatter(ds=ds2_copy, x=x_var, y=y_var, color=obs_color, alpha=0.8)
                    except Exception as error:
                        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                            print('scatter error:', error, file=f)
                        pass
                # axis labels
                plt.xlabel(xlab, fontsize=fontsize)
                plt.ylabel(ylab, fontsize=fontsize)
                # legend
                plt.legend(leg_text, loc=leg_loc)
                # plot title
                if title != None:
                    plt.title(title, fontsize=title_size)
                # set axis aspect ratio to 1
                if scatter == True:
                    if ds2 != None:
                        min_val = min(ds1_min, ds2_min)
                        max_val = max(ds1_max, ds2_max)
                    else:
                        min_val = ds1_min
                        max_val = ds1_max
                    plt.xlim(min_val, max_val)
                    plt.ylim(min_val, max_val)
                    plt.gca().set_aspect('equal')
                    plt.axline((0,0), slope=1, linewidth=2, color='black')
                # add trendline if full data timeseries
                if time_scale == 'full':
                    if ds2 != None:
                        ds1_poly = ds1_copy.polyfit(dim='time', deg=1, skipna=True)
                        ds2_poly = ds2_copy.polyfit(dim='time', deg=1, skipna=True)
                        ds1_poly_var = var + '_polyfit_coefficients'
                        ds2_poly_var = var + '_polyfit_coefficients'
                        b1, m1 = ds1_poly[ds1_poly_var].values
                        b2, m2 = ds2_poly[ds2_poly_var].values
                        yfit1 = xr.polyval(ds1_copy['time'], ds1_poly[ds1_poly_var])
                        yfit2 = xr.polyval(ds2_copy['time'], ds2_poly[ds2_poly_var])
                        plt.plot(ds1_copy['time'], yfit1, color='black')
                        plt.plot(ds2_copy['time'], yfit2, color='orange')
                    else:
                        ds1_poly = ds1_copy.polyfit(dim='time', deg=1, skipna=True)
                        ds1_poly_var = var + '_polyfit_coefficients'
                        b1, m1 = ds1_poly[ds1_poly_var].values
                        yfit1 = xr.polyval(ds1_copy['time'], ds1_poly[ds1_poly_var])
                        plt.plot(ds1_copy['time'], yfit1, color='black')
                # save plot to file
                plt.savefig(file_dir + file_part + file_end + '.png')
                # close figure
                plt.close()
            except Exception as error:
                with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
                    print(error, file=f)
                pass
    ###### full time seris plots
    # plot controls
    leg_loc = 1
    cru_color = 'tab:orange'
    obs_color = 'purple'
    leg_text = ['CRUJRA v2.3', 'Observations']
    plot_dpi = 150
    text_size = 16
    x_lab = 'time'
    time_scale = 'years'
    title = None
    title_size = 18
    bias = False
    scatter = False
    # uncorrected cru and observations
    graph_time(ds_cru, ds_obs, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '') 
    # Additive biascorrected cru and observations 
    title = 'Abc'
    leg_text = ['CRUJRA abc', 'Observations']
    graph_time(ds_cru_abc, ds_obs, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_abc') 
    # Multiplicative biascorrected cru and observations 
    title = 'Mbc'
    leg_text = ['CRUJRA mbc', 'Observations']
    graph_time(ds_cru_mbc, ds_obs, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_mbc') 
    ###### climate trend graphs
    cftime_units = 'days since 1901-01-01 00:00:00'
    ds_cru_cft = ds_cru.copy(deep=True)
    ds_obs_cft = ds_obs.copy(deep=True)
    ds_cru_abc_cft = ds_cru_abc.copy(deep=True)
    ds_cru_mbc_cft = ds_cru_mbc.copy(deep=True)
    ds_cru_cft.coords['time'] = cft.date2num(ds_cru_cft.indexes['time'], cftime_units)
    ds_obs_cft.coords['time'] = cft.date2num(ds_obs_cft.indexes['time'], cftime_units)
    ds_cru_abc_cft.coords['time'] = cft.date2num(ds_cru_abc_cft.indexes['time'], cftime_units)
    ds_cru_mbc_cft.coords['time'] = cft.date2num(ds_cru_mbc_cft.indexes['time'], cftime_units)
    x_lab = cftime_units
    time_scale = 'full'
    title = 'CRUJRAv2.3'
    graph_time(ds_cru_cft, ds_obs_cft, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_trend') 
    title = 'Abc CRUJRAv2.3'
    graph_time(ds_cru_abc_cft, ds_obs_cft, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_abc_trend') 
    title = 'Mbc CRUJRAv2.3'
    graph_time(ds_cru_mbc_cft, ds_obs_cft, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_mbc_trend') 
    ##### daily mean plots 
    x_lab = 'hours per week of year'
    time_scale = 'doy'
    title = 'Mym'
    # graph uncorrected cru and observations
    graph_time(ds_cru_mym, ds_obs_mym, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_mym') 
    ###### daily additive bias 
    title = 'ABias'
    leg_text = ['Obs - Cru']
    cru_color = 'tab:green'
    bias = 'add'
    #  graph uncorrected cru and observations
    graph_time(ds_ab, None, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                 time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_abias') 
    ###### daily multiplicative bias 
    title = 'MBias'
    leg_text = ['Obs / Cru']
    cru_color = 'tab:olive'
    bias = 'multiply'
    # graph uncorrected cru and observations
    graph_time(ds_mb, None, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_mbias') 
    ###### R squared plots
    # align cru, obs, abias, and mbias, xarray.datasets through their only dimension - time
    ds_cru2, ds_obs2, ds_abias2, ds_mbias2 = xr.align(ds_cru, ds_obs, ds_cru_abc, ds_cru_mbc)
    # dictionary to rename climate variables to obs_var
    original_data_vars = ['FSDS','FLDS','PBOT','RAIN','SNOW','QBOT','TBOT','WIND']
    new_obs_vars = ['obs_FSDS','obs_FLDS','obs_PBOT','obs_RAIN','obs_SNOW','obs_QBOT','obs_TBOT','obs_WIND']
    new_cru_vars = ['cru_FSDS','cru_FLDS','cru_PBOT','cru_RAIN','cru_SNOW','cru_QBOT','cru_TBOT','cru_WIND']
    cru_dict = dict(zip(original_data_vars, new_cru_vars)) 
    obs_dict = dict(zip(original_data_vars, new_obs_vars))
    # subset dict to only variables that exist in observations to rename
    obs_dict = {k: obs_dict[k] for k in ds_obs2.data_vars}
    with open(Path(config['site_dir'] + 'debug.txt'), 'w') as f:
        print(obs_dict, file=f)
    # rename datasets for combination
    ds_obs2 = ds_obs2.rename(obs_dict)
    ds_cru2 = ds_cru2.rename(cru_dict)
    ds_abias2 = ds_abias2.rename(cru_dict)
    ds_mbias2 = ds_mbias2.rename(cru_dict)
    # add obs data to cru datasets
    for var in list(ds_obs2.data_vars):
        ds_obs_data = ds_obs2[var]
        ds_cru2[var] = ds_obs_data
        ds_abias2[var] = ds_obs_data
        ds_mbias2[var] = ds_obs_data
    # new grouping coordinate must be integer data type and added to time dimenion
    ds_cru3 = ds_cru2.assign_coords(groupvar = ('time', map_groups(ds_cru2.indexes['time'], 'weekofyear', 'hour', config)))
    ds_abias3 = ds_abias2.assign_coords(groupvar = ('time', map_groups(ds_abias2.indexes['time'], 'weekofyear', 'hour', config)))
    ds_mbias3 = ds_mbias2.assign_coords(groupvar = ('time', map_groups(ds_mbias2.indexes['time'], 'weekofyear', 'hour', config)))
    # groupby new coordinate
    ds_cru3 = ds_cru3.groupby('groupvar').mean() 
    ds_abias3 = ds_abias3.groupby('groupvar').mean() 
    ds_mbias3 = ds_mbias3.groupby('groupvar').mean() 
    # create additive plot of uncorrected cru vs obs and corrected cru vs obs to show correction toward 1:1 line
    title = 'Abc Mym'
    leg_text = ['Cru vs Obs','Abc vs Obs']
    cru_color = 'tab:orange'
    time_scale = False
    bias = False
    scatter = True
    graph_time(ds_cru3, ds_abias3, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_abc_mym_rsqr') 
    # create additive plot of uncorrected cru vs obs and corrected cru vs obs to show correction toward 1:1 line
    title = 'Mbc Mym'
    leg_text = ['Cru vs Obs','Mbc vs Obs']
    graph_time(ds_cru3, ds_mbias3, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_mbc_mym_rsqr') 
    # create additive plot of uncorrected cru vs obs and corrected cru vs obs to show correction toward 1:1 line
    title = 'Abc All'
    leg_text = ['Cru vs Obs','Abc vs Obs']
    graph_time(ds_cru2, ds_abias2, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_abc_rsqr') 
    # create additive plot of uncorrected cru vs obs and corrected cru vs obs to show correction toward 1:1 line
    title = 'Mbc All'
    leg_text = ['Cru vs Obs','Mbc vs Obs']
    graph_time(ds_cru2, ds_mbias2, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_mbc_rsqr') 

# Generate PDF report using Reportlab
def climate_pdf_report(config_list):
    # calculate scaled image widths from desired height
    def get_height(path, width):
        try:
            img = utils.ImageReader(path)
            iw, ih = img.getSize()
            aspect = ih/ float(iw)
            return width*aspect
        except Exception as error:
            pass
    # swamp coordinates to 0,0 at top left going down and right with increasing numbers
    # pdf uses bottomup where 0,0 is bottom left of page with increasing numbers going up and right
    def coord(x, y, page_height, image_height, unit):
        x = x * unit
        y = page_height - image_height - y * unit
        return x, y 
    # place single type of plot on to pdf page from left to right, top to bottom
    def add_plots(c, site_dir, drawn_width, page_h, obj_h, inch, v_row, v_space, file_list, col_pos, col_num):
        # set column iterator
        col_iter = 0
        # loop through climate drivers
        for file_name in file_list:
            # get image file path
            image_path = site_dir + file_name
            # place image into pdf
            try:
                c.drawImage(image_path, *coord(col_pos[col_iter],v_row,page_h,obj_h,inch), width=drawn_width, height=obj_h, mask='auto')
            except Exception as error:
                pass
            # add to col iterator up to col number, then reset
            if col_iter < (col_num - 1):
                col_iter += 1
            elif col_iter == (col_num - 1):
                # reset to first column
                col_iter = 0
                # move down a row when column count runs out
                v_row += v_space
    # text wrapping
    def text_wrap(c, colx, coly, lines, set_char_width, font_type, font_size):
        # start textobject
        text_box = c.beginText(colx, coly) 
        # set wordspace to zero as standard
        wordspace = 0
        # calculate first string length (special case of preceeding spaces in paragraph
        full_width = stringWidth('x'*set_char_width, font_type, font_size)
        char_width = stringWidth(lines[0].rstrip(), font_type, font_size)
        space_count = lines[0].rstrip().count(' ')
        space_to_fill = full_width - char_width
        add_space = space_to_fill/space_count
        if add_space < 20:
            text_box.setWordSpace(wordspace+add_space)
        text_box.textLine(lines[0].rstrip())
        # loop through all other lines in paragraph
        for line in lines[1:]:
            full_width = stringWidth('x'*set_char_width, font_type, font_size)
            char_width = stringWidth(line.strip(), font_type, font_size)
            space_count = line.strip().count(' ')
            space_to_fill = full_width - char_width
            add_space = space_to_fill/space_count
            if add_space < 20:
                text_box.setWordSpace(wordspace+add_space)
            text_box.textLine(line.strip())
        c.drawText(text_box)
        return text_box.getY()

    # define page size
    WIDTH, HEIGHT = letter
    # set todays data and a figure directory (should probably move this to the config file eventually)
    DATE = date.today()
    logo_dir = '/scratch/jw2636/wrpmip/logos/'
    doc = docx.Document(Path(logo_dir + 'bc_report_text.docx'))
    full_text = []
    for para in doc.paragraphs:
        full_text.append(str(para.text))
    para_width = 95
    p1_lines = textwrap.wrap(full_text[0], width=para_width, break_long_words=False, tabsize=4)
    p2_lines = textwrap.wrap(full_text[1], width=para_width, break_long_words=False, tabsize=4)
    p3_lines = textwrap.wrap(full_text[2], width=para_width, break_long_words=False, tabsize=4)
    p4_lines = textwrap.wrap(full_text[3], width=para_width, break_long_words=False, tabsize=4)
    # create PDF doc
    c = canvas.Canvas('/projects/warpmip/shared/forcing_data/biascorrected_forcing/bc_report.pdf',  pagesize=letter) 
    ###### cover page
    drawn_width = WIDTH
    image_path = logo_dir + 'pdf_header.png'
    obj_h = get_height(image_path, drawn_width)
    c.drawImage(image_path, *coord(0, 0, HEIGHT, obj_h, inch), width=drawn_width, height=obj_h, mask='auto')
    c.setFont('Helvetica', 24)  
    c.drawCentredString(*coord(4.25, 3.75, HEIGHT, 0.2, inch), f"Site Bias-Correction Report")
    c.setFont('Helvetica', 16)
    c.drawCentredString(*coord(4.25, 4.05, HEIGHT, 0.2, inch), 'Jon M Wells')
    c.setFont('Helvetica', 12)
    c.drawCentredString(*coord(4.25, 4.30, HEIGHT, 0.2, inch), f'{DATE}')
    #pos_y = 6 
    #for font in c.getAvailableFonts():
    #    c.setFont(font, 12)
    #    c.drawString(*coord(0.5, pos_y, HEIGHT, 0.1, inch), font)
    #    pos_y += 0.2
    # place graph on front page
    drawn_width = WIDTH * 0.6
    image_path = logo_dir + 'crujra.png'
    obj_h = get_height(image_path, drawn_width)
    c.drawImage(image_path, *coord(1.75, 4.75, HEIGHT, obj_h, inch), width=drawn_width, height=obj_h, mask='auto')
    # place logos on front page
    drawn_width = WIDTH * 0.2
    image_path = logo_dir + 'nau.png'
    obj_h = get_height(image_path, drawn_width)
    c.drawImage(image_path, 0.4*inch, 0.85*inch, width=drawn_width, height=obj_h, mask='auto')
    image_path = logo_dir + 'ecoss.png'
    obj_h = get_height(image_path, drawn_width)
    c.drawImage(image_path, 2.5*inch, 0.9*inch, width=drawn_width, height=obj_h, mask='auto')
    image_path = logo_dir + 'lbnl.png'
    obj_h = get_height(image_path, drawn_width)
    c.drawImage(image_path, 4.4*inch, 0.7*inch, width=drawn_width, height=obj_h, mask='auto')
    image_path = logo_dir + 'wcrc.png'
    obj_h = get_height(image_path, drawn_width)
    c.drawImage(image_path, 6.3*inch, 0.7*inch, width=drawn_width, height=obj_h, mask='auto')
    col_x, col_y = coord(4.25,5.5,HEIGHT,0.2,inch)
    c.drawCentredString(col_x-0.1*inch, col_y-2.2*inch, 'Jan 2004')
    c.setFont('Helvetica', 8)
    c.drawString(*coord(8, 10.65, HEIGHT, 0, inch), f'{c.getPageNumber()}')
    c.showPage()
    
    sub_title = [
        'CRUJRA vs Observations',
        'Multi-Year Means (Mym) and Additive/Multiplicative Bias Plots (Abias/Mbias)',
        'Additive/Multiplicative Biascorrection Results (Abc/Mbc) of CRUJRAv2.3',
        'Biascorrection R-squared: Multi-Year Means (Mym) and All Data (All)',
        'Climate Trends before/after correction']
    toc_sub_title = [
        '- CRUJRA vs Observations',
        '- Multi-Year Means (Mym) and Additive/Multiplicative Bias Plots (Abias/Mbias)',
        '- Additive/Multiplicative Biascorrection Results (Abc/Mbc) of CRUJRAv2.3',
        '- Biascorrection R-squared: Multi-Year Means (Mym) and All Data (All)',
        '- Climate Trends before/after correction']
    ######## table of contents
    col_x, col_y = coord(0.5,0.6,HEIGHT,0.2,inch)
    c.setFont('Helvetica', 16)
    toc_header = ['Table of Contents:', 'Document Overview', 'CRUJRAv2.3 Bias-Correction by Site:']  
    c.drawString(col_x, col_y, toc_header[0])
    col_y -= 0.4*inch
    c.setFont('Helvetica', 12)  
    c.drawString(col_x+0.5*inch, col_y, toc_header[1])
    page_num = 3
    pg_col = 7.5
    site_col_adj = 1.0
    sub_col_adj = 1.5
    c.setFont('Helvetica', 9)  
    c.drawRightString(pg_col*inch, col_y, str(page_num))
    page_num += 1
    c.linkAbsolute(toc_header[1], toc_header[1], (col_x+0.5*inch, col_y-0.05*inch, col_x+3*inch, col_y+0.15*inch))
    col_y -= 0.25*inch
    c.setFont('Helvetica', 12)  
    c.drawString(col_x+0.5*inch, col_y, toc_header[2])
    c.setFont('Helvetica', 9)  
    c.drawRightString(pg_col*inch, col_y, str(page_num))
    col_y -= 0.2*inch
    for site in config_list:
        c.setFont('Helvetica-Bold', 10)  
        config = read_config(site)
        site_name = config['site_name']
        c.drawString(col_x+site_col_adj*inch, col_y, site_name+':')
        c.setFont('Helvetica', 9)  
        c.drawRightString(pg_col*inch, col_y, str(page_num))
        c.linkAbsolute(site_name, site_name, (col_x+site_col_adj*inch, col_y-0.05*inch, col_x+4*inch, col_y+0.15*inch))
        col_y -= 0.175*inch
        sub_page_num = [page_num, page_num+1, page_num+2, page_num+4, page_num+6]
        for pos in range(0,5):
            c.setFont('Helvetica', 9)
            c.drawString(col_x+sub_col_adj*inch, col_y, toc_sub_title[pos])
            c.drawRightString(pg_col*inch, col_y, str(sub_page_num[pos]))
            c.linkAbsolute(toc_sub_title[pos], site_name+' '+sub_title[pos], (col_x+sub_col_adj*inch, col_y-0.05*inch, col_x+7*inch, col_y+0.15*inch))
            col_y -= 0.15*inch
        page_num += 8
        col_y -= 0.025*inch
    c.setFont('Helvetica', 8)
    c.drawString(*coord(8, 10.65, HEIGHT, 0, inch), f'{c.getPageNumber()}')
    c.showPage()

    ###### document overview page
    col_x, col_y = coord(0.3,0.5,HEIGHT,0.2,inch)
    c.setFont('Helvetica', 16)  
    c.drawString(col_x, col_y, toc_header[1])
    c.bookmarkPage(toc_header[1])
    c.setFont('Helvetica', 14)  
    c.drawString(col_x+0.35*inch, col_y-0.55*inch, 'Problem')
    c.setFont('Helvetica', 10)
    col_y = text_wrap(c, col_x+0.45*inch, col_y-0.75*inch, p1_lines, para_width, 'Helvetica', 10)
    c.setFont('Helvetica', 14)  
    c.drawString(col_x+0.35*inch, col_y-0.2*inch,'Approach')
    c.setFont('Helvetica', 10)
    col_y = text_wrap(c, col_x+0.45*inch, col_y-0.45*inch, p2_lines, para_width, 'Helvetica', 10)
    col_y = text_wrap(c, col_x+0.45*inch, col_y, p3_lines, para_width, 'Helvetica', 10)
    c.setFont('Helvetica', 14)  
    c.drawString(col_x+0.35*inch, col_y-0.2*inch,'Results')
    c.setFont('Helvetica', 10)
    text_wrap(c, col_x+0.45*inch, col_y-0.45*inch, p4_lines, para_width, 'Helvetica', 10)
    c.setFont('Helvetica', 8)
    c.drawString(*coord(8, 10.65, HEIGHT, 0, inch), f'{c.getPageNumber()}') 
    c.showPage()

    ######## site pages
    # climate driver list
    driver_list = ['shortwave', 'longwave', 'air_temperature', 'air_pressure', 'specific_humidity', 'wind_speed', 'shortwave']
    # loop through sites
    for site in config_list:
        # get site info from config file
        config = read_config(site)
        site_name = config['site_name']
        with open(Path(config['site_dir'] + 'debug.txt'), 'w') as f:
            print(full_text, file=f)
        ###### first site specific page
        # create page and add graphs/text
        c.setFont('Helvetica', 16)
        c.drawString(*coord(0.5, 0.5, HEIGHT, 0.2, inch),  f'{site_name}')
        # bookmark pages for each site
        c.bookmarkPage(site_name)
        c.setFont('Helvetica', 12)
        c.drawString(*coord(0.75, 0.8, HEIGHT, 0.2, inch), sub_title[0])
        c.bookmarkPage(site_name+' '+sub_title[0])
        # plot settings for cru vs obs plots
        col_num = 2
        col_pos = [1.25, 4.5]
        v_row = 0.9
        v_space = 2.25
        drawn_width = WIDTH/2 - 1.25*inch
        image_path = config['site_dir'] + 'shortwave.png'
        obj_h = get_height(image_path, drawn_width)
        file_list = [s + '.png' for s in driver_list]
        # place plots into pdf
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list, col_pos, col_num) 
        # add page number and end page
        c.setFont('Helvetica', 8)
        c.drawString(*coord(8, 10.65, HEIGHT, 0, inch), f'{c.getPageNumber()}')
        c.showPage()
        ####### next site specific page
        # add next section header
        c.setFont('Helvetica', 12)
        c.drawString(*coord(0.75, 0.8, HEIGHT, 0.2, inch), sub_title[1])
        c.bookmarkPage(site_name+' '+sub_title[1]) 
        # plot info
        file_ends = ['_mym.png','_abias.png','_mbias.png']
        file_products = itertools.product(driver_list, file_ends)
        file_list = [list(i)[0] + list(i)[1] for i in file_products]
        col_num = 3
        col_pos = [1.25, 3.5, 5.75]
        v_row = 0.9
        v_space = 1.35
        drawn_width = WIDTH/3 - 1.1*inch
        image_path = config['site_dir'] + 'shortwave_mym.png'
        obj_h = get_height(image_path, drawn_width)
        # add plots to pdf 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list, col_pos, col_num) 
        # add page number and end page
        c.setFont('Helvetica', 8)
        c.drawString(*coord(8, 10.65, HEIGHT, 0, inch), f'{c.getPageNumber()}')
        c.showPage()
        ####### next page
        # add next section header
        c.setFont('Helvetica', 12)
        c.drawString(*coord(0.75, 0.8, HEIGHT, 0.2, inch), sub_title[2])
        c.bookmarkPage(site_name+' '+sub_title[2])
        # plot settings for cru vs obs plots
        col_num = 1
        col1_pos = [1.25]
        col2_pos = [4.5]
        v_row = 0.9
        v_space = 2.25
        drawn_width = WIDTH/2 - 1.25*inch
        image_path = config['site_dir'] + 'shortwave_abc.png'
        obj_h = get_height(image_path, drawn_width)
        file_list1 = [s + '_abc.png' for s in driver_list[:4]]
        file_list2 = [s + '_mbc.png' for s in driver_list[:4]]
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list1, col1_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list2, col2_pos, col_num) 
        # add page number and end page
        c.setFont('Helvetica', 8)
        c.drawString(*coord(8, 10.65, HEIGHT, 0, inch), f'{c.getPageNumber()}')
        c.showPage()
        file_list1 = [s + '_abc.png' for s in driver_list[4:]]
        file_list2 = [s + '_mbc.png' for s in driver_list[4:]]
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list1, col1_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list2, col2_pos, col_num) 
        # add page number and end page
        c.setFont('Helvetica', 8)
        c.drawString(*coord(8, 10.65, HEIGHT, 0, inch), f'{c.getPageNumber()}')
        c.showPage() 
        ####### next page
        # add next section header
        c.setFont('Helvetica', 12)
        c.drawString(*coord(0.75, 0.8, HEIGHT, 0.2, inch), sub_title[3])
        c.bookmarkPage(site_name+' '+sub_title[3])
        # plot settings for cru vs obs plots
        col_num = 1
        col1_pos = [0.25]
        col2_pos = [2.15]
        col3_pos = [4.05]
        col4_pos = [5.95]
        v_row = 0.9
        v_space = 2.25
        drawn_width = WIDTH/3 - 0.5*inch
        image_path = config['site_dir'] + 'shortwave_abc.png'
        obj_h = get_height(image_path, drawn_width)
        file_list1 = [s + '_abc_mym_rsqr.png' for s in driver_list[:4]]
        file_list2 = [s + '_abc_rsqr.png' for s in driver_list[:4]]
        file_list3 = [s + '_mbc_mym_rsqr.png' for s in driver_list[:4]]
        file_list4 = [s + '_mbc_rsqr.png' for s in driver_list[:4]]
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list1, col1_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list2, col2_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list3, col3_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list4, col4_pos, col_num) 
        # add page number and end page
        c.setFont('Helvetica', 8)
        c.drawString(*coord(8, 10.65, HEIGHT, 0, inch), f'{c.getPageNumber()}')
        c.showPage()
        file_list1 = [s + '_abc_mym_rsqr.png' for s in driver_list[4:]]
        file_list2 = [s + '_abc_rsqr.png' for s in driver_list[4:]]
        file_list3 = [s + '_mbc_mym_rsqr.png' for s in driver_list[4:]]
        file_list4 = [s + '_mbc_rsqr.png' for s in driver_list[4:]]
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list1, col1_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list2, col2_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list3, col3_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list4, col4_pos, col_num) 
        # add page number and end page
        c.setFont('Helvetica', 8)
        c.drawString(*coord(8, 10.65, HEIGHT, 0, inch), f'{c.getPageNumber()}')
        c.showPage() 
        ####### next page
        # add next section header
        c.setFont('Helvetica', 12)
        c.drawString(*coord(0.75, 0.8, HEIGHT, 0.2, inch), sub_title[4])
        c.bookmarkPage(site_name+' '+sub_title[4])
        # plot settings for cru vs obs plots
        col_num = 1
        col1_pos = [0.45]
        col2_pos = [3.00]
        col3_pos = [5.55]
        v_row = 0.9
        v_space = 2.25
        drawn_width = WIDTH/3 - 0.20*inch
        image_path = config['site_dir'] + 'shortwave_abc.png'
        obj_h = get_height(image_path, drawn_width)
        file_list1 = [s + '_trend.png' for s in driver_list[:4]]
        file_list2 = [s + '_abc_trend.png' for s in driver_list[:4]]
        file_list3 = [s + '_mbc_trend.png' for s in driver_list[:4]]
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list1, col1_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list2, col2_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list3, col3_pos, col_num) 
        # add page number and end page
        c.setFont('Helvetica', 8)
        c.drawString(*coord(8, 10.65, HEIGHT, 0, inch), f'{c.getPageNumber()}')
        c.showPage()
        file_list1 = [s + '_trend.png' for s in driver_list[4:]]
        file_list2 = [s + '_abc_trend.png' for s in driver_list[4:]]
        file_list3 = [s + '_mbc_trend.png' for s in driver_list[4:]]
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list1, col1_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list2, col2_pos, col_num) 
        add_plots(c, config['site_dir'], drawn_width, HEIGHT, obj_h, inch, v_row, v_space, file_list3, col3_pos, col_num) 
        # add page number and end page
        c.setFont('Helvetica', 8)
        c.drawString(*coord(8, 10.65, HEIGHT, 0, inch), f'{c.getPageNumber()}')
        c.showPage() 
    # output pdf
    c.save()

###########################################################################################################################
# Regional simulaitons
###########################################################################################################################i

# remove previous cleaned model simulation folders
def regional_dir_prep(f):
    # read config file
    config = read_config(f)
    # remake directory per folder 
    Path(config['output_dir']+config['model_name']).mkdir(parents=True, exist_ok=True)
    Path(config['output_dir']+config['model_name']).chmod(0o762)

# collect file lists per model and simulations based on configuration files
def regional_simulation_files(f):
    # read config file
    config = read_config(f[0])
    # read the simulation type
    sim_type = f[1]
    # read all CRUJRA input file names from reanalysis directory
    dir_name = sim_type + '_dir'
    sim_str = sim_type + '_str'
    if config['model_name'] != 'ecosys':
        sim_files = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], config[sim_str])))
    else:
        # deal with non-standard file and variable chunking in ecosys
        # this will eventually need to be updated for other files: water, SOC, etc.
        sim_files1 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'daily_C_flux')))
        sim_files2 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'soil_temp1')))
        sim_files3 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'soil_temp2')))
        sim_files4 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'daily_water')))
        sim_files = [sim_files1, sim_files2, sim_files3, sim_files4]
    # loop through files in reanalysis archive linking config, in, out files
    merged_file = config['output_dir'] + config['model_name'] + '/WrPMIP_Pan-Arctic_' + config['model_name'] + '_' + sim_type + '.zarr'
    # combine processing info into list
    info_list = [config, sim_type, sim_files, merged_file]
    return info_list 

# function to fix timesteps in preprocess step and subset to variables of interest
def preprocess_JSBACH(ds, config):
    # replace time indexes, decimal numbers not allowed in cftime, also rh ends with fillvalues
    # generally a dimension shouldnt have fill values it should simply be missing
    if ds.sizes['time'] >= 1000:
        # if index is daily then replace with integers (this fixes decimal issue and fill value issue
        ds = ds.assign_coords({"time": range(730, 8765+1)})
    # replace attributes that are lost from the coord assignment step
    ds.time.attrs['standard_name'] = 'time'
    ds.time.attrs['units'] = 'days since 1998-01-01 00:00:00'
    ds.time.attrs['calendar'] = 'proleptic_gregorian'
    ds.time.attrs['axis'] = 'T'
    # reverse lat index
    ds = ds.sortby('lat', ascending=True) #reindex(lat=ds.lat[::-1])
    return ds

# preprocess files that are parsed by time
def preprocess_ecosys(ds, config, sim_type):
        # copy the file
        dsc = ds.copy(deep=True)
        # decode to cftime
        dsc = xr.decode_cf(dsc, use_cftime=True)
        # save time values outside of dataset
        ds_time = dsc.time.values
        # create and fill empty list with variables
        ds_list = []
        for i in dsc.data_vars:
            ds_sub = dsc[i]
            ds_list.append(ds_sub)
        # combine the datarrays back into a dataset
        dsc = xr.combine_by_coords(ds_list)
        # rename doy to time for dim and coord 
        dsc = dsc.rename_dims({'doy': 'time'})
        dsc = dsc.rename({'doy': 'time'})
        # set coord values to time index and set attributes
        dsc.coords['time'] = ds_time
        #dsc['time'].attrs['long_name'] = "time"
        #dsc['time'].attrs['long_name'] = "time"
        #dsc['time'].attrs['units'] = 'days since 1901-01-01 00:00:00'
        # calculate TotalResp and remake attributes
        if 'ECO_RH' in dsc.data_vars:
            dsc['TotalResp'] = dsc['ECO_RH'] + (dsc['ECO_GPP'] - dsc['ECO_NPP'])
            dsc['TotalResp'].attrs['long_name'] = 'Autotrophic + heterotrophic respiration'
            dsc['TotalResp'].attrs['units'] = 'gC m-2 day-1'
        # reorder lat
        dsc = dsc.sortby('lat', ascending=True)
        # change to noleap calendar to remove leapdays
        dsc = dsc.convert_calendar('noleap', use_cftime=True)
        # return preprocessed file to open_mfdataset
        return dsc

# preprocess files that are parsed by time
def preprocess_time(ds, config):
    var_keep = config['subset_vars']
    return ds[var_keep]

# preprocess files that are parsed by variable
def preprocess_var(ds, config):
    # return only subset variables of interest from config file
    return ds

# merge files depending on simulation outputs
def process_simulation_files(f, top_config):
    # read in config, input files, and output file
    config = f[0]
    sim_type = f[1]
    sim_files = f[2]
    out_file = f[3]
    # check if simulation dataset exists for model
    data_check = 'has_' + sim_type
    # context manager to shut up dask chunk warning
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # check if dataset exists
        if config[data_check] == "True":
            with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'w') as pf:
                print(f, file=pf)
            if sim_type == 'b1':
                chunk_read = config['nc_read']['b1_chunks']
                zarr_chunk_out = 150
            else:
                chunk_read = config['nc_read']['b2_chunks']
                zarr_chunk_out = 365
            # assign engine used to open netcdf files from config
            engine = config['nc_read']['engine']
            # set kwargs to mask_and_scale=True and decode_times=False for individual files passed to open_mfdataset
            kwargs = {"mask_and_scale": False, "decode_times": False}
            # match the combination type with how files should be merged
            match config['merge_type']:
                case 'time':
                    # create functools.partial function to pass subset variable through to preprocess
                    partial_time = partial(preprocess_time, config=config) 
                    if config['model_name'] == 'UVic-ESCM':
                        # open using mfdataset which will auto merge variables, but preprocess away incorrect time indexes 
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_time, **kwargs)
                        # extract fill value from ra/rh
                        fill_value = ds['L_veggpp'].attrs['FillValue']
                        # remove all fill values before calculating TotalResp
                        ds['L_veggpp'] = ds['L_veggpp'].where(ds['L_veggpp'] != fill_value)
                        ds['L_vegnpp'] = ds['L_vegnpp'].where(ds['L_vegnpp'] != fill_value)
                        ds['L_soilresp'] = ds['L_soilresp'].where(ds['L_soilresp'] != fill_value)
                        # calculate auto_resp
                        ds['auto_resp'] = ds['L_veggpp'] - ds['L_vegnpp']
                        # calculate TotalResp as auto + hetero resp
                        ds['TotalResp'] = ds['auto_resp'] + ds['L_soilresp']
                        ds['TotalResp'].encoding['_FillValue'] = fill_value
                        ds['L_gndtemp'].encoding['_FillValue'] = fill_value
                        ds = ds.drop_vars(['L_veggpp','L_vegnpp','L_soilresp','auto_resp'])
                        # choose arctic crass pft to reduce dimensions of TotalResp
                        ds = ds.sel(pft=3, method="nearest")
                        ds = ds.drop_vars('pft')
                        with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                            print(ds, file=pf)
                    else:
                        # open using mfdataset and merge using combine_by_coords
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_time, **kwargs)
                case 'variables':
                    if config['model_name'] == 'JSBACH':
                        # create functools.partial function to pass subset variable through to preprocess
                        partial_JSBACH = partial(preprocess_JSBACH, config=config) 
                        # open using mfdataset which will auto merge variables, but preprocess away incorrect time indexes 
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_JSBACH, **kwargs)
                        # extract fill value from ra/rh
                        fill_value = ds['ra'].attrs['_FillValue']
                        d_type = ds['ra'].encoding['dtype']
                        orig_shape = ds['ra'].encoding['original_shape']
                        # remove all fill values before calculating TotalResp
                        ds['ra'] = ds['ra'].where(ds['ra'] != fill_value)
                        ds['rh'] = ds['rh'].where(ds['rh'] != fill_value)
                        # calculate total resp and assign _FillValue
                        ds['TotalResp'] = ds['ra'] + ds['ra']
                        ds['TotalResp'].encoding['_FillValue'] = fill_value
                        ds['TotalResp'].encoding['dtype'] = d_type
                        ds['TotalResp'].encoding['original_shape'] = orig_shape
                        # clear soil_temperatures encoding
                        fill_value = ds['soil_temperature'].attrs['_FillValue']
                        d_type = ds['soil_temperature'].encoding['dtype']
                        orig_shape = ds['soil_temperature'].encoding['original_shape']
                        file_source = ds['soil_temperature'].encoding['source']
                        ds['soil_temperature'].encoding = {}
                        ds['soil_temperature'].attrs = {}
                        ds['soil_temperature'].encoding['source'] = file_source
                        ds['soil_temperature'].encoding['dtype'] = d_type
                        ds['soil_temperature'].encoding['original_shape'] = orig_shape
                        ds['soil_temperature'].encoding['_FillValue'] = fill_value
                        # subset to variables of interest
                        ds = ds[config['subset_vars']]
                    else:
                        # create functools.partial function to pass subset variable through to preprocess
                        partial_var = partial(preprocess_var, config=config) 
                        # auto merge by variables
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_var, **kwargs)
                        ds = ds[config['subset_vars']]
                case 'ecosys':
                    partial_ecosys = partial(preprocess_ecosys, config=config, sim_type=sim_type) 
                    # open using mfdataset which will auto merge variables, but preprocess away incorrect time indexes 
                    ds1 = xr.open_mfdataset(sim_files[0], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ecosys, **kwargs)
                    ds2 = xr.open_mfdataset(sim_files[1], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ecosys, **kwargs)
                    ds3 = xr.open_mfdataset(sim_files[2], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ecosys, **kwargs)
                    ds4 = xr.open_mfdataset(sim_files[3], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ecosys, **kwargs)
                    # merge TotalResp and SoilTemps
                    ds = xr.merge([ds1,ds2,ds3,ds4])
                    with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                        print(ds, file=pf)
                    # loop through soil layer dataarrays, for each soil layer add a depth coord and expand dim
                    ds_list = []
                    layer_iter = 0
                    layer_interface = config['soil_depths']
                    layers=['TMAX_SOIL_1','TMAX_SOIL_2','TMAX_SOIL_3','TMAX_SOIL_4','TMAX_SOIL_5', \
                            'TMAX_SOIL_6','TMAX_SOIL_7','TMAX_SOIL_8','TMAX_SOIL_9','TMAX_SOIL_10', 'TMAX_SOIL_11'] 
                    for layer in layers:
                        # expand a dimension to include site and save to list
                        ds2 = ds.assign_coords({'SoilDepth': layer_interface[layer_iter]})
                        ds2[layer].attrs = {}
                        ds2[layer] = ds2[layer].expand_dims('SoilDepth')
                        ds2 = ds2.rename({layer: 'SoilTemp'})
                        ds_list.append(ds2['SoilTemp'])
                        layer_iter += 1
                    # merge all the SoilTemp layers into 4D dataarray
                    ds_soiltemp = xr.combine_by_coords(ds_list)
                    with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                        print(ds, file=pf)
                    ds = ds.drop_vars(layers)
                    with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                        print(ds, file=pf)
                    ds = xr.merge([ds, ds_soiltemp])
                    with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                        print(ds, file=pf)
                    # add long name and unit attributes
                    ds['SoilTemp'].attrs['long_name'] = 'Soil temperature by layer'
                    ds['SoilTemp'].attrs['units'] = 'Degree C' 
                    # reorde variable dimenions and remove chunking
                    ds['SoilTemp'] = ds['SoilTemp'].transpose('time', 'SoilDepth', 'lat', 'lon')
                    ds['TotalResp'] = ds['TotalResp'].transpose('time', 'lat', 'lon')
                    #ds = ds.chunk({'time': 300, 'SoilDepth': -1, 'lat': -1, 'lon': -1})
                    # subset to variables of interest
                    ds = ds[config['subset_vars']]
                    with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                        print(ds, file=pf)
                    # change data variables to float32 to save space
                    ds = ds.astype('float32')
                    with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                        print(ds, file=pf)
                    # assign encoding fillvalues
                    for var in ds.data_vars:
                        ds[var].encoding['_FillValue'] = config['nc_write']['fillvalue']
                    with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                        print(ds, file=pf)
                        print(ds['SoilTemp'].encoding, file=pf)
                        print(ds['TotalResp'].encoding, file=pf)
                case 'none':
                    # open single file and save to new file name
                    ds = xr.open_dataset(sim_files[0], engine=engine, chunks=chunk_read, mask_and_scale=False, decode_times=False)
                    # subset variables
                    ds = ds[config['subset_vars']]
            # rename variables
            ds = ds.rename(config['rename_subset'])
            # fix models that have non-CF time conforming dates
            match config['model_name']:
                case 'UVic-ESCM':
                    if sim_type in ['b1']:
                        # dates incorrectly start from non-cftime standards of 0-1-1
                        # to fix this I read in xarray attributes and update reference time
                        units, reference_date = ds.time.attrs['units'].split('since') 
                        reference_date = '1901-01-01'
                        # the numbers themselves are also wrong, must be integers, but are instread decimal dates
                        # to deal with this I have to remake the index with xr.cftime_range
                        new_index = xr.cftime_range(start=reference_date, periods=ds.sizes['time'], \
                            calendar='noleap', freq='AS-JUL')
                        # reassign new index/
                        ds = ds.assign_coords({'time': new_index})
                    else:
                        # fix non-cftime conforming dates similarly for b2,otc,sf
                        units, reference_date = ds.time.attrs['units'].split('since') 
                        reference_date = '2000-01-01'
                        # remake integer time index in cf conforming integers
                        new_index = xr.cftime_range(start=reference_date, periods=ds.sizes['time'], \
                            calendar='noleap', freq='5D')
                        # to recreate UVic date exactly have to shift first day of the year to the 2.5th day
                        ds = ds.assign_coords({'time': new_index})
                    # assign depth intergers to meter depths
                    ds = ds.assign_coords({'SoilDepth': config['soil_depths']})
            # decode cftime
            ds = xr.decode_cf(ds, use_cftime=True)
            # change calendar for JSBACH
            if config['model_name'] in ['JSBACH']:
                var_enc = {}
                # save encodings 
                for var in ds.data_vars:
                    item_enc = {}
                    for enc, enc_val in ds[var].encoding.items():
                        item_enc[enc] = enc_val
                    var_enc[var] = item_enc
                # convert calendar, removes encoding
                ds = ds.convert_calendar('noleap')
                # replace encoding
                for var in ds.data_vars:
                    for enc, enc_val in var_enc[var].items():
                        ds[var].encoding[enc] = enc_val 
            # multindex and unstack lndgrids so that lat lon become dimensions instead
            if config['model_name'] in ['ELM2-NGEE','LPJ-GUESS']:
                ds = ds.set_index(lndgrid=['lat','lon'])
                ds = ds.unstack() 
            # convert lon from -180-180 to 0-360
            if config['model_name'] in ['ELM2-NGEE', 'ecosys']:
                ds = ds.assign_coords({'lon': (ds.lon % 360)})
                ds = ds.sortby('lon', ascending=True)
                with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                    print('adjusted lon index', file=pf)
                    print(ds['SoilTemp'].encoding, file=pf)
                    print(ds['TotalResp'].encoding, file=pf)
            # adjust units
            for var in config['data_units']:
                # remove _FillValues and missing values
                if config['model_name'] != 'UVic-ESCM':
                    ds[var] = ds[var].where(ds[var] != ds[var].encoding['_FillValue'])
                else: 
                    ds[var] = ds[var].where(ds[var] < 2e36)
                # scale units
                if config['data_units'][var]['scale_type'] == 'add':
                    ds[var] = ds[var] + config['data_units'][var]['scale_value']
                    ds[var] = ds[var].assign_attrs({"units": config['data_units'][var]['units']})
                if config['data_units'][var]['scale_type'] == 'multiply':
                    ds[var] = ds[var] * config['data_units'][var]['scale_value']
                    ds[var] = ds[var].assign_attrs({"units": config['data_units'][var]['units']})
            for var in config['coords_units']:
                # scale units
                if config['coords_units'][var]['scale_type'] == 'add':
                    ds[var] = ds[var] + config['coords_units'][var]['scale_value']
                    ds[var] = ds[var].assign_attrs({"units": config['coords_units'][var]['units']})
                if config['coords_units'][var]['scale_type'] == 'multiply':
                    ds[var] = ds[var] * config['coords_units'][var]['scale_value']
                    ds[var] = ds[var].assign_attrs({"units": config['coords_units'][var]['units']})
            # add empty dataframe for missing values like WTD/ALT
            missing_vars = [i for i in top_config['combined_vars'] if i not in ds.data_vars]
            for var in missing_vars:
                ds[var] = ds['TotalResp'].copy(deep=True)
                ds[var].loc[:] = np.nan
            with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                print('missing vars added with all nans:',file=pf)
                print(missing_vars, file=pf)
                print(ds, file=pf)
            # create 10cm soil temp average 
            # !!!!!!!!! THIS IS INCORRECT AVERAGE UNTIL REDONE WEHN LAYER THICKNESS/INTERFACES ARE KNOWN !!!!!!
            # need to create function to weight temp average by layer thickness and deal with partial thickness when 10cm is between nodes
            ds['SoilTemp_10cm'] = ds['SoilTemp'].sel(SoilDepth=slice(0.0,0.11)).mean(dim='SoilDepth')           
            with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                print('ds including 10cm soil temp:',file=pf)
                print(ds, file=pf)
            # rescale from g C m-2 s-1 to g C m-2 day-1
            ds['TotalResp'] = ds['TotalResp'] * 86400  
            # set zarr compression and encoding
            compress = Zstd(level=3) #, shuffle=Blosc.BITSHUFFLE)
            # clear all chunk and fill value encoding/attrs
            for var in ds:
                try:
                    del ds[var].encoding['chunks']
                except:
                    pass
                try:
                    del ds[var].encoding['_FillValue']
                except:
                    pass
                try:
                    del ds[var].attrs['_FillValue']
                except:
                    pass
            dim_chunks = {
                'time': zarr_chunk_out,
                'SoilDepth': -1,
                'lat': -1,
                'lon': -1}
            ds = ds.chunk(dim_chunks) 
            with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                print('data rechunked',file=pf)
                print(ds, file=pf)
            encode = {var: {'_FillValue': np.NaN, 'compressor': compress} for var in ds.data_vars}
            #    'SoilTemp': {'_FillValue': np.NaN, 'compressor': compress},
            #    'TotalResp': {'_FillValue': np.NaN, 'compressor': compress}}
            #comp = dict(compressor = compress, _FillValue=np.NaN)
            # encoding
            #encode = {var: comp for var in ds.data_vars}
            # output to zarr file
            ds.to_zarr(out_file, encoding=encode, mode="w")
            with open(Path(config['output_dir'] + config['model_name'] + '/' + sim_type + '_debug.txt'), 'a') as pf:
                print('encoded output using:', file=pf)
                print(encode, file=pf)
            # close file connections
            try:
                ds.close()
            except Exception as error:
                print(error)
                pass

# create sub folder for site files
def site_dir_prep(f):
    # read config file
    config = read_config(f)
    # remove cleaned model folder
    #rmv_dir(config['output_dir']+config['model_name']+'/sites')
    # remake directory per folder 
    Path(config['output_dir']+config['model_name']+'/sites').mkdir(parents=True, exist_ok=True)
    Path(config['output_dir']+config['model_name']+'/sites').chmod(0o762)

# collect file lists per model and simulations based on configuration files
def subsample_site_list(f, gps):
    # read config file
    config = read_config(f[0])
    # read the simulation type
    sim_type = f[1]
    # read all CRUJRA input file names from reanalysis directory
    dir_name = config['output_dir'] + config['model_name'] + '/'
    zarr_file = glob.glob("{}*{}*.zarr".format(dir_name, sim_type))
    with open(Path(config['output_dir'] + config['model_name'] + '/sites/' + sim_type + '_debug.txt'), 'w') as pf:
        print(zarr_file, file=pf)
    # file to output a site subset
    site_file = config['output_dir'] + config['model_name'] + '/sites/WrPMIP_sites_' + config['model_name'] + '_' + sim_type + '.zarr'
    # combine processing info into list
    if (config['model_name'] == 'JSBACH') & (sim_type == 'b1'):
        info_list = [config, sim_type, gps, ['None'], site_file]
    elif (config['model_name'] == 'ecosys') & (sim_type == 'b1'):
        info_list = [config, sim_type, gps, ['None'], site_file]
    else:
        info_list = [config, sim_type, gps, zarr_file, site_file]
    return info_list 

# read in data to combine each models 
def subsample_sites(f):
    # read in config, input files, and output filc
    config = f[0]
    sim_type = f[1]
    site_gps = f[2]
    regional_file = f[3][0]
    site_file = f[4]
    # check if simulation dataset exists for model
    data_check = 'has_' + sim_type
    # if the data exists read in the zarr file and select sites
    if config[data_check] == "True":
        # read in zarr file to dataset
        ds = xr.open_zarr(regional_file, use_cftime=True, mask_and_scale=False)
        for var in ds:
            del ds[var].encoding['chunks']
        #if sim_type == 'b1':
        #    start_time, end_time = config['sim_length']['b1']
        #    ds = ds.sel(time=slice(start_time, end_time))
        #elif sim_type in ['b2','otc','sf']:
        #    start_time, end_time = config['sim_length']['b2']
        #    ds = ds.sel(time=slice(start_time, end_time))
        with open(Path(config['output_dir'] + config['model_name'] + '/sites/' + sim_type + '_debug.txt'), 'a') as pf:
            print(ds, file=pf)
        # make single netcdf output for jeralyn/ILAMB
        #if (config['model_name'] == 'CLM5'):
        #    ds_copy = ds.copy(deep=True)
        #    ds_copy = ds_copy[['TotalResp','TSOI_10CM']]
        #    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
        #        complevel=config['nc_write']['complevel'],_FillValue=config['nc_write']['fillvalue'])
        #    # encoding
        #    encoding = {var: comp for var in ds_copy.data_vars}
        #    # write site's bias corrected netcdf
        #    ds_copy.to_netcdf(Path('/projects/warpmip/shared/CLM5_2001-2021_pan-arctic_' + sim_type + '_er-soiltemp.nc'), mode="w", \
        #        format=config['nc_write']['format'],\
        #        engine=config['nc_write']['engine'])   
        #    del ds_copy
        #    # remove TSOI_10CM
        #    ds = ds.drop_vars(['TSOI_10CM'])
        # create list for ds after gps selection
        ds_list = []
        for site in site_gps:
            # try to mask

            # choose the neatest grid cell to the site
            ds_sub = ds.sel(lon=site_gps[site]['lon'], lat=site_gps[site]['lat'], method='nearest').copy()
            with open(Path(config['output_dir'] + config['model_name'] + '/sites/' + sim_type + '_debug.txt'), 'a') as pf:
                print(ds_sub, file=pf)
                print(site, file=pf)
            # expand a dimension to include site and save to list
            ds_sub = ds_sub.assign_coords({'site': site})
            ds_sub = ds_sub.expand_dims('site')
            ds_sub = ds_sub.reset_coords(['lat','lon'])
            ds_sub['lat'] = ds_sub['lat'].expand_dims('site')
            ds_sub['lon'] = ds_sub['lon'].expand_dims('site')
            ds_sub = ds_sub.chunk({'time': -1})
            with open(Path(config['output_dir'] + config['model_name'] + '/sites/' + sim_type + '_debug.txt'), 'a') as pf:
                print(ds_sub, file=pf)
            ds_list.append(ds_sub)
        with open(Path(config['output_dir'] + config['model_name'] + '/sites/' + sim_type + '_debug.txt'), 'a') as pf:
            print('datasets appended', file=pf)
        # combine site dimension to have multiple sites
        #try:
        ds_sites = xr.combine_by_coords(ds_list)
        #except Exception as error:
        #    with open(Path(config['output_dir'] + config['model_name'] + '/sites/' + sim_type + '_debug.txt'), 'a') as pf:
        #        print(error, file=pf)
        with open(Path(config['output_dir'] + config['model_name'] + '/sites/' + sim_type + '_debug.txt'), 'a') as pf:
            print('merged datasets',file=pf)
            print(ds_sites, file=pf)
        # set zarr compression and encoding
        #compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        dim_chunks = {
            'SoilDepth': -1,
            'time': -1,
            'site': -1}
        ds_sites = ds_sites.chunk(dim_chunks) 
        with open(Path(config['output_dir'] + config['model_name'] + '/sites/' + sim_type + '_debug.txt'), 'a') as pf:
            print('data rechunked',file=pf)
            print(ds_sites, file=pf)
        compress = None #Zstd(level=3) #, shuffle=Blosc.BITSHUFFLE)
        encode = {var: {'_FillValue': np.NaN, 'compressor': compress} for var in ds.data_vars}
        #encode = {
        #    'SoilTemp': {'_FillValue': np.NaN, 'compressor': compress},
        #    'TotalResp': {'_FillValue': np.NaN, 'compressor': compress},
        #    'lat': {'_FillValue': np.NaN, 'compressor': compress}, 
        #    'lon': {'_FillValue': np.NaN, 'compressor': compress}} 
        # remove encoding of fill value before resaving
        for var in ds_sites:
            try:
                del ds_sites[var].encoding['_FillValue']
            except:
                pass
            try:
                del ds_sites[var].attrs['_FillValue']
            except:
                pass
        #with open(Path(config['output_dir'] + config['model_name'] + '/sites/' + sim_type + '_debug.txt'), 'a') as pf:
        #    print('past comp',file=pf)
        #    print(comp, file=pf)
        ## encoding
        #encoding = {var: comp for var in ds_sites.data_vars}
        #with open(Path(config['output_dir'] + config['model_name'] + '/sites/' + sim_type + '_debug.txt'), 'a') as pf:
        #    print('past encoding',file=pf)
        #    print(encoding, file=pf)
        # write zarr
        ds_sites.to_zarr(site_file, encoding=encode,  mode="w")
        with open(Path(config['output_dir'] + config['model_name'] + '/sites/' + sim_type + '_debug.txt'), 'a') as pf:
            print('past zarr out',file=pf)
            print(ds_sites, file=pf)
        # close file connections
        ds.close()

# create sub folder for site files
def site_sim_dir_prep(f):
    # read config file
    config = read_config(f)
    # remove cleaned model folder
    #rmv_dir(config['output_dir']+config['model_name']+'/sites_sims')
    # remake directory per folder 
    Path(config['output_dir']+config['model_name']+'/sites_sims').mkdir(parents=True, exist_ok=True)
    Path(config['output_dir']+config['model_name']+'/sites_sims').chmod(0o762)

# combine b2/otc/sf from same models as another dimension simulation
def aggregate_simulation_types(f):
    # read config file
    config = read_config(f)
    config = read_config(f)
    # open b2, otc, sf zarr files, add simulation dimension
    ds_list = []
    for sim in ['b2','otc','sf']:
        try:
            # create name of file to open fro site folder
            ds_file = config['output_dir'] + config['model_name'] + '/sites/WrPMIP_sites_' + config['model_name'] + '_' + sim + '.zarr'
            out_file = config['output_dir'] + config['model_name'] + '/sites_sims/WrPMIP_sites_' + config['model_name'] + '_warming.zarr'
            # open site file
            ds = xr.open_zarr(ds_file, use_cftime=True, mask_and_scale=False) 
            #for var in ds:
            #    del ds[var].encoding['chunks']
            # assign simulation coordinate
            ds = ds.assign_coords({'sim': sim})
            ds = ds.expand_dims('sim')
            # append dataset to list for later merging
            ds_list.append(ds)
        except Exception as error:
            pass
    # merge dataset
    ds_sites = xr.merge(ds_list)
    with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/warming_debug.txt'), 'a') as pf:
        print(ds_sites, file=pf)
    # create dict of delta names for creation
    new_name_dict = {"TotalResp": 'deltaTotalResp', "SoilTemp_10cm": 'deltaSoilTemp'}
    # convert response values of interest into delta values
    ds_list = []
    for sim_sel in ['otc','sf']:
        try:
            # select warming treatment, subtract control
            ds_sub = ds_sites.sel(sim=sim_sel).copy()
            ds_control = ds_sites.sel(sim='b2').copy()
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/warming_debug.txt'), 'a') as pf:
                print(ds_sub, file=pf)
                print(ds_control, file=pf)
            # create list of newly created delta variable names and calculate warming - control
            for response in new_name_dict.keys():
                ds_sub[new_name_dict[response]] = ds_sub[response] - ds_control[response]
            # calculate Q10
            warmed_temp = ds_sub['SoilTemp_10cm']
            control_temp = ds_control['SoilTemp_10cm']
            ds_sub['q10'] = (ds_sub['TotalResp']/ds_control['TotalResp']) ** (10/(warmed_temp - control_temp))
            keep_list = ['q10']
            keep_list.extend(list(new_name_dict.values())) 
            # subset to only new variables
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/warming_debug.txt'), 'a') as pf:
                print(ds_sub, file=pf)
            ds_sub = ds_sub[keep_list]
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/warming_debug.txt'), 'a') as pf:
                print(ds_sub, file=pf)
            # add simulation dimension back and expand dims to recreate same NetCDF shape
            ds_sub = ds_sub.assign_coords({'sim': sim_sel})
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/warming_debug.txt'), 'a') as pf:
                print(ds_sub, file=pf)
            ds_sub = ds_sub.expand_dims('sim')
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/warming_debug.txt'), 'a') as pf:
                print(ds_sub, file=pf)
            # append dataset to list for merging
            ds_list.append(ds_sub)
        except Exception as error:
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/warming_debug.txt'), 'a') as pf:
                print(error, file=pf)
            pass
    # recreate original dataarray shape
    ds_delta = xr.combine_by_coords(ds_list)
    with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/warming_debug.txt'), 'a') as pf:
        print(ds_delta, file=pf)
    # add delta responses dataarrays to original dataset
    for var in keep_list:
        ds_sites[var] = ds_delta[var]
    # set zarr compression and encoding
    #compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    #comp = dict(chunks={})
    # encoding
    #encoding = {var: comp for var in ds_sites.data_vars}
    # write zarr
    #ds_sites.chunk({'time': -1}).to_zarr(out_file, mode="w")
    ds_sites.to_zarr(out_file, mode="w")

# create sub folder for site files
def combined_dir_prep(f):
    # read config file
    config = read_config(f)
    # clear and remake combined folder
    #rmv_dir(config['output_dir']+'/combined')
    # remake directory per folder 
    Path(config['output_dir']+'/combined').mkdir(parents=True, exist_ok=True)
    Path(config['output_dir']+'/combined').chmod(0o762)

# combine all models
def aggregate_models_warming(f):
    # open site_sim files for each model, add model dimension
    ds_list = []
    for model in f:
        try:
            # read config file
            config = read_config(model)
            # create file path
            ds_file = config['output_dir'] + config['model_name'] + '/sites_sims/WrPMIP_sites_' + config['model_name'] + '_warming.zarr'
            # open site warming period file
            ds = xr.open_zarr(ds_file, use_cftime=True, mask_and_scale=False) 
            # test subsetting
            #ds = ds[['TotalResp','SoilTemp']]
            # assign simulation coordinate
            ds = ds.assign_coords({'model': config['model_name']})
            ds = ds.expand_dims('model')
            # append dataset to list for later merging
            ds_list.append(ds)
        except Exception as error:
            pass
    # merge dataset
    ds_sites = xr.merge(ds_list)
    with open(Path(config['output_dir'] + '/combined/2000-2021_debug.txt'), 'a') as pf:
        print(ds_sites, file=pf)
    # set zarr compression and encoding
    #compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    #comp = dict(chunks={})
    # encoding
    #encoding = {var: comp for var in ds_sites.data_vars}
    # write zarr
    out_file = config['output_dir'] + 'combined/WrPMIP_all_models_sites_2000-2021.zarr'
    #ds_sites.chunk({'time': -1}).to_zarr(out_file, mode="w")
    ds_sites.to_zarr(out_file, mode="w")
    
# combine all models
def aggregate_models_baseline(f):
    # open site_sim files for each model, add model dimension
    ds_list = []
    for model in f:
        # read config file
        config = read_config(model)
        # if the data exists read in the zarr file and select sites
        if config['has_b1'] == 'True':
            # create file path
            ds_file = config['output_dir'] + config['model_name'] + '/sites/WrPMIP_sites_' + config['model_name'] + '_b1.zarr'
            # open site warming period file
            ds = xr.open_zarr(ds_file, use_cftime=True, mask_and_scale=False) 
            #for var in ds:
            #    del ds[var].encoding['chunks']
            # assign simulation coordinate
            ds = ds.assign_coords({'model': config['model_name']})
            ds = ds.expand_dims('model')
            # append dataset to list for later merging
            ds_list.append(ds)
    # if list isnt empty continue
    if len(ds_list) > 0:    
        # merge dataset
        ds_sites = xr.merge(ds_list)
        with open(Path(config['output_dir'] + '/combined/1901-2000_debug.txt'), 'a') as pf:
            print(ds_sites, file=pf)
        # set zarr compression and encoding
        #compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        #comp = dict(chunks={})
        # encoding
        #encoding = {var: comp for var in ds_sites.data_vars}
        # write zarr
        out_file = config['output_dir'] + 'combined/WrPMIP_all_models_sites_1901-2000.zarr'
        #ds_sites.chunk({'time': -1}).to_zarr(out_file, mode="w")
        ds_sites.to_zarr(out_file, mode="w")    
  
# create sub folder for site files
def plot_dir_prep(f, config_file):
    # read config file
    config = read_config(config_file)
    # clear and remake combined folder
    #rmv_dir(config['output_dir']+'/combined/' + f)
    # remake directory per folder 
    Path(config['output_dir'] + '/combined/' + f).mkdir(parents=True, exist_ok=True)
    Path(config['output_dir'] + '/combined/' + f).chmod(0o762)

# line plots using plotnine
def plotnine_lines(f, config, out_dir):
    # bring in and select data similarly as graph_lines
    # read in inputs
    sites = f[0]
    var = f[1]
    models = f[2]
    sims = f[3]
    plot_num = f[4]
    # set file read location from merged daily data
    combined_file = config['output_dir'] + 'combined/WrPMIP_all_models_sites_2000-2021.zarr'
    ds = xr.open_zarr(combined_file, use_cftime=True, mask_and_scale=True)
    # function to subset only summer months
    def is_summer(month):
        return (month >= 5) & (month <= 9)
    # check maximum Total Respiration value for plots
   # ds_mean = ds[var].sel(time=is_summer(ds[var].time.dt.month)).resample(time='A').mean('time')
   # annual_var_max = np.unique(ds_mean.max().values).max()
   # annual_var_min = np.unique(ds_mean.min().values).min()
   # daily_var_max = np.unique(ds[var].max().values).max()
   # daily_var_min = np.unique(ds[var].min().values).min()
    # subsample data
    da = ds[var].sel(site=sites, model=models, sim=sims, time=slice('2000-01-01','2020-12-31'))
    # deal with variable site/model/sims and variable depth increments
    listed = False
    groups = 'site'
    other_vars = ['sim', 'model']
    if isinstance(var, str) & isinstance(models, str) & isinstance(sites, str) & isinstance(sims, str):
        file_name = var + '_by_time_' + models + '_' + sites[:7] + '_' + sims
    if isinstance(sites, list):
        sites_chopped = []
        for i in sites:
            sites_chopped.append(i[:7]) 
        file_part = models + '_' + '_'.join(sites_chopped) + '_' + sims 
        listed = True
        groups = 'site'
        other_vars = ['model', 'sim']
    if isinstance(models, list):
        file_part = '_'.join(models) + '_' + sites[:7] + '_' + sims
        listed = True
        groups = 'model'
        other_vars = ['sim', 'site']
    if isinstance(sims, list):
        file_part = models + '_' + sites[:7] + '_' + '_'.join(sims)
        listed = True
        groups = 'sim'
        other_vars = ['model', 'site']
    # create file name if any lists present
    if listed == True:
        file_name = var + '_by_time_' + file_part
    # manipulate data for ggplot input format
    # daily data
    pd_df = da.to_dataframe()
    pd_df = pd_df.reset_index()
    pd_df['ID'] =  pd_df[groups].astype(str).str.cat(pd_df[other_vars].astype(str), sep='_')
    pd_df = pd_df.set_index('ID', drop=False)
    pd_df[var].replace([np.inf, -np.inf], np.nan, inplace=True)
    pd_df = pd_df.dropna()
    # annulize for second plot
    da = da.sel(time=is_summer(da['time.month'])) 
    ds_annual = da.resample(time='AS').sum('time')
    # annudal data
    pd_df_annual = ds_annual.to_dataframe()
    pd_df_annual = pd_df_annual.reset_index()
    pd_df_annual['ID'] =  pd_df_annual[groups].astype(str).str.cat(pd_df_annual[other_vars].astype(str), sep='_')
    pd_df_annual = pd_df_annual.set_index('ID', drop=False)
    pd_df_annual[var].replace([np.inf, -np.inf], np.nan, inplace=True)
    pd_df_annual = pd_df_annual.dropna()
    # output csv to inspect
    pd_df.to_csv(config['output_dir'] + 'combined/' + out_dir + '/' + file_name + '.csv')
    pd_df_annual.to_csv(config['output_dir'] + 'combined/' + out_dir + '/' + file_name + '_annual.csv')
    # custome color map
    plot_colors = ['blue','gold','red','olive','purple','orange','green','cyan','magenta','brown','gray','black']
    # create color column for consistent colors on plots

    # create axis labels
    if var == 'TotalResp':
        x_label = r'time (day)'
        y_label = r'Summer Ecosystem Respiration (g C $m^{-2}$ $day^{-1}$)'
    elif var == 'q10':
        x_label = r'time (day)'
        y_label = r'q10 (unitless)'
    # plotnine graph daily
    p = ggplot(pd_df, aes(x='time', y=var, group='ID', color='ID')) + \
        labs(x=x_label, y=y_label) + \
        geom_line() + \
        scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y')) + \
        scale_color_manual(plot_colors) + \
        guides(color = guide_legend(reverse=True)) + \
        theme_bw() + \
        theme(
            axis_text_x = element_text(angle = 90),
            axis_line = element_line(colour = "black"),
            panel_grid_major = element_blank(),
            panel_grid_minor = element_blank(),
            panel_border = element_blank(),
            panel_background = element_blank()
        )
        #scale_y_continuous(limits=(daily_var_min,daily_var_max)) + \
    # plotnine graph annual
    p2 = ggplot(pd_df_annual, aes(x='time', y=var, group='ID', color='ID')) + \
        labs(x=x_label, y=y_label) + \
        geom_line() + \
        scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y')) + \
        scale_color_manual(plot_colors) + \
        guides(color = guide_legend(reverse=True)) + \
        theme_bw() + \
        theme(
            axis_text_x = element_text(angle = 90),
            axis_line = element_line(colour = "black"),
            panel_grid_major = element_blank(),
            panel_grid_minor = element_blank(),
            panel_border = element_blank(),
            panel_background = element_blank()
        )
        #scale_y_continuous(limits=(annual_var_min,annual_var_max)) + \
    # output graph
    p.save(filename=file_name+'.png', path=config['output_dir']+'combined/'+out_dir, dpi=300)
    p2.save(filename=file_name+'_annual.png', path=config['output_dir']+'combined/'+out_dir, dpi=300)

# line plots using plotnine
def plotnine_scatter(f, config, out_dir):
    # bring in and select data similarly as graph_lines
    # read in inputs
    sites = f[0]
    var = f[1]
    models = f[2]
    sims = f[3]
    plot_num = f[4]
    # set file read location from merged daily data
    combined_file = config['output_dir'] + 'combined/WrPMIP_all_models_sites_2000-2021.zarr'
    func_curves_file = '/projects/warpmip/shared/ted_data/response_curve_points2.csv'
    ds = xr.open_zarr(combined_file, use_cftime=True, mask_and_scale=True)
    func_curves = pd.read_csv(func_curves_file)
    # function to subset only summer months
    def is_summer(month):
        return (month >= 5) & (month <= 7)
    # check maximum Total Respiration value for plots
    #ds = ds.where(ds['SoilTemp'] < 150)
    #ds_mean = ds[['TotalResp','SoilTemp']].sel(time=is_summer(ds[['TotalResp','SoilTemp']].time.dt.month)).resample(time='M').mean('time')
    #monthly_soilT_max = np.unique(ds_mean['SoilTemp'].max().values).max()
    #monthly_soilT_min = np.unique(ds_mean['SoilTemp'].min().values).min()
    #monthly_er_max = np.unique(ds_mean['TotalResp'].max().values).max()
    #monthly_er_min = np.unique(ds_mean['TotalResp'].min().values).min()
    #daily_soilT_max = np.unique(ds['SoilTemp'].max().values).max()
    #daily_soilT_min = np.unique(ds['SoilTemp'].min().values).min()
    #daily_er_max = np.unique(ds['TotalResp'].max().values).max()
    #daily_er_min = np.unique(ds['TotalResp'].min().values).min()
    # subsample data
    var_list = [var]
    var_list.extend(['SoilTemp_10cm'])
    ds = ds[var_list].sel(site=sites, model=models, sim=sims, time=slice('2000-01-01','2020-12-31'))
    ds = ds.sel(time=is_summer(ds['time.month']))
    # deal with variable site/model/sims and variable depth increments
    listed = False
    groups = 'site'
    color_base = 'none'
    other_vars = ['sim', 'model']
    if isinstance(var, str) & isinstance(models, str) & isinstance(sites, str) & isinstance(sims, str):
        file_name = var + '_by_time_' + models + '_' + sites[:7] + '_' + sims
    if isinstance(sites, list):
        sites_chopped = []
        for i in sites:
            sites_chopped.append(i[:7]) 
        file_part = models + '_' + '_'.join(sites_chopped) + '_' + sims 
        listed = True
        groups = 'site'
        color_base = 'site'
        other_vars = ['model', 'sim']
    if isinstance(models, list):
        file_part = '_'.join(models) + '_' + sites[:7] + '_' + sims
        listed = True
        groups = 'model'
        color_base = 'model'
        other_vars = ['sim', 'site']
    if isinstance(sims, list):
        file_part = models + '_' + sites[:7] + '_' + '_'.join(sims)
        listed = True
        groups = 'sim'
        color_base = 'sim'
        other_vars = ['model', 'site']
    # create file name if any lists present
    if listed == True:
        file_name = var + '_by_SoilTemp_' + file_part
    # manipulate data for ggplot input format
    # daily data
    pd_df = ds.to_dataframe()
    pd_df_test = ds.unstack()
    pd_df = pd_df.reset_index()
    #pd_df['ID'] =  pd_df[groups].astype(str).str.cat(pd_df[other_vars].astype(str), sep='_')
    pd_df['ID'] =  pd_df[groups].astype(str)
    pd_df = pd_df.sort_values(by=['ID'])
    pd_df[var].replace([np.inf, -np.inf], np.nan, inplace=True)
    pd_df = pd_df.dropna()
    # monthly aggregation
    ds_monthly = ds.resample(time='M').mean('time')
    # monthly data
    pd_df_monthly = ds_monthly.to_dataframe()
    pd_df_monthly = pd_df_monthly.reset_index()
    #pd_df_monthly['ID'] =  pd_df_monthly[groups].astype(str).str.cat(pd_df_monthly[other_vars].astype(str), sep='_')
    pd_df_monthly['ID'] =  pd_df_monthly[groups].astype(str)
    pd_df_monthly = pd_df_monthly.sort_values(by=['ID'])
    pd_df_monthly[var].replace([np.inf, -np.inf], np.nan, inplace=True)
    pd_df_monthly = pd_df_monthly.dropna()
    # custome color map
    def color_mapper(value, cmap_name='plasma', vmin=64, vmax=79):
        # norm = plt.Normalize(vmin, vmax)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap_name)  # PiYG
        rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
        color = matplotlib.colors.rgb2hex(rgb)
        return color
    #plot_colors = ['blue','gold','red','olive','purple','orange','green','cyan','magenta','brown','gray','black']
    #fill_blanks = ['#ffffff00'] * len(plot_colors)
    # make color column for plotting consistent colors
    pd_df['color'] = 'dodgerblue'
    pd_df_monthly['color'] = 'dodgerblue'
    if color_base == 'sim':
        pd_df.loc[pd_df['sim'] == 'b2', 'color'] = 'blue' 
        pd_df.loc[pd_df['sim'] == 'otc', 'color'] = 'gold' 
        pd_df.loc[pd_df['sim'] == 'sf', 'color'] = 'red'
        pd_df_monthly.loc[pd_df_monthly['sim'] == 'b2', 'color'] = 'blue' 
        pd_df_monthly.loc[pd_df_monthly['sim'] == 'otc', 'color'] = 'gold' 
        pd_df_monthly.loc[pd_df_monthly['sim'] == 'sf', 'color'] = 'red'
    elif color_base == 'model':
        pd_df.loc[pd_df['model'] == 'CLM5', 'color'] = '#cc0000' 
        pd_df.loc[pd_df['model'] == 'CLM5-ExIce', 'color'] = '#e69138' 
        pd_df.loc[pd_df['model'] == 'ELM2-NGEE', 'color'] = '#f1c232' 
        pd_df.loc[pd_df['model'] == 'ecosys', 'color'] = '#6aa84f' 
        pd_df.loc[pd_df['model'] == 'JSBACH', 'color'] = '#45818e' 
        pd_df.loc[pd_df['model'] == 'UVic-ESCM', 'color'] = '#3c78d8' 
        pd_df_monthly.loc[pd_df_monthly['model'] == 'CLM5', 'color'] = '#cc0000' 
        pd_df_monthly.loc[pd_df_monthly['model'] == 'CLM5-ExIce', 'color'] = '#e69138' 
        pd_df_monthly.loc[pd_df_monthly['model'] == 'ELM2-NGEE', 'color'] = '#f1c232' 
        pd_df_monthly.loc[pd_df_monthly['model'] == 'ecosys', 'color'] = '#6aa84f' 
        pd_df_monthly.loc[pd_df_monthly['model'] == 'JSBACH', 'color'] = '#45818e' 
        pd_df_monthly.loc[pd_df_monthly['model'] == 'UVic-ESCM', 'color'] = '#3c78d8' 
    elif color_base == 'site':
        pd_df.loc[pd_df['site'] == 'USA-Atqasuk',       'color'] = '#cc0000'#color_mapper(70.5)
        pd_df.loc[pd_df['site'] == 'USA-Utqiagvik',     'color'] = '#e69138'#color_mapper(71.3)
        pd_df.loc[pd_df['site'] == 'USA-Toolik',        'color'] = '#f1c232'#color_mapper(68.8)
        pd_df.loc[pd_df['site'] == 'USA-EightMileLake', 'color'] = '#6aa84f'#color_mapper(63.9)
        pd_df.loc[pd_df['site'] == 'SWE-Abisko',        'color'] = '#45818e'#color_mapper(68.4)
        pd_df.loc[pd_df['site'] == 'SVA-Adventdalen',   'color'] = '#3c78d8'#color_mapper(78.2)
        pd_df.loc[pd_df['site'] == 'RUS-Seida',         'color'] = '#3d85c6'#color_mapper(67.1)
        pd_df.loc[pd_df['site'] == 'GRE-Zackenburg',    'color'] = '#674ea7'#color_mapper(74.5)
        pd_df.loc[pd_df['site'] == 'GRE-Disko',         'color'] = '#a64d79'#color_mapper(69.3)
        pd_df.loc[pd_df['site'] == 'CAN-DaringLake',    'color'] = '#85200c'#color_mapper(64.9)
        pd_df_monthly.loc[pd_df_monthly['site'] == 'USA-Atqasuk',       'color'] = '#cc0000'#color_mapper(70.5) 
        pd_df_monthly.loc[pd_df_monthly['site'] == 'USA-Utqiagvik',     'color'] = '#e69138'#color_mapper(71.3)
        pd_df_monthly.loc[pd_df_monthly['site'] == 'USA-Toolik',        'color'] = '#f1c232'#color_mapper(68.8)
        pd_df_monthly.loc[pd_df_monthly['site'] == 'USA-EightMileLake', 'color'] = '#6aa84f'#color_mapper(63.9)
        pd_df_monthly.loc[pd_df_monthly['site'] == 'SWE-Abisko',        'color'] = '#45818e'#color_mapper(68.4)
        pd_df_monthly.loc[pd_df_monthly['site'] == 'SVA-Adventdalen',   'color'] = '#3c78d8'#color_mapper(78.2)
        pd_df_monthly.loc[pd_df_monthly['site'] == 'RUS-Seida',         'color'] = '#3d85c6'#color_mapper(67.1)
        pd_df_monthly.loc[pd_df_monthly['site'] == 'GRE-Zackenburg',    'color'] = '#674ea7'#color_mapper(74.5)
        pd_df_monthly.loc[pd_df_monthly['site'] == 'GRE-Disko',         'color'] = '#a64d79'#color_mapper(69.3)
        pd_df_monthly.loc[pd_df_monthly['site'] == 'CAN-DaringLake',    'color'] = '#85200c'#color_mapper(64.9)
    if groups == 'site':
        with open(Path(config['output_dir'] + 'combined/scatter_debug.txt'), 'w') as pf:
            with pd.option_context('display.max_columns', None):
                print('daily data:', file=pf)
                print(pd_df.dtypes, file=pf)
                print(pd_df, file=pf)
                print('scaled monthly data subset:', file=pf)
                print(pd_df_monthly.dtypes, file=pf)
                print(pd_df_monthly, file=pf)
    # remove repiration values below zero respiration and zero soil temperature
    pd_df_scaled_daily = pd_df.loc[np.logical_and(pd_df.TotalResp > 0, pd_df.SoilTemp_10cm > 0)]
    pd_df_scaled_monthly = pd_df_monthly.loc[np.logical_and(pd_df_monthly.TotalResp > 0,pd_df_monthly.SoilTemp_10cm > 0)]
    if groups == 'site':
        with open(Path(config['output_dir'] + 'combined/scatter_debug.txt'), 'a') as pf:
            with pd.option_context('display.max_columns', None):
                print('scaled daily data subset:', file=pf)
                print(pd_df_scaled_daily.dtypes, file=pf)
                print(pd_df_scaled_daily, file=pf)
                print('scaled monthly data subset:', file=pf)
                print(pd_df_scaled_monthly.dtypes, file=pf)
                print(pd_df_scaled_monthly, file=pf)
    # scale respiration for each ID group by value closest to zero
    for name, group in pd_df_scaled_daily.groupby('ID', observed=True):
        try:
            # select respiration value at lowest soil temp
            resp_at_zero_daily = group.nsmallest(5,'SoilTemp_10cm').TotalResp.iloc[0]
            # divide all groups values by resp_at_zero
            pd_df_scaled_daily.loc[pd_df_scaled_daily['ID'] == name, 'TotalResp'] /= resp_at_zero_daily
        except:
            continue
    # scale respiration for each ID group by value closest to zero
    for name, group in pd_df_scaled_monthly.groupby('ID', observed=True):
        try:
            # select respiration value at lowest soil temp
            resp_at_zero_monthly = group.nsmallest(5,'SoilTemp_10cm').TotalResp.iloc[0]
            # divide all groups values by resp_at_zero
            pd_df_scaled_monthly.loc[pd_df_scaled_monthly['ID'] == name, 'TotalResp'] /= resp_at_zero_monthly
        except:
            continue
    if groups == 'site':
        with open(Path(config['output_dir'] + 'combined/scatter_debug.txt'), 'a') as pf:
            with pd.option_context('display.max_columns', None):
                print('scaled daily data subset:', file=pf)
                print(pd_df_scaled_daily.dtypes, file=pf)
                print(pd_df_scaled_daily, file=pf)
                print('scaled monthly data subset:', file=pf)
                print(pd_df_scaled_monthly.dtypes, file=pf)
                print(pd_df_scaled_monthly, file=pf)
    # xaxis label
    if var == 'TotalResp':
        x_label = r'Soil Temperature ($^\circ$C)'
        y_label = r'Ecosystem Respiration (g C $m^{-2}$ $d^{-1}$)'
        y_label_scaled = r'Ecosystem Respiration (normalized)'
        # calculate exponential fit curves
        # define exponent function for optimization
        def exp_func(x,a,b):
            return a * np.exp(b*x)
        # define exponential regression function
        def exp_regress(x_data, y_data):
            p0 = [1, 0.1]
            popt, pcov = curve_fit(exp_func, x_data.to_numpy(), y_data.to_numpy(), p0)
            return popt
        # calculate daily data fits
        pd_df_scaled_daily['a'] = np.nan
        pd_df_scaled_daily['b'] = np.nan
        for name, group in pd_df_scaled_daily.groupby('ID', observed=True):
            try:
                if groups == 'site':
                    with open(Path(config['output_dir'] + 'combined/scatter_debug.txt'), 'a') as pf:
                        print(name, file=pf)
                        print(group, file=pf)
                coefs = exp_regress(group['SoilTemp_10cm'], group[var])
                pd_df_scaled_daily.loc[pd_df_scaled_daily['ID'] == name, 'a'] = coefs[0]
                pd_df_scaled_daily.loc[pd_df_scaled_daily['ID'] == name, 'b'] = coefs[1]
            except:
                continue
        pd_df_scaled_daily['exp_pred'] = pd_df_scaled_daily['a']*np.exp(pd_df_scaled_daily['b']*pd_df_scaled_daily['SoilTemp_10cm']) 
        pd_df_scaled_daily = pd_df_scaled_daily.dropna()
        if groups == 'site':
            with open(Path(config['output_dir'] + 'combined/scatter_debug.txt'), 'a') as pf:
                with pd.option_context('display.max_columns', None):
                    print('daily scale factor:', file=pf)
                    print(pd_df_scaled_daily, file=pf)
        # calculate monthly data fits
        pd_df_scaled_monthly['a'] = np.nan
        pd_df_scaled_monthly['b'] = np.nan
        for name, group in pd_df_scaled_monthly.groupby('ID', observed=True):
            try:
                if groups == 'site':
                    with open(Path(config['output_dir'] + 'combined/scatter_debug.txt'), 'a') as pf:
                        print(name, file=pf)
                        print(group, file=pf)
                coefs = exp_regress(group['SoilTemp_10cm'], group[var])
                pd_df_scaled_monthly.loc[pd_df_scaled_monthly['ID'] == name, 'a'] = coefs[0]
                pd_df_scaled_monthly.loc[pd_df_scaled_monthly['ID'] == name, 'b'] = coefs[1]
            except:
                continue
        pd_df_scaled_monthly['exp_pred'] = pd_df_scaled_monthly['a']*np.exp(pd_df_scaled_monthly['b']*pd_df_scaled_monthly['SoilTemp_10cm']) 
        pd_df_scaled_monthly = pd_df_scaled_monthly.dropna()
        #pd_df['exp_pred'] = np.nan
        #pd_df_monthly['exp_pred'] = np.nan
    if groups == 'site':
        with open(Path(config['output_dir'] + 'combined/scatter_debug.txt'), 'a') as pf:
            with pd.option_context('display.max_columns', None):
                print('scaled daily data subset:', file=pf)
                print(pd_df_scaled_daily.dtypes, file=pf)
                print(pd_df_scaled_daily, file=pf)
                print('scaled monthly data subset:', file=pf)
                print(pd_df_scaled_monthly.dtypes, file=pf)
                print(pd_df_scaled_monthly, file=pf)
                print('unique ID values', file=pf)
                print(pd_df_scaled_daily['ID'].unique(), file=pf)
                print(pd_df_scaled_monthly['ID'].unique(), file=pf)
                print('unique color values', file=pf)
                print(pd_df_scaled_daily['color'].unique(), file=pf)
                print(pd_df_scaled_monthly['color'].unique(), file=pf)
    # output csv to inspect
    pd_df.to_csv(config['output_dir'] + 'combined/' + out_dir + '/' + file_name + '.csv')
    pd_df_monthly.to_csv(config['output_dir'] + 'combined/' + out_dir + '/' + file_name + '_monthly.csv')
    pd_df_scaled_daily.to_csv(config['output_dir'] + 'combined/' + out_dir + '/' + file_name + '_scaled_daily.csv')
    pd_df_scaled_monthly.to_csv(config['output_dir'] + 'combined/' + out_dir + '/' + file_name + '_scaled_monthly.csv')
    # create the correct categorical labels for plots
    for factor_col in ['ID','color']:
        pd_df[factor_col] = pd_df[factor_col].astype('category')
        pd_df_monthly[factor_col] = pd_df_monthly[factor_col].astype('category')
        pd_df_scaled_monthly[factor_col] = pd_df_scaled_monthly[factor_col].astype('category')
        pd_df_scaled_daily[factor_col] = pd_df_scaled_daily[factor_col].astype('category')
   # elif var == 'q10':
   #     x_label = r'Soil Temperature ($^\circ$C)'
   #     y_label = r'q10 (unitless)'
   #     pd_df['exp_pred'] = np.nan
   #     pd_df_monthly['exp_pred'] = np.nan
    # plotnine graph daily
    p = ggplot(pd_df, aes(x='SoilTemp_10cm', y=var, group='color', color='color')) + \
        labs(x=x_label, y=y_label) + \
        geom_point(alpha=0.5, fill='None', size=1.2) + \
        scale_color_identity(name=groups, guide='legend', labels=list(pd_df['ID'].unique())) + \
        xlim(0,30) + \
        ylim(0,30) + \
        theme_bw() + \
        theme(
            axis_text_x = element_text(angle = 90),
            axis_line = element_line(colour = "black"),
            legend_text=element_text(size=8),
            legend_key=element_rect(fill = "white"),
            panel_grid_major = element_blank(),
            panel_grid_minor = element_blank(),
            panel_border = element_blank(),
            panel_background = element_blank()
        )
        #scale_color_manual(labels=pd_df['ID'].unique(), values=pd_df['color'].unique()) + \
    #if var == 'TotalResp':
    #    p = p + geom_line(aes(y='exp_pred'))
    # plotnine graph monthly
    p2 = ggplot(pd_df_monthly, aes(x='SoilTemp_10cm', y=var, group='color', color='color')) + \
        labs(x=x_label, y=y_label) + \
        geom_point(alpha=0.5, fill='None', size=1.2) + \
        scale_color_identity(name=groups, guide='legend', labels=list(pd_df_monthly['ID'].unique())) + \
        xlim(0,30) + \
        ylim(0,30) + \
        theme_bw() + \
        theme(
            axis_text_x = element_text(angle = 90),
            axis_line = element_line(colour = "black"),
            legend_text=element_text(size=8),
            legend_key=element_rect(fill = "white"),
            panel_grid_major = element_blank(),
            panel_grid_minor = element_blank(),
            panel_border = element_blank(),
            panel_background = element_blank()
        )
    #    # example of axis controls for plots 
    #    #scale_x_continuous(limits=(0, 30)) + \
    #    #scale_y_continuous(limits=(0, 8e-5)) + \
    # scaled plots with jeralyn's functional benchmrk curves
    # plotnine graph daily
    p3 = ggplot(pd_df_scaled_daily, aes(x='SoilTemp_10cm', y=var, group='color',  color='color')) + \
        labs(x=x_label, y=y_label_scaled) + \
        geom_point(alpha=0.5, fill='None', size=1.2) + \
        geom_line(aes(y='exp_pred'), size=1) + \
        scale_color_identity(name=groups, guide='legend', labels=list(pd_df_scaled_daily['ID'].unique())) + \
        geom_ribbon(func_curves, aes(x=func_curves['soil_temp'], \
            ymin=func_curves['allsites_ctl_lower_bound'], \
            ymax=func_curves['allsites_ctl_upper_bound']), \
            fill='grey', alpha=0.5, inherit_aes=False) + \
        geom_line(func_curves, aes(x=func_curves['soil_temp'], y=func_curves['allsites_ctl_reco']), \
            color='black', linetype='dashed', size=1, inherit_aes=False) + \
        xlim(0,30) + \
        ylim(0,30) + \
        theme_bw() + \
        theme(
            axis_text_x = element_text(angle = 90),
            axis_line = element_line(colour = "black"),
            legend_text=element_text(size=8),
            legend_key=element_rect(fill = "white"),
            panel_grid_major = element_blank(),
            panel_grid_minor = element_blank(),
            panel_border = element_blank(),
            panel_background = element_blank()
        )
        #scale_color_manual(labels=pd_df_scaled_daily['ID'].unique(), values=pd_df_scaled_daily['color'].unique()) + \
    #if var == 'TotalResp':
    #    p = p + geom_line(aes(y='exp_pred'))
    # plotnine graph monthly
    p4 = ggplot(pd_df_scaled_monthly, aes(x='SoilTemp_10cm', y=var, group='color', color='color')) + \
        labs(x=x_label, y=y_label_scaled) + \
        geom_point(alpha=0.5, fill='None', size=1.2) + \
        geom_line(aes(y='exp_pred'), size=1)+ \
        scale_color_identity(name=groups, guide='legend', labels=list(pd_df_scaled_monthly['ID'].unique())) + \
        geom_ribbon(func_curves, aes(x=func_curves['soil_temp'], \
            ymin=func_curves['allsites_ctl_lower_bound'], \
             ymax=func_curves['allsites_ctl_upper_bound']), \
            fill='grey', alpha=0.5, inherit_aes=False) + \
        geom_line(func_curves, aes(x=func_curves['soil_temp'], y=func_curves['allsites_ctl_reco']), \
            color='black', linetype='dashed', size=1, inherit_aes=False) + \
        xlim(0,30) + \
        ylim(0,30) + \
        theme_bw() + \
        theme(
            axis_text_x = element_text(angle = 90),
            axis_line = element_line(colour = "black"),
            legend_text=element_text(size=8),
            legend_key=element_rect(fill = "white"),
            panel_grid_major = element_blank(),
            panel_grid_minor = element_blank(),
            panel_border = element_blank(),
            panel_background = element_blank()
        )
        #scale_color_manual(labels=pd_df_scaled_monthly['ID'].unique(), values=pd_df_scaled_monthly['color'].unique()) + \
    # output graph
    p.save(filename=file_name+'.png', path=config['output_dir']+'combined/'+out_dir, \
        height=5, width=8, units='in', dpi=300)
    p2.save(filename=file_name+'_monthly.png', path=config['output_dir']+'combined/'+out_dir, \
        height=5, width=8, units='in', dpi=300)
    p3.save(filename=file_name+'_scaled.png', path=config['output_dir']+'combined/'+out_dir, \
        height=5, width=8, units='in', dpi=300)
    p4.save(filename=file_name+'_scaled_monthly.png', path=config['output_dir']+'combined/'+out_dir, \
        height=5, width=8, units='in', dpi=300)

def plotnine_scatter_delta(f, config, out_dir): 
    # bring in and select data similarly as graph_lines
    sites = f[0]
    var = f[1]
    models = f[2]
    sims = f[3]
    plot_num = f[4]
    # set file read location from merged daily data
    combined_file = config['output_dir'] + 'combined/WrPMIP_all_models_sites_2000-2021.zarr'
    ds = xr.open_zarr(combined_file, use_cftime=True, mask_and_scale=True)
    # function to subset only summer months
    def is_summer(month):
        return (month >= 5) & (month <= 9)
    # check maximum Total Respiration value for plots
    #ds = ds.where(ds['SoilTemp'] < 150)
    #ds_mean = ds[['TotalResp','SoilTemp']].sel(time=is_summer(ds[['TotalResp','SoilTemp']].time.dt.month)).resample(time='M').mean('time')
    #monthly_soilT_max = np.unique(ds_mean['SoilTemp'].max().values).max()
    #monthly_soilT_min = np.unique(ds_mean['SoilTemp'].min().values).min()
    #monthly_er_max = np.unique(ds_mean['TotalResp'].max().values).max()
    #monthly_er_min = np.unique(ds_mean['TotalResp'].min().values).min()
    #daily_soilT_max = np.unique(ds['SoilTemp'].max().values).max()
    #daily_soilT_min = np.unique(ds['SoilTemp'].min().values).min()
    #daily_er_max = np.unique(ds['TotalResp'].max().values).max()
    #daily_er_min = np.unique(ds['TotalResp'].min().values).min()
    # subsample data
    var_list = [var]
    var_list.extend(['deltaSoilTemp'])
    ds = ds[var_list].sel(site=sites, model=models, sim=sims, time=slice('2000-01-01','2020-12-31'))
    ds = ds.sel(time=is_summer(ds['time.month']))
    # deal with variable site/model/sims and variable depth increments
    listed = False
    groups = 'site'
    other_vars = ['sim', 'model']
    if isinstance(var, str) & isinstance(models, str) & isinstance(sites, str) & isinstance(sims, str):
        file_name = var + '_by_time_' + models + '_' + sites[:7] + '_' + sims
    if isinstance(sites, list):
        sites_chopped = []
        for i in sites:
            sites_chopped.append(i[:7]) 
        file_part = models + '_' + '_'.join(sites_chopped) + '_' + sims 
        listed = True
        groups = 'site'
        other_vars = ['model', 'sim']
    if isinstance(models, list):
        file_part = '_'.join(models) + '_' + sites[:7] + '_' + sims
        listed = True
        groups = 'model'
        other_vars = ['sim', 'site']
    if isinstance(sims, list):
        file_part = models + '_' + sites[:7] + '_' + '_'.join(sims)
        listed = True
        groups = 'sim'
        other_vars = ['model', 'site']
    # create file name if any lists present
    if listed == True:
        file_name = var + '_by_SoilTemp_' + file_part
    # manipulate data for ggplot input format
    # daily data
    pd_df = ds.to_dataframe()
    pd_df = pd_df.reset_index()
    pd_df['ID'] =  pd_df[groups].astype(str).str.cat(pd_df[other_vars].astype(str), sep='_')
    pd_df = pd_df.set_index('ID', drop=False)
    pd_df[var].replace([np.inf, -np.inf], np.nan, inplace=True)
    pd_df = pd_df.dropna()
    # aggregate to monthly means
    ds_monthly = ds.resample(time='M').mean('time')
    # monthly data
    pd_df_monthly = ds_monthly.to_dataframe()
    pd_df_monthly = pd_df_monthly.reset_index()
    pd_df_monthly['ID'] =  pd_df_monthly[groups].astype(str).str.cat(pd_df_monthly[other_vars].astype(str), sep='_')
    pd_df_monthly = pd_df_monthly.set_index('ID', drop=False)
    pd_df_monthly[var].replace([np.inf, -np.inf], np.nan, inplace=True)
    pd_df_monthly = pd_df_monthly.dropna()
    # custome color map
    plot_colors = ['blue','gold','red','olive','purple','orange','green','cyan','magenta','brown','gray','black']
    fill_blanks = ['#ffffff00'] * len(plot_colors)
    # create color column for consistent plotting
    
    # x and y labels
    x_label = r'Delta Soil Temperature ($^\circ$C)'
    y_label = r'Delta Ecosystem Respiration (g C $m^{-2}$ $d^{-1}$)'
    # plotnine graph daily
    p = ggplot(pd_df, aes(x='deltaSoilTemp', y=var, group='ID', color='ID')) + \
        labs(x=x_label, y=y_label) + \
        geom_point() + \
        geom_smooth(method = 'lm', se=False) + \
        scale_fill_manual(values = fill_blanks) + \
        scale_color_manual(plot_colors) + \
        guides(color = guide_legend(reverse=True)) + \
        theme_bw() + \
        theme(
            axis_text_x = element_text(angle = 90),
            axis_line = element_line(colour = "black"),
            legend_text=element_text(size=8),
            panel_grid_major = element_blank(),
            panel_grid_minor = element_blank(),
            panel_border = element_blank(),
            panel_background = element_blank()
        )
    # plotnine graph annual
    # add back if x/y limits needed: scale_x_continuous(limits=(0, 30)) + \
    # add back if x/y limits needed:scale_y_continuous(limits=(0, 8e-5)) + \
    p2 = ggplot(pd_df_monthly, aes(x='deltaSoilTemp', y=var, group='ID', color='ID')) + \
        labs(x=x_label, y=y_label) + \
        geom_point() + \
        geom_smooth(method = 'lm', se=False) + \
        scale_fill_manual(values = fill_blanks) + \
        scale_color_manual(plot_colors) + \
        guides(color = guide_legend(reverse=True)) + \
        theme_bw() + \
        theme(
            axis_text_x = element_text(angle = 90),
            axis_line = element_line(colour = "black"),
            legend_text=element_text(size=8),
            panel_grid_major = element_blank(),
            panel_grid_minor = element_blank(),
            panel_border = element_blank(),
            panel_background = element_blank()
        )
    # output graph
    p.save(filename=file_name+'.png', path=config['output_dir']+'combined/'+out_dir, \
        height=5, width=8, units='in', dpi=300)
    p2.save(filename=file_name+'_monthly.png', path=config['output_dir']+'combined/'+out_dir, \
        height=5, width=8, units='in', dpi=300)

def process_ted_data(config):
    # define ted data location 
    obs_raw_file = '/projects/warpmip/shared/ted_data/flux_daily.csv'
    obs_processed_file = '/projects/warpmip/shared/ted_data/USA-EML_processed_data.csv'
    obs_processed_file_monthly = '/projects/warpmip/shared/ted_data/USA-EML_processed_data_monthly.csv'
    obs_processed_file_annual = '/projects/warpmip/shared/ted_data/USA-EML_processed_data_annual.csv'
    # read in csv file, parse datstime from date column
    pd_obs = pd.read_csv(obs_raw_file)
    # Subset, Harmonize needed columns and factors 
    pd_obs = pd_obs[['date','plot.id','treatment','reco.sum','t10.filled.mean','wtd','td']]
    pd_obs = pd_obs.rename(columns={
            'date': 'time',
            'plot.id': 'plot',
            'treatment': 'sim',
            'reco.sum': 'obs_TotalResp',
            't10.filled.mean': 'obs_SoilTemp',
            'wtd': 'obs_WTD',
            'td': 'obs_ALT'})
    pd_obs.loc[pd_obs.sim == 'Control','sim'] = 'b2'
    pd_obs.loc[pd_obs.sim == 'Air Warming','sim'] = 'otc'
    pd_obs.loc[pd_obs.sim == 'Soil Warming','sim'] = 'sf'
    pd_obs = pd_obs[pd_obs['sim'] != 'Air + Soil Warming']
    # scale and change sign
    pd_obs['obs_ALT'] = (pd_obs['obs_ALT']/100)*(-1)
    pd_obs['obs_WTD'] = (pd_obs['obs_WTD']/100)*(-1)
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'w') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 20):
            print('Healy observations after subset, hamonization:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    # set dimensions to multiindex
    pd_obs = pd_obs.set_index(['time','plot','sim'])
    # remove duplicated values (all from 2018) some error in heidis code 
    pd_obs = pd_obs[~pd_obs.index.duplicated()]
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 20):
            print('duplicated removal:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    # create datetime index
    pd_obs = pd_obs.reset_index()
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('reset index:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    pd_obs['time'] = pd.to_datetime(pd_obs['time'])
    pd_obs['time'] = pd_obs['time'].dt.strftime('%Y-%m-%d')
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('datetime and string edit:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    datetime_index = pd.DatetimeIndex(pd_obs['time'])
    pd_obs['month'] = datetime_index.month
    pd_obs['year'] = datetime_index.year 
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('month and year added:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    pd_obs = pd_obs.set_index(datetime_index)
    pd_obs = pd_obs.drop(columns=['time'])
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('set index with time:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    pd_obs = pd_obs.loc[(pd_obs.index > datetime(year=2009,month=1,day=1)) & (pd_obs.index < datetime(year=2021,month=12,day=31))]
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('time subset to 2009-2021:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    pd_obs = pd_obs.reset_index()
    pd_obs = pd_obs.set_index(['time','plot','sim'])
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('index reset:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    # aggregate daily data to year/month/plot/simi
    pd_obs_monthly = pd_obs.groupby([
        pd.Grouper(level='time', freq='M'),
        pd.Grouper(level='plot'),
        pd.Grouper(level='sim')
    ]).agg({
        'obs_TotalResp': 'sum',
        'obs_SoilTemp': 'mean',
        'obs_WTD': 'mean',
        'obs_ALT': 'max'})
    pd_obs_monthly = pd_obs_monthly.reset_index()
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('Healy observations month/plot/treatment aggregation:', file=pf)
            print(pd_obs_monthly.dtypes, file=pf)
            print(pd_obs_monthly, file=pf)
    # aggregate across years to create the correct n for mean/std calculation (n= 48/4 = 12)
    pd_obs_monthly = pd_obs.groupby(['month','plot','sim']).agg({
        'obs_TotalResp': 'mean',
        'obs_SoilTemp': 'mean',
        'obs_WTD': 'mean',
        'obs_ALT': 'mean'})
    pd_obs_monthly = pd_obs_monthly.reset_index()
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('Healy observations month/plot/treatment aggregation:', file=pf)
            print(pd_obs_monthly.dtypes, file=pf)
            print(pd_obs_monthly, file=pf)
    # aggregate across plots for final mean/std
    pd_obs_monthly_mean = pd_obs_monthly.groupby(['month','sim']).agg({
        'obs_TotalResp':'mean',
        'obs_SoilTemp':'mean',
        'obs_WTD':'mean',
        'obs_ALT':'mean'})
    pd_obs_monthly_mean = pd_obs_monthly_mean.reset_index()
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None):
            print('Healy observations monthly mean:', file=pf)
            print(pd_obs_monthly_mean.dtypes, file=pf)
            print(pd_obs_monthly_mean, file=pf)
    def std(x):
        return np.std(x, ddof=1)
    pd_obs_monthly_std = pd_obs_monthly.groupby(['month','sim']).agg({
        'obs_TotalResp': std,
        'obs_SoilTemp': std,
        'obs_WTD': std,
        'obs_ALT': std})
    pd_obs_monthly_std = pd_obs_monthly_std.reset_index()
    pd_obs_monthly_std = pd_obs_monthly_std.rename(columns={
        'obs_TotalResp':'obs_TotalResp_std',
        'obs_SoilTemp':'obs_SoilTemp_std',
        'obs_WTD':'obs_WTD_std',
        'obs_ALT':'obs_ALT_std'})
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None):
            print('Healy observations monthly std:', file=pf)
            print(pd_obs_monthly_std.dtypes, file=pf)
            print(pd_obs_monthly_std, file=pf)
    pd_obs_monthly_mean = pd.merge(pd_obs_monthly_mean, pd_obs_monthly_std, on=['month','sim'])
    #pd_obs_monthly_mean['month'] = pd.DatetimeIndex(pd_obs_monthly_mean['time']).month
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None):
            print('Healy observations monthly mean and std merged:', file=pf)
            print(pd_obs_monthly_mean.dtypes, file=pf)
            print(pd_obs_monthly_mean, file=pf)
    # annual
    pd_obs_annual = pd_obs.groupby(['year','plot','sim']).agg({
        'obs_TotalResp':'sum',
        'obs_SoilTemp':'mean',
        'obs_WTD':'mean',
        'obs_ALT':'max'})
    pd_obs_annual = pd_obs_annual.reset_index()
    #pd_obs_annual = pd_obs_annual.set_index(pd.DatetimeIndex(pd_obs_annual['time']))
    pd_obs_annual_mean = pd_obs_annual.groupby(['year','sim']).agg({
        'obs_TotalResp':'mean',
        'obs_SoilTemp':'mean',
        'obs_WTD':'mean',
        'obs_ALT':'max'})
    pd_obs_annual_mean = pd_obs_annual_mean.reset_index()
    pd_obs_annual_std = pd_obs_annual.groupby(['year','sim']).agg({
        'obs_TotalResp': std,
        'obs_SoilTemp': std,
        'obs_WTD': std,
        'obs_ALT': std}) 
    pd_obs_annual_std = pd_obs_annual_std.reset_index()
    pd_obs_annual_std = pd_obs_annual_std.rename(columns={
        'obs_TotalResp':'obs_TotalResp_std',
        'obs_SoilTemp':'obs_SoilTemp_std',
        'obs_WTD':'obs_WTD_std',
        'obs_ALT':'obs_ALT_std'})
    pd_obs_annual_mean = pd.merge(pd_obs_annual_mean, pd_obs_annual_std, on=['year','sim'])
    pd_obs_annual_mean['time'] = pd_obs_annual_mean['year'].astype(str) + '-01-01'
    # add site column and EML name for merging during plotting
    pd_obs_monthly_mean['site'] = 'USA-EightMileLake'
    pd_obs_annual_mean['site'] = 'USA-EightMileLake'
    # set index as time
    # print out data
    with open(Path('/projects/warpmip/shared/ted_data/ted_debug.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None):
            print('Healy observations monthly mean:', file=pf)
            print(pd_obs_monthly_mean, file=pf)
            print('Healy observations monthly std:', file=pf)
            print(pd_obs_monthly_std, file=pf)
            print('Healy observations annual mean:', file=pf)
            print(pd_obs_annual_mean, file=pf)
            print('Healy observations annual std:', file=pf)
            print(pd_obs_annual_std, file=pf)
    # output files to csv for import by graphing functions
   # pd_obs = pd_obs.drop(columns=['time'])
    pd_obs = pd_obs.reset_index() 
    pd_obs.to_csv(obs_processed_file, index=False)
    pd_obs_monthly_mean.to_csv(obs_processed_file_monthly, index=False)
    pd_obs_annual_mean.to_csv(obs_processed_file_annual, index=False)

def schadel_plots_env(var, config, out_dir):
    # create folder for output
    Path(config['output_dir'] + '/combined/' + out_dir).mkdir(parents=True, exist_ok=True)
    Path(config['output_dir'] + '/combined/' + out_dir).chmod(0o762)
    # set file read location from merged daily data
    combined_file = config['output_dir'] + 'combined/WrPMIP_all_models_sites_2000-2021.zarr'
    ds = xr.open_zarr(combined_file, use_cftime=True, mask_and_scale=True)
    # read in Healy obs data harmonized to WrPMIP simulation labels/factors
    obs_processed_file_monthly = '/projects/warpmip/shared/ted_data/USA-EML_processed_data_monthly.csv'
    obs_processed_file_annual = '/projects/warpmip/shared/ted_data/USA-EML_processed_data_annual.csv'
    pd_obs_monthly = pd.read_csv(obs_processed_file_monthly)
    pd_obs_annual = pd.read_csv(obs_processed_file_annual, parse_dates=['time'])
    with open(Path(config['output_dir'] + '/combined/schadel_debug'+var+'.txt'), 'w') as pf:
        with pd.option_context('display.max_columns', 10):
            print('monthly obs subset:', file=pf)
            print(pd_obs_monthly, file=pf)
            print('annual obs subset:', file=pf)
            print(pd_obs_annual, file=pf)
    # function to subset only summer months
    def is_summer(month):
        return (month >= 5) & (month <= 9)
    # subsample xarray dataset
    da = ds[var].sel(time=slice('2000-01-01','2020-12-31'))
    #da = da.sel(time=is_summer(da['time.month']))
    with open(Path(config['output_dir'] + '/combined/schadel_debug'+var+'.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', 10):
            print('summer data subset:', file=pf)
            print(da, file=pf)
    # aggregate sims (collapse models and sites into one value per model simulation)
    da_monthly = da.resample(time='M').mean('time')
    da_monthly = da_monthly.groupby('time.month').mean('time')
    if var == 'ALT':
        da_monthly = da.resample(time='M').max('time')
        da_monthly = da_monthly.groupby('time.month').mean('time')
    if var == 'TotalResp':
        da_monthly = ds.resample(time='M').sum('time')
        da_monthly = da_monthly.groupby('time.month').mean('time')
    with open(Path(config['output_dir'] + '/combined/schadel_debug'+var+'.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', 10):
            print('variable resampled to monthly timestep:', file=pf)
            print(da_monthly, file=pf)
    # monthly data
    pd_df_monthly = da_monthly.to_dataframe()
    with open(Path(config['output_dir'] + '/combined/schadel_debug'+var+'.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', 10):
            print('pandas dataframe conversion:', file=pf)
            print(pd_df_monthly, file=pf)
    pd_df_monthly = pd_df_monthly.reset_index()
    pd_df_monthly['ID'] =  pd_df_monthly['model'].astype(str).str.cat(pd_df_monthly[['site','sim']].astype(str), sep='_')
    pd_df_monthly = pd_df_monthly.set_index('ID', drop=False)    
    pd_df_monthly[var].replace([np.inf, -np.inf], np.nan, inplace=True)
    pd_df_monthly = pd_df_monthly.dropna()
    # aggregate to monthly means
    da_annual = da.groupby('time.year').mean('time')
    if var == 'ALT':
        da_annual = da.groupby('time.year').max('time')
    if var == 'TotalResp':
        da_annual = da.groupby('time.year').sum('time')
    # monthly data
    pd_df_annual = da_annual.to_dataframe()
    pd_df_annual = pd_df_annual.reset_index()
    pd_df_annual['ID'] =  pd_df_annual['model'].astype(str).str.cat(pd_df_annual[['site','sim']].astype(str), sep='_')
    pd_df_annual = pd_df_annual.set_index('ID', drop=False)
    pd_df_annual[var].replace([np.inf, -np.inf], np.nan, inplace=True)
    pd_df_annual = pd_df_annual.dropna()
    with open(Path(config['output_dir'] + '/combined/schadel_debug'+var+'.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', 10):
            print('pandas dataframe conversion:', file=pf)
            print(pd_df_annual.dtypes, file=pf)
    # deal with differences in positive vs negative numerical depths
    ## monthly
    if var == 'ALT':
        pd_df_annual.loc[pd_df_annual['model'] != 'ecosys', var] *= -1
        pd_df_monthly.loc[pd_df_monthly['model'] != 'ecosys', var] *= -1
    if var == 'WTD':    
        pd_df_annual.loc[pd_df_annual['model'] != 'JSBACH', var] *= -1
        pd_df_monthly.loc[pd_df_monthly['model'] != 'JSBACH', var] *= -1
    # subset to only healy
    pd_df_monthly = pd_df_monthly.loc[pd_df_monthly['site'] == 'USA-EightMileLake']
    pd_df_annual = pd_df_annual.loc[pd_df_annual['site'] == 'USA-EightMileLake']
    #pd_obs_annual.to_csv(config['output_dir'] + 'combined/' + out_dir + '/' + var + '_obs_annual.csv')
    pd_df_annual['year'] = pd.to_datetime(pd_df_annual['year'], format='%Y')
    pd_obs_annual['year'] = pd.to_datetime(pd_obs_annual['year'], format='%Y')
    # output csv files
    pd_df_monthly.to_csv(config['output_dir'] + 'combined/' + out_dir + '/' + var + '_monthly.csv')
    pd_obs_monthly.to_csv(config['output_dir'] + 'combined/' + out_dir + '/' + var + '_obs_monthly.csv')
    pd_df_annual.to_csv(config['output_dir'] + 'combined/' + out_dir + '/' + var + '_annual.csv')
    pd_obs_annual.to_csv(config['output_dir'] + 'combined/' + out_dir + '/' + var + '_obs_annual.csv')
    pd_df_annual['year'] = pd.to_datetime(pd_df_annual['year'], format='%Y')
    pd_obs_annual['year'] = pd.to_datetime(pd_obs_annual['year'], format='%Y')
    # plot labels
    if var == 'TotalResp':
        y_label = r'Summer Ecosystem Respiration (g C $m^{-2}$)'
        obs_str = 'obs_TotalResp'
        ymax_str = 'obs_TotalResp + obs_TotalResp_std'
        ymin_str = 'obs_TotalResp - obs_TotalResp_std'
    elif var == 'ALT':
        y_label = r'Max Active Layer Thickness (m)'
        obs_str = 'obs_ALT'
        ymax_str = 'obs_ALT + obs_ALT_std'
        ymin_str = 'obs_ALT - obs_ALT_std'
        # remove ALT below 10m
        pd_df_annual.loc[pd_df_annual['ALT'] < -10, 'ALT'] = np.nan
        pd_df_annual = pd_df_annual.dropna()
        pd_df_monthly.loc[pd_df_monthly['ALT'] < -10, 'ALT'] = np.nan
        pd_df_monthly = pd_df_monthly.dropna()
    elif var == 'WTD':
        y_label = r'Water Table Depth (m)'
        obs_str = 'obs_WTD'
        ymax_str = 'obs_WTD + obs_WTD_std'
        ymin_str = 'obs_WTD - obs_WTD_std'
    elif var == 'SoilTemp_10cm':
        y_label = r'10cm Soil Temperature ($^\circ$C)'
        obs_str = 'obs_SoilTemp'
        ymax_str = 'obs_SoilTemp + obs_SoilTemp_std'
        ymin_str = 'obs_SoilTemp - obs_SoilTemp_std'
    # subset control and warming subsets for plotting points by black/red
    pd_obs_annual_control = pd_obs_annual[pd_obs_annual['sim'] == 'b2']
    pd_obs_annual_otc = pd_obs_annual[pd_obs_annual['sim'] == 'otc']
    pd_obs_annual_sf = pd_obs_annual[pd_obs_annual['sim'] == 'sf']
    pd_obs_monthly_control = pd_obs_monthly[pd_obs_monthly['sim'] == 'b2']
    pd_obs_monthly_otc = pd_obs_monthly[pd_obs_monthly['sim'] == 'otc']
    pd_obs_monthly_sf = pd_obs_monthly[pd_obs_monthly['sim'] == 'sf']
    # plot annual change
    p1 = ggplot(pd_df_annual, aes(x='year', y=var, color='model', linetype='sim')) + \
        geom_line() + \
        geom_point(mapping=aes(x='year', y=obs_str), color='red', data=pd_obs_annual_sf, inherit_aes=False) + \
        geom_point(mapping=aes(x='year', y=obs_str), color='black', data=pd_obs_annual_control, inherit_aes=False) + \
        geom_errorbar(mapping=aes(x='year', ymax=ymax_str, ymin=ymin_str), color='red', data=pd_obs_annual_sf, inherit_aes=False) + \
        geom_errorbar(mapping=aes(x='year', ymax=ymax_str, ymin=ymin_str), color='black', data=pd_obs_annual_control, inherit_aes=False) + \
        scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y')) + \
        labs(x=r'time (year)', y=y_label) + \
        theme_bw() + \
        theme(
            axis_text_x = element_text(angle = 90),
            axis_line = element_line(colour = "black"),
            panel_grid_major = element_blank(),
            panel_grid_minor = element_blank(),
            panel_border = element_blank(),
            panel_background = element_blank()
        )
        #scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y')) + \
    p2 = ggplot(pd_df_monthly, aes(x='month', y=var, color='model', linetype='sim')) + \
        geom_line() + \
        scale_x_continuous(breaks = np.arange(np.round(min(pd_df_monthly.month)), np.round(max(pd_df_monthly.month)), step = 2)) + \
        geom_point(data=pd_obs_monthly_sf, mapping=aes(x='month', y=obs_str), color='red', inherit_aes=False) + \
        geom_errorbar(data=pd_obs_monthly_sf, mapping=aes(x='month', ymax=ymax_str, ymin=ymin_str), color='red', inherit_aes=False) + \
        geom_point(data=pd_obs_monthly_control, mapping=aes(x='month', y=obs_str), color='black', inherit_aes=False) + \
        geom_errorbar(data=pd_obs_monthly_control, mapping=aes(x='month', ymax=ymax_str, ymin=ymin_str), color='black', inherit_aes=False) + \
        labs(x=r'time (month)', y=y_label) + \
        theme_bw() + \
        theme(
            axis_text_x = element_text(angle = 90),
            axis_line = element_line(colour = "black"),
            panel_grid_major = element_blank(),
            panel_grid_minor = element_blank(),
            panel_border = element_blank(),
            panel_background = element_blank()
        )
        #scale_x_datetime(breaks=date_breaks('month'), labels=date_format('%M')) + \
        #guides(color = guide_legend(reverse=True)) + \
        #scale_color_manual(plot_colors) + \
    # output graph
    p1.save(filename=var+'_by_time_annual.png', path=config['output_dir']+'combined/'+out_dir, \
        height=5, width=8, units='in', dpi=300)
    p2.save(filename=var+'_by_time_monthly.png', path=config['output_dir']+'combined/'+out_dir, \
        height=5, width=8, units='in', dpi=300)

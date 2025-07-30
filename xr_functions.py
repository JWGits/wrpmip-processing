# import packages
import os
import shutil
import glob
import json
import traceback
import gzip
import itertools
import math
import random
import string
from datetime import datetime, date
from pathlib import Path
import zarr
from multiprocessing import Pool
import netCDF4 as nc
import xarray as xr
import xesmf as xe
import rioxarray as rxr
import cftime as cft
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from pandas import option_context
from reportlab.lib import utils
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib as matplotlib
import matplotlib.path as mpath
from cycler import cycler
import cartopy.crs as ccrs
import cartopy
import seaborn as sns
import nc_time_axis
import docx
import textwrap
import dask.config
from numcodecs import Blosc, Zlib, Zstd
from functools import partial
from cycler import cycler
import zipfile
import scipy
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
                shutil.rmtree(string)
    except Exception as error:
        print(error)
        pass

# create rand string generator
def rand_str_gen(count=1000):
    # create empty list for generated strings
    generated_strs = set()
    while True:
        if len(generated_strs) == count:
            return
        # create random 16 digit string
        candidate_str = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        # check if in generated
        if candidate_str not in generated_strs:
            generated_strs.add(candidate_str)
            yield candidate_str

# mv old files within /projects folder
def subdir_list(cur_dir):
    try:
        dir_list = [x[0] for x in os.walk(cur_dir, topdown=False)]
    except Exception as error:
        print(error)
    return dir_list

# mv old files within /projects folder
def mv_subdir_list(cur_dir, new_dir):
    try:
        # call string generator
        dir_list = subdir_list(cur_dir)
        str_gen = rand_str_gen(len(dir_list)*2)
        # loop through new project directory for list of sub dir
        move_list = []
        for i in dir_list:
            # generate unique string and add to scratch dir
            new_out = new_dir + next(str_gen)
            # add to move list
            move_list.append([i, new_out])
    except Exception as error:
        print(error)
    return move_list

# mv old files within /projects folder
def mv_dir(cur_dir, new_dir):
    try:
        # make project directory folder
        Path(new_dir).mkdir(parents=True, exist_ok=True)
        # move regional folder to new project location
        shutil.move(cur_dir, new_dir)
    except Exception as error:
        print(error)
        pass

# read config file from sys.arg
def read_config(sys_argv_file):
    with open(sys_argv_file) as f:
        config = json.load(f)
    return config

# function to extract position of all sublists within a list
def extract_sublist(lst, position):
    return [item[position] for item in lst]

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

# make graph circular
def add_circle_boundary(ax):
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

# functions to select seasons
def is_summer(month):
    return (month >= 5) & (month <= 9)
def is_maes_summer(month):
    return (month >= 6) & (month <= 8)
def is_winter(month):
    return (month > 10) | (month < 4 )

# functions to select time since warming onset
def is_initial(year):
    return (year <= 5)
def is_middle(year):
    return (year > 5) & (year <= 10)
def is_longterm(year):
    return (year > 10) & (year <= 15)

# calculate RSME across models for all grids
def root_mean_squared_error(dataset, var, agg_over):
    ds_means = dataset[var].mean(dim=agg_over)
    rsme = np.sqrt(((dataset[var] - ds_means)**2).mean(dim=agg_over))
    return rsme

# shift array
def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

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

# create list of files to unzip that is passed to dask cluster
def gunzip_dir_list(config):
    # define empty list to hold file paths
    file_info = []
    # loop through all directories listed in config_unzip
    for f_dir in config['dir']['in']:
        # create list of netcdfs in each directory with .gz ending
        files = sorted(glob.glob("{}*.gz".format(f_dir)))
        #loop through individual files
        for f in files: 
            # check pull filename from globbed path to use as directory for unzipped files
            if ('nhlat_monthly_1901_2000' in f) or ('1a' in f):
                dir_name = 'Baseline_1901-2000'
            elif ('standard_historical' in f) or ('1a' in f):
                dir_name = 'Baseline_2000-2021'
            elif ('temp1_historical' in f) or ('1a' in f):
                dir_name = 'OTC'
            elif ('ksnow0p5_historical' in f) or ('1a' in f):
                dir_name = 'Snow_fence'
            # create unziped file location
            unzipped_dir = Path(config['dir']['out']+dir_name) 
            # append to list of zip files to unzip
            file_info.append([f, unzipped_dir])
    # return file info list
    return file_info  

# copy to tmp from project folder and unzip to scratch
def ungunzip_file(f, config):
    # pull file, tmp file, and unzipped file paths from input
    src_file = f[0]
    unzip_dir = f[1]
    unzip_file = Path(str(unzip_dir)+'/'+str(Path(src_file).stem))
    # make directory for output files
    Path(unzip_dir).mkdir(parents=True, exist_ok=True)
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
    with open(Path(config['site_dir'] + 'debug_gzip.txt'), 'a') as f:
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
    with open(Path(config['site_dir'] + 'debug_filelist.txt'), 'a') as f:
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
    ds_mask = ds_mask.assign_coords({'lat': extract_sublist(ds_mask.yc.values, 0)})
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
    ds = ds[keep].isel(lndgrid=grid)
    # add snow and rain to make total precip
    ds['PRECIP'] = ds['RAIN'] + ds['SNOW']
    # shift the index by offset from GMT time at location 
    ds.coords['time'] = ds.indexes['time'].round('h') #.shift(config['cru_GMT_adj'], 'H')
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
    # Final QQ for clm created hourly cru data
    ds['FSDS'] = ds['FSDS'].where(ds['FSDS'] > 0, 0.0)
    ds['FLDS'] = ds['FLDS'].where(ds['FLDS'] > 0, 0.0)
    ds['RAIN'] = ds['RAIN'].where(ds['RAIN'] > 0, 0.0)
    ds['SNOW'] = ds['SNOW'].where(ds['SNOW'] > 0, 0.0)
    ds['QBOT'] = ds['QBOT'].where(ds['QBOT'] > 0, 0.0)
    ds['WIND'] = ds['WIND'].where(ds['WIND'] > 0, 0.0)
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
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
            print(obs_data.head(), file=f)
            print(obs_data.dtypes, file=f)
    # handle site specific idiosyncrasies
    site = config['site_name']
    match site:
        case 'USA-EightMileLake':
            # convert numerical timestamp to string for datetime.strptime
            obs_data['time'] = obs_data['TIMESTAMP_START'].astype(str)
            # remove -9999 values, as numbers are float values have to use np.isclose
            #num_cols = obs_data.select_dtypes(np.number).columns
            #obs_data[num_cols] = obs_data[num_cols].mask(np.isclose(obs_data[num_cols].values, -9999))
            # change -9999 fill values to NAs
            obs_data = obs_data.replace(-9999, np.NaN)
            # convert TBOT from celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # Convert from kPa to Pa
            obs_data.loc[:,'PBOT'] = obs_data['PBOT'] * 1000
            # convert RH to SH
            obs_data.loc[obs_data['RH'] < 0, 'RH'] = 0
            obs_data.loc[obs_data['RH'] > 100, 'RH'] = 100
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            obs_data = obs_data.drop(columns=['RH'])
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
                print(obs_data, file=f)
        case 'USA-Toolik':
            # fix hourly timestep - cannot have 24 as hour value, only 0:23 for datetime.strptime
            obs_data.loc[:,'hour'] = obs_data['hour'] - 100
            obs_data['hour'] = obs_data['hour'].astype(int)
            # combine date and hour columns for timestamp -  need to pad hours with preceeding zeros
            obs_data['time'] = obs_data['date'].astype(str) + " " + obs_data['hour'].astype(str).str.zfill(4)
            with option_context('display.max_rows', 10, 'display.max_columns', 10):
                with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
                    print(obs_data.head(), file=f)
                    print(obs_data.dtypes, file=f)
            # convert TBOT from celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # change mbar -> Pa
            obs_data.loc[:,'PBOT'] = obs_data['PBOT'] * 100
            obs_data.loc[obs_data['PBOT'] < 90000, 'PBOT'] = np.NaN 
            # convert RH to SH
            obs_data.loc[obs_data['RH'] < 0, 'RH'] = 0
            obs_data.loc[obs_data['RH'] > 100, 'RH'] = 100
            obs_data['QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT']) 
            obs_data = obs_data.drop(columns=['RH'])
            # precip
            obs_data['PRECIP'] = obs_data['PRECIP']/3600
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
                print(obs_data, file=f)
        case 'SWE-Abisko':
            # abiskos data is very messy and has all kinds of non-numeric character strings which confuses python
            # python then turns all columns into objects (strings) which breaks all the math code
            # to fix this I have to force all columns to numeric which makes all non-numbers into NaNs
            cols_to_num = ['TBOT','FSDS','FLDS','PBOT','RH','WIND']
            for col in cols_to_num:
                obs_data[col] = pd.to_numeric(obs_data[col], errors='coerce')
            # convert numerical timestamp to string for datetime.strptime
            obs_data['time'] = obs_data['Timestamp (UTC)'].astype(str)
            # remove -9999 values, as numbers are float values have to use np.isclose
            #num_cols = obs_data.select_dtypes(np.number).columns
            #obs_data[num_cols] = obs_data[num_cols].mask(np.isclose(obs_data[num_cols].values, -6999))
            # change -6999 fill values to NAs
            #obs_data = obs_data.replace(-6999, np.NaN)
            # convert TBOT from celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # clean FLDS values
            obs_data.loc[obs_data['FLDS'] < 50,  'FLDS'] = np.NaN 
            # Convert from mbar to Pa
            obs_data.loc[:,'PBOT'] = obs_data['PBOT'] * 100
            obs_data.loc[obs_data['PBOT'] < 95000, 'PBOT'] = np.NaN 
            # convert RH to SH
            obs_data.loc[obs_data['RH'] < 0, 'RH'] = 0
            obs_data.loc[obs_data['RH'] > 100, 'RH'] = 100
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            obs_data = obs_data.drop(columns=['RH'])
            # precip
            obs_data['PRECIP'] = obs_data['PRECIP']/3600
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
                print(obs_data, file=f)
                print(obs_data.dtypes, file=f)
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
            obs_data = obs_data.resample('1h').mean()
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
            #obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            obs_data = obs_data.drop(columns=['RH','WIND'])
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            # No pressure given so I'll calculate air pressure given a rough reference (400m ~ 96357Pa) and plug in sites TBOT
            # into barometric pressure equation to estimate the pressure a few meters up at site elevation (424m)
            # basically adding temperature variability to reference pressure through barometric pressure function
            # I'm only doing this because I need pressure to convert RH to SH
            #g0 = 9.80665 # gavitational constat in m/s2
            #M0 = 0.0289644 # molar mass of air kg/mol
            #R0 = 8.3144598 # universal gas constant - J/(mol K)
            #hb = 0 # reference level, here just below sites elevation
            #Pb = 101325 # estimated reference pressure at 400 meters and 0 degre C 
            #obs_data.loc[:,'PBOT'] = Pb*np.exp((-g0*M0*(424-hb))/(R0*obs_data['TBOT']))
            #obs_data.loc[obs_data['PBOT'] < 90000, 'PBOT'] = np.NaN 
            # convert RH to SH
            #obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            # precip
            #obs_data['PRECIP'] = obs_data['PRECIP']/3600
            # remove uneeded columns
            obs_data = obs_data.drop(columns=['RH','PRECIP'])
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
                print(obs_data, file=f)
        case 'USA-Utqiagvik':
            # subset to BD for Barrow in strSitCom
            obs_data = obs_data.loc[obs_data['SITE'] == 'BD']
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            # remove -9999 values, as numbers are float values have to use np.isclose
            num_cols = obs_data.select_dtypes(np.number).columns
            obs_data[num_cols] = obs_data[num_cols].mask(np.isclose(obs_data[num_cols].values, -999.9))
            # remove zero wind speeds, select 2014 forward
            obs_data.loc[obs_data['WIND'] <= 0, 'WIND'] = np.nan
            obs_data.loc[obs_data.time < '2014-01-01', 'WIND'] = np.nan
            # scale celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # precip
            obs_data['PRECIP'] = obs_data['PRECIP']/3600
            # calculate SWIN from PAR
            #obs_data.loc[:,'FSDS'] = obs_data['PAR']/2.1#4.57)/0.46
            obs_data = obs_data.drop(columns = ['PAR','PRECIP'])
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
                print(obs_data, file=f)
        case 'USA-Atqasuk':
            # subset to BD for Barrow in strSitCom
            obs_data = obs_data.loc[obs_data['SITE'] == 'AD']
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            # remove -9999 values, as numbers are float values have to use np.isclose
            num_cols = obs_data.select_dtypes(np.number).columns
            obs_data[num_cols] = obs_data[num_cols].mask(np.isclose(obs_data[num_cols].values, -999.9))
            # remove zero wind speeds
            obs_data.loc[obs_data['WIND'] <= 0.0, 'WIND'] = np.nan
            obs_data.loc[obs_data.time < '2014-01-01', 'WIND'] = np.nan
            obs_data.loc[obs_data.time > '2016-01-01', 'WIND'] = np.nan
            # scale celsius to kelvin
            obs_data.loc[:,'TBOT'] = obs_data['TBOT'] + 273.15
            # precip
            obs_data['PRECIP'] = obs_data['PRECIP']/3600
            # calculate SWIN from PAR
            #obs_data.loc[:,'FSDS'] = obs_data['PAR']/2.1#4.57)/0.46
            obs_data = obs_data.drop(columns = ['PAR','PRECIP','WIND'])
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            obs_data.loc[obs_data['WIND'] <= 0, 'WIND'] = np.nan
            # calculate SH from RH/TBOT/PBOT
            obs_data.loc[obs_data['RH'] < 0, 'RH'] = 0
            obs_data.loc[obs_data['RH'] > 100, 'RH'] = 100
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            obs_data = obs_data.drop(columns=['RH'])
            # precip
            obs_data['PRECIP'] = obs_data['PRECIP']/3600
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            obs_data.loc[obs_data['RH'] < 0, 'RH'] = 0
            obs_data.loc[obs_data['RH'] > 100, 'RH'] = 100
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            # remove zero wind speeds
            obs_data.loc[obs_data['WIND'] <= 0, 'WIND'] = np.nan
            # precip
            obs_data['PRECIP'] = obs_data['PRECIP']/3600
            # remove uneeded columns
            obs_data = obs_data.drop(columns=['RH'])
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            obs_data.loc[obs_data['RH'] < 0, 'RH'] = 0
            obs_data.loc[obs_data['RH'] > 100, 'RH'] = 100
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            # remove zero wind speeds
            obs_data.loc[obs_data['WIND'] <= 0, 'WIND'] = np.nan
            # precip
            obs_data['PRECIP'] = obs_data['PRECIP']/3600
            # remove uneeded columns
            obs_data = obs_data.drop(columns=['RH'])
            # replace odd data on first rows of observation data with nan
            obs_data.loc[obs_data['FLDS'] < 100, 'FLDS'] = np.nan
            obs_data.loc[obs_data['FLDS'] > 400, 'FLDS'] = np.nan
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            quality_flags = ['good','suspect','missing','unknown']
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
            # use loc to replace all -9999 values
            obs_data.loc[obs_data['TBOT'] < -900.0, 'TBOT'] = np.nan
            obs_data.loc[obs_data['PBOT'] < -900.0, 'PBOT'] = np.nan
            obs_data.loc[obs_data['FSDS'] < -900.0, 'FSDS'] = np.nan
            obs_data.loc[obs_data['FLDS'] < -900.0, 'FLDS'] = np.nan
            obs_data.loc[obs_data['RH'] < -900, 'RH'] = np.nan
            # scale celsius to kelvin
            obs_data['TBOT'] = obs_data['TBOT'] + 273.15
            # scale mbar to kpa
            obs_data['PBOT'] = obs_data['PBOT']*100
            # calculate SH from RH/TBOT/PBOT
            obs_data.loc[obs_data['RH'] < 0, 'RH'] = 0
            obs_data.loc[obs_data['RH'] > 100, 'RH'] = 100
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            # remove zero wind speeds
            obs_data.loc[obs_data['WIND'] <= 0, 'WIND'] = np.nan
            # precip, less than zero remove and scale to rate per second
            obs_data.loc[obs_data['PRECIP'] < 0, 'PRECIP'] = np.nan
            obs_data['PRECIP'] = obs_data['PRECIP']/3600
            # remove uneeded columns
            obs_data = obs_data.drop(columns=['RH'])
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
                print('data adjusted', file=f)
                print(obs_data, file=f)
                print(obs_data.head(), file=f)
                print(obs_data.dtypes, file=f)
        case 'GRE-Disko':
            ##### read in air temp
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
            quality_flags = ['good','suspect','missing','unknown']
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
            # use loc to replace all -9999 values
            obs_data.loc[obs_data['TBOT'] < -900, 'TBOT'] = np.nan
            obs_data.loc[obs_data['PBOT'] < -900, 'PBOT'] = np.nan
            obs_data.loc[obs_data['FSDS'] < -900, 'FSDS'] = np.nan
            obs_data.loc[obs_data['FLDS'] < -900, 'FLDS'] = np.nan
            obs_data.loc[obs_data['RH'] < -900, 'RH'] = np.nan
            # scale celsius to kelvin
            obs_data['TBOT'] = obs_data['TBOT'] + 273.15
            # scale mbar to kpa
            obs_data['PBOT'] = obs_data['PBOT']*100
            # calculate SH from RH/TBOT/PBOT
            obs_data.loc[obs_data['RH'] < 0, 'RH'] = 0
            obs_data.loc[obs_data['RH'] > 100, 'RH'] = 100
            obs_data.loc[:,'QBOT'] = specific_humidity(obs_data['RH'], obs_data['TBOT'], obs_data['PBOT'])
            # remove zero wind speeds
            obs_data.loc[obs_data['WIND'] <= 0, 'WIND'] = np.nan
            # precip, less than zero remove and scale to rate per second
            obs_data.loc[obs_data['PRECIP'] < 0, 'PRECIP'] = np.nan
            obs_data['PRECIP'] = obs_data['PRECIP']/3600
            # remove uneeded columns
            obs_data = obs_data.drop(columns=['RH'])
            with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
                print('data adjusted', file=f)
                print(obs_data, file=f)
                print(obs_data.head(), file=f)
                print(obs_data.dtypes, file=f)
    try:        
        # make final QQ adjustments
        try:
            obs_data.loc[obs_data['FSDS'] < 0, 'FSDS'] = 0
        except:
            pass
        try:
            obs_data.loc[obs_data['FLDS'] < 0, 'FLDS'] = 0
        except:
            pass
        try:
            obs_data.loc[obs_data['WIND'] < 0, 'WIND'] = 0
        except:
            pass
        try:
            obs_data.loc[obs_data['PRECIP'] < 0, 'PRECIP'] = 0
        except:
            pass
        try:
            obs_data.loc[obs_data['QBOT'] < 0, 'QBOT'] = 0
        except:
            pass
        # drop old date/time columns
        obs_data = obs_data.drop(columns=config['obs']['f1']['datetime_cols'], errors='ignore')    
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
            print(obs_data['time'], file=f)
            print('past1', file=f)
        # remove duplicate timestamps
        obs_data = obs_data.drop_duplicates(subset='time')
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
            print('past2', file=f)
        # create datetime values from numerical timestamp after conversion to string
        obs_data.loc[:,'time'] = obs_data['time'].apply(lambda x: datetime.strptime(x, config['obs']['f1']['datetime_format']))
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
            print('past3', file=f)
        # create new index that can fill missing timesteps
        new_index = pd.date_range(start=obs_data.at[obs_data.index[0],'time'], \
                                  end=obs_data.at[obs_data.index[-1],'time'], freq=config['obs']['f1']['freq'])
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
            print('past4', file=f)
        # set dateime as index
        print(obs_data['time'].dtypes)
        obs_data = obs_data.set_index(['time'])
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
            print('past5', file=f)
        # create xarray dataset
        ds = obs_data.to_xarray()
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
            print('past6', file=f)
        # use reindex to add the missing timesteps and fill data values with na as default
        ds = ds.reindex({"time": new_index})
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
            print('past7', file=f)
        # convert to 365_day calendar using dataset.convert_calendar (drops leap days)
        ds = ds.convert_calendar("365_day")
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
            print('past8', file=f)
        # shift time index by offset from GMT described in observational dataset
        ds.coords['time'] = ds.indexes['time'].shift(config['obs_GMT_adj'], 'h')
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
            print('past9', file=f)
            print(ds, file=f)
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
        with open(Path(config['site_dir'] + 'debug_obs.txt'), 'a') as f:
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
        elif sub_win == '':
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
                    for weekofyear in range(1,53):
                        avg_sub_idx.append('W'+str(weekofyear).zfill(3))
        case 'month':
            match sub_win:
                case 'hour':
                    # loop avg and sub windows to create list
                    for month in range(1,13):
                        for hour in range(0,24):
                            avg_sub_idx.append('W'+str(month).zfill(3)+'S'+str(hour).zfill(3))
                case '':
                    for month in range(1,13):
                        avg_sub_idx.append('W'+str(month).zfill(3))
    # zip/map a list of integers to list of strings that represent
    mapped_dict = dict(zip(avg_sub_idx, range(1, len(avg_sub_idx)+1)))
    # apply dictionary to datasets input time (as data array
    new_groups = time.map(lambda x: mapped_dict[user_groupby(x, avg_win, sub_win)])
    return new_groups

# convert simulation and obvervation netcdf to multi-year daily means    
def multiyear_means(f_iter):
    # take values from zip for config and nc_type(i.e. CRUJRA vs Obs)
    config = read_config(f_iter[0])
    nc_type = f_iter[1] 
    # create input file name
    nc_file = Path(config['site_dir'] + nc_type + "_" + config['site_name'] + "_dat.nc")
    # create file output name
    nc_out_dw = Path(config['site_dir'] + nc_type + "_" + config['site_name'] + "_dw_mym.nc")
    nc_out_w  = Path(config['site_dir'] + nc_type + "_" + config['site_name'] + "_w_mym.nc")
    nc_out_m  = Path(config['site_dir'] + nc_type + "_" + config['site_name'] + "_m_mym.nc")
    # open netcdf file using context manager and xarray
    try:
        with xr.open_dataset(nc_file, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
            ds = ds_tmp.load()
    except:
        # if no obs data repeat procedure on cru data; allows rest of code to execute and bias=0 so no correction (crujra from cell is maintained) 
        nc_file = Path(config['site_dir'] + "CRUJRA_" + config['site_name'] + "_dat.nc")
        with xr.open_dataset(nc_file, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
            ds = ds_tmp.load()
        pass
        #return
    # remove zero precipitation events for average rate of rainfall when it occurs
    #if 'PRECIP' in list(ds.data_vars):
    #    ds['PRECIP'].loc[ds['PRECIP'] <= 0] = np.nan
    # create user defined groupby values to do non-standard time averages
    # new grouping coordinate must be integer data type and added to time dimenion
    ds_dw = ds.assign_coords(groupvar = ('time', map_groups(ds.indexes['time'], 'weekofyear', 'hour', config)))
    ds_w  = ds.assign_coords(groupvar = ('time', map_groups(ds.indexes['time'], 'weekofyear', '', config)))
    ds_m  = ds.assign_coords(groupvar = ('time', map_groups(ds.indexes['time'], 'month', '', config)))
    # groupby new coordinate
    ds_dw = ds_dw.groupby('groupvar').mean() 
    ds_w = ds_w.groupby('groupvar').mean() 
    ds_m = ds_m.groupby('groupvar').mean() 
    ## set netcdf write characteristics for xarray.to_netcdf()
    comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
    # create encoding
    encoding = {var: comp for var in ds.data_vars}
    # save file
    ds_dw.to_netcdf(nc_out_dw, mode="w", \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
    ds_w.to_netcdf(nc_out_w, mode="w", \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
    ds_m.to_netcdf(nc_out_m, mode="w", \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])

# calculate additive/multiplicative offset (bias)
def bias_calculation(f_iter):
    # read in config
    config = read_config(f_iter[0])
    bias_type = f_iter[1]
    time_avg = f_iter[2]
    with open(Path(config['site_dir'] + '/debug_' + time_avg + '.txt'), 'w') as f:
        print('bias calculation started', file=f)
    # create file names for CRUJRA and obs mydm
    f_cru = Path(config['site_dir'] + "CRUJRA" + "_" + config['site_name'] + "_" +  time_avg + "_mym.nc")
    f_obs = Path(config['site_dir'] + "Obs" + "_" + config['site_name'] + "_" +  time_avg + "_mym.nc")
    bias_file = Path(config['site_dir'] + bias_type + "_" + config['site_name'] + "_" +  time_avg + "_mym.nc")
    # read cru
    with xr.open_dataset(f_cru, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_cru = ds_tmp.load()
    # read obs
    try:
        with xr.open_dataset(f_obs, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
            ds_obs = ds_tmp.load()
    except:
        ds_obs = ds_cru.copy(deep=True)
        pass
    # list of vlimate variables to attempt to adjust
    keep = ['FSDS','FLDS','PBOT','PRECIP','RAIN','SNOW','QBOT','TBOT','WIND']
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
            if bc[var].isnull().sum() < 0.1*len(bc[var]):
                bc[var] = bc[var].interpolate_na(dim='groupvar', method='nearest')
        except Exception as error:
            with open(Path(config['site_dir'] + '/debug_' + time_avg + '.txt'), 'a') as f:
                print(error, file=f)
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
    time_avg = f_iter[2]
    with open(Path(config['site_dir'] + '/debug_' + time_avg + '.txt'), 'w') as f:
        print('bias correction started', file=f)
    # create file name for fully adjusted dataset
    cru_file = Path(config['site_dir'] + "CRUJRA_" + config['site_name'] + "_allyears.nc")
    cru_bc_file = Path(config['site_dir'] + bias_type + "_" + config['site_name'] + "_" +  time_avg + "_allyears.nc")
    bias_file = Path(config['site_dir'] + bias_type + "_" + config['site_name'] + "_" +  time_avg + "_mym.nc")
    # read in bias file, otherwise skip and output unadjusted cru product file
    with xr.open_dataset(bias_file, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_bias = ds_tmp.load()
    # read in concatenated CRUJRA site file
    with xr.open_dataset(cru_file, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_cru = ds_tmp.load()
    # list of vlimate variables to attempt to adjust
    keep = ['FSDS','FLDS','PBOT','PRECIP','RAIN','SNOW','QBOT','TBOT','WIND']
    ds_cru = ds_cru[keep]
    # create user defined groupby values to do non-standard time averages
    # new grouping coordinate must be integer data type and added to time dimenion
    if time_avg == 'dw':
        main_window = 'weekofyear'
        sub_window = 'hour'
    elif time_avg == 'w':
        main_window = 'weekofyear'
        sub_window = ''
    elif time_avg == 'm':
        main_window = 'month'
        sub_window = ''
    ds_cru = ds_cru.assign_coords(groupvar = ('time', map_groups(ds_cru.indexes['time'], main_window, sub_window, config)))
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
            with open(Path(config['site_dir'] + '/debug_' + time_avg + '.txt'), 'a') as f:
                print(error, file=f)
            pass
    # final QQ for correct files
    try:
        ds_cru['FSDS'] = ds_cru['FSDS'].where(ds_cru['FSDS'] > 0, 0)
    except:
        pass
    try:
        ds_cru['PRECIP'] = ds_cru['PRECIP'].where(ds_cru['PRECIP'] > 0, 0)
    except:
        pass
    try:
        ds_cru['WIND'] = ds_cru['WIND'].where(ds_cru['WIND'] > 0, 0)
    except:
        pass
    try:
        ds_cru['QBOT'] = ds_cru['QBOT'].where(ds_cru['QBOT'] > 0, 0.000001)
    except:
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

# summary stats on crujra 
def combine_climate_and_corrections(input_list):
    # read in config file with list of site folder
    config_files = input_list[0]
    dir_out = input_list[1] 
    kwargs = {'mask_and_scale': True}
    # collect file names from each for for Mbias crujra files
    file_list = []
    for site in config_files:
        site_config = read_config(site)
        crujra_file = Path(site_config['site_dir'] + '/CRUJRA_' + site_config['site_name'] + '_allyears.nc')
        mbias_file = Path(site_config['site_dir'] + '/MBias_' + site_config['site_name'] + '_dw_allyears.nc')
        data_file = Path(site_config['site_dir'] + '/Obs_' + site_config['site_name'] + '_dat.nc')
        file_list.append([site_config['site_name'], site_config, crujra_file, mbias_file, data_file])  
    # loop through files appending the site dimension
    ds_list_cru = []
    ds_list_mbias = []
    ds_list_obs = []
    iterator = 1    
    for file_name in file_list:
        # extract site info list
        site_name = file_name[0]
        site_config = file_name[1]
        crujra_ncfile = file_name[2]
        mbias_ncfile = file_name[3]
        obs_ncfile = file_name[4]
        # open site file
        with xr.open_dataset(crujra_ncfile, engine=site_config['nc_read']['engine'], decode_cf=True, use_cftime=True, **kwargs) as ds_tmp:
            ds_cru = ds_tmp.load()
        with xr.open_dataset(mbias_ncfile, engine=site_config['nc_read']['engine'], decode_cf=True, use_cftime=True, **kwargs) as ds_tmp:
            ds_mbias = ds_tmp.load()
        try:
            with xr.open_dataset(obs_ncfile, engine=site_config['nc_read']['engine'], decode_cf=True, use_cftime=True, **kwargs) as ds_tmp:
                ds_obs = ds_tmp.load()
        except:
            pass
        # remove RAIN and SNOW
        ds_cru = ds_cru.drop_vars(['RAIN','SNOW'])
        ds_cru = ds_cru.assign_coords({'site': iterator})
        ds_cru = ds_cru.assign_coords({'data_type': 'cru'})
        ds_cru = ds_cru.assign({'site_name': site_name})
        ds_cru = ds_cru.assign({'lon': site_config['lon']})
        ds_cru = ds_cru.assign({'lat': site_config['lat']})
        ds_cru = ds_cru.expand_dims('site')
        ds_cru = ds_cru.expand_dims('data_type')
        ds_mbias = ds_mbias.drop_vars(['RAIN','SNOW'])
        if site_name in ['CAN-WanderingRiver','SVA-Endalen']:
            ds_obs = ds_mbias.copy(deep=True)
            for var in ['FSDS','FLDS','PBOT','QBOT','TBOT','WIND','PRECIP']:
                ds_obs[var].loc[:] = np.nan
        ds_mbias = ds_mbias.assign_coords({'site': iterator})
        ds_mbias = ds_mbias.assign_coords({'data_type': 'mbc'})
        ds_mbias = ds_mbias.assign({'site_name': site_name})
        ds_mbias = ds_mbias.assign({'lon': site_config['lon']})
        ds_mbias = ds_mbias.assign({'lat': site_config['lat']})
        ds_mbias = ds_mbias.expand_dims('site')
        ds_mbias = ds_mbias.expand_dims('data_type')
        ds_obs = ds_obs.assign_coords({'site': iterator})
        ds_obs = ds_obs.assign_coords({'data_type': 'obs'})
        ds_obs = ds_obs.assign({'site_name': site_name})
        ds_obs = ds_obs.assign({'lon': site_config['lon']})
        ds_obs = ds_obs.assign({'lat': site_config['lat']})
        ds_obs = ds_obs.expand_dims('site')
        ds_obs = ds_obs.expand_dims('data_type')
        # append dataset to list for later merging
        ds_list_cru.append(ds_cru)
        ds_list_mbias.append(ds_mbias)
        ds_list_obs.append(ds_obs)
        iterator += 1
    # merge dataset
    ds_sites_cru = xr.merge(ds_list_cru)
    ds_sites_mbias = xr.merge(ds_list_mbias)
    ds_sites_obs = xr.merge(ds_list_obs)
    # reindex time by
    start_time = '1901-01-01 00:00:00'
    end_time = '2021-12-31 23:00:00'
    ds_sites_cru = ds_sites_cru.reindex({'time': xr.cftime_range(start=start_time, end=end_time, freq='h', calendar='noleap')})
    ds_sites_mbias = ds_sites_mbias.reindex({'time': xr.cftime_range(start=start_time, end=end_time, freq='h', calendar='noleap')})
    ds_sites_obs = ds_sites_obs.reindex({'time': xr.cftime_range(start=start_time, end=end_time, freq='h', calendar='noleap')})
    # final merged file
    ds_final = xr.merge([ds_sites_cru,ds_sites_mbias,ds_sites_obs])
    # output files
    config = read_config(config_files[0])
    cru_out = dir_out + 'CRUJRA_sites.nc' 
    mbias_out = dir_out + 'MBias_sites.nc' 
    obs_out = dir_out + 'Obs_sites.nc' 
    final_out = dir_out + 'Combined_climate_sites.nc'
    ds_sites_cru.to_netcdf(cru_out, mode="w", \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
    ds_sites_mbias.to_netcdf(mbias_out, mode="w", \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
    ds_sites_obs.to_netcdf(obs_out, mode="w", \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
    ds_final.to_netcdf(final_out, mode="w", \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])

# summarize final corrections
def correction_summary():
    # define final file location
    out_dir = '/projects/warpmip/shared/forcing_data/biascorrected_forcing/'
    file_name = '/projects/warpmip/shared/forcing_data/biascorrected_forcing/Combined_climate_sites.nc'
    # read in file
    kwargs = {'mask_and_scale': True}
    with xr.open_dataset(file_name, engine='netcdf4', decode_cf=True, use_cftime=True, **kwargs) as ds_tmp:
        ds = ds_tmp.load()
    ds['PBOT'] = ds['PBOT'] / 1000
    with open(Path(out_dir + 'debug_summary.txt'), 'w') as f:
        print('Combined dataset:', file=f)
        print(ds, file=f)
        print('time coords', file=f)
        print(ds['time'].values, file=f)
    # # plot data
    # for var in ['FSDS','FLDS','PBOT','QBOT','TBOT','WIND','PRECIP']:
    #     fig = plt.figure(figsize=(10,10))
    #     kwargs = {'alpha': 0.8}
    #     ds[var].sel(time=slice('1990','2022')).plot(col='site', col_wrap=4, hue='data_type', **kwargs)
    #     plt.savefig(Path(out_dir + var + '_compare.png'), dpi=300)
    #     plt.close(fig)
    # define function to take RSME between to data types
    def rmse_between_data_types(ds, y_pred, y, var, dim_reduce):
        differences = ds[var].sel(data_type=y_pred) - ds[var].sel(data_type=y)
        with open(Path(out_dir + 'debug_summary.txt'), 'a') as f:
            print('differences for ' + y_pred + ' and ' + var + ':', file=f)
            print(differences, file=f)
        diff_sqrd = differences ** 2
        with open(Path(out_dir + 'debug_summary.txt'), 'a') as f:
            print('differences squared for ' + y_pred + ' and ' + var + ':', file=f)
            print(diff_sqrd, file=f)
        mean_diff_sqrd = diff_sqrd.mean(dim=dim_reduce, skipna=True)
        with open(Path(out_dir + 'debug_summary.txt'), 'a') as f:
            print('mean differences squared for ' + y_pred + ' and ' + var + ':', file=f)
            print(mean_diff_sqrd, file=f)
        rmse_value = np.sqrt(mean_diff_sqrd)
        with open(Path(out_dir + 'debug_summary.txt'), 'a') as f:
            print('rmse values for ' + y_pred + ' and ' + var + ':', file=f)
            print(rmse_value, file=f)
        return rmse_value
    # calculate RSME of crujra vs obs / mbias vs obs
    site_names = ds['site_name'].sel(data_type='obs').values
    climate_name = 'Climate'
    data_name = 'Driver'
    tbot_cru_rmse = rmse_between_data_types(ds,'cru','obs','TBOT','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'TBOT': 'RMSE', 'index': 'Site'})
    tbot_cru_rmse[climate_name] = 'Crujra_v2.3'
    tbot_cru_rmse[data_name] = 'TBOT (K)'
    tbot_mbias_rmse = rmse_between_data_types(ds,'mbc','obs','TBOT','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'TBOT': 'RMSE', 'index': 'Site'})
    tbot_mbias_rmse[climate_name] = 'Mbc'
    tbot_mbias_rmse[data_name] = 'TBOT (K)'
    pbot_cru_rmse = rmse_between_data_types(ds,'cru','obs','PBOT','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'PBOT': 'RMSE', 'index': 'Site'})
    pbot_cru_rmse[climate_name] = 'Crujra_v2.3'
    pbot_cru_rmse[data_name] = 'PBOT (kPa)'
    pbot_mbias_rmse = rmse_between_data_types(ds,'mbc','obs','PBOT','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'PBOT': 'RMSE', 'index': 'Site'})
    pbot_mbias_rmse[climate_name] = 'Mbc'
    pbot_mbias_rmse[data_name] = 'PBOT (kPa)'
    qbot_cru_rmse = rmse_between_data_types(ds,'cru','obs','QBOT','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'QBOT': 'RMSE', 'index': 'Site'})
    qbot_cru_rmse[climate_name] = 'Crujra_v2.3'
    qbot_cru_rmse[data_name] = 'QBOT (kg/kg)'
    qbot_mbias_rmse = rmse_between_data_types(ds,'mbc','obs','QBOT','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'QBOT': 'RMSE', 'index': 'Site'})
    qbot_mbias_rmse[climate_name] = 'Mbc'
    qbot_mbias_rmse[data_name] = 'QBOT (kg/kg)'
    fsds_cru_rmse = rmse_between_data_types(ds,'cru','obs','FSDS','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'FSDS': 'RMSE', 'index': 'Site'})
    fsds_cru_rmse[climate_name] = 'Crujra_v2.3'
    fsds_cru_rmse[data_name] = 'FSDS (W/m2)'
    fsds_mbias_rmse = rmse_between_data_types(ds,'mbc','obs','FSDS','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'FSDS': 'RMSE', 'index': 'Site'})
    fsds_mbias_rmse[climate_name] = 'Mbc'
    fsds_mbias_rmse[data_name] = 'FSDS (W/m2)'
    flds_cru_rmse = rmse_between_data_types(ds,'cru','obs','FLDS','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'FLDS': 'RMSE', 'index': 'Site'})
    flds_cru_rmse[climate_name] = 'Crujra_v2.3'
    flds_cru_rmse[data_name] = 'FLDS (W/m2)'
    flds_mbias_rmse = rmse_between_data_types(ds,'mbc','obs','FLDS','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'FLDS': 'RMSE', 'index': 'Site'})
    flds_mbias_rmse[climate_name] = 'Mbc'
    flds_mbias_rmse[data_name] = 'FLDS (W/m2)'
    wind_cru_rmse = rmse_between_data_types(ds,'cru','obs','WIND','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'WIND': 'RMSE', 'index': 'Site'})
    wind_cru_rmse[climate_name] = 'Crujra_v2.3'
    wind_cru_rmse[data_name] = 'WIND (m/s)'
    wind_mbias_rmse = rmse_between_data_types(ds,'mbc','obs','WIND','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'WIND': 'RMSE', 'index': 'Site'})
    wind_mbias_rmse[climate_name] = 'Mbc'
    wind_mbias_rmse[data_name] = 'WIND (m/s)'
    prec_cru_rmse = rmse_between_data_types(ds,'cru','obs','PRECIP','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'PRECIP': 'RMSE', 'index': 'Site'})
    prec_cru_rmse[climate_name] = 'Crujra_v2.3'
    prec_cru_rmse[data_name] = 'PRECIP (mm/s)'
    prec_mbias_rmse = rmse_between_data_types(ds,'mbc','obs','PRECIP','time').to_dataframe().set_index(site_names).reset_index().rename(columns={'PRECIP': 'RMSE', 'index': 'Site'})
    prec_mbias_rmse[climate_name] = 'Mbc'
    prec_mbias_rmse[data_name] = 'PRECIP (mm/s)'
    # check data
    with open(Path(out_dir + 'debug_summary.txt'), 'a') as f:
        print('CRUJRA vs obs:', file=f)
        print(tbot_cru_rmse, file=f)
        print(tbot_cru_rmse.values, file=f)
        print('Mbias vs obs:', file=f)
        print(tbot_mbias_rmse, file=f)
        print(tbot_mbias_rmse.values, file=f)
    # combine data into long format
    df_tbot = tbot_cru_rmse.merge(tbot_mbias_rmse, on=['Site','Climate','Driver','RMSE'], how='outer')
    df_pbot = pbot_cru_rmse.merge(pbot_mbias_rmse, on=['Site','Climate','Driver','RMSE'], how='outer')
    df_qbot = qbot_cru_rmse.merge(qbot_mbias_rmse, on=['Site','Climate','Driver','RMSE'], how='outer')
    df_fsds = fsds_cru_rmse.merge(fsds_mbias_rmse, on=['Site','Climate','Driver','RMSE'], how='outer')
    df_flds = flds_cru_rmse.merge(flds_mbias_rmse, on=['Site','Climate','Driver','RMSE'], how='outer')
    df_wind = wind_cru_rmse.merge(wind_mbias_rmse, on=['Site','Climate','Driver','RMSE'], how='outer')
    df_prec = prec_cru_rmse.merge(prec_mbias_rmse, on=['Site','Climate','Driver','RMSE'], how='outer')
    with open(Path(out_dir + 'debug_summary.txt'), 'a') as f:
        print('Merged tbot dataframe:', file=f)
        print(df_tbot, file=f)
    # concat the responses
    df = pd.concat([df_tbot,df_pbot,df_qbot,df_fsds,df_flds,df_wind,df_prec], ignore_index=True)
    df = df[['Climate','Driver','Site','RMSE']]
    df = df.sort_values(['Site','Climate','Driver'])
    with open(Path(out_dir + 'debug_summary.txt'), 'a') as f:
        print('Merged dataframe of all climate drivers:', file=f)
        print(df, file=f)
    # make faceted seabonr plot
    fig = plt.figure()
    sns.set(style='white', font_scale=1.2)
    g = sns.catplot(kind='bar', data=df, x='Site', y='RMSE', hue='Climate', col='Driver', color=['orange','dodgerblue'], col_wrap=4, \
                    palette=sns.color_palette(['orange','dodgerblue']), sharey=False, height=6, aspect=0.8, edgecolor='none')
    sns.move_legend(g, "upper left", bbox_to_anchor=(.80, .37), frameon=False, title='Climate Source', fontsize=20, title_fontsize='xx-large') 
    g.tick_params(axis='x', which='both', rotation=90, labelsize=12) 
    g.fig.set_figheight(15)
    g.fig.set_figwidth(20)
    axes = g.axes.flatten()
    axes[0].set_ylabel('RMSE (W/m2)')
    axes[0].set_title('FLDS')
    axes[1].set_ylabel('RMSE (W/m2)')
    axes[1].set_title('FSDS')
    axes[2].set_ylabel('RMSE (kPa)')
    axes[2].set_title('PBOT')
    axes[3].set_ylabel('RMSE (mm/s)')
    axes[3].set_title('PRECIP')
    axes[4].set_ylabel('RMSE (kg/kg)')
    axes[4].set_title('QBOT')
    axes[5].set_ylabel('RMSE (k)')
    axes[5].set_title('TBOT')
    axes[6].set_ylabel('RMSE (m/s)')
    axes[6].set_title('WIND')
    plt.tight_layout()
    #matplotlib.rcParams['axes.grid'] = True
    #matplotlib.rcParams['savefig.transparent'] = True 
    plt.savefig(Path(out_dir + 'RMSE_barplot.png'), dpi=300, transparent=True)
    plt.close(fig)
 
# function to take correct mbc files and combine them together
def combine_corrected_climate(input_list):
    # read in config file with list of site folder
    config_files = input_list[0]
    nc_out = Path(input_list[1]) 
    # collect file names from each for for Mbias crujra files
    file_list = []
    for site in config_files:
        site_config = read_config(site)
        site_file = Path(site_config['site_dir'] + '/MBias_' + site_config['site_name'] + '_dw_allyears.nc')
        file_list.append([site_config['site_name'], site_config, site_file])  
    # loop through files appending the site dimension
    ds_list = []
    iterator = 1    
    for file_name in file_list:
        # extract site info list
        site_name = file_name[0]
        site_config = file_name[1]
        site_ncfile = file_name[2]
        # open site file
        with xr.open_dataset(site_ncfile, engine=site_config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
            ds = ds_tmp.load()
        # remove RAIN and SNOW
        ds = ds.drop_vars(['RAIN','SNOW'])
        # assign simulation coordinate
        ds = ds.assign_coords({'site': iterator})
        ds = ds.assign({'site_name': site_name})
        ds = ds.assign({'lon': site_config['lon']})
        ds = ds.assign({'lat': site_config['lat']})
        ds = ds.expand_dims('site')
        # append dataset to list for later merging
        ds_list.append(ds)
        iterator += 1
    # merge dataset
    ds_sites = xr.merge(ds_list)
    # set netcdf write characteristics for xarray.to_netcdf()
    config = read_config(config_files[0])
    #comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
    #            complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
    ## create encoding
    #encoding = {var: comp for var in ds_sites.data_vars}
    # add long name and units to all data variables
    ds_sites['FSDS'] = ds_sites['FSDS'].assign_attrs(
        units='W/m2', description='Downwelling shortwave radiation')
    ds_sites['FLDS'] = ds_sites['FLDS'].assign_attrs(
        units='W/m2', description='Downwelling longwave radiation')
    ds_sites['WIND'] = ds_sites['WIND'].assign_attrs(
        units='m/s', description='Wind speed')
    ds_sites['TBOT'] = ds_sites['TBOT'].assign_attrs(
        units='K', description='Air temperature')
    ds_sites['PBOT'] = ds_sites['PBOT'].assign_attrs(
        units='Pa', description='Air pressure')
    ds_sites['QBOT'] = ds_sites['QBOT'].assign_attrs(
        units='kg/kg', description='Specific humidity')
    ds_sites['PRECIP'] = ds_sites['PRECIP'].assign_attrs(
        units='mm/s', description='Total precipitation rate (before rain/snow separation)')
    ds_sites['site_name'] = ds_sites['site_name'].assign_attrs(
        description='Abbreviated country and full site name')
    ds_sites['lon'] = ds_sites['lon'].assign_attrs(
        units = 'degrees east',
        description='longitude; [-180 180] centered on prime meridian')
    ds_sites['lat'] = ds_sites['lat'].assign_attrs(
        units = 'degrees north',
        description='latitude; [-90 90] centered on equator')
    # drop groupvar
    ds_sites = ds_sites.drop_vars('groupvar')
    # select timeframe
    start_time = '1901-01-01 00:00:00'
    end_time = '2021-12-31 23:00:00'
    ds_sites = ds_sites.sel(time=slice(start_time, end_time))
    # delete previous attributes
    global_attrs = list(ds_sites.attrs)
    with open(Path('/projects/warpmip/shared/forcing_data/biascorrected_forcing/debug_nc.txt'), 'w') as f:
        print(global_attrs, file=f)
    for item in global_attrs:
        del ds_sites.attrs[item]
    with open(Path('/projects/warpmip/shared/forcing_data/biascorrected_forcing/debug_nc.txt'), 'a') as f:
        for varname, da in ds_sites.data_vars.items():
            print(da.attrs, file=f)
    # change global file attributes
    ds_sites.attrs.update(
        creator='Jon M Wells',
        project='The Warming Permafrost Model Intercomparion Project (WrPMIP)',
        updated=pd.Timestamp.now(tz='MST').strftime('%c'),
        timestep='1-hourly data interpolated from 6H CRUJRAv2.3 product using CLM5',
        method='1-hourly CRUJRAv2.3 (1901-2021) biascorrected by diurnal weekly multi-year means',
        sites='14 sites as focus of WrPMIP site-level simulation experiments'
    )
    # print file output for debug
    with option_context('display.max_rows', 10, 'display.max_columns', 10):
        with open(Path('/projects/warpmip/shared/forcing_data/biascorrected_forcing/debug_nc.txt'), 'a') as f:
            print(ds_sites, file=f)
    # save file
    ds_sites.to_netcdf(nc_out, mode="w", \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])

# function to ingest CLM5 formatted 14-site climate files and replace data with biascorrected forcing
def replace_clm_14site_climate(input_list):
    # read in config_clmsites
    config = read_config(input_list[0])
    # read in bc file location
    bc_file = Path(input_list[1])
    surf_file = Path(input_list[2]) 
    # remove previous copy of crujra folder
    rmv_dir(config['new_dir'])
    # remake directory for subset files, grant permission
    Path(config['new_dir']).mkdir(parents=True, exist_ok=True)
    # open surface dataset
    with xr.open_dataset(surf_file, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds = ds_tmp.load()
    try:   
        # add experimental warming onset for OTCs
        with open(Path(config['new_dir']+'/debug_siteclimate.txt'), 'w') as f:
            print('Starting surface data addition of warming onset:\n', file=f)
            print(ds, file=f)
            print('Longitude of current gridcells:\n', file=f)
            print(ds['LONGXY'], file=f)
        # define OTC and SF experimental start dates at each site
        otc_start = [2008,2015,1994,1996,2005,2011,2012,2013,2007,2017,2012,2003,2002,1994]
        sf_start = [2008,2015,1994,1996,2005,2011,2017,2012,2007,2017,2012,2003,2002,1994]
        site_names = ['USA-EightMileLake','USA-Toolik','USA-Utqiagvik','USA-Atqasuk','CAN-DaringLake',
                     'CAN-WanderingRiver','CAN-CambridgeBay','GRE-Disko','GRE-Zackenberg','NOR-Iskoras',
                     'RUS-Seida','SVA-Adventdalen','SVA-Endalen','SWE-Abisko']
        # create copy of original ds to make final adjustments
        ds_adj = ds.copy(deep=True)
        # replace Svalbard ORGANIC values with Zackenburg as next closest site in lat
        ds_adj['ORGANIC'].loc[:,11] = ds_adj['ORGANIC'][:,8]
        ds_adj['ORGANIC'].loc[:,12] = ds_adj['ORGANIC'][:,8]
        # change all natveg percentages to 100%, and other patch percentages to zero
        ds_adj['PCT_NATVEG'].values = [100.0] * 14
        ds_adj['PCT_CROP'].values =     [0.0] * 14
        ds_adj['PCT_GLACIER'].values =  [0.0] * 14
        ds_adj['PCT_LAKE'].values =     [0.0] * 14
        ds_adj['PCT_URBAN'].values =   [[0.0] * 14] * 3 
        ds_adj['PCT_WETLAND'].values =  [0.0] * 14
        # add site names variable
        ds_adj['site_name'] = ds_adj['LONGXY']
        ds_adj['site_name'].values = site_names
        del ds_adj.site_name.attrs['long_name']
        del ds_adj.site_name.attrs['units']
        ds_adj['site_name'] = ds_adj['site_name'].assign_attrs(
             description='Site names for each experimental site (gridcell)')
        # adjust individual sites to have more realistic pfts, especially dominant species
        # bare ground to zero based on observational data generally collected from vegetated collars
        # PFTs split only into evergreen shrub(bes), deciduous shrub(bds), and arctic grass (c3ag)
        for grid in range(0,14):
            # subset zberock
            zbed = ds_adj['zbedrock'][grid]
            # check if zbedrock is below 5m, if so replace with 5m
            if zbed < 5.0:
                zbed = 5.0
            # get current grids list of PFT percentages
            pfts = ds_adj['PCT_NAT_PFT'][:,grid]
            # replace all pfts with zero
            pfts.loc[0:15] = 0.0
            # do specific things for each site based on reduced site info
            match grid:
                case 0: # USA-EightMileLake
                     pfts.loc[[9,11,12]] = [35.0,35.0,30.0]
                case 1: # USA-Toolik
                     pfts.loc[[9,11,12]] = [35.0,35.0,30.0]
                case 2: # USA-Utqiagvik
                     pfts.loc[[9,11,12]] = [20.0,50.0,30.0]
                case 3: # USA-Atqasuk
                     pfts.loc[[9,11,12]] = [50.0,20.0,30.0]
                case 4: # CAN-DaringLake
                     pfts.loc[[9,11,12]] = [50.0,20.0,30.0]
                case 5: # CAN-WanderingRiver
                     pfts.loc[[9,11,12]] = [35.0,35.0,30.0]
                case 6: # CAN-CambridgeBay
                     pfts.loc[[9,11,12]] = [35.0,35.0,30.0]
                case 7: # GRE-Disko
                     pfts.loc[[9,11,12]] = [27.0,55.0,18.0]
                case 8: # GRE-Zackenburg
                     pfts.loc[[9,11,12]] = [35.0,35.0,30.0]
                case 9: # NOR-Iskoras
                     pfts.loc[[9,11,12]] = [35.0,35.0,30.0]
                case 10: # RUS-Seida
                     pfts.loc[[9,11,12]] = [15.0,70.0,15.0]
                case 11: # SVA-Adventdalen
                     pfts.loc[[9,11,12]] = [50.0,20.0,30.0]
                case 12: # SVA-Endalen
                     pfts.loc[[9,11,12]] = [50.0,20.0,30.0]
                case 13: # SWE-Abisko
                     pfts.loc[[9,11,12]] = [35.0,35.0,30.0]
            # assign manipulated PFT/zbed values back to grid
            ds_adj['PCT_NAT_PFT'].loc[:,grid] = pfts
            ds_adj['zbedrock'].loc[grid] = zbed
        # create OTC and SF surface files from adjusted file
        ds_otc = ds_adj.copy(deep=True)
        ds_sf = ds_adj.copy(deep=True)
        # copy dataset that has correct dims into warming_onset var, replace values
        ds_otc['warming_onset'] = ds_otc['LONGXY']
        ds_otc['warming_onset'].values = otc_start
        del ds_otc.warming_onset.attrs['long_name']
        del ds_otc.warming_onset.attrs['units']
        ds_otc['warming_onset'] = ds_otc['warming_onset'].assign_attrs(
            units='year', description='Year of experimental warming onset for OTCs')
        ds_sf['warming_onset'] = ds_sf['LONGXY']
        ds_sf['warming_onset'].values = sf_start
        del ds_sf.warming_onset.attrs['long_name']
        del ds_sf.warming_onset.attrs['units']
        ds_sf['warming_onset'] = ds_sf['warming_onset'].assign_attrs(
            units='year', description='Year of experimental warming onset for SFs')
        # check surf data addition
        with open(Path(config['new_dir']+'/debug_siteclimate.txt'), 'a') as f:
            print('Site Names:\n', file=f)
            print(ds_adj['site_name'], file=f)
            print('zbedrock:\n', file=f)
            print(ds_adj['zbedrock'], file=f)
            print('PCT_CROP:\n', file=f)
            print(ds_adj['PCT_CROP'], file=f)
            print('PCT_GLACIER:\n', file=f)
            print(ds_adj['PCT_GLACIER'], file=f)
            print('PCT_LAKE:\n', file=f)
            print(ds_adj['PCT_LAKE'], file=f)
            print('PCT_URBAN:\n', file=f)
            print(ds_adj['PCT_URBAN'], file=f)
            print('PCT_WETLAND:\n', file=f)
            print(ds_adj['PCT_WETLAND'], file=f)
            print('SAND percent:\n', file=f)
            print(ds_adj['PCT_SAND'], file=f)
            print('CLAY percent:\n', file=f)
            print(ds_adj['PCT_CLAY'], file=f)
            print('ORGANIC percent:\n', file=f)
            print(ds_adj['ORGANIC'], file=f)
            print('PCT_NATVEG variable:\n', file=f)
            print(ds_adj['PCT_NATVEG'], file=f)
            print('PCT_NAT_PFT variable:\n', file=f)
            print(ds_adj['PCT_NAT_PFT'], file=f)
            for i in range(1,15):
                grid_index = i - 1
                pfts = ds_adj['PCT_NAT_PFT'][:,grid_index]
                print('\nGrid ' + str(i) + ':', file=f)
                print(pfts, file=f)
                print('Total percent: ' + str(sum(pfts)), file=f)
            print('GRE-Zackenburg zbedrock/ORGANIC:\n', file=f)
            print(ds_adj['zbedrock'][8], file=f)
            print(ds_adj['ORGANIC'][:,8], file=f)
            print('SVA-Adventdalen zbedrock/ORGANIC:\n', file=f)
            print(ds_adj['zbedrock'][11], file=f)
            print(ds_adj['ORGANIC'][:,11], file=f)
            print('SVA-Endalen zbedrock/ORGANIC:\n', file=f)
            print(ds_adj['zbedrock'][12], file=f)
            print(ds_adj['ORGANIC'][:,12], file=f)
            print('updated surface data - OTC:\n', file=f)
            print(ds_otc, file=f)
            print('warming_onset variable - OTC:\n', file=f)
            print(ds_otc['warming_onset'], file=f)
            print('updated surface data - SF:\n', file=f)
            print(ds_sf, file=f)
            print('warming_onset variable - SF:\n', file=f)
            print(ds_sf['warming_onset'], file=f)
        # output OTC/SF surfdata
        adj_fname = str(surf_file.parent) + '/' + str(surf_file.stem) + '_update.nc'
        otc_fname = str(surf_file.parent) + '/' + str(surf_file.stem) + '_otc.nc'
        sf_fname = str(surf_file.parent) + '/' + str(surf_file.stem) + '_sf.nc'
        ## set netcdf write characteristics for xarray.to_netcdf()
        comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
        # set encoding output otc file
        encoding = {var: comp for var in ds_adj.data_vars}
        ds_adj.to_netcdf(adj_fname, mode="w", encoding=encoding, \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
        # set encoding output otc file
        encoding = {var: comp for var in ds_otc.data_vars}
        ds_otc.to_netcdf(otc_fname, mode="w", encoding=encoding, \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
        # set encoding output sf file
        encoding = {var: comp for var in ds_sf.data_vars}
        ds_sf.to_netcdf(sf_fname, mode="w", encoding=encoding, \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
    except Exception as error:
        with open(Path(config['new_dir']+'/debug_siteclimate.txt'), 'a') as f:
            print(error, file=f)
    # open biascorrected climate dataset file
    with xr.open_dataset(bc_file, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds = ds_tmp.load()
    # count nan by data array
    fsds_nan = ds['FSDS'].isnull().sum() 
    flds_nan = ds['FLDS'].isnull().sum()
    pbot_nan = ds['PBOT'].isnull().sum()
    tbot_nan = ds['TBOT'].isnull().sum()
    qbot_nan = ds['QBOT'].isnull().sum()
    prec_nan = ds['PRECIP'].isnull().sum()
    wind_nan = ds['WIND'].isnull().sum()
    # create list of precip/solar/TPQW files by year (1901-2021)
    year_list = list(range(1901,2021+1,1))
    file_list = []
    for year in year_list:
        files = glob.glob("{nc_path}/*{nc_year}.nc".format(nc_path=config['file_dir'], nc_year=year)) 
        prec_file = [f for f in files if 'Prec' in f]
        solr_file = [f for f in files if 'Solr' in f]
        tpqw_file = [f for f in files if 'TPQWL' in f]
        file_list.append([year, prec_file[0], solr_file[0], tpqw_file[0]])
    # reset debug print statement before loop
    with open(Path(config['new_dir']+'/debug_siteclimate.txt'), 'a') as f:
        print('Starting data replacement loop:\n', file=f)
    # loop through 1901-2021 file groups
    for file_group in file_list:
        # open annual files for precip/solar/tpqwl file for that year
        with open(Path(config['new_dir']+'/debug_siteclimate.txt'), 'a') as f:
            print(file_group[1], file=f)
            print(file_group[2], file=f)
            print(file_group[3], file=f)
        # slice bc subset by year
        time_start = str(file_group[0]) + '-01-01 00:00:00'
        time_end = str(file_group[0]) + '-12-31 23:00:00'
        ds_sub = ds.sel(time=slice(time_start,time_end)).copy(deep=True)
        # print statement to debug before data change
        with option_context('display.max_rows', 10, 'display.max_columns', 10):
            with open(Path(config['new_dir']+'/debug_siteclimate.txt'), 'a') as f:
                print('\nYear of subset data:\n', file=f)
                print(file_group[0], file=f)
                print(time_start, file=f)
                print(time_end, file=f)
                print('\nBiascorrected data:\n', file=f)
                print(ds_sub, file=f) 
                print('\nFSDS nan count:\n', file=f)
                print(fsds_nan, file=f) 
                print('\nFLDS nan count:\n', file=f)
                print(flds_nan, file=f) 
                print('\nPBOT nan count:\n', file=f)
                print(pbot_nan, file=f) 
                print('\nTBOT nan count:\n', file=f)
                print(tbot_nan, file=f) 
                print('\nQBOT nan count:\n', file=f)
                print(qbot_nan, file=f) 
                print('\nPRECIP nan count:\n', file=f)
                print(prec_nan, file=f) 
                print('\nWIND nan count:\n', file=f)
                print(wind_nan, file=f) 
        # reformat names to match output dataarrays
        ds_sub = ds_sub.rename({
            'lon':'LONGXY',
            'lat':'LATIXY',
            'PRECIP':'PRECTmms',
            'PBOT':'PSRF'})
        # convert lon to [0 360]
        ds_sub['LONGXY'] =(ds_sub['LONGXY'] % 360)
        # sort dataset by longitude
        ds_sub = ds_sub.sortby('LONGXY')
        # assign new site coords after longitude ordering, rename lon, drop site coord
        ds_sub = ds_sub.rename_dims({'site':'lon'})
        #ds_sub = ds_sub.drop_vars('site')
        # add dimension with 1 value for lat
        ds_sub = ds_sub.expand_dims(dim='lat')
        # transpose order
        ds_sub = ds_sub.transpose('time','lat','lon') 
        # load clm site files for precip/solar/tpqwl
        with xr.open_dataset(Path(file_group[1]), engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
            ds_prec = ds_tmp.load()
        with xr.open_dataset(Path(file_group[2]), engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
            ds_solr = ds_tmp.load()
        with xr.open_dataset(Path(file_group[3]), engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
            ds_tpqw = ds_tmp.load()
        # print statement to debug before data change
        with option_context('display.max_rows', 10, 'display.max_columns', 10):
            with open(Path(config['new_dir']+'/debug_siteclimate.txt'), 'a') as f:
                print(ds_sub['site_name'], file=f)
                print('\nReformatted Biascorrected data:\n', file=f)
                print(ds_sub, file=f)
                print('\nCLM5 CRUJRAv2.3 Precip site formatted data:\n', file=f)
                print(ds_prec, file=f)
                print('\nCLM5 CRUJRAv2.3 Radiation site formatted data:\n', file=f)
                print(ds_solr, file=f)
                print('\nCLM5 CRUJRAv2.3 Other site formatted data:\n', file=f)
                print(ds_tpqw, file=f)
        # adjust datafiles to have 365_day calendar and hourly timestep
        new_index = pd.date_range(start=time_start, end=time_end, freq='1H')
        with open(Path(config['new_dir']+'/debug_siteclimate.txt'), 'a') as f:
            print('\nnew index for dated:\n', file=f)
            print(new_index,file=f)
        new_index = new_index[~((new_index.day == 29) & (new_index.month == 2))]
        ds_prec = ds_prec.reindex({"time": new_index})
        ds_solr = ds_solr.reindex({"time": new_index})
        ds_tpqw = ds_tpqw.reindex({"time": new_index})
        ds_prec = ds_prec.convert_calendar("365_day")
        ds_solr = ds_solr.convert_calendar("365_day")
        ds_tpqw = ds_tpqw.convert_calendar("365_day")
        # reindexing added time dimension to lat/lon, remove
        ds_prec['LONGXY'] = ds_prec['LONGXY'].isel(time=1).drop('time')    
        ds_solr['LONGXY'] = ds_solr['LONGXY'].isel(time=1).drop('time')
        ds_tpqw['LONGXY'] = ds_tpqw['LONGXY'].isel(time=1).drop('time')
        ds_prec['LATIXY'] = ds_prec['LATIXY'].isel(time=1).drop('time')    
        ds_solr['LATIXY'] = ds_solr['LATIXY'].isel(time=1).drop('time')
        ds_tpqw['LATIXY'] = ds_tpqw['LATIXY'].isel(time=1).drop('time')
        # replace climate with bc climate
        ds_prec['PRECTmms'] = ds_sub['PRECTmms'] 
        ds_solr['FSDS'] = ds_sub['FSDS'] 
        ds_tpqw['TBOT'] = ds_sub['TBOT'] 
        ds_tpqw['PSRF'] = ds_sub['PSRF'] 
        ds_tpqw['QBOT'] = ds_sub['QBOT'] 
        ds_tpqw['WIND'] = ds_sub['WIND'] 
        ds_tpqw['FLDS'] = ds_sub['FLDS'] 
        # count nana by data array
        fsds_nan = ds_solr['FSDS'].isnull().sum() 
        flds_nan = ds_tpqw['FLDS'].isnull().sum()
        pbot_nan = ds_tpqw['PSRF'].isnull().sum()
        tbot_nan = ds_tpqw['TBOT'].isnull().sum()
        qbot_nan = ds_tpqw['QBOT'].isnull().sum()
        prec_nan = ds_prec['PRECTmms'].isnull().sum()
        wind_nan = ds_tpqw['WIND'].isnull().sum()
        # print statement to debug after data change
        with option_context('display.max_rows', 10, 'display.max_columns', 10):
            with open(Path(config['new_dir']+'/debug_siteclimate.txt'), 'a') as f:
                print('\nBC Precip dataset:\n', file=f)
                print(ds_prec, file=f)
                print('\nBC Radiation dataset:\n', file=f)
                print(ds_solr, file=f)
                print('\nBC Other dataset:\n', file=f)
                print(ds_tpqw, file=f)
                print('\nFSDS nan count:\n', file=f)
                print(fsds_nan, file=f) 
                print('\nFLDS nan count:\n', file=f)
                print(flds_nan, file=f) 
                print('\nPBOT nan count:\n', file=f)
                print(pbot_nan, file=f) 
                print('\nTBOT nan count:\n', file=f)
                print(tbot_nan, file=f) 
                print('\nQBOT nan count:\n', file=f)
                print(qbot_nan, file=f) 
                print('\nPRECIP nan count:\n', file=f)
                print(prec_nan, file=f) 
                print('\nWIND nan count:\n', file=f)
                print(wind_nan, file=f) 
        # save annual output files to new folder with same file names
        prec_file_out = config['new_dir'] + str(Path(file_group[1]).name) 
        solr_file_out = config['new_dir'] + str(Path(file_group[2]).name)
        tpqw_file_out = config['new_dir'] + str(Path(file_group[3]).name)
        ## set netcdf write characteristics for xarray.to_netcdf()
        comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
        try: 
            encoding = {var: comp for var in ds_prec.data_vars}
            ds_prec.to_netcdf(prec_file_out, mode="w", encoding=encoding, \
                    format=config['nc_write']['format'], \
                    engine=config['nc_write']['engine'])
            encoding = {var: comp for var in ds_solr.data_vars}
            ds_solr.to_netcdf(solr_file_out, mode="w", encoding=encoding, \
                    format=config['nc_write']['format'], \
                    engine=config['nc_write']['engine'])
            encoding = {var: comp for var in ds_tpqw.data_vars}
            ds_tpqw.to_netcdf(tpqw_file_out, mode="w", encoding=encoding, \
                    format=config['nc_write']['format'], \
                    engine=config['nc_write']['engine'])
        except Exception as error:
            with open(Path(config['new_dir']+'/debug_siteclimate.txt'), 'a') as f:
                print(error, file=f)
            
# funtion to output pdf report
def plot_site_graphs(input_list):
    # read in config
    config = read_config(input_list[0])
    time_avg = input_list[1]
    # create file names to load obs, cru, A/M bias corrected cru netcdfs, and A/M bias netcdfs
    f_obs = Path(config['site_dir'] + "Obs" + "_" + config['site_name'] + "_dat.nc")
    f_cru = Path(config['site_dir'] + "CRUJRA" + "_" + config['site_name'] + "_allyears.nc")
    f_cru_abc = Path(config['site_dir'] + "ABias" + "_" + config['site_name'] + "_" + time_avg + "_allyears.nc")
    f_cru_mbc = Path(config['site_dir'] + "MBias" + "_" + config['site_name'] + "_" + time_avg + "_allyears.nc")
    f_ab = Path(config['site_dir'] + "ABias" + "_" + config['site_name'] + "_" +  time_avg + "_mym.nc")
    f_mb = Path(config['site_dir'] + "MBias" + "_" + config['site_name'] + "_" +  time_avg + "_mym.nc")
    f_cru_mym = Path(config['site_dir'] + "CRUJRA" + "_" + config['site_name'] + "_" +  time_avg + "_mym.nc")
    f_obs_mym = Path(config['site_dir'] + "Obs" + "_" + config['site_name'] + "_" +  time_avg + "_mym.nc")
    # read all the files into memory using context manager
    with xr.open_dataset(f_cru, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
        ds_cru = ds_tmp.load()
    try:
        with xr.open_dataset(f_obs, engine=config['nc_read']['engine'], decode_cf=True, use_cftime=True) as ds_tmp:
            ds_obs = ds_tmp.load()
    except:
        ds_obs = ds_cru.copy(deep=True)
        pass
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
    # print statment
    with open(Path(config['site_dir'] + '/debug_' + time_avg + '.txt'), 'w') as f:
        print('data for plotting uploaded', file=f)
    # define function to graph
    def graph_time(ds1, ds2, year_start, year_end, xlab, title, title_size, file_dir, bias, scatter,\
                    time_scale, cru_color, obs_color, leg_text, leg_loc, fontsize, dpi, file_end):
        for var in ['FSDS','FLDS','TBOT','PBOT','QBOT','WIND','PRECIP']:
            try:
                # set typical scaling
                add_or_mult = 'none'
                # case match for scaling and yaxis titles
                match var:
                    case 'FSDS':
                        ylab = 'Shortwave (W/$m^{2}$)'
                        file_part = 'shortwave_' + time_avg
                    case 'FLDS':
                        ylab = 'Longwave (W/$m^{2}$)'
                        file_part = 'longwave_' + time_avg
                    case 'TBOT':
                        if bias == False:
                            factor = -273.15
                            add_or_mult = 'add'
                        ylab = 'Air Temperature ($^\circ$C)'
                        file_part = 'air_temperature_' + time_avg
                    case 'PBOT':
                        if bias == False:
                            factor = 0.001
                            add_or_mult = 'multiply'
                        ylab = 'Pressure (kPa)'
                        file_part = 'air_pressure_' + time_avg
                    case 'QBOT':
                        ylab = 'Specific Humidity (kg/kg)'
                        file_part = 'specific_humidity_' + time_avg   
                    case 'WIND':
                        ylab = 'Wind Speed (m/s)'   
                        file_part = 'wind_speed_' + time_avg
                    case 'PRECIP':
                        ylab = 'Total Precipitation (mm/s)'   
                        file_part = 'precipitation_' + time_avg
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
                        with open(Path(config['site_dir'] + '/debug_' + time_avg +'.txt'), 'a') as f:
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
                        with open(Path(config['site_dir'] + '/debug_' + time_avg +'.txt'), 'a') as f:
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
                with open(Path(config['site_dir'] + '/debug_' + time_avg +'.txt'), 'a') as f:
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
    try:
        graph_time(ds_cru, ds_obs, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '') 
    except Exception as error:
        with open(Path(config['site_dir'] + '/debug_' + time_avg + '.txt'), 'a') as f:
            print(error, file=f)
        pass
    # Additive biascorrected cru and observations 
    try:
        title = 'Abc'
        leg_text = ['CRUJRA abc', 'Observations']
        graph_time(ds_cru_abc, ds_obs, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_abc') 
    except Exception as error:
        with open(Path(config['site_dir'] + '/debug_' + time_avg +'.txt'), 'a') as f:
            print(error, file=f)
        pass
    # Multiplicative biascorrected cru and observations 
    try:
        title = 'Mbc'
        leg_text = ['CRUJRA mbc', 'Observations']
        graph_time(ds_cru_mbc, ds_obs, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_mbc') 
    except Exception as error:
        with open(Path(config['site_dir'] + '/debug_' + time_avg + '.txt'), 'a') as f:
            print(error, file=f)
        pass
    ###### climate trend graphs
    try:
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
    except:
        pass
    ##### daily mean plots 
    if time_avg == 'dw':
        x_lab = 'hours per week of year'
    elif time_avg == 'w':
        x_lab = 'week of year'
    elif time_avg == 'm':
        x_lab = 'month of year'
    try:
        time_scale = 'doy'
        title = 'Mym'
        # graph uncorrected cru and observations
        graph_time(ds_cru_mym, ds_obs_mym, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_mym') 
    except:
        pass
    ###### daily additive bias 
    try:
        title = 'ABias'
        leg_text = ['Obs - Cru']
        cru_color = 'tab:green'
        bias = 'add'
        #  graph uncorrected cru and observations
        graph_time(ds_ab, None, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                 time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_abias') 
    except:
        pass
    ###### daily multiplicative bias 
    try:
        title = 'MBias'
        leg_text = ['Obs / Cru']
        cru_color = 'tab:olive'
        bias = 'multiply'
        # graph uncorrected cru and observations
        graph_time(ds_mb, None, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_mbias') 
    except:
        pass
    ###### R squared plots
    try:
        # align cru, obs, abias, and mbias, xarray.datasets through their only dimension - time
        ds_cru_org2, ds_obs_org2, ds_abias_org2, ds_mbias_org2 = xr.align(ds_cru, ds_obs, ds_cru_abc, ds_cru_mbc)
        # dictionary to rename climate variables to obs_var
        original_data_vars = ['FSDS','FLDS','PBOT','PRECIP','RAIN','SNOW','QBOT','TBOT','WIND']
        new_obs_vars = ['obs_FSDS','obs_FLDS','obs_PBOT','obs_PRECIP','obs_RAIN','obs_SNOW','obs_QBOT','obs_TBOT','obs_WIND']
        new_cru_vars = ['cru_FSDS','cru_FLDS','cru_PBOT','cru_PRECIP','cru_RAIN','cru_SNOW','cru_QBOT','cru_TBOT','cru_WIND']
        cru_dict = dict(zip(original_data_vars, new_cru_vars)) 
        obs_dict = dict(zip(original_data_vars, new_obs_vars))
        # subset dict to only variables that exist in observations to rename
        obs_dict = {k: obs_dict[k] for k in ds_obs_org2.data_vars}
        with open(Path(config['site_dir'] + '/debug_' + time_avg + '.txt'), 'a') as f:
            print(obs_dict, file=f)
        # rename datasets for combination
        ds_obs_org2 = ds_obs_org2.rename(obs_dict)
        ds_cru_org2 = ds_cru_org2.rename(cru_dict)
        ds_abias_org2 = ds_abias_org2.rename(cru_dict)
        ds_mbias_org2 = ds_mbias_org2.rename(cru_dict)
        # add obs data to cru datasets
        for var in list(ds_obs_org2.data_vars):
            ds_obs_data = ds_obs_org2[var]
            ds_cru_org2[var] = ds_obs_data
            ds_abias_org2[var] = ds_obs_data
            ds_mbias_org2[var] = ds_obs_data
        # copy original data to new variable names for zero removal
        ds_cru2 = ds_cru_org2.copy(deep=True)
        ds_abias2 = ds_abias_org2.copy(deep=True)
        ds_mbias2 = ds_mbias_org2.copy(deep=True)
        # remove zero rain events for r-square plot and seasonal averages
        #if 'cru_PRECIP' in list(ds_cru2.data_vars):
        #    ds_cru2['cru_PRECIP'].loc[ds_cru2['cru_PRECIP'] <= 0] = np.nan
        #    ds_cru2['obs_PRECIP'].loc[ds_cru2['obs_PRECIP'] <= 0] = np.nan
        #    ds_abias2['cru_PRECIP'].loc[ds_abias2['cru_PRECIP'] <= 0] = np.nan
        #    ds_abias2['obs_PRECIP'].loc[ds_abias2['obs_PRECIP'] <= 0] = np.nan
        #    ds_mbias2['cru_PRECIP'].loc[ds_mbias2['cru_PRECIP'] <= 0] = np.nan
        #    ds_mbias2['obs_PRECIP'].loc[ds_mbias2['obs_PRECIP'] <= 0] = np.nan
        # new grouping coordinate must be integer data type and added to time dimenion
        if time_avg == 'dw':
            main_window = 'weekofyear'
            sub_window = 'hour'
        elif time_avg == 'w':
            main_window = 'weekofyear'
            sub_window = ''
        elif time_avg == 'm':
            main_window = 'month'
            sub_window = ''
        ds_cru3 = ds_cru2.assign_coords(groupvar = ('time', map_groups(ds_cru2.indexes['time'], main_window, sub_window, config)))
        ds_abias3 = ds_abias2.assign_coords(groupvar = ('time', map_groups(ds_abias2.indexes['time'], main_window, sub_window, config)))
        ds_mbias3 = ds_mbias2.assign_coords(groupvar = ('time', map_groups(ds_mbias2.indexes['time'], main_window, sub_window, config)))
        with open(Path(config['site_dir'] + '/debug_' + time_avg + '.txt'), 'a') as f:
            print(ds_cru3, file=f)
            print(ds_abias3, file=f)
            print(ds_mbias3, file=f)
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
        graph_time(ds_cru_org2, ds_abias_org2, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                    time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_abc_rsqr') 
        # create additive plot of uncorrected cru vs obs and corrected cru vs obs to show correction toward 1:1 line
        title = 'Mbc All'
        leg_text = ['Cru vs Obs','Mbc vs Obs']
        graph_time(ds_cru_org2, ds_mbias_org2, config['year_start'], config['year_end'], x_lab, title, title_size, config['site_dir'], bias, scatter, \
                    time_scale, cru_color, obs_color, leg_text, leg_loc, text_size, plot_dpi, '_mbc_rsqr') 
    except Exception as error:
        with open(Path(config['site_dir'] + '/debug_' + time_avg + '.txt'), 'a') as f:
            print(error, file=f)
        pass

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
    driver_list = ['shortwave_dw', 'longwave_dw', 'air_temperature_dw', 'air_pressure_dw', 'specific_humidity_dw', 'wind_speed_dw', 'precipitation_dw']
    # loop through sites
    for site in config_list:
        # get site info from config file
        config = read_config(site)
        site_name = config['site_name']
        with open(Path(config['site_dir'] + 'debug.txt'), 'a') as f:
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
        image_path = config['site_dir'] + 'shortwave_dw.png'
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
        image_path = config['site_dir'] + 'shortwave_dw_mym.png'
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
        image_path = config['site_dir'] + 'shortwave_dw_abc.png'
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
        image_path = config['site_dir'] + 'shortwave_dw_abc.png'
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
        image_path = config['site_dir'] + 'shortwave_dw_abc.png'
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
def regional_dir_prep(config):
    # remake directory per folder 
    Path(config['output_dir'] + 'zarr_output/' + config['model_name']).mkdir(parents=True, exist_ok=True)
    Path(config['output_dir'] + 'zarr_output/' + config['model_name']).chmod(0o762)

# collect file lists per model and simulations based on configuration files
def regional_simulation_files(input_list):
    # read config file
    config = input_list[0]
    # read the simulation type
    sim_type = input_list[1]
    # read all CRUJRA input file names from reanalysis directory
    dir_name = sim_type + '_dir'
    sim_str = sim_type + '_str'
    if config['model_name'] not in ['ecosys','ELM1-ECA','ORCHIDEE-MICT-teb','JULES']:
        sim_files = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], config[sim_str])))
    elif config['model_name'] in ['ELM1-ECA','JULES']:
        sim_files = sorted(glob.glob("{}*{}*.zarr".format(config[dir_name], config[sim_str])))
    elif config['model_name'] in ['ecosys']:
        # deal with non-standard file and variable chunking in ecosys
        # this will eventually need to be updated for other files: water, SOC, etc.
        sim_files1 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'daily_C_flux')))
        sim_files2 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'soil_temp1')))
        sim_files3 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'soil_temp2')))
        sim_files4 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'daily_water')))
        sim_files5 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'SOC1')))
        sim_files6 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'SOC2')))
        sim_files = [sim_files1, sim_files2, sim_files3, sim_files4, sim_files5, sim_files6]
    elif config['model_name'] in ['ORCHIDEE-MICT-teb']:
        sim_files1 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'sechiba')))
        sim_files2 = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], 'stomate')))
        sim_files = [sim_files1, sim_files2]
    # loop through files in reanalysis archive linking config, in, out files
    merged_file = config['output_dir'] + 'zarr_output/' + config['model_name'] + '/WrPMIP_Pan-Arctic_' + config['model_name'] + '_' + sim_type + '.zarr'
    # combine processing info into list
    info_list = [config, sim_type, sim_files, merged_file]
    return info_list 

# rechunk LPJ-GUESS-ML
def list_lpjguessml(input_list):
    # read config file
    config = input_list[0]
    # read the simulation type
    sim_type = input_list[1]
    # read all CRUJRA input file names from reanalysis directory
    dir_name = sim_type + '_dir'
    sim_str = sim_type + '_str'
    sim_dir = Path(config[dir_name]).stem
    # make output dir
    Path(config['output_dir'] + sim_dir).mkdir(parents=True, exist_ok=True)
    Path(config['output_dir'] + sim_dir).chmod(0o762)
    with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_' + sim_type + '.txt'), 'w') as pf:
        print('sim_dir: ', file=pf)
        print(sim_dir, file=pf)
    sim_files = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], config[sim_str])))
    #sim_files = [x for x in sim_files if 'Soil' in x]
    with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_' + sim_type + '.txt'), 'a') as pf:
        print('sim files: ', file=pf)
        print(sim_files, file=pf)
        print('output file list: ', file=pf)
    file_list = []
    for f in sim_files:
        file_name = Path(f).name
        output_file = config['output_dir'] + sim_dir + '/' + file_name
        file_list.append([f, output_file])
    with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_' + sim_type + '.txt'), 'a') as pf:
        print(file_list, file=pf)
    return file_list

# rechunk JULES
def list_jules_daily(input_list):
    # read config file
    config = input_list[0]
    # read the simulation type
    sim_type = input_list[1]
    # read all CRUJRA input file names from reanalysis directory
    dir_name = sim_type + '_dir'
    sim_str = sim_type + '_str'
    sim_dir = Path(config[dir_name]).stem
    # make output dir
    Path(config['output_dir'] + sim_dir).mkdir(parents=True, exist_ok=True)
    Path(config['output_dir'] + sim_dir).chmod(0o762)
    with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_' + sim_type + '.txt'), 'w') as pf:
        print('sim_dir: ', file=pf)
        print(sim_dir, file=pf)
    sim_files = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], config[sim_str])))
    if sim_type in ['b2','otc','sf']:
        # identify unique variable string:
        var_unique = []
        for i in range(len(sim_files)):
            var_unique.append("_".join(str(Path(sim_files[i]).name).split("_")[6:8]))
        var_unique = list(set(var_unique))
        # for each unique variable pull
        file_string = "_".join(str(Path(sim_files[0]).name).split("_")[0:6])
        new_sim_files = []
        for var in var_unique:
            output_file = config['output_dir'] + sim_dir + '/' + file_string + '_' + var + '_daily_2000_2022.zarr'
            files_to_combine = [item for item in sim_files if var in item]
            file_subset = [item for item in sim_files if var not in item]
            if var == 'SoilTemp_nhlat':
                files_to_combine = [item for item in files_to_combine if 'top5cm' not in item]
            if var in ['ALT_sthf','Fdepth_sthf']:
                continue
            new_sim_files.append([files_to_combine, output_file, sim_type])
    else:
        new_sim_files = []
    #sim_files = [x for x in sim_files if 'Soil' in x]
    with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_' + sim_type + '.txt'), 'a') as pf:
        print('sim files: ', file=pf)
        print(new_sim_files, file=pf)
    return new_sim_files

# rechunk JULES
def list_jules_monthly(input_list):
    # read config file
    config = input_list[0]
    # read the simulation type
    sim_type = input_list[1]
    # read all CRUJRA input file names from reanalysis directory
    dir_name = sim_type + '_dir'
    sim_str = sim_type + '_str'
    sim_dir = Path(config[dir_name]).stem
    # make output dir
    Path(config['output_dir'] + sim_dir).mkdir(parents=True, exist_ok=True)
    Path(config['output_dir'] + sim_dir).chmod(0o762)
    with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_' + sim_type + '_monthly.txt'), 'w') as pf:
        print('sim_dir: ', file=pf)
        print(sim_dir, file=pf)
    monthly_files = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], '_monthly_')))
    #sim_files = [x for x in sim_files if 'Soil' in x]
    file_list = []
    remove_list = ['WTD_nhlat_monthly','TVeg_nhlat_monthly','TotalResp_nhlat_monthly','SWE_nhlat_monthly', \
                   'SoilTemp_nhlat_monthly','SoilMoist_nhlat_monthly','SoilIce_nhlat_monthly','SnowDepth_nhlat_monthly', \
                   'Qs_nhlat_monthly','Qsb_nhlat_monthly','Qle_nhlat_monthly','Qh_nhlat_monthly','NPP_nhlat_monthly','Nmineral_nhlat_monthly',\
                   'NEP_nhlat_monthly','NEE_nhlat_monthly','LAI_nhlat_monthly','HeteroResp_vr_nhlat_monthly', \
                   'GPP_nhlat_monthly','fPAR_nhlat_monthly','Fdepth_sthf_nhlat_monthly','Fdepth_nhlat_monthly', \
                   'CLitter_nhlat_monthly','AutoResp_nhlat_monthly','ALT_sthf_nhlat_monthly','ALT_nhlat_monthly']
    for f in monthly_files:
        file_name = Path(f).stem
        if sim_type in ['b2','otc','sf']:
            output_file = config['output_dir'] + sim_dir + '/' + file_name.replace('monthly','daily') + '.zarr'
            if any(item in f for item in remove_list):
                continue
        else:
            output_file = config['output_dir'] + sim_dir + '/' + file_name + '.zarr'
            if any(item in f for item in ['ALT_sthf','Fdepth_sthf']):
                continue
        file_list.append([f, output_file, sim_type])
    with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_' + sim_type + '_monthly.txt'), 'a') as pf:
        print(file_list, file=pf)
    return file_list

def rechunk_jules_daily(input_list, config):
    input_files = input_list[0]
    output_file = input_list[1]
    sim_type = input_list[2]
    # open and combine files
    def preprocess_jules_rechunk(ds, config):
        ds = xr.decode_cf(ds, use_cftime=True)
        if 'bnds' in ds.dims.keys():
            ds = ds.drop_dims('bnds')
        if 'month' in ds.coords:
            ds = ds.reset_coords('month', drop=True)
        ds = ds.sortby(['time'])
        with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_rechunk_daily.txt'), 'a') as pf:
            print(ds, file=pf)
        return ds
    partial_jules_rechunk = partial(preprocess_jules_rechunk, config=config)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        with xr.open_mfdataset(input_files, parallel=True, engine=config['nc_read']['engine'], chunks={'time': 73}, preprocess=partial_jules_rechunk, mask_and_scale=True, decode_times=True) as ds:
            with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_rechunk_daily.txt'), 'a') as pf:
                print('duplicated times: ', file=pf)
                print(ds.data_vars, file=pf)
                print(sum(ds.time.to_index().duplicated()), file=pf)
            if sum(ds.time.to_index().duplicated()) > 0:
                ds = ds.drop_duplicates('time')
            # change calendar to no_leap
            ds = ds.convert_calendar('noleap', use_cftime=True)
            ds['time'] = [t - xr.coding.cftime_offsets.Hour(12) for t in ds['time'].values]
            ds = ds.reindex({'time': xr.cftime_range(start='2000-01-01 00:00:00', end='2021-12-31 00:00:00', freq='D', calendar='noleap')})
            with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_rechunk_daily.txt'), 'a') as pf:
                print(ds, file=pf)
            # output to zarr
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
            if 'depth' in ds.dims.keys():
                ds = ds.rename({'depth': 'SoilDepth'})
                chunk_dict = {'time': 73, 'SoilDepth': 20, 'lat': 62, 'lon': 720}
                ds = ds.chunk(chunk_dict)
            elif 'sclayer' in ds.dims.keys():
                ds = ds.rename({'sclayer': 'SoilDepth'})
                chunk_dict = {'time': 73, 'SoilDepth': 20, 'lat': 62, 'lon': 720}
                ds = ds.chunk(chunk_dict)
            else:
                chunk_dict = {'time': 73, 'lat': 62, 'lon': 720}
                ds = ds.chunk(chunk_dict)
            ds = ds.transpose('time', 'SoilDepth', 'lat', 'lon', missing_dims='ignore')
            if 'SoilDepth' in ds.dims.keys():
                ds = ds.assign_coords({'SoilDepth': config['soil_depths']})
            compress = Zstd(level=6)
            encode = {var: {'_FillValue': np.nan, 'compressor': compress} for var in ds.data_vars}
            ds.to_zarr(output_file, encoding=encode, mode="w")
            with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_rechunk_daily.txt'), 'a') as pf:
                print(ds.data_vars, file=pf)
                print('Finished with above dataset', file=pf)

def rechunk_jules_monthly(input_list, config):
    input_file = input_list[0]
    output_file = input_list[1]
    sim_type = input_list[2]
    # open and combine files
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        with xr.open_dataset(input_file, engine=config['nc_read']['engine'], chunks={'time': 12}, mask_and_scale=True, decode_times=True) as ds:
            # change calendar to no_leap
            ds = ds.convert_calendar('noleap', use_cftime=True)
            ds['time'] = [t - xr.coding.cftime_offsets.Hour(12) for t in ds['time'].values]
            # sum Cpools
            if 'scpool' in ds.dims.keys():
                ds = ds.sum(dim='scpool',skipna=True)
            # remove extraneous dimensions/coords
            if 'bnds' in ds.dims.keys():
                ds = ds.drop_dims('bnds')
            if 'month' in ds.coords:
                ds = ds.reset_coords('month', drop=True)
            # for all but b1 change to daily data
            if sim_type in ['b2','otc','sf']:
                ds = ds.resample(time='1D').ffill()
                ds = ds.reindex({'time': xr.cftime_range(start='2000-01-01 00:00:00', end='2021-12-31 00:00:00', freq='D', calendar='noleap')})
            with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_rechunk_monthly.txt'), 'a') as pf:
                print(ds, file=pf)
            # output to zarr
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
            if 'depth' in ds.dims.keys():
                ds = ds.rename({'depth': 'SoilDepth'})
                chunk_dict = {'time': 73, 'SoilDepth': 20, 'lat': 62, 'lon': 720}
                ds = ds.chunk(chunk_dict)
            elif 'sclayer' in ds.dims.keys():
                ds = ds.rename({'sclayer': 'SoilDepth'})
                chunk_dict = {'time': 73, 'SoilDepth': 20, 'lat': 62, 'lon': 720}
                ds = ds.chunk(chunk_dict)
            else:
                chunk_dict = {'time': 73, 'lat': 62, 'lon': 720}
                ds = ds.chunk(chunk_dict)
            ds = ds.transpose('time', 'SoilDepth', 'lat', 'lon', missing_dims='ignore')
            if 'SoilDepth' in ds.dims.keys():
                ds = ds.assign_coords({'SoilDepth': config['soil_depths']})
            compress = Zstd(level=6)
            encode = {var: {'_FillValue': np.nan, 'compressor': compress} for var in ds.data_vars}
            ds.to_zarr(output_file, encoding=encode, mode="w")
            with open(Path(config['output_dir'] + 'debug_' + config['model_name'] + '_rechunk_monthly.txt'), 'a') as pf:
                print(ds.data_vars, file=pf)
                print('Finished with above dataset', file=pf)

def rechunk_elmeca(input_list):
    # read config file
    config = input_list[0]
    # read the simulation type
    sim_type = input_list[1]
    # read all CRUJRA input file names from reanalysis directory
    dir_name = sim_type + '_dir'
    sim_str = sim_type + '_str'
    # make output dir
    Path(config['output_dir'] + sim_type).mkdir(parents=True, exist_ok=True)
    Path(config['output_dir'] + sim_type).chmod(0o762)
    sim_files = sorted(glob.glob("{}*{}*.nc".format(config[dir_name], config[sim_str])))
    #sim_files = [x for x in sim_files if 'Soil' in x]
    with open(Path(config['output_dir'] + 'debug_elmeca.txt'), 'w') as pf:
        print('LPJ-GUESS-ML rechunking...', file=pf)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # function to chop up file input list
        def chunk_gen(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        if sim_type == 'b1':
            chunk_read = config['nc_read']['b1_chunks']
            zarr_chunk_out = 73
        else:
            chunk_read = config['nc_read']['b2_chunks']
            zarr_chunk_out = 73
        # assign engine used to open netcdf files from config
        engine = config['nc_read']['engine']
        # set kwargs to mask_and_scale=True and decode_times=False for individual files passed to open_mfdataset
        kwargs = {"mask_and_scale": True, "decode_times": False}
        # create functools.partial function to pass subset variable through to preprocess
        partial_time = partial(preprocess_time, config=config) 
        # loop through file chunks and output to zarr to fix issues
        year_iterator = 1
        # loop through and open chunked file list, 0 to 6204 is from 2000-01-01 to 2016-12-31 noleap
        # starting from 2017-01-01 time index was restarted for unknown reason in files
        for file_chunk in chunk_gen(sim_files, 6205):
            with xr.open_mfdataset(file_chunk, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_time, **kwargs) as ds_tmp:
                with open(Path(config['output_dir'] + 'debug_elmeca.txt'), 'a') as pf:
                    print(ds_tmp, file=pf)
                ######## deal with levdcmp issue
                # identify variables with problematic dim
                problem_dim = 'levdcmp'
                matching_variables = []
                for var_name, data_array in ds_tmp.data_vars.items():
                    if problem_dim in data_array.dims:
                        matching_variables.append(var_name)
                with open(Path(config['output_dir'] + 'debug_elmeca.txt'), 'a') as pf:
                    print("problem variables:", file=pf)
                    print(matching_variables, file=pf)
                # subset out these variables
                ds_problem_vars = ds_tmp[matching_variables]
                with open(Path(config['output_dir'] + 'debug_elmeca.txt'), 'a') as pf:
                    print("problem ds:", file=pf)
                    print(ds_problem_vars, file=pf)
                # remove problem variables from dataset
                ds_tmp = ds_tmp.drop_dims(problem_dim)
                with open(Path(config['output_dir'] + 'debug_elmeca.txt'), 'a') as pf:
                    print("cleaned ds:", file=pf)
                    print(ds_tmp, file=pf)
                # fix dimension name of problem variables (currently a dataset b/c SOIL[1N,2N,3N]_vr variables)
                ds_problem_vars = ds_problem_vars.rename({"levdcmp":"levgrnd"})
                with open(Path(config['output_dir'] + 'debug_elmeca.txt'), 'a') as pf:
                    print("changed levdcmp to levgrnd ds:", file=pf)
                    print(ds_problem_vars, file=pf)
                ds_problem_vars = ds_problem_vars.assign_coords({"levgrnd": ds_tmp.coords['levgrnd'].values})
                with open(Path(config['output_dir'] + 'debug_elmeca.txt'), 'a') as pf:
                    print("replaced levdcmp values with lebgrnd values ds:", file=pf)
                    print(ds_problem_vars, file=pf)
                # add variables back to dataset
                ds_tmp = ds_tmp.merge(ds_problem_vars)
                with open(Path(config['output_dir'] + 'debug_elmeca.txt'), 'a') as pf:
                    print("merged ds:", file=pf)
                    print(ds_tmp, file=pf)
                ########
                # create file name for output
                file_out = Path(config['output_dir']+ sim_type +'/elm_eca_' + sim_type + '_' + str(year_iterator).zfill(2)+'.zarr')
                # set zarr compression and encoding
                compress = Zstd(level=6)
                # clear all chunk and fill value encoding/attrs
                for var in ds_tmp:
                    try:
                        del ds_tmp[var].encoding['chunks']
                    except:
                        pass
                    try:
                        del ds_tmp[var].encoding['_FillValue']
                    except:
                        pass
                    try:
                        del ds_tmp[var].attrs['_FillValue']
                    except:
                        pass
                dim_chunks = {
                    'time': zarr_chunk_out,
                    'levgrnd': -1,
                    'lat': -1,
                    'lon': -1}
                ds_tmp = ds_tmp.chunk(dim_chunks)
                # if after 2017 add 6205 to time value to fix
                if year_iterator > 1:
                    ds_tmp = ds_tmp.assign_coords({'time': ds_tmp.time + 6205}) 
                # encode and output to file 
                encode = {var: {'_FillValue': np.nan, 'compressor': compress} for var in ds_tmp.data_vars}
                ds_tmp.to_zarr(file_out, encoding=encode, mode="w")
            # iterate to next year for file names
            year_iterator += 1

def elm_zstore_list(dir_in, dir_out):
    zarr_stores = sorted(glob.glob("{}*.zarr".format(dir_in)))
    zarr_list = []
    for zstore_in in zarr_stores:
        f_parts = str(Path(zstore_in).stem).split('_')
        zstore_out = dir_out + '/' + '_'.join([f_parts[0],f_parts[1],'b1',f_parts[3]]) + '.zarr'
        zarr_list.append([zstore_in,zstore_out])
    return zarr_list

def zstore_copy(input_list):
    z_in = input_list[0]
    z_out = input_list[1]
    output_dir = Path(z_out).parent
    if not Path(output_dir).is_dir():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).chmod(0o762)
    z1 = zarr.DirectoryStore(z_in)
    z2 = zarr.DirectoryStore(z_out)
    zarr.copy_store(z1, z2)
    
def rechunk_lpjguessml(input_list, config):
    input_file = input_list[0]
    output_file = input_list[1]
    with open(Path(config['output_dir'] + 'debug_lpjguess.txt'), 'w') as pf:
        print('LPJ-GUESS-ML rechunking...', file=pf)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds = xr.open_dataset(input_file, engine=config['nc_read']['engine'], mask_and_scale=True, decode_times=False)
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        if 'month' in input_file:
            if 'nsoil' in ds.dims.keys():
                ds = ds.rename({'nsoil': 'SoilDepth'})
                chunk_dict = {'time': 73, 'SoilDepth': 15, 'lat': 120, 'lon': 720}
                ds = ds.chunk(chunk_dict)
            else:
                chunk_dict = {'time': 73, 'lat': 120, 'lon': 720}
                ds = ds.chunk(chunk_dict)
        else:
            if 'nsoil' in ds.dims.keys():
                ds = ds.rename({'nsoil': 'SoilDepth'})
                chunk_dict = {'time': 73, 'SoilDepth': 15, 'lat': 120, 'lon': 720}
                ds = ds.chunk(chunk_dict)
            else:
                chunk_dict = {'time': 73, 'lat': 120, 'lon': 720}
                ds = ds.chunk(chunk_dict)
        ds = ds.transpose('time', 'SoilDepth', 'lat', 'lon', missing_dims='ignore')
        with open(Path(config['output_dir'] + 'debug_lpjguess.txt'), 'a') as pf:
            print(ds, file=pf)
            print(ds.encoding, file=pf)
        # clear encoding chunks
        with open(Path(config['output_dir'] + 'debug_lpjguess.txt'), 'a') as pf:
            print(ds.encoding, file=pf)
        # output netcdf
        comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'], _FillValue=np.nan, chunksizes=list(chunk_dict.values()))
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(output_file, mode="w", encoding=encoding, \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
        # try to close file connection
        try:
            ds.close()
        except:
            pass
        # try to delete data
        try:
            del ds
        except:
            pass

def rechunk_lpjguess(input_list, config):
    input_file = input_list[0]
    output_file = input_list[1]
    with open(Path(config['output_dir'] + 'debug_lpjguess.txt'), 'w') as pf:
        print('LPJ-GUESS rechunking...', file=pf)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds = xr.open_dataset(input_file, engine=config['nc_read']['engine'], mask_and_scale=True, decode_times=False)
        #ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        if 'month' in input_file:
            if 'nsoil' in ds.dims.keys():
                ds = ds.rename({'nsoil': 'SoilDepth'})
                chunk_dict = {'time': 73, 'SoilDepth': 15, 'lat': 108, 'lon': 675}
                ds = ds.chunk(chunk_dict)
            elif 'nsnow' in ds.dims.keys():
                chunk_dict = {'time': 73, 'lat': 108, 'lon': 675, 'nsnow': 5}
                ds = ds.chunk(chunk_dict)
            elif 'nsompool' in ds.dims.keys():
                chunk_dict = {'time': 73, 'lat': 108, 'lon': 675, 'nsompool': 11}
                ds = ds.chunk(chunk_dict)
            else:
                chunk_dict = {'time': 73, 'lat': 108, 'lon': 675}
                ds = ds.chunk(chunk_dict)
        elif 'year' in input_file:
            if 'nsoil' in ds.dims.keys():
                ds = ds.rename({'nsoil': 'SoilDepth'})
                chunk_dict = {'time': 12, 'SoilDepth': 15, 'lat': 108, 'lon': 675}
                ds = ds.chunk(chunk_dict)
            elif 'nsnow' in ds.dims.keys():
                chunk_dict = {'time': 12, 'lat': 108, 'lon': 675, 'nsnow': 5}
                ds = ds.chunk(chunk_dict)
            elif 'nsompool' in ds.dims.keys():
                chunk_dict = {'time': 12, 'lat': 108, 'lon': 675, 'nsompool': 11}
                ds = ds.chunk(chunk_dict)
            else:
                chunk_dict = {'time': 12, 'lat': 108, 'lon': 675}
                ds = ds.chunk(chunk_dict)
        else:
            if 'nsoil' in ds.dims.keys():
                ds = ds.rename({'nsoil': 'SoilDepth'})
                chunk_dict = {'time': 73, 'SoilDepth': 15, 'lat': 108, 'lon': 675}
                ds = ds.chunk(chunk_dict)
            elif 'nsnow' in ds.dims.keys():
                chunk_dict = {'time': 73, 'lat': 108, 'lon': 675, 'nsnow': 5}
                ds = ds.chunk(chunk_dict)
            elif 'nsompool' in ds.dims.keys():
                chunk_dict = {'time': 73, 'lat': 108, 'lon': 675, 'nsompool': 11}
                ds = ds.chunk(chunk_dict)
            else:
                chunk_dict = {'time': 73, 'lat': 108, 'lon': 675}
                ds = ds.chunk(chunk_dict)
        ds = ds.transpose('time', 'SoilDepth', 'lat', 'lon', 'nsnow', 'nsompool', missing_dims='ignore')
        with open(Path(config['output_dir'] + 'debug_lpjguess.txt'), 'a') as pf:
            print(ds, file=pf)
            print(ds.encoding, file=pf)
        # clear encoding chunks
        with open(Path(config['output_dir'] + 'debug_lpjguess.txt'), 'a') as pf:
            print(ds.encoding, file=pf)
        # output netcdf
        comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'], _FillValue=np.nan, chunksizes=list(chunk_dict.values()))
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(output_file, mode="w", encoding=encoding, \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
        # try to close file connection
        try:
            ds.close()
        except:
            pass
        # try to delete data
        try:
            del ds
        except:
            pass
    
# preprocess files that are parsed by variable
def preprocess_JULES(ds, config, sim_type):
    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
        print('opening' + ds.encoding['source'], file=pf)
        print(ds, file=pf)
    return ds

# function to fix timesteps in preprocess step and subset to variables of interest
def preprocess_JSBACH(ds, config, sim_type):
    dsc = ds.copy(deep=True)
    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
        print('opening' + dsc.encoding['source'], file=pf)
        print(dsc, file=pf)
    # replace time indexes, decimal numbers not allowed in cftime, also rh ends with fillvalues
    # generally a dimension shouldnt have fill values it should simply be missing
    dsc.time.attrs['units'] = 'days since 1998-01-01 00:00:00'
    if dsc.sizes['time'] > 1000:
        # if index is daily then replace with integers (this fixes decimal issue and fill value issue
        dsc = dsc.assign_coords({"time": np.arange(730,8765+1).astype(int)})
        dsc.time.attrs['standard_name'] = 'time'
        dsc.time.attrs['units'] = 'days since 1998-01-01 00:00:00'
        dsc.time.attrs['calendar'] = 'proleptic_gregorian'
        dsc.time.attrs['axis'] = 'T'
        u, c = np.unique(dsc.time.values, return_counts=True)
        dup = u[c > 1] 
    else:
        dsc = xr.decode_cf(dsc, use_cftime=True)
        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
            print('ds after decode cf:', file=pf)
            print(dsc, file=pf)
        new_index = xr.cftime_range(start='2000-01-01', periods=8036, calendar='proleptic_gregorian', freq='1D')
        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
            print('new index:', file=pf)
            print(new_index, file=pf)
        dsc = dsc.reindex({'time': new_index})
        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
            print('ds after reindex:', file=pf)
            print(dsc, file=pf)
        dsc = dsc.resample(time='1D').ffill() #interpolate('linear')
        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
            print('ds after resample:', file=pf)
            print(dsc, file=pf)
        dsc = dsc.assign_coords({'time': np.arange(730,8765+1)})
        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
            print('ds after replacing index with integers:', file=pf)
            print(dsc, file=pf)
        dsc.time.attrs['standard_name'] = 'time'
        dsc.time.attrs['units'] = 'days since 1998-01-01 00:00:00'
        dsc.time.attrs['calendar'] = 'proleptic_gregorian'
        dsc.time.attrs['axis'] = 'T'
        u, c = np.unique(dsc.time.values, return_counts=True)
        dup = u[c > 1] 
    # reverse lat index
    dsc = dsc.sortby('lat', ascending=True) #reindex(lat=ds.lat[::-1])
    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
        print('proprocess complete:', file=pf)
        print(dsc, file=pf)
        print('duplicate times:', file=pf)
        print(dup, file=pf)
        print('time:', file=pf)
        print(dsc.time, file=pf)
    return dsc

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
    # reorder lat
    dsc = dsc.sortby('lat', ascending=True)
    # change to noleap calendar to remove leapdays
    dsc = dsc.convert_calendar('noleap', use_cftime=True)
    # return preprocessed file to open_mfdataset
    return dsc

def preprocess_ORCHIDEE_MICT(ds, config, sim_type):
    # copy the dataset
    dsc = ds.copy(deep=True)
    # sort latitude to monotonic increasing values
    dsc = dsc.sortby('latitude')
    # read file name to find year, only for b2/otc/sf
    if sim_type in ['b2','otc','sf']:
        file_name = dsc.encoding['source']
        year = file_name.split('.')[0].split('_')[-1]
        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
            print(file_name, file=pf)
            print(year, file=pf)
        # create cftime string for year
        start_string = year + '-01-01'
        new_index = xr.cftime_range(start=start_string, periods=365, calendar='noleap', freq='1D')
        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
            print('new index: ', file=pf)
            print(new_index, file=pf)
        # replace 1-365 days in time coord with cftimeindex
        dsc = dsc.assign_coords({'time': new_index})
        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
            print('updated dataset: ', file=pf)
            print(dsc, file=pf)
    # subset
    var_keep = config['subset_vars']
    return dsc[var_keep]

def preprocess_ORCHIDEE_MICT_teb(ds, config, sim_type):
    # copy the dataset
    dsc = ds.copy(deep=True)
    # subset variables / rename sechiba/stomate file sets
    if 'sechiba' in dsc.encoding['source']:
        # rename soilth to SoilDepth
        dsc = dsc.rename({'solth':'SoilDepth', 'time_counter':'time'})
        # subset sechiba
        dsc = dsc[config['subset_sechiba']]
        # take mean soil temp across PFTs
        dsc['ptn'] = dsc['ptn'].mean(dim='veget', keep_attrs=True, skipna=True)
    elif 'stomate' in dsc.encoding['source']:
        # rename ndeep to SoilDepth
        try:
            dsc = dsc.rename({'ndeep':'SoilDepth', 'time_counter':'time'})
        except Exception as error:
            with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                print('Error while trying to rename ndeep in stomate file:', file=pf)
                print(error, file=pf)
                print('file name:', file=pf)
                print(dsc.encoding['source'], file=pf)
        # subset stomate
        dsc = dsc[config['subset_stomate']]
        # take mean of ALT depth across PFTs
        dsc['alt'] = dsc['alt'].mean(dim='veget', keep_attrs=True, skipna=True)
        # remove chunking
        dsc = dsc.chunk({'time': -1})
    # assign same soil depths as other ORCHIDEE-MICT outputs, stomate has integers, sechiba has slightly diff values from other submission
    dsc = dsc.assign_coords({'SoilDepth': config['soil_depths']})
    # take sum of everything else (e.g. fluxes, pools)
    dsc = dsc.sum(dim='veget', keep_attrs=True, skipna=True)
    # change time from seconds since 2002-01-01 00:00:00 to cftime
    dsc = xr.decode_cf(dsc, use_cftime=True)
    if 'time_centered' in ds.coords:
        dsc = dsc.drop_vars(['time_centered'])
    # forward fill month start values to all days in month
    #dsc = dsc.resample(time='1D').ffill() 
    # print updated data to file for debugging
    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
        print('updated dataset: ', file=pf)
        print(dsc, file=pf)
    return dsc

# preprocess files that are parsed by variable
def preprocess_CLASSIC(ds, config, sim_type):
    dsc = ds.copy(deep=True)
    # check for TotalResp file to fix mislabeled variable (should be TotalResp but is HeteroResp in file)
    if 'TotalResp' in dsc.encoding['source']:
        dsc['TotalResp'] = dsc['HeteroResp']
        dsc = dsc.drop_vars(['HeteroResp'])
    # check for NEE file to fix mislabeled variable (should be NEE but is HeteroResp in file)
    if 'NEE' in dsc.encoding['source']:
        dsc['NEE'] = dsc['HeteroResp']
        dsc = dsc.drop_vars(['HeteroResp'])
    # print dataset info
    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
        print(dsc, file=pf)
    # return only subset variables of interest from config file
    return dsc

# preprocess files that are parsed by variable
def preprocess_LPJ_GUESS_ML(ds, config, sim_type):
    dsc = ds.copy(deep=True)
    # check for TotalResp file to fix mislabeled variable (should be TotalResp but is HeteroResp in file)
    if 'NSOM' in dsc.encoding['source']:
        for var in dsc.data_vars:
            var_rename = 'N_' + var
            dsc[var_rename] = dsc[var]
            dsc = dsc.drop_vars([var])
    # check for NEE file to fix mislabeled variable (should be NEE but is HeteroResp in file)
    if 'CSOM' in dsc.encoding['source']:
        for var in dsc.data_vars:
            var_rename = 'C_' + var
            dsc[var_rename] = dsc[var]
            dsc = dsc.drop_vars([var])
    # print dataset info
    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
        print(dsc, file=pf)
        print('encoding for fillvalue', file=pf)
        for var in dsc.data_vars:
            print('variable: ' + var, file=pf)
            print(dsc[var].encoding['_FillValue'], file=pf)
    # return only subset variables of interest from config file
    return dsc

def preprocess_LPJ_GUESS(ds, config, sim_type):
    dsc = ds.copy(deep=True)
    # check for snowice and simulation 1d to correct nsnow values
    if ('snowice' in dsc.encoding['source']) and ('1d' in dsc.encoding['source']):
        dsc = dsc.assign_coords({'nsnow': np.arange(0,5)})
    if len(dsc.lat.values) > 108:
        dsc = dsc.drop_sel(lat = 67.265625)
    # print file info for debugging
    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
        print(dsc, file=pf)
        with xr.set_options(display_max_rows=200, display_style="text"): 
            print('lat:', file=pf)
            print(dsc.lat.values, file=pf)
        print('time:', file=pf)
        print(dsc.time.values, file=pf)
        print('encoding for fillvalue', file=pf)
        for var in dsc.data_vars:
            print('variable: ' + var, file=pf)
            print(dsc[var].encoding, file=pf)
            print(dsc[var].attrs, file=pf)
    return dsc

# preprocess files that are parsed by variable
def preprocess_var(ds, config):
    # return only subset variables of interest from config file
    return ds

# preprocess files that are parsed by time
def preprocess_time(ds, config):
    var_keep = config['subset_vars']
    return ds[var_keep]

# merge files depending on simulation outputs
def process_simulation_files(input_list, top_config):
    # read in config, input files, and output file
    config = input_list[0]
    sim_type = input_list[1]
    sim_files = input_list[2]
    out_file = input_list[3]
    # check if simulation dataset exists for model
    data_check = 'has_' + sim_type
    # context manager to keep dask from auto-rechunking
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # check if dataset exists
        if config[data_check] == "True":
            with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'w') as pf:
                print(input_list, file=pf)
            if sim_type == 'b1':
                chunk_read = config['nc_read']['b1_chunks']
                zarr_chunk_out = 73
            else:
                chunk_read = config['nc_read']['b2_chunks']
                zarr_chunk_out = 73
            #if config['model_name'] in ['ORCHIDEE-MICT-teb']:
            #    chunk_read = {'time': 12}
            #    zarr_chunk_out = 12
            # assign engine used to open netcdf files from config
            engine = config['nc_read']['engine']
            # set kwargs to mask_and_scale=True and decode_times=False for individual files passed to open_mfdataset
            kwargs = {"mask_and_scale": True, "decode_times": False}
            zarr_kwargs = dict(decode_cf=True, mask_and_scale=True, decode_times=False)
            # match the combination type with how files should be merged
            match config['merge_type']:
                case 'time':
                    # create functools.partial function to pass subset variable through to preprocess
                    partial_time = partial(preprocess_time, config=config) 
                    if config['model_name'] == 'UVic-ESCM':
                        # open using mfdataset which will auto merge variables, but preprocess away incorrect time indexes 
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_time, **kwargs)
                        # # calculate auto_resp
                        ds['AutoResp'] = ds['L_veggpp'] - ds['L_vegnpp']
                        # sum pfts to remove pft dimension
                        ds = ds.sum(dim='pft', keep_attrs=True, skipna=True)
                        # calculate TotalResp as auto + hetero resp
                        ds['TotalResp'] = ds['AutoResp'] + ds['L_soilresp']
                        # mask Uvic
                        ds = ds.where(ds.G_mskt < 1)
                        # drop vars/mask
                        ds = ds.drop_vars(['L_vegnpp','G_mskt'])
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print(ds, file=pf)
                    elif config['model_name'] == 'ELM1-ECA':
                        # open zarr files
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine="zarr", preprocess=partial_time, **zarr_kwargs)
                        # check files
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print(ds, file=pf)
                        # join multiple vertically resolved soil pools
                        ds['SoilN_Layers'] = ds['SOIL1N_vr'] + ds['SOIL2N_vr'] + ds['SOIL3N_vr']
                        # drop vr soiln variables
                        ds = ds.drop_vars(['SOIL1N_vr','SOIL2N_vr','SOIL3N_vr'])
                    elif config['model_name'] == 'ORCHIDEE-MICT':
                        partial_ORCHIDEE_MICT = partial(preprocess_ORCHIDEE_MICT, config=config, sim_type=sim_type)
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ORCHIDEE_MICT, **kwargs)
                        # combine vertically resolved soil pool variables
                        ds['SoilC_Layers'] = ds['deepC_a'] + ds['deepC_s'] + ds['deepC_p']
                        # drop individual pools
                        ds = ds.drop_vars(['deepC_a','deepC_s','deepC_p'])
                    elif config['model_name'] == 'ORCHIDEE-MICT-teb':
                        partial_ORCHIDEE_MICT_teb = partial(preprocess_ORCHIDEE_MICT_teb, config=config, sim_type=sim_type)
                        ds1 = xr.open_mfdataset(sim_files[0], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ORCHIDEE_MICT_teb, **kwargs)
                        ds2 = xr.open_mfdataset(sim_files[1], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ORCHIDEE_MICT_teb, **kwargs)
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print('open_mfdatasets completed', file=pf)
                            print(ds1, file=pf)
                            print(ds2, file=pf)
                        # combine datasets
                        ds = ds1.merge(ds2)
                        # calculate TotalResp
                        ds['TotalResp'] = ds['rh'] + ds['ra']
                        # combine vertically resolved soil pool variables
                        ds['SoilC_Layers'] = ds['deepC_a'] + ds['deepC_s'] + ds['deepC_p']
                        # drop individual pools
                        ds = ds.drop_vars(['deepC_a','deepC_s','deepC_p'])
                        # debug dataset
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print('combined dataset: ', file=pf)
                            print(ds, file=pf)
                    elif config['model_name'] == 'CLM5':
                        # open using mfdataset and merge using combine_by_coords
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_time, **kwargs)
                        # output landmask for 
                        output_land_vars = ['landmask','pftmask','nbedrock','area','landfrac']
                        if sim_type == 'b2':    
                            clm5_landmask = ds[output_land_vars]
                            clm5_land_file = config['output_dir']+'zarr_output/clm5_landmask.nc'
                            clm5_landmask.to_netcdf(clm5_land_file, mode="w", format=config['nc_write']['format'], engine=config['nc_write']['engine'])
                        # drop land mask and old dims
                        ds = ds.drop_vars(output_land_vars)
                        # pull each var with levsoi, reindex/rename to levgrnd values, assign back to dasaset
                        for var in ds.data_vars:
                            if 'levsoi' in ds[var].dims:
                                tmp_var = ds[var].copy(deep=True)
                                tmp_var = tmp_var.reindex({'levsoi': ds.levgrnd.values})
                                tmp_var = tmp_var.rename({'levsoi': 'levgrnd'})
                                ds[var] = tmp_var
                        ds = ds.drop_dims(['levsoi'])
                    elif config['model_name'] == 'CLM5-ExIce':
                        # open using mfdataset and merge using combine_by_coords
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_time, **kwargs)
                        # pull each var with levsoi, reindex/rename to levgrnd values, assign back to dasaset
                        for var in ds.data_vars:
                            if 'levsoi' in ds[var].dims:
                                tmp_var = ds[var].copy(deep=True)
                                tmp_var = tmp_var.reindex({'levsoi': ds.levgrnd.values})
                                tmp_var = tmp_var.rename({'levsoi': 'levgrnd'})
                                ds[var] = tmp_var
                        # drop land mask
                        ds = ds.drop_dims(['levsoi'])
                    else:
                        # open using mfdataset and merge using combine_by_coords
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_time, **kwargs)
                case 'variables':
                    if config['model_name'] == 'JULES':
                        # create functools.partial function to pass subset variable through to preprocess
                        partial_JULES = partial(preprocess_JULES, config=config, sim_type=sim_type) 
                        # open using mfdataset which will auto merge variables, but preprocess away incorrect time indexes 
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine="zarr", chunks='auto', preprocess=partial_JULES, **zarr_kwargs)
                        # # subset to variables of interest
                        if sim_type in ['b1']:
                            subset_vars = [item for item in config['subset_vars'] if item not in ['SoilMoist','TotSoilN','SoilC_vr','SoilN_vr','CN']]
                        else:
                            subset_vars = config['subset_vars']
                        if sim_type in ['b2','otc']: 
                            ds['NEE'] = ds['NEE'].sum(dim='SoilDepth')
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print('open_mfdataset completed', file=pf)
                            print(ds, file=pf)
                        ds = ds[subset_vars]
                    elif config['model_name'] == 'JSBACH':
                        # create functools.partial function to pass subset variable through to preprocess
                        partial_JSBACH = partial(preprocess_JSBACH, config=config, sim_type=sim_type) 
                        # open using mfdataset which will auto merge variables, but preprocess away incorrect time indexes 
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_JSBACH, **kwargs)
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print('open_mfdataset completed', file=pf)
                            print(ds, file=pf)
                        # invert sign on rh (it's negetive for some reason)
                        ds['rh'] = ds['rh'] * -1
                        # calculate TotalResp from ra and now positive rh
                        ds['TotalResp'] = ds['rh'] + ds['ra']
                        # subset to variables of interest
                        ds = ds[config['subset_vars']]
                    elif config['model_name'] == 'CLASSIC':
                        # create functools.partial function to pass subset variable through to preprocess
                        partial_CLASSIC = partial(preprocess_CLASSIC, config=config, sim_type=sim_type) 
                        # auto merge by variables
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_CLASSIC, combine='nested', **kwargs)
                        ds = ds[config['subset_vars']]
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print(ds, file=pf)
                    elif config['model_name'] == 'LPJ-GUESS-ML':
                        # create functools.partial function to pass subset variable through to preprocess
                        partial_LPJ_GUESS_ML = partial(preprocess_LPJ_GUESS_ML, config=config, sim_type=sim_type) 
                        # auto merge by variables
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_LPJ_GUESS_ML, combine='nested', **kwargs)
                        # calculate total soil C and N
                        ds['SoilC_Total'] =  ds['C_SOILMETA'] + ds['C_SOILSTRUCT'] + ds['C_SOILFWD'] + ds['C_SOILCWD'] + ds['C_SOILMICRO'] + ds['C_SLOWSOM'] + ds['C_PASSIVESOM']
                        ds['SoilN_Total'] =  ds['N_SOILMETA'] + ds['N_SOILSTRUCT'] + ds['N_SOILFWD'] + ds['N_SOILCWD'] + ds['N_SOILMICRO'] + ds['N_SLOWSOM'] + ds['N_PASSIVESOM']
                        # subset vars
                        ds = ds[config['subset_vars']]
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print(ds, file=pf)
                    elif config['model_name'] == 'LPJ-GUESS':
                        # create functools.partial function to pass subset variable through to preprocess
                        partial_LPJ_GUESS = partial(preprocess_LPJ_GUESS, config=config, sim_type=sim_type) 
                        # auto merge by variables
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print('begin opening datasets for ' + sim_type, file=pf)
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_LPJ_GUESS, combine='nested', **kwargs)
                        # calculate CN
                        if sim_type in ['b2','otc','sf']:
                            ds['SoilC_Total'] =  ds.SOMpool.sel(nsompool=['SOILSTRUCT','SOILMICRO','SOILMETA','SLOWSOM','PASSIVESOM']).sum(dim='nsompool')
                        # calculate totalresp
                        ds['TotalResp'] = ds['AutoResp'] + ds['HeteroResp']
                        # subset variables of interest, if b1 then dont select SoilC as it cannot be calculated from the available data files
                        if sim_type == 'b1':
                            subset_vars = [i for i in config['subset_vars'] if i not in ['SoilC_Total']]
                        else:
                            subset_vars = config['subset_vars']
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print('variables used for subsetting: ' + sim_type, file=pf)
                            print(subset_vars, file=pf)
                        ds = ds[subset_vars]
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print(ds, file=pf)
                    else:
                        # create functools.partial function to pass subset variable through to preprocess
                        partial_var = partial(preprocess_var, config=config) 
                        # auto merge by variables
                        ds = xr.open_mfdataset(sim_files, parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_var, combine='nested', **kwargs)
                        ds = ds[config['subset_vars']]
                        with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                            print(ds, file=pf)
                case 'ecosys':
                    partial_ecosys = partial(preprocess_ecosys, config=config, sim_type=sim_type) 
                    # open using mfdataset which will auto merge variables, but preprocess away incorrect time indexes 
                    ds1 = xr.open_mfdataset(sim_files[0], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ecosys, **kwargs)
                    ds2 = xr.open_mfdataset(sim_files[1], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ecosys, **kwargs)
                    ds3 = xr.open_mfdataset(sim_files[2], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ecosys, **kwargs)
                    ds4 = xr.open_mfdataset(sim_files[3], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ecosys, **kwargs)
                    if sim_type not in ['otc','sf']:
                        ds5 = xr.open_mfdataset(sim_files[4], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ecosys, **kwargs)
                        ds6 = xr.open_mfdataset(sim_files[5], parallel=True, engine=engine, chunks=chunk_read, preprocess=partial_ecosys, **kwargs)
                    # merge TotalResp and SoilTemps
                    ds = xr.merge([ds1, ds2, ds3, ds4])
                    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                        print(ds, file=pf)
                    # loop through soil layer dataarrays, for each soil layer add a depth coord and expand dim
                    ds_list_temp = []
                    layer_center = config['soil_depths']
                    temp_layers=['TMAX_SOIL_1','TMAX_SOIL_2','TMAX_SOIL_3','TMAX_SOIL_4','TMAX_SOIL_5', \
                            'TMAX_SOIL_6','TMAX_SOIL_7','TMAX_SOIL_8','TMAX_SOIL_9','TMAX_SOIL_10', 'TMAX_SOIL_11'] 
                    layer_iter = 0
                    for layer in temp_layers:
                        # expand a dimension to include site and save to list
                        ds_temp = ds.assign_coords({'SoilDepth': layer_center[layer_iter]})
                        ds_temp[layer].attrs = {}
                        ds_temp[layer] = ds_temp[layer].expand_dims('SoilDepth')
                        ds_temp = ds_temp.rename({layer: 'SoilTemp'})
                        ds_list_temp.append(ds_temp['SoilTemp'])
                        layer_iter += 1
                    # merge all the SoilTemp layers into 4D dataarray
                    ds_soiltemp = xr.combine_by_coords(ds_list_temp)
                    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                        print(ds, file=pf)
                    ds = ds.drop_vars(temp_layers)
                    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                        print(ds, file=pf)
                    ds = xr.merge([ds, ds_soiltemp])
                    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                        print(ds, file=pf)
                    # calculate TotalResp and remake attributes
                    ds['AutoResp'] = ds['ECO_GPP'] - ds['ECO_NPP']
                    ds['TotalResp'] = ds['ECO_RH'] + ds['AutoResp']
                    # deal with soilC that does not exist in every simulation
                    if sim_type in ['b2']:
                        # merge soilc variables into dataset
                        ds = xr.merge([ds, ds5, ds6]) 
                        # add all soil layer variables to SoilC
                        ds['SoilC_Total'] = ds['SOC_1'] + ds['SOC_2'] + ds['SOC_3'] + ds['SOC_4'] + ds['SOC_5'] + ds['SOC_6'] + \
                                    ds['SOC_7'] + ds['SOC_8'] + ds['SOC_9'] + ds['SOC_10'] + ds['SOC_11']
                        # approximate 1m calculation
                        ds['SoilC_1m'] = ds['SOC_1'] + ds['SOC_2'] + ds['SOC_3'] + ds['SOC_4'] + ds['SOC_5'] + ds['SOC_6'] + \
                                    ds['SOC_7'] + ds['SOC_8'] + ((15/35)*ds['SOC_9'])
                    elif sim_type in ['otc','sf']:
                        ds['SoilC_Total'] = ds['ECO_RH']
                        ds['SoilC_Total'].loc[:] = np.nan
                    # add long name and unit attributes
                    ds['SoilC_Total'].attrs['long_name'] = 'Total soil organic carbon'
                    ds['SoilC_Total'].attrs['units'] = 'gC/m2'
                    ds['TotalResp'].attrs['long_name'] = 'Autotrophic + heterotrophic respiration'
                    ds['TotalResp'].attrs['units'] = 'gC m-2 day-1'
                    ds['SoilTemp'].attrs['long_name'] = 'Soil temperature by layer'
                    ds['SoilTemp'].attrs['units'] = 'Degree C'
                    # reorde variable dimenions and remove chunking
                    ds['SoilTemp'] = ds['SoilTemp'].transpose('time', 'SoilDepth', 'lat', 'lon')
                    ds['SoilC_Total'] = ds['SoilC_Total'].transpose('time', 'lat', 'lon')
                    ds['ECO_RH'] = ds['ECO_RH'].transpose('time', 'lat', 'lon')
                    ds['ECO_GPP'] = ds['ECO_GPP'].transpose('time', 'lat', 'lon')
                    ds['AutoResp'] = ds['AutoResp'].transpose('time', 'lat', 'lon')
                    # subset to variables of interest
                    ds = ds[config['subset_vars']]
                    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                        print(ds, file=pf)
                    # change data variables to float32 to save space
                    ds = ds.astype('float32')
                    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                        print(ds, file=pf)
                case 'elm2':
                    # open single file and save to new file name
                    ds = xr.open_dataset(sim_files[0], engine=engine, chunks=chunk_read, **kwargs)
                    # subset variables
                    ds = ds[config['subset_vars']]
                    # sum all soil C and N pools
                    ds['SoilC_Layers'] = ds['SOIL1C_vr'] + ds['SOIL2C_vr'] + ds['SOIL3C_vr'] + ds['SOIL4C_vr']   
                    ds['SoilN_Layers'] = ds['SOIL1N_vr'] + ds['SOIL2N_vr'] + ds['SOIL3N_vr'] + ds['SOIL4N_vr']   
                    ds['SoilC_Total'] = ds['TOTSOMC']
                    # remove variables used for calculation
                    ds = ds.drop_vars(['SOIL1C_vr','SOIL2C_vr','SOIL3C_vr','SOIL4C_vr', \
                                       'SOIL1N_vr','SOIL2N_vr','SOIL3N_vr','SOIL4N_vr','TOTSOMC'])
                    # fix any remaining variables with levdcmp instead of levgrnd
                    problem_dim = 'levdcmp'
                    matching_variables = []
                    for var_name, data_array in ds.data_vars.items():
                        if problem_dim in data_array.dims:
                            matching_variables.append(var_name)
                    ds_problem_vars = ds[matching_variables]
                    ds = ds.drop_dims(problem_dim)
                    ds_problem_vars = ds_problem_vars.rename({"levdcmp":"levgrnd"})
                    ds_problem_vars = ds_problem_vars.assign_coords({"levgrnd": ds.coords['levgrnd'].values})
                    ds = ds.merge(ds_problem_vars)
            # rename variables
            ds = ds.rename(config['rename_subset'])
            with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                print('variables renamed', file=pf)
                print(ds, file=pf)
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
                        # reassign new index/
                        ds = ds.assign_coords({'time': new_index})
            # assign depth intergers to meter depths
            if config['model_name'] in ['UVic-ESCM','ORCHIDEE-MICT','JULES']:
                ds = ds.assign_coords({'SoilDepth': config['soil_depths']})
            # decode cftime
            ds = xr.decode_cf(ds, use_cftime=True)
            if (config['model_name'] == 'LPJ-GUESS') and (sim_type in ['b2','otc','sf']):
                ds = ds.reindex({'time': xr.cftime_range(start='2000-01-01 00:00:00', end='2022-12-31 00:00:00', freq='D', calendar='noleap')})
                with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                    print('LPJ_GUESS reindexed', file=pf)
                    print(ds, file=pf)
                    print('ds time encoding', file=pf)
                    print(ds.time.encoding, file=pf)
                    print('ds time attrs', file=pf)
                    print(ds.time.attrs, file=pf)
            with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                print('time decoded', file=pf)
                print(ds, file=pf)
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
                with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                    print('calendar changed', file=pf)
                    print(ds, file=pf)
            # multindex and unstack lndgrids so that lat lon become dimensions instead
            if config['model_name'] in ['ELM2-NGEE']:
                ds = ds.set_index(lndgrid=['lat','lon'])
                ds = ds.unstack() 
            # convert lon from -180-180 to 0-360
            if config['model_name'] in ['ELM2-NGEE', 'ecosys', 'CLASSIC', 'LPJ-GUESS-ML', 'LPJ-GUESS', 'ORCHIDEE-MICT', 'ORCHIDEE-MICT-teb','JULES']:
                ds = ds.assign_coords({'lon': (ds.lon % 360)})
                ds = ds.sortby('lon', ascending=True)
                with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                    print('adjusted lon index', file=pf)
            # remove zeros from RE,GPP,NEE
            for var in ds.data_vars:
                if var in ['AutoResp','HeteroResp','TotalResp','GPP','NEE']:
                    ds[var] = ds[var].where(ds[var] != 0.0) 
            # remove unlabeled missing values by 
            if config['model_name'] in ['UVic-ESCM','ORCHIDEE-MICT','ORCHIDEE-MICT-teb']:
                for var in ds.data_vars:
                    ds[var] = ds[var].where((ds[var] > -990.0)&(ds[var] < 1.0e20))
            # adjust units
            missing_vars = [i for i in top_config['combined_vars'] if i not in ds.data_vars]
            data_adj_vars = [i for i in config['data_units'] if i not in missing_vars]
            for var in data_adj_vars:
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
            # add empty dataframe for missing dataarrays like WTD/ALT/Cpool
            ds_fill_3d = ds['TotalResp'].copy(deep=True)
            ds_fill_3d.loc[:] = np.nan
            try:
                ds_fill_4d = ds_fill_3d.expand_dims(dim={'SoilDepth': range(1,len(config['soil_depths'])+1)}, axis=1)
            except Exception as error:
                with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                    print('No soil depth included:', file=pf)
                    print(error, file=pf)
                pass
            for var in missing_vars:
                if var not in ['SoilTemp','SoilMoist','SoilC_Layers','SoilN_Layers']:
                    ds[var] = ds_fill_3d
                else:
                    ds[var] = ds_fill_4d
            # rescale from g C m-2 s-1 to g C m-2 day-1
            ds['AutoResp'] = ds['AutoResp'] * 86400  
            ds['HeteroResp'] = ds['HeteroResp'] * 86400  
            ds['TotalResp'] = ds['TotalResp'] * 86400  
            ds['GPP'] = ds['GPP'] * 86400
            if 'NEE' not in missing_vars:
                ds['NEE'] = ds['NEE'] * 86400
            else: 
                ds['NEE'] = ds['TotalResp'] - ds['GPP']
            # replace any remaining missing_values with np.nan
            # print out combined dataset for debugging
            with open(Path(config['output_dir'] + 'zarr_output/debug_process_simulation.txt'), 'a') as pf:
                with np.printoptions(threshold=np.inf):
                    print(config['model_name'] + ':\n', file=pf)
                    print('Combined dataset before rechunking', file=pf)
                    print(ds, file=pf)
                    print('time:', file=pf)
                    print(ds['time'].head(), file=pf)
                    print('lat:', file=pf)
                    print(ds['lat'].values, file=pf)
                    print('lon:', file=pf)
                    print(ds['lon'].values, file=pf)
                    if (sim_type != 'b1') and (config['model_name'] != 'LPJ-GUESS'):
                        print('SoilDepth:', file=pf)
                        print(ds['SoilDepth'].values, file=pf)
            # output clm5 lat,lon
            if (config['model_name'] == 'CLM5') and (sim_type == 'b2'):
                lat_df = ds['lat'].to_dataframe()
                lon_df = ds['lon'].to_dataframe()
                soild_df = ds['SoilDepth'].to_dataframe()
                lat_df.to_csv(config['output_dir']+'zarr_output/clm5_lat.csv', index=False, float_format="%g")
                lon_df.to_csv(config['output_dir']+'zarr_output/clm5_lon.csv', index=False, float_format="%g")
                soild_df.to_csv(config['output_dir']+'zarr_output/clm5_soild.csv', index=False, float_format="%g")
            # convert all soiltemp data variables to float32 to save space
            varlist_float32 = ['SoilTemp','SoilMoist','SoilC_Layers','SoilN_Layers']
            for var in ds.data_vars:
                if var in varlist_float32:
                    ds[var] = ds[var].astype('float32')
                else:
                    ds[var] = ds[var].astype('float64')
            # assign simulation diension
            ds = ds.assign_coords({'sim': sim_type})
            ds = ds.expand_dims('sim')
            # rechunk
            dim_chunks = {
                'sim': 1,
                'time': zarr_chunk_out,
                'SoilDepth': -1,
                'lat': -1,
                'lon': -1}
            # LPJ-GUESS is only model that gave me SoilTemp with different dimensions across simulation types
            # to deal with initial baseline not having SoilTemp by layers I have to chunk b1 differently from all other models
            # I will likely simply remove b1 from processing as there are other issues like monthly vs yearly timesteps across variables
            if (sim_type == 'b1') and (config['model_name'] == 'LPJ-GUESS'):
                dim_chunks = {
                    'sim': 1,
                    'time': zarr_chunk_out,
                    'lat': -1,
                    'lon': -1}
            ds = ds.chunk(dim_chunks) 
            # clear all chunk and fill value encoding/attrs
            for var in ds.data_vars:
                try:
                    del ds[var].encoding['chunks']
                except:
                    pass
                try:
                    del ds[var].encoding['_FillValue']
                except:
                    pass
                try:
                    del ds[var].encoding['coordinates']
                except:
                    pass
                try:
                    del ds[var].attrs['_FillValue']
                except:
                    pass
                with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                    print('data encoding',file=pf)
                    print(ds[var].encoding, file=pf)
                    print('data attrs',file=pf)
                    print(ds[var].attrs, file=pf)
                    print('time attrs',file=pf)
                    print(ds[var].time.attrs, file=pf)
                    print('time encoding',file=pf)
                    print(ds[var].time.encoding, file=pf)
            with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                print('data rechunked',file=pf)
                print(ds, file=pf)
            # set zarr compression and encoding
            compress = Zstd(level=6)
            encode = {var: {'_FillValue': np.nan, 'compressor': compress} for var in ds.data_vars}
            # output to zarr file
            ds.to_zarr(out_file, encoding=encode, mode="w")
            with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_' + sim_type + '.txt'), 'a') as pf:
                print('encoded output using:', file=pf)
                print(encode, file=pf)
            # close file connections
            try:
                ds.close()
            except Exception as error:
                print(error)
                pass

# output variable subset netcdf
def aggregate_regional_sims(config):
    # set output file location
    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_agg_sims.txt'), 'w') as pf:
        print(config, file=pf)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # create files to combine sim datasets
        ds_list = []
        for sim in ['b2','otc','sf']:
            # create name of file to open for model/sim zarr files
            ds_file = config['output_dir'] + 'zarr_output/' + config['model_name'] + \
                      '/WrPMIP_Pan-Arctic_' + config['model_name'] + '_' + sim + '.zarr'
            ds_list.append(ds_file)
        # set expected chunks for reading
        read_chunks = {'sim': 1,'time': 73,'SoilDepth': -1,'lat': -1,'lon': -1}
        # merge datasets with open_mfdataset / dask
        zarr_kwargs = dict(decode_cf=True, mask_and_scale=False)
        #with xr.open_mfdataset(ds_list, engine='zarr', chunks=read_chunks, combine='nested', concat_dim='sim', **zarr_kwargs) as ds:
        with xr.open_mfdataset(ds_list, engine='zarr', chunks=read_chunks, use_cftime=True, **zarr_kwargs) as ds:
            with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_agg_sims.txt'), 'a') as pf:
                print('dataset:', file=pf)
                print(ds, file=pf)
                print('time encoding:', file=pf)
                print(ds.time.encoding, file=pf)
                print('time attrs:', file=pf)
                print(ds.time.attrs, file=pf)
                print('sim encoding:', file=pf)
                print(ds.sim.encoding, file=pf)
                print('sim attrs:', file=pf)
                print(ds.sim.attrs, file=pf)
            # remove fillvalue to save output
            for var in ds.data_vars:
                del ds[var].attrs['_FillValue']
                with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_agg_sims.txt'), 'a') as pf:
                    print(var + ' encoding:', file=pf)
                    print(ds[var].encoding, file=pf)
                    print(var + ' attrs:', file=pf)
                    print(ds[var].attrs, file=pf)
            ds = ds.assign_coords({'sim': [1,2,3]})
            ds = ds.chunk(read_chunks)
            # set zarr compression and encoding
            compress = Zstd(level=6) 
            encode = {var: {'_FillValue': np.nan, 'compressor': compress} for var in ds.data_vars}
            # merge dataset
            out_file = config['output_dir'] + 'zarr_output/' + config['model_name'] + \
                        '/WrPMIP_Pan-Arctic_' + config['model_name'] + '_sims_merged.zarr'
            ds.to_zarr(out_file, encoding=encode, mode="w")

# define function to calculate layer thickness and bottom node interface depths
def soil_node_calculations(config):
    # assign empty lists to fill node thickness / bottom interface
    current_interface = 0
    node_thickness = []
    node_interface = []
    indicies = []
    node_centers = config['soil_depths']
    # loop through depth center list
    for index, node in enumerate(node_centers, start=0):
        half_thickness = node - current_interface
        current_interface = node + half_thickness
        current_thickness = 2 * half_thickness
        node_thickness.append(current_thickness)
        node_interface.append(current_interface)
        if current_interface <= 1:
            indicies.append(index)
    # checl index of interface above / below 1m
    if not indicies:
        indicies = [0]
    max_index = max(indicies)
    last_index = max_index + 1
    indicies_1m = indicies
    indicies_1m.append(last_index)
    soildepths_1m = node_centers[:last_index]
    # calculate depth from bottom
    diff_thick = node_thickness[last_index] - node_thickness[max_index]
    #1m thicknesses to multiple by
    node_thickness_1m = node_thickness[:max_index]
    node_thickness_1m.append(diff_thick)        
    # return thickness and interfaces
    return node_thickness, node_interface, indicies_1m, soildepths_1m, node_thickness_1m

# output final harmonized pan-arctic regional database
def harmonize_regional_models(config):
    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_harmonization.txt'), 'w') as pf:     
        print('harmonizing ' + config['model_name'] + ':', file=pf)
    read_chunks = {'sim': 1,'time': 73,'SoilDepth': -1,'lat': -1,'lon': -1}
    # context manager to keep dask from auto-rechunking
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # create name of file to open for each model
        ds_file = config['output_dir'] + 'zarr_output/' + config['model_name'] + \
                    '/WrPMIP_Pan-Arctic_' + config['model_name'] + '_sims_merged.zarr'
        out_file = config['output_dir'] + 'zarr_output/' + config['model_name'] + \
                    '/WrPMIP_Pan-Arctic_' + config['model_name'] + '_sims_harmonized.zarr'
        # open model zarr
        zarr_kwargs = dict(decode_cf=True, mask_and_scale=False)
        with xr.open_zarr(ds_file, chunks=read_chunks, chunked_array_type='dask', use_cftime=True, **zarr_kwargs) as ds:
            with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_harmonization.txt'), 'a') as pf:
                print('dataset:', file=pf)
                print(ds, file=pf)
            # assign model names as new dimension to merge on
            mod = config['model_name']
            mod_dict = config['models']
            ds = ds.assign_coords({'model': mod_dict[mod]})
            ds = ds.expand_dims('model')
            # save clm lat,lon,soil depths for interpolation of other models
            clm_lon = pd.read_csv(config['output_dir']+'zarr_output/clm5_lon.csv', index_col=False, float_precision='round_trip')['lon']
            clm_lat = pd.read_csv(config['output_dir']+'zarr_output/clm5_lat.csv', index_col=False, float_precision='round_trip')['lat']
            clm_soild =  pd.read_csv(config['output_dir']+'zarr_output/clm5_soild.csv', index_col=False, float_precision='round_trip')['SoilDepth'] 
            # calculate node thickness, bottom interfaces, Indicies to include for 1m calculation, and node
            node_thickness, node_interface, indicies_1m, soildepths_1m, node_thickness_1m = soil_node_calculations(config)
            with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_harmonization.txt'), 'a') as pf:
                print('list of node_thickness:', file=pf)
                print(node_thickness, file=pf)
                print('list of node interface:', file=pf)
                print(node_interface, file=pf)
                print('list of indexes included to 1m:', file=pf)
                print(indicies_1m, file=pf)
                print('list of soil depths included to 1m:', file=pf)
                print(soildepths_1m, file=pf)
                print('list of node thickness included to 1m:', file=pf)
                print(node_thickness_1m, file=pf)
            # create dataarray to scale all layers by depth thickness for gC / m2
            multiplier = xr.DataArray(
                node_thickness,
                dims=['SoilDepth'],
                coords=[config['soil_depths']],
            )
            if config['model_name'] in ['CLM5','CLM-ExIce']:
                ds_soilc = ds['SoilC_Layers'] * multiplier
                ds['SoilC_Total'] = ds_soilc.sum(dim='SoilDepth')
                del ds_soilc
            if config['model_name'] in ['ELM2-NGEE']:
                ds_soiln = ds['SoilN_Layers'] * multiplier
                ds['SoilN_Total'] = ds_soiln.sum(dim='SoilDepth')
                del ds_soiln
            # calculate SoilC_1m and SoilN_1m
            multiplier_1m = xr.DataArray(
                node_thickness_1m,
                dims=['SoilDepth'],
                coords=[soildepths_1m],
            )
            if config['model_name'] in ['CLASSIC','ELM2-NGEE','JULES','ORCHIDEE-MICT','ORCHIDEE-MICT-teb']:
                ds_soilc_1m = ds['SoilC_Layers'].isel(SoilDepth=indicies_1m) * multiplier_1m
                ds['SoilC_1m'] = ds_soilc_1m.sum(dim='SoilDepth')
                del ds_soilc_1m
            if config['model_name'] in ['ELM1-ECA','ELM2-NGEE','JULES','ORCHIDEE-MICT','ORCHIDEE-MICT-teb']:
                ds_soiln_1m = ds['SoilN_Layers'].isel(SoilDepth=indicies_1m) * multiplier_1m
                ds['SoilN_1m'] = ds_soiln_1m.sum(dim='SoilDepth')
                del ds_soiln_1m
            # drop SoilC_Layers, and SoilN_Layers to save space
            ds = ds.drop_vars(['SoilC_Layers','SoilN_Layers'])
            # Calculate CN_1m and CN_total
            ds['CN_1m'] = ds['SoilC_1m'] / ds['SoilN_1m']
            ds['CN_Total'] = ds['SoilC_Total'] / ds['SoilN_Total']
            # subset/inpterplote models to clm5 grid/extent
            if config['model_name'] not in ['CLM5','CLM5-ExIce']:
                # read in clm lat/lon, make comparison da, interp_like to match grid size/extent
                fake_data = np.zeros((len(clm_lon),len(clm_lat)))
                like_array = xr.DataArray(
                                data=fake_data,
                                dims=['lon','lat'],
                                coords=dict(
                                    lon=(['lon'],clm_lon),
                                    lat=(['lat'],clm_lat)))
                ds = ds.interp_like(like_array, method="nearest", kwargs={'fill_value': np.nan})
                # mask with clm5 landmask
                with xr.open_dataset(config['output_dir']+'zarr_output/clm5_landmask.nc') as clm5_landinfo:
                    clm5_landmask = clm5_landinfo['landmask']
                ds = ds.where(clm5_landmask > 0)
                # interpolate the soil temperatures to CLM5 soil layer centers
                if config['model_name'] in ['LPJ-GUESS-ML','CLASSIC']:
                    ds['10cm_SoilTemp'] = ds['SoilTemp'].isel(SoilDepth = 0, drop=True)
                    try:
                        ds['10cm_SoilMoist'] = ds['SoilMoist'].isel(SoilDepth = 0, drop=True)
                    except:
                        pass
                if config['model_name'] not in ['LPJ-GUESS-ML']:
                    ds_soiltemp_interp = ds['SoilTemp'].interp(SoilDepth=clm_soild, method="linear", kwargs={'fill_value':'extrapolate'})
                    ds_soilmoist_interp = ds['SoilMoist'].interp(SoilDepth=clm_soild, method="linear", kwargs={'fill_value':'extrapolate'})
                    ds = ds.drop_dims(['SoilDepth'])
                    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_harmonization.txt'), 'a') as pf:
                        print('dropped soiltemp dataarray from dataset:', file=pf)
                        print(ds, file=pf)
                    ds['SoilTemp'] = ds_soiltemp_interp
                    ds['SoilMoist'] = ds_soilmoist_interp
                    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_harmonization.txt'), 'a') as pf:
                        print('added interp soiltemp dataarray to dataset:', file=pf)
                        print(ds, file=pf)
                        print('time frequency:', file=pf)
                        print(ds['time'], file=pf)
                    del ds_soiltemp_interp, ds_soilmoist_interp
                    # replace soilDepth dim with integers
                    ds = ds.assign_coords({'SoilDepth': range(0,len(clm_soild))})
                    # calculate 10cm soil average
                    coords = dict(SoilDepth=('SoilDepth', [0,1,2]))
                    weights = xr.DataArray([0.2,0.4,0.4], dims=('SoilDepth',), coords=coords)
                    if config['model_name'] not in ['LPJ-GUESS-ML','CLASSIC']:
                        ds['10cm_SoilTemp'] = ds['SoilTemp'].isel(SoilDepth = [0,1,2]).weighted(weights).mean(dim='SoilDepth')
                        ds['10cm_SoilMoist'] = ds['SoilMoist'].isel(SoilDepth = [0,1,2]).weighted(weights).mean(dim='SoilDepth')
                # if LPJ-GUESS-ML do special interpolation by grid
                if config['model_name'] in ['LPJ-GUESS-ML']:
                    ds_soiltemp_interp = ds['SoilTemp'].interp(SoilDepth=clm_soild, method="linear")
                    ds_soilmoist_interp = ds['SoilMoist'].interp(SoilDepth=clm_soild, method="linear")
                    ds = ds.drop_dims(['SoilDepth'])
                    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_harmonization.txt'), 'a') as pf:
                        print('dropped soiltemp dataarray from dataset:', file=pf)
                        print(ds, file=pf)
                    ds['SoilTemp'] = ds_soiltemp_interp
                    ds['SoilMoist'] = ds_soilmoist_interp
                    with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_harmonization.txt'), 'a') as pf:
                        print('added interp soiltemp dataarray to dataset:', file=pf)
                        print(ds, file=pf)
                        print('time frequency:', file=pf)
                        print(ds['time'], file=pf)
                    del ds_soiltemp_interp, ds_soilmoist_interp
                    # replace soilDepth dim with integers
                    ds = ds.assign_coords({'SoilDepth': range(0,len(clm_soild))})
            # calculate 10cm soil temp and moisture if CLM5
            if config['model_name'] in ['CLM5','CLM5-ExIce']:
                # replace soilDepth dim with integers
                ds = ds.assign_coords({'SoilDepth': range(0,len(clm_soild))})
                # calculate 10cm soil average
                coords = dict(SoilDepth=('SoilDepth', [0,1,2]))
                weights = xr.DataArray([0.2,0.4,0.4], dims=('SoilDepth',), coords=coords)
                ds['10cm_SoilTemp'] = ds['SoilTemp'].isel(SoilDepth = [0,1,2]).weighted(weights).mean(dim='SoilDepth')
                ds['10cm_SoilMoist'] = ds['SoilMoist'].isel(SoilDepth = [0,1,2]).weighted(weights).mean(dim='SoilDepth')
            # fix orchidee monthly data by forward filling to daily
            if config['model_name'] in ['ORCHIDEE-MICT-teb','UVic-ESCM']:
                ds = ds.resample(time='1D').ffill()
            # transpose all datasets into same vairable order for alignment
            ds = ds.transpose('model','sim','time','SoilDepth','lat','lon', missing_dims='ignore')
            # reindex time from 2000-01-01 00:00:00 to 2021-12-31 00:00:00
            ds = ds.reindex({'time': xr.cftime_range(start='2000-01-01 00:00:00', end='2022-12-31 00:00:00', freq='D', calendar='noleap')})
            # check harmonzied dim
            with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_harmonization.txt'), 'a') as pf:
                with np.printoptions(threshold=np.inf):
                    print(mod + ':\n', file=pf)
                    print(ds, file=pf)
                    print('time:', file=pf)
                    print(ds['time'], file=pf)
                    print('lat:', file=pf)
                    print(ds['lat'].values, file=pf)
                    print('lon:', file=pf)
                    print(ds['lon'].values, file=pf)
                    print('SoilDepth:', file=pf)
                    print(ds['SoilDepth'].values, file=pf)
            # set zarr compression and encoding
            compress = Zstd(level=6) #, shuffle=Blosc.BITSHUFFLE)
            dim_chunks = {'model': 1,'sim': 1,'time': 73,'SoilDepth': 25,'lat': 60,'lon': 720}
            ds = ds.chunk(dim_chunks)
            with open(Path(config['output_dir'] + 'zarr_output/' + config['model_name'] + '/debug_harmonization.txt'), 'a') as pf:
                print('rechunked', file=pf)
                print(ds, file=pf)
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
            #'chunks': ds[var].encoding['chunks'] 
            encode = {var: {'_FillValue': np.nan, 'compressor': compress} for var in ds.data_vars}
            # output regional zarr of entire pan-acrtic for all models and b2,otc,sf simulations 
            ds.to_zarr(out_file, encoding=encode, mode="w")

# output variable subset netcdf
def aggregate_regional_models(config_list):
    # context manager to not change chunk sizes
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # open each models zarr file and append to list for merge 
        ds_list = []
        for config in config_list:
            # name of each models harmonized zarr store
            ds_file = config['output_dir'] + 'zarr_output/' + config['model_name'] + \
                      '/WrPMIP_Pan-Arctic_' + config['model_name'] + '_sims_harmonized.zarr'
            ds_list.append(ds_file)
        # set expected chunks for reading
        read_chunks = {'model': 1,'sim': 1,'time': 73,'SoilDepth': 25,'lat': 60,'lon': 720}
        # merge datasets with open_mfdataset / dask
        zarr_kwargs = dict(decode_cf=True, mask_and_scale=False)
        with xr.open_mfdataset(ds_list, engine='zarr', chunks=read_chunks, use_cftime=True, combine='nested', concat_dim='model', combine_attrs='drop', **zarr_kwargs) as ds:
            # assign correct model names
            ds = ds.assign_coords({'model': list(config['models'].keys())})
            ds = ds.assign_coords({'sim': list(config['sims'].keys())})
            # assinchange soil depths to clm5
            clm_soild =  pd.read_csv(config['output_dir']+'zarr_output/clm5_soild.csv', index_col=False, float_precision='round_trip')['SoilDepth'] 
            ds = ds.assign_coords({'SoilDepth': clm_soild})
            # print out final dataset info
            with open(Path(config['output_dir'] + 'zarr_output/debug_agg_harm_models.txt'), 'a') as pf:
                 print('final harmonized zarr database', file=pf)
                 print(ds, file=pf)
            # set zarr compression and encoding 
            compress = Zstd(level=6)
            encode = {var: {'_FillValue': np.nan, 'compressor': compress} for var in ds.data_vars}
            # output final harmonized zarr store 
            out_file = config['output_dir'] + 'zarr_output/WrPMIP_Pan-Arctic_models_harmonized.zarr'
            ds.to_zarr(out_file, encoding=encode, mode="w")

# open final zarr to create netcdfs
def regional_model_zarrs_to_netcdfs(input_list):
    config = input_list[0]
    year = input_list[1]
    # loop through all years of netcdf data
    #for year in range(2000,2022):
    # try to open, slice and save
    try:
        # create date-like string for slicing
        start_year = str(year)+'-01-01'
        end_year = str(year)+'-12-31'
        # create read in file name from config
        zarr_in = config['output_dir'] + 'zarr_output/' + config['model_name'] + \
                  '/WrPMIP_Pan-Arctic_' + config['model_name'] + '_sims_harmonized.zarr'
        # open zarr file in a context manager to avoid file collisions
        with xr.open_zarr(zarr_in, chunks=None, use_cftime=True, mask_and_scale=False) as ds_tmp:
            ds = ds_tmp.sel(time=slice(start_year,end_year)).copy(deep=True)
            ds = ds.persist()
            ds_tmp.close()
        # create yearly output name
        ncdf_year_out = config['output_dir'] + 'netcdf_output/' + config['model_name'] + \
                    '/WrPMIP_Pan-Arctic_' + config['model_name'] + '_sims_harmonized_' + str(year) + '.nc'
        # remove encoding issues created by zarr for netcdf output
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
        # set netcdf encoding for yearly file
        comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
        encoding = {var: comp for var in ds.data_vars}
        # output final netcdf(s)
        ds.to_netcdf(ncdf_year_out, mode="w", encoding=encoding, \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
        # remove persisted data
        del ds
    except:
        pass

def regional_harmonized_zarr_to_monthly_netcdfs(input_list):
    config = input_list[0]
    year = input_list[1]
    month = input_list[2]
    try:
        # create subfolder for fonal outputs if not there
        Path(config['output_dir'] + '/netcdf_output/WrPMIP_Pan-Arctic_models_harmonized').mkdir(parents=True, exist_ok=True)
        Path(config['output_dir'] + '/netcdf_output/WrPMIP_Pan-Arctic_models_harmonized').chmod(0o762)
        zarr_in = config['output_dir'] + 'zarr_output/WrPMIP_Pan-Arctic_models_harmonized.zarr'
        # loop through all years
        #for year in range(2000,2022):
        # define days in month
        m = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        # loop through yearly subset and output by month
        #for month in range(0,12):
        # calculate the last day of each month for slicing
        day = m[month]
        month = month + 1
        # slice monthly dates
        month_start = str(year) + '-' + str(month).zfill(2) + '-01'
        month_end = str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
        # create monthly output file
        ncdf_month_out = config['output_dir'] + 'netcdf_output/WrPMIP_Pan-Arctic_models_harmonized/WrPMIP_Pan-Arctic_models_harmonized_' \
                    + str(year) + '_' + str(month).zfill(2) + '.nc'
        # open and subset year slice by months in context manager
        with xr.open_zarr(zarr_in, chunks=None, use_cftime=True, mask_and_scale=False) as ds_tmp:
            ds = ds_tmp.sel(time=slice(month_start, month_end)).copy(deep=True)
            ds = ds.persist()
            ds_tmp.close()
        # remove encoding issues caused by zarr for netcdf
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
        # set netcdf encoding
        comp = dict(zlib=config['nc_write']['zlib'], shuffle=config['nc_write']['shuffle'],\
                complevel=config['nc_write']['complevel'],_FillValue=None) #config['nc_write']['fillvalue'])
        encoding = {var: comp for var in ds.data_vars}
        # output final netcdf(s)
        ds.to_netcdf(ncdf_month_out, mode="w", encoding=encoding, \
                format=config['nc_write']['format'], \
                engine=config['nc_write']['engine'])
        # remove persisted data
        del ds
    except Exception as error:
        with open(Path(config['output_dir'] + 'netcdf_output/debug_model_netcdfs.txt'), 'a') as pf:
            print(error, file=pf)
        pass

def harmonized_netcdf_output(config, agg='daily', by='year', subset_model_list=[], subset_sim_list=[], subset_var_list=[]):
    try:
        # create file load location
        zarr_in = config['output_dir'] + 'zarr_output/WrPMIP_Pan-Arctic_models_harmonized.zarr'
        # open file and slice by year in context manager
        read_chunks = {'model': 1,'sim': 1,'time': 73,'SoilDepth': 25,'lat': 60,'lon': 720}
        with xr.open_zarr(zarr_in, chunks=read_chunks, chunked_array_type='dask', use_cftime=True, mask_and_scale=False) as ds:
            # subset models if subset_list given
            if subset_model_list:
                ds = ds.sel(model = subset_model_list) 
            if subset_sim_list:
                ds = ds.sel(sim = subset_sim_list) 
            if subset_var_list:
                ds = ds[subset_var_list] 
            # if monthly aggregate by month
            if agg == 'monthly':
                ds = ds.resample(time='MS').mean()
                ds = ds.transpose('model','sim','time','SoilDepth','lat','lon', missing_dims='ignore')
                ds = ds.chunk({'model':1, 'sim':1, 'time':-1, 'SoilDepth': 25, 'lat':60, 'lon':720})
            # remove encoding issues caused by zarr for netcdf
            for var in ds:
                ds[var].drop_encoding()
                ds[var].drop_attrs()
                if var == 'AutoResp':
                    ds[var] = ds[var].assign_attrs(description='Autotrophic ecosystem respiration rate (always positive)', units='gC/m2/day', _FillValue='nan')
                elif var == 'HeteroResp':
                    ds[var] = ds[var].assign_attrs(description='Heterotrophic ecosystem respiration rate (always positive)', units='gC/m2/day', _FillValue='nan')
                elif var == 'TotalResp':
                    ds[var] = ds[var].assign_attrs(description='Total ecosystem respiration rate (always positive)', units='gC/m2/day', _FillValue='nan')
                elif var == 'GPP':
                    ds[var] = ds[var].assign_attrs(description='Gross primary productivity rate (always positive)', units='gC/m2/day', _FillValue='nan')
                elif var == 'NEE':
                    ds[var] = ds[var].assign_attrs(description='Net ecosystem exchange rate (positive to atmosphere)', units='gC/m2/day', _FillValue='nan')
                elif var == 'SoilMoist':
                    ds[var] = ds[var].assign_attrs(description='Soil moisture by layer', units='gH2O/m2', _FillValue='nan')
                elif var == 'SoilTemp':
                    ds[var] = ds[var].assign_attrs(description='Soil temperature by layer', units='°C', _FillValue='nan')
                elif var == '10cm_SoilTemp':
                    ds[var] = ds[var].assign_attrs(description='Soil temperature, 10cm weighted mean', units='°C', _FillValue='nan')
                    ds = ds.rename({"10cm_SoilTemp":"SoilTemp_10cm"})
                elif var == '10cm_SoilMoist':
                    ds[var] = ds[var].assign_attrs(description='Soil Moisture, 10cm weighted mean', units='g/m2', _FillValue='nan')
                    ds = ds.rename({"10cm_SoilMoist":"SoilMoist_10cm"})
                elif var == 'AirTemp':
                    ds[var] = ds[var].assign_attrs(description='Near-surface air temperature', units='°C', _FillValue='nan')
                elif var == 'VegTemp':
                    ds[var] = ds[var].assign_attrs(description='Vegetation temperature', units='°C', _FillValue='nan')
                elif var == 'ALT':
                    ds[var] = ds[var].assign_attrs(description='Active layer thickness (positive into the soil)', units='m', _FillValue='nan')
                elif var == 'WTD':
                    ds[var] = ds[var].assign_attrs(description='Water table depth (perched and non-perched models combined)', units='m', _FillValue='nan')
                elif var == 'SoilC_Total':
                    ds[var] = ds[var].assign_attrs(description='Total soil column soil organic carbon', units='gC/m2', _FillValue='nan')
                elif var == 'SoilC_1m':
                    ds[var] = ds[var].assign_attrs(description='Total soil organic carbon to 1 meter depth', units='gC/m2', _FillValue='nan')
                elif var == 'SoilN_Total':
                    ds[var] = ds[var].assign_attrs(description='Total soil column soil organic nitrogen', units='gC/m2', _FillValue='nan')
                elif var == 'SoilN_1m':
                    ds[var] = ds[var].assign_attrs(description='Total soil organic nitrogen to 1 meter depth', units='gC/m2', _FillValue='nan')
                elif var == 'CN_Total':
                    ds[var] = ds[var].assign_attrs(description='Carbon to nitrogen ratio of entire soil column', units='unitless', _FillValue='nan')
                elif var == 'CN_1m':
                    ds[var] = ds[var].assign_attrs(description='Carbon to nitrogen ratio of soil to 1m depth', units='unitless', _FillValue='nan')
            ds.attrs = {
                'Date': date.today().strftime("%B %d, %Y"),
                'Creator': 'Jon M Wells (jon.wells@nau.edu)',
                'Project': 'The Warming Permafrost Model Intercomparison Project (WrPMIP)',
                'Description': 'Monthly mean outputs from harmonized daily database'                  
            }
            ds = ds.persist()
            ds_list = []
            ncdf_list =[]
            if by == 'year':
                for year in np.arange(2000,2022):
                    ds_tmp = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
                    ncdf_out = config['output_dir'] + 'netcdf_output/WrPMIP_Pan-Arctic_harmonized_' + str(agg) + '_means_' + str(year) + '.nc'
                    ds_list.append(ds_tmp)
                    ncdf_list.append(ncdf_out)
            elif by == 'model':
                for model in ds.model.values:
                    ds_tmp = ds.sel(model=model)
                    ncdf_out = config['output_dir'] + 'netcdf_output/WrPMIP_Pan-Arctic_harmonized_' + str(agg) + '_means_' + str(model) + '.nc'
                    ds_list.append(ds_tmp)
                    ncdf_list.append(ncdf_out)
            xr.save_mfdataset(ds_list, ncdf_list, mode='w', format=config['nc_write']['format'], engine=config['nc_write']['engine'])
    except Exception as error:
        with open(Path(config['output_dir'] + 'netcdf_output/debug_model_netcdfs.txt'), 'a') as pf:
            print(error, file=pf)
        pass

# CAVM geotiff to netcdf conversion
def cavm_geotiff_to_netcdf():
    # geotiff CAVM file
    f = '/projects/warpmip/shared/raster_cavm_v1.tif'
    # netcdf output
    f_out = '/projects/warpmip/shared/0.5_cavm.nc' 
    img_out = '/projects/warpmip/shared/cavm.png'
    # open geotiff convert from projection to lat/lon
    geo = rxr.open_rasterio(f, masked=True)
    geo = geo.rio.reproject("EPSG:4326")
    with open(Path('/projects/warpmip/shared/debug_cavm.txt'), 'w') as pf:
        print(geo, file=pf)
    # convert to xarray dataset
    geo_ds = geo.to_dataset('band')
    with open(Path('/projects/warpmip/shared/debug_cavm.txt'), 'a') as pf:
        print(geo_ds, file=pf)
    # remove spatial ref, rename var and x/y, sort lat
    geo_ds = geo_ds.drop_vars('spatial_ref')
    geo_ds = geo_ds.rename({1: 'cavm', 'x':'lon', 'y':'lat'})
    geo_ds = geo_ds.sortby('lat')
    # remove land,water,glacier,etc cells from cavm to only have tundra cells
    geo_ds = geo_ds.where(geo_ds.cavm < 80, np.nan)
    geo_ds = geo_ds.where(geo_ds.lat > 55, drop=True)
    with open(Path('/projects/warpmip/shared/debug_cavm.txt'), 'a') as pf:
        print(geo_ds, file=pf)
    # use xesmf to regrid to 0.5 x 0.5 degree grid
    ds_out = xe.util.grid_global(0.5,0.5)
    ds_out = ds_out.where(ds_out.lat > 55, drop=True)
    with open(Path('/projects/warpmip/shared/debug_cavm.txt'), 'a') as pf:
        print('global grid created', file=pf)
        print(ds_out, file=pf)
    regridder = xe.Regridder(geo_ds, ds_out, 'nearest_s2d', periodic=True)
    with open(Path('/projects/warpmip/shared/debug_cavm.txt'), 'a') as pf:
        print('regridder created', file=pf)
    geo_out = regridder(geo_ds['cavm'])
    geo_out = geo_out.to_dataset(name='cavm')
    lon = geo_out['lon'][0,:]
    lat = geo_out['lat'][:,0]
    with open(Path('/projects/warpmip/shared/debug_cavm.txt'), 'a') as pf:
        print('saving lat', file=pf)
        print(lat, file=pf)
        print('saving lon', file=pf)
        print(lon, file=pf)
    geo_out = geo_out.drop_vars(['lat','lon'])
    geo_out = geo_out.rename_dims({'x':'lon', 'y':'lat'})
    geo_out = geo_out.assign_coords({'lon': lon.values, 'lat': lat.values})
    with open(Path('/projects/warpmip/shared/debug_cavm.txt'), 'a') as pf:
        print('geo_ds regridded', file=pf)
        print(geo_out, file=pf)
    #geo_out = geo_out.where(geo_out.cavm == np.nan, 1)
    # plot cavm data
    fig = plt.figure(figsize=(8,6))
    # Set the axes using the specified map projection
    ax=plt.axes(projection=ccrs.Orthographic(0,90))
    # Make a mesh plot
    cs=ax.pcolormesh(geo_out['lon'], geo_out['lat'], geo_out['cavm'], transform = ccrs.PlateCarree(), cmap='viridis')
    ax.coastlines()
    #ax.gridlines()
    cbar = plt.colorbar(cs,shrink=0.7,location='left',label='cavm_groups')
    ax.yaxis.set_ticks_position('left')
    #ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
    add_circle_boundary(ax)
    #geo_out.cavm.plot.imshow()
    with open(Path('/projects/warpmip/shared/debug_cavm.txt'), 'a') as pf:
        print('plotted', file=pf)
    plt.savefig(img_out, dpi=300)
    plt.close(fig)
    with open(Path('/projects/warpmip/shared/debug_cavm.txt'), 'a') as pf:
        print('saved', file=pf)
    # output to netcdf file
    geo_out.to_netcdf(f_out, mode="w", format='NETCDF4_CLASSIC', engine='netcdf4')

def reccap2_plot():
    # load recap2 netcdf
    reccap2_filename = '/projects/warpmip/shared/RECCAP2_permafrost_regions_isimip3.nc'
    reccap2 = xr.open_dataset(reccap2_filename)
    reccap2 = reccap2.rename({'latitude': 'lat', 'longitude': 'lon'})
    reccap2 = reccap2['permafrost_region_mask']
    #reccap2 = reccap2.reindex_like(ds.TotalResp, method='nearest')
    reccap2 = reccap2.where(reccap2 < 1e35)
    out_geo = '/scratch/jw2636/processed_outputs/regional/figures/reccap2_mask.png' 
    # Set figure size
    fig_reccap = plt.figure(figsize=(8,6))
    # Set the axes using the specified map projection
    ax=plt.axes(projection=ccrs.Orthographic(0,90))
    # Make a mesh plot
    cs=ax.pcolormesh(reccap2['lon'], reccap2['lat'], reccap2, transform = ccrs.PlateCarree(), cmap='viridis')
    ax.coastlines()
    #ax.gridlines()
    cbar = plt.colorbar(cs,shrink=0.7,location='left',label='reccap2')
    ax.yaxis.set_ticks_position('left')
    #ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
    add_circle_boundary(ax)
    plt.savefig(out_geo, dpi=300)
    plt.close(fig_reccap)

def warming_treatment_data(input_list):
    # context manager to not change chunk sizes
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # assign config and variable list from input_list
        config = input_list[0]
        var_list = input_list[1]
        with open(Path(config['output_dir'] + 'figures/debug_graph_data.txt'), 'w') as pf:
            print('graphing started:', file=pf)
        # open zarr file
        zarr_file = config['output_dir'] + 'zarr_output/WrPMIP_Pan-Arctic_models_harmonized.zarr'
        read_chunks = {'model': 1,'sim': 1,'time': 73,'SoilDepth': 25,'lat': 60,'lon': 720}
        ds = xr.open_zarr(zarr_file, chunks=read_chunks, chunked_array_type='dask', use_cftime=True) 
        with open(Path(config['output_dir'] + 'figures/debug_graph_data.txt'), 'a') as pf:
            print('data loaded:', file=pf)
            print(ds, file=pf)
        # select variables
        ds = ds[var_list]
        # change to -180/180 coords
        ds['lon'] =('lon', (((ds.lon.values + 180) % 360) - 180))
        ds = ds.sortby(['lon'])
        # select time window and resample to annual cummulative flux and annual mean environmental variables
        ds = ds.sel(time=slice('2000-01-01','2022-12-31'))
        var_sub_cum = ['AutoResp','HeteroResp','TotalResp','GPP','NEE']
        ds_summer_cum = ds[var_sub_cum].copy(deep=True)
        ds_summer_cum = ds_summer_cum.sel(time=is_summer(ds_summer_cum['time.month'])).groupby('time.year').sum(dim=['time'], skipna=True)
        var_sub_mean = ['VegTemp','AirTemp','10cm_SoilTemp','10cm_SoilMoist','ALT','WTD','SoilC','SoilN','CN']
        ds_summer_mean = ds[var_sub_mean].copy(deep=True)
        ds_summer_mean = ds_summer_mean.sel(time=is_summer(ds_summer_mean['time.month'])).groupby('time.year').mean(dim=['time'], skipna=True)
        ds_summer_max = ds[['ALT', 'WTD']].copy(deep=True)
        ds_summer_max = ds_summer_max.sel(time=is_summer(ds_summer_max['time.month'])).groupby('time.year').max(dim=['time'], skipna=True)
        ds_summer = ds_summer_cum.merge(ds_summer_mean)
        ds_summer['ALT_MAX'] = ds_summer_max['ALT']
        ds_summer['WTD_MAX'] = ds_summer_max['WTD']
        ds_summer = ds_summer.chunk({'sim':-1, 'year':-1, 'lat':-1, 'lon':-1}).persist()
        #ds_winter = ds_winter_cum.merge(ds_winter_mean)
        with open(Path(config['output_dir'] + 'figures/debug_graph_data.txt'), 'a') as pf:
            print('annual means and sums:', file=pf)
            print(ds_summer, file=pf)
        # create list for grids around sites
        ds_list = []
        ds_summer_list = []
        ds_winter_list = []
        site_gps = config['site_gps'] 
        # loop through sites and select grids
        for site in site_gps:
            # find middle grid center
            ds_center = ds_summer.sel(lon=site_gps[site]['lon'], lat=site_gps[site]['lat'], method='nearest')
            ds_summer_sub = ds_summer.sel(lon=slice(ds_center['lon']-2.0, ds_center['lon']+2.0), lat=slice(ds_center['lat']-2.0, ds_center['lat']+2.0)).copy(deep=True)
            #ds_sub = ds.sel(lon=slice(ds_center['lon']-2.0, ds_center['lon']+2.0), lat=slice(ds_center['lat']-2.0, ds_center['lat']+2.0)).copy(deep=True)
            #ds_winter_sub = ds_winter.sel(lon=slice(ds_center['lon']-2.0, ds_center['lon']+2.0), lat=slice(ds_center['lat']-2.0, ds_center['lat']+2.0)).copy()
            with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
                print(ds_center, file=pf)
                print(ds_summer_sub, file=pf)
                print(site, file=pf)
            # calculate summer cum fluxes and summer mean responses for everything else
            ds_summer_sub = ds_summer_sub.stack(gridcell=['lon','lat'], create_index=False) 
            ds_summer_sub = ds_summer_sub.drop_vars(['lat','lon']) 
            ds_summer_sub = ds_summer_sub.assign_coords({'gridcell': np.arange(1,ds_summer_sub.gridcell.size+1)})
            #ds_sub = ds_sub.stack(gridcell=['lon','lat'], create_index=False)
            #ds_sub = ds_sub.drop_vars(['lat','lon']) 
            #ds_sub = ds_sub.assign_coords({'gridcell': np.arange(1,ds_sub.gridcell.size+1)})
            with open(Path(config['output_dir'] + 'figures/debug_graph_data.txt'), 'a') as pf:
                print('ds_summer_sub:', file=pf)
                print(ds_summer_sub, file=pf)
            # expand a dimension to include site and save to list
            ds_summer_sub = ds_summer_sub.assign_coords({'site': site})
            ds_summer_sub = ds_summer_sub.expand_dims('site')
            #ds_sub = ds_sub.assign_coords({'site': site})
            #ds_sub = ds_sub.expand_dims('site')
            with open(Path(config['output_dir'] + 'figures/debug_graph_data.txt'), 'a') as pf:
                print(ds_summer_sub, file=pf)
            ds_summer_list.append(ds_summer_sub)
            #ds_list.append(ds_sub)
        with open(Path(config['output_dir'] + 'figures/debug_graph_data.txt'), 'a') as pf:
            print('datasets appended', file=pf)
        ds.close()
        # combine site dimension to have multiple sites
        ds_summer_sites = xr.combine_by_coords(ds_summer_list).chunk({'site':-1, 'year':-1, 'sim':-1, 'gridcell':-1}).persist()
        #ds_sites = xr.combine_by_coords(ds_list).chunk({'site':-1, 'time': -1, 'sim':-1, 'gridcell':-1}).persist()
        # calculate the normalized effectsize at sites
        #ds_sites_clean = ds_sites.where(ds_sites['TotalResp'] > 8)
        #ds_sites_deltas = ds_sites_clean.sel(sim='otc') - ds_sites_clean.sel(sim='b2')
        #ds_sites_deltas = ds_sites_deltas.where(ds_sites_deltas['10cm_SoilTemp'] > 0.05)
        #ds_sites_effectsize = (ds_sites_deltas['TotalResp'] / ds_sites_clean['TotalResp'].sel(sim='b2')) * 100
        #ds_sites_effectsize = ds_sites_effectsize.to_dataset(name='TotalResp')
        #ds_sites_nef = ds_sites_effectsize / ds_sites_deltas['10cm_SoilTemp'] 
        ds_summer_sites_clean = ds_summer_sites.where(ds_summer_sites['TotalResp'] > 8) # cell selection for carbon
        ds_summer_sites_deltas = ds_summer_sites_clean.sel(sim='otc') - ds_summer_sites_clean.sel(sim='b2') # otc - b2 delta
        ds_summer_sites_deltas = ds_summer_sites_deltas.where(ds_summer_sites_deltas['10cm_SoilTemp'] > 0.05) # cell selection for warming
        ds_summer_sites_effectsize = (ds_summer_sites_deltas['TotalResp'] / ds_summer_sites_clean['TotalResp'].sel(sim='b2')) * 100 # otc percent change from b2
        ds_summer_sites_effectsize = ds_summer_sites_effectsize.to_dataset(name='TotalResp')
        ds_summer_sites_nef = ds_summer_sites_effectsize / ds_summer_sites_deltas['10cm_SoilTemp'] # effectsize normalized by delta 10cm_SoilTemp
        with open(Path(config['output_dir'] + 'figures/debug_graph_data.txt'), 'a') as pf:
            print('ds_summer_sites_clean:', file=pf)
            print(ds_summer_sites_clean, file=pf)
            print('ds_summer_sites_deltas:', file=pf)
            print(ds_summer_sites_deltas, file=pf)
            print('ds_summer_sites_effectsize:', file=pf)
            print(ds_summer_sites_effectsize, file=pf)
            print('ds_summer_sites_nef:', file=pf)
            print(ds_summer_sites_nef, file=pf)
        # count values
        # count_total = ds_summer_sites['TotalResp'].count()
        # count_total_nonnan = ds_summer_sites['TotalResp'].isnull().sum()
        # count_total_sf = ds_summer_sites['TotalResp'].sel(sim='sf').count() 
        # count_total_otc = ds_summer_sites['TotalResp'].sel(sim='otc').count() 
        # count_total_b2 = ds_summer_sites['TotalResp'].sel(sim='b2').count() 
        # count_total_nonnan_sf = ds_summer_sites['TotalResp'].sel(sim='sf').isnull().sum()
        # count_total_nonnan_otc = ds_summer_sites['TotalResp'].sel(sim='otc').isnull().sum()
        # count_total_nonnan_b2 = ds_summer_sites['TotalResp'].sel(sim='b2').isnull().sum()
        # count_clean_sf = ds_summer_sites_clean['TotalResp'].sel(sim='sf').isnull().sum()
        # count_clean_otc = ds_summer_sites_clean['TotalResp'].sel(sim='otc').isnull().sum()
        # count_clean_b2 = ds_summer_sites_clean['TotalResp'].sel(sim='b2').isnull().sum()
        # count_deltas_total = ds_summer_sites_deltas['TotalResp'].count()
        # count_deltas_clean = ds_summer_sites_deltas['TotalResp'].isnull().sum()
        # count_ef_total = ds_summer_sites_effectsize.count()
        # count_ef_clean = ds_summer_sites_effectsize.isnull().sum()
        # count_nef_total = ds_summer_sites_nef.count()
        # count_nef_clean = ds_summer_sites_nef.isnull().sum()
        # count_list = ['total','total_nonnan','total_sf','total_otc','total_b2','total_nonnan_sf','total_nonnan_otc','total_nonnan_b2',\
        #               'clean_sf','clean_otc','clean_b2','delta_total','delta_clean','ef_total','ef_clean','nef_total','nef_clean']
        # count_values = [count_total, count_total_nonnan, count_total_sf, count_total_otc, count_total_b2, count_total_nonnan_sf, \
        #                 count_total_nonnan_otc, count_total_nonnan_b2, count_clean_sf, count_clean_otc, count_clean_b2, count_deltas_total, \
        #                 count_deltas_clean, count_ef_total, count_ef_clean, count_nef_total, count_nef_clean]
        # df_counts = pd.DataFrame({'calc': count_list, 'count': count_values})
        # df_counts.to_csv(config['output_dir'] + 'figures/ds_counts.csv', index=False)
        # normalize C,N,C:N for Resp v N plots
        ds_summer_sites_nef['TotalResp_std'] = ds_summer_sites_nef['TotalResp'].std(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['SoilN_b2'] = ds_summer_sites_clean['SoilN'].sel(sim='b2').mean(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['SoilC_b2'] = ds_summer_sites_clean['SoilC'].sel(sim='b2').mean(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['CN_b2'] = ds_summer_sites_clean['CN'].sel(sim='b2').mean(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['SoilN_b2_std'] = ds_summer_sites_clean['SoilN'].sel(sim='b2').std(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['SoilC_b2_std'] = ds_summer_sites_clean['SoilC'].sel(sim='b2').std(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['CN_b2_std'] = ds_summer_sites_clean['CN'].sel(sim='b2').std(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['SoilN_b2_modmean'] = ds_summer_sites_clean['SoilN'].sel(sim='b2').mean(dim=['gridcell','year','site'], skipna=True)
        ds_summer_sites_nef['SoilC_b2_modmean'] = ds_summer_sites_clean['SoilC'].sel(sim='b2').mean(dim=['gridcell','year','site'], skipna=True)
        ds_summer_sites_nef['CN_b2_modmean'] = ds_summer_sites_clean['CN'].sel(sim='b2').mean(dim=['gridcell','year','site'], skipna=True)
        ds_summer_sites_nef['SoilN_b2_centered'] = (ds_summer_sites_clean['SoilN'].sel(sim='b2')/ds_summer_sites_nef['SoilN_b2_modmean']).mean(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['SoilC_b2_centered'] = (ds_summer_sites_clean['SoilC'].sel(sim='b2')/ds_summer_sites_nef['SoilC_b2_modmean']).mean(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['CN_b2_centered'] = (ds_summer_sites_clean['CN'].sel(sim='b2')/ds_summer_sites_nef['CN_b2_modmean']).mean(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['SoilN_b2_centered_std'] = (ds_summer_sites_clean['SoilN'].sel(sim='b2')/ds_summer_sites_nef['SoilN_b2_modmean']).std(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['SoilC_b2_centered_std'] = (ds_summer_sites_clean['SoilC'].sel(sim='b2')/ds_summer_sites_nef['SoilC_b2_modmean']).std(dim=['gridcell','year'], skipna=True)
        ds_summer_sites_nef['CN_b2_centered_std'] = (ds_summer_sites_clean['CN'].sel(sim='b2')/ds_summer_sites_nef['CN_b2_modmean']).std(dim=['gridcell','year'], skipna=True)
        # output cleaned, delta, effectsize, nef datasets
            # ds_sites,
            # ds_sites_clean,
            # ds_sites_deltas,
            # ds_sites_effectsize,
            # ds_sites_nef,
        ds_list = [
            ds_summer,
            ds_summer_sites, 
            ds_summer_sites_clean, 
            ds_summer_sites_deltas, 
            ds_summer_sites_effectsize, 
            ds_summer_sites_nef]
            # config['output_dir'] + 'figures/ds_sites.nc',
            # config['output_dir'] + 'figures/ds_sites_clean.nc',
            # config['output_dir'] + 'figures/ds_sites_deltas.nc',
            # config['output_dir'] + 'figures/ds_sites_effectsize.nc',
            # config['output_dir'] + 'figures/ds_sites_nef.nc',
        ncdf_list = [
            config['output_dir'] + 'figures/ds_summer.nc',
            config['output_dir'] + 'figures/ds_summer_sites.nc',
            config['output_dir'] + 'figures/ds_summer_sites_clean.nc',
            config['output_dir'] + 'figures/ds_summer_sites_deltas.nc',
            config['output_dir'] + 'figures/ds_summer_sites_effectsize.nc',
            config['output_dir'] + 'figures/ds_summer_sites_nef.nc']
        xr.save_mfdataset(ds_list, ncdf_list, mode='w', format=config['nc_write']['format'], engine=config['nc_write']['engine'])

def warming_treatment_effect_graphs(input_list):
    # context manager to not change chunk sizes
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # graph type (mean, instantaneous)
        config = input_list[0]
        var_list = input_list[1]
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'w') as pf:
            print('graphing started:', file=pf)
        # open zarr file
        zarr_file = config['output_dir'] + 'zarr_output/WrPMIP_Pan-Arctic_models_harmonized.zarr'
        read_chunks = {'model': 1,'sim': 1,'time': 73,'SoilDepth': 25,'lat': 60,'lon': 720}
        ds = xr.open_zarr(zarr_file, chunks=read_chunks, chunked_array_type='dask', use_cftime=True) 
        # select soil layer
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('data loaded:', file=pf)
            print(ds, file=pf)
        # select variables of interest
        ds = ds[var_list]
        # change to -180/180 coords
        ds['lon'] =('lon', (((ds.lon.values + 180) % 360) - 180))
        # use sortby to enforce monotonically increasing dims for xarray
        ds = ds.sortby(['lon'])
        # sort models (alphabetically didn't work as capitalization seems to matter)
        #desired_order = ['CLASSIC','CLM5','CLM5-ExIce','ecosys','ELM1-ECA','ELM2-NGEE','JULES','JSBACH', \
        #                 'LPJ-GUESS','LPJ-GUESS-ML','ORCHIDEE-MICT','ORCHIDEE-MICT-teb','UVic-ESCM']
        #ds_order = xr.DataArray(data=range(0,len(desired_order)), dims=['model'], coords={'model': desired_order})
        #ds = ds.sortby(ds_order)
        # calculate summer cum fluxes and summer mean responses for everything else
        #ds = ds.sel(model=["CLM5","CLM5-ExIce","ELM2-NGEE","UVic-ESCM","ecosys","ELM1-ECA","CLASSIC"])
        ds = ds.sel(time=slice('2000-01-01','2020-12-31'))
        var_sub_cum = ['AutoResp','HeteroResp','TotalResp','GPP','NEE']
        ds_summer_cum = ds[var_sub_cum].copy(deep=True)
        ds_summer_cum = ds_summer_cum.sel(time=is_summer(ds_summer_cum['time.month'])).groupby('time.year').sum(dim=['time'], skipna=True)
        ds_winter_cum = ds[var_sub_cum].copy(deep=True)
        ds_winter_cum = ds_winter_cum.sel(time=is_winter(ds_winter_cum['time.month'])).groupby('time.year').sum(dim=['time'], skipna=True)
        var_sub_mean = ['VegTemp','AirTemp','10cm_SoilTemp','10cm_SoilMoist','ALT','WTD','SoilC','SoilN','CN']
        ds_summer_mean = ds[var_sub_mean].copy(deep=True)
        ds_summer_mean = ds_summer_mean.sel(time=is_summer(ds_summer_mean['time.month'])).groupby('time.year').mean(dim=['time'], skipna=True)
        ds_winter_mean = ds[var_sub_mean].copy(deep=True)
        ds_winter_mean = ds_winter_mean.sel(time=is_winter(ds_winter_mean['time.month'])).groupby('time.year').mean(dim=['time'], skipna=True)
        ds_summer = ds_summer_cum.merge(ds_summer_mean).persist()
        #ds_summer.to_netcdf(Path(config['output_dir'] + 'figures/ds_summer.nc'), mode="w", format='NETCDF4_CLASSIC', engine='netcdf4')
        ds_efd_summer = ds_summer.copy(deep=True)
        ds_summer_deltas = ds_summer.sel(sim='otc') - ds_summer.sel(sim='b2')
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('ds summer delta (otc - b2)', file=pf)
            print(ds_summer_deltas, file=pf)
            print('TotalResp from ds_summer_deltas', file=pf)
            print(ds_summer_deltas['TotalResp'], file=pf)
            print('TotalResp from ds_summer selected for baseline', file=pf)
            print(ds_summer['TotalResp'].sel(sim='b2'), file=pf)
            print('10cm_soiltemp from summer deltas', file=pf)
            print(ds_summer_deltas['10cm_SoilTemp'], file=pf)
        ds_efd_summer['TotalResp_efd'] = (ds_summer_deltas['TotalResp'] / ds_summer['TotalResp'].sel(sim='b2')) / ds_summer_deltas['10cm_SoilTemp'] 
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('ds effect size normed by soil warming', file=pf)
            print(ds_efd_summer, file=pf)
        ds_efd_summer['GPP_efd'] = (ds_summer_deltas['GPP'] / ds_summer['GPP'].sel(sim='b2')) / ds_summer_deltas['10cm_SoilTemp'] 
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('ds effect size normed by temp difference', file=pf)
            print(ds_efd_summer, file=pf)
        ds_winter = ds_winter_cum.merge(ds_winter_mean).persist()
        #ds_winter.to_netcdf(Path(config['output_dir'] + 'figures/ds_winter.nc'), mode="w", format='NETCDF4_CLASSIC', engine='netcdf4')
        ds_efd_winter = ds_winter.copy(deep=True)
        ds_winter_deltas = ds_winter.sel(sim='sf') - ds_winter.sel(sim='b2')
        ds_efd_winter['TotalResp_efd'] = (ds_winter_deltas['TotalResp'] / ds_winter['TotalResp'].sel(sim='b2')) / ds_winter_deltas['10cm_SoilTemp'] 
        ds_efd_winter['GPP_efd'] = (ds_winter_deltas['GPP'] / ds_winter['GPP'].sel(sim='b2')) / ds_winter_deltas['10cm_SoilTemp'] 
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('summer calculation of 10cm soiltemp', file=pf)
            print(ds_summer, file=pf)
            print('winter calculation of 10cm soiltemp', file=pf)
            print(ds_winter, file=pf)
        ds_summer_geomean = ds_summer.mean(dim=['lat','lon'], skipna=True)
        ds_summer_geomean_deltas = ds_summer_geomean.sel(sim='otc') - ds_summer_geomean.sel(sim='b2')
        ds_winter_geomean = ds_winter.mean(dim=['lat','lon'], skipna=True)
        ds_winter_geomean_deltas = ds_winter_geomean.sel(sim='sf') - ds_winter_geomean.sel(sim='b2')
        cavm_f = '/projects/warpmip/shared/0.5_cavm.nc' 
        cavm = xr.open_dataset(cavm_f)
        ds_summer_geomean_cavm = ds_summer.where(cavm.cavm > 0)
        ds_summer_geomean_cavm = ds_summer_geomean_cavm.mean(dim=['lat','lon'], skipna=True)
        ds_summer_geomean_cavm_deltas = ds_summer_geomean_cavm.sel(sim='otc') - ds_summer_geomean_cavm.sel(sim='b2')
        ds_summer_effectsize = ds_summer_geomean_cavm_deltas.copy(deep=True)
        ds_summer_effectsize['TotalResp'] = ds_summer_effectsize['TotalResp'] / ds_summer_geomean_cavm['TotalResp'].sel(sim='b2')
        ds_summer_effectsize['GPP'] = ds_summer_effectsize['GPP'] / ds_summer_geomean_cavm['GPP'].sel(sim='b2')
        ds_summer_effectsize['NEE'] = ds_summer_effectsize['NEE'] / ds_summer_geomean_cavm['NEE'].sel(sim='b2')
        ds_winter_geomean_cavm = ds_winter.where(cavm.cavm > 0)
        ds_winter_geomean_cavm = ds_winter_geomean_cavm.mean(dim=['lat','lon'], skipna=True)
        ds_winter_geomean_cavm_deltas = ds_winter_geomean_cavm.sel(sim='sf') - ds_winter_geomean_cavm.sel(sim='b2')
        ds_winter_effectsize = ds_winter_geomean_cavm_deltas.copy(deep=True)
        ds_winter_effectsize['TotalResp'] = ds_winter_effectsize['TotalResp'] / ds_winter_geomean_cavm['TotalResp'].sel(sim='b2')
        ds_winter_effectsize['GPP'] = ds_winter_effectsize['GPP'] / ds_winter_geomean_cavm['GPP'].sel(sim='b2')
        ds_winter_effectsize['NEE'] = ds_winter_effectsize['NEE'] / ds_winter_geomean_cavm['NEE'].sel(sim='b2')
        # subsample sites
        # create list for ds after gps selection
        ds_summer_list = []
        ds_winter_list = []
        site_gps = config['site_gps'] 
        for site in site_gps:
            # find middle grid center
            ds_center = ds_summer.sel(lon=site_gps[site]['lon'], lat=site_gps[site]['lat'], method='nearest')
            ds_summer_sub = ds_summer.sel(lon=slice(ds_center['lon']-2.0, ds_center['lon']+2.0), lat=slice(ds_center['lat']-2.0, ds_center['lat']+2.0)).copy()
            ds_winter_sub = ds_winter.sel(lon=slice(ds_center['lon']-2.0, ds_center['lon']+2.0), lat=slice(ds_center['lat']-2.0, ds_center['lat']+2.0)).copy()
            with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
                print(ds_center, file=pf)
                print(ds_summer_sub, file=pf)
                print(site, file=pf)
            # calculate summer cum fluxes and summer mean responses for everything else
            ds_summer_sub = ds_summer_sub.stack(gridcell=['lon','lat'], create_index=False) 
            ds_summer_sub = ds_summer_sub.drop_vars(['lat','lon']) 
            ds_summer_sub = ds_summer_sub.assign_coords({'gridcell': np.arange(1,ds_summer_sub.gridcell.size+1)})
            ds_winter_sub = ds_winter_sub.stack(gridcell=['lon','lat'], create_index=False)
            ds_winter_sub = ds_winter_sub.drop_vars(['lat','lon']) 
            ds_winter_sub = ds_winter_sub.assign_coords({'gridcell': np.arange(1,ds_summer_sub.gridcell.size+1)})
            #ds_summer_sub = ds_summer_sub.mean(dim=['lat','lon'], skipna=True)
            #ds_winter_sub = ds_winter_sub.mean(dim=['lat','lon'], skipna=True)
            with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
                print('ds_summer_sub:', file=pf)
                print(ds_summer_sub, file=pf)
            #ds_sub['site_mean'] = ds_sub.mean(dim=['lon','lat'], skipna=True)
            #ds_sub['site_std'] = ds_sub.std(dim=['lon','lat'], skipna=True)
            # expand a dimension to include site and save to list
            ds_summer_sub = ds_summer_sub.assign_coords({'site': site})
            ds_summer_sub = ds_summer_sub.expand_dims('site')
            ds_winter_sub = ds_winter_sub.assign_coords({'site': site})
            ds_winter_sub = ds_winter_sub.expand_dims('site')
            #ds_sub = ds_sub.reset_coords(['lat','lon'])
            #ds_sub['lat'] = ds_sub['lat'].expand_dims('site')
            #ds_sub['lon'] = ds_sub['lon'].expand_dims('site')
            with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
                print(ds_summer_sub, file=pf)
            ds_summer_list.append(ds_summer_sub)
            ds_winter_list.append(ds_winter_sub)
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('datasets appended', file=pf)
        # combine site dimension to have multiple sites
        ds_summer_sites = xr.combine_by_coords(ds_summer_list)
        ds_winter_sites = xr.combine_by_coords(ds_winter_list)
        # calculate the temp differences for otc - ctrl
        ds_summer_site_all = ds_summer_sites.copy(deep=True)
        ds_winter_site_all = ds_winter_sites.copy(deep=True)
        ds_summer_sites = ds_summer_sites.where(ds_summer_sites.TotalResp > 8)
        ds_summer_site_grids = ds_summer_sites.copy(deep=True)
        ds_summer_site_clean = ds_summer_sites.where(ds_summer_sites['10cm_SoilTemp'] > 0.05) #wrong
        # hsmd of normalized effect size
        ds_summer_site_nef_hsmd = ds_summer_site_clean['TotalResp'].sel(sim='otc') / ds_summer_site_clean['TotalResp'].sel(sim='b2')
        ds_summer_site_nef_hsmd = ds_summer_site_nef_hsmd.to_dataset(name='resp_ratio_otc')
        ds_summer_site_nef_hsmd['resp_ratio_b2'] = ds_summer_site_clean['TotalResp'].sel(sim='b2') / ds_summer_site_clean['TotalResp'].sel(sim='b2')
        ds_summer_site_nef_hsmd['temp_delta'] = ds_summer_site_clean['10cm_SoilTemp'].sel(sim='otc') / ds_summer_site_clean['10cm_SoilTemp'].sel(sim='b2')
        ds_summer_site_nef_otc = ds_summer_site_nef_hsmd['resp_ratio_otc'] / ds_summer_site_nef_hsmd['temp_delta']
        ds_summer_site_nef_b2 = ds_summer_site_nef_hsmd['resp_ratio_b2'] / ds_summer_site_nef_hsmd['temp_delta']
        ds_summer_site_nef_otc = ds_summer_site_nef_otc.assign_coords({'sim': 'otc'})
        ds_summer_site_nef_otc = ds_summer_site_nef_otc.expand_dims('sim')
        ds_summer_site_nef_b2 = ds_summer_site_nef_b2.assign_coords({'sim': 'b2'})
        ds_summer_site_nef_b2 = ds_summer_site_nef_b2.expand_dims('sim')
        ds_nef = xr.concat([ds_summer_site_nef_otc,ds_summer_site_nef_b2], dim='sim')
        ds_summer_site_nef_hsmd['nef_otc'] = ds_nef.sel(sim='otc').mean(dim='gridcell', skipna=False)
        ds_summer_site_nef_hsmd['nef_b2'] = ds_nef.sel(sim='b2').mean(dim='gridcell', skipna=False)
        ds_summer_site_nef_hsmd['nef_psdev'] = ds_nef.sel(sim=['otc','b2']).std(dim=['gridcell','sim'], skipna=False)
        ds_summer_site_nef_hsmd['nef_hsmd'] = (ds_summer_site_nef_hsmd['nef_otc'] - ds_summer_site_nef_hsmd['nef_b2']) / ds_summer_site_nef_hsmd['nef_psdev']
        #ds_summer_site_nef_hsmd.to_netcdf(Path(config['output_dir'] + 'figures/ds_summer_site_nef_hsmd.nc'), mode="w", format='NETCDF4_CLASSIC', engine='netcdf4')
        # er hsmd
        ds_summer_site_means_otc = ds_summer_site_clean.sel(sim='otc').mean(dim='gridcell', skipna=True)
        ds_summer_site_means_b2 = ds_summer_site_clean.sel(sim='b2').mean(dim='gridcell', skipna=True)
        ds_summer_site_pooledsd = ds_summer_site_clean.sel(sim=['otc','b2']).std(dim=['gridcell','sim'], skipna=True)
        ds_summer_site_hsmd = (ds_summer_site_means_otc - ds_summer_site_means_b2) / ds_summer_site_pooledsd 
        ds_summer_site_hsmd['SoilN_b2'] = ds_summer_site_means_b2['SoilN']
        ds_summer_site_hsmd['SoilC_b2'] = ds_summer_site_means_b2['SoilC']
        ds_summer_site_hsmd['CN_b2'] = ds_summer_site_means_b2['CN']
        #ds_summer_site_hsmd.to_netcdf(Path(config['output_dir'] + 'figures/ds_summer_site_hsmd.nc'), mode="w", format='NETCDF4_CLASSIC', engine='netcdf4')
        #ds_summer_site_hsmd.to_netcdf(Path(config['output_dir'] + 'figures/ds_grids_hsmd.nc'), mode="w", format='NETCDF4_CLASSIC', engine='netcdf4')
        #ds_summer_site_means_otc_all = ds_summer_site_all.sel(sim='otc').mean(dim='gridcell', skipna=True)
        #ds_summer_site_means_b2_all = ds_summer_site_all.sel(sim='b2').mean(dim='gridcell', skipna=True)
        #ds_summer_site_pooledsd_all = ds_summer_site_all.sel(sim=['otc','b2']).std(dim=['gridcell','sim'], skipna=True)
        #ds_summer_site_hsmd_all = (ds_summer_site_means_otc_all - ds_summer_site_means_b2_all) / ds_summer_site_pooledsd_all
        #ds_summer_site_hsmd_all['SoilN'] = ds_summer_site_means_b2_all['SoilN']
        #ds_summer_site_hsmd_all['SoilC'] = ds_summer_site_means_b2_all['SoilC']
        #ds_summer_site_hsmd_all['CN'] = ds_summer_site_means_b2_all['CN']
        ds_summer_site_deltas_all = ds_summer_site_all.sel(sim='otc') - ds_summer_site_all.sel(sim='b2')
        ds_summer_site_deltas = ds_summer_sites.sel(sim='otc') - ds_summer_sites.sel(sim='b2')
        ds_summer_site_deltas = ds_summer_site_deltas.where(ds_summer_site_deltas['10cm_SoilTemp'] > 0.05)
        df_sum_site_delta = ds_summer_site_deltas['TotalResp'].to_dataframe()
        df_sum_site_delta.to_csv(Path(config['output_dir'] + 'figures/summer_site_er_delta.csv'))
        df_baseline_er = ds_summer_sites['TotalResp'].sel(sim='b2').to_dataframe()
        df_baseline_er.to_csv(Path(config['output_dir'] + 'figures/summer_site_er_baseline.csv'))
        df_sum_temp_delta = ds_summer_site_deltas['10cm_SoilTemp'].to_dataframe()
        df_sum_temp_delta.to_csv(Path(config['output_dir'] + 'figures/summer_site_temp_delta.csv'))
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('site summer deltas', file=pf)
            print(ds_summer_site_deltas, file=pf)
            print('site summer deltas: TotalResp', file=pf)
            print(ds_summer_site_deltas['TotalResp'], file=pf)
            print('site summer TotalResp selected for baseline', file=pf)
            print(ds_summer_sites['TotalResp'].sel(sim='b2'), file=pf)
        ds_site_efd_summer = ds_summer_site_deltas['TotalResp'] / ds_summer_sites['TotalResp'].sel(sim='b2') * 100
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('site summer efd', file=pf)
            print(ds_site_efd_summer, file=pf)
        ds_site_efd_summer = ds_site_efd_summer / ds_summer_site_deltas['10cm_SoilTemp']
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('ds sites summer effect size normalized by soil warming', file=pf)
            print(ds_site_efd_summer, file=pf)
        # output to csv to check numbers
        ds_summer_site_mean = ds_summer_sites.mean(dim='site', skipna=True)
        ds_summer_site_mean_deltas = ds_summer_site_mean.sel(sim='otc') - ds_summer_site_mean.sel(sim='b2')
        ds_winter_site_deltas = ds_winter_sites.sel(sim='sf') - ds_winter_sites.sel(sim='b2')
        ds_site_efd_winter = ds_winter_site_deltas['TotalResp'] / ds_winter_sites['TotalResp'].sel(sim='b2') * 100
        ds_site_efd_winter = ds_site_efd_winter / ds_winter_site_deltas['10cm_SoilTemp']
        ds_winter_site_mean = ds_winter_sites.mean(dim='site', skipna=True)
        ds_winter_site_mean_deltas = ds_winter_site_mean.sel(sim='sf') - ds_winter_site_mean.sel(sim='b2')
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('summer deltas', file=pf)
            print(ds_summer_site_deltas, file=pf)
        # fix grids for previous plots
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('nef before mean', file=pf)
            print(ds_site_efd_summer, file=pf)
            print('ds summer site clean', file=pf)
            print(ds_summer_site_clean, file=pf)
        ds_summer_site_nef_time = ds_site_efd_summer.mean(dim=['gridcell'], skipna=True)
        ds_summer_site_nef = ds_site_efd_summer.mean(dim=['gridcell','year'], skipna=True).to_dataset(name='TotalResp')
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('nef after mean', file=pf)
            print(ds_summer_site_nef, file=pf)
        ds_summer_site_nef['TotalResp_std'] = ds_site_efd_summer.std(dim=['gridcell','year'], skipna=True)
        ds_summer_site_nef['SoilN_b2_modmean'] = ds_summer_site_clean['SoilN'].sel(sim='b2').mean(dim=['gridcell','year','site'], skipna=True)
        ds_summer_site_nef['SoilC_b2_modmean'] = ds_summer_site_clean['SoilC'].sel(sim='b2').mean(dim=['gridcell','year','site'], skipna=True)
        ds_summer_site_nef['CN_b2_modmean'] = ds_summer_site_clean['CN'].sel(sim='b2').mean(dim=['gridcell','year','site'], skipna=True)
        ds_summer_site_nef['SoilN_b2'] = ds_summer_site_clean['SoilN'].sel(sim='b2').mean(dim=['gridcell','year'], skipna=True) / ds_summer_site_nef['SoilN_b2_modmean']
        ds_summer_site_nef['SoilC_b2'] = ds_summer_site_clean['SoilC'].sel(sim='b2').mean(dim=['gridcell','year'], skipna=True) / ds_summer_site_nef['SoilC_b2_modmean']
        ds_summer_site_nef['CN_b2'] = ds_summer_site_clean['CN'].sel(sim='b2').mean(dim=['gridcell','year'], skipna=True) / ds_summer_site_nef['CN_b2_modmean']
        ds_summer_site_nef['SoilN_b2_temp'] = ds_summer_site_clean['SoilN'].sel(sim='b2') / ds_summer_site_nef['SoilN_b2_modmean']
        ds_summer_site_nef['SoilC_b2_temp'] = ds_summer_site_clean['SoilC'].sel(sim='b2') / ds_summer_site_nef['SoilC_b2_modmean']
        ds_summer_site_nef['CN_b2_temp'] = ds_summer_site_clean['CN'].sel(sim='b2') / ds_summer_site_nef['CN_b2_modmean']
        ds_summer_site_nef['SoilN_b2_std'] = ds_summer_site_nef['SoilN_b2_temp'].std(dim=['gridcell','year'], skipna=True)
        ds_summer_site_nef['SoilC_b2_std'] = ds_summer_site_nef['SoilC_b2_temp'].std(dim=['gridcell','year'], skipna=True)
        ds_summer_site_nef['CN_b2_std'] = ds_summer_site_nef['CN_b2_temp'].std(dim=['gridcell','year'], skipna=True)
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('nef after adding other variables', file=pf)
            print(ds_summer_site_nef, file=pf)
        ds_summer_sites = ds_summer_sites.mean(dim='gridcell')
        ds_summer_site_mean = ds_summer_site_mean.mean(dim='gridcell')
        ds_summer_site_deltas = ds_summer_site_deltas.mean(dim='gridcell')
        ds_summer_site_deltas_all = ds_summer_site_deltas_all.mean(dim='gridcell')
        ds_summer_site_mean_deltas = ds_summer_site_mean_deltas.mean(dim='gridcell')
        ds_winter_sites = ds_winter_sites.mean(dim='gridcell')
        ds_winter_site_mean = ds_winter_site_mean.mean(dim='gridcell')
        ds_winter_site_deltas = ds_winter_site_deltas.mean(dim='gridcell')
        ds_winter_site_mean_deltas = ds_winter_site_mean_deltas.mean(dim='gridcell')
        # # sort models in alphabetical order (for somereaons ecosys is last, looks like capitals sort first...)
        # # having to create a list in the order I want to sort properly
        # desired_order = ["CLASSIC","CLM5","CLM5-ExIce","ecosys","ELM1-ECA","ELM2-NGEE","JSBACH","LPJ-GUESS-ML","LPJ-GUESS","ORCHIDEE-MICT","ORCHIDEE-MICT-teb","UVic-ESCM"]
        # ds_order = xr.DataArray(data=range(0,len(desired_order)), dims=['model'], coords={'model': desired_order})
        # ds_summer = ds_summer.sortby(ds_order)
        # ds_summer_deltas = ds_summer_deltas.sortby(ds_order)
        # ds_summer_effectsize = ds_summer_effectsize.sortby(ds_order)
        # ds_summer_site_all = ds_summer_site_all.sortby(ds_order)
        # ds_summer_site_hmsd = ds_summer_site_hsmd.sortby(ds_order)
        # ds_summer_site_deltas = ds_summer_site_deltas.sortby(ds_order)
        # ds_summer_site_mean_deltas = ds_summer_site_mean_deltas.sortby(ds_order)
        # ds_summer_geomean_cavm_deltas = ds_summer_geomean_cavm_deltas.sortby(ds_order)
        # ds_summer_geomean_deltas = ds_summer_geomean_deltas.sortby(ds_order)
        # ds_efd_summer = ds_efd_summer.sortby(ds_order)
        # ds_site_efd_summer = ds_site_efd_summer.sortby(ds_order)
        # ds_winter = ds_winter.sortby(ds_order)
        # ds_winter_deltas = ds_winter_deltas.sortby(ds_order)
        # ds_winter_effectsize = ds_winter_effectsize.sortby(ds_order)
        # ds_winter_site_all = ds_winter_site_all.sortby(ds_order)
        # ds_winter_site_deltas = ds_winter_site_deltas.sortby(ds_order)
        # ds_winter_site_mean_deltas = ds_winter_site_mean_deltas.sortby(ds_order)
        # ds_winter_geomean_cavm_deltas = ds_winter_geomean_cavm_deltas.sortby(ds_order)
        # ds_winter_geomean_deltas = ds_winter_geomean_deltas.sortby(ds_order)
        # ds_efd_winter = ds_efd_winter.sortby(ds_order)
        #### Summer ##########################
        # graph delta 10cm soil temp by site/model through time
        cb_pal = ["#000000","#004949","#009292","#ff6db6","#ffb6db","#490092","#006ddb","#b66dff","#6db6ff","#b6dbff","#920000","#924900","#db6d00","#24ff24","#ffff6d"]
        matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=cb_pal)
        y_lab = 'Delta Mean Summer 10cm Soil Temperature (°C)' 
        x_lab = 'Year'
        cmap_WhRd = matplotlib.colors.LinearSegmentedColormap.from_list('custom_RdBu', ['#D3D3D3','#F5A886','#CF5246','#AB162A','#7f0000'], N=256)
        def modify_legend(axes, **kwargs):
            oldleg = axes.get_legend()
            props = dict(
                handles=oldleg.legend_handles,
                labels=[t.get_text() for t in oldleg.texts],
                title=oldleg.get_title().get_text()
            )
            props.update(kwargs) # kwargs takes precedence over props
            axes.legend(**props)
            return 
        # eval plots of each variable
        for var in var_list:
            for mod in ds_summer.model.values:
                output_file = 'figures/' + mod + '/' + mod + '_' + var + '_summer_cum.png'
                with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
                    print('ds_summer variable and mode subset', file=pf)
                    print(ds_summer[var].sel(model=mod), file=pf)
                fig_eval = plt.figure(figsize=(8,6))
                p = ds_summer[var].sel(model=mod, year=2010).plot(x='lon', y='lat', col='sim', col_wrap=3, robust=True, transform=ccrs.PlateCarree(), \
                    subplot_kws={'projection': ccrs.Orthographic(0,90)}, cbar_kwargs={'label': var}, \
                    cmap='coolwarm') #'RdBu_r')
                for ax in p.axs.flat:
                    ax.coastlines()
                    ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
                    add_circle_boundary(ax)
                plt.savefig(Path(config['output_dir'] + output_file), dpi=300)
                plt.close(fig_eval)
                #
            output_file = 'figures/ensemble_summer_' + var + '.png'
            fig_eval = plt.figure(figsize=(8,6))
            p = ds_summer[var].sel(year=2010, sim='b2').plot(x='lon', y='lat', col='model', col_wrap=4, robust=True, transform=ccrs.PlateCarree(), \
                subplot_kws={'projection': ccrs.Orthographic(0,90)}, cbar_kwargs={'label': var}, \
                cmap='coolwarm') #'RdBu_r')
            for ax in p.axs.flat:
                ax.coastlines()
                ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
                add_circle_boundary(ax)
            plt.savefig(Path(config['output_dir'] + output_file), dpi=300)
            plt.close(fig_eval)
        # ER - summer
        fig_effectsize = plt.figure(figsize=(8,6))
        er_tmp = ds_summer_effectsize['TotalResp'] * 100
        er_model_mean = er_tmp.mean(dim='model', skipna=True)
        er_tmp.plot(x='year', hue='model')# col='model', col_wrap=3, hue='site')
        plt.plot(er_tmp.year, er_model_mean, linestyle='--', color='black', label='ensemble')
        plt.title('', fontsize=20)
        y_lab = 'Summer ER Effectsize (%)' 
        x_lab = 'Year' 
        plt.xlabel(x_lab, fontsize=16)
        plt.ylabel(y_lab, fontsize=16)
        plt.ylim((-20, 175))
        plt.xticks(np.arange(2000,2022,5))
        plt.savefig(Path(config['output_dir'] + 'figures/ER_OTC_effectsize_by_model.png'), dpi=300)
        plt.close(fig_effectsize)
        # ER - winter
        fig_effectsize = plt.figure(figsize=(8,6))
        er_tmp = ds_winter_effectsize['TotalResp'] * 100
        er_model_mean = er_tmp.mean(dim='model', skipna=True)
        er_tmp.plot(x='year', hue='model')# col='model', col_wrap=3, hue='site')
        plt.plot(er_tmp.year, er_model_mean, linestyle='--', color='black', label='ensemble')
        plt.title('', fontsize=20)
        y_lab = 'Winter ER Effectsize (%)' 
        x_lab = 'Year' 
        plt.xlabel(x_lab, fontsize=16)
        plt.ylabel(y_lab, fontsize=16)
        plt.ylim((-20, 175))
        plt.xticks(np.arange(2000,2022,5))
        plt.savefig(Path(config['output_dir'] + 'figures/ER_SF_effectsize_by_model.png'), dpi=300)
        plt.close(fig_effectsize)
        # GPP - summer
        fig_effectsize = plt.figure(figsize=(8,6))
        gpp_tmp = ds_summer_effectsize['GPP'] * 100
        gpp_model_mean = gpp_tmp.mean(dim='model', skipna=True)
        gpp_tmp.plot(x='year', hue='model')# col='model', col_wrap=3, hue='site')
        plt.plot(gpp_tmp.year, gpp_model_mean, linestyle='--', color='black', label='ensemble')
        plt.title('', fontsize=20)
        y_lab = 'Summer GPP Effectsize (%)' 
        x_lab = 'Year' 
        plt.xlabel(x_lab, fontsize=16)
        plt.ylabel(y_lab, fontsize=16)
        plt.ylim((-20, 175))
        plt.xticks(np.arange(2000,2022,5))
        plt.savefig(Path(config['output_dir'] + 'figures/GPP_OTC_effectsize_by_model.png'), dpi=300)
        plt.close(fig_effectsize)
        # GPP - winter
        fig_effectsize = plt.figure(figsize=(8,6))
        gpp_tmp = ds_winter_effectsize['GPP'] * 100
        gpp_model_mean = gpp_tmp.mean(dim='model', skipna=True)
        gpp_tmp.plot(x='year', hue='model')# col='model', col_wrap=3, hue='site')
        plt.plot(gpp_tmp.year, gpp_model_mean, linestyle='--', color='black', label='ensemble')
        plt.title('', fontsize=20)
        y_lab = 'Winter GPP Effectsize (%)' 
        x_lab = 'Year' 
        plt.xlabel(x_lab, fontsize=16)
        plt.ylabel(y_lab, fontsize=16)
        plt.ylim((-20, 175))
        plt.xticks(np.arange(2000,2022,5))
        plt.savefig(Path(config['output_dir'] + 'figures/GPP_SF_effectsize_by_model.png'), dpi=300)
        plt.close(fig_effectsize)
        # graph site mean delta 10cm soil temp by model
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('hedge smd before graph:', file=pf)
            print(ds_summer_site_hsmd, file=pf)
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_summer_site_hsmd.plot.scatter(x='year', y='TotalResp', col='model', col_wrap=4, hue='site')
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_hsmd_by_site_through_time_scatter.png'), dpi=300)
        plt.close(fig_effectsize)
        #
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_summer_site_hsmd['TotalResp'].plot(x='year', col='model', col_wrap=4, hue='site')
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_hsmd_by_site_through_time_line.png'), dpi=300)
        plt.close(fig_effectsize)
        #
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_summer_site_nef_time.plot.scatter(x='year', y='TotalResp', col='model', col_wrap=4, hue='site', ylim=(-50,150))
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_nef_by_site_through_time_scatter.png'), dpi=300)
        plt.close(fig_effectsize)
        #
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_summer_site_nef_time.plot(x='year', col='model', col_wrap=4, hue='site')
        plt.ylim(-50,150)
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_nef_by_site_through_time_line.png'), dpi=300)
        plt.close(fig_effectsize)
        #
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_summer_site_hsmd.plot.scatter(x='year', y='TotalResp', col='model', col_wrap=4, hue='site')
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_hsmd_by_time.png'), dpi=300)
        plt.close(fig_effectsize)
        #
        fig_effectsize = plt.figure(figsize=(10,6))
        model_list = ['CLASSIC','ELM1-ECA','ELM2-NGEE','LPJ-GUESS-ML']
        p = ds_summer_site_hsmd.sel(model=model_list).plot.scatter(x='SoilN_b2', y='TotalResp', col='model', col_wrap=2, hue='site', sharex=False)
        for ax in p.axs.flat:
            try:
                x_data = ax.collections[0].get_offsets().data[:, 0]    
                y_data = ax.collections[0].get_offsets().data[:, 1]    
                npmask = np.isfinite(x_data) & np.isfinite(y_data)
                ax.set_xlim(0, max(x_data[npmask])+ 0.1*max(x_data[npmask]))
                m1, b1 = np.polyfit(x_data[npmask], y_data[npmask], 1)
                ax.plot(x_data, m1*x_data + b1, 'r')
            except:
                pass
        p.set_xlabels('Soil Nitrogen (gN/m2)')
        plt.subplots_adjust(wspace=0.1, right=0.8)
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_hsmd_by_soiln.png'), dpi=300)
        plt.close(fig_effectsize)
        #
        fig_effectsize = plt.figure(figsize=(8,6))
        model_list = ['CLASSIC','ELM1-ECA','ELM2-NGEE','LPJ-GUESS-ML']
        p = ds_summer_site_nef.sel(model=model_list).plot.scatter(x='SoilN_b2', y='TotalResp', col='model', col_wrap=2, hue='site')
        plt.xlim(0,3)
        for ax in p.axs.flat:
            try:
                with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
                    print('axes:', file=pf)
                    print(ax.get_title(), file=pf)
                x_data = ax.collections[0].get_offsets().data[:, 0]    
                y_data = ax.collections[0].get_offsets().data[:, 1]    
                npmask = np.isfinite(x_data) & np.isfinite(y_data)
                ax.set_xlim(0, max(x_data[npmask])+ 0.1*max(x_data[npmask]))
                m1, b1 = np.polyfit(x_data[npmask], y_data[npmask], 1)
                ax.plot(x_data, m1*x_data + b1, 'r')
            except:
                pass
        p.set_xlabels('Mean Centered Soil Nitrogen')
        p.set_ylabels('Normalized Effect Size (%/°C)')
        plt.subplots_adjust(wspace=0.1, right=0.8)
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_nef_by_soiln.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_effectsize)
        #
        fig_effectsize = plt.figure(figsize=(8,6))
        model_list = ['CLASSIC','ELM1-ECA','ELM2-NGEE','LPJ-GUESS-ML','JULES']
        #ds_summer_site_nef.to_netcdf(Path(config['output_dir'] + 'figures/ds_summer_site_nef.nc'), mode="w", format='NETCDF4_CLASSIC', engine='netcdf4')
        p = ds_summer_site_nef.sel(model=model_list).plot.scatter(x='SoilN_b2', y='TotalResp', col='model', col_wrap=2, hue='site')
        plt.xlim(0,2.5)
        plt.ylim(-30,150)
        mod_iter = iter(model_list)
        for ax in p.axs.flat:
            try:
                with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
                    print('axes:', file=pf)
                    print(ax.get_title(), file=pf)
                x_data = ax.collections[0].get_offsets().data[:, 0]    
                y_data = ax.collections[0].get_offsets().data[:, 1]    
                npmask = np.isfinite(x_data) & np.isfinite(y_data)
                #ax.set_xlim(0, max(x_data[npmask])+ 0.3*max(x_data[npmask]))
                viridis_cmap = plt.get_cmap('viridis')
                values = np.linspace(0,1,14)
                vmap_iter = iter([viridis_cmap(value) for value in values])
                with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
                    print('cmap:', file=pf)
                    print(viridis_cmap, file=pf)
                    print(vmap_iter, file=pf)
                for site in ds_summer_site_nef.site.values:
                    model_name = next(mod_iter)
                    site_color = next(vmap_iter)
                    x_val = ds_summer_site_nef['SoilN_b2'].sel(model=model_name, site=site)
                    y_val = ds_summer_site_nef['TotalResp'].sel(model=model_name, site=site) 
                    if not np.isnan(x_val) and not np.isnan(y_val): 
                        ax.errorbar(
                            x=x_val, \
                            y=y_val, \
                            ecolor=site_color, fmt='none', \
                            xerr=ds_summer_site_nef['SoilN_b2_std'].sel(model=model_name, site=site), \
                            yerr=ds_summer_site_nef['TotalResp_std'].sel(model=model_name, site=site))
                m1, b1 = np.polyfit(x_data[npmask], y_data[npmask], 1)
                ax.plot(x_data, m1*x_data + b1, 'r')
            except Exception as error:
                with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
                    print('error:', file=pf)
                    print(error, file=pf)
                pass
        p.set_xlabels('Mean Centered Soil Nitrogen')
        p.set_ylabels('Normalized Effect Size (%/°C)')
        plt.subplots_adjust(wspace=0.1, right=0.8)
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_nef_by_soiln_errorbar.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_effectsize)
        #
        fig_effectsize = plt.figure(figsize=(8,6))
        p = ds_summer_site_hsmd.plot.scatter(x='SoilC_b2', y='TotalResp', col='model', col_wrap=4, hue='site', sharex=False)
        for ax in p.axs.flat:
            try:
                x_data = ax.collections[0].get_offsets().data[:, 0]    
                y_data = ax.collections[0].get_offsets().data[:, 1]    
                npmask = np.isfinite(x_data) & np.isfinite(y_data)
                #ax.set_xlim(0, max(x_data[npmask])+ 0.1*max(x_data[npmask]))
                m1, b1 = np.polyfit(x_data[npmask], y_data[npmask], 1)
                ax.plot(x_data, m1*x_data + b1, 'r')
            except:
                pass
        p.set_xlabels('Soil Carbon (gC/m2)')
        plt.subplots_adjust(wspace=0.1, right=0.8)
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_hsmd_by_soilc.png'), dpi=300)
        plt.close(fig_effectsize)
        #
        fig_effectsize = plt.figure(figsize=(8,6))
        p = ds_summer_site_hsmd.sel(model=model_list).plot.scatter(x='CN_b2', y='TotalResp', col='model', col_wrap=2, hue='site')
        for ax in p.axs.flat:
            try:
                x_data = ax.collections[0].get_offsets().data[:, 0]    
                y_data = ax.collections[0].get_offsets().data[:, 1]    
                npmask = np.isfinite(x_data) & np.isfinite(y_data)
                m1, b1 = np.polyfit(x_data[npmask], y_data[npmask], 1)
                ax.plot(x_data, m1*x_data + b1, 'r')
            except:
                pass
        p.set_xlabels('Soil CN')
        plt.subplots_adjust(wspace=0.1, right=0.8)
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_hsmd_by_cn.png'), dpi=300)
        plt.close(fig_effectsize)
        #matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=cb_pal)
        y_lab = 'Delta Mean Summer 10cm Soil Temperature (°C)' 
        x_lab = 'Year' 
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_summer_site_deltas['10cm_SoilTemp'].plot(x='year', col='model', col_wrap=4, hue='site')
        #plt.title('Mean Summer Warming', fontsize=20)
        #plt.xlabel(x_lab, fontsize=16)
        #plt.ylabel(y_lab, fontsize=16)
        plt.savefig(Path(config['output_dir'] + 'figures/10cm_SoilTemp_OTC_warming_effect_by_site_through_time.png'), dpi=300)
        plt.close(fig_effectsize)
        #
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_summer_site_deltas_all['10cm_SoilTemp'].plot(x='year', col='model', col_wrap=4, hue='site')
        plt.savefig(Path(config['output_dir'] + 'figures/10cm_SoilTemp_OTC_warming_effect_by_site_through_time_all.png'), dpi=300)
        plt.close(fig_effectsize)
        #
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_summer_site_mean_deltas['10cm_SoilTemp'].plot(x='year', hue='model')
        #plt.title('Tundra Sites Mean Summer Warming', fontsize=20)
        plt.xlabel(x_lab, fontsize=16)
        plt.ylabel(y_lab, fontsize=16)
        plt.ylim((-1, 4))
        plt.savefig(Path(config['output_dir'] + 'figures/10cm_SoilTemp_OTC_warming_effect_by_sitemeans_through_time.png'), dpi=300)
        plt.close(fig_effectsize)
        # graph cavm mean delta 10cm soil temp by model
        #matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=cb_pal)
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_summer_geomean_cavm_deltas['10cm_SoilTemp'].plot(x='year', hue='model')
        #plt.title('CAVM Mean Summer Warming', fontsize=20)
        plt.xlabel(x_lab, fontsize=16)
        plt.ylabel(y_lab, fontsize=16)
        plt.ylim((-1, 4))
        plt.savefig(Path(config['output_dir'] + 'figures/10cm_SoilTemp_OTC_warming_effect_by_cavm_means_through_time.png'), dpi=300)
        plt.close(fig_effectsize)
        # graph total simulation delta
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_summer_geomean_deltas['10cm_SoilTemp'].plot(x='year', hue='model')
        #plt.title('55N Regional Mean Summer Warming', fontsize=20)
        plt.xlabel(x_lab, fontsize=16)
        plt.ylabel(y_lab, fontsize=16)
        plt.ylim((-1, 4))
        plt.savefig(Path(config['output_dir'] + 'figures/10cm_SoilTemp_OTC_warming_effect_55N_through_time.png'), dpi=300)
        plt.close(fig_effectsize)
        # graph ER effect by soil delta increase
        fig_effectsize = plt.figure(figsize=(8,6))
        er_tmp = ds_summer_effectsize.copy(deep=True)
        er_tmp['TotalResp'] = er_tmp['TotalResp'] * 100
        er_tmp.drop_sel(model='JSBACH').plot.scatter(x='10cm_SoilTemp', y='TotalResp', hue='model')
        plt.title('', fontsize=20)
        #plt.title('CAVM Delta ER by Delta OTC warming', fontsize=20)
        plt.xlabel('Delta Mean Summer 10cm Soil Temperature (°C)', fontsize=16)
        plt.ylabel('Summer CAVM ER Effect Size (%)', fontsize=16)
        #plt.axline((0,0), slope=1, linestyle='--', c='black', zorder=-100)
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_Delta-delta_ER_by_soiltemp.png'), dpi=300)
        plt.close(fig_effectsize)
        # graph ER effect by soil delta increase
        fig_effectsize = plt.figure(figsize=(8,6))
        er_tmp = ds_winter_effectsize.copy(deep=True)
        er_tmp['TotalResp'] = er_tmp['TotalResp'] * 100
        er_tmp.plot.scatter(x='10cm_SoilTemp', y='TotalResp', hue='model')
        plt.title('', fontsize=20)
        #plt.title('CAVM Delta ER by Delta SF warming', fontsize=20)
        plt.xlabel('Delta Mean Winter 10cm Soil Temperature (°C)', fontsize=16)
        plt.ylabel('Winter CAVM ER Effect Size (%)', fontsize=16)
        #plt.axline((0,0), slope=1, linestyle='--', c='black', zorder=-100)
        plt.savefig(Path(config['output_dir'] + 'figures/SF_Delta-delta_ER_by_soiltemp.png'), dpi=300)
        plt.close(fig_effectsize)
        # graph the gridded responses
        cavm_outline = gpd.read_file('/projects/warpmip/shared/cavm_shapefiles/physiog_la.shp')
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('cavm projection:', file=pf)
            print(cavm_outline.crs, file=pf)
        cavm_outline['geometry'] = cavm_outline['geometry'].rotate(180, origin=(90,-180))
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('cavm geodataset:', file=pf)
            print(cavm_outline, file=pf)
            print('cavm columns:', file=pf)
            print(cavm_outline.columns.to_list(), file=pf)
        cavm_union = cavm_outline.dissolve(by='physiog')
        cavm_outer = cavm_union.geometry.iloc[0]
        with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            print('cavm after dissolve:', file=pf)
            print(cavm_union, file=pf)
            print('cavm outer:', file=pf)
            print(cavm_outer, file=pf)
        fig_effectsize = plt.figure(figsize=(10,10))
        p = ds_summer_deltas['10cm_SoilTemp'].sel(year=2010).plot(col='model', col_wrap=4, robust=True, transform=ccrs.PlateCarree(), \
                subplot_kws={'projection': ccrs.Orthographic(0,90)}, cbar_kwargs={'label': 'Delta Mean Summer Soil Temperature (°C)'}, \
                cmap='coolwarm') #'RdBu_r')
        for ax in p.axs.flat:
            ax.coastlines()
            ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
            add_circle_boundary(ax)
            cavm_outline.plot(ax=ax, edgecolor="purple", facecolor="none")
            cavm_union.plot(ax=ax, edgecolor='yellow', facecolor='none')
        plt.subplots_adjust(wspace=0.1, right=0.8)
        #plt.axline((0,0), slope=1, linestyle='--', c='black', zorder=-100)
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_55N_year2010_10cm_soiltemp_by_model.png'), dpi=300)
        plt.close(fig_effectsize)
        # graph the gridded responses
        fig_effectsize = plt.figure(figsize=(10,10))
        p = ds_efd_summer['TotalResp_efd'].sel(year=2010).plot(col='model', col_wrap=4, robust=True, transform=ccrs.PlateCarree(), \
                subplot_kws={'projection': ccrs.Orthographic(0,90)}, cbar_kwargs={'label': 'ER Effect Size Delta(%/°C)'}, \
                cmap='coolwarm')#'RdBu_r')
        for ax in p.axs.flat:
            ax.coastlines()
            ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
            add_circle_boundary(ax)
        plt.subplots_adjust(wspace=0.1, right=0.8)
        #plt.axline((0,0), slope=1, linestyle='--', c='black', zorder=-100)
        plt.savefig(Path(config['output_dir'] + 'figures/OTC_55N_year2010_ER_effect_size_delta_by_model.png'), dpi=300)
        plt.close(fig_effectsize)
        #### winter ##########################
        # graph delta 10cm soil temp by site/model through time
        cb_pal = ["#000000","#004949","#009292","#ff6db6","#ffb6db","#490092","#006ddb","#b66dff","#6db6ff","#b6dbff","#920000","#924900","#db6d00","#24ff24","#ffff6d"]
        matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=cb_pal)
        y_lab = 'Delta Mean Winter 10cm Soil Temperature (°C)' 
        x_lab = 'Year' 
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_winter_site_deltas['10cm_SoilTemp'].plot(x='year', col='model', col_wrap=4, hue='site')
        #plt.title('Mean Winter Warming', fontsize=20)
        #plt.xlabel(x_lab, fontsize=16)
        #plt.ylabel(y_lab, fontsize=16)
        plt.savefig(Path(config['output_dir'] + 'figures/10cm_SoilTemp_SF_warming_effect_by_site_through_time.png'), dpi=300)
        plt.close(fig_effectsize)
        # graph site mean delta 10cm soil temp by model
        #matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=cb_pal)
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_winter_site_mean_deltas['10cm_SoilTemp'].plot(x='year', hue='model')
        #plt.title('Tundra Sites Mean Winter Warming', fontsize=20)
        plt.xlabel(x_lab, fontsize=16)
        plt.ylabel(y_lab, fontsize=16)
        plt.ylim((-1, 4))
        plt.savefig(Path(config['output_dir'] + 'figures/10cm_SoilTemp_SF_warming_effect_by_sitemeans_through_time.png'), dpi=300)
        plt.close(fig_effectsize)
        # graph cavm mean delta 10cm soil temp by model
        #matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=cb_pal)
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_winter_geomean_cavm_deltas['10cm_SoilTemp'].plot(x='year', hue='model')
        #plt.title('CAVM Mean Winter Warming', fontsize=20)
        plt.xlabel(x_lab, fontsize=16)
        plt.ylabel(y_lab, fontsize=16)
        plt.ylim((-1, 4))
        plt.savefig(Path(config['output_dir'] + 'figures/10cm_SoilTemp_SF_warming_effect_by_cavm_means_through_time.png'), dpi=300)
        plt.close(fig_effectsize)
        # graph total simulation delta
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_winter_geomean_deltas['10cm_SoilTemp'].plot(x='year', hue='model')
        #plt.title('55N Regional Mean Winter Warming', fontsize=20)
        plt.xlabel(x_lab, fontsize=16)
        plt.ylabel(y_lab, fontsize=16)
        plt.ylim((-1, 4))
        plt.savefig(Path(config['output_dir'] + 'figures/10cm_SoilTemp_SF_warming_effect_55N_through_time.png'), dpi=300)
        plt.close(fig_effectsize)
        # graph the gridded responses
        fig_effectsize = plt.figure(figsize=(10,8))
        p = ds_winter_deltas['10cm_SoilTemp'].sel(year=2010).plot(col='model', col_wrap=4, robust=True, transform=ccrs.PlateCarree(), \
                subplot_kws={'projection': ccrs.Orthographic(0,90)}, cbar_kwargs={'label': 'Delta Winter Soil Temperature (°C)'}, \
                cmap=cmap_WhRd, vmin=0, vmax=7)
        for ax in p.axs.flat:
            ax.coastlines()
            ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
            add_circle_boundary(ax)
        plt.subplots_adjust(wspace=0.1, right=0.8)
        #plt.axline((0,0), slope=1, linestyle='--', c='black', zorder=-100)
        plt.savefig(Path(config['output_dir'] + 'figures/SF_55N_year2010_10cm_soiltemp_by_model.png'), dpi=300)
        plt.close(fig_effectsize)
        # effectsize delta
        fig_effectsize = plt.figure(figsize=(10,10))
        p = ds_efd_winter['TotalResp_efd'].sel(year=2010).plot(col='model', col_wrap=4, robust=True, transform=ccrs.PlateCarree(), \
                subplot_kws={'projection': ccrs.Orthographic(0,90)}, cbar_kwargs={'label': 'ER Effect Size Delta(%/°C)'}, \
                cmap='coolwarm')#'RdBu_r')
        for ax in p.axs.flat:
            ax.coastlines()
            ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
            add_circle_boundary(ax)
        plt.subplots_adjust(wspace=0.1, right=0.8)
        #plt.axline((0,0), slope=1, linestyle='--', c='black', zorder=-100)
        plt.savefig(Path(config['output_dir'] + 'figures/SF_55N_year2010_ER_effect_size_delta_by_model.png'), dpi=300)
        plt.close(fig_effectsize)
        # #######
        # ## effect size normed by temp delta
        # #######
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_site_efd_summer.mean(dim=['gridcell','site']).plot(x='year', hue='model')
        ds_site_efd_summer.mean(dim=['gridcell','site','model']).plot(x='year', color='black', linestyle='--', label='Ensemble')
        plt.title('Site-based Normalized Effect Size', fontsize=20)
        plt.xticks(np.arange(2000,2022,5))
        plt.xlabel('Year', fontsize=16)
        plt.ylabel('Normalized Effect Size (%/°C)', fontsize=16)
        plt.savefig(Path(config['output_dir'] + 'figures/Sites_eft_by_model.png'), dpi=300)
        plt.close(fig_effectsize)
        
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_site_efd_summer.mean(dim='gridcell').plot(x='year', col='model', col_wrap=4, hue='site')
        #plt.title('Mean Winter Warming', fontsize=20)
        #plt.xlabel(x_lab, fontsize=16)
        plt.ylim((-60, 60))
        #plt.ylabel(y_lab, fontsize=16)
        plt.savefig(Path(config['output_dir'] + 'figures/Sites_eft_by_model_2.png'), dpi=300)
        plt.close(fig_effectsize)
        # ds_sites_ef_numerator = ds_summer_site_grids['TotalResp'].sel(sim='otc') - ds_summer_site_grids['TotalResp'].sel(sim='b2')
        # ds_sites_ef_denominator = ds_summer_site_grids['TotalResp'].sel(sim='b2', drop=True)
        # ds_sites_c = ds_summer_site_grids['SoilC'].sel(sim='b2', drop=True)
        # ds_sites_n = ds_summer_site_grids['SoilN'].sel(sim='b2', drop=True)
        # ds_sites_cn = ds_summer_site_grids['CN'].sel(sim='b2', drop=True)
        # 
        # ds_sites_efd_numerator = (ds_sites_ef_numerator / ds_sites_ef_denominator) * 100
        # ds_sites_efd_denominator =  ds_summer_site_grids['10cm_SoilTemp'].sel(sim='otc') - ds_summer_site_grids['10cm_SoilTemp'].sel(sim='b2')
        # ds_sites_efd_denominator = ds_sites_efd_denominator.where(ds_sites_efd_denominator > 0.05)
        # ds_sites_efd_final = ds_sites_efd_numerator / ds_sites_efd_denominator 
        # 
        # ds_out = ds_sites_efd_final.to_dataset(name='efd')
        # ds_out['SoilC'] = ds_sites_c
        # ds_out['SoilN'] = ds_sites_n
        # ds_out['CN'] = ds_sites_cn
        # ds_out.to_netcdf(Path(config['output_dir'] + 'figures/ds_grids.nc'), mode="w", format='NETCDF4_CLASSIC', engine='netcdf4')
        
        #for mod in ds_site_efd_summer['model'].values:
            ##### test 1 loops with xarray plot
            # fig_ef, axs = plt.subplots(nrows=4,ncols=4)
            # for ax, site in zip(axs.ravel(), ds_sites_efd_final.site.values):
            #     ds_sites_efd_final.sel(model=mod, site=site).plot(ax=ax, x='year', hue='gridcell')
            #     ax.get_legend().set_visible(False)
            #     ax.set_title(site, fontsize=20)
            #     ax.set_xlabel('', fontsize=16)
            #     ax.set_ylim((-60, 60))
            # plt.tight_layout()
            ##### test 2 seaborn
            # efd_df = ds_sites_efd_final.sel(model=mod).to_dataframe(name='efd')
            # efd_df = efd_df.reset_index()
            # with open(Path(config['output_dir'] + 'figures/debug_soiltemp_graphs.txt'), 'a') as pf:
            #     print('efd to dataframe', file=pf)
            #     print(efd_df, file=pf)
            # efd_df.to_csv(Path(config['output_dir'] + 'figures/efd_to_dataframe.csv'))
            # g = sns.FacetGrid(data=efd_df, col='site', col_wrap=4, hue='gridcell')
            # g.map_dataframe(sns.lineplot, 'year', 'efd')
            ###### xarray plots 
            # fig_effectsize = plt.figure(figsize=(8,6))
            # ds_sites_efd_final.sel(model=mod).plot(x='year', col='site', col_wrap=4, hue='gridcell')
            # plt.ylim((-60, 60))
            # plt.savefig(Path(config['output_dir'] + 'figures/site_efd_eval_' + str(mod) + '.png'), dpi=300)
            # plt.close(g.fig)
            # 
            # fig_effectsize = plt.figure(figsize=(8,6))
            # ds_sites_ef_numerator.sel(model=mod).plot(x='year', col='site', col_wrap=4, hue='gridcell')
            # #plt.title('Mean Winter Warming', fontsize=20)
            # #plt.xlabel(x_lab, fontsize=16)
            # plt.ylim((-100, 400))
            # #plt.ylabel(y_lab, fontsize=16)
            # plt.savefig(Path(config['output_dir'] + 'figures/site_delta_flux_eval_' + str(mod) + '.png'), dpi=300)
            # plt.close(fig_effectsize)
            # 
            # fig_effectsize = plt.figure(figsize=(8,6))
            # ds_sites_ef_denominator.sel(model=mod).plot(x='year', col='site', col_wrap=4, hue='gridcell')
            # #plt.title('Mean Winter Warming', fontsize=20)
            # #plt.xlabel(x_lab, fontsize=16)
            # plt.ylim((-100, 400))
            # #plt.ylabel(y_lab, fontsize=16)
            # plt.savefig(Path(config['output_dir'] + 'figures/site_baseline_flux_eval_' + str(mod) + '.png'), dpi=300)
            # plt.close(fig_effectsize)
        
            # fig_effectsize = plt.figure(figsize=(8,6))
            # ds_sites_efd_numerator.sel(model=mod).plot(x='year', col='site', col_wrap=4, hue='gridcell')
            # #plt.title('Mean Winter Warming', fontsize=20)
            # #plt.xlabel(x_lab, fontsize=16)
            # plt.ylim((-100, 400))
            # #plt.ylabel(y_lab, fontsize=16)
            # plt.savefig(Path(config['output_dir'] + 'figures/site_ef_eval_' + str(mod) + '.png'), dpi=300)
            # plt.close(fig_effectsize)
            # 
            # fig_effectsize = plt.figure(figsize=(8,6))
            # ds_sites_efd_denominator.sel(model=mod).plot(x='year', col='site', col_wrap=4, hue='gridcell')
            # #plt.title('Mean Winter Warming', fontsize=20)
            # #plt.xlabel(x_lab, fontsize=16)
            # plt.ylim((-100, 400))
            # #plt.ylabel(y_lab, fontsize=16)
            # plt.savefig(Path(config['output_dir'] + 'figures/site_delta_temp_eval_' + str(mod) + '.png'), dpi=300)
            # plt.close(fig_effectsize)
            ##### gridspec / subplotspec
            # fig = plt.figure(figsize=(10, 8))
            # outer = matplotlib.gridspec.GridSpec(4, 4, wspace=0.1, hspace=0.1)
            # sites = iter(ds_sites_efd_final.site.values)
            # for i in range(16):
            #     inner = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.05, hspace=0.1)
            #     for j in range(2):
            #         ax = plt.Subplot(fig, inner[j])
            #         t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (i, j))
            #         t.set_ha('center')
            #         ax.set_xticks([])
            #         ax.set_yticks([])
            #         fig.add_subplot(ax)
            # plt.savefig(Path(config['output_dir'] + 'figures/site_efd_grid_' + str(mod) + '.png'), dpi=300)
            # plt.close(fig_effectsize)

        # #######
        # ## effectsize normalzed by delta soil temp
        # #######
        # ds_summer = ds_summer.mean(dim=['lat','lon'], skipna=True)
        # ds_winter = ds_winter.mean(dim=['lat','lon'], skipna=True)
        # # ER - summer
        # fig_effectsize = plt.figure(figsize=(8,6))
        # er_tmp = ds_efd_summer['TotalResp_efd']
        # er_model_mean = er_tmp.mean(dim='model', skipna=True)
        # er_tmp.plot(x='year', hue='model')# col='model', col_wrap=3, hue='site')
        # plt.plot(er_tmp.year, er_model_mean, linestyle='--', color='black', label='ensemble')
        # plt.title('', fontsize=20)
        # y_lab = 'Summer ER Effectsize Delta (%/°C)' 
        # x_lab = 'Year' 
        # plt.xlabel(x_lab, fontsize=16)
        # plt.ylabel(y_lab, fontsize=16)
        # #plt.ylim((-20, 100))
        # plt.savefig(Path(config['output_dir'] + 'figures/ER_OTC_effectsizei_delta_by_model.png'), dpi=300)
        # plt.close(fig_effectsize)
        # # ER - winter
        # fig_effectsize = plt.figure(figsize=(8,6))
        # er_tmp = ds_efd_winter['TotalResp_efd']
        # er_model_mean = er_tmp.mean(dim='model', skipna=True)
        # er_tmp.plot(x='year', hue='model')# col='model', col_wrap=3, hue='site')
        # plt.plot(er_tmp.year, er_model_mean, linestyle='--', color='black', label='ensemble')
        # plt.title('', fontsize=20)
        # y_lab = 'Winter ER Effectsize Delta (%/°C)' 
        # x_lab = 'Year' 
        # plt.xlabel(x_lab, fontsize=16)
        # plt.ylabel(y_lab, fontsize=16)
        # #plt.ylim((-20, 100))
        # plt.savefig(Path(config['output_dir'] + 'figures/ER_SF_effectsize_delta_by_model.png'), dpi=300)
        # plt.close(fig_effectsize)
        # # GPP - summer
        # fig_effectsize = plt.figure(figsize=(8,6))
        # gpp_tmp = ds_efd_summer['GPP_efd']
        # gpp_model_mean = gpp_tmp.mean(dim='model', skipna=True)
        # gpp_tmp.plot(x='year', hue='model')# col='model', col_wrap=3, hue='site')
        # plt.plot(gpp_tmp.year, gpp_model_mean, linestyle='--', color='black', label='ensemble')
        # plt.title('', fontsize=20)
        # y_lab = 'Summer GPP Effectsize Delta (%/°C)' 
        # x_lab = 'Year' 
        # plt.xlabel(x_lab, fontsize=16)
        # plt.ylabel(y_lab, fontsize=16)
        # #plt.ylim((-20, 100))
        # plt.savefig(Path(config['output_dir'] + 'figures/GPP_OTC_effectsize_delta_by_model.png'), dpi=300)
        # plt.close(fig_effectsize)
        # # GPP - winter
        # fig_effectsize = plt.figure(figsize=(8,6))
        # gpp_tmp = ds_efd_winter['GPP_efd']
        # gpp_model_mean = gpp_tmp.mean(dim='model', skipna=True)
        # gpp_tmp.plot(x='year', hue='model')# col='model', col_wrap=3, hue='site')
        # plt.plot(gpp_tmp.year, gpp_model_mean, linestyle='--', color='black', label='ensemble')
        # plt.title('', fontsize=20)
        # y_lab = 'Winter GPP Effectsize Delta (%/°C)' 
        # x_lab = 'Year' 
        # plt.xlabel(x_lab, fontsize=16)
        # plt.ylabel(y_lab, fontsize=16)
        # #plt.ylim((-20, 100))
        # plt.savefig(Path(config['output_dir'] + 'figures/GPP_SF_effectsize_delta_by_model.png'), dpi=300)
        # plt.close(fig_effectsize)

# Maes graphs
def maes_graphs(input_list):
    # context manager to not change chunk sizes
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # graph type (mean, instantaneous)
        config = input_list[0]
        var_list = input_list[1]
        with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'w') as pf:
            print('graphing started:', file=pf)
        # open zarr file
        zarr_file = config['output_dir'] + 'zarr_output/WrPMIP_Pan-Arctic_models_harmonized.zarr'
        ds = xr.open_zarr(zarr_file, chunks='auto', chunked_array_type='dask', use_cftime=True, mask_and_scale=False) 
        # select soil layer
        with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
            print('data loaded:', file=pf)
            print(ds, file=pf)
        # select variables of interest
        ds = ds[var_list]
        # change to -180/180 coords
        ds['lon'] =('lon', (((ds.lon.values + 180) % 360) - 180))
        # use sortby to enforce monotonically increasing dims for xarray
        ds = ds.sortby(['lon'])
        # select surface soil temp
        #ds = ds.isel(SoilDepth = 0, drop=True)
        with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
            print('subset selected', file=pf)
            print(ds, file=pf)
        # subsample sites
        # create list for ds after gps selection
        ds_list = []
        site_gps = config['site_gps'] 
        for site in site_gps:
            # find middle grid center
            ds_center = ds.sel(lon=site_gps[site]['lon'], lat=site_gps[site]['lat'], method='nearest')
            ds_sub = ds.sel(lon=slice(ds_center['lon']-0.75, ds_center['lon']+0.75), lat=slice(ds_center['lat']-0.75, ds_center['lat']+0.75)).copy()
            with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
                print(ds_center, file=pf)
                print(ds_sub, file=pf)
                print(site, file=pf)
            ds_sub['site_mean'] = ds_sub.mean(dim=['lon','lat'], skipna=True)
            ds_sub['site_std'] = ds_sub.std(dim=['lon','lat'], skipna=True)
            # expand a dimension to include site and save to list
            ds_sub = ds_sub.assign_coords({'site': site})
            ds_sub = ds_sub.expand_dims('site')
            #ds_sub = ds_sub.reset_coords(['lat','lon'])
            #ds_sub['lat'] = ds_sub['lat'].expand_dims('site')
            #ds_sub['lon'] = ds_sub['lon'].expand_dims('site')
            with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
                print(ds_sub, file=pf)
            ds_list.append(ds_sub)
        with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
            print('datasets appended', file=pf)
        # combine site dimension to have multiple sites
        ds_sites = xr.combine_by_coords(ds_list)
        with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
            print('datasets combined', file=pf)
        # # select by CAVM
        # cavm_f = '/projects/warpmip/shared/0.5_cavm.nc' 
        # cavm = xr.open_dataset(cavm_f)
        # ds = ds.where(cavm.cavm > 0).persist()
        # with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #     print('cavm grids selected', file=pf)
        #     print(ds, file=pf)
        # # summer average
        # ds = ds.sel(model=["CLM5","CLM5-ExIce","ELM2-NGEE","UVic-ESCM","ecosys","ELM1-ECA","CLASSIC"])
        # ds = ds.sel(time=slice('2000-01-01','2020-12-31'))
        # ds_annual_summer_cum_geoavg = ds.sel(time=is_summer(ds['time.month'])).groupby('time.year').sum(dim=['time'], skipna=True).mean(dim=['lat','lon'], skipna=True)
        # ds_annual_summer_mean_geoavg = ds.sel(time=is_summer(ds['time.month'])).groupby('time.year').mean(dim=['time','lat','lon'], skipna=True)
        ds_sites = ds_sites.sel(model=["CLM5","CLM5-ExIce","ELM2-NGEE","UVic-ESCM","ecosys","ELM1-ECA","CLASSIC"])
        ds_sites = ds_sites.sel(time=slice('2000-01-01','2020-12-31'))
        var_sub_cum = ['TotalResp','GPP','NEE']
        ds_annual_summer_cum_sites = ds_sites[var_sub_cum].sel(time=is_maes_summer(ds_sites['time.month'])).groupby('time.year').sum(dim=['time'], skipna=True)
        var_sub_mean = ['SoilTemp','ALT','WTD','SoilC','SoilN','CN']
        ds_annual_summer_mean_sites = ds_sites[var_sub_mean].sel(time=is_maes_summer(ds_sites['time.month'])).groupby('time.year').mean(dim=['time'], skipna=True)
        ds_summer_merged = ds_annual_summer_cum_sites.merge(ds_annual_summer_mean_sites)
        with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
            # print('subset to geospatial average of annual summer cumulative fluxes of tundra cells', file=pf)
            # print(ds_annual_summer_cum_geoavg, file=pf)
            # print('create summer mean nitrogen values through time', file=pf)
            # print(ds_annual_summer_mean_geoavg, file=pf)
            print('sites through time summer cumulative flux', file=pf)
            print(ds_annual_summer_cum_sites, file=pf)
            print('sites through time summer means for other variables', file=pf)
            print(ds_annual_summer_mean_sites, file=pf)
            print('merged cum and means for summer by year', file=pf)
            print(ds_summer_merged, file=pf)
        # # warm - control / control
        # ds_diff = (ds_annual_summer_cum_geoavg.sel(sim='otc', drop=True) - ds_annual_summer_cum_geoavg.sel(sim='b2', drop=True)) / ds_annual_summer_cum_geoavg.sel(sim='b2', drop=True)
        # with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #     print('warmed - control / control', file=pf)
        #     print(ds_diff, file=pf)
        # # pooled standard deviation, time
        # model_stdevs = ds_annual_summer_cum_geoavg['TotalResp'].sel(sim=['b2','otc']).std(dim=['sim','year'], skipna=True)
        # model_stdevs_sqrd = model_stdevs * model_stdevs
        # pooled_variance = model_stdevs_sqrd.sum(dim='model') / model_stdevs_sqrd.count(dim='model')
        # pooled_stdev = np.sqrt(pooled_variance.values)
        # with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #     print('model stdevs', file=pf)
        #     print(model_stdevs.values, file=pf)
        #     print('model stdevs squared', file=pf)
        #     print(model_stdevs_sqrd.values, file=pf)
        #     print('pooled model variance avg', file=pf)
        #     print(pooled_variance.values, file=pf)
        #     print('pooled stdev after sqrt', file=pf)
        #     print(pooled_stdev, file=pf)
        warmed_stdevs_sites = ds_summer_merged.sel(sim='otc', drop=True).std(dim=['year'], skipna=True)
        contrl_stdevs_sites = ds_summer_merged.sel(sim='b2', drop=True).std(dim=['year'], skipna=True)
        with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
            print('warmed/control stdevs per model', file=pf)
            print(warmed_stdevs_sites, file=pf)
            print(contrl_stdevs_sites, file=pf)
        warmed_stdevs_sqrd_sites = warmed_stdevs_sites * warmed_stdevs_sites
        contrl_stdevs_sqrd_sites = contrl_stdevs_sites * contrl_stdevs_sites
        with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
            print('warmed/control stdevs per model squared', file=pf)
            print(warmed_stdevs_sqrd_sites, file=pf)
            print(contrl_stdevs_sqrd_sites, file=pf)
        avg_sqrd_stdev_sites = (warmed_stdevs_sqrd_sites + contrl_stdevs_sqrd_sites) / 2
        with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
            print('average warmed+control pooled stdev per model', file=pf)
            print(avg_sqrd_stdev_sites, file=pf)
        pooled_stdev_sites = np.sqrt(avg_sqrd_stdev_sites)
        with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
            print('pooled warmed+control stdev per model after sqrt', file=pf)
            print(pooled_stdev_sites, file=pf)
        # # Hedge's SMD 
        # ds_hsmd = (ds_annual_summer_cum_geoavg.sel(sim='otc', drop=True) - ds_annual_summer_cum_geoavg.sel(sim='b2', drop=True)) / pooled_stdev
        ds_hsmd_sites = (ds_summer_merged.sel(sim='otc', drop=True) - ds_summer_merged.sel(sim='b2', drop=True)) / pooled_stdev_sites
        with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
            # print('Hedges SMD', file=pf)
            # print(ds_hsmd, file=pf)
            print('Hedges SMD sites', file=pf)
            print(ds_hsmd_sites, file=pf)
        ##### site plot of Hedges SMD by model
        # plot totalresp through time
        cb_pal = ["#000000","#004949","#009292","#ff6db6","#ffb6db","#490092","#006ddb","#b66dff","#6db6ff","#b6dbff","#920000","#924900","#db6d00","#24ff24","#ffff6d"]
        matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=cb_pal)
        fig_effectsize = plt.figure(figsize=(8,6))
        ds_hsmd_sites['TotalResp'].plot(x='year', col='model', col_wrap=3, hue='site')
        plt.savefig(Path(config['output_dir'] + 'figures/ER_HSMD_sites_through_time_by_model.png'), dpi=300)
        plt.close(fig_effectsize)
        # ##### effect site plot
        # # plot totalresp through time
        # fig_effectsize = plt.figure(figsize=(8,6))
        # #ds_sub_time = pd.to_datetime([f'{a:04}-07-01' for a in ds_diff['year'].values])
        # ds_sub_time = np.arange(0,len(ds_diff['year'].values))
        # cb_pal = ["#000000","#004949","#009292","#ff6db6","#ffb6db","#490092","#006ddb","#b66dff","#6db6ff","#b6dbff","#920000","#924900","#db6d00","#24ff24","#ffff6d"]
        # color_iter = iter(cb_pal)
        # ax = plt.gca()
        # y_lab = 'ER effect size (%)' 
        # for mod in ds_diff['model'].values:
        #     ds_sub_mod_var = ds_diff['TotalResp'].sel(model=mod).values * 100
        #     #ds_sub_mod_sd = ds_stdev.sel(model=mod).values
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('time values from arange', file=pf)
        #         print(ds_sub_time, file=pf)
        #         print('ER values', file=pf)
        #         print(ds_sub_mod_var, file=pf)
        #     line_color = next(color_iter)
        #     idx = np.isfinite(ds_sub_time) & np.isfinite(ds_sub_mod_var)
        #     b, m = polyfit(ds_sub_time[idx], ds_sub_mod_var[idx], 1)
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('fit slope', file=pf)
        #         print(m, file=pf)
        #         print('fit intercept', file=pf)
        #         print(b, file=pf)
        #     ds_fit = np.array(b + m * ds_sub_time)
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('fit values', file=pf)
        #         print(ds_fit, file=pf)
        #     plt.scatter(ds_sub_time, ds_sub_mod_var, label=mod, c=line_color)
        #     plt.plot(ds_sub_time, ds_fit, '-', c=line_color)
        # #plt.plot(mod_time, mod_var, linestyle='--', color='black', label='ensemble')
        # plt.xlabel('Year', fontsize=16)
        # plt.axvspan(4.5, 9.5, facecolor='lightskyblue', alpha=0.2, zorder=-100) 
        # plt.axvspan(14.5, 22, facecolor='lightskyblue', alpha=0.2, zorder=-100) 
        # #y_lab = 'Delta ' + str(var) + ' (' + units + ')' 
        # ax.set_xlim([-0.5,20.5])
        # y_lab = 'ER effect size (%)' 
        # plt.ylabel(y_lab, fontsize=16) 
        # plt.legend() 
        # #plt.ylim((-0.5, 1.0))
        # plt.savefig(Path(config['output_dir'] + 'figures/Maes_fig_3b_effectsize.png'), dpi=300)
        # plt.close(fig_effectsize)
        # with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #     print('effect size graph output', file=pf)
        # ##### hedges SMD plot
        # # plot totalresp through time
        # fig_hsmd = plt.figure(figsize=(8,6))
        # #ds_sub_time = pd.to_datetime([f'{a:04}-07-01' for a in ds_diff['year'].values])
        # ax = plt.gca()
        # color_iter = iter(cb_pal)
        # for mod in ds_hsmd['model'].values:
        #     ds_sub_mod_var = ds_hsmd['TotalResp'].sel(model=mod).values
        #     #ds_sub_mod_sd = ds_stdev.sel(model=mod).values
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('time values from arange', file=pf)
        #         print(ds_sub_time, file=pf)
        #         print('ER values', file=pf)
        #         print(ds_sub_mod_var, file=pf)
        #     line_color = next(color_iter)
        #     plt.scatter(ds_sub_time, ds_sub_mod_var, label=mod, c=line_color)
        #     ds_sub_time_age1 = ds_sub_time[5:10]
        #     ds_sub_time_age2 = ds_sub_time[10:15]
        #     ds_sub_mod_var_age1 = ds_sub_mod_var[5:10]
        #     ds_sub_mod_var_age2 = ds_sub_mod_var[10:15]
        #     idx1 = np.isfinite(ds_sub_time_age1) & np.isfinite(ds_sub_mod_var_age1)
        #     idx2 = np.isfinite(ds_sub_time_age2) & np.isfinite(ds_sub_mod_var_age2)
        #     b1, m1 = polyfit(ds_sub_time_age1[idx1], ds_sub_mod_var_age1[idx1], 1)
        #     b2, m2 = polyfit(ds_sub_time_age2[idx2], ds_sub_mod_var_age2[idx2], 1)
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('fit slope', file=pf)
        #         print(m1, file=pf)
        #         print('fit intercept', file=pf)
        #         print(b1, file=pf)
        #     ds_fit1 = np.array(b1 + m1 * ds_sub_time_age1)
        #     ds_fit2 = np.array(b2 + m2 * ds_sub_time_age2)
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('fit values', file=pf)
        #         print(ds_fit1, file=pf)
        #     plt.plot(ds_sub_time_age1, ds_fit1, '-', c=line_color)
        #     plt.plot(ds_sub_time_age2, ds_fit2, '-', c=line_color)
        # #plt.plot(mod_time, mod_var, linestyle='--', color='black', label='ensemble')
        # plt.xlabel('Year', fontsize=16)
        # plt.axvspan(4.5, 9.5, facecolor='lightskyblue', alpha=0.2, zorder=-100) 
        # plt.axvspan(14.5, 22, facecolor='lightskyblue', alpha=0.2, zorder=-100) 
        # x_vals = np.arange(5,10,1)
        # y_vals = 1.35 + (-0.13 * x_vals)
        # plt.plot(x_vals, y_vals, '--', c='black')
        # x_vals = np.arange(10,15,1)
        # y_vals = -1.45 + (0.15 * x_vals)
        # plt.plot(x_vals, y_vals, '--', c='black', label='Observed')
        # ax.set_xlim([-0.5,20.5])
        # #y_lab = 'Delta ' + str(var) + ' (' + units + ')' 
        # y_lab = 'ER Hedges SMD' 
        # plt.ylabel(y_lab, fontsize=16) 
        # plt.legend() 
        # #plt.ylim((-0.5, 1.0))
        # plt.savefig(Path(config['output_dir'] + 'figures/Maes_fig_3b_hsmd.png'), dpi=300)
        # plt.close(fig_hsmd)
        # with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #     print('Hedges SMD graph output', file=pf)
        # ##### ER Hedges SMD by N%
        # # plot totalresp through time
        # fig_effectsize = plt.figure(figsize=(8,6))
        # #ds_sub_time = pd.to_datetime([f'{a:04}-07-01' for a in ds_diff['year'].values])
        # color_iter = iter(cb_pal)
        # ax = plt.gca()
        # for mod in ["ELM2-NGEE","ELM1-ECA","CLASSIC"]:
        #     ds_sub_time = ds_annual_summer_mean_geoavg['SoilN'].sel(model=mod, sim='b2').values * (1/(1.35*100000))
        #     ds_sub_mod_var = ds_hsmd['TotalResp'].sel(model=mod).values
        #     #ds_sub_mod_sd = ds_stdev.sel(model=mod).values
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('time values from arange', file=pf)
        #         print(ds_sub_time, file=pf)
        #         print('ER values', file=pf)
        #         print(ds_sub_mod_var, file=pf)
        #     line_color = next(color_iter)
        #     idx = np.isfinite(ds_sub_time) & np.isfinite(ds_sub_mod_var)
        #     b, m = polyfit(ds_sub_time[idx], ds_sub_mod_var[idx], 1)
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('fit slope', file=pf)
        #         print(m, file=pf)
        #         print('fit intercept', file=pf)
        #         print(b, file=pf)
        #     ds_fit = np.array(b + m * ds_sub_time)
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('fit values', file=pf)
        #         print(ds_fit, file=pf)
        #     plt.scatter(ds_sub_time, ds_sub_mod_var, label=mod, c=line_color)
        #     plt.plot(ds_sub_time, ds_fit, '-', c=line_color)
        # #plt.plot(mod_time, mod_var, linestyle='--', color='black', label='ensemble')
        # plt.xlabel('TN control (%)', fontsize=16)
        # x_vals = np.arange(0,1,0.1)
        # y_vals_c = 0.75 + (-0.6 * x_vals)
        # plt.plot(x_vals, y_vals_c, '--', c='black', label='Observed')
        # y_vals_b = 0.5 + (-0.6 * x_vals)
        # plt.plot(x_vals, y_vals_b, '--', c='blue', label='95CI')
        # y_vals_t = 1.0 + (-0.6 * x_vals)
        # plt.plot(x_vals, y_vals_t, '--', c='blue')
        # #plt.fill_between(x_vals, y_vals_c + y_vals_t, y_vals_c - y_vals_b,  facecolor='lightskyblue', alpha=0.2, zorder=-100)
        # #y_lab = 'Delta ' + str(var) + ' (' + units + ')' 
        # #ax.set_xlim([0,22])
        # y_lab = 'ER Hedges SMD (%)' 
        # plt.ylabel(y_lab, fontsize=16) 
        # plt.legend() 
        # #plt.ylim((-0.5, 1.0))
        # plt.savefig(Path(config['output_dir'] + 'figures/Maes_fig_5a.png'), dpi=300)
        # plt.close(fig_effectsize)
        # with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #     print('N effect graphed', file=pf)
        # ##### ER Hedges SMD by CN ratio
        # # plot totalresp through time
        # fig_effectsize = plt.figure(figsize=(8,6))
        # #ds_sub_time = pd.to_datetime([f'{a:04}-07-01' for a in ds_diff['year'].values])
        # color_iter = iter(cb_pal)
        # ax = plt.gca()
        # for mod in ["ELM2-NGEE","ELM1-ECA","CLASSIC"]:
        #     ds_sub_time = ds_annual_summer_mean_geoavg['CN'].sel(model=mod, sim='b2').values
        #     ds_sub_mod_var = ds_hsmd['TotalResp'].sel(model=mod).values
        #     #ds_sub_mod_sd = ds_stdev.sel(model=mod).values
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('time values from arange', file=pf)
        #         print(ds_sub_time, file=pf)
        #         print('ER values', file=pf)
        #         print(ds_sub_mod_var, file=pf)
        #     line_color = next(color_iter)
        #     idx = np.isfinite(ds_sub_time) & np.isfinite(ds_sub_mod_var)
        #     b, m = polyfit(ds_sub_time[idx], ds_sub_mod_var[idx], 1)
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('fit slope', file=pf)
        #         print(m, file=pf)
        #         print('fit intercept', file=pf)
        #         print(b, file=pf)
        #     ds_fit = np.array(b + m * ds_sub_time)
        #     with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #         print('fit values', file=pf)
        #         print(ds_fit, file=pf)
        #     plt.scatter(ds_sub_time, ds_sub_mod_var, label=mod, c=line_color)
        #     plt.plot(ds_sub_time, ds_fit, '-', c=line_color)
        # #plt.plot(mod_time, mod_var, linestyle='--', color='black', label='ensemble')
        # plt.xlabel('CN control (unitless)', fontsize=16)
        # x_vals = np.arange(0,30,1)
        # y_vals_c = 0.15 + (0.03 * x_vals)
        # plt.plot(x_vals, y_vals_c, '--', c='black', label='Observed')
        # y_vals_b = -0.25 + (0.03 * x_vals)
        # plt.plot(x_vals, y_vals_b, '--', c='blue', label='95CI')
        # y_vals_t = 0.6 + (0.03 * x_vals)
        # plt.plot(x_vals, y_vals_t, '--', c='blue')
        # #plt.fill_between(x_vals, y_vals_c + y_vals_t, y_vals_c - y_vals_b, facecolor='lightskyblue', alpha=0.2, zorder=-100)
        # #y_lab = 'Delta ' + str(var) + ' (' + units + ')' 
        # #ax.set_xlim([0,22])
        # y_lab = 'ER Hedges SMD (%)' 
        # plt.ylabel(y_lab, fontsize=16) 
        # plt.legend() 
        # #plt.ylim((-0.5, 1.0))
        # plt.savefig(Path(config['output_dir'] + 'figures/Maes_fig_5b.png'), dpi=300)
        # plt.close(fig_effectsize)
        # with open(Path(config['output_dir'] + 'figures/debug_maes_graphs.txt'), 'a') as pf:
        #     print('CN effect graphed', file=pf)

# regional graphing
def regional_model_graphs(input_list):
    # context manager to not change chunk sizes
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # graph type (mean, instantaneous)
        config = input_list[0]
        var_list = input_list[1]
        soil = 0
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'w') as pf:
            print('graphing started:', file=pf)
        # open zarr file
        zarr_file = config['output_dir'] + 'zarr_output/WrPMIP_Pan-Arctic_models_harmonized.zarr'
        ds = xr.open_zarr(zarr_file, chunks='auto', chunked_array_type='dask', use_cftime=True, mask_and_scale=False) 
        # select soil layer
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('data loaded:', file=pf)
            print(ds, file=pf)
        # change to -180/180 coords
        ds['lon'] =('lon', (((ds.lon.values + 180) % 360) - 180))
        # use sortby to enforce monotonically increasing dims for xarray
        ds = ds.sortby(['lon'])
        # SoilDepth collapsed to comparable dimensions by selecting top layer,
        ds = ds.isel(SoilDepth = soil, drop=True)
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('soil depth isel:', file=pf)
            print(ds, file=pf)
        # mask by RECCAP2 permafrost extent
        # # load recap2 netcdf
        # reccap2_filename = '/projects/warpmip/shared/RECCAP2_permafrost_regions_isimip3.nc'
        # reccap2 = xr.open_dataset(reccap2_filename)
        # reccap2 = reccap2.rename({'latitude': 'lat', 'longitude': 'lon'})
        # reccap2 = reccap2['permafrost_region_mask']
        # reccap2 = reccap2.reindex_like(ds.TotalResp, method='nearest')
        # reccap2 = reccap2.where(reccap2 < 1e35)
        # with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
        #     print(reccap2, file=pf)
        # out_geo = config['output_dir'] + 'figures/reccap2_mask.png' 
        # # Set figure size
        # fig_reccap = plt.figure(figsize=(8,6))
        # # Set the axes using the specified map projection
        # ax=plt.axes(projection=ccrs.Orthographic(0,90))
        # # Make a mesh plot
        # cs=ax.pcolormesh(reccap2['lon'], reccap2['lat'], reccap2, transform = ccrs.PlateCarree(), cmap='viridis')
        # ax.coastlines()
        # #ax.gridlines()
        # cbar = plt.colorbar(cs,shrink=0.7,location='left',label='reccap2')
        # ax.yaxis.set_ticks_position('left')
        # #ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
        # add_circle_boundary(ax)
        # plt.savefig(out_geo, dpi=300)
        # plt.close(fig_reccap)
        # cavm
        cavm_f = '/projects/warpmip/shared/0.5_cavm.nc' 
        cavm = xr.open_dataset(cavm_f)
        ds = ds.where(cavm.cavm > 0).persist()
        ##########
        # nee
        #ds = ds.where(ds.lat > 75, np.nan).persist()
        ##########
        # all daily response then averaged across all models and times,
        # thus geospatially explicit 20-year ensemble means/stdevs: 
        avg_type = ['time','model']
        ds_geo_means = ds.mean(dim=avg_type, skipna=True).persist()
        ds_geo_stdev = ds.std(dim=avg_type, skipna=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('geo means', file=pf)
        # all summer (may-sept) daily responses then averaged across all models and times, 
        # thus geospatially explicit 20-year summer ensemble means/stdevs:
        ds_geo_summer_means = ds.sel(time=is_summer(ds['time.month'])).mean(dim=avg_type, skipna=True).persist()
        ds_geo_summer_stdev = ds.sel(time=is_summer(ds['time.month'])).std(dim=avg_type, skipna=True).persist()
        # all winter (oct-april) daily responses then averaged across all models and times, 
        # thus geospatially explicit 20-year winter ensemble means/stdevs:
        ds_geo_winter_means = ds.sel(time=is_winter(ds['time.month'])).mean(dim=avg_type, skipna=True).persist()
        ds_geo_winter_stdev = ds.sel(time=is_winter(ds['time.month'])).std(dim=avg_type, skipna=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('summer/winter means', file=pf)
        # all daily response then averaged across all models,lats,lons within monthly groups,
        # thus Pan-Arctic 20-year monthly ensemble mean/stdev:
        avg_type = ['time','model','lat','lon']
        ds_geo_month_mod_mean = ds.groupby('time.month').mean(dim=avg_type, skipna=True).persist()
        ds_geo_month_mod_stdev = ds.groupby('time.month').std(dim=avg_type, skipna=True).persist()
        # all daily response then averaged across all models,lats,lons within monthly groups,
        # thus Pan-Arctic monthly model mean/stdev:
        avg_type = ['time','lat','lon']
        ds_geo_month_means = ds.groupby('time.month').mean(dim=avg_type, skipna=True).persist()
        ds_geo_month_stdev = ds.groupby('time.month').std(dim=avg_type, skipna=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('monthly geo means', file=pf)
        # all daily response then averaged across all lats,lons,
        # thus Pan-Arctic mean/stdev for each model through time
        avg_type = ['lat','lon']
        ds_time_means = ds.mean(dim=avg_type, skipna=True).persist()
        ds_time_stdev = ds.std(dim=avg_type, skipna=True).persist()
        # all daily response then averaged across all models,lats,lons,
        # thus Pan-Arctic ensemble mean/stdev through time
        avg_type = ['model','lat','lon']
        ds_time_mod_mean = ds.mean(dim=avg_type, skipna=True).persist()
        ds_time_mod_stdev = ds.std(dim=avg_type, skipna=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('done with means before time shift', file=pf)
        ##########  
        # create difference arrays
        ds_otc_diff = ds.sel(sim='otc', drop=True) - ds.sel(sim='b2', drop=True)
        ds_otc_diff2 = ds_otc_diff.copy(deep=True)
        ds_otc_diff3 = ds_otc_diff.copy(deep=True)
        ds_otc_effect_size = ds_otc_diff / ds.sel(sim='b2', drop=True)
        ds_otc_effect_size.persist()
        ds_otc_effect_size_temp = ds_otc_effect_size / ds_otc_diff['SoilTemp']
        ds_otc_effect_size_temp.persist()
        ds_sf_diff = ds.sel(sim='sf', drop=True) - ds.sel(sim='b2', drop=True)
        ds_sf_diff2 = ds_sf_diff.copy(deep=True)
        ds_sf_diff3 = ds_sf_diff.copy(deep=True)
        ds_sf_effect_size = ds_sf_diff / ds.sel(sim='b2', drop=True)
        ds_sf_effect_size.persist()
        ds_sf_effect_size_temp = ds_sf_effect_size / ds_sf_diff['SoilTemp']
        ds_sf_effect_size_temp.persist()
        ds_sf_otc_diff = ds.sel(sim='sf') - ds.sel(sim='otc')
        ds_sf_otc_diff2 = ds_sf_otc_diff.copy(deep=True)
        # calculate model averages for effect size calculation
        ds_mod_means_b2 = ds.sel(sim='b2', drop=True).groupby('model').mean(dim=['time','model','lat','lon'], skipna=True)
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('mod means for normalization', file=pf)
            print(ds_mod_means_b2, file=pf)
            print(ds_mod_means_b2['TotalResp'].values, file=pf)
            print('effect size dataset', file=pf)
            print(ds_otc_effect_size, file=pf)
            print(ds_sf_effect_size, file=pf)
        # create pandas multiindex
        idx_month = ds['time.month'].values
        idx_season = idx_month.copy()
        idx_peak = idx_month.copy()
        idx_season[(idx_season < 5) | (idx_season > 9)] = 2
        idx_peak[(idx_peak < 5) | (idx_peak > 9)] = 2
        idx_season[(idx_season >= 5) & (idx_season <= 9)] = 1
        idx_peak[(idx_peak < 5) | (idx_peak > 9)] = 2
        idx_peak[(idx_peak >= 5) & (idx_peak <= 9)] = 1
        idx_year = ds['time.year'].values
        idx_year_shift = shift5(idx_year, 4, fill_value=1999)
        year_month_season_idx = pd.MultiIndex.from_arrays([idx_year_shift, idx_month, idx_season], names=['year','month','season'])
        year_season_idx = pd.MultiIndex.from_arrays([idx_year_shift, idx_season], names=['year','season'])
        year_peak_idx = pd.MultiIndex.from_arrays([idx_year_shift, idx_peak], names=['year','peak'])
        year_month_idx = pd.MultiIndex.from_arrays([idx_year_shift, idx_month], names=['year','month'])
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('pandas multindex creation:', file=pf)
            print(year_season_idx, file=pf)
            print(year_peak_idx, file=pf)
            print(year_month_idx, file=pf)
        ds2 = ds.copy(deep=True)
        ds2.coords['year_month'] = ('time', year_month_idx)
        ds.coords['year_season'] = ('time', year_season_idx)
        ds_otc_effect_size.coords['year_month'] = ('time', year_month_idx)
        ds_sf_effect_size.coords['year_month'] = ('time', year_month_idx)
        #ds.coords['season'] = idx_season
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('assign year_month to time coordinate:', file=pf)
            print(ds, file=pf)
        # group by month within yea
        ds_geo_otc_month_ef = ds_otc_effect_size.groupby('year_month').mean(dim=['time'], skipna=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('otc effect size geospatial timeseries:', file=pf)
            print(ds_geo_otc_month_ef, file=pf)
        #ds_otc_month_ef = ds_otc_effect_size.where(ds_otc_effect_size.lat >= 75, np.nan).groupby('year_month').mean(dim=['time','lat','lon'], skipna=True).persist()
        ds_otc_month_ef = ds_otc_effect_size.groupby('year_month').mean(dim=['time','lat','lon'], skipna=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('otc effect size timeseries:', file=pf)
            print(ds_otc_month_ef, file=pf)
        year_index = ds_otc_month_ef['year'].values
        month_index = ds_otc_month_ef['month'].values
        year_month_index = tuple(zip(year_index, month_index))
        f_string_index = [f'{a:04}-{b:02}-01' for a, b in year_month_index]
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('year index:', file=pf)
            print(year_index, file=pf)
            print('month index:', file=pf)
            print(month_index, file=pf)
            print('combined index:', file=pf)
            print(year_month_index, file=pf)
            print('date index:', file=pf)
            print(f_string_index, file=pf)
        ds_otc_multiindex = pd.to_datetime(f_string_index)
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('otc effect size multiindex:', file=pf)
            print(ds_otc_multiindex, file=pf)
        #ds_otc_month_ef_res = ds_otc_month_ef.reset_index('year_month').persist() 
        #with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
        #    print('otc reset multiindex:', file=pf)
        #    print(ds_otc_month_ef_res, file=pf)
        #ds_otc_summer_ef = ds_otc_effect_size.where(ds_otc_effect_size.lat >= 75, np.nan).sel(month=[6,7,8]).groupby('year').mean(dim=['year_month'], skipna=True).persist()
        #ds_otc_winter_ef = ds_otc_effect_size.where(ds_otc_effect_size.lat >= 75, np.nan).sel(month=[12,1,2]).groupby('year').mean(dim=['year_month'], skipna=True).persist()
        #with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
        #    print('otc effect size summer/winter averages:', file=pf)
        #    print(ds_otc_summer_ef, file=pf)
        #    print(ds_otc_winter_ef, file=pf)
        ds_otc_seasonal_ef = ds_otc_month_ef.groupby('month').mean(dim=['year_month'], skipna=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('otc seasonal effect size:', file=pf)
            print(ds_otc_seasonal_ef, file=pf)
        #
        ds_geo_sf_month_ef = ds_sf_effect_size.groupby('year_month').mean(dim=['time'], skipna=True).persist()
        #ds_sf_month_ef = ds_sf_effect_size.where(ds_sf_effect_size.lat > 75, np.nan).groupby('year_month').mean(dim=['time','lat','lon'], skipna=True).persist()
        ds_sf_month_ef = ds_sf_effect_size.groupby('year_month').mean(dim=['time','lat','lon'], skipna=True).persist()
        #ds_otc_summer_ef = ds_otc_effect_size.where(ds_otc_effect_size.lat >= 75, np.nan).sel(month=[6,7,8]).groupby('year').mean(dim=['year_month'], skipna=True).persist()
        #ds_otc_winter_ef = ds_otc_effect_size.where(ds_otc_effect_size.lat >= 75, np.nan).sel(month=[12,1,2]).groupby('year').mean(dim=['year_month'], skipna=True).persist()
        ds_sf_seasonal_ef = ds_sf_month_ef.groupby('month').mean(dim=['year_month'], skipna=True).persist()
        #
        ds_season_mean = ds.groupby('year_season').mean(dim=['time','lat','lon'], skipna=True).persist()
        ds_summer_mean = ds_season_mean.sel(season=1, drop=True).persist()
        ds_season_ensemble = ds.groupby('year_season').mean(dim=['time','model','lat','lon'], skipna=True).persist()
        ds_summer_ensemble = ds_season_ensemble.sel(season=1, drop=True).persist()
        ds_season_stdev = ds.groupby('year_season').std(dim=['time','lat','lon'], skipna=True).persist()
        ds_summer_stdev = ds_season_stdev.sel(season=1, drop=True).persist()
        ds_season_ensemble_stdev = ds.groupby('year_season').std(dim=['time','model','lat','lon'], skipna=True).persist()
        ds_summer_ensemble_stdev = ds_season_ensemble_stdev.sel(season=1, drop=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('mean/sd by month within year:', file=pf)
            print(ds_season_mean, file=pf)
            print(ds_summer_mean, file=pf)
            print(ds_season_stdev, file=pf)
            print(ds_summer_stdev, file=pf)
        # add year_month index to diff values
        ds_otc_diff.coords['year_season'] = ('time', year_season_idx)
        ds_sf_diff.coords['year_season'] = ('time', year_season_idx)
        ds_sf_otc_diff.coords['year_season'] = ('time', year_season_idx)
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('diff calc', file=pf)
            print(ds_otc_diff, file=pf)
            print(ds_sf_diff, file=pf)
            print(ds_sf_otc_diff, file=pf)
        #
        ds_geo_mean_timeseries = ds2.groupby('year_month').mean(dim=['time'], skipna=True).persist()
        # 
        avg_type = ['time','lat','lon']
        ds_otc_season_mean = ds_otc_diff.groupby('year_season').mean(dim=avg_type, skipna=True).persist()
        ds_otc_season_stdev = ds_otc_diff.groupby('year_season').std(dim=avg_type, skipna=True).persist()
        ds_otc_summer_mean = ds_otc_season_mean.sel(season=1, drop=True).persist()
        ds_otc_summer_stdev = ds_otc_season_stdev.sel(season=1, drop=True).persist()
        ds_otc_winter_mean = ds_otc_season_mean.sel(season=2, drop=True).persist()
        ds_otc_winter_stdev = ds_otc_season_stdev.sel(season=2, drop=True).persist()
        #
        ds_sf_season_mean = ds_sf_diff.groupby('year_season').mean(dim=avg_type, skipna=True).persist()
        ds_sf_season_stdev = ds_sf_diff.groupby('year_season').std(dim=avg_type, skipna=True).persist()
        ds_sf_summer_mean = ds_sf_season_mean.sel(season=1, drop=True).persist()
        ds_sf_summer_stdev = ds_sf_season_stdev.sel(season=1, drop=True).persist()
        ds_sf_winter_mean = ds_sf_season_mean.sel(season=2, drop=True).persist()
        ds_sf_winter_stdev = ds_sf_season_stdev.sel(season=2, drop=True).persist()
        #
        ds_sf_otc_season_mean = ds_sf_otc_diff.groupby('year_season').mean(dim=avg_type, skipna=True).persist()
        ds_sf_otc_season_stdev = ds_sf_otc_diff.groupby('year_season').std(dim=avg_type, skipna=True).persist()
        ds_sf_otc_summer_mean = ds_sf_otc_season_mean.sel(season=1, drop=True).persist()
        ds_sf_otc_summer_stdev = ds_sf_otc_season_stdev.sel(season=1, drop=True).persist()
        # ensemble
        avg_type = ['time','model','lat','lon']
        ds_otc_season_ensemble = ds_otc_diff.groupby('year_season').mean(dim=avg_type, skipna=True).persist()
        ds_sf_season_ensemble = ds_sf_diff.groupby('year_season').mean(dim=avg_type, skipna=True).persist()
        ds_sf_otc_season_ensemble = ds_sf_otc_diff.groupby('year_season').mean(dim=avg_type, skipna=True).persist()
        ds_otc_summer_ensemble = ds_otc_season_ensemble.sel(season=1, drop=True).persist()
        ds_sf_summer_ensemble = ds_sf_season_ensemble.sel(season=1, drop=True).persist()
        ds_otc_winter_ensemble = ds_otc_season_ensemble.sel(season=2, drop=True).persist()
        ds_sf_winter_ensemble = ds_sf_season_ensemble.sel(season=2, drop=True).persist()
        ds_sf_otc_summer_ensemble = ds_sf_otc_season_ensemble.sel(season=1, drop=True).persist()
        # add year_month index to diff values
        ds_otc_diff2.coords['year_month'] = ('time', year_month_idx)
        ds_sf_diff2.coords['year_month'] = ('time', year_month_idx)
        ds_otc_diff3.coords['year_season'] = ('time', year_peak_idx)
        ds_sf_diff3.coords['year_season'] = ('time', year_peak_idx)
        ds_sf_otc_diff2.coords['year_month'] = ('time', year_month_idx)
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('diff calc', file=pf)
            print(ds_otc_diff2, file=pf)
            print(ds_sf_diff2, file=pf)
            print(ds_sf_otc_diff2, file=pf)
        # 
        avg_type = ['time','lat','lon']
        ds_geo_otc_season_mean = ds_otc_diff2.groupby('year_month').mean(dim=avg_type, skipna=True).persist()
        ds_geo_sf_season_mean = ds_sf_diff2.groupby('year_month').mean(dim=avg_type, skipna=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('season timeseries:', file=pf)
            print(ds_geo_otc_season_mean, file=pf)
            print(ds_geo_sf_season_mean, file=pf)
        ds_geo_otc_season_mean = ds_geo_otc_season_mean.sel(month = 7, drop=True).persist() 
        ds_geo_sf_season_mean = ds_geo_sf_season_mean.sel(month = 7, drop=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('summer timeseries averages:', file=pf)
            print(ds_geo_otc_season_mean, file=pf)
            print(ds_geo_sf_season_mean, file=pf)
        ds_geo_otc_year_month_mean = ds_otc_diff2.groupby('year_month').mean(dim=avg_type, skipna=True).persist()
        ds_geo_sf_year_month_mean = ds_sf_diff2.groupby('year_month').mean(dim=avg_type, skipna=True).persist()
        ds_geo_otc_month_mod_mean = ds_geo_otc_year_month_mean.groupby('month').mean(dim=['year_month'], skipna=True).persist()
        ds_geo_sf_month_mod_mean = ds_geo_sf_year_month_mean.groupby('month').mean(dim=['year_month'], skipna=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('diff geo month means', file=pf)
            print(ds_geo_otc_month_mod_mean, file=pf)
        ds_otc_season_mod_normalized = ds_geo_otc_season_mean.groupby('model') / ds_mod_means_b2
        ds_sf_season_mod_normalized = ds_geo_sf_season_mean.groupby('model') / ds_mod_means_b2
        ds_otc_year_mod_normalized = ds_geo_otc_year_month_mean.groupby('model') / ds_mod_means_b2
        ds_sf_year_mod_normalized = ds_geo_sf_year_month_mean.groupby('model') / ds_mod_means_b2
        ds_otc_month_mod_normalized = ds_geo_otc_month_mod_mean.groupby('model') / ds_mod_means_b2
        ds_sf_month_mod_normalized = ds_geo_sf_month_mod_mean.groupby('model') / ds_mod_means_b2
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('diff geo month means normalized', file=pf)
            print(ds_otc_month_mod_normalized, file=pf)
        ds_otc_mean_effectsize = ds_otc_year_mod_normalized.groupby('month').mean(dim=['model'], skipna=True).persist()
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('diff geo mean effect size by month', file=pf)
            print(ds_otc_mean_effectsize, file=pf)
        #
        ds_geo_sf_year_month_mean = ds_sf_diff2.groupby('year_month').mean(dim=avg_type, skipna=True).persist()
        ds_geo_sf_month_mean = ds_geo_sf_year_month_mean.groupby('month').mean(dim=['year_month','model'], skipna=True).persist()
        # 
        avg_type = ['time','lat','lon']
        ds_otc_year_month_mean = ds_otc_diff2.groupby('year_month').mean(dim=avg_type, skipna=True).persist()
        ds_otc_month_mean = ds_otc_year_month_mean.groupby('month').mean(dim=['year_month'], skipna=True).persist()
        ds_otc_year_month_stdev = ds_otc_diff2.groupby('year_month').std(dim=avg_type, skipna=True).persist()
        #
        ds_sf_year_month_mean = ds_sf_diff2.groupby('year_month').mean(dim=avg_type, skipna=True).persist()
        ds_sf_month_mean = ds_sf_year_month_mean.groupby('month').mean(dim=['year_month'], skipna=True).persist()
        ds_sf_year_month_stdev = ds_sf_diff2.groupby('year_month').std(dim=avg_type, skipna=True).persist()
        #
        ds_sf_otc_month_mean = ds_sf_otc_diff2.groupby('year_month').mean(dim=avg_type, skipna=True).persist()
        ds_sf_otc_month_stdev = ds_sf_otc_diff2.groupby('year_month').std(dim=avg_type, skipna=True).persist()
        # ensemble
        avg_type = ['time','model','lat','lon']
        ds_otc_year_month_ensemble = ds_otc_diff2.groupby('year_month').mean(dim=avg_type, skipna=True).persist()
        ds_otc_month_ensemble = ds_otc_year_month_ensemble.groupby('month').mean(dim=['year_month'], skipna=True).persist()
        ds_sf_year_month_ensemble = ds_sf_diff2.groupby('year_month').mean(dim=avg_type, skipna=True).persist()
        ds_sf_month_ensemble = ds_sf_year_month_ensemble.groupby('month').mean(dim=['year_month'], skipna=True).persist()
        ds_sf_otc_year_month_ensemble = ds_sf_otc_diff2.groupby('year_month').mean(dim=avg_type, skipna=True).persist()
        # debug outputs 
        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
            print('soil subset with sel', file=pf)
            print(ds_geo_means, file=pf)
            print(ds_geo_summer_means, file=pf)
            print(ds_geo_winter_means, file=pf)
            print(ds_geo_month_means, file=pf)
            print(ds_time_means, file=pf)
            print(ds_time_mod_mean, file=pf)
            print(ds['sim'].values, file=pf)
            print(str(ds['sim'].values), file=pf)
            print(ds['model'].values, file=pf)
            print(str(ds['model'].values), file=pf)
        # graph for each variable
        for var in var_list: 
            # make necessary folders
            Path(config['output_dir'] + 'figures/' + var).mkdir(parents=True, exist_ok=True)
            Path(config['output_dir'] + 'figures/' + var).chmod(0o762)
            max_geo_means = max(ds_geo_means[var].max().values, ds_geo_summer_means[var].max().values, ds_geo_winter_means[var].max().values)
            min_geo_means = min(ds_geo_means[var].min().values, ds_geo_summer_means[var].min().values, ds_geo_winter_means[var].min().values)
            max_geo_stdev = max(ds_geo_means[var].max().values, ds_geo_summer_means[var].max().values, ds_geo_winter_means[var].max().values)
            max_geo_season_means = ds_geo_month_means[var].max().values
            min_geo_season_means = ds_geo_month_means[var].min().values
            max_geo_season_stdev = ds_geo_month_means[var].max().values
            if var in ['TotalResp','GPP']:
                units = 'g C m-2 day-1'
            elif var in ['ALT', 'WTD']:
                units = 'm'
            elif var in ['SoilTemp']:
                units = '°C'  
            for sim in ds['sim'].values:
                ##############################################################
                
                ##############################################################
                if sim in ['otc', 'sf']:
                    with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                        print('starting year_diff plots -- summer', file=pf)
                    if sim == 'otc':
                        out_year = config['output_dir'] + 'figures/'+ var + '/Pan-Arctic_mean_' + var + '_year_diff_' + str(sim) + '.png' 
                        ds_month = ds_otc_season_mod_normalized[var] * 100 #.sel(model=['CLM5','CLM5-ExIce','ELM2-NGEE','JSBACH','ecosys','ELM1-ECA'])
                        #ds_stdev = ds_otc_summer_stdev[var].sel(year=slice(2000,2021))
                        #mod_time = ds_otc_summer_ensemble['year'].sel(year=slice(2000,2021)).values 
                        #mod_var = ds_otc_summer_ensemble[var].sel(year=slice(2000,2021)).values 
                    elif sim == 'sf':
                        out_year = config['output_dir'] + 'figures/'+ var + '/Pan-Arctic_mean_' + var + '_year_diff_' + str(sim) + '.png' 
                        ds_month = ds_sf_season_mod_normalized[var] * 100 #.sel(model=['CLM5','CLM5-ExIce','ELM2-NGEE','JSBACH','ecosys','ELM1-ECA'])
                        #ds_stdev = ds_sf_summer_stdev[var].sel(year=slice(2000,2021))
                        #mod_time = ds_sf_summer_ensemble['year'].sel(year=slice(2000,2021)).values 
                        #mod_var = ds_sf_summer_ensemble[var].sel(year=slice(2000,2021)).values
                    #y_min = min(ds_otc_month_ef[var].min().values, ds_sf_month_ef[var].min().values)
                    #y_max = max(ds_otc_month_ef[var].max().values, ds_sf_month_ef[var].max().values)
                    fig_year_diff_summer = plt.figure(figsize=(8,6))
                    #ds_sub_time = pd.to_datetime([f'{a:04}-{b:02}-01' for a, b in tuple(zip(ds_otc_month_ef['year'].values,ds_otc_month_ef['month'].values))])
                    ds_sub_time = pd.to_datetime([f'{a:04}-01-01' for a in ds_otc_season_mod_normalized['year'].values])
                    for mod in ds_month['model'].values:
                        ds_sub_mod_var = ds_month.sel(model=mod).values
                        #ds_sub_mod_sd = ds_stdev.sel(model=mod).values
                        plt.plot(ds_sub_time, ds_sub_mod_var, label=mod)
                    #plt.plot(mod_time, mod_var, linestyle='--', color='black', label='ensemble')
                    plt.xlabel('Year', fontsize=16)
                    #y_lab = 'Delta ' + str(var) + ' (' + units + ')' 
                    y_lab = str(var)+ ' Change from Baseline Mean (%)' 
                    plt.ylabel(y_lab, fontsize=16) 
                    plt.legend() 
                    #plt.ylim((-0.5, 1.0))
                    plt.savefig(out_year, dpi=300)
                    plt.close(fig_year_diff_summer)
                #####
                # if sim in ['otc', 'sf']:
                #     with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                #         print('starting year_diff plots -- winter', file=pf)
                #     if sim == 'otc':
                #         out_year = config['output_dir'] + 'figures/'+ var + '/Pan-Arctic_mean_winter' + var + '_year_diff_' + str(sim) + '.png' 
                #         ds_month = ds_otc_month_ef[var]
                #         #ds_stdev = ds_otc_winter_stdev[var].sel(year=slice(2000,2021))
                #         #mod_time = ds_otc_winter_ensemble['year'].sel(year=slice(2000,2021)).values 
                #         #mod_var = ds_otc_winter_ensemble[var].sel(year=slice(2000,2021)).values 
                #     elif sim == 'sf':
                #         out_year = config['output_dir'] + 'figures/'+ var + '/Pan-Arctic_mean_winter' + var + '_year_diff_' + str(sim) + '.png' 
                #         ds_month = ds_sf_month_ef[var]
                #         #ds_stdev = ds_sf_winter_stdev[var].sel(year=slice(2000,2021))
                #         #mod_time = ds_sf_winter_ensemble['year'].sel(year=slice(2000,2021)).values 
                #         #mod_var = ds_sf_winter_ensemble[var].sel(year=slice(2000,2021)).values
                #     y_min = min(ds_otc_month_ef[var].min().values, ds_sf_month_ef[var].min().values)
                #     y_max = max(ds_otc_month_ef[var].max().values, ds_sf_month_ef[var].max().values)
                #     fig_year_diff_winter = plt.figure(figsize=(8,6))
                #     ds_sub_time = pd.to_datetime([f'{a}-{b}-01' for a, b in ds_month['year_month']])
                #     for mod in ds_month['model'].values:
                #         ds_sub_mod_var = ds_month.sel(model=mod).values
                #         #ds_sub_mod_sd = ds_stdev.sel(model=mod).values
                #         plt.plot(ds_sub_time, ds_sub_mod_var, label=mod)
                #     #plt.plot(mod_time, mod_var, linestyle='--', color='black', label='ensemble')
                #     plt.xlabel('Year')
                #     y_lab = 'Delta Mean Winter ' + str(var) + ' (' + units + ')' 
                #     plt.ylabel(y_lab) 
                #     plt.legend() 
                #     plt.ylim((y_min, y_max))
                #     plt.savefig(out_year, dpi=300)
                #     plt.close(fig_year_diff_winter)
                ##############################################################
                out_year = config['output_dir'] + 'figures/'+ var + '/Pan-Arctic_mean_' + var + '_year_' + str(sim) + '.png' 
                ds_month = ds_summer_mean[var].sel(sim=sim, year=slice(2001,2020))
                ds_stdev = ds_summer_stdev[var].sel(sim=sim, year=slice(2001,2020))
                with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                    print('ds_month:', file=pf)
                    print(ds_month, file=pf)
                    print(ds_stdev, file=pf)
                fig_year = plt.figure(figsize=(8,6))
                for mod in ds_month['model'].values:
                    ds_sub_time = ds_month['year'].values
                    ds_sub_mod_var = ds_month.sel(model=mod).values
                    ds_sub_mod_sd = ds_stdev.sel(model=mod).values
                    with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                        print('ds_month for ' + str(mod) + ':', file=pf)
                        print(mod, file=pf)
                        print(ds_sub_time, file=pf)
                        print(ds_sub_mod_var, file=pf)
                        print(ds_sub_mod_sd, file=pf)
                    plt.errorbar(x=ds_sub_time, y=ds_sub_mod_var, yerr=ds_sub_mod_sd, fmt='o', label=mod)
                mod_time = ds_summer_ensemble['year'].sel(year=slice(2001,2020)).values 
                mod_var = ds_summer_ensemble[var].sel(sim=sim, year=slice(2001,2020)).values 
                with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                    print('ds_month_mod for ' + str(sim) + ':', file=pf)
                    print(mod_time, file=pf)
                    print(mod_var, file=pf)
                plt.plot(mod_time, mod_var, linestyle='--', color='black', label='ensemble')
                plt.xlabel('Year')
                y_lab = 'Mean Summer ' + str(var) + ' (' + units + ')' 
                plt.ylabel(y_lab) 
                plt.legend() 
                plt.savefig(out_year, dpi=300)
                plt.close(fig_year)
                ##############################################################
                if sim in ['otc', 'sf']:
                    with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                        print('starting season_diff plots -- summer', file=pf)
                    if sim == 'otc':
                        out_month = config['output_dir'] + 'figures/'+ var + '/Pan-Arctic_mean_' + var + '_seasonal_diff_' + str(sim) + '.png' 
                        ds_month = ds_otc_month_mod_normalized[var] * 100 #.sel(model=['CLM5','CLM5-ExIce','ELM2-NGEE','JSBACH','ecosys','ELM1-ECA'])
                        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                            print('ds_month for diff plots:' + str(sim) + ':', file=pf)
                            print(ds_month, file=pf)
                        #mod_time = ds_otc_month_ensemble['month'].values 
                        #mod_var = ds_otc_month_ensemble[var].values
                    elif sim == 'sf':
                        out_month = config['output_dir'] + 'figures/'+ var + '/Pan-Arctic_mean_' + var + '_seasonal_diff_' + str(sim) + '.png' 
                        ds_month = ds_sf_month_mod_normalized[var] * 100 #.sel(model=['CLM5','CLM5-ExIce','ELM2-NGEE','JSBACH','ecosys','ELM1-ECA'])
                        with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                            print('ds_month for diff plots:' + str(sim) + ':', file=pf)
                            print(ds_month, file=pf)
                        #mod_time = ds_sf_month_ensemble['month'].values 
                        #mod_var = ds_sf_month_ensemble[var].values
                    #y_min = min(ds_otc_month_mean[var].min().values, ds_sf_otc_month_mean[var].min().values)
                    #y_max = max(ds_otc_month_mean[var].max().values, ds_sf_otc_month_mean[var].max().values)
                    
                    #mods = list(ds_month['model'].values)
                    #mod_num = len(mods)
                    #years = np.unique(ds_month['year'].values)
                    #years_num = len(years)
                    #plt_shapes = ["o","v","^","<",">","s","p","*","+","D","x"]
                    #plt_colors = matplotlib.get_cmap('viridis', years_num)
                    #shape_dict = dict(zip(mods, plt_shapes[:mod_num]))
                    #color_dict = dict(zip(years, plt_colors))
                    fig_month_diff = plt.figure(figsize=(8,6))
                    for mod in ds_month['model'].values:
                        ds_sub_time = ds_month['month'].values
                        ds_sub_mod_var = ds_month.sel(model=mod).values
                        plt.plot(ds_sub_time, ds_sub_mod_var, label=mod)
                    #plt.plot(mod_time, mod_var, linestyle='--', color='black', label='ensemble')  
                    plt.xlabel('Month', fontsize=16)  
                    y_lab = str(var) + ' Change from Baseline Mean (%)'
                    plt.ylabel(y_lab, fontsize=16)  
                    plt.legend()
                    #plt.ylim((-0.5, 1.0))
                    plt.savefig(out_month, dpi=300)
                    plt.close(fig_month_diff)
                #####
                if sim in ['sf']:
                    with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                        print('starting season_diff plots -- combined', file=pf)
                    out_month = config['output_dir'] + 'figures/'+ var + '/Pan-Arctic_mean_' + var + '_seasonal_diff_combined.png' 
                    ds_month_otc = ds_otc_seasonal_ef[var]
                    ds_month_sf = ds_sf_seasonal_ef[var]
                    y_min = min(ds_month_otc.min().values, ds_month_sf.min().values)
                    y_max = max(ds_month_otc.max().values, ds_month_sf.max().values)
                    fig_month_diff_both = plt.figure(figsize=(8,6))
                    for mod in ds_month['model'].values:
                        ds_sub_time_otc = ds_month_otc['month'].values
                        ds_sub_mod_var_otc = ds_month_otc.sel(model=mod).values
                        ds_sub_time_sf = ds_month_sf['month'].values
                        ds_sub_mod_var_sf = ds_month_sf.sel(model=mod).values
                        plt.plot(ds_sub_time_otc, ds_sub_mod_var_otc, label=mod, linestyle='solid')
                        plt.plot(ds_sub_time_sf, ds_sub_mod_var_sf, label=mod, linestyle='dashed')
                    #plt.plot(mod_time, mod_var, linestyle='--', color='black', label='ensemble')  
                    plt.xlabel('Month')  
                    y_lab = 'Delta Mean Monthly ' + str(var) + ' (' + units + ')'
                    plt.ylabel(y_lab)  
                    plt.legend()
                    plt.ylim((y_min, y_max))
                    plt.savefig(out_month, dpi=300)
                    plt.close(fig_month_diff_both)
                ##############################################################
                out_month = config['output_dir'] + 'figures/'+ var + '/Pan-Arctic_mean_' + var + '_seasonal_' + str(sim) + '.png' 
                ds_month = ds_geo_month_means[var].sel(sim=sim)
                with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                    print('ds_month:', file=pf)
                    print(ds_month, file=pf)
                fig_month = plt.figure(figsize=(8,6))
                for mod in ds_month['model'].values:
                    ds_sub_time = ds_month['month'].values
                    ds_sub_mod_var = ds_month.sel(model=mod).values
                    with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                        print('ds_month for ' + str(mod) + ':', file=pf)
                        print(mod, file=pf)
                        print(ds_sub_time, file=pf)
                        print(ds_sub_mod_var, file=pf)
                    plt.plot(ds_sub_time, ds_sub_mod_var, label=mod)
                mod_time = ds_geo_month_mod_mean['month'].values 
                mod_var = ds_geo_month_mod_mean[var].sel(sim=sim).values 
                with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                    print('ds_month_mod for ' + str(sim) + ':', file=pf)
                    print(mod_time, file=pf)
                    print(mod_var, file=pf)
                plt.plot(mod_time, mod_var, linestyle='--', color='black', label='ensemble')  
                plt.xlabel('Month')  
                y_lab = 'Mean Monthly ' + str(var) + ' (' + units + ')' 
                plt.ylabel(y_lab)  
                plt.legend()
                plt.savefig(out_month, dpi=300)
                plt.close(fig_month)
                ##############################################################
                out_time = config['output_dir'] + 'figures/'+ var + '/Pan-Arctic_mean_' + var + '_timeseries_' + str(sim) + '.png' 
                ds_time = ds_time_means[var].sel(sim=sim, time=slice('2001-01-01','2020-12-31'))
                fig_time = plt.figure(figsize=(8,6))
                for mod in ds_time['model'].values:
                    plt.plot(ds_time['time'].values, ds_time.sel(model=mod).values, label=mod)
                plt.plot(ds_time_mod_mean['time'].sel(time=slice('2001-01-01','2020-12-31')).values, \
                         ds_time_mod_mean[var].sel(sim=sim, time=slice('2001-01-01','2020-12-31')).values, \
                         linestyle='--', color='black', label='ensemble')  
                plt.xlabel('Date')  
                y_lab = 'Mean Daily ' + str(var) + ' (' + units + ')' 
                plt.ylabel(y_lab)  
                plt.legend()
                plt.savefig(out_time, dpi=300)
                plt.close(fig_time)
                #####################################
                # for year in [2001, 2005, 2010, 2020]: #range(2000,2022):
                #     for month in range(1,13):
                #         if var in ['NEE']:
                #             # for mod in ds_geo_mean_timeseries['model'].values:
                #             #     out_geo = config['output_dir'] + 'figures/' + var + '/Regional_harmonized_'+ var +'_brwn-green_'+str(mod)+'_'+str(sim)+'_'+str(year)+'_'+str(month)+'.png' 
                #             #     geodiff = ds_geo_mean_timeseries[var].sel(sim=sim, year=2010, month=month, model=mod).copy(deep=True)
                #             #     with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                #             #         print('NEE subset from ds_geo_mean_timeseries:' + str(mod) + ':', file=pf)
                #             #         print(geodiff, file=pf)
                #             #     # Set figure size
                #             #     fig_geo_diff = plt.figure(figsize=(8,6))
                #             #     # Set the axes using the specified map projection
                #             #     ax=plt.axes(projection=ccrs.Orthographic(0,90))
                #             #     # Make a mesh plot
                #             #     cs=ax.pcolormesh(geodiff['lon'], geodiff['lat'], geodiff, clim=(-10,10), transform = ccrs.PlateCarree(), cmap='BrBG')
                #             #     ax.coastlines()
                #             #     ax.gridlines()
                #             #     cbar = plt.colorbar(cs,shrink=0.7,location='left',label=var)
                #             #     ax.yaxis.set_ticks_position('left')
                #             #     ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
                #             #     add_circle_boundary(ax)
                #             #     plt.savefig(out_geo, dpi=300)
                #             #     plt.close(fig_geo_diff)
                #             # xarray plot
                #             out_geo = config['output_dir'] + 'figures/' + var + '/Regional_harmonized_'+ var +'_brwn-green_xarray_'+str(sim)+'_'+str(year)+'_'+str(month)+'.png' 
                #             ds_month = ds_geo_mean_timeseries[var].sel(sim=sim, year=year, month=month, model=['CLM5','CLM5-ExIce','ELM2-NGEE','JSBACH','ecosys','ELM1-ECA'])
                #             p = ds_month.plot(x='lon', y='lat', col='model', col_wrap=3, cmap='BrBG_r', \
                #                             transform=ccrs.PlateCarree(), vmin=-10, vmax=10, \
                #                             subplot_kws={"projection": ccrs.Orthographic(0,90)}, \
                #                             cbar_kwargs={'label': 'NEE (gC m-2 day-1)'})
                #             for ax in p.axs.flat:
                #                 ax.coastlines()
                #                 ax.gridlines()
                #                 ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
                #                 add_circle_boundary(ax) 
                #             plt.draw()
                #             plt.savefig(out_geo, dpi=300)
                #             plt.close('all')
                #             if sim in ['otc','sf']:
                #                 out_geo = config['output_dir'] + 'figures/' + var + '/Regional_harmonized_'+ var +'_brwn-green_xarray_diff'+str(sim)+'_'+str(year)+'_'+str(month)+'.png' 
                #                 ds_month = ds_geo_mean_timeseries[var].sel(sim=sim, year=year, month=month, model=['CLM5','CLM5-ExIce','ELM2-NGEE','JSBACH','ecosys','ELM1-ECA']) - \
                #                            ds_geo_mean_timeseries[var].sel(sim='b2', year=year, month=month, model=['CLM5','CLM5-ExIce','ELM2-NGEE','JSBACH','ecosys','ELM1-ECA'])
                #                 p = ds_month.plot(x='lon', y='lat', col='model', col_wrap=3, cmap='BrBG_r', \
                #                                 transform=ccrs.PlateCarree(), vmin=-5, vmax=5, \
                #                                 subplot_kws={"projection": ccrs.Orthographic(0,90)}, \
                #                                 cbar_kwargs={'label': 'Delta NEE (gC m-2 day-1)'})
                #                 for ax in p.axs.flat:
                #                     ax.coastlines()
                #                     ax.gridlines()
                #                     ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
                #                     add_circle_boundary(ax) 
                #                 plt.draw()
                #                 plt.savefig(out_geo, dpi=300)
                #                 plt.close('all')
                ##############################################################
                if sim in ['otc','sf']:
                    with open(Path(config['output_dir'] + 'figures/debug_cartopy.txt'), 'a') as pf:
                        print('starting geo_diff plots', file=pf)
                    #y_min = min(ds_geo_otc_month_mean[var].min().values, ds_geo_sf_month_mean[var].min().values)
                    #y_max = max(ds_geo_otc_month_mean[var].max().values, ds_geo_sf_month_mean[var].max().values)
                    for month in range(1,13):
                        for mod in ds_geo_otc_month_ef['model'].values:
                            out_geo = config['output_dir'] + 'figures/' + var + '/Regional_harmonized_'+ var +'_'+ str(mod)+'_geodiff_'+str(sim)+'_month'+str(month)+'.png' 
                            if sim == 'otc':
                                geodiff = ds_geo_otc_month_ef[var].sel(year=2010, month=month, model=mod).copy(deep=True)
                            if sim == 'sf':
                                geodiff = ds_geo_sf_month_ef[var].sel(year=2010, month=month, model=mod).copy(deep=True)
                            # Set figure size
                            fig_geo_diff = plt.figure(figsize=(8,6))
                            # Set the axes using the specified map projection
                            ax=plt.axes(projection=ccrs.Orthographic(0,90))
                            # Make a mesh plot
                            cs=ax.pcolormesh(geodiff['lon'], geodiff['lat'], geodiff, transform = ccrs.PlateCarree(), cmap='viridis')
                            ax.coastlines()
                            #ax.gridlines()
                            cbar = plt.colorbar(cs,shrink=0.7,location='left',label=var)
                            ax.yaxis.set_ticks_position('left')
                            ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
                            add_circle_boundary(ax)
                            plt.savefig(out_geo, dpi=300)
                            plt.close(fig_geo_diff)
                ##############################################################
                for season in [[ds_geo_means[var],'annual'], [ds_geo_summer_means[var],'summer'], [ds_geo_winter_means[var],'winter']]:
                    ###############################################################
                    # create outfile
                    out_sgeom = config['output_dir'] + 'figures/' + var + '/Regional_harmonized_'+ var +'_'+season[1]+'_geomean_'+str(sim)+'.png' 
                    # subset seasonal data to perturbation simulation of interest
                    sgeom = season[0].sel(sim=sim).copy(deep=True)
                    # Set figure size
                    fig_sgeom = plt.figure(figsize=(8,6))
                    # Set the axes using the specified map projection
                    ax=plt.axes(projection=ccrs.Orthographic(0,90))
                    # Make a mesh plot
                    cs=ax.pcolormesh(sgeom['lon'], sgeom['lat'], sgeom, clim=(min_geo_means,max_geo_means),
                                transform = ccrs.PlateCarree(),cmap='viridis')
                    ax.coastlines()
                    #ax.gridlines()
                    cbar = plt.colorbar(cs,shrink=0.7,location='left',label=var)
                    ax.yaxis.set_ticks_position('left')
                    ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
                    add_circle_boundary(ax)
                    plt.savefig(out_sgeom, dpi=300)
                    plt.close(fig_sgeom)
                    ################################################################
                    # create outfile
                    out_sgeosd = config['output_dir'] + 'figures/' + var + '/Regional_harmonized_'+ var +'_'+season[1]+'_geosd_'+str(sim)+'.png' 
                    # subset seasonal data to perturbation simulation of interest
                    sgeosd = season[0].sel(sim=sim).copy(deep=True)
                    # Set figure size
                    fig_sgeosd = plt.figure(figsize=(8,6))
                    # Set the axes using the specified map projection
                    ax=plt.axes(projection=ccrs.Orthographic(0,90))
                    # Make a mesh plot
                    cs=ax.pcolormesh(sgeosd['lon'], sgeosd['lat'], sgeosd, clim=(0,max_geo_stdev),
                                transform = ccrs.PlateCarree(),cmap='Reds')
                    ax.coastlines()
                    #ax.gridlines()
                    cbar = plt.colorbar(cs,shrink=0.7,location='left',label=var)
                    ax.yaxis.set_ticks_position('left')
                    ax.set_extent([-180,180,55,90], crs=ccrs.PlateCarree())
                    add_circle_boundary(ax)
                    plt.savefig(out_sgeosd, dpi=300)
                    plt.close(fig_sgeosd)
                    ###############################################################


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
    with open(Path(config['output_dir'] + config['model_name'] + '/sites/debug_' + sim_type + '.txt'), 'w') as pf:
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
        with open(Path(config['output_dir'] + config['model_name'] + '/sites/debug_' + sim_type + '.txt'), 'a') as pf:
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
            with open(Path(config['output_dir'] + config['model_name'] + '/sites/debug_' + sim_type + '.txt'), 'a') as pf:
                print(ds_sub, file=pf)
                print(site, file=pf)
            # expand a dimension to include site and save to list
            ds_sub = ds_sub.assign_coords({'site': site})
            ds_sub = ds_sub.expand_dims('site')
            ds_sub = ds_sub.reset_coords(['lat','lon'])
            ds_sub['lat'] = ds_sub['lat'].expand_dims('site')
            ds_sub['lon'] = ds_sub['lon'].expand_dims('site')
            ds_sub = ds_sub.chunk({'time': -1})
            with open(Path(config['output_dir'] + config['model_name'] + '/sites/debug_' + sim_type + '.txt'), 'a') as pf:
                print(ds_sub, file=pf)
            ds_list.append(ds_sub)
        with open(Path(config['output_dir'] + config['model_name'] + '/sites/debug_' + sim_type + '.txt'), 'a') as pf:
            print('datasets appended', file=pf)
        # combine site dimension to have multiple sites
        #try:
        ds_sites = xr.combine_by_coords(ds_list)
        #except Exception as error:
        #    with open(Path(config['output_dir'] + config['model_name'] + '/sites/debug_' + sim_type + '.txt'), 'a') as pf:
        #        print(error, file=pf)
        with open(Path(config['output_dir'] + config['model_name'] + '/sites/debug_' + sim_type + '.txt'), 'a') as pf:
            print('merged datasets',file=pf)
            print(ds_sites, file=pf)
        # set zarr compression and encoding
        #compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        dim_chunks = {
            'SoilDepth': -1,
            'time': -1,
            'site': -1}
        ds_sites = ds_sites.chunk(dim_chunks) 
        with open(Path(config['output_dir'] + config['model_name'] + '/sites/debug_' + sim_type + '.txt'), 'a') as pf:
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
        #with open(Path(config['output_dir'] + config['model_name'] + '/sites/debug_' + sim_type + '.txt'), 'a') as pf:
        #    print('past comp',file=pf)
        #    print(comp, file=pf)
        ## encoding
        #encoding = {var: comp for var in ds_sites.data_vars}
        #with open(Path(config['output_dir'] + config['model_name'] + '/sites/debug_' + sim_type + '.txt'), 'a') as pf:
        #    print('past encoding',file=pf)
        #    print(encoding, file=pf)
        # write zarr
        ds_sites.to_zarr(site_file, encoding=encode,  mode="w")
        with open(Path(config['output_dir'] + config['model_name'] + '/sites/debug_' + sim_type + '.txt'), 'a') as pf:
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
    with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/debug_warming.txt'), 'a') as pf:
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
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/debug_warming.txt'), 'a') as pf:
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
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/debug_warming.txt'), 'a') as pf:
                print(ds_sub, file=pf)
            ds_sub = ds_sub[keep_list]
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/debug_warming.txt'), 'a') as pf:
                print(ds_sub, file=pf)
            # add simulation dimension back and expand dims to recreate same NetCDF shape
            ds_sub = ds_sub.assign_coords({'sim': sim_sel})
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/debug_warming.txt'), 'a') as pf:
                print(ds_sub, file=pf)
            ds_sub = ds_sub.expand_dims('sim')
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/debug_warming.txt'), 'a') as pf:
                print(ds_sub, file=pf)
            # append dataset to list for merging
            ds_list.append(ds_sub)
        except Exception as error:
            with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/debug_warming.txt'), 'a') as pf:
                print(error, file=pf)
            pass
    # recreate original dataarray shape
    ds_delta = xr.combine_by_coords(ds_list)
    with open(Path(config['output_dir'] + config['model_name'] + '/sites_sims/debug_warming.txt'), 'a') as pf:
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
    with open(Path(config['output_dir'] + '/combined/debug_2000-2021.txt'), 'a') as pf:
        print(ds_sites, file=pf)
    # write zarr
    out_file_zarr = config['output_dir'] + 'combined/WrPMIP_all_models_sites_2000-2021.zarr'
    #ds_sites.chunk({'time': -1}).to_zarr(out_file, mode="w")
    ds_sites.to_zarr(out_file_zarr, mode="w")
    # also output aggregate netcdf as well
    comp = dict(zlib=True, shuffle=False,\
            complevel=0,_FillValue=None) #config['nc_write']['fillvalue'])
    encoding = {var: comp for var in ds_sites.data_vars}
    out_file_nc = config['output_dir'] + 'combined/WrPMIP_all_models_sites_2000-2021.nc'
    ds_sites.to_netcdf(out_file_nc, mode="w", encoding=encoding, \
            format='NETCDF4_CLASSIC', \
            engine='netcdf4')
    
    
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
        with open(Path(config['output_dir'] + '/combined/debug_1901-2000.txt'), 'a') as pf:
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
    try:
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
        #     ds_mean = ds[var].sel(time=is_summer(ds[var].time.dt.month)).resample(time='A').mean('time')
        #     annual_var_max = np.unique(ds_mean.max().values).max()
        #     annual_var_min = np.unique(ds_mean.min().values).min()
        #     daily_var_max = np.unique(ds[var].max().values).max()
        #     daily_var_min = np.unique(ds[var].min().values).min()
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

        pd_df['ID'] = pd_df['ID'].astype('category')
        pd_df_annual['ID'] = pd_df_annual['ID'].astype('category')
        # create axis labels
        if var == 'TotalResp':
            x_label = r'time (day)'
            y_label = r'Summer Ecosystem Respiration (g C $m^{-2}$ $day^{-1}$)'
        elif var == 'q10':
            x_label = r'time (day)'
            y_label = r'q10 (unitless)'
        with open(Path(config['output_dir'] + 'combined/debug_line.txt'), 'w') as pf:
            with pd.option_context('display.max_columns', None):
                print('daily data:', file=pf)
                print(pd_df.dtypes, file=pf)
                print(pd_df, file=pf)
                print('scaled monthly data subset:', file=pf)
                print(pd_df_annual.dtypes, file=pf)
                print(pd_df_annual, file=pf)
        # plotnine graph daily
        p = ggplot(pd_df, aes(x='time', y=var, group='ID', color='ID')) + \
            labs(x=x_label, y=y_label) + \
            geom_line() + \
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
            #scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y')) + \
            #scale_y_continuous(limits=(annual_var_min,annual_var_max)) + \
            #scale_color_manual(plot_colors) + \
        # output graph
        p.save(filename=file_name+'.png', path=config['output_dir']+'combined/'+out_dir, dpi=300)
        p2.save(filename=file_name+'_annual.png', path=config['output_dir']+'combined/'+out_dir, dpi=300)
    except Exception as error:
        with open(Path(config['output_dir'] + 'combined/debug_line.txt'), 'w') as pf:
            print(error, file=pf)
        

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
        with open(Path(config['output_dir'] + 'combined/debug_scatter.txt'), 'w') as pf:
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
        with open(Path(config['output_dir'] + 'combined/debug_scatter.txt'), 'a') as pf:
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
        with open(Path(config['output_dir'] + 'combined/debug_scatter.txt'), 'a') as pf:
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
                    with open(Path(config['output_dir'] + 'combined/debug_scatter.txt'), 'a') as pf:
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
            with open(Path(config['output_dir'] + 'combined/debug_scatter.txt'), 'a') as pf:
                with pd.option_context('display.max_columns', None):
                    print('daily scale factor:', file=pf)
                    print(pd_df_scaled_daily, file=pf)
        # calculate monthly data fits
        pd_df_scaled_monthly['a'] = np.nan
        pd_df_scaled_monthly['b'] = np.nan
        for name, group in pd_df_scaled_monthly.groupby('ID', observed=True):
            try:
                if groups == 'site':
                    with open(Path(config['output_dir'] + 'combined/debug_scatter.txt'), 'a') as pf:
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
        with open(Path(config['output_dir'] + 'combined/debug_scatter.txt'), 'a') as pf:
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
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'w') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 20):
            print('Healy observations after subset, hamonization:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    # set dimensions to multiindex
    pd_obs = pd_obs.set_index(['time','plot','sim'])
    # remove duplicated values (all from 2018) some error in heidis code 
    pd_obs = pd_obs[~pd_obs.index.duplicated()]
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 20):
            print('duplicated removal:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    # create datetime index
    pd_obs = pd_obs.reset_index()
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('reset index:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    pd_obs['time'] = pd.to_datetime(pd_obs['time'])
    pd_obs['time'] = pd_obs['time'].dt.strftime('%Y-%m-%d')
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('datetime and string edit:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    datetime_index = pd.DatetimeIndex(pd_obs['time'])
    pd_obs['month'] = datetime_index.month
    pd_obs['year'] = datetime_index.year 
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('month and year added:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    pd_obs = pd_obs.set_index(datetime_index)
    pd_obs = pd_obs.drop(columns=['time'])
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('set index with time:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    pd_obs = pd_obs.loc[(pd_obs.index > datetime(year=2009,month=1,day=1)) & (pd_obs.index < datetime(year=2021,month=12,day=31))]
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None, 'display.max_rows', 100):
            print('time subset to 2009-2021:', file=pf)
            print(pd_obs.dtypes, file=pf)
            print(pd_obs, file=pf)
    pd_obs = pd_obs.reset_index()
    pd_obs = pd_obs.set_index(['time','plot','sim'])
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
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
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
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
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
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
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
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
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', None):
            print('Healy observations monthly std:', file=pf)
            print(pd_obs_monthly_std.dtypes, file=pf)
            print(pd_obs_monthly_std, file=pf)
    pd_obs_monthly_mean = pd.merge(pd_obs_monthly_mean, pd_obs_monthly_std, on=['month','sim'])
    #pd_obs_monthly_mean['month'] = pd.DatetimeIndex(pd_obs_monthly_mean['time']).month
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
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
    with open(Path('/projects/warpmip/shared/ted_data/debug_ted.txt'), 'a') as pf:
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
    with open(Path(config['output_dir'] + '/combined/debug_schadel_'+var+'.txt'), 'w') as pf:
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
    with open(Path(config['output_dir'] + '/combined/debug_schadel_'+var+'.txt'), 'a') as pf:
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
    with open(Path(config['output_dir'] + '/combined/debug_schadel_'+var+'.txt'), 'a') as pf:
        with pd.option_context('display.max_columns', 10):
            print('variable resampled to monthly timestep:', file=pf)
            print(da_monthly, file=pf)
    # monthly data
    pd_df_monthly = da_monthly.to_dataframe()
    with open(Path(config['output_dir'] + '/combined/debug_schadel_'+var+'.txt'), 'a') as pf:
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
    with open(Path(config['output_dir'] + '/combined/debug_schadel_'+var+'.txt'), 'a') as pf:
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

# import functions module
import xr_functions as xrfx 
# import python packages
import sys
import itertools
from datetime import datetime
from pathlib import Path
from dask_mpi import initialize
from dask.distributed import Client, wait, performance_report

# main function
def main():
    # read in list of config files for each site
    config = xrfx.read_config(sys.argv[1])
    # print paths
    p_file = "/scratch/jw2636/wrpmip/python_codes/debug.txt"
    xrfx.rmv_file(p_file)
    with open(Path(p_file),"a") as printfile:
        for line in config['config_files']:
            printfile.write(line + '\n')                
    # start dask cluster
    with client:
        # copy, subset, resample netcdfs from each directory
        L = [client.submit(xrfx.full_reanalysis_list, f) for f in config['config_files']]
        # flatten list of returned futures list to parallize subset call
        full_list = []
        for site in L:
            for i in site.result():
                full_list.append(i)
        # subset all files across all of dasks resources
        L2 = [client.submit(xrfx.subset_reanalysis, f) for f in full_list]
        wait(L2)
        # crujra raw data test
        #La = [client.submit(xrfx.raw_dswrf_list, f) for f in config['config_files']]
        ## flatten list of returned futures list to parallize subset call
        #raw_list = []
        #for site in La:
        #    for i in site.result():
        #        raw_list.append(i)
        ## subset all files across all of dasks resources
        #Lb = [client.submit(xrfx.subset_raw_reanalysis, f) for f in raw_list]
        #wait(Lb)
        #Lc = [client.submit(xrfx.concat_dswrf, f) for f in config['config_files']]
        #wait(Lc)
        # for each site, create list of files that will be concatenated based on datayeas
        L3 = [client.submit(xrfx.cru_sitesubset_list, f) for f in config['config_files']]
        # use futures list by site to concat files
        data_list = []
        for site in L3:
            data_list.append(site.result())
        L4 = [client.submit(xrfx.concat_cru_sitesubset, f) for f in data_list]
        wait(L4)
        # combine observation data form csv files into netcdf for each site
        L5 = [client.submit(xrfx.combine_site_observations, f) for f in config['config_files']]
        wait(L5)
        # calculate the multiyear means of all climate forcing variables for each site
        L6 = [client.submit(xrfx.multiyear_means, f) for f in itertools.product(config['config_files'], ["CRUJRA","Obs"])]
        wait(L6)
        # calculate the bias between cru and obs for each site
        L7 = [client.submit(xrfx.bias_calculation, f) for f in itertools.product(config['config_files'], ["ABias","MBias"], ['dw','w','m'])] 
        wait(L7)  
        # use calculated bias to correct entire crujra product for each site
        L8 = [client.submit(xrfx.bias_correction, f) for f in itertools.product(config['config_files'], ["ABias","MBias"], ['dw','w','m'])] 
        wait(L8)  
        # plot graphs for each site
        L9 = [client.submit(xrfx.plot_site_graphs, f) for f in itertools.product(config['config_files'], ['dw','w','m'])] 
        wait(L9) 
    # combine climate observations and corrections
    nc_output_dir = '/projects/warpmip/shared/forcing_data/biascorrected_forcing/'
    xrfx.combine_climate_and_corrections([config['config_files'], nc_output_dir])
    # summarize climate corrections
    #xrfx.correction_summary()
    # create combined netcdf file
    nc_file_out = '/projects/warpmip/shared/forcing_data/biascorrected_forcing/2024-08-03_crujra_bc_1901-2021.nc'
    xrfx.combine_corrected_climate([config['config_files'], nc_file_out])
    # create clm site subset forcing files
    config_clmsites = '/scratch/jw2636/wrpmip/python_codes/config_clmclimate2.json'
    bc_climate_file = nc_file_out
    surf_file = '/projects/warpmip/shared/forcing_data/CRUJRAv2.3/biascorrected_14sites/surfdata_WrPMIP_14sites_CMIP6_simyr2000_c240215.nc'
    xrfx.replace_clm_14site_climate([config_clmsites, bc_climate_file, surf_file])
    # create pdf report        
    #xrfx.climate_pdf_report(config['config_files']) 

if __name__ == '__main__':
    # initialize dask cluster on resources requested by sbatch
    initialize()
    client=Client()
    # call man function
    main()

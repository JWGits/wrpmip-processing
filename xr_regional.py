from dask_mpi import initialize
initialize(interface='ib0', nthreads=1, memory_limit='25G', 
            worker_class='distributed.Worker', 
            local_directory='/tmp/dask_scratch')
import dask
import distributed 
from distributed import Client, wait
import xr_functions as xrfx 
import os
import shutil
import sys
import itertools
import threading
from datetime import datetime
from pathlib import Path

# main function
def main():
    # read in list of config files for each site
    config = xrfx.read_config(sys.argv[1])
    # print paths
    p_file = "/scratch/jw2636/wrpmip/python_codes/regional_debug.txt"
    xrfx.rmv_file(p_file)
    with open(Path(p_file),"a") as printfile:
        for line in config['config_files']:
            printfile.write(line + '\n')            
    # mkdir if it doesnt exist and store/append to each model config dictionary
    model_config_list = []
    for model in config['config_files']:
        mod_con = xrfx.read_config(model)
        mod_con.update({'output_dir': config['output_dir']})
        if Path(mod_con['output_dir'] + 'zarr_output/' + mod_con['model_name']).exists():
            # move existing directory to delete location
            shutil.move(mod_con['output_dir'] + 'zarr_output/' + mod_con['model_name'], config['delete_dir']) 
        else:
            # make model folders
            Path(mod_con['output_dir'] + 'zarr_output/' + mod_con['model_name']).mkdir(parents=True, exist_ok=True)
            Path(mod_con['output_dir'] + 'zarr_output/' + mod_con['model_name']).chmod(0o762)
            Path(mod_con['output_dir'] + 'netcdf_output/' + mod_con['model_name']).mkdir(parents=True, exist_ok=True)
            Path(mod_con['output_dir'] + 'netcdf_output/' + mod_con['model_name']).chmod(0o762)
            Path(mod_con['output_dir'] + 'figures/' + mod_con['model_name']).mkdir(parents=True, exist_ok=True)
            Path(mod_con['output_dir'] + 'figures/' + mod_con['model_name']).chmod(0o762)
        model_config_list.append(mod_con)
    # start subproccess to remove delete_dir
    threading.Thread(target = lambda : xrfx.rmv_dir(config['delete_dir'])).start()
    # start dask cluster
    with client:
        # make folders structure 
        L0 = [client.submit(xrfx.regional_dir_prep, f) for f in model_config_list]
        wait(L0)
        # create list of netcdf files to merge each models regional simulation outputs
        L1 = [client.submit(xrfx.regional_simulation_files, f) for f in itertools.product(model_config_list, ["b1","b2","otc","sf"])]
        full_list = []
        for model_sim in client.gather(L1):
            full_list.append(model_sim)
        # process each models regional simulation outputs towards harmonizable database
        L2 = [client.submit(xrfx.process_simulation_files, f, config) for f in full_list] 
        del full_list, L0, L1
        wait(L2)
        # aggregate each models regional simulation outputs to a single file 
        L_rsims = [client.submit(xrfx.aggregate_regional_sims, f) for f in model_config_list]
        wait(L_rsims)
        del L_rsims
        # harmonize each models combined files to clm5 dimensions by interpolation/extrapolation
        L_rm = [client.submit(xrfx.harmonize_regional_models, f) for f in model_config_list]
        wait(L_rm)
        del L_rm
        # aggregate all harmonized models into single zarr database
        L_rm2 = [client.submit(xrfx.aggregate_regional_models, model_config_list)]
        wait(L_rm2)
        del L_rm2
        # output netcdfs by year for each model, and by year/month for the final harmonized database
        L_rm3 = [client.submit(xrfx.regional_model_zarrs_to_netcdfs, f) for f in itertools.product(model_config_list,range(2000,2021))]
        wait(L_rm3)
        del L_rm3
        L_rm4 = [client.submit(xrfx.harmonized_model_zarr_to_monthly_netcdfs, f) for f in itertools.product(model_config_list, range(2000,2021),
            range(0,12))]
        wait(L_rm4)
        del L_rm4
        # graph regional outputs from harmonized zarr database
        L_rm5 = [client.submit(xrfx.regional_model_graphs, f) for f in [('mean','TotalResp',1)]]
        wait(L_rm5)
        del L_rm5
        L_rm6 = [client.submit(xrfx.regional_model_graphs, f) for f in [('mean','SoilTemp',1)]]
        wait(L_rm6)
        del L_rm6
        L_rm7 = [client.submit(xrfx.regional_model_graphs, f) for f in [('mean','ALT',1)]]
        wait(L_rm7)
        del L_rm7
        L_rm8 = [client.submit(xrfx.regional_model_graphs, f) for f in [('mean','WTD',1)]]
        wait(L_rm8)
        del L_rm8
        ### clear and recreate site subfolders
        #L3 = [client.submit(xrfx.site_dir_prep, f) for f in config['config_files']]
        #wait(L3)
        ## create list of files to process from harmonized regional zarr files to sites for each model with all variables
        #L4 = [client.submit(xrfx.subsample_site_list, f, config['site_gps']) for f in itertools.product(config['config_files'], ["b1","b2","otc","sf"])]
        #full_list = []
        #for model_sim in client.gather(L4):
        #    full_list.append(model_sim)
        ## process each simulation for all models
        #L5 = [client.submit(xrfx.subsample_sites, f) for f in full_list] 
        #del full_list, L2, L4 
        #wait(L5)
        ### create site_sim directories to aggregate comparable simulations (b2,otc,sf) into site netcdfs
        #L6 = [client.submit(xrfx.site_sim_dir_prep, f) for f in config['config_files']]
        #wait(L6)
        ## aggregate b2,otc,sf simulations into site netcdfs
        #L7 = [client.submit(xrfx.aggregate_simulation_types, f) for f in config['config_files']] 
        #wait(L7)
        ### create directories for combined files with all models
        #L8 = client.submit(xrfx.combined_dir_prep, config['config_files'][0])
        #wait(L8)
        ## aggregate all models for warming period (2000-2021) and baseline (1901-2000)
        #L9 = client.submit(xrfx.aggregate_models_warming, config['config_files'])  
        #L10 = client.submit(xrfx.aggregate_models_baseline, config['config_files'])  
        #wait(L9)
        #wait(L10)
        ## process teds data
        #L_obs = client.submit(xrfx.process_ted_data, config)
        #wait(L_obs)
        ## recreate schadel 2018 ERL article figures
        #L_erl = [client.submit(xrfx.schadel_plots_env, var, config, 'schadel_plots') for var in ['TotalResp','ALT','WTD','SoilTemp_10cm']] 
        #wait(L_erl)
        ## exploratory figures
        #sites = list(config['site_gps'].keys())
        #var = ['TotalResp']
        #models = []
        #for con in config['config_files']:
        #    mod_con = xrfx.read_config(con)
        #    models.append(mod_con['model_name'])
        #sims = ['b2','otc','sf']
        #plot_num = 1
        #plot_line_ind = []
        #plot_line_sites = []
        #plot_line_models = []
        #plot_line_sims = []
        #plot_scatter_ind = []
        #plot_scatter_sites = []
        #plot_scatter_models = []
        #plot_scatter_sims = []
        #plot_scatter_delta_ind = []
        #plot_scatter_delta_sites = []
        #plot_scatter_delta_models = []
        #plot_scatter_delta_sims = []
        #for f in [list(i) for i in itertools.product(sites,var,models,sims)]:
        #    f.append(plot_num)
        #    plot_line_ind.append(f)
        #    plot_num += 1
        #for f in [list(i) for i in itertools.product(var,models,sims)]:
        #    f.append(plot_num)
        #    f.append(sites)
        #    f = [f[i] for i in [4,0,1,2,3]]
        #    plot_line_sites.append(f)
        #    plot_num += 1
        #for f in [list(i) for i in itertools.product(sites,var,sims)]:
        #    f.append(plot_num)
        #    f.append(models)
        #    f = [f[i] for i in [0,1,4,2,3]]
        #    plot_line_models.append(f)
        #    plot_num += 1
        #for f in [list(i) for i in itertools.product(sites,var,models)]:
        #    f.append(plot_num)
        #    f.append(sims)
        #    f = [f[i] for i in [0,1,2,4,3]]
        #    plot_line_sims.append(f)
        #    plot_num += 1
        #for f in [list(i) for i in itertools.product(sites,var,models,sims)]:
        #    f.append(plot_num)
        #    plot_scatter_ind.append(f)
        #    plot_num += 1
        #for f in [list(i) for i in itertools.product(var,models,sims)]:
        #    f.append(plot_num)
        #    f.append(sites)
        #    f = [f[i] for i in [4,0,1,2,3]]
        #    plot_scatter_sites.append(f)
        #    plot_num += 1
        #for f in [list(i) for i in itertools.product(sites,var,sims)]:
        #    f.append(plot_num)
        #    f.append(models)
        #    f = [f[i] for i in [0,1,4,2,3]]
        #    plot_scatter_models.append(f)
        #    plot_num += 1
        #for f in [list(i) for i in itertools.product(sites,var,models)]:
        #    f.append(plot_num)
        #    f.append(sims)
        #    f = [f[i] for i in [0,1,2,4,3]]
        #    plot_scatter_sims.append(f)
        #    plot_num += 1
        #var = ['deltaTotalResp'] 
        #sims = ['otc','sf']
        #for f in [list(i) for i in itertools.product(sites,var,models,sims)]:
        #    f.append(plot_num)
        #    plot_scatter_delta_ind.append(f)
        #    plot_num += 1
        #for f in [list(i) for i in itertools.product(var,models,sims)]:
        #    f.append(plot_num)
        #    f.append(sites)
        #    f = [f[i] for i in [4,0,1,2,3]]
        #    plot_scatter_delta_sites.append(f)
        #    plot_num += 1
        #for f in [list(i) for i in itertools.product(sites,var,sims)]:
        #    f.append(plot_num)
        #    f.append(models)
        #    f = [f[i] for i in [0,1,4,2,3]]
        #    plot_scatter_delta_models.append(f)
        #    plot_num += 1
        #for f in [list(i) for i in itertools.product(sites,var,models)]:
        #    f.append(plot_num)
        #    f.append(sims)
        #    f = [f[i] for i in [0,1,2,4,3]]
        #    plot_scatter_delta_sims.append(f)
        #    plot_num += 1
        ## make plotting directories
        #line_plot_dirs = ['line_plots_ind', 'line_plots_sites', 'line_plots_models', 'line_plots_sims']
        #scatter_plot_dirs = ['scatter_plots_ind', 'scatter_plots_sites', 'scatter_plots_models', 'scatter_plots_sims']
        #scatter_delta_plot_dirs = ['scatter_delta_plots_ind', 'scatter_delta_plots_sites', 'scatter_delta_plots_models', 'scatter_delta_plots_sims']
        #L_lines = [client.submit(xrfx.plot_dir_prep, f, config['config_files'][0]) for f in line_plot_dirs]
        #L_scatter = [client.submit(xrfx.plot_dir_prep, f, config['config_files'][0]) for f in scatter_plot_dirs]
        #L_scatter_delta = [client.submit(xrfx.plot_dir_prep, f, config['config_files'][0]) for f in scatter_delta_plot_dirs]
        #wait(L_lines)  
        #wait(L_scatter)  
        #wait(L_scatter_delta)  
        ##submit plots to dask cluster          
        #L11 = [client.submit(xrfx.plotnine_lines, f, config, 'line_plots_ind') for f in plot_line_ind] 
        #L12 = [client.submit(xrfx.plotnine_lines, f, config, 'line_plots_sites') for f in plot_line_sites] 
        #L13 = [client.submit(xrfx.plotnine_lines, f, config, 'line_plots_models') for f in plot_line_models] 
        #L14 = [client.submit(xrfx.plotnine_lines, f, config, 'line_plots_sims') for f in plot_line_sims] 
        #L15 = [client.submit(xrfx.plotnine_scatter, f, config, 'scatter_plots_ind') for f in plot_scatter_ind] 
        #L16 = [client.submit(xrfx.plotnine_scatter, f, config, 'scatter_plots_sites') for f in plot_scatter_sites] 
        #L17 = [client.submit(xrfx.plotnine_scatter, f, config, 'scatter_plots_models') for f in plot_scatter_models] 
        #L18 = [client.submit(xrfx.plotnine_scatter, f, config, 'scatter_plots_sims') for f in plot_scatter_sims] 
        #L19 = [client.submit(xrfx.plotnine_scatter_delta, f, config, 'scatter_delta_plots_ind') for f in plot_scatter_delta_ind] 
        #L20 = [client.submit(xrfx.plotnine_scatter_delta, f, config, 'scatter_delta_plots_sites') for f in plot_scatter_delta_sites] 
        #L21 = [client.submit(xrfx.plotnine_scatter_delta, f, config, 'scatter_delta_plots_models') for f in plot_scatter_delta_models] 
        #L22 = [client.submit(xrfx.plotnine_scatter_delta, f, config, 'scatter_delta_plots_sims') for f in plot_scatter_delta_sims] 
        ## wait on plots to finish
        #wait(L11)
        #wait(L12)
        #wait(L13)
        #wait(L14)
        #wait(L15)
        #wait(L16)
        #wait(L17)
        #wait(L18)
        #wait(L19)
        #wait(L20)
        #wait(L21)
        #wait(L22)
        ###wait(L_del)

if __name__ == '__main__':
    # call client
    client=Client()
    # call main function; pass client info
    main()

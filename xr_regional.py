# using mpi, setup dask cluster using slurm-started mpi processes
# as I understand it, the sbatch call sends the entire script in replicate to all MPI processes
# intialize essentially then takes over each process and props up MPI0 as the lead script
# other processes are then assigned as client/scheduler/workers with the Client() call
# I believe placing initialize() first reduces the overhead of initial processing of the entire script on each parallel mpi process
# I had initialize() after ' __Main__' or within the main() function and it still worked but maybe slower?
from dask_mpi import initialize
initialize(nthreads=2, interface='ib0', local_directory='/scratch/jw2636/dask_scratch/')
# connect client to daskmpi; dask takes over mpi processes
from dask.distributed import Client, wait
client=Client()
# setup client on scheduler node to use port forwarding for dashboard; needed for mpi instantiation of client
from distributed.scheduler import logger
import socket
host = client.run_on_scheduler(socket.gethostname)
port = client.scheduler_info()['services']['dashboard']
login_node_address = "jw2636@monsoon.hpc.nau.edu" # Change this to the address/domain of your login node
logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}")
# import functions module
import xr_functions as xrfx 
# import python packages
import os
import shutil
import sys
import itertools
from datetime import datetime
from pathlib import Path
#import dask

# main function
def main(client):

    # read in list of config files for each site
    config = xrfx.read_config(sys.argv[1])
    # print paths
    p_file = "/scratch/jw2636/wrpmip/python_codes/regional_debug.txt"
    xrfx.rmv_file(p_file)
    with open(Path(p_file),"a") as printfile:
        for line in config['config_files']:
            printfile.write(line + '\n')            
    
    # mkdir if it doesnt exist 
    for model in config['config_files']:
        mod_con = xrfx.read_config(model)
        if Path(mod_con['output_dir']+mod_con['model_name']).exists():
            # move existing directory to delete location
            shutil.move(mod_con['output_dir']+mod_con['model_name'], config['delete_dir']) 
        else:
            # make model folders
            Path(mod_con['output_dir']+mod_con['model_name']).mkdir(parents=True, exist_ok=True)
            Path(mod_con['output_dir']+mod_con['model_name']).chmod(0o762)
    shutil.move(mod_con['output_dir']+'combined', config['delete_dir']) 
    # print client info 
    with open(Path(p_file),"a") as pf:
        print(client.scheduler_info(), file=pf)
    
    # start dask cluster
    with client:
        # start deleting previous folder
        L_del = client.submit(xrfx.rmv_dir, config['delete_dir'])
        ## remove previous folder structure and remake 
        L0 = [client.submit(xrfx.regional_dir_prep, f) for f in config['config_files']]
        wait(L0)
        # create list of netcdf files to concat/merge/copy for each mo del/simulation combination
        L1 = [client.submit(xrfx.regional_simulation_files, f) for f in itertools.product(config['config_files'], ["b1","b2","otc","sf"])]
        full_list = []
        for model_sim in client.gather(L1):
            full_list.append(model_sim)
        # process each simulation for all models
        L2 = [client.submit(xrfx.process_simulation_files, f) for f in full_list] 
        del full_list, L1
        wait(L2)
        # clear and recreate site subfolders
        L3 = [client.submit(xrfx.site_dir_prep, f) for f in config['config_files']]
        wait(L3)
        # create list of files to process from harmonized regional zarr files to sites for each model with all variables
        L4 = [client.submit(xrfx.subsample_site_list, f, config['site_gps']) for f in itertools.product(config['config_files'], ["b1","b2","otc","sf"])]
        full_list = []
        for model_sim in client.gather(L4):
            full_list.append(model_sim)
        # process each simulation for all models
        L5 = [client.submit(xrfx.subsample_sites, f) for f in full_list] 
        del full_list, L2, L3, L4
        wait(L5)
        # create site_sim directories to aggregate comparable simulations (b2,otc,sf) into site netcdfs
        L6 = [client.submit(xrfx.site_sim_dir_prep, f) for f in config['config_files']]
        wait(L6)
        # aggregate b2,otc,sf simulations into site netcdfs
        L7 = [client.submit(xrfx.aggregate_simulation_types, f) for f in config['config_files']] 
        wait(L7)
        # create directories for combined files with all models
        L8 = client.submit(xrfx.combined_dir_prep, config['config_files'][0])
        wait(L8)
        # aggregate all models for warming period (2000-2021) and baseline (1901-2000)
        L9 = client.submit(xrfx.aggregate_models_warming, config['config_files'])  
        L10 = client.submit(xrfx.aggregate_models_baseline, config['config_files'])  
        wait(L9)
        wait(L10)

        # plot figures
        sites = list(config['site_gps'].keys())
        var = ['TotalResp']
        models = []
        for con in config['config_files']:
            mod_con = xrfx.read_config(con)
            models.append(mod_con['model_name'])
        sims = ['b2','otc','sf']
        soild = [0]
        plot_num = 1
        plot_line_ind = []
        plot_line_sites = []
        plot_line_models = []
        plot_line_sims = []
        plot_scatter_ind = []
        plot_scatter_sites = []
        plot_scatter_models = []
        plot_scatter_sims = []
        plot_scatter_soild = []
        plot_scatter_delta_ind = []
        plot_scatter_delta_sites = []
        plot_scatter_delta_models = []
        plot_scatter_delta_sims = []
        plot_scatter_delta_soild = []
        for f in [list(i) for i in itertools.product(sites,var,models,sims)]:
            f.append(plot_num)
            plot_line_ind.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(var,models,sims)]:
            f.append(plot_num)
            f.append(sites)
            f = [f[i] for i in [4,0,1,2,3]]
            plot_line_sites.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(sites,var,sims)]:
            f.append(plot_num)
            f.append(models)
            f = [f[i] for i in [0,1,4,2,3]]
            plot_line_models.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(sites,var,models)]:
            f.append(plot_num)
            f.append(sims)
            f = [f[i] for i in [0,1,2,4,3]]
            plot_line_sims.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(sites,models,sims,soild)]:
            f.append(plot_num)
            plot_scatter_ind.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(models,sims,soild)]:
            f.append(plot_num)
            f.append(sites)
            f = [f[i] for i in [4,0,1,2,3]]
            plot_scatter_sites.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(sites,sims,soild)]:
            f.append(plot_num)
            f.append(models)
            f = [f[i] for i in [0,4,1,2,3]]
            plot_scatter_models.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(sites,models,soild)]:
            f.append(plot_num)
            f.append(sims)
            f = [f[i] for i in [0,1,4,2,3]]
            plot_scatter_sims.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(sites,models,sims)]:
            f.append(plot_num)
            f.append(soild)
            f = [f[i] for i in [0,1,2,4,3]]
            plot_scatter_soild.append(f)
            plot_num += 1
        
        sims = ['otc','sf']
        for f in [list(i) for i in itertools.product(sites,models,sims,soild)]:
            f.append(plot_num)
            plot_scatter_delta_ind.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(models,sims,soild)]:
            f.append(plot_num)
            f.append(sites)
            f = [f[i] for i in [4,0,1,2,3]]
            plot_scatter_delta_sites.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(sites,sims,soild)]:
            f.append(plot_num)
            f.append(models)
            f = [f[i] for i in [0,4,1,2,3]]
            plot_scatter_delta_models.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(sites,models,soild)]:
            f.append(plot_num)
            f.append(sims)
            f = [f[i] for i in [0,1,4,2,3]]
            plot_scatter_delta_sims.append(f)
            plot_num += 1
        for f in [list(i) for i in itertools.product(sites,models,sims)]:
            f.append(plot_num)
            f.append(soild)
            f = [f[i] for i in [0,1,2,4,3]]
            plot_scatter_delta_soild.append(f)
            plot_num += 1
        # make plotting directories
        line_plot_dirs = ['line_plots_ind', 'line_plots_sites', 'line_plots_models', 'line_plots_sims']
        scatter_plot_dirs = ['scatter_plots_ind', 'scatter_plots_sites', 'scatter_plots_models', 'scatter_plots_sims', 'scatter_plots_soild']
        scatter_delta_plot_dirs = ['scatter_delta_plots_ind', 'scatter_delta_plots_sites', 'scatter_delta_plots_models', 'scatter_delta_plots_sims', 'scatter_delta_plots_soild']
        L_lines = [client.submit(xrfx.plot_dir_prep, f, config['config_files'][0]) for f in line_plot_dirs]
        L_scatter = [client.submit(xrfx.plot_dir_prep, f, config['config_files'][0]) for f in scatter_plot_dirs]
        L_scatter_delta = [client.submit(xrfx.plot_dir_prep, f, config['config_files'][0]) for f in scatter_delta_plot_dirs]
        wait(L_lines)  
        wait(L_scatter)  
        wait(L_scatter_delta)  
        #submit plots to dask cluster          
        L11 = [client.submit(xrfx.plotnine_lines, f, config, 'line_plots_ind') for f in plot_line_ind] 
        L12 = [client.submit(xrfx.plotnine_lines, f, config, 'line_plots_sites') for f in plot_line_sites] 
        L13 = [client.submit(xrfx.plotnine_lines, f, config, 'line_plots_models') for f in plot_line_models] 
        L14 = [client.submit(xrfx.plotnine_lines, f, config, 'line_plots_sims') for f in plot_line_sims] 
        L15 = [client.submit(xrfx.plotnine_scatter, f, config, 'scatter_plots_ind') for f in plot_scatter_ind] 
        L16 = [client.submit(xrfx.plotnine_scatter, f, config, 'scatter_plots_sites') for f in plot_scatter_sites] 
        L17 = [client.submit(xrfx.plotnine_scatter, f, config, 'scatter_plots_models') for f in plot_scatter_models] 
        L18 = [client.submit(xrfx.plotnine_scatter, f, config, 'scatter_plots_sims') for f in plot_scatter_sims] 
        L19 = [client.submit(xrfx.plotnine_scatter, f, config, 'scatter_plots_soild') for f in plot_scatter_soild] 
        L20 = [client.submit(xrfx.plotnine_scatter_delta, f, config, 'scatter_delta_plots_ind') for f in plot_scatter_delta_ind] 
        L21 = [client.submit(xrfx.plotnine_scatter_delta, f, config, 'scatter_delta_plots_sites') for f in plot_scatter_delta_sites] 
        L22 = [client.submit(xrfx.plotnine_scatter_delta, f, config, 'scatter_delta_plots_models') for f in plot_scatter_delta_models] 
        L23 = [client.submit(xrfx.plotnine_scatter_delta, f, config, 'scatter_delta_plots_sims') for f in plot_scatter_delta_sims] 
        L24 = [client.submit(xrfx.plotnine_scatter_delta, f, config, 'scatter_delta_plots_soild') for f in plot_scatter_delta_soild] 
        # wait on plots to finish
        wait(L11)
        wait(L12)
        wait(L13)
        wait(L14)
        wait(L15)
        wait(L16)
        wait(L17)
        wait(L18)
        wait(L19)
        wait(L20)
        wait(L21)
        wait(L22)
        wait(L23)
        wait(L24)
        #wait(L_del)

if __name__ == '__main__':
    # call main function; pass client info
    main(client)

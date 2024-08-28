from dask_mpi import initialize
import dask
import distributed 
from distributed import Client, wait
import xr_functions as xrfx 
import os
import shutil
import sys
import time
import itertools
from mpi4py import MPI
from datetime import datetime
from pathlib import Path

# main function
def main():
    # initialize dask mpi
    initialize(interface='ib0', nthreads=1, memory_limit='10G', worker_class='distributed.Worker', local_directory='/tmp/dask_scratch', exit=False)
    # return all main functions that are not rank 1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 1:
        return
    # read in top level config file
    config = xrfx.read_config(sys.argv[1])
    # remove delete_dir (just in case), mv old regional output to delete_dir, remove delete_dir
    xrfx.rmv_dir(config['delete_dir'])
    xrfx.mv_dir(config['elm_output_dir'], config['delete_dir'])
    xrfx.mv_dir(config['lpj_output_dir'], config['delete_dir'])
    xrfx.mv_dir(config['lpj2_output_dir'], config['delete_dir'])
    xrfx.rmv_dir(config['delete_dir'])
    # create list of models config files to process, add global config info to each, create output folder locations
    lpj_config = []
    for model in config['lpj_config_file']:
        mod_con = xrfx.read_config(model)
        mod_con.update({'output_dir': config['lpj_output_dir']})
        mod_con.update({'nc_read': config['nc_read']})
        mod_con.update({'nc_write': config['nc_write']})
        lpj_config.append(mod_con)
    elm_config = []
    for model in config['elm_config_file']:
        mod_con = xrfx.read_config(model)
        mod_con.update({'output_dir': config['elm_output_dir']})
        mod_con.update({'nc_read': config['nc_read']})
        mod_con.update({'nc_write': config['nc_write']})
        elm_config.append(mod_con)
    lpj2_config = []
    for model in config['lpj2_config_file']:
        mod_con = xrfx.read_config(model)
        mod_con.update({'output_dir': config['lpj2_output_dir']})
        mod_con.update({'nc_read': config['nc_read']})
        mod_con.update({'nc_write': config['nc_write']})
        lpj2_config.append(mod_con)
    # start dask cluster
    with Client() as client:
        L0 = [client.submit(xrfx.list_lpjguessml, f) for f in itertools.product(lpj_config, ["b1","b2","otc","sf"])]
        full_list_lpjguessml = []
        for i in L0:
            for j in i.result():
                full_list_lpjguessml.append(j)
        del L0
        L1 = [client.submit(xrfx.list_lpjguessml, f) for f in itertools.product(lpj2_config, ["b1","b2","otc","sf"])]
        full_list_lpjguess = []
        for i in L1:
            for j in i.result():
                full_list_lpjguess.append(j)
        del L1
        L2 = [client.submit(xrfx.rechunk_lpjguessml, f, lpj_config[0]) for f in full_list_lpjguessml] 
        L3 = [client.submit(xrfx.rechunk_lpjguess, f, lpj2_config[0]) for f in full_list_lpjguess] 
        L4 = [client.submit(xrfx.rechunk_elmeca, f) for f in itertools.product(elm_config, ["b2","otc","sf"])] 
        wait(L2)
        wait(L3)
        wait(L4)
        # spin down client
        while len(client.scheduler_info()['workers']) < 1:
            time.sleep(1)
        client.retire_workers()
        time.sleep(1)
        client.shutdown()

if __name__ == '__main__':
    main()

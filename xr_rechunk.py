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
    initialize(interface='ib0', nthreads=1, memory_limit='6G', worker_class='distributed.Worker', local_directory='/tmp/dask_scratch', exit=False)
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
    xrfx.mv_dir(config['jules_output_dir'], config['delete_dir'])
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
    jules_config = []
    for model in config['jules_config_file']:
        mod_con = xrfx.read_config(model)
        mod_con.update({'output_dir': config['jules_output_dir']})
        mod_con.update({'nc_read': config['nc_read']})
        mod_con.update({'nc_write': config['nc_write']})
        jules_config.append(mod_con)
    # start dask cluster
    def batched(iterable, n):
        if n < 1:
            raise ValueError('n must be at least one')
        iterator = iter(iterable)
        while batch := tuple(itertools.islice(iterator, n)):
            yield batch
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
        L2 = [client.submit(xrfx.list_jules_daily, f) for f in itertools.product(jules_config, ["b1","b2","otc","sf"])]
        list_jules_daily = []
        for i in L2:
            for j in i.result():
                list_jules_daily.append(j)
        del L2
        L3 = [client.submit(xrfx.list_jules_monthly, f) for f in itertools.product(jules_config, ["b1","b2","otc","sf"])]
        list_jules_monthly = []
        for i in L3:
            for j in i.result():
                list_jules_monthly.append(j)
        del L3
        L4 = [client.submit(xrfx.rechunk_lpjguessml, f, lpj_config[0]) for f in full_list_lpjguessml] 
        L5 = [client.submit(xrfx.rechunk_lpjguess, f, lpj2_config[0]) for f in full_list_lpjguess] 
        L6 = [client.submit(xrfx.rechunk_elmeca, f) for f in itertools.product(elm_config, ["b2","otc","sf"])] 
        wait(L4)
        wait(L5)
        wait(L6)
        # copy elm b2 to b1
        dir_in = '/scratch/jw2636/processed_outputs/ELM-ECA/b2/'
        dir_out = '/scratch/jw2636/processed_outputs/ELM-ECA/b1/'
        zstores = client.submit(xrfx.elm_zstore_list, dir_in, dir_out).result()
        wait([client.submit(xrfx.zstore_copy, f) for f in zstores])
        with open(Path(jules_config[0]['output_dir'] + 'debug_JULES_rechunk_monthly.txt'), 'w') as pf:
            print('start monthly jules rechunk', file=pf)
        with open(Path(jules_config[0]['output_dir'] + 'debug_JULES_rechunk_daily.txt'), 'w') as pf:
            print('start daily jules rechunk', file=pf)
        #for chunk in batched(list_jules_monthly, 40):
        wait([client.submit(xrfx.rechunk_jules_monthly, f, jules_config[0]) for f in list_jules_monthly]) 
        #for chunk in batched(list_jules_daily, 40):
        wait([client.submit(xrfx.rechunk_jules_daily, f, jules_config[0]) for f in list_jules_daily]) 
        # spin down client
        while len(client.scheduler_info()['workers']) < 1:
            time.sleep(1)
        client.retire_workers()
        time.sleep(1)
        client.shutdown()

if __name__ == '__main__':
    main()

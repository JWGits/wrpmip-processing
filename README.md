# wrpmip-processing

Python code to parallel-process the Warming Permafrost Model Intercomparison Project (WrPMIP) simulation outputs using xarray and dask libraries.

## General workflow

This code is developed for HPC systems utilizing the Slurm scheduler (though other HPC/scheduler combinations that utilize Openmpi should work with bash script modifications). An Sbatch call is made with the bash script of interest and associated JSON configuration file.  

```
sbatch xr_regional.bash config_models.json
```

The bash file describes the resources requested from the scheduler, activates needed modules/libraries, and calls the associated python code using Openmpi's mpirun command, with configuration information (e.g. file locations, models to process, etc.) passed through to the python call as shown below.

```
mpirun -np 40 --map-by numa --bind-to core python3 xr_regional.py $1
```

The above code maps 40 processes to cores by numa setup but this will need to be tested on each HPC for proper use and edited to align with the SBATCH resource request. Inputs added to the command line after the bash script in an sbatch call are stored as $1 through $x and used here to pass the JSON config file through to the python call.

Within the python code a Dask scheduler, Dask client, and Dask workers are created based on MPI rank (in that order) using the dask_mpi.initialize() function. A Dask cluster is subsequently created within the python code using the dask.distributed.Client() function and all work to be done is parallelized using Dask's client.submit() framework. Functions are currently all listed in xr_functions.py and read in by an import statement at the top of the python script. Separating functions into groups based on their use in different python scripts is planned but not complete.  

Figures and graphs are generally created using plotnine, which is a nearly complete port of R's ggplot2 package into python. The graphing functions used are currently inefficient as they load and manipulate data for each figure. This will eventually be refactored into another processing function preceeding the graphing function calls when time permits.

## Monitoring the dask cluster

The first node in the resource list will be hosting the scheduler. You can ssh into the compute node (depending on your HPC setup and allowances) using windows powershell (or other shell) to forward the scheduler's communications to your own machine like below:

```
ssh -N -L 8787:nodename:8787 signin@hpc.address
```

You will be prompted for your password to sign in to the HPC node. The -N only forwards the port and doesn't execute commands while the -L defines the connection address. Dask schedulers will typically use 8787 as the port for sharing the dask dashboard. If you have many Dask users on your HPC you may need to configure the client to use a different port number when being setup and then use that new port number in place of 8787 when connecting to the compute node. You subsequently can view the dashboard in your browser using localhost as shown below.

```
localhost:8787
```

## Cautions and caveats

Dask is extremely useful for parallelizing heterogeneous jobs that don't work well in simplier parallel processing schemes like jobarrays. It is also quite powerful when used with libraries like xarray that can use dask-arrays to load subsets of NetCDF files and subsequently process larger-than-memory files with no file splittng or code modifications. The catch being that Dask clusters are fickle and easy to crash, especially due to memory overflows, though dask is under active development to address some of these concerns. Dask clusters also use ethernet or inifiniband resources to communicate between nodes across the HPC cluster which can be a problem for HPC resource management if parallel work is dependent on other processes/workers. The general advice is to avoid Dask if other simpler parrallelization schemes like jobarrays can be used effectively.

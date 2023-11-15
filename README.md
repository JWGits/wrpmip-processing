# wrpmip-processing

Python code to parallel-process WrPMIP simulation outputs using xarray and dask libraries.

## General workflow

Code is developed for HPC systems utilizing the Slurm scheduler (though other HPC/scheduler combinations that utilize Openmpi should work). An Sbatch call is made with the bash script of interest and associated json configuration file.  

```
sbatch xr_regional.bash config_models.json
```

The bash file describes the resources requested by the scheduler, activates needed modules/libraries, and calls the associate python code using Openmpi's mpirun command, with the configuration information (e.g. file locations, models to process, etc) passed through to the mpirun call as shown below.

```
mpirun -np 40 --map-by numa --bind-to core python3 xr_regional.py $1
```

The above code maps 40 processes to cores by numa setup but this is something that will need to be tested on each HPC for proper use and edited to align with the SBATCH resource call. Inputs added to the command line after the bash script in an sbatch call are stored as $1 through $x and used here to pass the json config file through to the python call.

Within the python code a dask scheduler, dask client, and dask workers are created based on MPI ranks (in that order) using dask-mpi's Initialize() function. A dask cluster is subsequently created using the Client() function and all work to be done is parallelized using Dask's client.submit() framework. Functions are currently all listed in xr_functions.py and read in by an import statement at the top of the python script. Separating functions into groups based on their use in different python scripts is planned but not complete.  

## Monitoring the dask cluster

The first node in the resource list will be hosting the scheduler. You can ssh into the compute node (depending on your HPC setup and allowances) using windows powershell to forward the schedulers port to your own machine like below:

```
ssh -N -L 8787:nodename:8787 signin@hpc.address
```

You will be prompted for your password to sign in to the HPC node. The -N only forwards the port and doesn't execute commands while the -L defines the connection address. Dask schedulers will typically use 8787 as the port for sharing the dask dashboard. If you have many dask users on your HPC you will need to configure the client to use a different port number when being setup and then use that new port number in place of 8787 when logining into the compute node. You subsequently can view the dashboard using the localhost command and the port number in your browser as shown below.

```
localhost:8787
```

# wrpmip-processing
Python code to parallel-process WrPMIP simulation outputs using xarray and dask libraries.

## General workflow
Code is developed for HPC systems utilizing the Slurm scheduler (though other HPC/scheduler combinations that utilize Openmpi should work). An Sbatch call is made with the bash script of interest and associated json configuration file.  

```
sbatch xr_regional.bash config_models.json
```

The bash file describes the resources requested by the scheduler, activates needed modules/libraries, and calls the associate python code using Openmpi's mpirun command, with the configuration information (e.g. file locations, models to process, etc) pass through to the mpirun call as shown below.

```
mpirun -np 40 --map-by numa, --bind-to core python3 xr_regional.py $1
```

The above code maps 40 processes to cores by numa setup but this is something that will need to be tested on HPCs for proper setup. The inputs passed to a bash script on the command line are stored as $1 through however many are added. Here the json config file is passed through to the mpirun/python call.

Within the python code a dask scheduler, client, and workers are created based on MPI ranks using dask-mpi's Initialize() function. A dask cluster is subsequently created using the Client() function and work to be done is parallelized using Dask's client.submit() framework. Functions are currently all listed in xr_functions.py and read in by an import statement at the top of the python script. Separating functions up into groups based on their purpose is planned but not complete.  

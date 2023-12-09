#!/bin/bash
#SBATCH -J dask-mpi
#SBATCH -p core
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH -t 01:00:00

# import bash shell environment
eval "$(conda shell.bash hook)"
# load anaconda module and conda env
module load openmpi/4.1.4
module load anaconda3
conda activate xarray2
# call python scripts
mpirun -np 40 --map-by numa --bind-to core python3 xr_regional.py $1
wait

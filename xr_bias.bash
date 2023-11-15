#!/bin/bash
#SBATCH -J dask-mpi
#SBATCH -p core
#SBATCH --nodes=7
#SBATCH --mem=20G
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=1
#SBATCH -t 00:15:00

# import bash shell environment
eval "$(conda shell.bash hook)"
# load anaconda module and conda env
module load openmpi
module load anaconda3
conda activate xarray
# call python scripts
mpirun -np 35 python3 xr_bias.py $1
wait

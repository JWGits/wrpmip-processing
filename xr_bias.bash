#!/bin/bash
#SBATCH -J dask-mpi
#SBATCH -p core
#SBATCH --nodes=14
#SBATCH --ntasks-per-node=1
#SBATCH --cores-per-socket=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH -t 00:45:00

# import bash shell environment
eval "$(conda shell.bash hook)"
# load anaconda module and conda env
module load openmpi/4.1.4
module load anaconda3
conda activate xarray2
# call python scripts
mpirun -np 14 --bind-to core --map-by socket --report-bindings python3 xr_bias.py $1
wait

#!/bin/bash
#SBATCH -J dask-mpi
#SBATCH -p core
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH -t 00:45:00

# import bash shell environment
eval "$(conda shell.bash hook)"
# load anaconda module and conda env
module load openmpi/4.1.4
module load anaconda3
conda activate xarray2
# call python scripts
mpirun -np 40 --map-by numa --bind-to core python3 xr_regional.py $1
#mpirun --may-by ppr:2:node --mca btl_tcp_if_include 172.16.3.0/24 --mca btl_base_verbose 100 --mca mpi_preconnect_all true python3 xr_regional.py $1
#--bind-to socket --rank-by socket
#srun -c4 python3 xr_regional.py $1
wait

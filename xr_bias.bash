#!/bin/bash
#SBATCH -J dask-mpi
#SBATCH -p core
#SBATCH --nodes=11
#SBATCH -C sl
#SBATCH --ntasks-per-node=4
#SBATCH --sockets-per-node=2
#SBATCH --cores-per-socket=2
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH -t 00:25:00

# import bash shell environment
eval "$(conda shell.bash hook)"
# load anaconda module and conda env
#module load openmpi/4.1.4
module load anaconda3
conda activate /scratch/jw2636/wrpmip/conda_envs/r_env

# call python scripts
mpirun -np 44 --bind-to core --map-by socket --report-bindings python3 xr_bias.py $1
wait

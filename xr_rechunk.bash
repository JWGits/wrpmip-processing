#!/bin/bash
#SBATCH -J dask-mpi
#SBATCH -p core
#SBATCH -C sl|amd
#SBATCH -t 00:40:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=6
#SBATCH --sockets-per-node=2
#SBATCH --cores-per-socket=3
#SBATCH --ntasks-per-socket=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G

# import bash shell environment
eval "$(conda shell.bash hook)"
# load anaconda module and conda env
module purge
module load anaconda3
conda activate /scratch/jw2636/wrpmip/conda_envs/r_env
module load openmpi/4.1.4
PMIX_MCA_gds=hash
# call python scripts
mpirun -np 60 --bind-to core --map-by socket --report-bindings python3 xr_rechunk.py $1
wait

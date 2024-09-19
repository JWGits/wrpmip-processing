#!/bin/bash
#SBATCH -J dask-mpi
#SBATCH -p core
#SBATCH -C sl
#SBATCH -t 01:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --sockets-per-node=2
#SBATCH --cores-per-socket=2
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40G

# import bash shell environment
eval "$(conda shell.bash hook)"
# load anaconda module and conda env
module purge
module load anaconda3
conda activate /scratch/jw2636/wrpmip/conda_envs/r_env
module load openmpi/4.1.4
PMIX_MCA_gds=hash
# call python scripts
mpirun -np 40 --bind-to core --map-by socket --report-bindings python3 xr_regional.py $1
wait

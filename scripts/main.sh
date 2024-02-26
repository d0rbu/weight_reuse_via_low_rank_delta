#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonestar6 AMD Milan nodes
#
#   *** MPI Job in Normal Queue ***
# 
# Last revised: October 22, 2021
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch milan.mpi.slurm" on a Lonestar6 login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do NOT use mpirun or mpiexec.
#
#   -- Max recommended MPI ranks per Milan node: 128
#      (start small, increase gradually).
#
#   -- If you're running out of memory, try running
#      fewer tasks per node to give each task more memory.
#
#----------------------------------------------------

#SBATCH -J weight-reuse-eval  # Job name
#SBATCH -o %j-out.txt         # Name of stdout output file
#SBATCH -e %j-err.txt         # Name of stderr error file
#SBATCH -p gpu-a100           # Queue (partition) name
#SBATCH -N 4                  # Total # of nodes 
#SBATCH -n 12                 # Total # of mpi tasks
#SBATCH -t 48:00:00           # Run time (hh:mm:ss)
#SBATCH --mail-type=all       # Send email at begin and end of job
#SBATCH -A MLL                # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=henryandrecastillo@gmail.com

# Any other commands must follow all #SBATCH directives...
unset PYTHONPATH
conda activate weights
cd $SCRATCH/weight_reuse_via_low_rank_delta

# Launch MPI code... 
ibrun python -m experiment.main         # Use ibrun instead of mpirun or mpiexec
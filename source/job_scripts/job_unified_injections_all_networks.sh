#!/bin/bash
#
#SBATCH --job-name=jobUnifiedInjs
#SBATCH --output=stdout_job_unified_injections_all_networks.txt
#SBATCH --error=stderr_job_unified_injections_all_networks.txt
#
#SBATCH --ntasks=1
#SBATCH --time=08:00:00 # HH:MM:SS
#SBATCH --mem-per-cpu=500 # MB, determined from mprof
#
#SBATCH --array=1-2048 

# argument/s: task_id (handle everything else inside python)
srun python3 /fred/oz209/jgardner/CEonlyPony/source/run_unified_injections_as_task.py $SLURM_ARRAY_TASK_ID

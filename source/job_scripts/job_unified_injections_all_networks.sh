#!/bin/bash
#
#SBATCH --job-name=jobUnifiedInjs
#SBATCH --output=stdout_job_unified_injections_all_networks.txt
#SBATCH --error=stderr_job_unified_injections_all_networks.txt
#
#SBATCH --ntasks=1
#SBATCH --time=04:00:00 # HH:MM:SS
#SBATCH --mem-per-cpu=250 # MB, determined from mprof (3.5 hr, 160 MB is likely enough)
#
#SBATCH --array=1-2048 

# argument/s: task_id (handle everything else inside python)
srun python3 /fred/oz209/jgardner/CEonlyPony/source/run_unified_injections_as_task.py $SLURM_ARRAY_TASK_ID

#!/bin/bash
#
#SBATCH --job-name=jobUnifiedInjs
#SBATCH --output=slurm_output_files/stdout_job_unified_injections_all_networks_JOB-ID_%A_TASK-ID_%a.txt
#SBATCH --error=slurm_output_files/stderr_job_unified_injections_all_networks_JOB-ID_%A_TASK-ID_%a.txt
#
#SBATCH --ntasks=1
#SBATCH --time=04:00:00 # HH:MM:SS
#SBATCH --mem-per-cpu=250 # MB, determined from mprof (3.5 hr, 160 MB is likely enough)
#
#SBATCH --array=0-2047 # shifted to 1-2048 below

# <https://slurm.schedmd.com/slurm.conf.html> --> The maximum job array task index value will be one less than MaxArraySize to allow for an index value of zero. 
# python scripts run with task_id's starting from zero
let "ONE_INDEXED_TASK_ID = ${SLURM_ARRAY_TASK_ID} + 1"

# argument/s: task_id (handle everything else inside python)
srun python3 /fred/oz209/jgardner/CEonlyPony/source/run_unified_injections_as_task.py ${ONE_INDEXED_TASK_ID}

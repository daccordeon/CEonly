#!/bin/bash
#
#SBATCH --job-name=tmp_jobUnifiedInjs
#SBATCH --output=slurm_output_files/stdout_tmp_job_unified_injections_all_networks_JOB-ID_%A_TASK-ID_%a.txt
#SBATCH --error=slurm_output_files/stderr_tmp_job_unified_injections_all_networks_JOB-ID_%A_TASK-ID_%a.txt
#
#SBATCH --ntasks=1
#SBATCH --time=04:00:00 # HH:MM:SS
#SBATCH --mem-per-cpu=250 # MB, determined from mprof (3.5 hr, 160 MB is likely enough)
#SBATCH --array=0-1

TEST_INDS=(54 145)
let "TEST_IND = ${TEST_INDS[$SLURM_ARRAY_TASK_ID]}"
# argument/s: task_id (handle everything else inside python)
srun python3 /fred/oz209/jgardner/CEonlyPony/source/run_unified_injections_as_task.py ${TEST_IND}

#!/bin/bash
#
#SBATCH --job-name=mprofUnifiedInjs
#SBATCH --output=slurm_output_files/stdout_job_memory_profile_unified_injections_JOB-ID_%A_TASK-ID_%a.txt
#SBATCH --error=slurm_output_files/stderr_job_memory_profile_unified_injections_JOB-ID_%A_TASK-ID_%a.txt
#
#SBATCH --ntasks=1
#SBATCH --time=08:00:00 # HH:MM:SS
#SBATCH --mem-per-cpu=1000 # MB
#
#SBATCH --array=1-4

TEST_INDS=(1 1024 1025 2048)
# slurm uses a base index of 1 but bash uses 0
let "TEST_IND = ${TEST_INDS[$SLURM_ARRAY_TASK_ID - 1]}"

srun mprof run -o "mprof_plot_unified_injs_${TEST_IND}.dat" /fred/oz209/jgardner/CEonlyPony/source/run_calculate_unified_injections_as_task.py ${TEST_IND}; mprof plot -o "mprof_plot_unified_injs_${TEST_IND}.pdf" "mprof_plot_unified_injs_${TEST_IND}.dat"

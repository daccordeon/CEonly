#!/bin/bash
#
#SBATCH --job-name=mprof_plotters
#SBATCH --output=stdout_job_memory_profile_plot_detection_rate_and_measurement_errors.txt
#SBATCH --error=stderr_job_memory_profile_plot_detection_rate_and_measurement_errors.txt
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00 # HH:MM:SS
#SBATCH --mem-per-cpu=10000 # MB
#
#SBATCH --array=1-4 # 1 network set, 2 science cases, 2 plots

# James Gardner, March 2022

NETWORK_INDEX=0 # 0 for first network set (B&S2022_SIX)
let "OFFSET_TASK_ID = $SLURM_ARRAY_TASK_ID + 4*$NETWORK_INDEX"
srun mprof run -o "mprof_plot_plotters_${OFFSET_TASK_ID}.dat" run_plot_detection_rate_and_measurement_errors.py ${OFFSET_TASK_ID}; mprof plot -o "mprof_plot_plotters_${OFFSET_TASK_ID}.pdf" "mprof_plot_plotters_${OFFSET_TASK_ID}.dat"

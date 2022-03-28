#!/bin/bash
#
#SBATCH --job-name=plotDRandERRs
#SBATCH --output=stdout_job_plot_detection_rate_and_measurement_errors.txt
#SBATCH --error=stderr_job_plot_detection_rate_and_measurement_errors.txt
#
#SBATCH --ntasks=1
#SBATCH --time=00:02:00 # HH:MM:SS, mprof maximum: 40 s
#SBATCH --mem-per-cpu=2000 # MB, mprof maximum: 800 MB
#
#SBATCH --array=1-40 # 10 network sets, 2 science cases with a unique waveform each, 2 plots

# James Gardner, March 2022
srun python3 -u run_plot_detection_rate_and_measurement_errors.py $SLURM_ARRAY_TASK_ID

#!/bin/bash
#
#SBATCH --job-name=plotDRandERRs
#SBATCH --output=stdout_plot_detection_rate_and_measurement_errors.txt
#SBATCH --error=stderr_plot_detection_rate_and_measurement_errors.txt
#
#SBATCH --ntasks=1
#SBATCH --time=04:00:00 # HH:MM:SS, overestimate
#SBATCH --mem-per-cpu=500 # MB, from running mprof with id=1
#
#SBATCH --array=1-20 # 10 network sets and 2 science cases with a unique waveform each

# James Gardner, March 2022
srun python3 -u run_plot_detection_rate_and_measurement_errors.py $SLURM_ARRAY_TASK_ID

#!/bin/bash
#
#SBATCH --job-name=plotDRandERRs
#SBATCH --output=stdout_job_plot_detection_rate_and_measurement_errors.txt
#SBATCH --error=stderr_job_plot_detection_rate_and_measurement_errors.txt
#
#SBATCH --ntasks=1
#SBATCH --time=00:05:00 # HH:MM:SS, mprof maximum: 70 s
#SBATCH --mem-per-cpu=10000 # MB, mprof maximum: 5500 MB but out-of-memory showed up in stderr again?
#
#SBATCH --array=1-4 # just BS_SIX for now. set to 1-40 to get all 10 network sets, 2 science cases with a unique waveform each, 2 plots

# James Gardner, March 2022
srun python3 /fred/oz209/jgardner/CEonlyPony/source/run_plot_detection_rate_and_measurement_errors.py $SLURM_ARRAY_TASK_ID

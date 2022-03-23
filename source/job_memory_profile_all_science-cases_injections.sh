#!/bin/bash
#
#SBATCH --job-name=mprof
#SBATCH --output=stdout_mprof.txt
#SBATCH --error=stderr_mprof.txt
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00 # HH:MM:SS
#SBATCH --mem-per-cpu=1000 # MB
#
#SBATCH --array=1-2

# James Gardner, March 2022

SCIENCE_CASES=('BNS' 'BBH')
#NUM_INJS_PER_ZBIN_PER_TASK_LIST=(100 10) # 10 numerical injections/zbin takes 3 minutes on a single core
NUM_INJS_PER_ZBIN_PER_TASK_LIST=(1000 1000) # want to know how long it'll take to scale up to B&S2022
let "SCIENCE_CASE_INDEX = $SLURM_ARRAY_TASK_ID - 1"
SCIENCE_CASE=${SCIENCE_CASES[$SCIENCE_CASE_INDEX]}
NUM_INJS_PER_ZBIN_PER_TASK=${NUM_INJS_PER_ZBIN_PER_TASK_LIST[$SCIENCE_CASE_INDEX]}

# ensure that @profile is enabled in detection_rates.py and that data files don't already exist (if they do, then just increment the task ID, currently 0, below)
#echo "mprof_plot_${SCIENCE_CASE}_${NUM_INJS_PER_ZBIN_PER_TASK}.pdf"
srun mprof run old_run_injections_for_hard-coded_network.py 0 ${NUM_INJS_PER_ZBIN_PER_TASK} ${SCIENCE_CASE}; mprof plot -o "mprof_plot_${SCIENCE_CASE}_${NUM_INJS_PER_ZBIN_PER_TASK}.pdf"


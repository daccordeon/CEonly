#!/bin/bash
#
#SBATCH --job-name=job_run_all_injections
#SBATCH --output=stdout_job_run_all_injections.txt
#SBATCH --error=stderr_job_run_all_injections.txt
#
#SBATCH --ntasks=1
#SBATCH --time=00:06:00 # HH:MM:SS
#SBATCH --mem-per-cpu=200 # MB, use mprof to determine required time and memory per task
#
#SBATCH --array=1-132 # last value is the number of independent jobs, 132 = 33 networks, 2 science cases, 2 tasks each

# tasks should go: net1-BNS, net1-BNS, net1-BBH, net1-BBH, net2-BNS, ...
# use task index to select a network and a science case (the latter of which uniquely determines a waveform)
NUM_TASKS_PER_NETWORK_SC_WF=2 # need to manually update number of tasks in array, to-do: automate this
SCIENCE_CASES=('BNS' 'BBH')
NUM_INJS_PER_ZBIN_PER_TASK_LIST=(100 10) # 10 numerical injections/zbin takes 3 minutes on a single core

NUM_SCS=${#SCIENCE_CASES[*]} # length of SCIENCE_CASES
# determine network in python script from task id
let "NETWORK_INDEX = ($SLURM_ARRAY_TASK_ID - 1)/($NUM_SCS*$NUM_TASKS_PER_NETWORK_SC_WF)" # bash '/' rounds down
let "SCIENCE_CASE_INDEX = (($SLURM_ARRAY_TASK_ID - 1) % ($NUM_SCS*$NUM_TASKS_PER_NETWORK_SC_WF))/$NUM_TASKS_PER_NETWORK_SC_WF"
SCIENCE_CASE=${SCIENCE_CASES[$SCIENCE_CASE_INDEX]}
# total number of injections is num_tasks*num_zbins*<below number>
NUM_INJS_PER_ZBIN_PER_TASK=${NUM_INJS_PER_ZBIN_PER_TASK_LIST[$SCIENCE_CASE_INDEX]}

# arguments: task_id, network_id, science_case, num_injs_per_zbin_per_task
srun python3 -u /home/jgardner/CEonlyPony/source/run_injections_for_network_id.py $SLURM_ARRAY_TASK_ID $NETWORK_INDEX $SCIENCE_CASE $NUM_INJS_PER_ZBIN_PER_TASK


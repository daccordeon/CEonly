#!/bin/bash
#
#SBATCH --job-name=jobRunAllInjLng # 15 non-whitespace characters long
#SBATCH --output=stdout_job_run_all_injections_at_scale.txt
#SBATCH --error=stderr_job_run_all_injections_at_scale.txt
#
#SBATCH --ntasks=1
#SBATCH --time=08:00:00 # HH:MM:SS
#SBATCH --mem-per-cpu=200 # MB, determined from mprof (175 MB, 50 min for BNS, 7 hr for BBH)
#
#SBATCH --array=1-2040 # number of independent jobs, 34 networks, 2 science cases, some number of tasks each set below, watch out for MaxArraySize=2048 and the maximum concurrent jobs of 1000 in /apps/slurm/etc/slurm.config

# tasks should go: net1-BNS, net1-BNS, net1-BBH, net1-BBH, net2-BNS, ...
# use task index to select a network and a science case (the latter of which uniquely determines a waveform)
NUM_TASKS_PER_NETWORK_SC_WF=30 # need to manually update number of tasks in array, to-do: automate this
SCIENCE_CASES=('BNS' 'BBH')
NUM_INJS_PER_ZBIN_PER_TASK_LIST=(1000 1000) 

NUM_NETWORKS=34
NUM_SCS=${#SCIENCE_CASES[*]} # length of SCIENCE_CASES
let "NUM_FILES = $NUM_NETWORKS*$NUM_SCS*$NUM_TASKS_PER_NETWORK_SC_WF"
# whether to automatically merge all the task files
MERGE_BOOL=0
# determine network in python script from task id
let "NETWORK_INDEX = ($SLURM_ARRAY_TASK_ID - 1)/($NUM_SCS*$NUM_TASKS_PER_NETWORK_SC_WF)" # bash '/' rounds down
let "SCIENCE_CASE_INDEX = (($SLURM_ARRAY_TASK_ID - 1) % ($NUM_SCS*$NUM_TASKS_PER_NETWORK_SC_WF))/$NUM_TASKS_PER_NETWORK_SC_WF"
SCIENCE_CASE=${SCIENCE_CASES[$SCIENCE_CASE_INDEX]}
# total number of injections is num_tasks*num_zbins*<below number>
NUM_INJS_PER_ZBIN_PER_TASK=${NUM_INJS_PER_ZBIN_PER_TASK_LIST[$SCIENCE_CASE_INDEX]}
# offset task_id if running on farnarkle and sstar concurrently, use 2040 to dodge first set
TASK_ID_OFFSET=0
let "TASK_ID = $SLURM_ARRAY_TASK_ID + $TASK_ID_OFFSET"

# arguments: task_id, network_id, science_case, num_injs_per_zbin_per_task
srun python3 -u /fred/oz209/jgardner/CEonlyPony/source/run_injections_for_network_id.py $TASK_ID $NETWORK_INDEX $SCIENCE_CASE $NUM_INJS_PER_ZBIN_PER_TASK $NUM_FILES $MERGE_BOOL

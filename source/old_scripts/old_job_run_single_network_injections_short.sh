#!/bin/bash
#
#SBATCH --job-name=old_job_run_injections
#SBATCH --output=stdout_old_job_run_single_network_injections_short.txt
#SBATCH --error=stderr_old_job_run_single_network_injections_short.txt
#
#SBATCH --ntasks=1
#SBATCH --time=00:08:00 # HH:MM:SS
#SBATCH --mem-per-cpu=200 # MB, use mprof to determine required time and memory per task
#
#SBATCH --array=1-60 # last value is the number of independent jobs

# total number of injections is num_tasks*num_zbins*<below number>
NUM_INJS_PER_ZBIN_PER_TASK=1000

# arguments: task_id, num_injs_per_task
srun python3 -u /fred/oz209/jgardner/CEonlyPony/source/old_scripts/old_run_injections_for_hard-coded_network.py $SLURM_ARRAY_TASK_ID $NUM_INJS_PER_ZBIN_PER_TASK

# guide for pleasingly (aka. embarrassingly) parallel scripting where lots of jobs are created that are each single-threaded
# https://supercomputing.swin.edu.au/docs/2-ozstar/oz-slurm-examples.html#embarrassingly-parallel-example
# provides a warning: "If the running time of your program is small (i.e. ten minutes or less), creating a job array will incur a lot of overhead and you should consider packing your jobs."
# --> each time my_program is run it should go for sufficiently long to make the overhead worth it
# local filesystem has 32 cores to bridge the gap from Blackwater?

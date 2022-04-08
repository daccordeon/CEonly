#!/bin/bash
#
#SBATCH --job-name=jobTestHybrid # 15 non-whitespace characters long
#SBATCH --output=stdout_job_test_hybridised.txt
#SBATCH --error=stderr_job_test_hybridised.txt
#
#SBATCH --array=1-40 # number of independent jobs
#
#SBATCH --nodes=1 # each job occurs on an individual node, however, the nodes are different
#SBATCH --ntasks=1 # 1 task per job
#SBATCH --cpus-per-task=4 # cores per task, maximum 30ish depending on how many out of 36 is left to by ozstar admin
#SBATCH --time=00:05:00 # HH:MM:SS
#SBATCH --mem-per-cpu=200 # MB, determined from mprof (175 MB, 50 min for BNS, 7 hr for BBH)

# test_script.py should use multiprocessing with the cpu count given by slurm
# -u not showing all print statements?
srun python3 -u /fred/oz209/jgardner/CEonlyPony/source/hybridised_job_test_script.py $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK $SLURM_JOB_NODELIST $SLURM_JOB_NUM_NODES $SLURM_NODEID $SLURM_LOCALID $SLURM_SUBMIT_DIR

# if the 2500 cores per project restriction exists, then hybridising from a base of 2040 cores is not going to be particularly profitable

# https://help.rc.ufl.edu/doc/Sample_SLURM_Scripts
# https://supercomputing.swin.edu.au/docs/2-ozstar/oz-slurm-examples.html

# for GPU programming see CuPy (https://cupy.dev/) for simple replacement of numpy/scipy commands: https://sulis-hpc.github.io/gettingstarted/batchq/gpu.html

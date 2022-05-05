#!/bin/bash
#
#SBATCH --job-name=jobUnifiedInjs
#SBATCH --output=slurm_output_files/stdout_job_unified_injections_all_networks_JOB-ID_%A_TASK-ID_%a.txt
#SBATCH --error=slurm_output_files/stderr_job_unified_injections_all_networks_JOB-ID_%A_TASK-ID_%a.txt
#
#SBATCH --ntasks=1
#SBATCH --time=04:00:00 # HH:MM:SS
#SBATCH --mem-per-cpu=250 # MB, determined from mprof (3.5 hr, 160 MB is likely enough)
#
#SBATCH --array=0-2047 # shifted to 1-2048 below

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Slurm script to process the unified injections in a large job array.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 
# License:
#     BSD 3-Clause License
# 
#     Copyright (c) 2022, James Gardner.
#     All rights reserved except for those for the gwbench code which remain reserved
#     by S. Borhanian; the gwbench code is included in this repository for convenience.
# 
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
# 
#     1. Redistributions of source code must retain the above copyright notice, this
#        list of conditions and the following disclaimer.
# 
#     2. Redistributions in binary form must reproduce the above copyright notice,
#        this list of conditions and the following disclaimer in the documentation
#        and/or other materials provided with the distribution.
# 
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
# 
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#     DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#     SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#     OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#     OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# <https://slurm.schedmd.com/slurm.conf.html> --> The maximum job array task index value will be one less than MaxArraySize to allow for an index value of zero. 
# python scripts run with task_id's starting from zero
let "ONE_INDEXED_TASK_ID = ${SLURM_ARRAY_TASK_ID} + 1"

# argument/s: task_id (handle everything else inside python)
srun python3 /fred/oz209/jgardner/CEonlyPony/source/run_calculate_unified_injections_as_task.py ${ONE_INDEXED_TASK_ID}

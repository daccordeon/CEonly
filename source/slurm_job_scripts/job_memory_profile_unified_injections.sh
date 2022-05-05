#!/bin/bash
#
#SBATCH --job-name=mprofUnifiedInjs
#SBATCH --output=slurm_output_files/stdout_job_memory_profile_unified_injections_JOB-ID_%A_TASK-ID_%a.txt
#SBATCH --error=slurm_output_files/stderr_job_memory_profile_unified_injections_JOB-ID_%A_TASK-ID_%a.txt
#
#SBATCH --ntasks=1
#SBATCH --time=08:00:00 # HH:MM:SS
#SBATCH --mem-per-cpu=1000 # MB
#
#SBATCH --array=1-4

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Slurm script to profile the memory usage of the processing unified injections script.
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

TEST_INDS=(1 1024 1025 2048)
# slurm uses a base index of 1 but bash uses 0
let "TEST_IND = ${TEST_INDS[$SLURM_ARRAY_TASK_ID - 1]}"

srun mprof run -o "mprof_plot_unified_injs_${TEST_IND}.dat" /fred/oz209/jgardner/CEonlyPony/source/run_calculate_unified_injections_as_task.py ${TEST_IND}; mprof plot -o "mprof_plot_unified_injs_${TEST_IND}.pdf" "mprof_plot_unified_injs_${TEST_IND}.dat"

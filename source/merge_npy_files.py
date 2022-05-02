#!/usr/bin/env python3
"""Short one-sentence description.

Long description.

Usage:
    Describe the typical usage.

License:
    BSD 3-Clause License

    Copyright (c) 2022, James Gardner.
    All rights reserved except for those for the gwbench code which remain reserved
    by S. Borhanian; the gwbench code is included in this repository for convenience.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import List, Set, Dict, Tuple, Optional, Union
from numpy.typing import NDArray
import numpy as np
import glob
import os, sys


def file_tag_from_task_file(file, cut_num_injs=False):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """returns file_tag from task output filenames, assumes that path is in file name"""
    file_tag = file.replace("_TASK_", "results_").split("results_")[1]
    if cut_num_injs:
        return file_tag.split("_INJS-PER-ZBIN_")[0]
    else:
        return file_tag


def merge_npy_files(
    output_filename, input_files=None, pattern=None, path="./", delete_input_files=False
):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """finds all .npy files matching pattern in path, saves a merged .npy at output_filename, or just merges those given as input_files"""
    # https://stackoverflow.com/questions/44164917/concatenating-numpy-arrays-from-a-directory
    if input_files is None:
        input_files = sorted(
            glob.glob(path + pattern)
        )  # sorted to make debugging printout easier to read
    arrays = []
    for input_file in input_files:
        arrays.append(np.load(input_file))
    # try to merge and if fails then don't delete input files
    try:
        # concatenate works with empty data arrays as long as they have shape=(0, 7) which is the case for without_rows_w_nan
        merged_array = np.concatenate(arrays)
        np.save(path + output_filename, merged_array)
    except:
        raise ValueError(
            "Something went wrong when concatenating arrays, check memory allocation."
        )
    else:
        if delete_input_files:
            for input_file in input_files:
                os.remove(input_file)


def merge_all_task_npy_files(
    path="/fred/oz209/jgardner/CEonlyPony/source/processed_injections_data/",
    pattern="results_NET_*_SCI-CASE_*_WF_*_INJS-PER-ZBIN_*_TASK_*.npy",
    delete_input_files=False,
):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """find all .npy outputs from each task from job_run_all_networks_injections.sh and merge them together to have one .npy file per network+sc+wf combination"""
    task_files = sorted(
        glob.glob(path + pattern)
    )  # sorted to make debugging printout easier to read
    # split into separate network+sc+wf combinations
    # dict(tag1=[net1-task1, net1-task2], tag2=[net2-task1, net2-task2], ...)
    dict_tag_task_files = dict()
    for file in task_files:
        # remove num_injs from file_tag to capture the last injection task which contains more injections since 1024 doesn't divide the injections remaining after initial filtering
        file_tag_net_sc_wf = file_tag_from_task_file(file, cut_num_injs=True)
        # if it is not already in the dict, then find all matches and add them
        if not file_tag_net_sc_wf in dict_tag_task_files.keys():
            dict_tag_task_files[file_tag_net_sc_wf] = [
                file
                for file in task_files
                if file_tag_net_sc_wf
                == file_tag_from_task_file(file, cut_num_injs=True)
            ]

    for file_tag_net_sc_wf, task_files_same_tag in dict_tag_task_files.items():
        # calculate total number of injections if all injections had well-conditioned FIMs, no longer assuming that all have the same initial number of injections. replace '.npy' to deal with non-task files
        #         total_num_injs_per_zbin = sum(
        #             [
        #                 int(
        #                     file.replace(".npy", "")
        #                     .replace("_TASK_", "_INJS-PER-ZBIN_")
        #                     .split("_INJS-PER-ZBIN_")[1]
        #                 )
        #                 for file in task_files_same_tag
        #             ]
        #         )
        # assuming tasks all for the same run of injections
        input_num_injs = (
            task_files_same_tag[0]
            .replace(".npy", "")
            .replace("_TASK_", "_INJS-PER-ZBIN_")
            .split("_INJS-PER-ZBIN_")[1]
        )
        output_filename = (
            f"results_{file_tag_net_sc_wf}_INJS-PER-ZBIN_{input_num_injs}.npy"
        )
        merge_npy_files(
            output_filename,
            input_files=task_files_same_tag,
            path=path,
            delete_input_files=delete_input_files,
        )


if __name__ == "__main__":
    # TODO: add progress bar, parallelise merging
    if len(sys.argv[1:]) == 1:
        delete_input_files = int(sys.argv[1])
    else:
        delete_input_files = 0
    merge_all_task_npy_files(delete_input_files=delete_input_files)

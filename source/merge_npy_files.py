#!/usr/bin/env python3
"""James Gardner, March 2022"""
import numpy as np
import glob
import os, sys


def file_tag_from_task_file(file, cut_num_injs=False):
    """returns file_tag from task output filenames, assumes that path is in file name"""
    file_tag = file.replace("_TASK_", "results_").split("results_")[1]
    if cut_num_injs:
        return file_tag.split("_INJS-PER-ZBIN_")[0]
    else:
        return file_tag


def merge_npy_files(
    output_filename, input_files=None, pattern=None, path="./", delete_input_files=False
):
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
    path="/fred/oz209/jgardner/CEonlyPony/source/data_redshift_snr_errs_sky-area/",
    pattern="results_NET_*_SCI-CASE_*_WF_*_INJS-PER-ZBIN_*_TASK_*.npy",
    delete_input_files=False,
):
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
    # to-do: add progress bar, parallelise merging
    if len(sys.argv[1:]) == 1:
        delete_input_files = int(sys.argv[1])
    else:
        delete_input_files = 0
    merge_all_task_npy_files(delete_input_files=delete_input_files)

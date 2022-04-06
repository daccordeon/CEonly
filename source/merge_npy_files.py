#!/usr/bin/env python3
"""James Gardner, March 2022"""
import numpy as np
import glob
import os

def file_tag_from_task_file(file):
    """returns file_tag from task output filenames, assumes that path is in file name"""
    return file.replace('_TASK_', 'results_').split('results_')[1]

def merge_npy_files(output_filename, input_files=None, pattern=None, path='./', delete_input_files=False):
    """finds all .npy files matching pattern in path, saves a merged .npy at output_filename, or just mergers those given as input_files"""
    # https://stackoverflow.com/questions/44164917/concatenating-numpy-arrays-from-a-directory
    if input_files is None:
        input_files = sorted(glob.glob(path + pattern)) # sorted to make debugging printout easier to read 
    arrays = []
    for input_file in input_files:
        arrays.append(np.load(input_file))
    # try to merge and if fails then don't delete input files
    try:
        merged_array = np.concatenate(arrays)
        np.save(path + output_filename, merged_array)
    except:
        raise ValueError('Something went wrong when concatenating arrays, check memory allocation.')
    else:
        if delete_input_files:
            for input_file in input_files:
                os.remove(input_file)        

def merge_all_task_npy_files(path='/fred/oz209/jgardner/CEonlyPony/source/data_redshift_snr_errs_sky-area/', pattern='results_NET_*_SCI-CASE_*_WF_*_INJS-PER-ZBIN_*_TASK_*.npy', delete_input_files=False):
    """find all .npy outputs from each task from job_run_all_networks_injections.sh and merge them together to have one .npy file per network+sc+wf combination"""
    task_files = sorted(glob.glob(path + pattern)) # sorted to make debugging printout easier to read 
    # split into separate network+sc+wf combinations
    # dict(tag1=[net1-task1, net1-task2], tag2=[net2-task1, net2-task2], ...)
    dict_tag_task_files = dict()
    for file in task_files:
        file_tag = file_tag_from_task_file(file)
        # if it is not already in the dict, then find all matches and add them    
        if not file_tag in dict_tag_task_files.keys():
            dict_tag_task_files[file_tag]=[file for file in task_files if file_tag == file_tag_from_task_file(file)] 
        
    for file_tag, task_files_same_tag in dict_tag_task_files.items():
        # calculate total number of injections if all injections had well-conditioned FIMs, subtlety that all assumed to have same initial number of injections
        total_num_injs_per_zbin = len(task_files_same_tag)*int(file_tag.replace('_TASK_', "_INJS-PER-ZBIN_").split("_INJS-PER-ZBIN_")[1])
        output_filename = f'results_{file_tag.split("_INJS-PER-ZBIN_")[0]}_INJS-PER-ZBIN_{total_num_injs_per_zbin}.npy'
        merge_npy_files(output_filename, input_files=task_files_same_tag, path=path, delete_input_files=delete_input_files)
    
if __name__ == '__main__':
    merge_all_task_npy_files(delete_input_files=True)

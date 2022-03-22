#!/usr/bin/env python3
"""James Gardner, March 2022"""
import numpy as np
import glob
import os

def merge_npy_files(path, output_filename, pattern='*.npy', delete_input_files=False):
    """finds all .npy files matching pattern in path, saves a merged .npy at output_filename"""	
    # https://stackoverflow.com/questions/44164917/concatenating-numpy-arrays-from-a-directory
    input_files = glob.glob(path+pattern)
    arrays = []
    for input_file in input_files:
        arrays.append(np.load(input_file))
        if delete_input_files:
            os.remove(input_file)
    merged_array = np.concatenate(arrays)
    np.save(path+output_filename, merged_array)

if __name__ == '__main__':
    file_tag = 'NET_A+_H..A+_L..V+_V..K+_K..A+_I_SCI-CASE_BBH_WF_lal_bbh_IMRPhenomHM_INJS-PER-ZBIN_10'
    pattern = f'results_{file_tag}_TASK_*.npy'
    output_filename = f'results_{file_tag}_MERGED.npy'
    merge_npy_files('data_redshift_snr_errs_sky-area/', output_filename, pattern=pattern, delete_input_files=True)


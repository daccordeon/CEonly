"""James Gardner, March 2022"""
import os
import numpy as np

from gwbench import network

def file_name_to_multiline_readable(file, two_rows_only=False, net_only=False):
    intermediate = file.replace('results_', '').replace('.npy', '').replace('NET_', 'network: ').replace('_SCI-CASE_', '\nscience case: ').replace('..', ', ')
    if net_only:
        return intermediate.split('\n')[0]
    else:
        if two_rows_only:
            return intermediate.replace('_WF_', ', waveform: ').replace('_NUM-INJS_', ", injections per bin: ")
        else:
            return intermediate.replace('_WF_', '\nwaveform: ').replace('_NUM-INJS_', "\ninjections per bin: ")

def find_files_given_networks(network_spec_list, science_case, specific_wf=None, print_progress=True):
    """returns a list of found files that match networks, science case, and specific wf, choosing those files with the greatest num_injs if multiple exist for a given network"""
    # finding file names
    net_labels = [network.Network(network_spec).label for network_spec in network_spec_list]
    
    file_list = os.listdir("data_redshift_snr_errs_sky-area")
    found_files = np.array([])
    for net_label in net_labels:
        # file_tag = f'NET_{net.label}_SCI-CASE_{science_case}_WF_..._NUM-INJS_{num_injs}'
        file_tag_partial = f'NET_{net_label}_SCI-CASE_{science_case}'
        # file is file_name
        matches = np.array([file for file in file_list if file_tag_partial in file])
        if len(matches) == 0:
            continue
        # [[f'NET_{net.label}_SCI-CASE_{science_case}', f'{wf_model_name}', f'{num_injs}', '.npy'], [...], ...]
        decomp_files = np.array([file.replace('.npy', '').replace('_WF_', '_NUM-INJS_').split('_NUM-INJS_') for file in matches])
        # appending is slow but this problem is small
        unique_wf_index_list = []
        for i, wf in enumerate(decomp_files[:,1]):
            # if specified a wf (with any auxillary), then skip those that don't match
            if specific_wf is not None:
                if wf != specific_wf:
                    continue
            # if multiple files with same tag, then select the one with the greatest number of injections
            num_injs = int(decomp_files[i,2])
            num_injs_list = [int(j) for j in decomp_files[:,2][decomp_files[:,1] == wf]]
            # covers the case where len(num_injs_list) = 1, i.e. unique wf
            if num_injs == max(num_injs_list):
                unique_wf_index_list.append(i)
        found_files = np.append(found_files, matches[list(set(unique_wf_index_list))]) #could flatten matches here
    found_files = found_files.flatten()
    if len(found_files) == 0:
        raise ValueError('No files found.')
    elif print_progress:
        print(f'Found {len(found_files)} file/s:', *found_files, sep='\n')
    return list(found_files)

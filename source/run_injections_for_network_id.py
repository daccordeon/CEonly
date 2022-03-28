#!/bin/env/python3
"""James Gardner, March 2022
script for a single task's worth of injections for a given task id (to determine the network), science case, and number of injections per redshift bin"""
from detection_rates import detection_rate_for_network_and_waveform
from merge_npy_files import merge_all_task_npy_files
from filename_search_and_manipulation import net_spec_styler
from networks import NET_LIST

import sys
import glob

# suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')

# input arguments from bash
# to-do: look into argparse for a kwarg alternative to sys.argv
task_id, network_id, science_case, num_injs_per_zbin_per_task, total_number_of_files, merge_bool = sys.argv[1:] # first argv is the script's name
task_id, network_id, num_injs_per_zbin_per_task, total_number_of_files, merge_bool = [int(x) for x in (task_id, network_id, num_injs_per_zbin_per_task, total_number_of_files, merge_bool)]

# --- network, waveform, and injection parameters ---
# tasks go through all 34 networks with two science cases for each
network_spec = NET_LIST[network_id]
# determine waveform based on science case
if science_case == 'BNS':
    #wf_model_name, wf_other_var_dic = 'lal_bns', dict(approximant='IMRPhenomD_NRTidalv2')
    wf_model_name, wf_other_var_dic = 'tf2_tidal', None # to-do: change to more accurate numerical once gwbench patch released
elif science_case == 'BBH':
    wf_model_name, wf_other_var_dic = 'lal_bbh', dict(approximant='IMRPhenomHM')
else:
    raise ValueError('Science case not recognised.')

# output file name, using SLURM_TASK_... will use file_tag once network initialised
file_name = f'SLURM_TASK_{task_id}'

# generate and save injections in a separate file, don't parallelise individual tasks
data_path = '/fred/oz209/jgardner/CEonlyPony/source/data_redshift_snr_errs_sky-area/'
# --- for data production ---
detection_rate_for_network_and_waveform(network_spec, science_case, wf_model_name, wf_other_var_dic, num_injs_per_zbin_per_task, generate_fig=False, show_fig=False, print_progress=False, print_reach=False, data_path=data_path, file_name=file_name, parallel=False)
# --- for testing job script ---
# print(f'task_id={task_id}, network_spec={network_spec}, science_case={science_case}, wf_model_name={wf_model_name}, wf_other_var_dic={wf_other_var_dic}, num_injs_per_zbin_per_task={num_injs_per_zbin_per_task}, file_name={file_name}')
# if wf_other_var_dic is not None:
#     file_tag = f'NET_{net_spec_styler(network_spec)}_SCI-CASE_{science_case}_WF_{wf_model_name}_{wf_other_var_dic["approximant"]}_INJS-PER-ZBIN_{num_injs_per_zbin_per_task}'
# else:
#     file_tag = f'NET_{net_spec_styler(network_spec)}_SCI-CASE_{science_case}_WF_{wf_model_name}_INJS-PER-ZBIN_{num_injs_per_zbin_per_task}'
# with open(f'results_{file_tag}_TASK_{task_id}.txt', 'w') as file:
#     file.write(f'task_id={task_id}, network_spec={network_spec}, science_case={science_case}, wf_model_name={wf_model_name}, wf_other_var_dic={wf_other_var_dic}, num_injs_per_zbin_per_task={num_injs_per_zbin_per_task}, file_name={file_name}')

# clean-up, if all task files exist, then call the merge script which only combines task files (therefore if previous data files exist they will be unharmed)
# to-do: check if this is ever likely to happen in two threads at the same time, check if the ``race condition'' is impossible
if merge_bool and (total_number_of_files == len(glob.glob(data_path + 'results_*_TASK_*'))):
    merge_all_task_npy_files(delete_input_files=True)

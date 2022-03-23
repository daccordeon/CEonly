#!/bin/env/python3
"""James Gardner, March 2022
script for a single task's worth of injections for a given task id (to determine the network), science case, and number of injections per redshift bin"""
from detection_rates import detection_rate_for_network_and_waveform
from merge_npy_files import merge_all_task_npy_files

import sys
from networks import NET_LIST
import glob

# suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')

# input arguments from bash
# to-do: look into argparse for a kwarg alternative to sys.argv
task_id, network_id, science_case, num_injs_per_zbin_per_task, total_number_of_files = sys.argv[1:] # first argv is the script's name
task_id, network_id, num_injs_per_zbin_per_task, total_number_of_files = [int(x) for x in (task_id, network_id, num_injs_per_zbin_per_task, total_number_of_files)]

# --- network, waveform, and injection parameters ---
# tasks go through all 34 networks with two science cases for each
network_spec = NET_LIST[network_id]
# determine waveform based on science case
if science_case == 'BNS':
    #wf_model_name, wf_other_var_dic = 'lal_bns', dict(approximant='IMRPhenomD_NRTidalv2')
    wf_model_name, wf_other_var_dic = 'tf2_tidal', None
elif science_case == 'BBH':
    wf_model_name, wf_other_var_dic = 'lal_bbh', dict(approximant='IMRPhenomHM')
else:
    raise ValueError('Science case not recognised.')

# output file name, using SLURM_TASK_... will use file_tag once network initialised
file_name = f'SLURM_TASK_{task_id}'

# generate and save injections in a separate file, don't parallelise individual tasks
data_path = '/home/jgardner/CEonlyPony/source/data_redshift_snr_errs_sky-area/'
#print(f'task_id={task_id}, network_spec={network_spec}, science_case={science_case}, wf_model_name={wf_model_name}, wf_other_var_dic={wf_other_var_dic}, num_injs_per_zbin_per_task={num_injs_per_zbin_per_task}, file_name={file_name}')
detection_rate_for_network_and_waveform(network_spec, science_case, wf_model_name, wf_other_var_dic, num_injs_per_zbin_per_task, generate_fig=False, show_fig=False, print_progress=False, print_reach=False, data_path=data_path, file_name=file_name, parallel=False)

# clean-up, if all files exist, then call the merge script
# to-do: check if this is ever likely to happen in two threads at the same time, check if the ``race condition'' is impossible
if total_number_of_files == len(glob.glob(data_path + 'results_*_TASK_*')):
    merge_all_task_npy_files(delete_input_files=True)


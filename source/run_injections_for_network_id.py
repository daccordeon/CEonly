#!/bin/env/python3
"""James Gardner, March 2022
script for a single task's worth of injections for a given task id (to determine the network), science case, and number of injections per redshift bin"""
from detection_rates import detection_rate_for_network_and_waveform

import sys
from useful_functions import flatten_list
from networks import NET_DICT_LIST

# suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')

# input arguments from bash
task_id, network_id, science_case, num_injs_per_zbin_per_task = sys.argv[1:] # first argv is the script's name
task_id, network_id, num_injs_per_zbin_per_task = [int(x) for x in (task_id, network_id, num_injs_per_zbin_per_task)]

# --- network, waveform, and injection parameters ---
# tasks go through all 33 networks with two science cases for each
network_spec = flatten_list([net_dict['nets'] for net_dict in NET_DICT_LIST])[network_id]
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
print(f'task_id={task_id}, network_spec={network_spec}, science_case={science_case}, wf_model_name={wf_model_name}, wf_other_var_dic={wf_other_var_dic}, num_injs_per_zbin_per_task={num_injs_per_zbin_per_task}, file_name={file_name}')
#detection_rate_for_network_and_waveform(network_spec, science_case, wf_model_name, wf_other_var_dic, num_injs_per_zbin_per_task, generate_fig=False, show_fig=False, print_progress=False, print_reach=False, data_path=data_path, file_name=file_name, parallel=False)


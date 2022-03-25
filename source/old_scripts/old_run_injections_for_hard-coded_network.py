#!/bin/env/python3
"""James Gardner, March 2022
script for a single task's worth of injections for a hard-coded network, science case, and waveform"""
import sys
from gwbench import network
from detection_rates import detection_rate_for_network_and_waveform
from filename_search_and_manipulation import net_label_styler

# suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')

# input two arguments from bash
if len(sys.argv[1:]) == 2:
    task_id, num_injs_per_zbin_per_task = (int(arg) for arg in sys.argv[1:]) # first argv is the script's name
    #science_case = 'BBH'
    science_case = 'BNS'
else:
    task_id, num_injs_per_zbin_per_task, science_case = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

# --- network, waveform, and injection parameters ---
#network_spec = ['A+_H', 'A+_L', 'V+_V', 'K+_K', 'A+_I']
# the slowest network in the set?
# network_spec = ['A+_H', 'A+_L', 'K+_K', 'A+_I', 'ET_ET1', 'ET_ET2', 'ET_ET3']
# one of the missing networks from the first 30k run, chosen for having many detectors
network_spec = ['V+_V', 'K+_K', 'Voyager-CBO_H', 'Voyager-CBO_L', 'Voyager-CBO_I', 'CE2-40-CBO_S']

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
data_path = '/fred/oz209/jgardner/CEonlyPony/source/data_redshift_snr_errs_sky-area/'
#print(f'task_id={task_id}, network_spec={network_spec}, science_case={science_case}, wf_model_name={wf_model_name}, wf_other_var_dic={wf_other_var_dic}, num_injs_per_zbin_per_task={num_injs_per_zbin_per_task}, file_name={file_name}')
detection_rate_for_network_and_waveform(network_spec, science_case, wf_model_name, wf_other_var_dic, num_injs_per_zbin_per_task, generate_fig=False, show_fig=False, print_progress=False, print_reach=False, data_path=data_path, file_name=file_name, parallel=False)

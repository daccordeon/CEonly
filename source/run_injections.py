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
task_id, num_injs_per_zbin_per_task = (int(arg) for arg in sys.argv[1:]) # first argv is the script's name

# --- network, waveform, and injection parameters ---
network_spec = ['A+_H', 'A+_L', 'V+_V', 'K+_K', 'A+_I']
science_case = 'BBH'
wf_model_name, wf_other_var_dic = 'lal_bbh', dict(approximant='IMRPhenomHM')

# output file name, using SLURM_TASK_... will use file_tag once network initialised
file_name = f'SLURM_TASK_{task_id}'

# generate and save injections in a separate file, don't parallelise individual tasks
data_path = '/home/jgardner/CEonlyPony/source/data_redshift_snr_errs_sky-area/'
detection_rate_for_network_and_waveform(network_spec, science_case, wf_model_name, wf_other_var_dic, num_injs_per_zbin_per_task, generate_fig=False, show_fig=False, print_progress=False, print_reach=False, data_path=data_path, file_name=file_name, parallel=False)


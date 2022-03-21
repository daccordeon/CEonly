#!/bin/env/python3

import sys
from gwbench import network
from detection_rates import detection_rate_for_network_and_waveform
from filename_search_and_manipulation import net_label_styler

# suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')

# input two arguments from bash
task_id, num_injs_per_task = (int(arg) for arg in sys.argv[1:]) # first argv is the script's name

# --- network, waveform, and injection parameters ---
network_spec = ['A+_H', 'A+_L', 'V+_V', 'K+_K', 'A+_I']
science_case = 'BBH'
wf_model_name, wf_other_var_dic = 'lal_bbh', dict(approximant='IMRPhenomHM')
# number of injections per redshift bin (6 bins)
num_injs_per_zbin = num_injs_per_task

# output file name
# to-do: avoid initialising network just to get label
if wf_other_var_dic is not None:
    file_tag = f'NET_{net_label_styler(network.Network(network_spec).label)}_SCI-CASE_{science_case}_WF_{wf_model_name}_{wf_other_var_dic["approximant"]}_NUM-INJS_{num_injs_per_zbin}'
else:
    file_tag = f'NET_{net_label_styler(network.Network(network_spec).label)}_SCI-CASE_{science_case}_WF_{wf_model_name}_NUM-INJS_{num_injs_per_zbin}'
file_name = f'{file_tag}_task-{task_id}.npy'
    
# generate and save injections in a separate file
data_path = '/home/jgardner/CEonlyPony/source/data_injections/'
detection_rate_for_network_and_waveform(network_spec, science_case, wf_model_name, wf_other_var_dic, num_injs_per_zbin, generate_fig=False, show_fig=False, print_progress=False, print_reach=False, data_path=data_path, file_name=file_name)

#!/usr/bin/env python3
"""James Gardner, March 2022
generates detection rate vs redshift and CDF sky area and measurement errors etc. plots from saved data given a task array"""
from networks import *
from detection_rates import *
from measurement_errors import *
from useful_functions import flatten_list

import sys

# suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')

# input argument from bash
task_id = int(sys.argv[1])

net_dict = NET_DICT_LIST[(task_id - 1) // 2]
# to-do: add back functionality to include benchmarks such as HLVKI+ in CE-only plots
network_set = net_dict['nets']
network_label = net_dict['label']

science_case = ('BNS', 'BBH')[(task_id - 1) % 2]
if science_case == 'BNS':
    #wf_model_name, wf_other_var_dic = 'lal_bns', dict(approximant='IMRPhenomD_NRTidalv2')
    wf_model_name, wf_other_var_dic = 'tf2_tidal', None # to-do: change to more accurate numerical once gwbench patch released
elif science_case == 'BBH':
    wf_model_name, wf_other_var_dic = 'lal_bbh', dict(approximant='IMRPhenomHM')
else:
    raise ValueError('Science case not recognised.')

if wf_other_var_dic is not None:
    plot_label = f'NET_{network_label}_SCI-CASE_{science_case}_WF_{wf_model_name}_{wf_other_var_dic["approximant"]}'
    plot_title = f'Networks: {network_label}, science-case: {science_case}, waveform: {wf_model_name} {wf_other_var_dic["approximant"]}'    
else:
    plot_label = f'NET_{network_label}_SCI-CASE_{science_case}_WF_{wf_model_name}'
    plot_title = f'Networks: {network_label}, science-case: {science_case}, waveform: {wf_model_name}'        

data_path = '/fred/oz209/jgardner/CEonlyPony/source/data_redshift_snr_errs_sky-area/'

# --- detection rate plot ---
# compare to Fig 1 and 2 in Borhanian and Sathya 2022
# being lazy and not specifying unique waveform using specify_waveform, assuming that other waveforms not present
compare_detection_rate_of_networks_from_saved_results(network_set, science_case, plot_label=plot_label, show_fig=False, data_path=data_path, print_progress=False)

# --- measurement errors plot ---
# compare to Fig 3 and 4 in B&S 2022
# normalises CDF to dlog(value) and thresholds by low SNR level (defaults to 10)
ymin_CDF = 1e-4
collate_measurement_errs_CDFs_of_networks(network_set, science_case, plot_label=plot_label, plot_title=plot_title, full_legend=False, print_progress=False, show_fig=False, normalise_count=True, xlim_list=None, threshold_by_SNR=True, CDFmin=ymin_CDF, data_path=data_path)
# additionally, for more direct comparison, use B&S2022's xlim_list which is a hard coded option in measurement_errors.py 
if net_dict == BS2022_SIX:
    collate_measurement_errs_CDFs_of_networks(network_set, science_case, plot_label=plot_label+'_XLIMS_preset', plot_title=plot_title+', XLIMS: preset to B&S2022', full_legend=False, print_progress=False, show_fig=True, normalise_count=True, xlim_list='B&S2022', threshold_by_SNR=False, CDFmin=ymin_CDF, data_path=data_path)

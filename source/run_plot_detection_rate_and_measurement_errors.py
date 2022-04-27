#!/usr/bin/env python3
"""James Gardner, March 2022
generates detection rate vs redshift and CDF sky area and measurement errors etc. plots from saved data given a task array"""
from networks import *
from plot_collated_detection_rate import *
from plot_collated_PDFs_and_CDFs import *
from useful_functions import flatten_list

import sys

# suppress warnings
from warnings import filterwarnings

filterwarnings("ignore")

# --- user input
task_id = int(sys.argv[1])
seed = 12345
norm_tag = "GWTC2"
# ---
# netDict1-sc1-plot1, netDict1-sc1-plot2, netDict1-sc2-plot1, netDict1-sc2-plot2, netDict2...
net_dict = NET_DICT_LIST[(task_id - 1) // 4]
science_case = ("BNS", "BBH")[((task_id - 1) // 2) % 2]
plot_index = (task_id - 1) % 2

# to-do: add back functionality to include benchmarks such as HLVKI+ in CE-only plots
network_set = net_dict["nets"]
network_label = net_dict["label"]

if science_case == "BNS":
    # wf_model_name, wf_other_var_dic = 'lal_bns', dict(approximant='IMRPhenomD_NRTidalv2')
    wf_model_name, wf_other_var_dic = (
        "tf2_tidal",
        None,
    )  # to-do: change to more accurate numerical once gwbench patch released
elif science_case == "BBH":
    wf_model_name, wf_other_var_dic = "lal_bbh", dict(approximant="IMRPhenomHM")
else:
    raise ValueError("Science case not recognised.")

if wf_other_var_dic is not None:
    plot_label = f'NET_{network_label}_SCI-CASE_{science_case}_WF_{wf_model_name}_{wf_other_var_dic["approximant"]}'
    plot_title = f'Networks: {network_label}, science-case: {science_case}, waveform: {wf_model_name} {wf_other_var_dic["approximant"]}'
else:
    plot_label = f"NET_{network_label}_SCI-CASE_{science_case}_WF_{wf_model_name}"
    plot_title = f"Networks: {network_label}, science-case: {science_case}, waveform: {wf_model_name}"

data_path = "/fred/oz209/jgardner/CEonlyPony/source/data_redshift_snr_errs_sky-area/"

# print(network_set, science_case, plot_label, plot_index)

if plot_index == 0:
    # --- detection rate plot ---
    # compare to Fig 1 and 2 in Borhanian and Sathya 2022
    # being lazy and not specifying unique waveform using specify_waveform, assuming that other waveforms not present
    compare_detection_rate_of_networks_from_saved_results(
        network_set,
        science_case,
        plot_label=plot_label,
        show_fig=False,
        data_path=data_path,
        print_progress=False,
        parallel=False,
        norm_tag=norm_tag,
    )
elif plot_index == 1:
    # --- measurement errors plot ---
    # compare to Fig 3 and 4 in B&S 2022
    # normalises CDF to dlog(value) and thresholds by low SNR level (defaults to 10)
    args = network_set, science_case
    kwargs = dict(
        full_legend=False,
        print_progress=False,
        show_fig=False,
        normalise_count=True,
        threshold_by_SNR=True,
        CDFmin=1e-4,
        data_path=data_path,
        num_bins=40,
        contour=False,
        parallel=False,
        seed=seed,
        norm_tag=norm_tag,
    )
    if net_dict == BS2022_SIX:
        # additionally, for more direct comparison, use B&S2022's xlim_list which is a hard coded option in plot_collated_PDFs_and_CDFs.py
        collate_measurement_errs_CDFs_of_networks(
            *args,
            plot_label=plot_label + "_XLIMS_preset",
            plot_title=plot_title + ", XLIMS: preset to B&S2022",
            xlim_list="B&S2022",
            linestyles_from_BS2022=True,
            **kwargs,
        )
    else:
        collate_measurement_errs_CDFs_of_networks(
            *args,
            plot_label=plot_label,
            plot_title=plot_title,
            xlim_list=None,
            linestyles_from_BS2022=False,
            **kwargs,
        )

"""James Gardner, April 2022.
calculating the same set of injections for a set of networks.
based on the old calculate_injections and gwbench's multi_network.py example script"""


def multi_network_results_for_injection(network_specs, inj_params, wf_dict, deriv_dict):
    pass


def multi_network_results_for_injections_file(
    network_specs,
    injections_file,
    process_injs_per_task,
    base_params,
    wf_dict,
    deriv_dict,
    misc_dict,
):
    pass


############################################################################
### Precalculate the unique components common among all networks
############################################################################

# calculate the unique detector repsponse derivatives
if sym_derivs:
    unique_loc_net = network.unique_locs_det_responses(
        network_specs,
        f,
        inj_params,
        deriv_symbs_string,
        wf_model_name,
        wf_other_var_dic,
        conv_cos,
        conv_log,
        use_rot,
        num_cores,
    )
else:
    unique_loc_net = network.unique_locs_det_responses(
        network_specs,
        f,
        inj_params,
        deriv_symbs_string,
        wf_model_name,
        wf_other_var_dic,
        conv_cos,
        conv_log,
        use_rot,
        num_cores,
        step,
        method,
        order,
        d_order_n,
    )

# get the unique PSDs for the various detector technologies
unique_tec_net = network.unique_tecs(network_specs, f)

############################################################################
### Perform the analysisa of each network from the unique components
############################################################################

output = {}

for num, network_spec in enumerate(network_specs):
    net = network.Network(network_spec)
    # get the correct network from the unique components calculated above
    net.get_det_responses_psds_from_locs_tecs(unique_loc_net, unique_tec_net)
    net.calc_snrs(only_net=only_net)
    net.calc_errors(only_net=only_net)
    net.calc_sky_area_90(only_net=only_net)

    snr, errs, cov, fisher, inv_err = net.get_snrs_errs_cov_fisher_inv_err_for_key(
        key="network"
    )
    cond_num = net.cond_num
    well_cond = net.wc_fisher

    output[f"network {num}"] = {
        "network_spec": network_spec,
        "snr": snr,
        "errs": errs,
        "cov": cov,
        "cond_num": cond_num,
        "well_cond": well_cond,
        "inv_err": inv_err,
    }

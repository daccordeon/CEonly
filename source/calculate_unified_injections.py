"""James Gardner, April 2022.
calculating the same set of injections for a set of networks.
based on the old calculate_injections and gwbench's multi_network.py example script"""
from useful_functions import (
    without_rows_w_nan,
    parallel_map,
    HiddenPrints,
    PassEnterExit,
)
from generate_injections import filter_bool_for_injection, fisco_obs_from_Mc_eta
from network_subclass import NetworkExtended

import numpy as np
import os
from gwbench import network
from gwbench.basic_relations import f_isco_Msolar


def multi_network_results_for_injection(
    network_specs,
    inj,
    base_params,
    wf_dict,
    deriv_dict,
    misc_settings_dict,
    debug=False,
):
    """returns the benchmark as a dict of tuples for a single injection using the inj and base_params and the settings dicts through the networks in network_specs"""
    output_if_injection_fails = dict(
        (
            (repr(network_spec), tuple(np.nan for _ in range(7)))
            for network_spec in network_specs
        )
    )
    varied_keys = [
        "Mc",
        "eta",
        "chi1x",
        "chi1y",
        "chi1z",
        "chi2x",
        "chi2y",
        "chi2z",
        "DL",
        "iota",
        "ra",
        "dec",
        "psi",
        "z",
    ]
    varied_params = dict(zip(varied_keys, inj))
    z = varied_params.pop("z")
    inj_params = dict(**base_params, **varied_params)

    # subtlety, if V+ (or aLIGO) is present in any network, then f is truncated for V+ for all networks (since f is shared below). to-do: figure out how common this is
    aLIGO_or_Vplus_used = ("aLIGO" in deriv_dict["unique_tecs"]) or (
        "V+" in deriv_dict["unique_tecs"]
    )
    if not filter_bool_for_injection(
        inj,
        misc_settings_dict["redshifted"],
        wf_dict["coeff_fisco"],
        wf_dict["science_case"],
        aLIGO_or_Vplus_used=aLIGO_or_Vplus_used,
        debug=debug,
    ):
        return output_if_injection_fails
    fmin, fmax = 5.0, wf_dict["coeff_fisco"] * fisco_obs_from_Mc_eta(
        inj_params["Mc"],
        inj_params["eta"],
        redshifted=misc_settings_dict["redshifted"],
        z=z,
    )
    if aLIGO_or_Vplus_used:
        fmax_bounds = (11, 1024)
    else:
        fmax_bounds = (6, 1024)
    fmax = float(max(min(fmax, fmax_bounds[1]), fmax_bounds[0]))
    # df linearly transitions from 1/16 Hz (fine from B&S2022) to 10 Hz (coarse to save computation time)
    df = ((fmax - fmax_bounds[0]) / (fmax_bounds[1] - fmax_bounds[0])) * 10 + (
        (fmax_bounds[1] - fmax) / (fmax_bounds[1] - fmax_bounds[0])
    ) * 1 / 16
    f = np.arange(fmin, fmax + df, df)

    # passing parameters to gwbench, hide stdout (i.e. prints) if not debugging, stderr should still show up
    if not debug:
        entry_class = HiddenPrints
    else:
        entry_class = PassEnterExit
    with entry_class():
        # precalculate the unique components (detector derivatives and PSDs) common among all networks
        # calculate the unique detector response derivatives
        loc_net_args = (
            network_specs,
            f,
            inj_params,
            deriv_dict["deriv_symbs_string"],
            wf_dict["wf_model_name"],
            wf_dict["wf_other_var_dic"],
            deriv_dict["conv_cos"],
            deriv_dict["conv_log"],
            misc_settings_dict["use_rot"],
            misc_settings_dict["num_cores"],
        )
        if not deriv_dict["numerical_over_symbolic_derivs"]:
            unique_loc_net = network.unique_locs_det_responses(*loc_net_args)
        else:
            # update eta if too close to its maximum value for current step size, https://en.wikipedia.org/wiki/Chirp_mass#Definition_from_component_masses
            eta_max = 0.25
            deriv_dict["numerical_deriv_settings"]["step"] = min(
                deriv_dict["numerical_deriv_settings"]["step"],
                (eta_max - inj_params["eta"]) / 10,
            )
            unique_loc_net = network.unique_locs_det_responses(
                *loc_net_args,
                deriv_dict["numerical_deriv_settings"]["step"],
                deriv_dict["numerical_deriv_settings"]["method"],
                deriv_dict["numerical_deriv_settings"]["order"],
                deriv_dict["numerical_deriv_settings"]["n"],
            )
        # get the unique PSDs for the various detector technologies
        unique_tec_net = network.unique_tecs(network_specs, f)

        # perform the analysis of each network from the unique components
        multi_network_results_dict = dict()
        for i, network_spec in enumerate(network_specs):
            # if net only exists here, then the subclass is pointless. re-factor to justify using subclass
            net = network.Network(network_spec)
            # get the correct network from the unique components calculated above. this avoids the need to .set_net_vars, .set_wf_vars, .setup_ant_pat_lpf_psds, .calc_det_responses, .calc_det_responses_derivs_num/sym,
            net.get_det_responses_psds_from_locs_tecs(unique_loc_net, unique_tec_net)
            # calculate the network SNRs
            net.calc_snrs(only_net=misc_settings_dict["only_net"])
            # calculate the Fisher and covariance matrices, then error estimates
            net.calc_errors(only_net=misc_settings_dict["only_net"])
            # calculate the 90%-credible sky area (in [deg]^2)
            net.calc_sky_area_90(only_net=misc_settings_dict["only_net"])

            # to-do: if using gwbench 0.7, still introduce a limit on net.cond_num based on machine precision errors that mpmath is blind to
            if not net.wc_fisher:
                if debug:
                    print(
                        f"Rejected injection for {network_spec} (but not whole multi network) because of ill-conditioned FIM"
                    )
                multi_network_results_dict[repr(network_spec)] = tuple(
                    np.nan for _ in range(7)
                )
            else:
                # convert sigma_cos(iota) into sigma_iota
                abs_err_iota = abs(net.errs["cos_iota"] / np.sin(inj_params["iota"]))
                multi_network_results_dict[repr(network_spec)] = (
                    z,
                    net.snr,
                    net.errs["log_Mc"],
                    net.errs["log_DL"],
                    net.errs["eta"],
                    abs_err_iota,
                    net.errs["sky_area_90"],
                )

    return multi_network_results_dict


def multi_network_results_for_injections_file(
    results_file_name,
    network_specs,
    injections_file,
    process_injs_per_task,
    base_params,
    wf_dict,
    deriv_dict,
    misc_settings_dict,
    data_path="/fred/oz209/jgardner/CEonlyPony/source/data_redshift_snr_errs_sky-area/",
    debug=False,
):
    """benchmarks the first process_injs_per_task number of injections from injections_file + base_params for each of the networks in network_specs for the science_case and other settings in the three dict.'s provided, saves the results as a .npy file in results_file_name at data_path. num_injs_per_redshift_bin is the total number of injections from the injections file across all tasks (used for labelling)."""
    inj_data = np.load(injections_file)
    if process_injs_per_task is None:
        process_injs_per_task = len(inj_data)
    # only process the first process_injs_per_task of inj_data
    process_inj_data = inj_data[:process_injs_per_task]

    # check if results files already exist
    # can't pass net_copy because of memory constraints, want to stay low (200 MB), to do: test if this actually affects scheduling
    results_file_name_list = []
    for network_spec in network_specs:
        net = NetworkExtended(
            network_spec,
            wf_dict["science_case"],
            wf_dict["wf_model_name"],
            wf_dict["wf_other_var_dic"],
            process_injs_per_task,
            file_name=results_file_name,
            data_path=data_path,
        )
        # includes path
        results_file_name_list.append(net.file_name_with_path)
        if net.results_file_exists:
            raise Exception("Some results file/s already exist, aborting process.")

    # list of multi_network_results_dict's from each injection
    multi_network_results_dict_list = parallel_map(
        lambda inj: multi_network_results_for_injection(
            network_specs,
            inj,
            base_params,
            wf_dict,
            deriv_dict,
            misc_settings_dict,
            debug=debug,
        ),
        process_inj_data,
        parallel=misc_settings_dict["num_cores"] is not None,
        num_cpus=misc_settings_dict["num_cores"],
    )

    # convert results into numpy arrays for each network,
    for i, network_spec in enumerate(network_specs):
        results = np.array(
            [
                multi_network_results_dict[repr(network_spec)]
                for multi_network_results_dict in multi_network_results_dict_list
            ]
        )
        results = without_rows_w_nan(results)
        if len(results) == 0:
            print(network_spec, multi_network_results_dict_list)
            raise ValueError("All calculated values are NaN.")
        np.save(results_file_name_list[i], results)

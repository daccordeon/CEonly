"""Calculates the same set of injections for a set of networks.

Based on the old calculate_injections.py and gwbench's multi_network.py example script, this processes the injections (e.g. in data_raw_injections/) in union for each network in a set and saves the results (e.g. in data_processed_injections/). This is faster than the previous implementation if detectors are shared between the networks because the detector responses are only calculated once.

Usage:
    See the example in run_unified_injections.py.

License:
    BSD 3-Clause License

    Copyright (c) 2022, James Gardner.
    All rights reserved except for those for the gwbench code which remain reserved
    by S. Borhanian; the gwbench code is included in this repository for convenience.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from typing import List, Set, Dict, Tuple, Optional, Union
from numpy.typing import NDArray
import os
import numpy as np

from gwbench import network
from gwbench.basic_relations import f_isco_Msolar

from useful_functions import (
    without_rows_w_nan,
    parallel_map,
    HiddenPrints,
    PassEnterExit,
)
from generate_injections import filter_bool_for_injection, fisco_obs_from_Mc_eta
from network_subclass import NetworkExtended


def multi_network_results_for_injection(
    network_specs: List[List[str]],
    inj: NDArray[np.float64],
    base_params: Dict[str, Union[int, float]],
    wf_dict: Dict[str, Union[str, Optional[Dict[str, str]], bool, int]],
    deriv_dict: Dict[
        str,
        Union[
            str,
            Tuple[str, ...],
            List[Set[str]],
            bool,
            Optional[Dict[str, Union[float, str, int]]],
        ],
    ],
    misc_settings_dict: Dict[str, Optional[int]],
    debug: bool = False,
) -> Dict[str, Tuple[float, ...]]:
    """Returns the benchmark as a dict of tuples for a single injection using the inj and base_params and the settings dicts through the networks in network_specs.

    If a single network fails an injection, then the unified results will save it as a np.nan in all networks so that the universe of injections is the same between each network. TODO: check that this doesn't bias the results away from loud sources that we care about.

    Args:
        network_specs: Networks to pass to gwbench's multi-network pipeline.
        inj: Injection parameters for each injection, e.g. chirp mass and luminosity distance.
        base_params: Common parameters among injections, e.g. time of coalesence.
        wf_dict: Waveform dictionary of model name and options, also contains the science case string.
        deriv_dict: Derivative options dictionary.
        misc_settings_dict: Options for gwbench, e.g. whether to account for Earth's rotation about its axis.
        debug: Whether to debug.

    Returns:
        Dict[str, Tuple[float]]: Keys are repr(network_spec). Each value is (redshift, SNR, logMc err, logDL err, eta err, iota err, 90%-credible sky-area in sqr degrees) or a tuple of seven np.nan's if the injection failed in any network.
    """
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

    # subtlety, if V+ (or aLIGO) is present in any network, then f is truncated for V+ for all networks (since f is shared below). TODO: figure out how common this is
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

            # TODO: if using gwbench 0.7, still introduce a limit on net.cond_num based on machine precision errors that mpmath is blind to
            # if the FIM is zero, then the condition number is NaN and matrix is ill-conditioned (according to gwbench). TODO: try catching this by converting warnings to errors following <https://stackoverflow.com/questions/5644836/in-python-how-does-one-catch-warnings-as-if-they-were-exceptions#30368735> --> 54 and 154 converged in a second run
            if not net.wc_fisher:
                # unified injection rejection so that cosmological resampling can be uniform across networks, this now means that the number of injections is equal to that of the weakest network in the set but leads to a better comparison
                if debug:
                    print(
                        f"Rejected injection for {network_spec} and, therefore, all networks in the multi-network because of ill-conditioned FIM ({net.fisher}) with condition number ({net.cond_num}) greater than 1e15"
                    )
                return dict(
                    (repr(network_spec_2), tuple(np.nan for _ in range(7)))
                    for network_spec_2 in network_specs
                )
            #                 multi_network_results_dict[repr(network_spec)] = tuple(
            #                     np.nan for _ in range(7)
            #                 )
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
    results_file_name: str,
    network_specs: List[List[str]],
    injections_file: str,
    num_injs_per_redshift_bin: int,
    process_injs_per_task: Optional[int],
    base_params: Dict[str, Union[int, float]],
    wf_dict: Dict[str, Union[str, Optional[Dict[str, str]], bool, int]],
    deriv_dict: Dict[
        str,
        Union[
            str,
            Tuple[str, ...],
            List[Set[str]],
            bool,
            Optional[Dict[str, Union[float, str, int]]],
        ],
    ],
    misc_settings_dict: Dict[str, Optional[int]],
    data_path: str = "./data_processed_injections/task_files/",
    debug: int = False,
) -> None:
    """Runs the injections in the given file through the given set of networks and saves them as a .npy file.

    Benchmarks the first process_injs_per_task number of injections from injections_file + base_params for each of the networks in network_specs for the science_case and other settings in the three dict.'s provided, saves the results as a .npy file in results_file_name at data_path in the form (number of surviving injections, 7) with the columns of (redshift, SNR, logMc err, logDL err, eta err, iota err, 90%-credible sky-area in sqr degrees).

    Args:
        results_file_name: Output .npy filename template for each of the network results. Of the form f"SLURM_TASK_{task_id}" if to be generated automatically later. TODO: check whether this works without the task_id format.
        network_specs: Set of networks to analyse.
        injections_file: Input injections filename with path.
        num_injs_per_redshift_bin: Total number of injections from the injections file across all tasks (used for labelling).
        process_injs_per_task: Number of injections to process, does all of them if None.
        base_params: Common parameters among injections, e.g. time of coalesence.
        wf_dict: Waveform dictionary of model name and options, also contains the science case string.
        deriv_dict: Derivative options dictionary.
        misc_settings_dict: Options for gwbench, e.g. whether to account for Earth's rotation about its axis.
        data_path: Path to the output processed data file for the task.
        debug: Whether to debug.

    Raises:
        Exception: If any of the target results files already exist.
    """
    inj_data: NDArray[NDArray[np.float64]] = np.load(injections_file)
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
            num_injs_per_redshift_bin,
            file_name=results_file_name,
            data_path=data_path,
        )
        # includes path, injs-per-zbin is num_injs_per_redshift_bin input to generate_injections (e.g. will be 250k)
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
            print(
                "All calculated values are NaN (might not be this network's fault however). Saving empty array with shape=(0, 7).",
                results_file_name_list[i],
                multi_network_results_dict_list,
            )
            # now just saving an empty array if all results are NaN, some saved injs have high losses, one could have all failures
        #             raise ValueError("All calculated values are NaN.")
        np.save(results_file_name_list[i], results)

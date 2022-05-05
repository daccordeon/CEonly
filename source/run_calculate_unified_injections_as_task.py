#!/usr/bin/env python3
"""Runs a pre-generated set of injections through a given set of networks using the multi-network feature of gwbench.

Using a specified task index, finds the corresponding injection parameters data file (.npy) and calls calculate_unified_injections.py on the injections with the options set below. An output file (.npy) is produced.

Usage:
    Called in a job array by a slurm bash script, e.g.
    $ python3 run_calculate_unified_injections_as_task.py TASK_ID

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
# TODO: update Tuple to tuple when upgraded to Python 3.9+, similarly throughout codebase
from typing import List, Set, Dict, Tuple, Optional, Union
import sys
import glob

from lal import GreenwichMeanSiderealTime

from networks import NET_LIST, BS2022_SIX
from generate_symbolic_derivatives import generate_symbolic_derivatives
from calculate_unified_injections import multi_network_results_for_injections_file


def settings_from_task_id(
    task_id: int,
    inj_data_path: str = "./data_raw_injections/task_files/",
) -> Tuple[str, Dict[str, Union[str, Optional[Dict[str, str]], bool, int]], int]:
    """Returns injection file (with path), waveform parameters in a dictionary, and number of injections for the given task id.

    Args:
        task_id: Slurm task ID from 1 to 2048.
        inj_data_path: Path to injection files.

    Raises:
        ValueError: If there are no matching or more than one matching injections files.
            Also, if the science case is not recognised to set the waveform parameters.
    """
    # TODO: rewrite injection_file_name in generate_injections to use it here?
    matches = glob.glob(inj_data_path + f"*_TASK_{task_id}.npy")
    if len(matches) != 1:
        raise ValueError(
            f"Number of matches in data_raw_injections/ path is not one: {len(matches)}"
        )
    # includes absolute path
    file = matches[0]
    science_case, num_injs_per_redshift_bin_str = (
        file.replace("_INJS-PER-ZBIN_", "_SCI-CASE_")
        .replace("_TASK_", "_SCI-CASE_")
        .replace(".npy", "_SCI-CASE_")
        .split("_SCI-CASE_")[1:3]
    )
    num_injs_per_redshift_bin = int(num_injs_per_redshift_bin_str)

    if science_case == "BNS":
        wf_dict = dict(
            wf_model_name="tf2_tidal",
            wf_other_var_dic=None,
            numerical_over_symbolic_derivs=False,
            coeff_fisco=4,
        )
        # TODO: change to more accurate numerical waveform once gwbench 0.7 released
    #         wf_dict = dict(science_case=science_case, wf_model_name='lal_bns', wf_other_var_dic=dict(approximant='IMRPhenomD_NRTidalv2'), numerical_over_symbolic_derivs=True, coeff_fisco = 4)
    elif science_case == "BBH":
        wf_dict = dict(
            wf_model_name="lal_bbh",
            wf_other_var_dic=dict(approximant="IMRPhenomHM"),
            numerical_over_symbolic_derivs=True,
            coeff_fisco=8,
        )
    else:
        raise ValueError("Science case not recognised.")
    wf_dict["science_case"] = science_case

    return file, wf_dict, num_injs_per_redshift_bin


# --- user inputs
task_id = int(sys.argv[1])
# ignore single detector network that is ill-conditioned (sky localisation really poor?) for BNS --> more relevant now that injections are rejected uniformly
# TODO: update mprof if more networks used
# network_specs = [net_spec for net_spec in NET_LIST if net_spec != ['CE2-40-CBO_C']]
network_specs = BS2022_SIX["nets"]
# 1464 is the maximum injs_per_task except for the last task, how many of those (counting from the start of the file) do we use?
process_injs_per_task = None  # defaults to maximum available
# process_injs_per_task = 10
debug = False
# ---

results_file_name = f"SLURM_TASK_{task_id}"
injection_file_name, wf_dict, num_injs_per_redshift_bin = settings_from_task_id(task_id)
# settings: whether to account for the rotation of the earth, whether to only calculate results for the whole network, whether the masses are already redshifted by the injections module, whether to parallelize and if so on how many cores
misc_settings_dict = dict(use_rot=True, only_net=True, redshifted=True, num_cores=None)
tecs, locs = zip(
    *[
        det_spec.split("_")
        for network_spec in network_specs
        for det_spec in network_spec
    ]
)
unique_tecs, unique_locs = list(set(tecs)), list(set(locs))

# derivative settings
# sym_derivs = numerical_over_symbolic_derivs
deriv_dict: Dict[
    str,
    Union[
        str,
        Tuple[str, ...],
        List[Set[str]],
        bool,
        Optional[Dict[str, Union[float, str, int]]],
    ],
] = dict(
    deriv_symbs_string="Mc eta DL tc phic iota ra dec psi",
    conv_cos=("dec", "iota"),
    conv_log=(
        "Mc",
        "DL",
        "lam_t",
    ),  # no error if lam_t not present since gwbench uses ``key in conv_log:''
    unique_tecs=unique_tecs,
    unique_locs=unique_locs,
)
deriv_dict["numerical_over_symbolic_derivs"] = wf_dict["numerical_over_symbolic_derivs"]
if not deriv_dict["numerical_over_symbolic_derivs"]:
    deriv_dict["numerical_deriv_settings"] = None
    # TODO: Slurm gets upset when multiple tasks try to create the derivatives if there aren't any there already, so run in series using `$ python3 generate_symbolic_derivatives.py`. Presently, this just performs a check that they exist but hopefully won't regenerate them in parallel.
    generate_symbolic_derivatives(
        wf_dict["wf_model_name"],
        wf_dict["wf_other_var_dic"],
        deriv_dict["deriv_symbs_string"],
        deriv_dict["unique_locs"],
        misc_settings_dict["use_rot"],
        print_progress=False,
    )
else:
    deriv_dict["numerical_deriv_settings"] = dict(
        step=1e-9, method="central", order=2, n=1
    )

# using lam_t from Ssohrab on 20220421 and gmst0 from gwbench's multi_network.py
base_params = {
    "tc": 0,
    "phic": 0,
    "gmst0": GreenwichMeanSiderealTime(1247227950.0),
}
if wf_dict["science_case"] == "bns" or "tidal" in wf_dict["wf_model_name"]:
    # these can be calculated if m1, m2, Love number, and EoS (i.e. radii) known
    base_params["lam_t"] = 600  # combined dimensionless tidal deformability
    base_params["delta_lam_t"] = 0

if debug:
    print(
        results_file_name,
        network_specs,
        injection_file_name,
        num_injs_per_redshift_bin,
        process_injs_per_task,
        base_params,
        wf_dict,
        deriv_dict,
        misc_settings_dict,
    )

multi_network_results_for_injections_file(
    results_file_name,
    network_specs,
    injection_file_name,
    num_injs_per_redshift_bin,
    process_injs_per_task,
    base_params,
    wf_dict,
    deriv_dict,
    misc_settings_dict,
    debug=debug,
)

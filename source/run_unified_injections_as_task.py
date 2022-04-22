"""James Gardner, April 2022
called in a job array by a slurm bash script, runs a pre-generated set of injections through a given set of networks using the multi-network feature of gwbench"""
from calculate_unified_injections import multi_network_results_for_injections_file
from networks import NET_LIST, BS2022_SIX
from basic_benchmarking import generate_symbolic_derivatives

from lal import GreenwichMeanSiderealTime
import sys
import glob


def settings_from_task_id(
    task_id, inj_data_path="/fred/oz209/jgardner/CEonlyPony/source/injections/"
):
    """returns science case, waveform parameters, and injection file (with path) for the given task_id"""
    # to-do: rewrite injection_file_name in generate_injections to use it here?
    matches = glob.glob(inj_data_path + f"*_TASK_{task_id}.npy")
    if len(matches) != 1:
        raise ValueError(
            f"Number of matches in injections/ path is not one: {len(matches)}"
        )
    # includes absolute path
    file = matches[0]
    science_case, num_injs_per_redshift_bin = (
        file.replace("_NUM-INJS-PER-ZBIN_", "_SCI-CASE_")
        .replace("_TASK_", "_SCI-CASE_")
        .replace(".npy", "_SCI-CASE_")
        .split("_SCI-CASE_")[1:3]
    )
    num_injs_per_redshift_bin = int(num_injs_per_redshift_bin)

    if science_case == "BNS":
        wf_dict = dict(
            wf_model_name="tf2_tidal",
            wf_other_var_dic=None,
            numerical_over_symbolic_derivs=False,
            coeff_fisco=4,
        )
        # to-do: change to more accurate numerical waveform once gwbench 0.7 released
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
# ignore single detector network that is ill-conditioned (sky localisation really poor?) for BNS
# network_specs = [net_spec for net_spec in NET_LIST if net_spec != ['CE2-40-CBO_C']]
network_specs = BS2022_SIX["nets"]
# 1464 is the maximum injs_per_task except for the last task, how many of those (counting from the start of the file) do we use?
process_injs_per_task = None  # defaults to maximum available
# process_injs_per_task = 10
debug = False
# ---

results_file_name = f"SLURM_TASK_{task_id}"
injection_file_name, wf_dict, num_injs_per_redshift_bin = settings_from_task_id(task_id)
misc_settings_dict = dict(use_rot=1, only_net=1, redshifted=1, num_cores=None)
unique_locs = list(
    set(
        [
            det_spec.split("_")[-1]
            for network_spec in network_specs
            for det_spec in network_spec
        ]
    )
)

# sym_derivs = numerical_over_symbolic_derivs
deriv_dict = dict(
    deriv_symbs_string="Mc eta DL tc phic iota ra dec psi",
    conv_cos=("dec", "iota"),
    conv_log=(
        "Mc",
        "DL",
        "lam_t",
    ),  # no error if lam_t not present since gwbench uses ``key in conv_log:''
    unique_locs=unique_locs,
)
deriv_dict["numerical_over_symbolic_derivs"] = wf_dict["numerical_over_symbolic_derivs"]
if not deriv_dict["numerical_over_symbolic_derivs"]:
    deriv_dict["numerical_deriv_settings"] = None
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
    process_injs_per_task,
    base_params,
    wf_dict,
    deriv_dict,
    misc_settings_dict,
    debug=debug,
)

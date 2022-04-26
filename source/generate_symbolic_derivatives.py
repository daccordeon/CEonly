"""James Gardner, April 2022"""
from gwbench import wf_class as wfc
from gwbench import detector_response_derivatives as drd

import os


def generate_symbolic_derivatives(
    wf_model_name,
    wf_other_var_dic,
    deriv_symbs_string,
    locs,
    use_rot,
    output_path=None,
    print_progress=True,
):
    """generate symbolic derivatives, from generate_lambdified_functions.py from S. Borhanian 2020
    use network's wf_model_name, wf_other_var_dic, deriv_symbs_string, and use_rot
    will print 'Done.' when finished unless all files already exist in which it will print as such

    # # how to print settings as a sanity check
    # print('wf_model_name = \'{}\''.format(wf.wf_model_name))
    # print('wf_other_var_dic = {}'.format(wf.wf_other_var_dic))
    # print('deriv_symbs_string = \'{}\''.format(deriv_symbs_string))
    # print('use_rot = %i'%use_rot)"""
    # skip if derivatives already exist
    file_names = [
        "par_deriv_WFM_"
        + wf_model_name
        + "_VAR_"
        + deriv_symbs_string.replace(" ", "_")
        + "_DET_"
        + key
        + ".dat"
        for key in locs
    ]
    file_names.append(
        "par_deriv_WFM_"
        + wf_model_name
        + "_VAR_"
        + deriv_symbs_string.replace(" ra", "")
        .replace(" dec", "")
        .replace(" psi", "")
        .replace(" ", "_")
        + "_DET_"
        + "pl_cr"
        + ".dat"
    )
    path = "lambdified_functions/"
    file_names_existing = [
        file_name for file_name in file_names if os.path.isfile(path + file_name)
    ]
    if len(file_names_existing) < len(file_names):
        # if a file doesn't exist, generate them all again
        # to-do: make this more efficient and just generate the missing files, or, do it in parallel
        # waveform
        wf = wfc.Waveform(wf_model_name, wf_other_var_dic)
        # lambidified detector reponses and derivatives
        drd.generate_det_responses_derivs_sym(
            wf,
            deriv_symbs_string,
            locs=locs,
            use_rot=use_rot,
            user_lambdified_functions_path=output_path,
        )
    elif print_progress:
        print("All lambdified derivatives already exist.")

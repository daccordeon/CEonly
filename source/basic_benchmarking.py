"""benchmarking.py
James Gardner, March 2022"""
from useful_functions import *

from gwbench import network
from gwbench import wf_class as wfc
from gwbench import detector_response_derivatives as drd

import os
        
def generate_symbolic_derivatives(wf_model_name, wf_other_var_dic, deriv_symbs_string, locs, use_rot, output_path=None, print_progress=True):
    """generate symbolic derivatives, from generate_lambdified_functions.py from S. Borhanian 2020
    use network's wf_model_name, wf_other_var_dic, deriv_symbs_string, and use_rot
    will print 'Done.' when finished unless all files already exist in which it will print as such
    
    # # how to print settings as a sanity check
    # print('wf_model_name = \'{}\''.format(wf.wf_model_name))
    # print('wf_other_var_dic = {}'.format(wf.wf_other_var_dic))
    # print('deriv_symbs_string = \'{}\''.format(deriv_symbs_string))
    # print('use_rot = %i'%use_rot)"""
    # skip if derivatives already exist
    file_names = ['par_deriv_WFM_' + wf_model_name + '_VAR_' + deriv_symbs_string.replace(' ', '_') + '_DET_' + key + '.dat' for key in locs]
    file_names.append('par_deriv_WFM_' + wf_model_name + '_VAR_' + deriv_symbs_string.replace(' ra', '').replace(' dec', '').replace(' psi', '').replace(' ', '_') + '_DET_' + 'pl_cr' + '.dat')
    path = 'lambdified_functions/'
    file_names_existing = [file_name for file_name in file_names if os.path.isfile(path + file_name)]
    if len(file_names_existing) < len(file_names):
        # if a file doesn't exist, generate them all again
        # to-do: make this more efficient and just generate the missing files, or, do it in parallel
        # waveform
        wf = wfc.Waveform(wf_model_name, wf_other_var_dic)
        # lambidified detector reponses and derivatives
        drd.generate_det_responses_derivs_sym(wf, deriv_symbs_string, locs=locs, use_rot=use_rot,
                                              user_lambdified_functions_path=output_path)   
    elif print_progress:
        print('All lambdified derivatives already exist.')
        
def basic_network_benchmarking(net, only_net=True, numerical_over_symbolic_derivs=True, numerical_deriv_settings=dict(step=1e-9, method='central', order=2, n=1), hide_prints=True):
    """computes network SNR, measurement errors, and sky area using gwbench FIM analysis
    no return, saves results natively in network (net.snr, net.errs)
    assumes that network is already set up, with waveform set etc."""
    if hide_prints:
        entry_class = HiddenPrints
    else:
        entry_class = PassEnterExit
    with entry_class():
        # compute the WF polarizations and their derivatives
        net.calc_wf_polarizations()
        if numerical_over_symbolic_derivs:
            # --- numerical differentiation ---
            net.calc_wf_polarizations_derivs_num(**numerical_deriv_settings)
        else:
            # --- symbolic differentiation ---
            net.load_wf_polarizations_derivs_sym()
            net.calc_wf_polarizations_derivs_sym()

        # setup antenna patterns, location phase factors, and PSDs
        net.setup_ant_pat_lpf_psds()

        # compute the detector responses and their derivatives
        net.calc_det_responses()
        if numerical_over_symbolic_derivs:       
            # --- numerical differentiation ---
            net.calc_det_responses_derivs_num(**numerical_deriv_settings)
        else:
            # --- symbolic differentiation ---
            net.load_det_responses_derivs_sym()
            net.calc_det_responses_derivs_sym()

        # calculate the network and detector SNRs
        net.calc_snrs(only_net=only_net)
        # calculate the Fisher and covariance matrices, then error estimates
        net.calc_errors(only_net=only_net) #cond_sup=# 1e15 (default) or None (allows all)
        # calculate the 90%-credible sky area (in [deg]^2)
        net.calc_sky_area_90(only_net=only_net)

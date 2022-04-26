"""benchmarking.py
James Gardner, March 2022"""
from useful_functions import *

from gwbench import network


def basic_network_benchmarking(
    net,
    only_net=True,
    numerical_over_symbolic_derivs=True,
    numerical_deriv_settings=dict(step=1e-9, method="central", order=2, n=1),
    hide_prints=True,
):
    """computes network SNR, measurement errors, and sky area using gwbench FIM analysis
    no return, saves results natively in network (net.snr, net.errs)
    assumes that network is already set up, with waveform set etc.
    ensure that step size in numerical_deriv_settings is sufficiently small, e.g. to not step out of bounds above eta = 0.25"""
    if hide_prints:
        entry_class = HiddenPrints
    else:
        entry_class = PassEnterExit
    with entry_class():
        # compute the WF polarizations and their derivatives
        # net.calc_wf_polarizations() # automatically done by calc_wf_polarizations_derivs_...
        # Ssohrab says that this is not required if det response derivs are taken
        #         if numerical_over_symbolic_derivs:
        #             # --- numerical differentiation ---
        #             net.calc_wf_polarizations_derivs_num(**numerical_deriv_settings)
        #         else:
        #             # --- symbolic differentiation ---
        #             net.load_wf_polarizations_derivs_sym()
        #             net.calc_wf_polarizations_derivs_sym()

        # setup antenna patterns, location phase factors, and PSDs, is this required?
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
        net.calc_errors(
            only_net=only_net
        )  # cond_sup=# 1e15 (default) or None (allows all)
        # calculate the 90%-credible sky area (in [deg]^2)
        net.calc_sky_area_90(only_net=only_net)

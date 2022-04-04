"""James Gardner, April 2022"""   
from calculate_injections import calculate_detection_rate_from_results
from useful_functions import HiddenPrints, parallel_map
from constants import SNR_THRESHOLD_LO, SNR_THRESHOLD_HI
from networks import DICT_NETSPEC_TO_COLOUR
from filename_search_and_manipulation import net_label_styler, file_name_to_multiline_readable, find_files_given_networks
from useful_plotting_functions import force_log_grid

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def collate_eff_detrate_vs_redshift(axs, zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, zaxis_plot, colours=None, label=None, parallel=True):
    """collate plots to replicate Fig 2 in B&S2022, adds curves to existing axs.
    use case: collate different networks with data generated/saved using detection_rate_for_network_and_waveform
    defaults to using the same colour"""
    if colours is None:
        colours = [None, None] # list is mutable, None is not

    # efficiency vs redshift
    # re-ordered plots to re-order legend
    line_lo, = axs[0].semilogx(zaxis_plot, det_eff_fits[0](zaxis_plot), color=colours[0], label=label)
    if colours[1] is None:
        colours[1] = line_lo.get_color()    
    line_hi, = axs[0].semilogx(zaxis_plot, det_eff_fits[1](zaxis_plot), color=colours[1], linestyle='--')    
    axs[0].plot(zavg_efflo_effhi[:,0], zavg_efflo_effhi[:,1], 'o', color=line_lo.get_color(), label=fr'$\rho$ > {SNR_THRESHOLD_LO}')
    axs[0].plot(zavg_efflo_effhi[:,0], zavg_efflo_effhi[:,2], 's', color=line_hi.get_color(), label=fr'$\rho$ > {SNR_THRESHOLD_HI}')

    # explicitly setting legend
#     plt.plot(np.linspace(1, 1000, 10), np.arange(10), 'o-', label='test')
#     plt.plot(np.linspace(1, 1000, 10), np.arange(10), 's--', label='test2')
#     plt.legend()
    
    # detection rate vs redshift
    # merger rate depends on star formation rate and the delay between formation and merger
    # use display_progress_bar in parallel_map to restore old p_map usage
    axs[1].loglog(zaxis_plot, parallel_map(lambda z : det_rate(z, snr_threshold=10),  zaxis_plot, parallel=parallel), color=line_lo.get_color())
    axs[1].loglog(zaxis_plot, parallel_map(lambda z : det_rate(z, snr_threshold=100), zaxis_plot, parallel=parallel), color=line_hi.get_color(), linestyle='--')

def compare_detection_rate_of_networks_from_saved_results(network_spec_list, science_case, save_fig=True, show_fig=True, plot_label=None, full_legend=False, specific_wf=None, print_progress=True, data_path='data_redshift_snr_errs_sky-area/', parallel=True):
    """replication of Fig 2 in B&S2022, use to check if relative detection rates are correct
    even if the absolute detection rate is wildly (1e9) off
    network_spec_list is assumed unique.
    uses uniformly sampled results in redshift to have good resolution along detection rate curve"""
    # finding file names
    net_labels = [net_label_styler('..'.join(network_spec)) for network_spec in network_spec_list]
    if plot_label is None:
        plot_label = f"SCI-CASE_{science_case}{''.join(tuple('_NET_'+l for l in net_labels))}"
    
    found_files = find_files_given_networks(network_spec_list, science_case, specific_wf=specific_wf, print_progress=print_progress, data_path=data_path, raise_error_if_no_files_found=False)
    if found_files is None or len(found_files) == 0:
        return

    # load file and add results to plot
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'wspace':0, 'hspace':0.05})
    zaxis_plot = np.geomspace(1e-2, 50, 100)
    
    axs[0].axhline(0, color='grey', linewidth=0.5)
    axs[0].axhline(1, color='grey', linewidth=0.5)
    axs[0].set_ylim((0-0.05, 1+0.05))
    axs[0].set_ylabel(r'detection efficiency, $\varepsilon$') 
    axs[1].set_ylim((1e-1, 6e5)) # to match B&S2022 Fig 2    
    axs[1].set_ylabel(r'detection rate, $D_R$ / $\mathrm{yr}^{-1}$')  
    fig.align_ylabels()
    axs[-1].set_xscale('log')
    axs[-1].set_xlim((zaxis_plot[0], zaxis_plot[-1]))
    axs[-1].xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=0.1*np.arange(1, 10), numticks=10))
    axs[-1].xaxis.set_minor_formatter(plt.NullFormatter())
    axs[-1].set_xlabel('redshift, z')

    colours_used = []
    for i, file in enumerate(found_files):
        results = np.load(data_path + file)
        with HiddenPrints():
            zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, _, _ = \
                calculate_detection_rate_from_results(results, science_case, print_reach=False)    
        # to not repeatedly plot merger rate
        if i == 0:
            axs[1].loglog(zaxis_plot, parallel_map(det_rate_limit, zaxis_plot, parallel=parallel), color='black', linewidth=3, label=f'{science_case} merger rate')
#             print(f'maximum detection rate at z={zaxis_plot[-1]} is {det_rate_limit(zaxis_plot[-1])}')
        
        if full_legend:
            label = file_name_to_multiline_readable(file, two_rows_only=True)
        else:
            label = file_name_to_multiline_readable(file, net_only=True)
            
        # net_spec is stylised from net_label, this is reflected in the keys of DICT_NETSPEC_TO_COLOUR
        net_spec = file.replace('NET_', '_SCI-CASE_').split('_SCI-CASE_')[1].split('..')
        
        if repr(net_spec) in DICT_NETSPEC_TO_COLOUR.keys():
            colour = DICT_NETSPEC_TO_COLOUR[repr(net_spec)]
            # avoid duplicating colours in plot
            if colour in colours_used:
                colour = None
            else:
                colours_used.append(colour)
        else:
            colour = None

        collate_eff_detrate_vs_redshift(axs, zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, zaxis_plot, label=label, colours=[colour, colour], parallel=parallel)

    handles, labels = axs[0].get_legend_handles_labels()
    # updating handles
    new_handles = list(np.array([[
        mlines.Line2D([], [], visible=False),
        mlines.Line2D([], [], marker='o', linestyle='-', color=handle.get_c()),
        mlines.Line2D([], [], marker='s', linestyle='--', color=handle.get_c())] for handle in handles[::3]]).flatten())
    axs[0].legend(handles=new_handles, labels=labels, handlelength=2, bbox_to_anchor=(1.04,1), loc="upper left")
    axs[1].legend(handlelength=2, loc="upper left")
    
    fig.canvas.draw()
    force_log_grid(axs[0], log_axis='x')
    force_log_grid(axs[-1], log_axis='both')    
    
    if save_fig:
        fig.savefig(f'plots/collated_eff_rate_vs_z/collated_eff_rate_vs_z_{plot_label}.pdf', bbox_inches='tight')
    if show_fig:
        plt.show(fig)
    plt.close(fig)

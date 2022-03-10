"""James Gardner, March 2022"""
from useful_functions import *
from constants import *
from filename_input_and_manipulations import *

def add_measurement_errs_CDFs_to_axs(axs, results_reordered, num_bins, colour, label):
    """add PDFs wrt dlog(x) and CDFs on log-log scale to axs"""
    for i, data in enumerate(results_reordered):
        data.sort() # sorts in place
        log_bins = np.geomspace(data.min(), data.max(), num_bins)
        # density vs weights is normalising integral (area under the curve) vs total count, integral behaves counter-intuitively visually with logarithmic axis 
#         label = legend_label if i==0 else None
        axs[0, i].hist(data, weights=np.full(len(data), 1/len(data)), histtype='step', bins=log_bins, color=colour, label=label)

        # # binned CDF
        # axs[1, i].hist(data, density=True, cumulative=True, histtype='step', color='k', bins=log_bins)
        # axs[1, i].hist(data, density=True, cumulative=-1, histtype='step', color='r', bins=log_bins)
        # unbinned CDF
        cdf = np.arange(len(data))/len(data)
        axs[1, i].plot(data, cdf, color=colour, label=label, zorder=2)
        if i == 0:
            # invert SNR CDF to ``highlight behaviour at large values'' - B&S2022
            axs[1, i].plot(data, 1-cdf, linestyle='--', color=colour, label=label)

def collate_measurement_errs_CDFs_of_networks(network_spec_list, science_case, specific_wf=None, num_bins=20, save_fig=True, show_fig=True, plot_label=None, full_legend=False, print_progress=True):
    """collate PDFs-dlog(x) and CDFs of SNR, sky-area, and measurement errs for given networks"""
    found_files = find_files_given_networks(network_spec_list, science_case, specific_wf=specific_wf, print_progress=print_progress)
    net_labels = [network.Network(network_spec).label for network_spec in network_spec_list]
    if plot_label is None:
        plot_label = ''.join(tuple('_NET_'+l for l in net_labels))[1:]

    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(2, 6, sharex='col', sharey='row', figsize=(16, 4), gridspec_kw=dict(wspace=0.05, hspace=None))
    
    # same colours manipulation as compare_networks_from_saved_results
    colours_used = []
    for i, file in enumerate(found_files):
        # redshift (z), integrated SNR (rho), measurement errors (logMc, logDL, eta, iota), 90% credible sky area
        # errs: fractional chirp mass, fractional luminosity distance, symmetric mass ratio, inclination angle
        results = np.load(f'data_redshift_snr_errs_sky-area/{file}')
                
        if full_legend:
            legend_label = file_name_to_multiline_readable(file, two_rows_only=True)
        else:
            legend_label = file_name_to_multiline_readable(file, net_only=True)
            
        net_spec = file.replace('NET_', '_SCI-CASE_').split('_SCI-CASE_')[1].split('..')

        if repr(net_spec) in DICT_KEY_NETSPEC_VAL_COLOUR.keys():
            colour = DICT_KEY_NETSPEC_VAL_COLOUR[repr(net_spec)]
            # avoid duplicating colours in plot
            if colour in colours_used:
                colour = None
            else:
                colours_used.append(colour)
        else:
            colour = None        
        
        # re-order results columns to have sky-area second
        results_reordered = (results.transpose()[i] for i in (1, -1, *range(2, 6)))
        add_measurement_errs_CDFs_to_axs(axs, results_reordered, num_bins, colour, legend_label) 
        
#     quantity_long_labels = (r'integrated SNR, $\rho$', r'90%-credible sky area, $\Omega$ / $\mathrm{deg}^2$', r'fractional chirp mass error, $\Delta\mathcal{M}/\mathcal{M}$', r'fractional luminosity distance error, $\Delta D_L/D_L$', r'symmetric mass ratio error, $\Delta\eta$', r'inclination angle error, $\Delta\iota$')
    quantity_short_labels = (r'SNR, $\rho$', r'$\Omega$ / $\mathrm{deg}^2$', r'$\Delta\mathcal{M}/\mathcal{M}$', r'$\Delta D_L/D_L$', r'$\Delta\eta$', r'$\Delta\iota$')

    for i in range(len(axs[1])):
        axs[1, i].axhline(1, color='lightgrey', zorder=1)
        axs[1, i].set(xscale='log', yscale='log', xlabel=quantity_short_labels[i])

    # hist handles are boxes, want lines and so borrow from 1, 1 to avoid dotted
    handles, _ = axs[1, 1].get_legend_handles_labels()
    axs[0, 0].legend(handles=handles, handlelength=1, bbox_to_anchor=(0, -1.7), loc="upper left")
    fig.suptitle(r'collated PDFs wrt $\mathrm{d}\log(x)$ and CDFs'+f'\n{plot_label}', y=0.9, verticalalignment='bottom')
    axs[0, 0].set(ylabel='count')
    axs[1, 0].set(ylabel='CDF', ylim=(1e-4, 1e0+1))
    axs[1, 0].legend(labels=['CDF', '1-CDF'], handles=[mlines.Line2D([], [], color='k'), mlines.Line2D([], [], color='k', linestyle='--')], handlelength=1)
    fig.align_labels()

    if save_fig:
        fig.savefig(f'plots/collated_CDFs_snr_errs_sky-area_{plot_label}.pdf')
    if show_fig:
        plt.show()
    plt.close(fig)

"""James Gardner, March 2022"""
from useful_functions import *
from constants import *
from filename_search_and_manipulation import *

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def add_measurement_errs_CDFs_to_axs(axs, results_reordered, num_bins, colour, label, normalise_count=True, threshold_by_SNR=True):
    """add PDFs wrt dlog(x) and CDFs on log-log scale to axs"""
    for i, data in enumerate(results_reordered):
        # using low SNR threshold as cut-off for all non-SNR quantities, this might leave few sources remaining (e.g. for HLVKI+)
        if threshold_by_SNR & (i != 0):
            data = data[results_reordered[0] > SNR_THRESHOLD_LO]
        
        data.sort() # sorts in place
        log_bins = np.geomspace(data.min(), data.max(), num_bins)
        # density vs weights is normalising integral (area under the curve) vs total count, integral behaves counter-intuitively visually with logarithmic axis 
        if normalise_count:
            # normalise wrt dlog(x)
            weights = np.full(len(data), 1/len(data))
        else:
            weights = None
        axs[0, i].hist(data, weights=weights, histtype='step', bins=log_bins, color=colour, label=label)
            
        # # binned CDF
        # axs[1, i].hist(data, density=True, cumulative=True, histtype='step', color='k', bins=log_bins)
        # axs[1, i].hist(data, density=True, cumulative=-1, histtype='step', color='r', bins=log_bins)
        # unbinned CDF
        cdf = np.arange(len(data))/len(data)
        if i == 0:
            # invert SNR CDF to ``highlight behaviour at large values'' - B&S2022
            axs[1, i].plot(data, 1-cdf, linestyle='--', color=colour, label=label, zorder=2)
        else:
            axs[1, i].plot(data, cdf, color=colour, label=label, zorder=2)            

def collate_measurement_errs_CDFs_of_networks(network_spec_list, science_case, specific_wf=None, num_bins=20, save_fig=True, show_fig=True, plot_label=None, full_legend=False, print_progress=True, xlim_list=None, normalise_count=True, threshold_by_SNR=True, plot_title=None, CDFmin=None):
    """collate PDFs-dlog(x) and CDFs of SNR, sky-area, and measurement errs for given networks"""
    found_files = find_files_given_networks(network_spec_list, science_case, specific_wf=specific_wf, print_progress=print_progress)
    net_labels = [network.Network(network_spec).label for network_spec in network_spec_list]
    if plot_label is None:
        plot_label = ''.join(tuple('_NET_'+l for l in net_labels))[1:]
    if plot_title is None:
        plot_title = plot_label

    plt.rcParams.update({'font.size': 14})
    wspace = 0.3 if xlim_list == 'B&S2022' else 0.1
    fig, axs = plt.subplots(2, 6, sharex='col', sharey='row', figsize=(16, 4), gridspec_kw=dict(wspace=wspace, hspace=0.1))
    
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
        results_reordered = [results.transpose()[i] for i in (1, -1, 2, 4, 3, 5)]
        add_measurement_errs_CDFs_to_axs(axs, results_reordered, num_bins, colour, legend_label, normalise_count=normalise_count, threshold_by_SNR=threshold_by_SNR) 
        
    quantity_short_labels = (r'SNR, $\rho$', r'$\Omega$ / $\mathrm{deg}^2$', r'$\Delta\mathcal{M}/\mathcal{M}$', r'$\Delta\eta$', r'$\Delta D_L/D_L$', r'$\Delta\iota$')

    # from B&S 2022
    if xlim_list is None:
        xlim_list = tuple((None, None) for _ in range(6))
    elif xlim_list == 'B&S2022':
        xlim_list = ((1e0, 1e2), (1e-2, 1e2), (1e-7, 1e-4), (1e-6, 1e-3), (1e-2, 1e1), (1e-3, 1e1))
    
    for i in range(len(axs[1])):
        axs[1, i].axhline(1, color='lightgrey', zorder=1)
        axs[1, i].set(xscale='log', yscale='log', xlabel=quantity_short_labels[i], xlim=xlim_list[i])

    for ax in axs[:, 0]:
        ax.axvspan(ax.get_xlim()[0], SNR_THRESHOLD_LO, alpha=0.5, color='lightgrey')
        
    # hist handles are boxes, want lines and so borrow from 1, 1 to avoid dotted
    handles, _ = axs[1, 1].get_legend_handles_labels()
    axs[0, 0].legend(handles=handles, handlelength=1, bbox_to_anchor=(0, -1.7), loc="upper left")
    title = r'collated PDFs wrt $\mathrm{d}\log(x)$ and CDFs'+f'\n{plot_title}'
    if threshold_by_SNR:
        title = title.replace('\n', f', non-SNR quantities thresholded by SNR > {SNR_THRESHOLD_LO}\n')
    else:
        title = title.replace('\n', f', without thresholding by SNR\n')
    fig.suptitle(title, y=0.9, verticalalignment='bottom')
    if normalise_count:
        axs[0, 0].set(ylabel='normalised\ncount wrt height')
    else:
        axs[0, 0].set(ylabel='count')
    if CDFmin == 'B&S2022':
        CDFmin = 1e-4
    axs[1, 0].set(ylabel='CDF', ylim=(CDFmin, 1e0+1))
    axs[1, 0].legend(labels=['1-CDF'], handles=[mlines.Line2D([], [], color='k', linestyle='--')], handlelength=1, loc='lower left')
    axs[1, 1].legend(labels=['CDF'],   handles=[mlines.Line2D([], [], color='k')], handlelength=1, loc='lower left')
    fig.align_labels()

    if save_fig:
        fig.savefig(f'plots/collated_PDFs_and_CDFs_snr_errs_sky-area/collated_PDFs_and_CDFs_snr_errs_sky-area_{plot_label}.pdf', bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close(fig)

"""James Gardner, March 2022"""
from useful_functions import *
from constants import *
from networks import DICT_NETSPEC_TO_COLOUR, BS2022_SIX
from filename_search_and_manipulation import *
from useful_plotting_functions import *

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def add_measurement_errs_CDFs_to_axs(axs, results_reordered, num_bins, colour, linestyle, label, normalise_count=True, threshold_by_SNR=True, contour=True):
    """add PDFs wrt dlog(x) and CDFs on log-log scale to axs.
    the bool contour controls whether to display contours on the CDFs between SNR > 100 (detected well) and SNR > 10 (detected) curves"""
    for i, data in enumerate(results_reordered):
        # using low SNR threshold as cut-off for all non-SNR quantities, this might leave few sources remaining (e.g. for HLVKI+)
        # to-do: rewrite lo and hi to reduce repetition; handle data_hi being empty 
        # add legend for contour like Kuns+2020
        data_hi_empty = False
        if threshold_by_SNR and (i != 0):
            data_lo = data[results_reordered[0] > SNR_THRESHOLD_LO]
            data_hi = data[results_reordered[0] > SNR_THRESHOLD_HI]
            # to-do: fix contour issue as loud sources should have lower errors 
            #if i == 1: print(f'number of sources with SNR > {SNR_THRESHOLD_HI}: {len(data_hi)} which is {len(data_hi)/len(data_lo):.1%} of those with SNR > {SNR_THRESHOLD_LO}, for {label}')
            if len(data_hi) == 0:
                data_hi_empty = True
        if (not (threshold_by_SNR and (i != 0))) or (not contour):
            data_lo = data
            data_hi_empty = True

        data_lo.sort()
        if not data_hi_empty:
            data_hi.sort()
            log_bins = np.geomspace(min(data_lo.min(), data_hi.min()), max(data_lo.max(), data_hi.max()), num_bins)
        else: 
            log_bins = np.geomspace(data_lo.min(), data_lo.max(), num_bins)            
        # density vs weights is normalising integral (area under the curve) vs total count, integral behaves counter-intuitively visually with logarithmic axis 
        if normalise_count:
            # normalise wrt dlog(x), i.e. to the height rather than the actual integrated area of each column
            weights_lo = np.full(len(data_lo), 1/len(data_lo))
            if not data_hi_empty: weights_hi = np.full(len(data_hi), 1/len(data_hi))
        else:
            weights_lo = None
            if not data_hi_empty: weights_hi = None
        
        if contour and (i != 0):
            linewidth_lo, label_lo = 0.5, None
        else:
            linewidth_lo, label_lo = None, label

        if not data_hi_empty:
            # trying with weights_lo, not normalised but check if it is the tail
            axs[0, i].hist(data_hi, weights=np.full(len(data_hi), 1/len(data_lo)), histtype='step', bins=log_bins, color=colour, linestyle=linestyle, label=label)
        axs[0, i].hist(data_lo, weights=weights_lo, histtype='step', bins=log_bins, color=colour, linestyle=linestyle, label=label_lo, linewidth=linewidth_lo)
        
        cdf_lo = np.arange(len(data_lo))/len(data_lo)    
        if i == 0:
            # invert SNR CDF to ``highlight behaviour at large values'' - B&S2022
            # unbinned CDF
            axs[1, i].plot(data_lo, 1 - cdf_lo, color=colour, linestyle=linestyle, zorder=2, label=label_lo)
        else:
            # add legend to ax[1, 1] so that handles can be used by ax[0, 0] later
            axs[1, i].plot(data_lo, cdf_lo, color=colour, linestyle=linestyle, zorder=2, linewidth=linewidth_lo, label=label_lo) 
            if not data_hi_empty: 
                cdf_hi = np.arange(len(data_hi))/len(data_hi)                            
                axs[1, i].plot(data_hi, cdf_hi, color=colour, linestyle=linestyle, zorder=2, label=label)
                axs[1, i].fill(np.append(data_hi, data_lo[::-1]), np.append(cdf_hi, cdf_lo[::-1]), color=colour, alpha=0.1)

def collate_measurement_errs_CDFs_of_networks(network_spec_list, science_case, specific_wf=None, num_bins=20, save_fig=True, show_fig=True, plot_label=None, full_legend=False, print_progress=True, xlim_list=None, normalise_count=True, threshold_by_SNR=True, plot_title=None, CDFmin=None, data_path='data_redshift_snr_errs_sky-area/', linestyles_from_BS2022=False, contour=False):
    """collate PDFs-dlog(x) and CDFs of SNR, sky-area, and measurement errs for given networks"""
    found_files = find_files_given_networks(network_spec_list, science_case, specific_wf=specific_wf, print_progress=print_progress, data_path=data_path, raise_error_if_no_files_found=False)
    if found_files is None or len(found_files) == 0:
        return
    net_labels = [net_label_styler('..'.join(network_spec)) for network_spec in network_spec_list]
    if plot_label is None:
        plot_label = ''.join(tuple('_NET_'+l for l in net_labels))[1:]
    if plot_title is None:
        plot_title = plot_label

    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(2, 6, sharex='col', sharey='row', figsize=(20, 6), gridspec_kw=dict(wspace=0.1, hspace=0.1))
    
    # same colours manipulation as compare_networks_from_saved_results
    colours_used = []
    for i, file in enumerate(found_files):
        # redshift (z), integrated SNR (rho), measurement errors (logMc, logDL, eta, iota), 90% credible sky area
        # errs: fractional chirp mass, fractional luminosity distance, symmetric mass ratio, inclination angle
        results = np.load(data_path + file)
                
        if full_legend:
            legend_label = file_name_to_multiline_readable(file, two_rows_only=True)
        else:
            legend_label = file_name_to_multiline_readable(file, net_only=True)
            
        # net_spec is stylised from net_label
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
            
        if linestyles_from_BS2022:
            linestyle = BS2022_SIX['linestyles'][[net_spec_styler(net) for net in BS2022_SIX['nets']].index(str(net_spec))]
        else:
            linestyle = None
        
        # re-order results columns to have sky-area second
        results_reordered = [results.transpose()[i] for i in (1, -1, 2, 4, 3, 5)]
        add_measurement_errs_CDFs_to_axs(axs, results_reordered, num_bins, colour, linestyle, legend_label, normalise_count=normalise_count, threshold_by_SNR=threshold_by_SNR, contour=contour) 
        
    quantity_short_labels = (r'SNR, $\rho$', r'$\Omega_{90}$ / $\mathrm{deg}^2$', r'$\Delta\mathcal{M}/\mathcal{M}$', r'$\Delta\eta$', r'$\Delta D_L/D_L$', r'$\Delta\iota$')

    # from B&S 2022
    if xlim_list is None:
        xlim_list = tuple((None, None) for _ in range(6))
    elif xlim_list == 'B&S2022':
        if science_case == 'BNS':
            xlim_list = [(1e0, 1e2), (1e-2, 1e2), (1e-7, 1e-4), (1e-6, 1e-3), (1e-2, 1e1), (1e-3, 1e1)]
        elif science_case == 'BBH':
            xlim_list = [(1e0, 1e3), (1e-3, 1e2), (1e-6, 1e-1), (1e-7, 1e-2), (1e-3, 1e0), (1e-3, 1e0)]
        else:
            raise ValueError('Science case not recognised.')
        # perturb the xlim_list to avoid ticklabels overhanging the boundary of each subplot, perturbation must work on any scale
        epsilon = 1e-5
        xlim_list = [(xlim[0]*(1 + epsilon), xlim[1]*(1 - epsilon)) for xlim in xlim_list]
    
    for i in range(len(axs[1])):         
        axs[1, i].axhline(1, color='lightgrey', zorder=1)
        axs[1, i].set(xscale='log', yscale='log', xlabel=quantity_short_labels[i], xlim=xlim_list[i])

    # SNR threshold
    for ax in axs[:, 0]:   
        ax.axvspan(ax.get_xlim()[0], SNR_THRESHOLD_LO, alpha=0.5, color='lightgrey')
        
    # sky area threshold
    for ax in axs[:, 1]:
        ax.axvspan(TOTAL_SKY_AREA_SQR_DEG, ax.get_xlim()[1], alpha=0.5, color='k')
        ax.axvline(EM_FOLLOWUP_SKY_AREA_SQR_DEG, color='k', linestyle='--', linewidth=1)       
        
    # hist handles are boxes, want lines and so borrow from 1, 1 to avoid dotted
    handles, _ = axs[1, 0].get_legend_handles_labels()
    axs[0, 0].legend(handles=handles, handlelength=1.5, bbox_to_anchor=(0, -1.35), loc="upper left")
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
    axs[1, 0].legend(labels=['1-CDF'], handles=[mlines.Line2D([], [], color='k')], handlelength=1, loc='lower left', frameon=False)
    if contour:
        axs[0, 1].legend(labels=[f'SNR>{SNR_THRESHOLD_LO}', f'SNR>{SNR_THRESHOLD_HI}'], handles=[mlines.Line2D([], [], color='k', linewidth=0.5), mlines.Line2D([], [], color='k')], handlelength=1, loc='upper left', frameon=False, labelspacing=0.03)
        add_SNR_contour_legend(axs[1, 1])
    else:
        axs[1, 1].legend(labels=['CDF'], handles=[mlines.Line2D([], [], color='k')], handlelength=1, loc='upper left', frameon=False)
    fig.align_labels()

    fig.canvas.draw()
    for ax in axs[0]:
        force_log_grid(ax, log_axis='x')
    for ax in axs[1]:
        force_log_grid(ax, log_axis='both')        
    
    if save_fig:
        fig.savefig(f'plots/collated_PDFs_and_CDFs_snr_errs_sky-area/collated_PDFs_and_CDFs_snr_errs_sky-area_{plot_label}.pdf', bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close(fig)

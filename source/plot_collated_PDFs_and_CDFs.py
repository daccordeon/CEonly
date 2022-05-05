"""Collates the distributions of the signal-to-noise ratio, 90%-credible sky-area, and measurement errors for different networks.

Uses the .npy processed injections data files.

Usage:
    Requires process injections data files to exist.
    Code structure is similar to plot_collated_detection_rates.py.
    See run_plot_collated_detection_rate_and_PDFs_and_CDFs_as_task.py.
    
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
from typing import List, Set, Dict, Tuple, Optional, Union, Type
from numpy.typing import NDArray
from results_class import InjectionResults
from constants import (
    SNR_THRESHOLD_LO,
    SNR_THRESHOLD_HI,
    SNR_THRESHOLD_MID,
    TOTAL_SKY_AREA_SQR_DEG,
    EM_FOLLOWUP_SKY_AREA_SQR_DEG,
)
from networks import DICT_NETSPEC_TO_COLOUR, BS2022_SIX
from filename_search_and_manipulation import (
    find_files_given_networks,
    network_spec_to_net_label,
    file_name_to_multiline_readable,
    network_spec_styler,
)
from useful_plotting_functions import add_SNR_contour_legend, force_log_grid
from networks import DICT_NETSPEC_TO_COLOUR, BS2022_SIX
from cosmological_redshift_resampler import (
    resample_redshift_cosmologically_from_results,
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from copy import deepcopy


def add_measurement_errs_CDFs_to_axs(
    axs: NDArray[Type[plt.Subplot]],
    resampled_results: NDArray[NDArray[np.float64]],
    num_bins: int,
    colour: Optional[str],
    linestyle: Optional[str],
    label: str,
    normalise_count: bool = True,
    threshold_by_SNR: bool = True,
    contour: bool = True,
    debug: bool = False,
) -> None:
    """Adds the distributions for SNR, sky area, and measurement errors plots for a given network onto existing axes.

    Takes array of results from file cosmologically re-sampled, add PDFs wrt dlog(x) for the x-axis variable x and CDFs on log-log scale to axes.

    Args:
        axs: Existing axes to add plots to.
        resampled_results: Cosmologically resampled results.
        num_bins: Number of sub-bins to split redshift into to calculate PDF/histogram.
        colour: Colour for plot curves.
        linestyle: Linestyle for plot curves.
        label: Legend label for plot.
        normalise_count: Whether to set the yaxis for the PDFs to be normalised to one instead of displaying the actual count histogram-like. The yscale is logarithmic either way.
        threshold_by_SNR: Whether to threshold the non-SNR quantities by SNR.
        contour: Whether to display contours between the different SNR thresholds for non-SNR quantities (between SNR > 100 (detected well) and SNR > 10 (detected)). This crowds the plot. TODO: figure out a way to display the information cleanly.
        debug: Whether to print debug statements.
    """
    # re-order results columns and transpose: [snr, sky-area, err_logMc, err_eta, err_logDL, err_iota]
    results_reordered = resampled_results.transpose()[(1, 6, 2, 4, 3, 5), :]
    snr = results_reordered[0]
    different_linestyle = "--" if linestyle != "--" else "-"

    for i, data in enumerate(results_reordered):
        # using low SNR threshold as cut-off for all non-SNR quantities, this might leave few sources remaining (e.g. for HLVKI+)
        # TODO: rewrite lo, mid, and hi to reduce repetition
        data_mid_empty = False
        data_hi_empty = False
        if threshold_by_SNR and (i != 0) and contour:
            data_lo = data[snr > SNR_THRESHOLD_LO]
            data_mid = data[snr > SNR_THRESHOLD_MID]
            data_hi = data[snr > SNR_THRESHOLD_HI]
            if debug and (i == 1):
                print(
                    f"number of sources with SNR > {SNR_THRESHOLD_HI}: {len(data_hi)} which is {len(data_hi)/len(data_lo):.1%} of those with SNR > {SNR_THRESHOLD_LO}, for {label}"
                )
                print(
                    f"number of sources with SNR > {SNR_THRESHOLD_MID}: {len(data_mid)} which is {len(data_mid)/len(data_lo):.1%} of those with SNR > {SNR_THRESHOLD_LO}, for {label}"
                )
            if len(data_mid) == 0:
                data_mid_empty = True
            if len(data_hi) == 0:
                data_hi_empty = True
        else:
            # deepcopy to avoid errors by sorting data in place and then filtering by the snr array determined pre-sort, redundant without sorting in place
            if i == 0 or not threshold_by_SNR:
                data_lo = deepcopy(data)
            else:
                data_lo = data[snr > SNR_THRESHOLD_LO]
                if debug and (i == 1):
                    print(
                        f"number of sources with SNR > {SNR_THRESHOLD_LO}: {len(data_lo)} which is {len(data_lo)/len(data):.1%} of all injections, for {label}"
                    )
            data_mid_empty = True
            data_hi_empty = True

        # don't sort in place, e.g. data_lo.sort(), since it can re-order data itself if data_lo was the whole column
        data_lo = np.sort(data_lo)
        # TODO: update bins in clever way to be consistent between all present (note that data_hi not empty implies the same about data_mid), if rewriting code to work with lo, mid, and hi then I may as well write it to be general between any list of snr thresholds to be able to change between different densities of plots
        if not data_hi_empty:
            data_hi = np.sort(data_hi)
            log_bins = np.geomspace(
                min(data_lo.min(), data_hi.min()),
                max(data_lo.max(), data_hi.max()),
                num_bins,
            )
        else:
            log_bins = np.geomspace(data_lo.min(), data_lo.max(), num_bins)
        # density vs weights is normalising integral (area under the curve) vs total count, integral behaves counter-intuitively visually with logarithmic axis
        if normalise_count:
            # normalise wrt dlog(x), i.e. to the height rather than the actual integrated area of each column
            weights_lo = np.full(len(data_lo), 1 / len(data_lo))
            # use lo normalisation to see the portion of the curve above the threshold, means that hi curve isn't normalised but this is okay
            if not data_hi_empty:
                weights_hi = np.full(len(data_hi), 1 / len(data_lo))
        else:
            weights_lo = None
            if not data_hi_empty:
                weights_hi = None

        if contour and (i != 0):
            linewidth_lo, label_lo = 0.5, None
        else:
            linewidth_lo, label_lo = None, label

        if not data_hi_empty:
            axs[0, i].hist(
                data_hi,
                weights=weights_hi,
                histtype="step",
                bins=log_bins,
                color=colour,
                linestyle=linestyle,
                label=label,
            )
        axs[0, i].hist(
            data_lo,
            weights=weights_lo,
            histtype="step",
            bins=log_bins,
            color=colour,
            linestyle=linestyle,
            label=label_lo,
            linewidth=linewidth_lo,
        )

        cdf_lo = np.arange(len(data_lo)) / len(data_lo)
        if i == 0:
            # invert SNR CDF to ``highlight behaviour at large values'' - B&S2022
            # unbinned CDF
            axs[1, i].plot(
                data_lo,
                1 - cdf_lo,
                color=colour,
                linestyle=linestyle,
                zorder=2,
                label=label_lo,
            )
        else:
            # add legend to ax[1, 1] so that handles can be used by ax[0, 0] later
            axs[1, i].plot(
                data_lo,
                cdf_lo,
                color=colour,
                linestyle=linestyle,
                zorder=2,
                linewidth=linewidth_lo,
                label=label_lo,
            )
            if not data_hi_empty:
                cdf_hi = np.arange(len(data_hi)) / len(data_hi)
                axs[1, i].plot(
                    data_hi,
                    cdf_hi,
                    color=colour,
                    linestyle=linestyle,
                    zorder=2,
                    label=label,
                )
                axs[1, i].fill(
                    np.append(data_hi, data_lo[::-1]),
                    np.append(cdf_hi, cdf_lo[::-1]),
                    color=colour,
                    alpha=0.1,
                )


def collate_measurement_errs_CDFs_of_networks(
    network_spec_list: List[List[str]],
    science_case: str,
    specific_wf: Optional[str] = None,
    num_bins: int = 20,
    save_fig: bool = True,
    show_fig: bool = True,
    plot_label: Optional[str] = None,
    full_legend: bool = False,
    print_progress: bool = True,
    xlim_list: Optional[Union[Tuple[tuple, ...], str]] = None,
    normalise_count: bool = True,
    threshold_by_SNR: bool = True,
    plot_title: Optional[str] = None,
    CDFmin: Optional[float] = None,
    data_path: str = "./data_processed_injections/",
    linestyles_from_BS2022: bool = False,
    contour: bool = False,
    parallel: bool = False,
    debug: bool = False,
    seed: Optional[int] = None,
    norm_tag: str = "GWTC3",
) -> None:
    """Collates distributions of SNR, sky-area, and measurement errors for different networks.

    Distributions are the probability density function (PDF) wrt the logarithmic axis (i.e. if a histogram with uniform logarithmic width bins was normalised to height instead of the actual integrated area) and the cumulative (CDF) function of the raw variable.

    Args:
        network_spec_list: Set of unique networks to compare.
        science_case: Science case, e.g. 'BNS'.
        specific_wf: If specified, then filters to only show the given waveform.
        num_bins: Number of sub-bins to split redshift into to calculate PDF/histogram.
        save_fig: Whether to save the plot.
        show_fig: Whether to show the plot interactively.
        plot_label: File name to save plot as.
        full_legend: Whether to display a verbose legend.
        print_progress: Whether to print progress statements.
        xlim_list: X-axis limits for the six plots, or a tag to load a preset.
        normalise_count: Whether to set the yaxis for the PDFs to be normalised to one instead of displaying the actual count histogram-like. The yscale is logarithmic either way.
        threshold_by_SNR: Whether to threshold the non-SNR quantities by SNR.
        plot_title: Sub-title for the plot.
        CDFmin: Minimum CDF value to plot.
        data_path: Path to processed injections data files.
        linestyles_from_BS2022: Whether to use the linesstyles from B&S2022
        contour: Whether to display contours between the different SNR thresholds for non-SNR quantities. This crowds the plot. TODO: figure out a way to display the information cleanly.
        parallel: Whether to parallelize the computation.
        debug: Whether to print debug statements.
        seed: Random seed for the cosmological resampling of the uniform-in-linear-redshift injections.
        norm_tag: Survey to normalise cosmological merger rates to.

    Raises:
        ValueError: If the science case is not recognised.
    """
    found_files = find_files_given_networks(
        network_spec_list,
        science_case,
        specific_wf=specific_wf,
        print_progress=print_progress,
        data_path=data_path,
        raise_error_if_no_files_found=False,
    )
    if found_files is None or len(found_files) == 0:
        return
    net_labels = [
        network_spec_to_net_label(network_spec, styled=True)
        for network_spec in network_spec_list
    ]
    if plot_label is None:
        plot_label = "".join(tuple("_NET_" + l for l in net_labels))[1:]
    if plot_title is None:
        plot_title = plot_label

    plt.rcParams.update({"font.size": 14})
    fig, axs = plt.subplots(
        2,
        6,
        sharex="col",
        sharey="row",
        figsize=(26, 4.875),
        gridspec_kw=dict(wspace=0.05, hspace=0.15),
    )

    # same colours manipulation as compare_networks_from_saved_results
    colours_used = []
    for i, file in enumerate(found_files):
        # redshift (z), integrated SNR (rho), measurement errors (logMc, logDL, eta, iota), 90% credible sky area
        # errs: fractional chirp mass, fractional luminosity distance, symmetric mass ratio, inclination angle
        results = InjectionResults(file, data_path=data_path, norm_tag=norm_tag)
        # re-sampling uniform results using a cosmological model, defaults to using a 10-year observation time
        # TODO: unify cosmological resampling between networks when comparing them, requires saving the same injections from benchmarking (i.e. going with the weakest network's rejections)
        resampled_results = resample_redshift_cosmologically_from_results(
            results, parallel=parallel, seed=seed, norm_tag=norm_tag, debug=debug
        )

        if full_legend:
            legend_label = file_name_to_multiline_readable(file, two_rows_only=True)
        else:
            legend_label = file_name_to_multiline_readable(file, net_only=True)

        if repr(results.network_spec) in DICT_NETSPEC_TO_COLOUR.keys():
            colour = DICT_NETSPEC_TO_COLOUR[repr(results.network_spec)]
            # avoid duplicating colours in plot
            if colour in colours_used:
                colour = None
            else:
                colours_used.append(colour)
        else:
            colour = None

        if linestyles_from_BS2022:
            linestyle = BS2022_SIX["linestyles"][
                [network_spec_styler(net) for net in BS2022_SIX["nets"]].index(
                    repr(results.network_spec)
                )
            ]
        else:
            linestyle = None

        if debug and (i == 0):
            print("- - -\n", plot_label)
        add_measurement_errs_CDFs_to_axs(
            axs,
            resampled_results,
            num_bins,
            colour,
            linestyle,
            legend_label,
            normalise_count=normalise_count,
            threshold_by_SNR=threshold_by_SNR,
            contour=contour,
            debug=debug,
        )

    quantity_short_labels = (
        r"SNR, $\rho$",
        r"$\Omega_{90}$ / $\mathrm{deg}^2$",
        r"$\Delta\mathcal{M}/\mathcal{M}$",
        r"$\Delta\eta$",
        r"$\Delta D_L/D_L$",
        r"$\Delta\iota$",
    )

    # from B&S 2022
    if xlim_list is None:
        xlim_list = tuple((None, None) for _ in range(6))
    elif xlim_list == "B&S2022":
        if science_case == "BNS":
            xlim_list = (
                (1e0, 1e2),
                (1e-2, 1e2),
                (1e-7, 1e-4),
                (1e-6, 1e-3),
                (1e-2, 1e1),
                (1e-3, 1e1),
            )
        elif science_case == "BBH":
            xlim_list = (
                (1e0, 1e3),
                (1e-3, 1e2),
                (1e-6, 1e-1),
                (1e-7, 1e-2),
                (1e-3, 1e0),
                (1e-3, 1e0),
            )
        else:
            raise ValueError("Science case not recognised.")
    elif xlim_list == "B&S2022-Figs11--12":
        if science_case == "BNS":
            xlim_list = (
                (1e-2, 5e3),
                (1e-5, 2e5),
                (4e-9, 2e-3),
                (2e-7, 5e-2),
                (5e-5, 4e1),
                (6e-5, 5e1),
            )
        elif science_case == "BBH":
            xlim_list = (
                (1e-2, 1e4),
                (1e-4, 1e7),
                (1e-7, 1e0),
                (1e-9, 2e-1),
                (5e-4, 1e1),
                (4e-4, 1e1),
            )
        else:
            raise ValueError("Science case not recognised.")

    # perturb the xlim_list to avoid ticklabels overhanging the boundary of each subplot, perturbation must work on any scale
    if "B&S2022" in xlim_list:
        epsilon = 1e-5
        xlim_list = [
            (xlim[0] * (1 + epsilon), xlim[1] * (1 - epsilon)) for xlim in xlim_list
        ]

    for i in range(len(axs[1])):
        axs[1, i].axhline(1, color="lightgrey", zorder=1)
        axs[1, i].set(
            xscale="log",
            yscale="log",
            xlabel=quantity_short_labels[i],
            xlim=xlim_list[i],
        )

    # SNR threshold
    for ax in axs[:, 0]:
        ax.axvspan(ax.get_xlim()[0], SNR_THRESHOLD_LO, alpha=0.5, color="lightgrey")

    # sky area threshold
    for ax in axs[:, 1]:
        ax.axvspan(TOTAL_SKY_AREA_SQR_DEG, ax.get_xlim()[1], alpha=0.5, color="k")
        ax.axvline(EM_FOLLOWUP_SKY_AREA_SQR_DEG, color="k", linestyle="--", linewidth=1)

    # hist handles are boxes, want lines and so borrow from 1, 1 to avoid dotted
    handles, _ = axs[1, 0].get_legend_handles_labels()
    axs[0, 0].legend(
        handles=handles, handlelength=1.5, bbox_to_anchor=(0, -1.55), loc="upper left"
    )
    title = r"collated PDFs wrt $\mathrm{d}\log(x)$ and CDFs" + f"\n{plot_title}"
    if threshold_by_SNR:
        title = title.replace(
            "\n", f", non-SNR quantities thresholded by SNR > {SNR_THRESHOLD_LO}\n"
        )
    else:
        title = title.replace("\n", f", without thresholding by SNR\n")
    fig.suptitle(title, y=0.9, verticalalignment="bottom")
    if normalise_count:
        axs[0, 0].set(ylabel="normalised\ncount wrt height")
    else:
        axs[0, 0].set(ylabel="count in bin")
    if CDFmin == "B&S2022":
        CDFmin = 1e-4
    axs[1, 0].set(ylabel="CDF", ylim=(CDFmin, 1 + 0.1))
    axs[1, 0].legend(
        labels=["1-CDF"],
        handles=[mlines.Line2D([], [], color="k")],
        handlelength=0.5,
        loc="lower left",
        frameon=False,
    )
    if contour:
        axs[0, 1].legend(
            labels=[f"SNR>{SNR_THRESHOLD_LO}", f"SNR>{SNR_THRESHOLD_HI}"],
            handles=[
                mlines.Line2D([], [], color="k", linewidth=0.5),
                mlines.Line2D([], [], color="k"),
            ],
            handlelength=1,
            loc="upper left",
            frameon=False,
            labelspacing=0.03,
        )
        add_SNR_contour_legend(axs[1, 1])
    else:
        axs[1, 1].legend(
            labels=["CDF"],
            handles=[mlines.Line2D([], [], color="k")],
            handlelength=0.5,
            loc="upper left",
            frameon=False,
        )
    fig.align_labels()

    fig.canvas.draw()
    for ax in axs[0]:
        ax.yaxis.set_tick_params(labelsize=10)
        force_log_grid(ax, log_axis="both")
    for ax in axs[1]:
        ax.yaxis.set_tick_params(labelsize=10)
        force_log_grid(ax, log_axis="both")

    if save_fig:
        fig.savefig(
            f"plots/collated_PDFs_and_CDFs_snr_errs_sky-area/collated_PDFs_and_CDFs_snr_errs_sky-area_{plot_label}.pdf",
            bbox_inches="tight",
        )
    if show_fig:
        plt.show()
    plt.close(fig)

"""Collates the detection rate plots for the different networks from the .npy processed injections data files.

Usage:
    Requires process injections data files to exist.
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
from useful_functions import HiddenPrints, parallel_map
from constants import SNR_THRESHOLD_LO, SNR_THRESHOLD_HI
from networks import DICT_NETSPEC_TO_COLOUR
from filename_search_and_manipulation import (
    net_label_styler,
    net_label_to_network_spec,
    network_spec_to_net_label,
    filename_to_netspec_sc_wf_injs,
    file_name_to_multiline_readable,
    find_files_given_networks,
)
from useful_plotting_functions import force_log_grid

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def collate_eff_detrate_vs_redshift(
    axs: NDArray[Type[plt.Subplot]],
    results: InjectionResults,
    zaxis_plot: NDArray,
    colours: Optional[List[Optional[str]]] = None,
    label: Optional[str] = None,
    parallel: bool = True,
) -> None:
    """Adds the detection efficiency and rate versus redshift plot for a given network onto existing axes.

    Usage: collate different networks with data generated/saved using InjectionResults.calculate_and_set_detection_rate.

    Args:
        axs: Axes to add plots onto.
        results: Results with calculate_and_set_detection_rate() already run.
        zaxis_plot: Redshift axis to plot over.
        colours: Colours for the detection efficiency and rate plots. Defaults to using the same colour for each plot.
        label: Legend label for the results.
        parallel: Whether to parallelize the computation.
    """
    if colours is None:
        colours = [None, None]  # list is mutable, None is not

    # efficiency vs redshift
    # re-ordered plots to re-order legend
    (line_lo,) = axs[0].semilogx(
        zaxis_plot, results.det_eff_fits[0](zaxis_plot), color=colours[0], label=label
    )
    if colours[1] is None:
        colours[1] = line_lo.get_color()
    (line_hi,) = axs[0].semilogx(
        zaxis_plot,
        results.det_eff_fits[1](zaxis_plot),
        color=colours[1],
        linestyle="--",
    )
    axs[0].plot(
        results.zavg_efflo_effhi[:, 0],
        results.zavg_efflo_effhi[:, 1],
        "o",
        color=line_lo.get_color(),
        label=rf"$\rho$ > {SNR_THRESHOLD_LO}",
    )
    axs[0].plot(
        results.zavg_efflo_effhi[:, 0],
        results.zavg_efflo_effhi[:, 2],
        "s",
        color=line_hi.get_color(),
        label=rf"$\rho$ > {SNR_THRESHOLD_HI}",
    )

    # explicitly setting legend
    #     plt.plot(np.linspace(1, 1000, 10), np.arange(10), 'o-', label='test')
    #     plt.plot(np.linspace(1, 1000, 10), np.arange(10), 's--', label='test2')
    #     plt.legend()

    # detection rate vs redshift
    # merger rate depends on star formation rate and the delay between formation and merger
    # use display_progress_bar in parallel_map to restore old p_map usage
    axs[1].loglog(
        zaxis_plot,
        parallel_map(
            lambda z: results.det_rate(z, snr_threshold=10),
            zaxis_plot,
            parallel=parallel,
        ),
        color=line_lo.get_color(),
    )
    axs[1].loglog(
        zaxis_plot,
        parallel_map(
            lambda z: results.det_rate(z, snr_threshold=100),
            zaxis_plot,
            parallel=parallel,
        ),
        color=line_hi.get_color(),
        linestyle="--",
    )


def compare_detection_rate_of_networks_from_saved_results(
    network_spec_list: List[List[str]],
    science_case: str,
    save_fig: bool = True,
    show_fig: bool = True,
    plot_label: Optional[str] = None,
    full_legend: bool = False,
    specific_wf: Optional[str] = None,
    print_progress: bool = True,
    data_path: str = "/fred/oz209/jgardner/CEonlyPony/source/data_processed_injections/",
    parallel: bool = True,
    debug: bool = False,
    norm_tag: str = "GWTC3",
) -> None:
    """Collates the detection efficiency and rate versus redshift plots for different networks.

    Replication of Fig 2 in B&S2022, use to check if detection rates are correct.
    Uses uniformly sampled results in redshift to have good resolution along detection rate curve, this is actually the main motivation for using a non-physical initial population.

    Args:
        network_spec_list: Set of unique networks to compare.
        science_case: Science case, e.g. 'BNS'.
        save_fig: Whether to save the plot.
        show_fig: Whether to show the plot interactively.
        plot_label: File name to save plot as.
        full_legend: Whether to display a verbose legend.
        specific_wf: If specified, then filters to only show the given waveform.
        print_progress: Whether to print progress statements.
        data_path: Path to processed injections data files.
        parallel: Whether to parallelize the computation.
        debug: Whether to print debug statements.
        norm_tag: Survey to normalise cosmological merger rates to.
    """
    # finding file names
    net_labels = [
        network_spec_to_net_label(network_spec, styled=True)
        for network_spec in network_spec_list
    ]
    if plot_label is None:
        plot_label = (
            f"SCI-CASE_{science_case}{''.join(tuple('_NET_' + l for l in net_labels))}"
        )

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

    # load file and add results to plot
    plt.rcParams.update({"font.size": 14})
    fig, axs = plt.subplots(
        2, 1, sharex=True, figsize=(6, 8), gridspec_kw={"wspace": 0, "hspace": 0.05}
    )
    zaxis_plot = np.geomspace(1e-2, 50, 100)

    axs[0].axhline(0, color="grey", linewidth=0.5)
    axs[0].axhline(1, color="grey", linewidth=0.5)
    axs[0].set_ylim((0 - 0.05, 1 + 0.05))
    axs[0].set_ylabel(r"detection efficiency, $\varepsilon$")
    axs[1].set_ylim((1e-1, 6e5))  # to match B&S2022 Fig 2
    axs[1].set_ylabel(r"detection rate, $D_R$ / $\mathrm{yr}^{-1}$")
    fig.align_ylabels()
    axs[-1].set_xscale("log")
    axs[-1].set_xlim((zaxis_plot[0], zaxis_plot[-1]))
    axs[-1].xaxis.set_minor_locator(
        plt.LogLocator(base=10.0, subs=0.1 * np.arange(1, 10), numticks=10)
    )
    axs[-1].xaxis.set_minor_formatter(plt.NullFormatter())
    axs[-1].set_xlabel("redshift, z")

    colours_used = []
    for i, file in enumerate(found_files):
        results = InjectionResults(file, data_path=data_path, norm_tag=norm_tag)
        with HiddenPrints():
            results.calculate_and_set_detection_rate(print_reach=False)
        # to not repeatedly plot merger rate
        if i == 0:
            axs[1].loglog(
                zaxis_plot,
                parallel_map(results.det_rate_limit, zaxis_plot, parallel=parallel),
                color="black",
                linewidth=3,
                label=f"{results.science_case} merger rate",
            )
        #             print(f'maximum detection rate at z={zaxis_plot[-1]} is {det_rate_limit(zaxis_plot[-1])}')

        if full_legend:
            label = file_name_to_multiline_readable(file, two_rows_only=True)
        else:
            label = file_name_to_multiline_readable(file, net_only=True)

        # network_spec is stylised from net_label, this is reflected in the keys of DICT_NETSPEC_TO_COLOUR
        network_spec, _, _, _, _ = filename_to_netspec_sc_wf_injs(file)

        if repr(network_spec) in DICT_NETSPEC_TO_COLOUR.keys():
            colour = DICT_NETSPEC_TO_COLOUR[repr(network_spec)]
            # avoid duplicating colours in plot
            if colour in colours_used:
                colour = None
            else:
                colours_used.append(colour)
        else:
            colour = None

        if debug:
            print("- - -\n", plot_label)
        collate_eff_detrate_vs_redshift(
            axs,
            results,
            zaxis_plot,
            label=label,
            colours=[colour, colour],
            parallel=parallel,
        )

    handles, labels = axs[0].get_legend_handles_labels()
    # updating handles
    new_handles = list(
        np.array(
            [
                [
                    mlines.Line2D([], [], visible=False),
                    mlines.Line2D(
                        [], [], marker="o", linestyle="-", color=handle.get_c()
                    ),
                    mlines.Line2D(
                        [], [], marker="s", linestyle="--", color=handle.get_c()
                    ),
                ]
                for handle in handles[::3]
            ]
        ).flatten()
    )
    axs[0].legend(
        handles=new_handles,
        labels=labels,
        handlelength=2,
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
    )
    axs[1].legend(handlelength=2, loc="upper left")

    fig.canvas.draw()
    force_log_grid(axs[0], log_axis="x")
    force_log_grid(axs[-1], log_axis="both")

    if save_fig:
        fig.savefig(
            f"plots/collated_eff_rate_vs_z/collated_eff_rate_vs_z_{plot_label}.pdf",
            bbox_inches="tight",
        )
    if show_fig:
        plt.show(fig)
    plt.close(fig)

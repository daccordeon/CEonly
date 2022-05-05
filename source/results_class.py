"""Results class to add structure and attributes to a processed injections .npy data file and calculate detection rates.

Usage:
    >> results = InjectionResults(processed_injections_file_name)
    >> results.print_results()
    >> results.plot_detection_rate()

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

from typing import List, Set, Dict, Tuple, Optional, Union, Callable
from numpy.typing import NDArray
from merger_and_detection_rates import *  # also loads Plank18
from useful_functions import without_rows_w_nan, sigmoid_3parameter, parallel_map
from constants import SNR_THRESHOLD_LO, SNR_THRESHOLD_HI
from filename_search_and_manipulation import (
    network_spec_to_net_label,
    net_label_styler,
    filename_to_netspec_sc_wf_injs,
)
from useful_plotting_functions import force_log_grid
from network_subclass import set_file_tags

import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import gmean
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.lines as mlines


class InjectionResults(object):
    """Class for results processing, besides the results array it has common things like the science case, network, and label.

    Information is stored in the filename of the data file, this is extracted for later reference. The columns of the data file are also labelled for readability. The detection efficiency and rate can also be calculated.

    Attributes:
        file_name (str): File name for processed results .npy data file without path (slightly more flexible than this).
        data_path (str): Path to the data file.
        file_name_with_path (str): File name for processed results .npy data file with path.
        results (NDArray[NDArray[np.float64]]): Loaded .npy data file, rows are different injections, columns are different variables.
        redshift (NDArray[np.float64]): Redshift of injections.
        snr (NDArray[np.float64]): Signal-to-noise ratio of injections.
        err_logMc (NDArray[np.float64]): Fractional measurement error of chirp mass of injections.
        err_logDL (NDArray[np.float64]): Fractional measurement error of luminosity distance of injections.
        err_eta (NDArray[np.float64]): Measurement error of symmetric mass ratio of injections.
        err_iota (NDArray[np.float64]): Measurement error of inclination angle of injections.
        sky_area_90 (NDArray[np.float64]): Measurement error of 90%-credible sky-area of injections.
        network_spec (List[str]): Network specification, e.g. ['A+_H', 'A+_L', 'V+_V', 'K+_K', 'A+_I'].
        science_case (str): Science case, e.g. 'BNS'.
        wf_model_name (str): Waveform model name.
        wf_other_var_dic (Optional[Dict[str, str]]): Waveform approximant dictionary.
        num_injs (int): Number of injections per major redshift bin from file name, e.g. for a task file this is the initial total number of injections per redshift major-bin not the initial number in the task or the number remaining.
        remaining_num_injs (int): Number of injections in the data file.
        label (str): Network label, e.g. 'A+_H..A+_L..V+_V..K+_K..A+_I'.
        file_tag (str): File tag for input/output.
        human_file_tag (str): Human-readable file tag.
        norm_tag (str): Survey tag to normalise cosmological merger rates to, e.g. .

        If the file is from a task, then the following attributes will also be set.
        task_id (int): Slurm task ID.
        injections_task_file_name (str): File name to injections task data file with path.
        initial_task_num_injs (int): Number of injections in the raw injections data file.

        If calculate_and_set_detection_rate is run, then the following attributes will also be set.
        zmin_plot (float): Minimum redshift for plotting.
        zmax_plot (float): Maximum redshift for plotting.
        zavg_efflo_effhi (NDArray[NDArray[np.float64]]): Detection efficiency across the redshift range, for each redshift sub-bin contains the geometric mean and the proportion of sources above the low and high SNR thresholds.
        det_eff_fits (List[Callable[[float], float]]): 3-parameter sigmoid fits to the low and high SNR threshold detection efficiency curves.
        det_rate_limit (Callable[[float], float]): Maximum possible detection rate, i.e. actual number of sources merger rate, at a given redshift.
        det_rate (Callable[[float, float], float]): Detection rate at a given redshift for a given SNR threshold (low or high).
    """

    def __init__(
        self, file_name: str, data_path: Optional[str] = None, norm_tag: str = "GWTC3"
    ):
        """Initialises InjectionResults with all non--detection rate attributes.

        Args:
            file_name: Filename of .npy processed injections data file with or without path depending on whether data_path is given.
            data_path: Path to the data file.
            norm_tag: Survey to normalise cosmological merger rates to.
        """
        if data_path is None:
            if "/" in file_name:
                if "/" == file_name[0]:
                    data_path = ""
                else:
                    # if some relative path is given, then assume it to be local
                    data_path = "./"
            else:
                data_path = "./data_processed_injections/"
                if "_TASK_" in file_name:
                    data_path += "task_files/"

        self.file_name, self.data_path = file_name, data_path
        self.file_name_with_path = self.data_path + self.file_name
        self.results = np.load(self.file_name_with_path)
        (
            self.redshift,
            self.snr,
            self.err_logMc,
            self.err_logDL,
            self.err_eta,
            self.err_iota,
            self.sky_area_90,
        ) = self.results.transpose()
        (
            self.network_spec,
            self.science_case,
            self.wf_model_name,
            self.wf_other_var_dic,
            self.num_injs,
        ) = filename_to_netspec_sc_wf_injs(self.file_name)
        self.remaining_num_injs = self.results.shape[0]
        self.label = network_spec_to_net_label(self.network_spec)
        set_file_tags(self)
        self.norm_tag = norm_tag
        if "_TASK_" in self.file_name:
            self.task_id = int(
                self.file_name.replace(".npy", "_TASK_").split("_TASK_")[1]
            )
            self.injections_task_file_name = glob.glob(
                f"./data_raw_injections/task_files/*TASK_{self.task_id}.npy"
            )[0]
            self.initial_task_num_injs = np.load(self.injections_task_file_name).shape[
                0
            ]

    def calculate_and_set_detection_rate(self, print_reach: bool = False) -> None:
        """Calculates detection rate and auxiliary quantities and sets them as attributes.

        Adapted from legacy calculate_detection_rate_from_results.

        Args:
            print_reach: Whether to print the horizon and reach of the network, i.e. the redshifts to achieve particular values of the detection efficiency wrt both SNR thresholds (0.1% and 50% respectively).

        Raises:
            ValueError: If the science case is not recognised. Also, not appearing in the execution of this function, but if the detection efficiency function calculated here is later used then it can raise this error if the SNR threshold is not recognised.
        """
        # count efficiency over sources in (z, z+Delta_z)
        self.zmin_plot, self.zmax_plot, num_zbins_fine = (
            1e-2,
            50,
            40,
        )  # eyeballing 40 bins from Fig 2
        redshift_bins_fine = list(
            zip(
                np.geomspace(self.zmin_plot, self.zmax_plot, num_zbins_fine)[:-1],
                np.geomspace(self.zmin_plot, self.zmax_plot, num_zbins_fine)[1:],
            )
        )  # redshift_bins are too wide
        self.zavg_efflo_effhi = np.empty((len(redshift_bins_fine), 3))
        for i, (zmin, zmax) in enumerate(redshift_bins_fine):
            z_snr_in_bin = self.results[:, 0:2][
                np.logical_and(zmin < self.redshift, self.redshift < zmax)
            ]
            if len(z_snr_in_bin) == 0:
                self.zavg_efflo_effhi[i] = [np.nan, np.nan, np.nan]
            else:
                self.zavg_efflo_effhi[i, 0] = gmean(
                    z_snr_in_bin[:, 0]
                )  # geometric mean, just using zmax is cleaner but less accurate
                self.zavg_efflo_effhi[i, 1] = np.mean(
                    z_snr_in_bin[:, 1] > SNR_THRESHOLD_LO
                )
                self.zavg_efflo_effhi[i, 2] = np.mean(
                    z_snr_in_bin[:, 1] > SNR_THRESHOLD_HI
                )
        self.zavg_efflo_effhi = without_rows_w_nan(self.zavg_efflo_effhi)

        # fit three-parameter sigmoids to efficiency curves vs redshift
        # using initial coeff guesses inspired by Table 9
        # returns popts, pcovs
        # needs high maxfev to converge
        # can use bounds and maxfev together, stack exchange lied!
        p0, bounds, maxfev = [5, 0.01, 0.1], [[0.03, 5e-5, 0.01], [600, 0.2, 2]], 1e5
        popt_lo, _ = curve_fit(
            sigmoid_3parameter,
            self.zavg_efflo_effhi[:, 0],
            self.zavg_efflo_effhi[:, 1],
            method="dogbox",
            p0=p0,
            bounds=bounds,
            maxfev=maxfev,
        )
        if np.all(self.zavg_efflo_effhi[:, 2] == 0):
            popt_hi = 1, -1, 1  # f(z) = 0
        else:
            popt_hi, _ = curve_fit(
                sigmoid_3parameter,
                self.zavg_efflo_effhi[:, 0],
                self.zavg_efflo_effhi[:, 2],
                method="dogbox",
                p0=p0,
                bounds=bounds,
                maxfev=maxfev,
            )
        popts = [popt_lo, popt_hi]

        #         perrs = [np.sqrt(np.diag(pcov)) for pcov in pcovs]
        # lambdas in list comprehension are unintuitive, be explicit unless confident, see:
        # https://stackoverflow.com/questions/6076270/lambda-function-in-list-comprehensions
        # det_eff_fits = [(lambda z : sigmoid_3parameter(z, *popt)) for popt in popts]
        self.det_eff_fits = [
            (lambda z: sigmoid_3parameter(z, *popts[0])),
            (lambda z: sigmoid_3parameter(z, *popts[1])),
        ]
        # print(f'input {p0}\noptimal {list(popt)}\nerrors {perr}')

        # from this point on, I sample the sigmoid fit to the raw data (e.g. for the detection rate)
        # detection efficiency, interpolate from sigmoid fit
        def _det_eff(z: float, snr_threshold: float) -> float:
            """Returns the detection efficiency of sources above a given threshold at a given redshift.

            Calculates the detection efficiency using the 3-parameter sigmoid fits rather than the raw data.

            Args:
                z: Redshift.
                snr_threshold: Signal-to-noise ratio threshold.

            Raises:
                ValueError: If the SNR threshold is not recognised.
            """
            if snr_threshold == 10.0:
                return self.det_eff_fits[0](z)
            elif snr_threshold == 100.0:
                return self.det_eff_fits[1](z)
            else:
                # TODO: add this feature
                raise ValueError(
                    "SNR thresholds other than 10 or 100 are not yet supported"
                )

        # calculate and print reach and horizon
        # want initial guess to be near the transition (high derivative) part of the sigmoid, how?
        reach_initial_guess = 0.1  # pulling from Table 3
        reach_eff, horizon_eff = 0.5, 0.001
        for snr_threshold in (10.0, 100.0):
            # fsolve finds a zero x* of f(x) near an initial guess x0
            reach = fsolve(
                lambda z: _det_eff(z, snr_threshold) - reach_eff, reach_initial_guess
            )[0]
            # use the reach solution as the initial guess for the horizon since strong local slope there
            horizon = fsolve(lambda z: _det_eff(z, snr_threshold) - horizon_eff, reach)[
                0
            ]
            if print_reach:
                print(
                    f"Given SNR threshold rho_* = {snr_threshold:3d}, reach ({1 - reach_eff:.1%}) z_r = {reach:.3f} and horizon ({1 - horizon_eff:.1%}) z_h = {horizon:.3f}"
                )
                if reach == reach_initial_guess:
                    print("! Reach converged to initial guess, examine local slope.")

        normalisations = merger_rate_normalisations_from_gwtc_norm_tag(self.norm_tag)
        if self.science_case == "BNS":
            merger_rate = lambda z: merger_rate_bns(z, normalisation=normalisations[0])
        elif self.science_case == "BBH":
            merger_rate = lambda z: merger_rate_bbh(z, normalisation=normalisations[1])
        else:
            raise ValueError("Science case not recognised.")

        def det_rate_limit(z0: float) -> float:
            """Returns the maximum possible detection rate, i.e. the total number of sources.

            Formula: $D_R(z, \rho_\ast)|_{\varepsilon=1}$ in B&S2022;i.e. "merger rate" in Fig 2, not R(z) but int R(z)/(1+z), i.e. if perfect efficiency.

            Args:
                z0: Maximum redshift to integrate detection rate from zero out to.
            """
            return detection_rate_limit(merger_rate, z0)

        def det_rate(z0: float, snr_threshold: float) -> float:
            """Returns the detection rate above a given threshold.

            Formula: $D_R(z, \rho_\ast)$ in B&S2022.

            Args:
                z0: Maximum redshift to integrate detection rate from zero out to.
                snr_threshold: Signal-to-noise ratio detection threshold.
            """
            return detection_rate(merger_rate, _det_eff, z0, snr_threshold)

        # TODO: do this using global?
        self.det_rate_limit = det_rate_limit
        self.det_rate = det_rate

    def plot_detection_rate(
        self,
        show_fig: bool = False,
        save_fig: bool = True,
        file_extension: Optional[str] = ".pdf",
        print_progress: bool = True,
        parallel: bool = True,
    ) -> None:
        """Plots three-panels of integrated SNR, detection efficiency, and detection rate versus redshift.

        Designed to replicate Fig 2 in B&S2022. Similar to calling the legacy plot_snr_eff_detrate_vs_redshift with given kwargs.

        Args:
            show_fig: Whether to show the plot interactively.
            save_fig: Whether to save the plot, uses a generated filename.
            file_extension: File extension to save plot as.
            print_progress: Whether to print progress statements.
            parallel: Whether to parallelize the computation.
        """
        # checking that detection rate exists and calculating it if it doesn't; pythonic way of try/except AttributeError is an uglier solution
        if any(
            attribute not in vars(self).keys()
            for attribute in (
                "zmin_plot",
                "zmax_plot",
                "zavg_efflo_effhi",
                "det_eff_fits",
                "det_rate_limit",
                "det_rate",
            )
        ):
            self.calculate_and_set_detection_rate()
        # switching to using the same colour but different linestyles for LO and HI SNR threshold
        # colours = 'darkred', 'red'
        colour = "C0"
        zaxis_plot = np.geomspace(self.zmin_plot, self.zmax_plot, 100)

        plt.rcParams.update({"font.size": 14})
        fig, axs = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(6, 12),
            gridspec_kw={"wspace": 0, "hspace": 0.05},
        )

        # SNR vs redshift
        # use integrated SNR rho from standard benchmarking, not sure if B&S2022 use matched filter
        axs[0].loglog(self.redshift, self.snr, ",")
        axs[0].axhspan(0, SNR_THRESHOLD_LO, alpha=0.5, color="lightgrey")
        axs[0].axhspan(0, SNR_THRESHOLD_HI, alpha=0.25, color="lightgrey")
        axs[0].set_ylabel(r"integrated SNR, $\rho$")
        axs[0].set_title(self.human_file_tag, fontsize=14)

        # efficiency vs redshift
        axs[1].axhline(0, color="grey", linewidth=0.5)
        axs[1].axhline(1, color="grey", linewidth=0.5)
        axs[1].plot(
            self.zavg_efflo_effhi[:, 0],
            self.zavg_efflo_effhi[:, 1],
            "o",
            color=colour,
            label=rf"$\rho$ > {SNR_THRESHOLD_LO}",
        )
        axs[1].plot(
            self.zavg_efflo_effhi[:, 0],
            self.zavg_efflo_effhi[:, 2],
            "s",
            color=colour,
            label=rf"$\rho$ > {SNR_THRESHOLD_HI}",
        )
        axs[1].semilogx(zaxis_plot, self.det_eff_fits[0](zaxis_plot), "-", color=colour)
        axs[1].semilogx(
            zaxis_plot, self.det_eff_fits[1](zaxis_plot), "--", color=colour
        )
        handles, labels = axs[1].get_legend_handles_labels()
        new_handles = list(
            np.array(
                [
                    [
                        mlines.Line2D([], [], marker="o", linestyle="-", color=colour),
                        mlines.Line2D([], [], marker="s", linestyle="--", color=colour),
                    ]
                    for handle in handles[::2]
                ]
            ).flatten()
        )
        axs[1].legend(handles=new_handles, labels=labels, handlelength=2)
        axs[1].set_ylim((0 - 0.05, 1 + 0.05))
        axs[1].set_ylabel(r"detection efficiency, $\varepsilon$")
        fig.align_ylabels()

        # detection rate vs redshift
        # merger rate depends on star formation rate and the delay between formation and merger
        # use display_progress_bar in parallel_map to restore old p_map usage
        axs[2].loglog(
            zaxis_plot,
            parallel_map(self.det_rate_limit, zaxis_plot, parallel=parallel),
            color="black",
            linewidth=1,
        )
        axs[2].loglog(
            zaxis_plot,
            parallel_map(
                lambda z: self.det_rate(z, snr_threshold=10),
                zaxis_plot,
                parallel=parallel,
            ),
            "-",
            color=colour,
        )
        axs[2].loglog(
            zaxis_plot,
            parallel_map(
                lambda z: self.det_rate(z, snr_threshold=100),
                zaxis_plot,
                parallel=parallel,
            ),
            "--",
            color=colour,
        )
        axs[2].set_ylim((1e-1, 6e5))  # to match B&S2022 Fig 2
        if print_progress:
            print("Detection rate calculated.")
        axs[2].set_ylabel(r"detection rate, $D_R$ / $\mathrm{yr}^{-1}$")
        axs[-1].set_xscale("log")
        axs[-1].set_xlim((self.zmin_plot, self.zmax_plot))
        axs[-1].xaxis.set_minor_locator(
            plt.LogLocator(base=10.0, subs=0.1 * np.arange(1, 10), numticks=10)
        )
        axs[-1].xaxis.set_minor_formatter(plt.NullFormatter())
        axs[-1].set_xlabel("redshift, z")

        fig.canvas.draw()
        force_log_grid(axs[0], log_axis="both")
        force_log_grid(axs[1], log_axis="x")
        force_log_grid(axs[2], log_axis="both")

        if save_fig:
            filename = (
                f"plots/snr_eff_rate_vs_redshift/snr_eff_rate_vs_redshift_{self.file_tag}"
                + file_extension
            )
            fig.savefig(
                filename,
                bbox_inches="tight",
            )
        if show_fig:
            plt.show(fig)
        plt.close(fig)

    def print_results(self) -> None:
        """Prints out a summary of the results.

        TODO: summarise results, currently just prints column headings and results.
        """
        print(
            r"results.results contains seven (7) columns: redshift $z$, integrated SNR $\rho$, measurement errors *(fractional chirp mass $\log{\mathcal{M}_c}$, fractional luminosity distance $\log{D_L}$, symmetric mass ratio $\eta$, inclination angle $\iota$), 90%-credible sky area $\Omega_{90}$"
        )
        for attribute, value in vars(self).items():
            print(f"{attribute}: {value}")

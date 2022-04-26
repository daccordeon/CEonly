"""James Gardner, April 2022"""
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
import matplotlib.pyplot as plt
from scipy.stats import gmean
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.lines as mlines


class InjectionResults(object):
    """class for results processing, besides the results array at results.results has attributes for common things like finding the science case, network_spec, and label. methods for calculating and plotting detection rate."""

    def __init__(
        self,
        file_name,
        data_path=None,
    ):
        if data_path is None:
            if "/" in file_name:
                if "/" == file_name[0]:
                    data_path = ""
                else:
                    data_path = "./"
            else:
                data_path = "/fred/oz209/jgardner/CEonlyPony/source/data_redshift_snr_errs_sky-area/"

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

    def calculate_and_set_detection_rate(self, print_reach=False):
        """calculate detection rate and auxiliary quantities and set as attributes of self (self.zavg_efflo_effhi, self.det_eff_fits, self.det_rate_limit, self.det_rate, self.zmin_plot, self.zmax_plot).
        adapted from legacy calculate_detection_rate_from_results"""
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
        def det_eff(z, snr_threshold):
            if snr_threshold == 10:
                return self.det_eff_fits[0](z)
            elif snr_threshold == 100:
                return self.det_eff_fits[1](z)
            else:
                # to-do: add this feature
                raise ValueError(
                    "SNR thresholds other than 10 or 100 are not yet supported"
                )

        # calculate and print reach and horizon
        # want initial guess to be near the transition (high derivative) part of the sigmoid, how?
        reach_initial_guess = 0.1  # pulling from Table 3
        reach_eff, horizon_eff = 0.5, 0.001
        for snr_threshold in (10, 100):
            # fsolve finds a zero x* of f(x) near an initial guess x0
            reach = fsolve(
                lambda z: det_eff(z, snr_threshold) - reach_eff, reach_initial_guess
            )[0]
            # use the reach solution as the initial guess for the horizon since strong local slope there
            horizon = fsolve(lambda z: det_eff(z, snr_threshold) - horizon_eff, reach)[
                0
            ]
            if print_reach:
                print(
                    f"Given SNR threshold rho_* = {snr_threshold:3d}, reach ({1 - reach_eff:.1%}) z_r = {reach:.3f} and horizon ({1 - horizon_eff:.1%}) z_h = {horizon:.3f}"
                )
                if reach == reach_initial_guess:
                    print("! Reach converged to initial guess, examine local slope.")

        if self.science_case == "BNS":
            merger_rate = merger_rate_bns
        elif self.science_case == "BBH":
            merger_rate = merger_rate_bbh
        else:
            raise ValueError("Science case not recognised.")

        def det_rate_limit(z0):
            return detection_rate_limit(merger_rate, z0)

        def det_rate(z0, snr_threshold):
            return detection_rate(merger_rate, det_eff, z0, snr_threshold)

        # to-do: do this using global?
        self.det_rate_limit = det_rate_limit
        self.det_rate = det_rate

    def plot_detection_rate(
        self, show_fig=True, print_progress=True, parallel=True, recursed=False
    ):
        """plot three-panels of integrated SNR, detection efficiency, and detection rate versus redshift to replicate Fig 2 in B&S2022. calls plot_snr_eff_detrate_vs_redshift with given kwargs"""
        try:
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
            axs[0].loglog(results[:, 0], results[:, 1], ",")
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
            axs[1].semilogx(
                zaxis_plot, self.det_eff_fits[0](zaxis_plot), "-", color=colour
            )
            axs[1].semilogx(
                zaxis_plot, self.det_eff_fits[1](zaxis_plot), "--", color=colour
            )
            handles, labels = axs[1].get_legend_handles_labels()
            new_handles = list(
                np.array(
                    [
                        [
                            mlines.Line2D(
                                [], [], marker="o", linestyle="-", color=colour
                            ),
                            mlines.Line2D(
                                [], [], marker="s", linestyle="--", color=colour
                            ),
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

            fig.savefig(
                f"plots/snr_eff_rate_vs_redshift/snr_eff_rate_vs_redshift_{self.file_tag}.pdf",
                bbox_inches="tight",
            )
            if show_fig:
                plt.show(fig)
            plt.close(fig)
        except AttributeError:
            if recursed:
                raise AttributeError(
                    "Attribute missing from InjectionResults instance that is not created by calculate_detection_rate upon recursion."
                )
            self.calculate_detection_rate()
            self.plot_detection_rate(
                show_fig=show_fig,
                print_progress=print_progress,
                parallel=parallel,
                recursed=True,
            )

    def print_results(self):
        """prints out summary of results
        to-do: summarise results, currently just prints column headings and results"""
        print(
            r"results.results contains seven (7) columns: redshift $z$, integrated SNR $\rho$, measurement errors *(fractional chirp mass $\log{\mathcal{M}_c}$, fractional luminosity distance $\log{D_L}$, symmetric mass ratio $\eta$, inclination angle $\iota$), 90%-credible sky area $\Omega_{90}$"
        )
        for attribute, value in vars(self).items():
            print(f"{attribute}: {value}")

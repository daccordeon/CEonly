"""James Gardner, March 2022"""
from useful_functions import *
from constants import *
from networks import DICT_NETSPEC_TO_COLOUR
from basic_benchmarking import *
from filename_search_and_manipulation import *

from gwbench.basic_relations import f_isco_Msolar

from scipy.stats import gmean
from scipy.optimize import curve_fit
from scipy.integrate import quad
from astropy.cosmology import Planck18
from tqdm.notebook import tqdm
from scipy.optimize import fsolve
import matplotlib.lines as mlines   

# --- don't use if just trying to profile time (since it slows down) and memory usage --- just comment out the lines below --- only to use mprof line-by-line or get timestamps in plot --- 
#from memory_profiler import profile
# --- to not profile memory usage ---
#def profile(func):
#    """identity function to blank decorator call"""
#    return func

@profile
def save_benchmark_from_generated_injections(net, redshift_bins, num_injs,
                                             mass_dict, spin_dict, redshifted,
                                             base_params, deriv_symbs_string, coeff_fisco,
                                             conv_cos, conv_log, use_rot, only_net,
                                             numerical_over_symbolic_derivs, numerical_deriv_settings,
                                             file_tag, data_path=None, file_name=None, parallel=True):
    """given network and variables, generate injections, benchmark, 
    and save results (snr, errors in logM logDL eta iota, sky area) as .npy
    to-do: tidy up number of arguments"""
    # injection and benchmarking
    # concatenate injection data from different bins
    inj_data = np.empty((len(redshift_bins)*num_injs, 14))
    for i, (zmin, zmax, seed) in enumerate(redshift_bins):
        cosmo_dict = dict(sampler='uniform', zmin=zmin, zmax=zmax)
        # transposed array to get [[Mc0, eta0, ..., z0], [Mc1, eta1, ..., z1], ...]
        # [Mc, eta, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, DL, iota, ra, dec, psi, z]    
        inj_data[i*num_injs:(i+1)*num_injs] = np.array(injections.injections_CBC_params_redshift(cosmo_dict, mass_dict, spin_dict, redshifted, num_injs=num_injs, seed=seed)).transpose()

    def calculate_benchmark_from_injection(inj):
        """given a 14-array of [Mc, eta, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, DL, iota, ra, dec, psi, z],
        returns a 7-tuple of the
        * redshift z,
        * integrated snr,
        * fractional Mc and DL and absolute eta and iota errors,
        * 90% sky area.
        sigma_log(Mc) = sigma_Mc/Mc is fractional error in Mc and similarly for DL, sigma_eta is absolute,
        while |sigma_cos(iota)| = |sigma_iota*sin(iota)| --> error in iota requires rescaling from output"""
        varied_keys = ['Mc', 'eta', 'chi1x', 'chi1y', 'chi1z', 'chi2x', 'chi2y', 'chi2z', 'DL', 'iota', 'ra', 'dec', 'psi', 'z']
        varied_params = dict(zip(varied_keys, inj))
        z = varied_params.pop('z')
        Mc, eta, iota = varied_params['Mc'], varied_params['eta'], varied_params['iota']

        Mtot = Mc/eta**0.6
        #fisco = (6**1.5*PI*Mtot)**-1 # missing some number of Msun, c=1, G=1 factors
        fisco = f_isco_Msolar(Mtot) #4.4/Mtot*1e3 # Hz # from https://arxiv.org/pdf/2011.05145.pdf
        fmin, fmax = 5., float(max(min(coeff_fisco*fisco, 1024), 10)) # to stop f being too small
        # select df from 1/16 (fine from B&S2022) to 10 (coarse) Hz
        df = (fmax-fmin)/(1024-fmin)*10+(1024-fmax)/(1024-fmin)*1/16
        f = np.arange(fmin, fmax, df)
        # to diagnose len(f) == 1 error
        if len(f) == 1: print(f, fmin, fmax, df, coeff_fisco, fisco)

        # net_copy is automatically deleted once out of scope (is copying necessary with Pool()?)
        net_copy = deepcopy(net)
        inj_params = dict(**base_params, **varied_params)
        net_copy.set_net_vars(f=f, inj_params=inj_params, deriv_symbs_string=deriv_symbs_string,
                              conv_cos=conv_cos, conv_log=conv_log, use_rot=use_rot)

        basic_network_benchmarking(net_copy, numerical_over_symbolic_derivs=numerical_over_symbolic_derivs, only_net=only_net,
                                   numerical_deriv_settings=numerical_deriv_settings, hide_prints=True)

        if net_copy.wc_fisher:
            # convert sigma_cos(iota) into sigma_iota
            abs_err_iota = abs(net_copy.errs['cos_iota']/np.sin(iota))
            return (z, net_copy.snr, net_copy.errs['log_Mc'], net_copy.errs['log_DL'], net_copy.errs['eta'],
                    abs_err_iota, net_copy.errs['sky_area_90'])
        else:
            # to-do: check if CE only is still ill-conditioned
            return (z, *np.full(6, np.nan))

    # calculate results: z, snr, errs (logMc, logDL, eta, iota), sky area
    # p_umap is unordered in redshift for greater speed (check)
    results = np.array(parallel_map(calculate_benchmark_from_injection, inj_data, num_cpus=os.cpu_count()-1, unordered=True, parallel=parallel))
    # filter out NaNs
    results = without_rows_w_nan(results)
    if len(results) == 0:
        raise ValueError('All calculated values are NaN, FIM is ill-conditioned.')
    if data_path is None:
        data_path = 'data_redshift_snr_errs_sky-area/'
    if file_name is None:
        file_name = f'results_{file_tag}.npy'
    np.save(data_path+file_name, results)  

# 4*pi to convert from Mpc^3 sr^-1 (sr is steradian) to Mpc^3
def differential_comoving_volume(z): return  4.*PI*Planck18.differential_comoving_volume(z).value
# normalisation of merger rate ($\dot{n}(z)$) to values in https://arxiv.org/pdf/2111.03606v2.pdf
# 1e-9 converts Gpc^-3 to Mpc^-3 to match Planck18, in Fig 2 of Ngetal2021: the ndot_F rate is in Gpc^-3 yr^-1
# injections.py mentions an old arXiv version: https://arxiv.org/pdf/2012.09876v1.pdf
# this states that the ndot form in injections.py is just a proportionality relation, need to normalise
def merger_rate_bns(z): return GWTC3_MERGER_RATE_BNS/injections.bns_md_merger_rate(0)*1e-9*injections.bns_md_merger_rate(z)*differential_comoving_volume(z)
def merger_rate_bbh(z): return GWTC3_MERGER_RATE_BBH/injections.mdbn_merger_rate(0)*1e-9*injections.mdbn_merger_rate(z)*differential_comoving_volume(z)

@profile
def calculate_detection_rate_from_results(results, science_case, print_reach=True):
    """calculting efficiency and detection rate for plotting from results"""
    # count efficiency over sources in (z, z+Delta_z)
    zmin_plot, zmax_plot, num_zbins_fine = 1e-2, 50, 40 # eyeballing 40 bins from Fig 2
    redshift_bins_fine = list(zip(np.geomspace(zmin_plot, zmax_plot, num_zbins_fine)[:-1],
                                  np.geomspace(zmin_plot, zmax_plot, num_zbins_fine)[1:])) # redshift_bins are too wide
    zavg_efflo_effhi = np.empty((len(redshift_bins_fine), 3))
    for i, (zmin, zmax) in enumerate(redshift_bins_fine):
        z_snr_in_bin = results[:,0:2][np.logical_and(zmin < results[:,0], results[:,0] < zmax)]
        if len(z_snr_in_bin) == 0:
            zavg_efflo_effhi[i] = [np.nan, np.nan, np.nan]
        else:
            zavg_efflo_effhi[i,0] = gmean(z_snr_in_bin[:,0]) # geometric mean, just using zmax is cleaner but less accurate
            zavg_efflo_effhi[i,1] = np.mean(z_snr_in_bin[:,1] > SNR_THRESHOLD_LO)
            zavg_efflo_effhi[i,2] = np.mean(z_snr_in_bin[:,1] > SNR_THRESHOLD_HI)
    zavg_efflo_effhi = without_rows_w_nan(zavg_efflo_effhi)    

    # fit three-parameter sigmoids to efficiency curves vs redshift
    # using initial coeff guesses inspired by Table 9
    # returns popts, pcovs
    # needs high maxfev to converge
    # can use bounds and maxfev together, stack exchange lied!
    p0, bounds, maxfev = [5, 0.01, 0.1], [[0.03,5e-5,0.01], [600,0.2,2]], 1e5
    popt_lo, _ = curve_fit(sigmoid_3parameter, zavg_efflo_effhi[:,0], zavg_efflo_effhi[:,1],
                             method='dogbox', p0=p0, bounds=bounds, maxfev=maxfev)
    if np.all(zavg_efflo_effhi[:,2] == 0):
        popt_hi = 1, -1, 1 # f(z) = 0
    else:
        popt_hi, _ = curve_fit(sigmoid_3parameter, zavg_efflo_effhi[:,0], zavg_efflo_effhi[:,2],
                             method='dogbox', p0=p0, bounds=bounds, maxfev=maxfev)
    popts = [popt_lo, popt_hi]
    
#         perrs = [np.sqrt(np.diag(pcov)) for pcov in pcovs]
    # lambdas in list comprehension are unintuitive, be explicit unless confident, see:
    # https://stackoverflow.com/questions/6076270/lambda-function-in-list-comprehensions
    # det_eff_fits = [(lambda z : sigmoid_3parameter(z, *popt)) for popt in popts]
    det_eff_fits = [(lambda z : sigmoid_3parameter(z, *popts[0])), (lambda z : sigmoid_3parameter(z, *popts[1]))]
    # print(f'input {p0}\noptimal {list(popt)}\nerrors {perr}')
    
    # from this point on, I sample the sigmoid fit to the raw data (e.g. for the detection rate)
    # detection efficiency, interpolate from sigmoid fit
    def det_eff(z, snr_threshold):
        if snr_threshold == 10:
            return det_eff_fits[0](z)
        elif snr_threshold == 100:
            return det_eff_fits[1](z)
        else:
            # to-do: add this feature
            raise ValueError("SNR thresholds other than 10 or 100 are not yet supported") 

    # calculate and print reach and horizon
    # want initial guess to be near the transition (high derivative) part of the sigmoid, how?
    reach_initial_guess = 0.1 # pulling from Table 3
    reach_eff, horizon_eff = 0.5, 0.001    
    for snr_threshold in (10, 100):
        # fsolve finds a zero x* of f(x) near an initial guess x0
        reach =   fsolve(lambda z : det_eff(z, snr_threshold) - reach_eff,   reach_initial_guess)[0]
        # use the reach solution as the initial guess for the horizon since strong local slope there
        horizon = fsolve(lambda z : det_eff(z, snr_threshold) - horizon_eff, reach)[0]
        if print_reach:
            print(f"Given SNR threshold rho_* = {snr_threshold:3d}, reach ({1-reach_eff:.1%}) z_r = {reach:.3f} and horizon ({1-horizon_eff:.1%}) z_h = {horizon:.3f}")
            if reach == reach_initial_guess:
                print('! Reach converged to initial guess, examine local slope.')

    if science_case == 'BNS':
        merger_rate = merger_rate_bns
    elif science_case == 'BBH':
        merger_rate = merger_rate_bbh
    else:
        raise ValueError('Science case not recognised.')    

    # i.e. "merger rate" in Fig 2, not R(z) but int R(z)/(1+z), i.e. if perfect efficiency, quad returns (value, error)
    # 1+z factor of time dilation of merger rate in observer frame z away
    def det_rate_limit(z0): return quad(lambda z : merger_rate(z)/(1+z), 0, z0)[0]
    # detection rate
    def det_rate(z0, snr_threshold): return quad(lambda z : det_eff(z, snr_threshold)*merger_rate(z)/(1+z), 0, z0)[0]    
    return zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, zmin_plot, zmax_plot

@profile
def plot_snr_eff_detrate_vs_redshift(results, science_case, zavg_efflo_effhi,
                                    det_eff_fits, det_rate_limit, det_rate,
                                    zmin_plot, zmax_plot,
                                    file_tag, human_file_tag, show_fig=True,
                                    print_progress=True, parallel=True):
    """plotting to replicate Fig 2 in B&S2022
    to-do: tidy up number of arguments"""   
    # switching to using the same colour but different linestyles for LO and HI SNR threshold
#     colours = 'darkred', 'red'
    colour = 'C0'
    zaxis_plot = np.geomspace(zmin_plot, zmax_plot, 100)

    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 12), gridspec_kw={'wspace':0, 'hspace':0.05})

    # SNR vs redshift
    # use integrated SNR rho from standard benchmarking, not sure if B&S2022 use matched filter
    axs[0].loglog(results[:,0], results[:,1], '.')
    axs[0].axhspan(0, SNR_THRESHOLD_LO, alpha=0.5,  color='lightgrey')
    axs[0].axhspan(0, SNR_THRESHOLD_HI, alpha=0.25, color='lightgrey')
    axs[0].set_ylabel(r'integrated SNR, $\rho$')
    axs[0].set_title(human_file_tag, fontsize=14)

    # efficiency vs redshift
    axs[1].axhline(0, color='grey', linewidth=0.5)
    axs[1].axhline(1, color='grey', linewidth=0.5)
    axs[1].plot(zavg_efflo_effhi[:,0], zavg_efflo_effhi[:,1], 'o', color=colour, label=fr'$\rho$ > {SNR_THRESHOLD_LO}')
    axs[1].plot(zavg_efflo_effhi[:,0], zavg_efflo_effhi[:,2], 's', color=colour, label=fr'$\rho$ > {SNR_THRESHOLD_HI}')
    axs[1].semilogx(zaxis_plot, det_eff_fits[0](zaxis_plot), '-',  color=colour)
    axs[1].semilogx(zaxis_plot, det_eff_fits[1](zaxis_plot), '--', color=colour)
    handles, labels = axs[1].get_legend_handles_labels()
    new_handles = list(np.array([[
        mlines.Line2D([], [], marker='o', linestyle='-',  color=colour),
        mlines.Line2D([], [], marker='s', linestyle='--', color=colour)] for handle in handles[::2]]).flatten())
    axs[1].legend(handles=new_handles, labels=labels, handlelength=2)   
    axs[1].set_ylim((0-0.05, 1+0.05))
    axs[1].set_ylabel(r'detection efficiency, $\varepsilon$')
    fig.align_ylabels()

    # detection rate vs redshift
    # merger rate depends on star formation rate and the delay between formation and merger
    # use display_progress_bar in parallel_map to restore old p_map usage    
    axs[2].loglog(zaxis_plot, parallel_map(det_rate_limit, zaxis_plot, parallel=parallel), color='black', linewidth=1)
    axs[2].loglog(zaxis_plot, parallel_map(lambda z : det_rate(z, snr_threshold=10),  zaxis_plot, parallel=parallel), '-',  color=colour)
    axs[2].loglog(zaxis_plot, parallel_map(lambda z : det_rate(z, snr_threshold=100), zaxis_plot, parallel=parallel), '--', color=colour)
    axs[2].set_ylim((1e-1, 6e5)) # to match B&S2022 Fig 2
    if print_progress: print('Detection rate calculated.')
    axs[2].set_ylabel(r'detection rate, $D_R$ / $\mathrm{yr}^{-1}$')  
    axs[-1].set_xscale('log')
    axs[-1].set_xlim((zmin_plot, zmax_plot))
    axs[-1].xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=0.1*np.arange(1, 10), numticks=10))
    axs[-1].xaxis.set_minor_formatter(plt.NullFormatter())
    axs[-1].set_xlabel('redshift, z')
    axs[0].grid(which='both', axis='both', color='lightgrey')
    axs[1].grid(which='both', axis='both', color='lightgrey')
    axs[2].grid(which='both', axis='both', color='lightgrey')    

    fig.savefig(f'plots/snr_eff_rate_vs_redshift/snr_eff_rate_vs_redshift_{file_tag}.pdf', bbox_inches='tight')
    if show_fig:
        plt.show(fig)
    plt.close(fig)
    
# Replicating Borhanian and Sathya 2022 injections and detection rates
@profile
def detection_rate_for_network_and_waveform(network_spec, science_case, wf_model_name, wf_other_var_dic, num_injs, generate_fig=True, show_fig=True, print_progress=True, print_reach=True, data_path=None, file_name=None, parallel=True):
    """initialises network, benchmarks against injections, calculates efficiency and detection rate, plots"""
    # initialisation
    locs = [x.split('_')[-1] for x in network_spec]
    net = network.Network(network_spec)
    net.set_wf_vars(wf_model_name=wf_model_name, wf_other_var_dic=wf_other_var_dic)
    
    if science_case == 'BNS':
        # injection settings - source
        mass_dict = dict(dist='gaussian', mean=1.35, sigma=0.15, mmin=1, mmax=2)
        spin_dict = dict(geom='cartesian', dim=1, chi_lo=-0.05, chi_hi=0.05)
        # zmin, zmax, seed (use same seeds to replicate results)
        # typo in AppA that starts at 0 rather than 0.02 (in main text)?
        redshift_bins = ((0.02, 0.5, 7669), (0.5, 1, 3103), (1, 2, 4431), (2, 4, 5526), (4, 10, 7035), (10, 50, 2785))
        coeff_fisco = 4 # fmax = 4*fisco for BNS, 8*fisco for BBH
    elif science_case == 'BBH':
        # following injection.py and GWTC-2 (AppB.2. Power Law + Peak mass model), to-do: update for GWTC-3?
        # m1 follows power peak, m2 follow uniform in (5 Msun, m1) --> change mmin to 5?
        mass_dict = dict(
            dist='power_peak_uniform',
            mmin       = 5, # 4.59 in GWTC-2, but changing to 5 here to get m2 in correct range
            mmax       = 86.22,
            m1_alpha   = 2.63,
            q_beta     = 1.26,
            peak_frac  = 0.1,
            peak_mean  = 33.07, # assuming that peak_mu is peak_mean?
            peak_sigma = 5.69,
            delta_m    = 4.82,
        )
        spin_dict = dict(geom='cartesian', dim=1, chi_lo=-0.75, chi_hi=0.75)
        redshift_bins = ((0.02, 0.5, 5485), (0.5, 1, 1054), (1, 2, 46), (2, 4, 5553), (4, 10, 5998), (10, 50, 4743))
        coeff_fisco = 8
    else:
        raise ValueError('Science case not recognised.')

    base_params = {
        'tc':    0,
        'phic':  0,
        'gmst0': 0, # assume zero given B2021
        # B&S2022 uses tidal waveforms but turns tides off
        # these can be calculated if m1, m2, Love number, and EoS (i.e. radii) known
        'lam_t': 0, # combined dimensionless tidal deformability
        'delta_lam_t': 0,
    }

    # derivative settings
    # assign with respect to which parameters to take derivatives for the FIM
    deriv_symbs_string = 'Mc eta DL tc phic iota ra dec psi'
    # assign which parameters to convert to log or cos versions for differentiation
    conv_cos = ('dec', 'iota')
    conv_log = ('Mc', 'DL', 'lam_t')

    # network settings: whether to include Earth's rotation and individual detector calculations
    use_rot = 1
    only_net = 1

    # injection settings - other: number of injections per redshift bin (over 6 bins)
    # to-do: refactor file_tag generation
    redshifted = 1 # whether sample masses already redshifted wrt z
    if wf_other_var_dic is not None:
        file_tag = f'NET_{net_label_styler(net.label)}_SCI-CASE_{science_case}_WF_{wf_model_name}_{wf_other_var_dic["approximant"]}_INJS-PER-ZBIN_{num_injs}'
        human_file_tag = f'network: {net_label_styler(net.label).replace("..", ", ")}\nscience case: {science_case}\nwaveform: {wf_model_name} with {wf_other_var_dic["approximant"]}\nnumber of injections per bin: {num_injs}'
    else:
        file_tag = f'NET_{net_label_styler(net.label)}_SCI-CASE_{science_case}_WF_{wf_model_name}_INJS-PER-ZBIN_{num_injs}'
        human_file_tag = f'network: {net_label_styler(net.label).replace("..", ", ")}\nscience case: {science_case}\nwaveform: {wf_model_name}\nnumber of injections per bin: {num_injs}'    
    
    if file_name is None:
        file_name = f'results_{file_tag}.npy'
    elif file_name[:11] == 'SLURM_TASK_':
        file_name = f'results_{file_tag}_TASK_{file_name[11:]}.npy'
    
    if print_progress: print('Network initialised.')
    # use symbolic derivatives if able
    if (wf_model_name == 'tf2') | (wf_model_name == 'tf2_tidal'):
        numerical_over_symbolic_derivs = False    
        generate_symbolic_derivatives(wf_model_name, wf_other_var_dic, deriv_symbs_string, locs, use_rot, print_progress=print_progress)
        numerical_deriv_settings = None
    else:
        numerical_over_symbolic_derivs = True
        numerical_deriv_settings = dict(step=1e-9, method='central', order=2, n=1) # default

    # ------------------------------------------------
    # generate results or skip if previously generated successfully (i.e. not ill-conditioned)
    if data_path is None:
        data_path = 'data_redshift_snr_errs_sky-area/'
    
    if not os.path.isfile(data_path+file_name):
        save_benchmark_from_generated_injections(net, redshift_bins, num_injs, mass_dict, spin_dict, redshifted, base_params, deriv_symbs_string, coeff_fisco, conv_cos, conv_log, use_rot, only_net, numerical_over_symbolic_derivs, numerical_deriv_settings, file_tag, data_path=data_path, file_name=file_name, parallel=parallel)
    else:
        if (not generate_fig) & print_progress:
            print('Results already exist; figure not (re)generated.')
            # to-do: increase efficiency by making this check sooner? this case seems unlikely.
            return

    if generate_fig:
        results = np.load(data_path+file_name)
        if print_progress: print('Results found and loaded.')

        # ------------------------------------------------
        # calculting efficiency and detection rate for plotting
        zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, zmin_plot, zmax_plot = \
            calculate_detection_rate_from_results(results, science_case, print_reach)
    
        if print_progress: print('Detection rate defined, now calculating...')
        
        # ------------------------------------------------
        # plotting
        plot_snr_eff_detrate_vs_redshift(results, science_case,
                                         zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, zmin_plot, zmax_plot,
                                         file_tag, human_file_tag, show_fig=show_fig,
                                         print_progress=print_progress, parallel=parallel)

# Collating different networks saved using the above method to generate B&S2022 Fig 2
def collate_eff_detrate_vs_redshift(axs,
                                    zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, zaxis_plot,
                                    colours=None, label=None):
    """collate plots to replicate Fig 2 in B&S2022, adds curves to existing axs
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
    # to-do: add parallel=True option
    axs[1].loglog(zaxis_plot, parallel_map(lambda z : det_rate(z, snr_threshold=10),  zaxis_plot),  color=line_lo.get_color())
    axs[1].loglog(zaxis_plot, parallel_map(lambda z : det_rate(z, snr_threshold=100), zaxis_plot), color=line_hi.get_color(), linestyle='--')

def compare_detection_rate_of_networks_from_saved_results(network_spec_list, science_case, save_fig=True, show_fig=True, plot_label=None, full_legend=False, specific_wf=None, print_progress=True, data_path='data_redshift_snr_errs_sky-area/'):
    """replication of Fig 2 in B&S2022, use to check if relative detection rates are correct
    even if the absolute detection rate is wildly (1e9) off
    network_spec_list is assumed unique"""
    # finding file names
    net_labels = [net_label_styler(network.Network(network_spec).label) for network_spec in network_spec_list]
    if plot_label is None:
        plot_label = f"SCI-CASE_{science_case}{''.join(tuple('_NET_'+l for l in net_labels))}"
    
    found_files = find_files_given_networks(network_spec_list, science_case, specific_wf=specific_wf, print_progress=print_progress, data_path=data_path)

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
    axs[0].grid(which='both', axis='both', color='lightgrey')   
    axs[-1].grid(which='both', axis='both', color='lightgrey')
    
    colours_used = []
    for i, file in enumerate(found_files):
        results = np.load(data_path + file)
        with HiddenPrints():
            zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, _, _ = \
                calculate_detection_rate_from_results(results, science_case, print_reach=False)    
        # to not repeatedly plot merger rate
        if i == 0:
            axs[1].loglog(zaxis_plot, parallel_map(det_rate_limit, zaxis_plot), color='black', linewidth=3, label=f'{science_case} merger rate')
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

        collate_eff_detrate_vs_redshift(axs, zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, zaxis_plot, label=label, colours=[colour, colour])

    handles, labels = axs[0].get_legend_handles_labels()
    # updating handles
    new_handles = list(np.array([[
        mlines.Line2D([], [], visible=False),
        mlines.Line2D([], [], marker='o', linestyle='-', color=handle.get_c()),
        mlines.Line2D([], [], marker='s', linestyle='--', color=handle.get_c())] for handle in handles[::3]]).flatten())
    axs[0].legend(handles=new_handles, labels=labels, handlelength=2, bbox_to_anchor=(1.04,1), loc="upper left")
    axs[1].legend(handlelength=2, loc="upper left")
    if save_fig:
        fig.savefig(f'plots/collated_eff_rate_vs_z/collated_eff_rate_vs_z_{plot_label}.pdf', bbox_inches='tight')
    if show_fig:
        plt.show(fig)
    plt.close(fig)

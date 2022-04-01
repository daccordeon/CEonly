"""James Gardner, April 2022"""
# to-do: update imports post-refactoring detection_rates.py
from merger_and_detection_rates import *
from useful_functions import *
from constants import *
from networks import DICT_NETSPEC_TO_COLOUR
from basic_benchmarking import *
from filename_search_and_manipulation import *
from useful_plotting_functions import force_log_grid

from gwbench.basic_relations import f_isco_Msolar

from scipy.stats import gmean
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.lines as mlines   
from scipy.stats import loguniform

def save_benchmark_from_generated_injections(net, science_case, tecs, redshift_bins, num_injs, mass_dict, spin_dict, redshifted, base_params, deriv_symbs_string, coeff_fisco, conv_cos, conv_log, use_rot, only_net, numerical_over_symbolic_derivs, numerical_deriv_settings, file_tag, data_path=None, file_name=None, parallel=True):
    """given network and variables, generate injections, benchmark, 
    and save results (snr, errors in logM logDL eta iota, sky area) as .npy
    to-do: tidy up number of arguments"""
    # injection and benchmarking
    # concatenate injection data from different bins
    inj_data = np.empty((len(redshift_bins)*num_injs, 14))
    for i, (zmin, zmax, seed) in enumerate(redshift_bins):
        cosmo_dict = dict(sampler='uniform', zmin=zmin, zmax=zmax)
        # transposed array to get [[Mc0, eta0, ..., z0], [Mc1, eta1, ..., z1], ...] from [Mc, eta, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, DL, iota, ra, dec, psi, z]    
        injection_params = np.array(injections.injections_CBC_params_redshift(cosmo_dict, mass_dict, spin_dict, redshifted, num_injs=num_injs, seed=seed))
        # changing z to logarithmically uniformly sampled, DL and the redshifted Mc change accordingly
        # seed is no longer used, to-do: would have to apply an appropriate transform to numpy random's uniform
        z_vec = loguniform.rvs(zmin, zmax, size=num_injs)
        DL_vec = Planck18.luminosity_distance(z_vec).value
        if redshifted:
            # undo existing shift from z's, then apply new shift to Mc
            injection_params[0] = injection_params[0]*(1. + z_vec)/(1. + injection_params[13])
        injection_params[8] = DL_vec
        injection_params[13] = z_vec
        inj_data[i*num_injs:(i+1)*num_injs] = injection_params.transpose()

    def calculate_benchmark_from_injection(inj):
        """given a 14-array of [Mc, eta, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, DL, iota, ra, dec, psi, z],
        returns a 7-tuple of the
        * redshift z,
        * integrated snr,
        * fractional Mc and DL and absolute eta and iota errors,
        * 90% sky area.
        sigma_log(Mc) = sigma_Mc/Mc is fractional error in Mc and similarly for DL, sigma_eta is absolute,
        while |sigma_cos(iota)| = |sigma_iota*sin(iota)| --> error in iota requires rescaling from output.
        if something goes wrong with the injection, then (z, *np.full(6, np.nan)) will be returned"""
        varied_keys = ['Mc', 'eta', 'chi1x', 'chi1y', 'chi1z', 'chi2x', 'chi2y', 'chi2z', 'DL', 'iota', 'ra', 'dec', 'psi', 'z']
        varied_params = dict(zip(varied_keys, inj))
        z = varied_params.pop('z')
        Mc, eta, iota = varied_params['Mc'], varied_params['eta'], varied_params['iota']
        output_if_injection_fails = (z, *np.full(6, np.nan))
        
        Mtot = Mc/eta**0.6
        # fisco_obs = (6**1.5*PI*(1+z)*Mtot)**-1 # with the mass redshifted by (1+z) in the observer frame (not clear in B&S2022), missing some number of Msun, c=1, G=1 factors
        fisco_obs = f_isco_Msolar((1 + z)*Mtot) #4.4/Mtot*1e3 # Hz # from https://arxiv.org/pdf/2011.05145.pdf
        # chosing fmax in 11 <= coeff_fisco*fisco <= 1024, truncating to boundary values, NB: B&S2022 doesn't include the lower bound which must be included to avoid an IndexError with the automatically truncated fmin from the V+ and aLIGO curves stored in gwbench that start at 10 Hz, this can occur for Mtot > 3520 Msun with massive BBH mergers although those masses are at least an order of magnitude beyond any observed so far
        fmin, fmax = 5., coeff_fisco*fisco_obs
        # lower bound on fmax can be anything greater than f_lowest_allowed_by_PSD + 1/16     
        # from hardcoded PSDs, if aLIGO or V+ (everything else), then threshold fmax >= 11 (6) Hz; fmax is $f_U$ in B&S2022
        are_aLIGO_or_Vplus_used_bool = ('aLIGO' in tecs) or ('V+' in tecs)
        if are_aLIGO_or_Vplus_used_bool:
            fmax_bounds = (11, 1024)
        else:
            fmax_bounds = (6, 1024)
        fmax = float(max(min(fmax, fmax_bounds[1]), fmax_bounds[0]))            
        # if BBH, then discard the injection by returning NaNs if fmax < 12 Hz (7 Hz) for aLIGO or V+ (everything else)
        if science_case == 'BBH':
            if are_aLIGO_or_Vplus_used_bool and (fmax < 12):
                return output_if_injection_fails
            elif (not are_aLIGO_or_Vplus_used_bool) and (fmax < 7):
                return output_if_injection_fails
        # df linearly transitions from 1/16 (fine from B&S2022) to 10 (coarse to save computation time) Hz
        df = (fmax-fmax_bounds[0])/(fmax_bounds[1]-fmax_bounds[0])*10+(fmax_bounds[1]-fmax)/(fmax_bounds[1]-fmax_bounds[0])*1/16
        f = np.arange(fmin, fmax, df)
        
        # net_copy is automatically deleted once out of scope (is copying necessary with Pool()?)
        net_copy = deepcopy(net)
        inj_params = dict(**base_params, **varied_params)
        net_copy.set_net_vars(f=f, inj_params=inj_params, deriv_symbs_string=deriv_symbs_string, conv_cos=conv_cos, conv_log=conv_log, use_rot=use_rot)

        basic_network_benchmarking(net_copy, numerical_over_symbolic_derivs=numerical_over_symbolic_derivs, only_net=only_net, numerical_deriv_settings=numerical_deriv_settings, hide_prints=True)

        if net_copy.wc_fisher:
            # convert sigma_cos(iota) into sigma_iota
            abs_err_iota = abs(net_copy.errs['cos_iota']/np.sin(iota))
            return (z, net_copy.snr, net_copy.errs['log_Mc'], net_copy.errs['log_DL'], net_copy.errs['eta'],
                    abs_err_iota, net_copy.errs['sky_area_90'])
        else:
            return output_if_injection_fails

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
    popt_lo, _ = curve_fit(sigmoid_3parameter, zavg_efflo_effhi[:,0], zavg_efflo_effhi[:,1], method='dogbox', p0=p0, bounds=bounds, maxfev=maxfev)
    if np.all(zavg_efflo_effhi[:,2] == 0):
        popt_hi = 1, -1, 1 # f(z) = 0
    else:
        popt_hi, _ = curve_fit(sigmoid_3parameter, zavg_efflo_effhi[:,0], zavg_efflo_effhi[:,2], method='dogbox', p0=p0, bounds=bounds, maxfev=maxfev)
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
        
    def det_rate_limit(z0):
        return detection_rate_limit(merger_rate, z0)

    def det_rate(z0, snr_threshold):
        return detection_rate(merger_rate, det_eff, z0, snr_threshold)
    
    return zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, zmin_plot, zmax_plot

def plot_snr_eff_detrate_vs_redshift(results, science_case, zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, zmin_plot, zmax_plot, file_tag, human_file_tag, show_fig=True, print_progress=True, parallel=True):
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
    
    fig.canvas.draw()
    force_log_grid(axs[0], log_axis='both')
    force_log_grid(axs[1], log_axis='x')
    force_log_grid(axs[2], log_axis='both')    

    fig.savefig(f'plots/snr_eff_rate_vs_redshift/snr_eff_rate_vs_redshift_{file_tag}.pdf', bbox_inches='tight')
    if show_fig:
        plt.show(fig)
    plt.close(fig)
    
def detection_rate_for_network_and_waveform(network_spec, science_case, wf_model_name, wf_other_var_dic, num_injs, generate_fig=True, show_fig=True, print_progress=True, print_reach=True, data_path=None, file_name=None, parallel=True):
    """initialises network, benchmarks against injections, calculates efficiency and detection rate, plots.
    use case: Replicating Borhanian and Sathya 2022 (B&S2022) injections and detection rates"""
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

    # detector technologies, necessary to know because gwbench has different frequency ranges for the PSDs
    tecs = [detector.tec for detector in net.detectors]
        
    # ------------------------------------------------
    # generate results or skip if previously generated successfully (i.e. not ill-conditioned)
    if data_path is None:
        data_path = 'data_redshift_snr_errs_sky-area/'
    
    if not os.path.isfile(data_path+file_name):
        save_benchmark_from_generated_injections(net, science_case, tecs, redshift_bins, num_injs, mass_dict, spin_dict, redshifted, base_params, deriv_symbs_string, coeff_fisco, conv_cos, conv_log, use_rot, only_net, numerical_over_symbolic_derivs, numerical_deriv_settings, file_tag, data_path=data_path, file_name=file_name, parallel=parallel)
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
        plot_snr_eff_detrate_vs_redshift(results, science_case, zavg_efflo_effhi, det_eff_fits, det_rate_limit, det_rate, zmin_plot, zmax_plot, file_tag, human_file_tag, show_fig=show_fig, print_progress=print_progress, parallel=parallel)

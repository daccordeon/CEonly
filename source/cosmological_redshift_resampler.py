"""James Gardner, April 2022
Following B&S2022 Section 4A: Incorporation of redshift-dependent merger rates
The units of $R(z)$ are $\frac{[\text{count}]}{\text{year}\, [\text{redshift}]}$ since the yearly number of sources in $(z_0,z_0+\Delta z)$ is $\int_{z_0}^{z_0+\Delta z} R(z) \text{d}z$ not $R_i=R(z^\ast)$ where $z^\ast$ is any point in, e.g. the geometric mean of, $(z_0,z_0+\Delta z)$.

Instead of $p_i = \frac{R_i}{\sum_j R_j}$, therefore, I use $q_i = \frac{R_i \Delta z_i}{\sum_j R_j \Delta z_j} \approx \frac{R_i \Delta z_i}{\int R(z) \text{d}z}$ where $\Delta z_i$ is the width of the bin as the probability of selecting the index $i$ of each bin. For a ten-year observation period $\tau=10$, I draw $\tau \int R(z) \text{d}z$ samples from the probability distribution which means that the expected number of samples in each bin is $\frac{\tau R_i \Delta z_i \int R(z) \text{d}z}{\sum_j R_j \Delta z_j}\approx \tau R_i \Delta z_i\approx \tau\int_{z_0}^{z_0+\Delta z} R(z) \text{d}z$ as it should be (instead of $\frac{\tau R_i \int R(z) \text{d}z}{\sum_j R_j}$).

Plotting $\tau R_i \Delta z_i$ against $z$ is *not* the same as plotting $\tau R(z)$ against $z$ since, as the width of each bin decreases, $\tau R_i \Delta z_i \rightarrow 0$. For example, for a uniform merger rate, splitting a bin in half means that there are now two bins next to each other with half the initial number of sources. On a plot, however, it just appears that the number of sources decreased since the split bins look the same."""
from merger_and_detection_rates import *
from useful_functions import flatten_list, parallel_map

import numpy as np
from scipy.stats import gmean
from scipy.stats import rv_discrete

def cosmological_redshift_sample(science_case, zmin=2e-2, zmax=50, num_subzbin=150, norm_tag='GWTC3', observation_time_in_years=10, parallel=True):
    """return the redshift sub-bins with index i and count n_i of the mergers within determined cosmologically in the observers frame.
    merger rate (R(z) in B&S2022) is in [count]/yr/[redshift] so multiply by the number of years and integrate against z to get the actual count.
    for 60k injections per large zbin, 150 samples containing ~2400 injections if uniformly sampled (varying due to randomness and linear sampling)
    to compare to Ssohrab's email use zmin, zmax = 1.7e-3, 50 and num_subzbin=N=75."""
    if norm_tag == 'GWTC3':
        normalisations = (GWTC3_MERGER_RATE_BNS, GWTC3_MERGER_RATE_BBH)
    elif norm_tag == 'GWTC2':
        normalisations = (GWTC2_MERGER_RATE_BNS, GWTC2_MERGER_RATE_BBH)
    
    subzbin = list(zip(np.geomspace(zmin, zmax, num_subzbin + 1)[:-1], np.geomspace(zmin, zmax, num_subzbin + 1)[1:]))
    # using geometric mean to find the log-centre of each bin
    subzbin_centres = [gmean(zbin) for zbin in subzbin]
    subzbin_widths = np.array([zbin[1] - zbin[0] for zbin in subzbin])

    # R_obs(z) in B&S2022, using the merger rate in the observer's frame like when calculating detection rate (this isn't clear in Section 4A of B&S2022)
    if science_case == 'BNS':
        merger_rate = lambda z : merger_rate_in_obs_frame(merger_rate_bns, z, normalisation=normalisations[0])
    elif science_case == 'BBH':
        merger_rate = lambda z : merger_rate_in_obs_frame(merger_rate_bbh, z, normalisation=normalisations[1])
    else:
        raise ValueError('Science case not recognised.')
    # R_i in B&S2022
    subzbin_merger_rate = np.array(parallel_map(merger_rate, subzbin_centres, parallel=parallel))
    # q_i by James instead of p_i from B&S2022, weighting by width of each bin to estimate the actual number of mergers: n_i will approximate the integral of R_obs(z) over the bin
    subzbin_weighted_probs = subzbin_merger_rate*subzbin_widths/np.sum(subzbin_merger_rate*subzbin_widths)

    # "the desired [total, cosmological] number" of mergers over 10 years, integrating the merger rate in the *source* frame over the redshift range
    num_draws = int(observation_time_in_years*quad(merger_rate, zmin, zmax)[0])
    drawn_indicies = rv_discrete(values=(range(num_subzbin), subzbin_weighted_probs)).rvs(size=num_draws)
    # n_i in B&S2022: sample i with probability p_i "up to the desired [total, cosmological] number" of mergers over 10 years
    subzbin_num_samples = np.array([np.sum(drawn_indicies == i) for i in range(num_subzbin)])

    return subzbin, subzbin_num_samples

def resample_redshift_cosmologically_from_results(results, science_case, print_progress=False, print_samples_with_replacement=False, **kwargs):
    """following B&S2022 Section 4A, use a cosmological model of the observed merger rate to uniformly sample n_i times from the saved results data in the subzbin with index i where n_i is determined cosmologically and, ultimately, phenomenologically.
    sampling without replacement is used unless there are insufficient injections, then replacement is used.
    if there aren't injections in a requested bin, then that bin is skipped
    kwargs are passed to cosmological_redshift_sample."""
    subzbin, subzbin_num_samples = cosmological_redshift_sample(science_case, **kwargs)
    
    results_zsorted = results[results[:, 0].argsort()]
    ind_left_end_in_res = np.searchsorted(results_zsorted[:, 0], [zbin[0] for zbin in subzbin], side='left')
    ind_right_end_in_res = np.searchsorted(results_zsorted[:, 0], [zbin[1] for zbin in subzbin], side='left')
    # subzbin endpoints in terms of indicies of results sorted by redshift
    subzbin_ind_in_res = list(zip(ind_left_end_in_res, ind_right_end_in_res))

    drawn_result_inds = []
    running_count_results_sampled = running_count_completed = num_times_sampled_w_replacement = 0
    for i, subzbin_res in enumerate(subzbin_ind_in_res):
        subzbin_num_sample = subzbin_num_samples[i]
        if subzbin_num_sample == 0:
            continue        
        # because indices of subzbin in results are both to the left, a[i-1] < v <= a[i], the different of indices gives the length
        num_res_in_subzbin = subzbin_res[1] - subzbin_res[0]
        # if there isn't any results in the requested bin, then skip that bin
        if num_res_in_subzbin == 0:
            continue

        running_count_results_sampled += num_res_in_subzbin
        running_count_completed += subzbin_num_sample

        try:
            # sampling without replacement
            drawn_result_inds.append(list(np.random.choice(range(subzbin_res[0], subzbin_res[1]), subzbin_num_sample, replace=False)))
        except ValueError:
            # sampling with replacement
            if print_progress and print_samples_with_replacement: print(f'Sampling with replacement because there are only {num_res_in_subzbin} results and {subzbin_num_sample} were requested in {subzbin[i]}')
            num_times_sampled_w_replacement += 1
            drawn_result_inds.append(list(np.random.choice(range(subzbin_res[0], subzbin_res[1]), subzbin_num_sample, replace=True)))
   
    if print_progress:
        if num_times_sampled_w_replacement > 0: print(f'Insufficient injections, sampling with replacement used {num_times_sampled_w_replacement} times')
        print(f'Number of results: {len(results_zsorted)}, number sampled: {running_count_results_sampled}, equal? {len(results_zsorted) == running_count_results_sampled}')
        print(f'Requested number of samples: {sum(subzbin_num_samples)}, number completed: {running_count_completed}, equal? {sum(subzbin_num_samples) == running_count_completed}')    

    drawn_result_inds = flatten_list(drawn_result_inds)
    return results_zsorted[drawn_result_inds]

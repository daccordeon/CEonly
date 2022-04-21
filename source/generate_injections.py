"""James Gardner, April 2022
generates injection data to later be piped to a results function, saves as .npy
based on old calculate_injections.py"""
from gwbench import injections
from gwbench.basic_relations import f_isco_Msolar, m1_m2_of_Mc_eta
import numpy as np
import glob


def injection_file_name(science_case, num_injs_per_redshift_bin, task_id=None):
    """returns the file name for the injection data, can also add a task_id"""
    file_name = f"injections_SCI-CASE_{science_case}_NUM-INJS-PER-ZBIN_{num_injs_per_redshift_bin}.npy"
    if task_id is not None:
        file_name = file_name.replace(".npy", f"_TASK_{task_id}.npy")
    return file_name


def filter_bool_for_injection(inj, redshifted, coeff_fisco, science_case, debug=False):
    """for a given injection, filter it out if (1) the masses are negative or (2) the fISCOobs is too low, print the reason. This is a little pointless because the network specific filtering still remains to be done for fISCOobs"""
    varied_keys = [
        "Mc",
        "eta",
        "chi1x",
        "chi1y",
        "chi1z",
        "chi2x",
        "chi2y",
        "chi2z",
        "DL",
        "iota",
        "ra",
        "dec",
        "psi",
        "z",
    ]
    varied_params = dict(zip(varied_keys, inj))
    Mc, eta = varied_params["Mc"], varied_params["eta"]
    # m1 and m2 are redshifted if Mc already has been. This error message is never seen, is just here for a legacy sanity check
    m1, m2 = m1_m2_of_Mc_eta(Mc, eta)
    if (m1 <= 0) or (m2 <= 0) or (Mc <= 0) or (eta > 0.25):
        if debug:
            print(
                f"rejecting injection for domain error: m1, m2, Mc, eta = {m1, m2, Mc, eta}, redshifted = {redshifted}"
            )
        return False

    Mtot = Mc / eta**0.6
    # fisco_obs = (6**1.5*PI*(1+z)*Mtot_source)**-1 # with the mass redshifted by (1+z) in the observer frame, missing some number of Msun, c=1, G=1 factors
    if redshifted:
        # 4.4/Mtot*1e3 # Hz # from https://arxiv.org/pdf/2011.05145.pdf
        fisco_obs = f_isco_Msolar(Mtot)
    else:
        fisco_obs = f_isco_Msolar((1.0 + z) * Mtot)
    # chosing fmax in 11 <= coeff_fisco*fisco <= 1024, truncating to boundary values, NB: B&S2022 doesn't include the lower bound which must be included to avoid an IndexError with the automatically truncated fmin from the V+ and aLIGO curves stored in gwbench that start at 10 Hz, this can occur for Mtot > 3520 Msun with massive BBH mergers although those masses are at least an order of magnitude beyond any observed so far
    fmax = coeff_fisco * fisco_obs
    # if BBH, then discard the injection by returning NaNs if fmax < 12 Hz (7 Hz) for aLIGO or V+ (everything else)
    if (science_case == "BBH") and (fmax < 7):
        if debug:
            print(
                f"rejecting BBH injection for high redshifted masses: {fmax, Mtot, redshifted}"
            )
        return False
    return True


def generate_injections(
    num_injs_per_redshift_bin,
    redshift_bins,
    mass_dict,
    spin_dict,
    redshifted,
    coeff_fisco,
    science_case,
    inj_data_path="/fred/oz209/jgardner/CEonlyPony/source/injections/",
):
    """for one science case, generates injection data uniformly and linearly sampled in redshift in each bin and saves as .npy, given the injections parameters and number of injections"""
    # 14 to accommodate [Mc, eta, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, DL, iota, ra, dec, psi, z]
    inj_data = np.empty((len(redshift_bins) * num_injs_per_redshift_bin, 14))
    for i, (zmin, zmax, seed) in enumerate(redshift_bins):
        cosmo_dict = dict(sampler="uniform", zmin=zmin, zmax=zmax)
        injection_params = np.array(
            injections.injections_CBC_params_redshift(
                cosmo_dict,
                mass_dict,
                spin_dict,
                redshifted,
                num_injs=num_injs_per_redshift_bin,
                seed=seed,
            )
        )
        # transposed array to get [[Mc0, eta0, ..., z0], [Mc1, eta1, ..., z1], ...]
        inj_data[
            i * num_injs_per_redshift_bin : (i + 1) * num_injs_per_redshift_bin
        ] = injection_params.transpose()

    # still have to additionally filter for V+ and aLIGO+ later
    inj_data_len_0 = len(inj_data)
    inj_data = inj_data[
        list(
            map(
                lambda inj: filter_bool_for_injection(
                    inj, redshifted, coeff_fisco, science_case
                ),
                inj_data,
            )
        )
    ]
    if len(inj_data) < inj_data_len_0:
        print(
            f"dropped {(inj_data_len_0 - len(inj_data))/inj_data_len_0:.2%} of injections for {science_case}"
        )

    inj_file_name = injection_file_name(science_case, num_injs_per_redshift_bin)
    np.save(inj_data_path + inj_file_name, inj_data)


def inj_params_for_science_case(science_case):
    """return the injection parameters to generate injections for a given science case"""
    if science_case == "BNS":
        # injection settings - source
        mass_dict = dict(dist="gaussian", mean=1.35, sigma=0.15, mmin=1, mmax=2)
        spin_dict = dict(geom="cartesian", dim=1, chi_lo=-0.05, chi_hi=0.05)
        # redshift_bins = ((zmin, zmax, seed), ...) (use same seeds from B&S2022 to replicate results)
        # typo in AppA that starts at 0 rather than 0.02 (in main text)?
        redshift_bins = (
            (0.02, 0.5, 7669),
            (0.5, 1, 3103),
            (1, 2, 4431),
            (2, 4, 5526),
            (4, 10, 7035),
            (10, 50, 2785),
        )
        coeff_fisco = 4  # fmax = 4*fisco for BNS, 8*fisco for BBH
    elif science_case == "BBH":
        # following injection.py and GWTC-2 (AppB.2. Power Law + Peak mass model), to-do: update for GWTC-3?
        # m1 follows power peak, m2 follow uniform in (5 Msun, m1) --> change mmin to 5?
        mass_dict = dict(
            dist="power_peak_uniform",
            mmin=5,  # 4.59 in GWTC-2, but changing to 5 here to get m2 in correct range
            mmax=86.22,
            m1_alpha=2.63,
            q_beta=1.26,
            peak_frac=0.1,
            peak_mean=33.07,  # assuming that peak_mu is peak_mean?
            peak_sigma=5.69,
            delta_m=4.82,
        )
        spin_dict = dict(geom="cartesian", dim=1, chi_lo=-0.75, chi_hi=0.75)
        redshift_bins = (
            (0.02, 0.5, 5485),
            (0.5, 1, 1054),
            (1, 2, 46),
            (2, 4, 5553),
            (4, 10, 5998),
            (10, 50, 4743),
        )
        coeff_fisco = 8
    else:
        raise ValueError("Science case not recognised.")
    return mass_dict, spin_dict, redshift_bins, coeff_fisco


def chop_injections_data_for_processing(
    job_array_size=2048,
    inj_data_path="/fred/oz209/jgardner/CEonlyPony/source/injections/",
):
    """split (chop) the saved injections data into different files for each of the parallel tasks later to run over.
    Given 2048 tasks in the job array (the maximum), split the task as evenly as possible between each task and science case"""
    files = [file for file in glob.glob(inj_data_path + "*") if "TASK" not in file]
    num_science_cases = len(files)
    # to-do: more efficiently allocate the tasks, currently is 1464 in each task and more in the last of each science case to pick up the remainder
    tasks_per_sc = job_array_size // num_science_cases
    for j, file in enumerate(files):
        # absolute path included
        science_case, num_injs_per_redshift_bin = (
            file.replace("_NUM-INJS-PER-ZBIN_", "_SCI-CASE_")
            .replace(".npy", "_SCI-CASE_")
            .split("_SCI-CASE_")[1:3]
        )
        num_injs_per_redshift_bin = int(num_injs_per_redshift_bin)
        inj_data = np.load(file)
        injs_per_task = len(inj_data) // tasks_per_sc
        chop_inds = [
            (i * injs_per_task, (i + 1) * injs_per_task) for i in range(tasks_per_sc)
        ]
        # extend last task to cover any remainder after // above
        chop_inds[-1] = (chop_inds[-1][0], -1)
        for i in range(tasks_per_sc):
            task_id = j * tasks_per_sc + i + 1
            task_file_name = injection_file_name(
                science_case, num_injs_per_redshift_bin, task_id=task_id
            )
            np.save(task_file_name, inj_data[chop_inds[i][0] : chop_inds[i][1]])


if __name__ == "__main__":
    num_injs_per_redshift_bin = 250000
    science_cases = ("BBH", "BNS")
    redshifted = 1

    for science_case in science_cases:
        mass_dict, spin_dict, redshift_bins, coeff_fisco = inj_params_for_science_case(
            science_case
        )
        generate_injections(
            num_injs_per_redshift_bin,
            redshift_bins,
            mass_dict,
            spin_dict,
            redshifted,
            coeff_fisco,
            science_case,
        )

    chop_injections_data_for_processing()

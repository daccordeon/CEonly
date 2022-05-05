#!/usr/bin/env python3
"""Generates astrophysical and observational parameters of injections and saves them as .npy.

Based on old calculate_injections.py.

Usage:
    $ python3 generate_injections.py

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
from typing import List, Set, Dict, Tuple, Optional, Union
from numpy.typing import NDArray
import numpy as np
import glob

from gwbench.basic_relations import f_isco_Msolar, m1_m2_of_Mc_eta, M_of_Mc_eta
from gwbench import injections


def fisco_obs_from_Mc_eta(
    Mc: float, eta: float, redshifted: bool = True, z: Optional[float] = None
) -> float:
    """Returns the frequency of the ISCO in the observer's frame.

    Formula: fisco_obs = (6**1.5*PI*(1+z)*Mtot_source)**-1
    Missing some number of Msun, c=1, G=1 factors.

    Args:
        Mc: Chirp mass.
        eta: Symmetric mass ratio.
        redshifted: Whether the masses are already redshifted inside gwbench.
        z: Redshift, required to redshift the masses.

    Raises:
        ValueError: If redshifted is false but redshift (z) isn't given.
    """
    Mtot = M_of_Mc_eta(Mc, eta)  # Mc / eta**0.6
    if redshifted:
        return f_isco_Msolar(Mtot)
    else:
        if z is None:
            raise ValueError("redshifted is False, provide z value to redshift now")
        return f_isco_Msolar((1.0 + z) * Mtot)


def injection_file_name(
    science_case: str, num_injs_per_redshift_bin: int, task_id: Optional[int] = None
) -> str:
    """Returns the file name for the raw injection data without path.

    Args:
        science_case: Science case.
        num_injs_per_redshift_bin: Number of injections per redshift major bin.
        task_id: Task ID.
    """
    file_name = f"injections_SCI-CASE_{science_case}_INJS-PER-ZBIN_{num_injs_per_redshift_bin}.npy"
    if task_id is not None:
        file_name = file_name.replace(".npy", f"_TASK_{task_id}.npy")
    return file_name


def filter_bool_for_injection(
    inj: NDArray[np.float64],
    redshifted: bool,
    coeff_fisco: int,
    science_case: str,
    debug: bool = False,
    aLIGO_or_Vplus_used: bool = False,
) -> bool:
    """Returns whether to filter out the injection.

    For a given injection, filter it out if (1) the masses are negative or (2) the fISCOobs is too low, and print the reason.
    The network specific filtering is called in calculate_unified_injections.py.

    Args:
        inj: Injection parameters, 14 long.
        redshifted: Whether masses are already redshifted.
        coeff_fisco: Co-efficient of frequency of ISCO.
        science_case: Science case.
        debug: Whether to debug.
        aLIGO_or_Vplus_used: Whether aLIGO or V+ is being analysed.
    """
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
    Mc, eta, z = varied_params["Mc"], varied_params["eta"], varied_params["z"]
    # m1 and m2 are redshifted if Mc already has been. This error message is never seen, is just here for a legacy sanity check
    m1, m2 = m1_m2_of_Mc_eta(Mc, eta)
    if (m1 <= 0) or (m2 <= 0) or (Mc <= 0) or (eta > 0.25):
        if debug:
            print(
                f"rejecting injection for domain error: m1, m2, Mc, eta = {m1, m2, Mc, eta}, redshifted = {redshifted}"
            )
        return False

    fisco_obs = fisco_obs_from_Mc_eta(Mc, eta, redshifted=redshifted, z=z)
    # chosing fmax in 11 <= coeff_fisco*fisco <= 1024, truncating to boundary values, NB: B&S2022 doesn't include the lower bound which must be included to avoid an IndexError with the automatically truncated fmin from the V+ and aLIGO curves stored in gwbench that start at 10 Hz, this can occur for Mtot > 3520 Msun with massive BBH mergers although those masses are at least an order of magnitude beyond any observed so far
    fmax = coeff_fisco * fisco_obs
    # if BBH, then discard the injection by returning NaNs if fmax < 12 Hz (7 Hz) for aLIGO or V+ (everything else)
    if (science_case == "BBH") and (
        (fmax < 7) or (aLIGO_or_Vplus_used and (fmax < 12))
    ):
        if debug:
            print(
                f"rejecting BBH injection for high redshifted masses: {fmax, fisco_obs, redshifted}"
            )
        return False
    return True


def generate_injections(
    num_injs_per_redshift_bin: int,
    redshift_bins: Tuple[Tuple[float, float, float], ...],
    mass_dict: Dict[str, Union[str, float, int]],
    spin_dict: Dict[str, Union[str, float, int]],
    redshifted: bool,
    coeff_fisco: int,
    science_case: str,
    inj_data_path: str = "/fred/oz209/jgardner/CEonlyPony/source/data_raw_injections/",
) -> None:
    """Generates raw injections data sampled uniformly linearly in redshift, saves as .npy.

    Args:
        num_injs_per_redshift_bin: Number of redshift major bins.
        redshift_bins: Redshift bins in the form ((minimum redshift, maximum redshift, random seed, ...).
        mass_dict: Injection settings for mass sampler.
        spin_dict: Injection settings for spin sampler.
        redshifted: Whether gwbench should redshift the masses.
        coeff_fisco: Coefficient of frequency of ISCO.
        science_case: Science case.
        inj_data_path: Path to output directory.
    """
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


def inj_params_for_science_case(
    science_case: str,
) -> Tuple[
    Dict[str, Union[str, float, int]],
    Dict[str, Union[str, float, int]],
    Tuple[Tuple[float, float, float], ...],
    int,
]:
    """Returns the injection parameters to pass to generate_injections for a given science case.

    Args:
        science_case: Science case.

    Raises:
        ValueError: If science case is not recognised.
    """
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
        # following injection.py and GWTC-2 (AppB.2. Power Law + Peak mass model), TODO: update for GWTC-3?
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
    job_array_size: int = 2048,
    inj_data_path: str = "/fred/oz209/jgardner/CEonlyPony/source/data_raw_injections/",
    output_data_path: str = "/fred/oz209/jgardner/CEonlyPony/source/data_raw_injections/task_files/",
) -> None:
    """Splits (chops) the saved injections data into different files for each of the parallel tasks later to run over.

    Given 2048 tasks in the job array (the maximum), split the task as evenly as possible between each task and science case.

    Args:
        job_array_size: Number of tasks in slurm job array.
        inj_data_path: Path to input injections data.
        output_data_path: Path to output (chopped) injections data.
    """
    files = [file for file in glob.glob(inj_data_path + "*.npy")]
    num_science_cases = len(files)
    # TODO: more efficiently allocate the tasks, currently is 1464 in each task and more in the last of each science case to pick up the remainder
    tasks_per_sc = job_array_size // num_science_cases
    for j, file in enumerate(files):
        # absolute path included
        science_case, num_injs_per_redshift_bin_str = (
            file.replace("_INJS-PER-ZBIN_", "_SCI-CASE_")
            .replace(".npy", "_SCI-CASE_")
            .split("_SCI-CASE_")[1:3]
        )
        num_injs_per_redshift_bin = int(num_injs_per_redshift_bin_str)
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
            np.save(
                output_data_path + task_file_name,
                inj_data[chop_inds[i][0] : chop_inds[i][1]],
            )


if __name__ == "__main__":
    # 250k injections to match B&S2022
    num_injs_per_redshift_bin = 250000
    science_cases = ("BBH", "BNS")
    redshifted = True

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

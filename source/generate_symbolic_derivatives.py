#!/usr/bin/env python3
"""Generate symbolic derivatives as lambdified functions for gwbench.

When run as a script: generate all symbolic derivatives for tf2_tidal at all standard locations ahead of benchmarking.
Slurm gets upset when multiple tasks try to create the derivatives if there aren't any there already, so run in series.

Usage:
    $ python3 generate_symbolic_derivatives.py

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
import os

from gwbench import wf_class as wfc
from gwbench import detector_response_derivatives as drd


def generate_symbolic_derivatives(
    wf_model_name: str,
    wf_other_var_dic: Optional[Dict[str, str]],
    deriv_symbs_string: str,
    locs: List[str],
    use_rot: bool,
    output_path: Optional[str] = None,
    print_progress: bool = True,
) -> None:
    """Generate symbolic derivatives, from generate_lambdified_functions.py from gwbench.

    Use network's wf_model_name, wf_other_var_dic, deriv_symbs_string, and use_rot.
    Will print 'Done.' when finished unless all files already exist in which it will print as such.

    Args:
        wf_model_name: Waveform model name.
        wf_other_var_dic: Waveform approximant.
        deriv_symbs_string: Symbols to take derivatives wrt.
        locs: Detector locations.
        use_rot: Whether to account for Earth's rotation.
        output_path: Output file path.
        print_progress: Whether to print progress.
    """
    # # how to print settings as a sanity check
    # print('wf_model_name = \'{}\''.format(wf.wf_model_name))
    # print('wf_other_var_dic = {}'.format(wf.wf_other_var_dic))
    # print('deriv_symbs_string = \'{}\''.format(deriv_symbs_string))
    # print('use_rot = %i'%use_rot)

    # skip if derivatives already exist
    file_names = [
        "par_deriv_WFM_"
        + wf_model_name
        + "_VAR_"
        + deriv_symbs_string.replace(" ", "_")
        + "_DET_"
        + key
        + ".dat"
        for key in locs
    ]
    file_names.append(
        "par_deriv_WFM_"
        + wf_model_name
        + "_VAR_"
        + deriv_symbs_string.replace(" ra", "")
        .replace(" dec", "")
        .replace(" psi", "")
        .replace(" ", "_")
        + "_DET_"
        + "pl_cr"
        + ".dat"
    )
    path = "lambdified_functions/"
    file_names_existing = [
        file_name for file_name in file_names if os.path.isfile(path + file_name)
    ]
    if len(file_names_existing) < len(file_names):
        # if a file doesn't exist, generate them all again
        # TODO: make this more efficient and just generate the missing files, or, do it in parallel
        # waveform
        wf = wfc.Waveform(wf_model_name, wf_other_var_dic)
        # lambidified detector reponses and derivatives
        drd.generate_det_responses_derivs_sym(
            wf,
            deriv_symbs_string,
            locs=locs,
            use_rot=use_rot,
            user_lambdified_functions_path=output_path,
        )
    elif print_progress:
        print("All lambdified derivatives already exist.")


if __name__ == "__main__":
    # tf2_tidal is used as a replacement for numerical BNS simulations until they become well-conditioned
    wf_model_name, wf_other_var_dic = "tf2_tidal", None
    deriv_symbs_string = "Mc eta DL tc phic iota ra dec psi"
    locs = ["H", "L", "V", "K", "I", "ET1", "ET2", "ET3", "C", "N", "S"]
    use_rot = 1

    generate_symbolic_derivatives(
        wf_model_name,
        wf_other_var_dic,
        deriv_symbs_string,
        locs,
        use_rot,
        print_progress=False,
    )

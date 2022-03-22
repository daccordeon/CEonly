#!/usr/bin/env python3
"""James Gardner, March 2022
generate all symbolic derivatives for tf2_tidal at all standard locations ahead of benchmarking
slurm gets upset when multiple tasks try to create the derivatives if there aren't any there already"""

if __name__ == "__main__":
    wf_model_name, wf_other_var_dic = 'tf2_tidal', None
    deriv_symbs_string = 'Mc eta DL tc phic iota ra dec psi'
    locs = ['H', 'L', 'V', 'K', 'I', 'ET1', 'ET2', 'ET3', 'C', 'N', 'S']
    use_rot = 1
        
    generate_symbolic_derivatives(wf_model_name, wf_other_var_dic, deriv_symbs_string, locs, use_rot, print_progress=False):

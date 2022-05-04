"""I/O methods to interact with information stored in the filename of data files.

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
import glob
import numpy as np


def net_label_styler(net_label: str) -> str:
    """Returns the stylised net_label to make CE unified and ET/Voyager less verbose.

    Assumes that ET's declared as 'ET_ET1','ET_ET2','ET_ET3' in network_spec.

    Args:
        net_label: Network label, e.g. 'A+_H..A+_L..V+_V..K+_K..A+_I'.
    """
    return (
        net_label.replace("CE2", "CE")
        .replace("ET_ET1..ET_ET2..ET_ET3", "ET_E")
        .replace("Voyager", "Voy")
    )


def network_spec_styler(network_spec: List[str]) -> str:
    """Returns a styled repr(network_spec) to make CE unified and ET/Voyager less verbose.

    Assumes ET's declared in order.

    Args:
        network_spec: Network specification, e.g. ['A+_H', 'A+_L', 'V+_V', 'K+_K', 'A+_I'].
    """
    return (
        repr(network_spec)
        .replace("CE2", "CE")
        .replace("'ET_ET1', 'ET_ET2', 'ET_ET3'", "'ET_E'")
        .replace("Voyager", "Voy")
    )


def net_label_to_network_spec(
    net_label: str, styled: bool = False
) -> Union[List[str], str]:
    """Returns a converted net_label to network_spec as in gwbench's network.py.

    Args:
        net_label: Network label.
        styled: Whether to style and return repr.
    """
    network_spec = net_label.split("..")
    if styled:
        return network_spec_styler(network_spec)
    else:
        return network_spec


def network_spec_to_net_label(network_spec: List[str], styled: bool = False) -> str:
    """Returns a converted network_spec to net_label as in gwbench's network.py.

    Args:
        network_spec: Network specification, e.g. ['A+_H', 'A+_L', 'V+_V', 'K+_K', 'A+_I'].
        styled: Whether to style.
    """
    net_label = "..".join(network_spec)
    if styled:
        return net_label_styler(net_label)
    else:
        return net_label


def filename_to_netspec_sc_wf_injs(
    filename: str,
) -> Tuple[List[str], str, str, Dict[str, str], int]:
    """Returns the network specification, waveform options, and number of injections from a processed results filename.

    Args:
        filename: A results_*.npy filename without path.
    """
    if "_TASK_" in filename:
        filename = filename[: filename.find("_TASK_")] + ".npy"
    # [1:-1] to cut out 'results' and '.npy'
    net_label, science_case, wf_str, num_injs = (
        filename.replace("_SCI-CASE_", "_NET_")
        .replace("_WF_", "_NET_")
        .replace("_INJS-PER-ZBIN_", "_NET_")
        .replace(".npy", "_NET_")
        .split("_NET_")[1:-1]
    )
    network_spec = net_label_to_network_spec(net_label)
    # TODO: update wf_str to more distinctly separate approximant, currently searching for IMR...
    if "_IMR" in wf_str:
        approximant_index = wf_str.find("_IMR")
        # + 1 to cut out _ before IMR
        wf_model_name, wf_other_var_dic = wf_str[:approximant_index], dict(
            approximant=wf_str[approximant_index + 1 :]
        )
    else:
        wf_model_name, wf_other_var_dic = wf_str, None
    num_injs = int(num_injs)
    return network_spec, science_case, wf_model_name, wf_other_var_dic, num_injs


def file_name_to_multiline_readable(
    file: str, two_rows_only: bool = False, net_only: bool = False
) -> str:
    """Returns a stylised file_name to be human readable across multiple lines, e.g. for titling a plot.

    Args:
        file: Processed results filename with or without path.
    """
    # remove path if present
    if "/" in file:
        file = file.split("/")[-1]
    intermediate = (
        file.replace("results_", "")
        .replace(".npy", "")
        .replace("NET_", "network: ")
        .replace("_SCI-CASE_", "\nscience case: ")
        .replace("..", ", ")
    )
    if net_only:
        return intermediate.split("\n")[0]
    else:
        if two_rows_only:
            return intermediate.replace("_WF_", ", waveform: ").replace(
                "_INJS-PER-ZBIN_", ", injections per bin: "
            )
        else:
            return intermediate.replace("_WF_", "\nwaveform: ").replace(
                "_INJS-PER-ZBIN_", "\ninjections per bin: "
            )


def find_files_given_networks(
    network_spec_list: List[List[str]],
    science_case: str,
    specific_wf: Optional[str] = None,
    print_progress: bool = True,
    data_path: str = "/fred/oz209/jgardner/CEonlyPony/source/data_processed_injections/",
    raise_error_if_no_files_found: bool = True,
) -> List[str]:
    """Returns a list of found files that match networks, science case, and specific wf.

    Chooses those files with the greatest num_injs if multiple exist for a given network; returned list of files do not have data_path prepended.

    Args:
        network_spec_list: Set of networks.
        science_case: Science case.
        specific_wf: Specific waveform to match for.
        print_progress: Whether to print progress.
        data_path: Processed injections data path.
        raise_error_if_no_files_found: Whether to raise an error if no matches found.

    Raises:
        ValueError: If no matches are found.
    """
    # finding file names
    net_labels = [
        net_label_styler(network_spec_to_net_label(network_spec))
        for network_spec in network_spec_list
    ]

    # return files wrt data_path (i.e. exclude the path from glob results)
    file_list = [file.replace(data_path, "") for file in glob.glob(data_path + "*.npy")]
    found_files = np.array([])
    for net_label in net_labels:
        # file_tag = f'NET_{net_label_styler(net.label)}_SCI-CASE_{science_case}_WF_..._INJS-PER-ZBIN_{num_injs}'
        file_tag_partial = f"NET_{net_label}_SCI-CASE_{science_case}"
        # file is file_name
        matches = np.array([file for file in file_list if file_tag_partial in file])
        if len(matches) == 0:
            continue
        # [[f'NET_{net_label_styler(net.label)}_SCI-CASE_{science_case}', f'{wf_model_name}', f'{num_injs}', '.npy'], [...], ...]
        decomp_files = np.array(
            [
                file.replace(".npy", "")
                .replace("_WF_", "_INJS-PER-ZBIN_")
                .split("_INJS-PER-ZBIN_")
                for file in matches
            ]
        )
        # appending is slow but this problem is small
        unique_wf_index_list = []
        for i, wf in enumerate(decomp_files[:, 1]):
            # if specified a wf (with any auxillary), then skip those that don't match
            if specific_wf is not None:
                if wf != specific_wf:
                    continue
            # if multiple files with same tag, then select the one with the greatest number of injections
            num_injs = int(decomp_files[i, 2])
            num_injs_list = [
                int(j) for j in decomp_files[:, 2][decomp_files[:, 1] == wf]
            ]
            # covers the case where len(num_injs_list) = 1, i.e. unique wf
            if num_injs == max(num_injs_list):
                unique_wf_index_list.append(i)
        found_files = np.append(
            found_files, matches[list(set(unique_wf_index_list))]
        )  # could flatten matches here
    found_files = found_files.flatten()
    if len(found_files) == 0:
        message = f"No files found in {data_path} for network {network_spec_list} and science case {science_case}"
        if specific_wf is not None:
            message += f"with waveform {specific_wf}"
        if raise_error_if_no_files_found:
            raise ValueError(message)
        else:
            print(message)
            return list(found_files)
    else:
        if print_progress:
            print(f"Found {len(found_files)} file/s:", *found_files, sep="\n")
        return list(found_files)

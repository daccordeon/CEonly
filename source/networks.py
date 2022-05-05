"""Networks of gravitational-wave detectors of which to benchmark the performance.

Colours for plotting are drawn from B&S2022 and <https://flatuicolors.com/palette/us>.

Usage:
    See run_unified_injections_as_task.py.

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

from useful_functions import flatten_list
from filename_search_and_manipulation import network_spec_styler

# colours and linestyles pulled from B&S2022
BS2022_SIX = dict(
    nets=[
        ["A+_H", "A+_L", "V+_V", "K+_K", "A+_I"],
        ["V+_V", "K+_K", "Voyager-CBO_H", "Voyager-CBO_L", "Voyager-CBO_I"],
        ["A+_H", "A+_L", "K+_K", "A+_I", "ET_ET1", "ET_ET2", "ET_ET3"],
        ["V+_V", "K+_K", "A+_I", "CE2-40-CBO_C"],
        ["K+_K", "A+_I", "ET_ET1", "ET_ET2", "ET_ET3", "CE2-40-CBO_C"],
        ["ET_ET1", "ET_ET2", "ET_ET3", "CE2-40-CBO_C", "CE2-40-CBO_S"],
    ],
    colours=[
        "#8c510aff",
        "#bf812dff",
        "#dfc27dff",
        "#80cdc1ff",
        "#35978fff",
        "#01665eff",
    ],
    label="BS2022-six",
    linestyles=[
        (0, (1, 1)),
        (0, (5, 1, 1, 1, 1, 1)),
        (0, (5, 1, 1, 1)),
        (0, (5, 1)),
        (0, (1, 2)),
        "-",
    ],
)

# --- CE only ---
# - One in the US, removed this ill-conditioned network from the set since injection rejection is now unified.
# CE_C = dict(nets=[["CE2-40-CBO_C"]], colours=["#2d3436"], label="CE_C")
# - One in the US, One in Australia
CE_CS = dict(
    nets=[
        ["CE2-40-CBO_C", "CE2-40-CBO_S"],
        ["CE2-40-CBO_C", "CE2-20-CBO_S"],
        ["CE2-40-CBO_C", "CE2-20-PMO_S"],
    ],
    colours=["#0984e3", "#74b9ff", "#ff7675"],
    label="CE_CS",
)
# - Two in the US --> done in B&S2022 with VKI+
CE_CN = dict(
    nets=[
        ["CE2-40-CBO_C", "CE2-40-CBO_N"],
        ["CE2-40-CBO_C", "CE2-20-CBO_N"],
        ["CE2-40-CBO_C", "CE2-20-PMO_N"],
    ],
    colours=["#0984e3", "#74b9ff", "#ff7675"],
    label="CE_CN",
)
# - Three (two in US, one in Aus)
CE_CNS = dict(
    nets=[
        ["CE2-40-CBO_C", "CE2-20-CBO_N", "CE2-40-CBO_S"],
        ["CE2-40-CBO_C", "CE2-20-CBO_N", "CE2-20-CBO_S"],
        ["CE2-40-CBO_C", "CE2-20-CBO_N", "CE2-20-PMO_S"],
        ["CE2-40-CBO_C", "CE2-20-PMO_N", "CE2-40-CBO_S"],
        ["CE2-40-CBO_C", "CE2-20-PMO_N", "CE2-20-CBO_S"],
        ["CE2-40-CBO_C", "CE2-20-PMO_N", "CE2-20-PMO_S"],
    ],
    colours=["#0984e3", "#74b9ff", "#ff7675", "#6c5ce7", "#a29bfe", "#fd79a8"],
    label="CE_CNS",
)

# --- CE_S with non-CE others ---
# - CE_S + A+ network
CE_S_AND_2G = dict(
    nets=[
        ["A+_H", "A+_L", "V+_V", "K+_K", "A+_I", "CE2-40-CBO_S"],
        ["A+_H", "A+_L", "V+_V", "K+_K", "A+_I", "CE2-20-CBO_S"],
        ["A+_H", "A+_L", "V+_V", "K+_K", "A+_I", "CE2-20-PMO_S"],
    ],
    colours=["#00cec9", "#81ecec", "#fab1a0"],
    label="CE_S_and_2G",
)
# - CE_S + Voyager notework
CE_S_AND_VOYAGER = dict(
    nets=[
        [
            "V+_V",
            "K+_K",
            "Voyager-CBO_H",
            "Voyager-CBO_L",
            "Voyager-CBO_I",
            "CE2-40-CBO_S",
        ],
        [
            "V+_V",
            "K+_K",
            "Voyager-CBO_H",
            "Voyager-CBO_L",
            "Voyager-CBO_I",
            "CE2-20-CBO_S",
        ],
        [
            "V+_V",
            "K+_K",
            "Voyager-CBO_H",
            "Voyager-CBO_L",
            "Voyager-CBO_I",
            "CE2-20-PMO_S",
        ],
    ],
    colours=["#00cec9", "#81ecec", "#fab1a0"],
    label="CE_S_and_Voy",
)
# - CE_S + ET, not yet examined, compare to CE_CN_AND_ET
CE_S_AND_ET = dict(
    nets=[
        ["ET_ET1", "ET_ET2", "ET_ET3", "CE2-40-CBO_S"],
        ["ET_ET1", "ET_ET2", "ET_ET3", "CE2-20-CBO_S"],
        ["ET_ET1", "ET_ET2", "ET_ET3", "CE2-20-PMO_S"],
    ],
    colours=["#00cec9", "#81ecec", "#fab1a0"],
    label="CE_S_and_ET",
)

# --- CE_CN with ET ---
CE_CN_AND_ET = dict(
    nets=[
        ["ET_ET1", "ET_ET2", "ET_ET3", "CE2-40-CBO_C", "CE2-40-CBO_N"],
        ["ET_ET1", "ET_ET2", "ET_ET3", "CE2-40-CBO_C", "CE2-20-CBO_N"],
        ["ET_ET1", "ET_ET2", "ET_ET3", "CE2-40-CBO_C", "CE2-20-PMO_N"],
    ],
    colours=["#00cec9", "#81ecec", "#fab1a0"],
    label="CE_CN_and_ET",
)

# --- H0 science case: A+_S to get sky localisation for the ~10 BNS this decade ---
NEMOLF_AND_2G = dict(
    nets=[
        ["A+_S", "A+_H", "A+_L", "V+_V", "K+_K", "A+_I"],
        ["A+_S", "A+_H", "A+_L", "V+_V", "K+_K"],
        ["A+_S", "A+_H", "A+_L"],
    ],
    colours=["#00b894", "#55efc4", "#e17055"],
    label="NEMO-LF_and_2G",
)

# list of 10 network sets (dicts) to run unified tasks over
NET_DICT_LIST: List[
    Dict[str, Union[List[List[str]], List[str], str, List[tuple, str]]]
] = [
    BS2022_SIX,
    #     CE_C,
    CE_CS,
    CE_CN,
    CE_CNS,
    CE_S_AND_2G,
    CE_S_AND_VOYAGER,
    CE_S_AND_ET,
    CE_CN_AND_ET,
    NEMOLF_AND_2G,
]
# list of network spec's within the broad set
NET_LIST = flatten_list([net_dict["nets"] for net_dict in NET_DICT_LIST])
# lookup table: given network_spec return colour, will override if same network_spec used twice
DICT_NETSPEC_TO_COLOUR = dict()
for net_dict in NET_DICT_LIST:
    for network_spec in net_dict["nets"]:
        # using network_spec's recovered from net_label in filename after applying net_label_styler
        DICT_NETSPEC_TO_COLOUR[network_spec_styler(network_spec)] = net_dict["colours"][
            net_dict["nets"].index(network_spec)
        ]

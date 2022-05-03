"""Extention (subclass) of gwbench's network class with input/output quality-of-life features.

Usage:
    >> net = NetworkExtended(
            network_spec, 
            science_case,
            wf_model_name,
            wf_other_var_dic,
            num_injs
        )

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
from useful_functions import insert_at_pattern
from filename_search_and_manipulation import net_label_styler

from gwbench import network
import os


def set_file_tags(obj : Any) -> None:
    """Sets the file_tag and human_file_tag of an instance of NetworkExtended or InjectionResults.

    TODO: Update obj's type hinting to Union[Type[NetworkExtended], Type[InjectionResults]].

    Args:
        obj: Object instance with attributes to generate file tags, e.g. NetworkExtended or InjectionResults. The instance must have the attributes: label, science_case, wf_model_name, wf_other_var_dic, num_injs.
    """
    obj.file_tag = f"NET_{net_label_styler(obj.label)}_SCI-CASE_{obj.science_case}_WF_{obj.wf_model_name}_INJS-PER-ZBIN_{obj.num_injs}"
    obj.human_file_tag = f'network: {net_label_styler(obj.label).replace("..", ", ")}\nscience case: {obj.science_case}\nwaveform: {obj.wf_model_name}\nnumber of injections per bin: {obj.num_injs}'
    if obj.wf_other_var_dic is not None:
        obj.file_tag = insert_at_pattern(
            obj.file_tag, f'_{obj.wf_other_var_dic["approximant"]}', "_INJS"
        )
        obj.human_file_tag = insert_at_pattern(
            obj.human_file_tag,
            f' with {obj.wf_other_var_dic["approximant"]}',
            "\nnumber of injections",
        )


class NetworkExtended(network.Network):
    """Subclass of gwbench's network class that adds functionality, e.g. filename generation and styling.

    Since it is a subclass, instances of it can be passed as you would the parent network.Network class (e.g. as net).

    Attributes:
        network_spec (List[str]): Network specification, e.g. ['A+_H', 'A+_L', 'V+_V', 'K+_K', 'A+_I'].
        science_case (str): Science case, e.g. 'BNS'.
        wf_model_name (str): Waveform model name.
        wf_other_var_dic (Optional[Dict[str, str]]): Waveform approximant dictionary.
        num_injs (int): Number of injections per major redshift bin.
        tecs (List[str]): Unique detector technologies in network.
        file_tag (str): File tag for input/output.
        human_file_tag (str): Human-readable file tag.
        file_name (Optional[str]): File name for processed results .npy data file without path.
        data_path (str): Path to the data file.
        file_name_with_path (str): File name for processed results .npy data file with path.
        results_file_exists (bool): Whether processed results .npy data file exists.        
        
        And all other attributes of the network.Network class as initialised by a network_spec.
    """

    def __init__(
        self,
        network_spec : List[str],
        science_case : str,
        wf_model_name : str,
        wf_other_var_dic : Optional[Dict[str, str]],
        num_injs : int,
        file_name : Optional[str]=None,
        data_path : str="/fred/oz209/jgardner/CEonlyPony/source/processed_injections_data/",
    ) -> None:
        """Initialises NetworkExtended with all attributes. 

        Args:
            network_spec: Network specification, e.g. ['A+_H', 'A+_L', 'V+_V', 'K+_K', 'A+_I'].
            science_case: Science case, e.g. 'BNS'.
            wf_model_name: Waveform model name.
            wf_other_var_dic: Waveform approximant dictionary.
            num_injs: Number of injections per major redshift bin.
            file_name: File name for processed results .npy data file without path. If blank, then it is generated from the created file_tag. If containing "SLURM_TASK_", then the task_id is also added to the file_name.
            data_path: Path to the data file. 
        """
        super().__init__(network_spec)
        self.network_spec = network_spec
        self.science_case = science_case
        self.wf_model_name = wf_model_name
        self.wf_other_var_dic = wf_other_var_dic
        self.num_injs = num_injs
        # detector technologies, necessary to know because gwbench has different frequency ranges for the PSDs
        self.tecs = list(set([detector.tec for detector in self.detectors]))

        # input/output standard: file name and plot label
        set_file_tags(self)
        if file_name is None:
            self.file_name = f"results_{self.file_tag}.npy"
        elif "SLURM_TASK_" in file_name:
            self.file_name = (
                f'results_{self.file_tag}_TASK_{file_name.split("SLURM_TASK_")[1]}.npy'
            )
        else:
            self.file_name = file_name
        self.data_path = data_path
        self.file_name_with_path = self.data_path + self.file_name
        self.results_file_exists = os.path.isfile(self.file_name_with_path)

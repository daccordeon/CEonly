"""James Gardner, April 2022
python classes for an extention (subclass) of gwbench's network class"""
from useful_functions import insert_at_pattern
from filename_search_and_manipulation import net_label_styler

from gwbench import network
import os

def set_file_tags(obj):
    """sets file_tag and human_file_tag given an instance (obj) of NetworkExtended or InjectionResults with attributes: label, science_case, wf_model_name, wf_other_var_dic, num_injs"""
    obj.file_tag = f'NET_{net_label_styler(obj.label)}_SCI-CASE_{obj.science_case}_WF_{obj.wf_model_name}_INJS-PER-ZBIN_{obj.num_injs}'
    obj.human_file_tag = f'network: {net_label_styler(obj.label).replace("..", ", ")}\nscience case: {obj.science_case}\nwaveform: {obj.wf_model_name}\nnumber of injections per bin: {obj.num_injs}'    
    if obj.wf_other_var_dic is not None:
        obj.file_tag = insert_at_pattern(obj.file_tag, f'_{obj.wf_other_var_dic["approximant"]}', '_INJS')
        obj.human_file_tag = insert_at_pattern(obj.human_file_tag, f' with {obj.wf_other_var_dic["approximant"]}', '\nnumber of injections')

class NetworkExtended(network.Network):
    """subclass of gwbench's network class that adds functionality, e.g. filename generation and styling. since it is a subclass instances of it can be passed as you would the parent network class (e.g. as net)"""
    def __init__(self, network_spec, science_case, wf_model_name, wf_other_var_dic, num_injs, file_name=None, data_path='/fred/oz209/jgardner/CEonlyPony/source/data_redshift_snr_errs_sky-area/'):
        super().__init__(network_spec)
        self.network_spec = network_spec
        self.science_case = science_case
        self.wf_model_name = wf_model_name
        self.wf_other_var_dic = wf_other_var_dic
        self.num_injs = num_injs
        # detector technologies, necessary to know because gwbench has different frequency ranges for the PSDs
        self.tecs = [detector.tec for detector in self.detectors]
        
        # input/output standard: file name and plot label 
        set_file_tags(self)
        if file_name is None:
            self.file_name = f'results_{self.file_tag}.npy'
        elif 'SLURM_TASK_' in file_name:
            self.file_name = f'results_{self.file_tag}_TASK_{file_name.split("SLURM_TASK_")[1]}.npy'
        else:
            self.file_name = file_name
        self.data_path = data_path
        self.file_name_with_path = self.data_path + self.file_name
        self.results_file_exists = os.path.isfile(self.file_name_with_path)

"""James Gardner, March 2022"""
from numpy import pi as PI

# low and high SNR thresholds
SNR_THRESHOLD_LO = 10 # for detection
SNR_THRESHOLD_HI = 100 # for high fidelity

# merger rates from Section IV-A in https://arxiv.org/abs/2111.03634v2.pdf
GWTC3_MERGER_RATE_BNS = 105.5
GWTC3_MERGER_RATE_BBH = 23.9
# as quoted in B&S2022
# to-do: add functionality to switch to old rates to compare directly to B&S2022
# GWTC2_MERGER_RATE_BNS = 320
# GWTC2_MERGER_RATE_BBH = 24

# network sets
# colours pulled from B&S2022 using Inkscape
BS2022_STANDARD_6 = dict(nets=[
    ['A+_H', 'A+_L', 'V+_V', 'K+_K', 'A+_I'],
    ['V+_V', 'K+_K', 'Voyager-CBO_H', 'Voyager-CBO_L', 'Voyager-CBO_I'],
    ['A+_H', 'A+_L', 'K+_K', 'A+_I', 'ET_ET1'],
    ['V+_V', 'K+_K', 'A+_I', 'CE1-40-CBO_C'],
    ['K+_K', 'A+_I', 'ET_ET1', 'CE1-40-CBO_C'],
    ['ET_ET1', 'CE1-40-CBO_C', 'CE1-40-CBO_S']],
    colours=['#8c510aff','#bf812dff','#dfc27dff','#80cdc1ff','#35978fff','#01665eff'])
# https://flatuicolors.com/palette/us
# updated networks after meeting on 2022-03-11, now also studying 20-CBO and only CE2
CE_ONLY_C_and_S = dict(nets=[
    ['CE2-40-CBO_C', 'CE2-40-CBO_S'],
    ['CE2-40-CBO_C', 'CE2-20-CBO_S'],
    ['CE2-40-CBO_C', 'CE2-20-PMO_S']],
    colours=['#0984e3', '#74b9ff', '#ff7675'])
CE_ONLY_C_and_N = dict(nets=[
    ['CE2-40-CBO_C', 'CE2-40-CBO_N'],
    ['CE2-40-CBO_C', 'CE2-20-CBO_N'],
    ['CE2-40-CBO_C', 'CE2-20-PMO_N']],
    colours=['#6c5ce7', '#a29bfe', '#fd79a8'])
CE_S_W_ET = dict(nets=[
    ['ET_ET1', 'CE2-40-CBO_S'],
    ['ET_ET1', 'CE2-20-CBO_S'],
    ['ET_ET1', 'CE2-20-PMO_S']],
    colours=['#00b894', '#55efc4', '#fab1a0'])
# list of net sets to replicate B&S2022 and study CE only science case 
NET_DICT_LIST = [BS2022_STANDARD_6, CE_ONLY_C_and_S, CE_ONLY_C_and_N, CE_S_W_ET]
# colour look-up table given net_spec
DICT_KEY_NETSPEC_VAL_COLOUR = dict()
for dict_nets_colours in NET_DICT_LIST:
    for net_spec in dict_nets_colours['nets']:
        DICT_KEY_NETSPEC_VAL_COLOUR[repr(net_spec)] = dict_nets_colours['colours'][dict_nets_colours['nets'].index(net_spec)]

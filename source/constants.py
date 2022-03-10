"""James Gardner, March 2022"""
from numpy import pi as PI

SNR_THRESHOLD_LO = 10 # for detection
SNR_THRESHOLD_HI = 100 # for high fidelity

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
CE_ONLY = dict(nets=[
    ['CE1-40-CBO_C', 'CE1-20-PMO_S'],
    ['CE1-40-CBO_C', 'CE1-40-CBO_S'],
    ['CE2-40-CBO_C', 'CE2-20-PMO_S'],
    ['CE2-40-CBO_C', 'CE2-40-CBO_S']],
    colours=['#a29bfe','#ff7675','#6c5ce7','#d63031'])
CE_S_W_ET = dict(nets=[
    ['CE1-20-PMO_S', 'ET_ET1'],
    ['CE1-40-CBO_S', 'ET_ET1'],
    ['CE2-20-PMO_S', 'ET_ET1'],
    ['CE2-40-CBO_S', 'ET_ET1']],
    colours=['#74b9ff','#fd79a8','#0984e3','#e84393'])
# colour look-up table given net_spec
DICT_KEY_NETSPEC_VAL_COLOUR = dict()
for dict_nets_colours in BS2022_STANDARD_6, CE_ONLY, CE_S_W_ET:
    for net_spec in dict_nets_colours['nets']:
        DICT_KEY_NETSPEC_VAL_COLOUR[repr(net_spec)] = dict_nets_colours['colours'][dict_nets_colours['nets'].index(net_spec)]

# detection rate (DR) hack coefficients to correct scale manually
HACK_DR_COEFF_BNS = 38
HACK_DR_COEFF_BBH = 24
def hack_coeff_default(science_case):
    if science_case == 'BNS':
        return HACK_DR_COEFF_BNS
    elif science_case == 'BBH':
        return HACK_DR_COEFF_BBH
    else:
        raise ValueError('Science case not recognised.')

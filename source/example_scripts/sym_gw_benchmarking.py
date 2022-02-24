# Copyright (C) 2020  Ssohrab Borhanian
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import argparse
import os
import sys

import numpy as np
from lal import GreenwichMeanSiderealTime

import gwbench.basic_relations as brs
import gwbench.injections as injections
import gwbench.network as network

np.set_printoptions(linewidth=200)

parser = argparse.ArgumentParser(description = '')
parser.add_argument("nodenum", help = "node number",type=int)
nodenum = parser.parse_args().nodenum

############################################################################
### User Choices
############################################################################

# user's choice: waveform to use
wf_model_name = 'tf2'
#wf_model_name = 'tf2_tidal'

if   wf_model_name == 'tf2':       wf_other_var_dic = None
elif wf_model_name == 'tf2_tidal': wf_other_var_dic = None

# user's choice: with respect to which parameters to take derivatives for the Fisher analysis
if 'tidal' in wf_model_name or 'bns' in wf_model_name: deriv_symbs_string = 'Mc eta chi1z chi2z DL tc phic iota lam_t ra dec psi'
else: deriv_symbs_string = 'Mc eta chi1z chi2z DL tc phic iota ra dec psi'

# user's choice: convert derivatives to cos or log for specific variables
conv_cos = ('dec','iota')
conv_log = ('Mc','DL','lam_t')

# if symbolic derivatives, take from generate_lambdified_functions.py
# if numeric  derivatives, user's decision
use_rot = 1
# 1 for True, 0 for False
# calculate SNRs, error matrices, and errors only for the network
only_net = 1

# user's choice to generate injection parameters
if 'tidal' in wf_model_name or 'bns' in wf_model_name:
    mmin      = 0.8
    mmax      = 3
    chi_lo    = -0.05
    chi_hi    = 0.05
else:
    mmin      = 5
    mmax      = 100
    chi_lo    = -0.75
    chi_hi    = 0.75

cosmo_dict = {'zmin':0, 'zmax':0.2, 'sampler':'uniform_comoving_volume_inversion'}
mass_dict  = {'dist':'uniform', 'mmin':mmin, 'mmax':mmax}
spin_dict  = {'dim':1, 'geom':'cartesian', 'chi_lo':chi_lo, 'chi_hi':chi_hi}

redshifted = 1
num_injs  = 100
seed  = 29378
file_path = None

injections_data = injections.injections_CBC_params_redshift(cosmo_dict,mass_dict,spin_dict,redshifted,num_injs,seed,file_path)

############################################################################
### Print the choices
############################################################################

#-----output run settings-----
print('---run settings---')
print('wf_model_name = ',wf_model_name)
print('wf_other_var_dic = ',wf_other_var_dic)
print('deriv_symbs_string = ',deriv_symbs_string)
print()
print('conv_cos = ',conv_cos)
print('conv_log = ',conv_log)
print()
print('use_rot = ',use_rot)
print('only_net = ',only_net)
print()
print('cosmo_dict = ',cosmo_dict)
print('mass_dict = ',mass_dict)
print('spin_dict = ',spin_dict)
print('seed = ',seed)
print('num_injs = ',num_injs)
print('file_path',file_path)
print('---run settings done---')
print()

############################################################################
### injection parameters
############################################################################

inj_params                    = dict()

inj_params['Mc']              = injections_data[0][nodenum]
inj_params['eta']             = injections_data[1][nodenum]
inj_params['chi1x']           = injections_data[2][nodenum]
inj_params['chi1y']           = injections_data[3][nodenum]
inj_params['chi1z']           = injections_data[4][nodenum]
inj_params['chi2x']           = injections_data[5][nodenum]
inj_params['chi2y']           = injections_data[6][nodenum]
inj_params['chi2z']           = injections_data[7][nodenum]
inj_params['DL']              = injections_data[8][nodenum]
inj_params['tc']              = 0.
inj_params['phic']            = 0.
inj_params['iota']            = injections_data[9][nodenum]

if 'tidal' in wf_model_name or 'bns' in wf_model_name:
    inj_params['lam_t']       = 600.
    inj_params['delta_lam_t'] = 0.

inj_params['ra']              = injections_data[10][nodenum]
inj_params['dec']             = injections_data[11][nodenum]
inj_params['psi']             = injections_data[12][nodenum]
inj_params['gmst0']           = GreenwichMeanSiderealTime(1247227950.)
inj_params['z']               = injections_data[13][nodenum]

print('injections parameter: ', inj_params)
print()

############################################################################
### User choices
############################################################################

network_spec = ['CE2-40-CBO_C','CE2-40-CBO_N','CE2-40-CBO_S']
print('network spec: ', network_spec)
print()

f_lo = 1.
f_hi = brs.f_isco_Msolar(brs.M_of_Mc_eta(inj_params['Mc'],inj_params['eta']))
df = 2.**-4
f = np.arange(f_lo,f_hi+df,df)

print('f_lo:', f_lo, '   f_hi:', f_hi, '   df:', df)
print()

############################################################################
### Symbolic GW Benchmarking
############################################################################

net = network.Network(network_spec)
net.set_wf_vars(wf_model_name,wf_other_var_dic)
net.set_net_vars(f,inj_params,deriv_symbs_string,conv_cos,conv_log,use_rot)
net.setup_ant_pat_lpf_psds()
#-----symbolic derivatives-------------------------------------
net.load_det_responses_derivs_sym()
net.calc_det_responses_derivs_sym()
#-------------------------------------------------------------
net.calc_snrs(only_net=only_net)
net.calc_errors(only_net=only_net)
net.calc_sky_area_90(only_net=only_net)

############################################################################
### Print results
############################################################################

net.print_detectors()
net.print_network()

print('Fisher analysis done.')

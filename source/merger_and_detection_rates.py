"""James Gardner, April 2022"""
from constants import PI, GWTC3_MERGER_RATE_BNS, GWTC3_MERGER_RATE_BBH

from scipy.integrate import quad
from astropy.cosmology import Planck18
from gwbench import injections

def differential_comoving_volume(z):
    """$\frac{\text{d}V}{\text{d}z}(z)$ in B&S2022; 4*pi to convert from Mpc^3 sr^-1 (sr is steradian) to Mpc^3"""
    return  4.*PI*Planck18.differential_comoving_volume(z).value

def merger_rate_bns(z, normalisation=GWTC3_MERGER_RATE_BNS):
    """$R(z)$ in B&S2022; normalisation of merger rate density $\dot{n}(z)$ in the source frame to GWTC3_MERGER_RATE_BNS in https://arxiv.org/pdf/2111.03606v2.pdf.
1e-9 converts Gpc^-3 to Mpc^-3 to match Planck18, in Fig 2 of Ngetal2021: the ndot_F rate is in Gpc^-3 yr^-1, injections.py cites v1 of an arXiv .pdf"""
    return normalisation/injections.bns_md_merger_rate(0)*1e-9*injections.bns_md_merger_rate(z)*differential_comoving_volume(z)

def merger_rate_bbh(z, normalisation=GWTC3_MERGER_RATE_BBH):
    """$R(z)$ in B&S2022; normalisation of merger rate density $\dot{n}(z)$ in the source frame to GWTC3_MERGER_RATE_BBH in https://arxiv.org/pdf/2111.03606v2.pdf.
1e-9 converts Gpc^-3 to Mpc^-3 to match Planck18, in Fig 2 of Ngetal2021: the ndot_F rate is in Gpc^-3 yr^-1, injections.py cites v1 of an arXiv .pdf"""
    return normalisation/injections.mdbn_merger_rate(0)*1e-9*injections.mdbn_merger_rate(z)*differential_comoving_volume(z)

def merger_rate_in_obs_frame(merger_rate, z, **kwargs):
    """1+z factor of time dilation of merger rate in observer frame z away. kwargs, e.g. normalisation, are passed to merger_rate"""
    return merger_rate(z, **kwargs)/(1+z)

def detection_rate(merger_rate, detection_efficiency, z0, snr_threshold, **kwargs):
    """$D_R(z, \rho_\ast)$ in B&S2022. quad returns (value, error). kwargs, e.g. normalisation, are passed to merger_rate"""
    return quad(lambda z : detection_efficiency(z, snr_threshold)*merger_rate_in_obs_frame(merger_rate, z, **kwargs), 0, z0)[0]

def detection_rate_limit(merger_rate, z0, **kwargs):
    """$D_R(z, \rho_\ast)|_{\varepsilon=1}$ in B&S2022;i.e. "merger rate" in Fig 2, not R(z) but int R(z)/(1+z), i.e. if perfect efficiency"""
    return detection_rate(merger_rate, lambda _, __ : 1, z0, None, **kwargs)

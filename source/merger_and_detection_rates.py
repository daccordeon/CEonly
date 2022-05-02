"""Short one-sentence description.

Long description.

Usage:
    Describe the typical usage.

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
from constants import (
    PI,
    GWTC3_MERGER_RATE_BNS,
    GWTC3_MERGER_RATE_BBH,
    GWTC2_MERGER_RATE_BNS,
    GWTC2_MERGER_RATE_BBH,
)

from scipy.integrate import quad
from astropy.cosmology import Planck18
from gwbench import injections


def merger_rate_normalisations_from_gwtc_norm_tag(norm_tag="GWTC3"):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """returns merger rates (BNS, BBH) normalisation from survey tag"""
    if norm_tag == "GWTC3":
        return (GWTC3_MERGER_RATE_BNS, GWTC3_MERGER_RATE_BBH)
    elif norm_tag == "GWTC2":
        return (GWTC2_MERGER_RATE_BNS, GWTC2_MERGER_RATE_BBH)
    else:
        raise ValueError("Normalisation not recognised.")


def differential_comoving_volume(z):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """$\frac{\text{d}V}{\text{d}z}(z)$ in B&S2022; 4*pi to convert from Mpc^3 sr^-1 (sr is steradian) to Mpc^3"""
    return 4.0 * PI * Planck18.differential_comoving_volume(z).value


def merger_rate_bns(z, normalisation=GWTC3_MERGER_RATE_BNS):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """$R(z)$ in B&S2022; normalisation of merger rate density $\dot{n}(z)$ in the source frame to GWTC3_MERGER_RATE_BNS in https://arxiv.org/pdf/2111.03606v2.pdf.
    1e-9 converts Gpc^-3 to Mpc^-3 to match Planck18, in Fig 2 of Ngetal2021: the ndot_F rate is in Gpc^-3 yr^-1, injections.py cites v1 of an arXiv .pdf"""
    return (
        normalisation
        / injections.bns_md_merger_rate(0)
        * 1e-9
        * injections.bns_md_merger_rate(z)
        * differential_comoving_volume(z)
    )


def merger_rate_bbh(z, normalisation=GWTC3_MERGER_RATE_BBH):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """$R(z)$ in B&S2022; normalisation of merger rate density $\dot{n}(z)$ in the source frame to GWTC3_MERGER_RATE_BBH in https://arxiv.org/pdf/2111.03606v2.pdf.
    1e-9 converts Gpc^-3 to Mpc^-3 to match Planck18, in Fig 2 of Ngetal2021: the ndot_F rate is in Gpc^-3 yr^-1, injections.py cites v1 of an arXiv .pdf"""
    return (
        normalisation
        / injections.mdbn_merger_rate(0)
        * 1e-9
        * injections.mdbn_merger_rate(z)
        * differential_comoving_volume(z)
    )


def merger_rate_in_obs_frame(merger_rate, z, **kwargs):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """1+z factor of time dilation of merger rate in observer frame z away. kwargs, e.g. normalisation, are passed to merger_rate"""
    return merger_rate(z, **kwargs) / (1 + z)


def detection_rate(merger_rate, detection_efficiency, z0, snr_threshold, **kwargs):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """$D_R(z, \rho_\ast)$ in B&S2022. quad returns (value, error). kwargs, e.g. normalisation, are passed to merger_rate"""
    return quad(
        lambda z: detection_efficiency(z, snr_threshold)
        * merger_rate_in_obs_frame(merger_rate, z, **kwargs),
        0,
        z0,
    )[0]


def detection_rate_limit(merger_rate, z0, **kwargs):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """$D_R(z, \rho_\ast)|_{\varepsilon=1}$ in B&S2022;i.e. "merger rate" in Fig 2, not R(z) but int R(z)/(1+z), i.e. if perfect efficiency"""
    return detection_rate(merger_rate, lambda _, __: 1, z0, None, **kwargs)

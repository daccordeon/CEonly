"""Cosmological merger rates of gravitational-wave sources and detection rates.

Usage:
    See results_class.py and plot_collated_detection_rate.py for how the detection rates and calculated and the plots are created.

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

from typing import List, Set, Dict, Tuple, Optional, Union, Callable, Any
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


def merger_rate_normalisations_from_gwtc_norm_tag(
    norm_tag: str = "GWTC3",
) -> Tuple[float, float]:
    """Returns merger rate normalisations from a survey tag.

    Args:
        norm_tag: Tag of the survey to normalise merger rates to, e.g. "GWTC3" or "GWTC2".

    Raises:
        ValueError: If the survey tag isn't recognised.
    """
    if norm_tag == "GWTC3":
        return (GWTC3_MERGER_RATE_BNS, GWTC3_MERGER_RATE_BBH)
    elif norm_tag == "GWTC2":
        return (GWTC2_MERGER_RATE_BNS, GWTC2_MERGER_RATE_BBH)
    else:
        raise ValueError("Normalisation not recognised.")


def differential_comoving_volume(z: float) -> float:
    """Returns the differential comoving volume at a given redshift.

    Uses the Planck18 cosmology.
    Follows the formula: $\frac{\text{d}V}{\text{d}z}(z)$ in B&S2022; 4*pi to convert from Mpc^3 sr^-1 (sr is steradian) to Mpc^3.

    Args:
        z: Redshift.
    """
    return 4.0 * PI * Planck18.differential_comoving_volume(z).value


def merger_rate_bns(z: float, normalisation: float = GWTC3_MERGER_RATE_BNS) -> float:
    """Returns the binary neutron-star merger rate at a given redshift.

    Formula: $R(z)$ in B&S2022; normalisation of merger rate density $\dot{n}(z)$ in the source frame to GWTC3_MERGER_RATE_BNS in https://arxiv.org/pdf/2111.03606v2.pdf.
    1e-9 converts Gpc^-3 to Mpc^-3 to match Planck18, in Fig 2 of Ngetal2021: the ndot_F rate is in Gpc^-3 yr^-1, injections.py cites v1 of an arXiv .pdf

    Args:
        z: Redshift.
        normalisation: Merger rate normalisation from merger_rate_normalisations_from_gwtc_norm_tag.
    """
    return (
        normalisation
        / injections.bns_md_merger_rate(0)
        * 1e-9
        * injections.bns_md_merger_rate(z)
        * differential_comoving_volume(z)
    )


def merger_rate_bbh(z: float, normalisation: float = GWTC3_MERGER_RATE_BBH) -> float:
    """Returns the binary black-hole merger rate at a given redshift.

    Formula: $R(z)$ in B&S2022; normalisation of merger rate density $\dot{n}(z)$ in the source frame to GWTC3_MERGER_RATE_BBH in https://arxiv.org/pdf/2111.03606v2.pdf.
    1e-9 converts Gpc^-3 to Mpc^-3 to match Planck18, in Fig 2 of Ngetal2021: the ndot_F rate is in Gpc^-3 yr^-1, injections.py cites v1 of an arXiv .pdf

    Args:
        z: Redshift.
        normalisation: Merger rate normalisation from merger_rate_normalisations_from_gwtc_norm_tag.
    """
    return (
        normalisation
        / injections.mdbn_merger_rate(0)
        * 1e-9
        * injections.mdbn_merger_rate(z)
        * differential_comoving_volume(z)
    )


def merger_rate_in_obs_frame(
    merger_rate: Callable[..., float], z: float, **kwargs: Any
) -> float:
    """Returns the merger rate at the given redshift as time dilated in the observer's frame.

    TODO: update type hints when typing Protocol for kwargs in Callable is updated/released, similarly throughout.

    Args:
        merger_rate: Merger rate function of the form merger_rate(z, **kwargs), e.g. merger_rate_bns.
        z: Redshift, give a time dilation factor of 1 + z.
        **kwargs: Options passed to merger_rate.
    """
    return merger_rate(z, **kwargs) / (1 + z)


def detection_rate(
    merger_rate: Callable[..., float],
    detection_efficiency: Callable[[float, float], float],
    z0: float,
    snr_threshold: float,
    **kwargs: Any
) -> float:
    """Returns the detection rate of a source type given its merger rate and detector efficiency.

    Formula: $D_R(z, \rho_\ast)$ in B&S2022.

    Args:
        merger_rate: Merger rate function of the form merger_rate(z, **kwargs), e.g. merger_rate_bns.
        detection_efficiency: Detection efficiency function of the form detection_efficiency(z, snr_threshold).
        z0: Maximum redshift to integrate detection rate from zero out to.
        snr_threshold: Signal-to-noise ratio detection threshold.
        **kwargs: Options passed to merger_rate.
    """
    # quad returns (value, error), [0] to get value
    return quad(
        lambda z: detection_efficiency(z, snr_threshold)
        * merger_rate_in_obs_frame(merger_rate, z, **kwargs),
        0,
        z0,
    )[0]


def detection_rate_limit(
    merger_rate: Callable[..., float], z0: float, **kwargs: Any
) -> float:
    """Returns the maximum possible detection rate, i.e. the total number of sources, of a source type given its merger rate.

    Formula: $D_R(z, \rho_\ast)|_{\varepsilon=1}$ in B&S2022;i.e. "merger rate" in Fig 2, not R(z) but int R(z)/(1+z), i.e. if perfect efficiency.

    Args:
        merger_rate: Merger rate function of the form merger_rate(z, **kwargs), e.g. merger_rate_bns.
        z0: Maximum redshift to integrate detection rate from zero out to.
        **kwargs: Options passed to merger_rate.
    """
    return detection_rate(merger_rate, lambda _, __: 1, z0, None, **kwargs)

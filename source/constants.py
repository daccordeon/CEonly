"""Code base--wide constants, e.g. signal-to-noise detection threshold and cosmological merger rates.

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
from numpy import pi as PI

# low (for detection) and high (for high fidelity) SNR thresholds
# middle threshold to better show contour between lo and hi, TODO: relate this to literature?
SNR_THRESHOLD_LO = 10
SNR_THRESHOLD_MID = 30
SNR_THRESHOLD_HI = 100

# merger rates from Section IV-A in https://arxiv.org/abs/2111.03634v2.pdf
GWTC3_MERGER_RATE_BNS, GWTC3_MERGER_RATE_BBH = 105.5, 23.9
# from B&S2022, TODO: double check these rates
GWTC2_MERGER_RATE_BNS, GWTC2_MERGER_RATE_BBH = 320, 23

# sky areas, https://en.wikipedia.org/wiki/Square_degree
TOTAL_SKY_AREA_SQR_DEG = 129600 / PI
MOON_SKY_AREA_SQR_DEG = PI * (0.5 / 2) ** 2  # varies with distance from Earth
# from Rana, check decadal predictions for ZTF
EM_FOLLOWUP_SKY_AREA_SQR_DEG = 10

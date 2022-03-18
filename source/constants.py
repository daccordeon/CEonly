"""James Gardner, March 2022"""
from numpy import pi as PI

# low (for detection) and high (for high fidelity) SNR thresholds
SNR_THRESHOLD_LO = 10
SNR_THRESHOLD_HI = 100

# merger rates from Section IV-A in https://arxiv.org/abs/2111.03634v2.pdf
# to-do: add functionality to switch to GWTC-2 rates to compare directly to B&S2022
GWTC3_MERGER_RATE_BNS = 105.5
GWTC3_MERGER_RATE_BBH = 23.9

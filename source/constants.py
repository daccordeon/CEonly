"""James Gardner, March 2022"""
from numpy import pi as PI

# low (for detection) and high (for high fidelity) SNR thresholds
# middle threshold to better show contour between lo and hi, to-do: relate this to literature?
SNR_THRESHOLD_LO = 10
SNR_THRESHOLD_MID = 30
SNR_THRESHOLD_HI = 100

# merger rates from Section IV-A in https://arxiv.org/abs/2111.03634v2.pdf
# to-do: add functionality to switch to GWTC-2 rates to compare directly to B&S2022
GWTC3_MERGER_RATE_BNS, GWTC3_MERGER_RATE_BBH = 105.5, 23.9
# from B&S2022, to-do: double check these rates
GWTC2_MERGER_RATE_BNS, GWTC2_MERGER_RATE_BBH = 320, 23

# sky areas, https://en.wikipedia.org/wiki/Square_degree
TOTAL_SKY_AREA_SQR_DEG = 129600 / PI
MOON_SKY_AREA_SQR_DEG = PI * (0.5 / 2) ** 2  # varies with distance from Earth
# from Rana, check decadal predictions for ZTF
EM_FOLLOWUP_SKY_AREA_SQR_DEG = 10

import math
import numpy as np
import warnings
from scipy.interpolate import InterpolatedUnivariateSpline



def cut_offs(freq, response, mode='band_pass', cut_off_value=-3):
    idx_lim = [0, len(freq) - 1]
    freq_lim = [freq[0], freq[-1]]
    cnt_crosses = 0
    cut_off_value = -cut_off_value

    for k in range(len(freq)-1):
        if (response[k] + cut_off_value > 0) & (response[k + 1] + cut_off_value < 0):
            cnt_crosses = cnt_crosses + 1
            idx_lim[0] = k
            m1 = (response[k + 1] - response[k]) / (freq[k + 1] - freq[k])
            q1 = response[k + 1] - m1 * freq[k + 1]
            freq_lim[0] = (-cut_off_value - q1) / m1

        if (response[k] + 3 < 0) & (response[k + 1] + 3 > 0):
            cnt_crosses = cnt_crosses + 1
            idx_lim[1] = k
            m2 = (response[k + 1] - response[k]) / (freq[k + 1] - freq[k])
            q2 = response[k + 1] - m2 * (freq[k + 1])
            freq_lim[1] = (-cut_off_value - q2) / m2

    if mode == 'band_pass' or mode == 'band-stop':
        if cnt_crosses != 2:
            warnings.warn(f"*** WARNING! Number of frequency values at -{cut_off_value} dB is not 2 ***\n")
        else:
            return freq_lim

    if mode == 'low_pass':
        if cnt_crosses > 1:
            warnings.warn(f"*** WARNING! More than one frequency value at -{cut_off_value} dB is found ***\n")
        return freq_lim[0]
    
    if mode == 'high_pass':
        if cnt_crosses > 1:
            warnings.warn(f"*** WARNING! More than one frequency value at -{cut_off_value} dB is found ***\n")
        return freq_lim[0]


def cut_offs2(freq, response, mode='band_pass', cut_off_value=-3):
    response2 = response-cut_off_value
    int_f = InterpolatedUnivariateSpline(freq, response2)
    crosses = int_f.roots()

    cnt_crosses = len(crosses)
    if mode == 'band_pass' or mode == 'band-stop':
        if cnt_crosses != 2:
            warnings.warn(f"*** WARNING! Number of frequency values at -{cut_off_value} dB is not 2 ***\n")
        else:
            return crosses

    if mode == 'low_pass':
        if cnt_crosses > 1:
            warnings.warn(f"*** WARNING! More than one frequency value at -{cut_off_value} dB is found ***\n")
        return crosses[0]

    if mode == 'high_pass':
        if cnt_crosses > 1:
            warnings.warn(f"*** WARNING! More than one frequency value at -{cut_off_value} dB is found ***\n")
        return crosses[0]



def bandwidth(freq, response, cut_off_value=-3, method=1):
    if method == 1:
        lims = cut_offs(freq, response, cut_off_value=cut_off_value)
    else:
        lims = cut_offs2(freq, response, cut_off_value=cut_off_value)
    if lims is None:
        return freq[1]-freq[0]
    elif len(lims) != 2:
        return freq[1]-freq[0]
    else:
        return abs(lims[1]-lims[0])


def central_frequency(freq, response, cut_off_value=-3, method=1):
    if method == 1:
        lims = cut_offs(freq, response, cut_off_value=cut_off_value)
    else:
        lims = cut_offs2(freq, response, cut_off_value=cut_off_value)
    if lims is None:
        return freq[0]
    elif len(lims) != 2:
        return freq[0]
    else:
        return (lims[1]+lims[0]) / 2


def min_band_attenuation(freq, response, f_low, f_high):
    pass
    return


def max_band_ripple(freq, response, f_low, f_high):
    pass
    return




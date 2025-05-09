# Author: J Asplet

# Functions to plot radiation patterns

# To-writes:
# 1. Rad pattern P (1-D)
# 2. Rad pattern SV/SH (1-D)
# 3. 2-D radiation patterns (for P/SV/SH)
# 4. Beach ball
# 5. 4-panel combination

import numpy as np
import matplotlib.pyplot as plt
# import obspy
from patterns import calc_rad_patterns, radiation_patterns_2d


def plot_radiation(ax, radiation_coeff, azimuths):
    '''
    Plits inout radiation pattern as a function of azimuth,
    '''

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')

    ml_assume = np.mean(np.abs(radiation_coeff))
    radiation_coeff = radiation_coeff / ml_assume
    ax.plot(azimuths[radiation_coeff > 0],
            radiation_coeff[radiation_coeff > 0],
            color='tab:blue')
    ax.plot(azimuths[radiation_coeff <= 0],
            radiation_coeff[radiation_coeff <= 0],
            color='tab:red')
    ax.plot(azimuths, np.ones(azimuths.shape), 'k--')

    return ax


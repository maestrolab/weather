import pickle
import numpy as np
from weather import process_noise
from weather.boom import boom_runner, process_data
from weather.plotting.noise import contour


def fidell_CTL(noise, growth=0.47, CTL=81.3, A_star=None):
    noise = np.asarray(noise)
    # for A star
    if CTL is None:
        CTL = -10*np.log10(-np.log(0.5))/growth + A_star/growth
        m = (10.**(CTL/10.))**growth
        A = - m*np.log(0.5)
    else:
        A_star = growth*CTL + 10*np.log10(-np.log(0.5))
        A = 10**(A_star/10)
    m = (10.**(noise/10.))**growth
    P = np.exp(-A/m)
    return P*100


day = '18'
month = '06'
year = '2018'
hour = '12'

filename = "../../data/noise/" + year + "_" + month + "_" + day + "_" + hour
data = process_noise(filename, transformation=fidell_CTL)
contour(data, levels=np.arange(0, 110, 10), label='% Hightly Annoyed')

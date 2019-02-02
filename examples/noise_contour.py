import pickle
import numpy as np
from weather.boom import boom_runner, process_data
from weather.plotting.noise import contour


def exterior_annoyance(PL):
    PL = np.asarray(PL)
    annoyance = np.zeros(PL.shape)
    for i in range(len(PL)):
        PL_0 = 72.412
        slope = 5.7410605
        if PL[i] <= PL_0:
            annoyance[i] = 0
        elif PL[i] < 89.866:
            annoyance[i] = slope*(PL[i]-PL_0)
        else:
            annoyance[i] = 100
    return(annoyance)


def interior_annoyance(PL):
    PL = np.asarray(PL)
    annoyance = np.zeros(PL.shape)
    for i in range(len(PL)):
        PL_0 = 44.475
        slope = 4.536
        if PL[i] <= PL_0:
            annoyance[i] = 0
        elif PL[i] < 55.519:
            annoyance[i] = slope*(PL[i]-PL_0)
        else:
            annoyance[i] = 100
    return(annoyance)


day = '18'
month = '06'
year = '2018'
hour = '12'

filename = "../data/noise/" + year + "_" + month + "_" + day + "_" + hour

contour(filename, transformation=exterior_annoyance,
        levels=np.arange(0, 110, 10), label='% More Annoyed')

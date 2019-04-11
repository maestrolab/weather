import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

from weather.scraper.flight_conditions import properties, Airframe


def expected(data, airFrame):
    alpha, V, lift_to_drag = data

    pdf = airFrame.pdf.score_samples(np.vstack([alpha.ravel(), V.ravel()]).T)
    pdf = np.exp(pdf.reshape(lift_to_drag.shape))
    expected_value = 0
    numerator_list = []
    denominator_list = []
    for i in range(len(lift_to_drag)):
        numerator = simps(lift_to_drag[i]*pdf[i], alpha[i])
        denominator = simps(pdf[i], alpha[i])
        numerator_list.append(numerator)
        denominator_list.append(denominator)
    numerator = simps(numerator_list, V[:, 0])
    denominator = simps(denominator_list, V[:, 0])
    expected_value = numerator/denominator
    return(expected_value)


mat = scipy.io.loadmat('../../data/msc/morphing.mat')

owl = mat['Owl'][0][0]
naca0012 = mat['NACA0012'][0][0]
naca4415 = mat['NACA4415'][0][0]

C172_props = properties({'Cl_alpha': 5.143, 'Cl_0': 0.31,
                         'planform': 16.1651, 'density': 0.770488088,
                         'mass_min': 618., 'mass_max': 919.,
                         'incidence': 0.})
C172 = Airframe(airframe='C172', timestamp=1549036800,
                filepath='../../data/flight_plan/v_aoa_pickles/icao24s_',
                properties=C172_props)
C172.retrieve_data()
C172.train_pdf(1000)

alpha, V, lift_to_drag = owl

plt.show()
print('owl', expected(owl, C172))
print('NACA0012', expected(naca0012, C172))
print('NACA4415', expected(naca4415, C172))

import scipy.io
from scipy.integrate import simps
import numpy as np
from weather.scraper.path import Airframe
import matplotlib.pyplot as plt


def expected(data, airFrame):
    alpha, V, lift_to_drag = data

    pdf = airFrame.generate_pdf(np.vstack([alpha.ravel(), V.ravel()]))
    pdf = pdf.reshape(lift_to_drag.shape)
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
    print(denominator)
    expected_value = numerator/denominator
    return(expected_value)


mat = scipy.io.loadmat('../../data/msc/morphing.mat')

owl = mat['Owl'][0][0]
naca0012 = mat['NACA0012'][0][0]
naca4415 = mat['NACA4415'][0][0]


typecodeList = ['B737', 'B747', 'B757', 'B767', 'B777', 'B787',
                'A310', 'A318', 'A319', 'A320', 'A321',
                'A330', 'A340', 'A350', 'A380', 'C172', 'C180',
                'C182']

airFrame = Airframe(typecode=typecodeList[15], timestamp=1549036800)
airFrame.retrieve_data()
alpha, V, lift_to_drag = owl
airFrame.plot_pdf()
plt.scatter(alpha, V)
# airFrame.plot_scatter()
plt.show()
print('owl', expected(owl, airFrame))
print('NACA0012', expected(naca0012, airFrame))
print('NACA4415', expected(naca4415, airFrame))

import scipy.io
from scipy.integrate import simps
import numpy as np
from weather.scraper.path import Airframe


def expected(data, airFrame):
    alpha, V, lift_to_drag = data

    pdf = airFrame.generate_pdf(np.vstack([alpha.ravel(), V.ravel()]))
    pdf = pdf.reshape(lift_to_drag.shape)
    expected_value = 0
    integrated_list = []
    for i in range(len(lift_to_drag)):
        temp = simps(lift_to_drag[i]*pdf[i], alpha[i])
        integrated_list.append(temp)
    expected_value = simps(integrated_list, V[:, 0])
    return(expected_value/sum(pdf.ravel()))


mat = scipy.io.loadmat('../../data/msc/morphing.mat')

owl = mat['Owl'][0][0]
naca0012 = mat['NACA0012'][0][0]
naca4415 = mat['NACA4415'][0][0]


typecodeList = ['B737', 'B747', 'B757', 'B767', 'B777', 'B787',
                'A310', 'A318', 'A319', 'A320', 'A321',
                'A330', 'A340', 'A350', 'A380', 'C172', 'C180',
                'C182']

airFrame = Airframe(typecode=typecodeList[15], timestamp=1549036800)

print('owl', expected(owl, airFrame))
print('NACA0012', expected(naca0012, airFrame))
print('NACA4415', expected(naca4415, airFrame))

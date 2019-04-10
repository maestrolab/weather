import pickle
from opensky_api import OpenSkyApi
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from weather.filehandling import output_reader
import time


class properties(object):
    def __init__(self, inputs):
        for key in inputs:
            setattr(self, key, inputs[key])


class Airframe(object):
    """ Object to store data from OpenSky and estimate probabilities based on
        the retrieved data

        inputs:
         - airframe: typecode for airframe (e.g. B737)
         - timestamp: Time when OpenSkyApi pulls data from
         - filepath: path to where information is retrieved
       """

    def __init__(self, timestamp=1549729462, airframe='B737',
                 filepath='../../data/flight_plan/v_aoa_pickles/icao24s_'):
        self.airframe = airframe
        self.timestamp = timestamp
        self.filepath = filepath

    def calculate_angle_of_attack(self, velocity, weight, density=0.770488088,
                                  planform=16.1651, Cl_alpha=0.0776):

        angleOfAttack = np.array([])

        # Data for Cessna 172 (cruise altitude and cruise speed)
        W = weight * 9.81  # [kg*m*s^2]
        rho = 0.770488088  # [kg/m^3]
        # u = 62.57 # [m/s]: cruise for Cessna 172
        S = 16.1651  # [m^2]
        Cl_alpha = 0.0776
        Cl_o = 0.33
        incidenceAngle = 1  # [deg]

        cos_alpha = np.linspace(np.cos(-5/180*np.pi), np.cos(15/180*np.pi),
                                len(velocity))
        cos_alpha = np.mean(cos_alpha)
        constant = 2*W/rho/(velocity)**2/S

        Cl = (1/cos_alpha) * constant

        # Defining Cl as average of possible Cls for Cessna 172 assuming -5 < AoA < 15 [deg]
        # Cl = np.mean(Cl)

        initial_AoA = (Cl-Cl_o)/Cl_alpha + incidenceAngle  # [deg]

        # Implementing fixed-point iteration:
        err = 0.001
        j = 0
        for initial_AoA_value in initial_AoA:
            next_AoA = 0
            i = 0
            while (abs(initial_AoA_value-next_AoA) > err).any() and \
                    i < 10000:
                # if statement included to set the initial_AoA_value to the
                #   previously calculated AoA
                if (next_AoA != np.array([0])).any():
                    initial_AoA_value = next_AoA

                # Calculation of part of the calculation for AoA (depends on
                #   velocity which is an array, therefore, need index)
                constant = 2*W/rho/(velocity[j])**2/S

                # Calculation of coefficient of lift and AoA
                Cl = (1/np.cos(initial_AoA_value/180*np.pi)) * constant
                next_AoA = (Cl-Cl_o)/Cl_alpha + incidenceAngle
                i += 1

            j += 1

            # Adding converged AoA value to list of AoA's for later use
            angleOfAttack = np.append(angleOfAttack, next_AoA)

        # self.angleOfAttack = initial_AoA
        self.aoa = angleOfAttack

    def retrieve_data(self):
        data = pickle.load(open(self.filepath + str(self.airframe) + '_' +
                                str(self.timestamp) + '.p', 'rb'))

        self.velocity = np.array(data['velocity'], dtype='float')
        self.climb_rate = np.array(data['vertrate'], dtype='float')


if __name__ == "__main__":

    C172_props = properties({'Cl_alpha': 0.0776, 'Cl_0': 0.33,
                             'planform': 16.1651, 'density': 0.770488088,
                             'Mass_min': 618, 'Mass_max': 919})

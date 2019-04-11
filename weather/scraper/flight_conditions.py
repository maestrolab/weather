import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fixed_point
from sklearn.neighbors.kde import KernelDensity

from opensky_api import OpenSkyApi
from weather.filehandling import output_reader


class properties(object):
    def __init__(self, inputs):
        for key in inputs:
            setattr(self, key, inputs[key])

        self.weight_min = self.mass_min * 9.81
        self.weight_max = self.mass_max * 9.81


class Airframe(object):
    """ Object to store data from OpenSky and estimate probabilities based on
        the retrieved data

        inputs:
         - airframe: typecode for airframe (e.g. B737)
         - timestamp: Time when OpenSkyApi pulls data from
         - filepath: path to where information is retrieved
         - properties: object properties with all airframe and other relevant
                      properties
       """

    def __init__(self, timestamp=1549729462, airframe='B737',
                 filepath='../../data/flight_plan/v_aoa_pickles/icao24s_',
                 properties=properties):
        self.airframe = airframe
        self.timestamp = timestamp
        self.filepath = filepath
        self.properties = properties

    def calculate_aoa(self, weight, velocity):
        # If input is not list/array convert it to one
        try:
            len(velocity)  # dummy test
        except(TypeError):
            velocity = [velocity]
        try:
            len(weight)  # dummy test
        except(TypeError):
            weight = [weight]

        # Implementing fixed-point iteration
        aoa = []
        for i in range(len(velocity)):
            aoa_i = fixed_point(self._aoa, 2., args=(weight[i], velocity[i]))
            aoa.append(float(aoa_i))
        return(aoa)

    def _aoa(self, aoa, weight, velocity):
        p = self.properties
        cos = np.cos(np.radians(aoa + p.incidence))
        aoa = (2*weight/(p.density*velocity**2*p.planform)*cos -
               p.Cl_0)/p.Cl_alpha
        return np.degrees(aoa)

    def plot_pdf(self, parameters=None,
                 xgrid=np.linspace(-5, 35, 1000),
                 ygrid=np.linspace(20, 75, 1000)):
        """Generate contour plot visualizing PDF of velocity vs. angle of
        attack."""

        X, Y = np.meshgrid(xgrid, ygrid)
        Z = np.exp(self.pdf.score_samples(np.array([X.ravel(), Y.ravel()]).T))
        Z = np.reshape(Z, X.shape)

        plt.figure()
        plt.contourf(X, Y, Z)
        plt.xlabel('Approximated Angle of Attack [degrees]')
        plt.ylabel('Velocity [m/s]')
        # plt.xlim(-2, 35)
        plt.colorbar()
        plt.show()

    def retrieve_data(self):
        data = pickle.load(open(self.filepath + str(self.airframe) + '_' +
                                str(self.timestamp) + '.p', 'rb'))

        self.velocity = np.array(data['velocity'], dtype='float')
        self.velocity = self.velocity.reshape(len(self.velocity), 1)
        self.climb_rate = np.array(data['vertrate'], dtype='float')
        self.climb_rate = self.climb_rate.reshape(len(self.climb_rate), 1)
        self.pdf_velocity = KernelDensity(kernel='gaussian').fit(self.velocity)
        self.pdf_climb = KernelDensity(kernel='gaussian').fit(self.climb_rate)

    def train_pdf(self, population=100000):
        """Description"""
        p = self.properties
        self.database = []
        for i in range(population):
            weight = np.random.uniform(p.weight_min, p.weight_max)
            velocity = self.pdf_velocity.sample()[0]
            aoa = self.calculate_aoa(weight, velocity)
            self.database.append([aoa[0], velocity[0]])

        self.database = np.array(self.database)
        self.pdf = KernelDensity(kernel='gaussian').fit(self.database)

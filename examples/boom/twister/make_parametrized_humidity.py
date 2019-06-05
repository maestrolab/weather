################################################################################
# HermiteSpline was written by Pedro leal
# Original file: spline_class.py (copied from there)
"""Tools for creating parametric surface descriptions and meshes."""
import numpy as np
class HermiteSpline:
    """  """

    def __init__(self, p0=0., p1=0, m0=1., m1=-.2, a=0.5, b=1.0):
        self._p0 = p0
        self._p1 = p1
        self._m0 = m0
        self._m1 = m1
        self._a = a
        self._b = b

    def __call__(self, parameter):
        p0 = self._p0
        p1 = self._p1
        m0 = self._m0
        m1 = self._m1

        dx = self._b - self._a

        output = np.zeros(parameter.shape)
        indexes = np.where((parameter >= self._a) & (parameter <= self._b))
        t = np.linspace(0, 1, len(indexes[0]))
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        output[indexes] = h00*p0 + h10*dx*m0 + h01*p1 + h11*dx*m1
        return output
################################################################################
def package_data(data1, data2=None, method='unpack'):
    '''package_data packs or unpacks data in the form [[data1, data2]]'''
    if method == 'pack':
        packed_data = [[data1[i],data2[i]] for i in range(len(data1))]
        return packed_data
    elif method == 'unpack':
        unpacked_data_1 = [d[0] for d in data1]
        unpacked_data_2 = [d[1] for d in data1]
        return unpacked_data_1, unpacked_data_2

def convert_to_celcius(temperature_F):
    if type(temperature_F) == list:
        temperature_F = np.array(temperature_F)
    temperature_C = (5/9)*(temperature_F-32)
    return temperature_C

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# from 'some_directory' import HermiteSpline

class ParametrizeHumidity:

    def __init__(self, altitudes, relative_humidities, temperatures, cruise_altitude):
        self.alts = altitudes
        self.rhs = relative_humidities
        self.temps = temperatures
        self._calculate_vapor_pressures()
        self.cruise_altitude = cruise_altitude

    def _calculate_vapor_pressures(self):
        saturation_vapor_pressures = []
        for t in self.temps:
            if t>=0:
                sat_vps = 0.61121*np.exp((18.678-(t/234.5))*(t/(257.14+t)))
            else:
                sat_vps = 0.61115*np.exp((23.036-(t/333.7))*(t/(279.82+t)))
            saturation_vapor_pressures.append(sat_vps)

        actual_vapor_pressures = [self.rhs[i]/100*
                                  saturation_vapor_pressures[i] for i in
                                  range(len(self.rhs))]
        self.saturation_vps = saturation_vapor_pressures
        self.vps = actual_vapor_pressures

    def hermite_spline(self, p0=0, p1=0, m0=None, m1=1, a=0, b=8000, n_points=100):
        if m0 == None:
            m0 = (self.vps[1]-self.vps[0])/(self.alts[1]-self.alts[0])

        self.spline = HermiteSpline(p0,p1,m0,m1,a,b)
        self.p_alts = np.linspace(a,b,n_points)
        self.p_vps = self.spline(parameter=self.p_alts)

        self.p_alts = np.append(self.p_alts, self.cruise_altitude)
        self.p_vps = np.append(self.p_vps, 0.0)

    def calculate_humidity_profile(self):
        self.p_saturation_vps = np.interp(self.p_alts, self.alts,
                                          self.saturation_vps)
        self.p_rhs = [100*self.p_vps[i]/self.p_saturation_vps[i] for i in range(
                                                               len(self.p_vps))]

    def RMSE(self):
        # Values for RMSE are found (finding values of parametrized profile at
        #   each altitude of the original profile)
        predicted_values = list(self.spline(parameter=np.array(self.alts)))
        self.rmse = mean_squared_error(self.vps, predicted_values)

    def plot(self, profile_type='vapor_pressures'):
        fig = plt.figure()
        if profile_type == 'vapor_pressures':
            plt.plot(self.vps, self.alts, label='Original')
            plt.plot(self.p_vps, self.p_alts, label='Parametrized')
            plt.xlabel('Vapor Pressure [kPa]')
            # plt.xlim()
        elif profile_type == 'relative_humidities':
            plt.plot(self.rhs, self.alts, label='Original')
            plt.plot(self.p_rhs, self.p_alts, label='Parametrized')
            plt.xlabel('Relative Humidities [%]')
            # plt.xlim()
        plt.ylabel('Altitude [m]')
        # plt.ylim(0,16000)
        plt.show()

#  WILL IMPORT ABOVE THIS LINE ONCE WORKING
################################################################################
import pickle
from weather.boom import boom_runner, prepare_weather_sBoom
from weather.scraper.twister import process_data

day = '18'
month = '06'
year = '2018'
hour = '12_'
lat = 32
lon = -100
alt_ft = 45000.
alt = alt_ft * 0.3048

data, altitudes = process_data(day, month, year, hour, alt,
                               directory='../../../data/weather/twister/',
                               convert_celcius_to_fahrenheit=True)

key = '%i, %i' % (lat, lon)
weather_data = data[key]
index = list(data.keys()).index(key)
height_to_ground = altitudes[index] / 0.3048  # In feet

# Parametrization process
profile_altitudes, relative_humidities = package_data(weather_data['humidity'])
profile_altitudes, temperatures = package_data(weather_data['temperature'])
temperatures = convert_to_celcius(temperatures)
cruise_altitude = altitudes[index]

p_profile = ParametrizeHumidity(profile_altitudes,relative_humidities,temperatures,
                                cruise_altitude)

m0_range = np.arange(-0.0001, -0.001, -0.0001)
for m0 in m0_range:
    p_profile.hermite_spline(p0=max(p_profile.vps),p1=0,m0=m0,m1=-1/5500,a=0,b=8500,n_points=5)
    p_profile.calculate_humidity_profile()
    p_humidity_profile = package_data(p_profile.p_alts, p_profile.p_rhs, method='pack')
    p_profile.plot()
    p_profile.RMSE()
    print(p_profile.rmse)
    p_profile.plot(profile_type='relative_humidities')

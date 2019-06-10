import numpy as np
import pickle

################################################################################
class SplineBumpHumidity:
    """  """

    def __init__(self, a=0, b=1, x=0, p0=0, p1=0, y=1, m0=0, m1=0, m=0):
        self._p0 = [p0, y]
        self._p1 = [y, p1]
        self._m0 = [m0, m]
        self._m1 = [m, m1]
        self._a = [a, x]
        self._b = [x, b]

    def __call__(self, parameter):
        index = int(np.where(parameter == self._a[1])[0])
        for i in range(2):
            spline = HermiteSpline(self._p0[i], self._p1[i],
                                   self._m0[i], self._m1[i],
                                   self._a[i], self._b[i])
            if i == 0:
                output = spline(parameter[:index+1])
            else:
                output = np.append(output[:-1], spline(parameter[index:]))
        return output

"""Tools for creating parametric surface descriptions and meshes."""
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
def convert_to_celcius(temperature_F):
    if type(temperature_F) == list:
        temperature_F = np.array(temperature_F)
    temperature_C = (5/9)*(temperature_F-32)
    return temperature_C

def package_data(data1, data2=None, method='unpack'):
    '''package_data packs or unpacks data in the form [[data1, data2]]'''
    if method == 'pack':
        packed_data = [[data1[i],data2[i]] for i in range(len(data1))]
        return packed_data
    elif method == 'unpack':
        unpacked_data_1 = [d[0] for d in data1]
        unpacked_data_2 = [d[1] for d in data1]
        return unpacked_data_1, unpacked_data_2

def prepare_standard_profiles(standard_profiles_path=
'./../../../data/weather/standard_profiles/standard_profiles.p'):
       standard_profiles = pickle.load(open(standard_profiles_path,'rb'))

       # Interpolate temperature values at altitudes in relative humidity profile
       temperature_altitudes, temperatures = package_data(standard_profiles['temperature'])
       rh_altitudes, rhs = package_data(standard_profiles['relative humidity'])
       pressure_altitudes, pressures = package_data(standard_profiles['pressure'])

       temperatures = np.interp(rh_altitudes, temperature_altitudes, temperatures)
       pressures = np.interp(rh_altitudes, pressure_altitudes, pressures)

       standard_profiles['temperature'] = package_data(rh_altitudes,
                                                        temperatures, method='pack')
       standard_profiles['pressure'] = package_data(rh_altitudes, pressures,
                                                    method='pack')

       return standard_profiles

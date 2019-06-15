import numpy as np
import pickle
import matplotlib.pyplot as plt

################################################################################
class SplineBumpHumidity:
    """  """

    def __init__(self, a=0, b=1, x=0, p0=0, p1=0, y=1, m0=0, m1=0, m=0):
        self._p0 = [p0, y]
        self._p1 = [y, p1]
        self._m0 = [m0, m]
        self._m1 = [-m, m1]
        self._a = [a, x]
        self._b = [x, b]

    def __call__(self, parameter):
        output = np.zeros(len(parameter))
        for i in range(2):
            spline = HermiteSpline(self._p0[i], self._p1[i],
                                   self._m0[i], self._m1[i],
                                   self._a[i], self._b[i])

            # Output is computed if x is in data set or not
            if not (parameter == self._a[1]).any():
                if i == 0:
                    output = np.append(output, 0)
                    index = np.where(parameter < self._a[1])
                    i1 = index[0][-1]+1
                else:
                    output[i1] = 0
                parameter = np.insert(parameter, i1, self._a[1])
                output += spline(parameter)
                parameter = np.delete(parameter, i1)
                if i == 1:
                    output = np.delete(output, i1)
            else:
                output += spline(parameter)
                i2 = np.where(parameter == self._a[1])
                output[i2[0]] = self._p0[1]

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
def calculate_vapor_pressures(humidities, temperatures, pressures):
    '''calculate_vapor_pressures calculates the saturation_vapor_pressures
    and the actual_vapor_pressures given a temperature and humidity
    profile (equations from Arden Buck (1981))
    '''
    saturation_vps = []
    for i in range(len(temperatures)):
        if temperatures[i]>=0:
            f = 1.0007+(3.46e-6*pressures[i])
            sat_vps = f*0.61121*np.exp(17.502*temperatures[i]/(240.97+temperatures[i]))
        elif temperatures[i]>-50:
            f = 1.0003+(4.18e-6*pressures[i])
            sat_vps = f*0.61115*np.exp(22.452*temperatures[i]/(272.55+temperatures[i]))
        else:
            f = 1.0003+(4.18e-6*pressures[i])
            sat_vps = f*0.61115*np.exp(22.542*temperatures[i]/(273.48+temperatures[i]))
        saturation_vps.append(sat_vps)
    actual_vps = [humidities[i]/100*saturation_vps[i] for i in range(len(humidities))]
    return actual_vps, saturation_vps

def combine_profiles(data, day, month, year, hour, alt, profile_type='humidity',
                     latitude={'min':13,'max':59},
                     longitude={'min':-144,'max':-52},
                     path='../../data/weather/twister/day_combined/'):
    '''combine_profiles combines all profile data from a given day
        Inputs:
         - day, month, year, hour, alt: specifies day to pull data from
           (reference weather/data/weather/twister/ directory for available
           values)
         - profile_type: data to be combined (temperature, humidity, pressure,
                         wind_x, wind_y)
         - latitude, longitude: range of latitudes and longitudes to pull data
                                from
    '''

    profiles = []
    for lat in range(latitude['min'], latitude['max']):
        for lon in range(longitude['min'], longitude['max']):
            key = '%i, %i' % (lat, lon)
            alts, profile_vals = package_data(data[key][profile_type])
            profiles.append(profile_vals)

    if hour[-1] == '_':
        hour = hour[:-1]

    path += '%s_%s_%s_%s_all_%s' % (year, month, day, hour, profile_type)
    all_profiles_in_day = open(path + '.p','wb')
    pickle.dump(profiles, all_profiles_in_day)
    all_profiles_in_day.close()

def convert_to_celcius(temperature_F):
    if type(temperature_F) == list:
        temperature_F = np.array(temperature_F)
    temperature_C = (5/9)*(temperature_F-32)
    return temperature_C

def initialize_sample_weights(profile, type = 'linear', flip = False,
                              inverse = False):
    sample_funs = {None:None, 'linear':1, 'quadratic':2, 'cubic':3, 'quartic':4,
                   'quintic':5}

    sample_weights = np.linspace(0,1,len(profile))
    if type == 'bump':
        sample_weights = -4*(sample_weights-0.5)**2+1
    else:
        sample_weights = sample_weights**sample_funs[type]

    if flip:
        sample_weights = np.flip(sample_weights)
    if inverse:
        sample_weights = np.reciprocal(sample_weights)

    return sample_weights

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

       standard_profiles['original'] = standard_profiles['pressure'][:]

       temperatures = np.interp(rh_altitudes, temperature_altitudes, temperatures)
       pressures = np.interp(rh_altitudes, pressure_altitudes, pressures)

       standard_profiles['temperature'] = package_data(rh_altitudes,
                                                        temperatures, method='pack')
       standard_profiles['pressure'] = package_data(rh_altitudes, pressures,
                                                    method='pack')

       return standard_profiles

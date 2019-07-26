import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

    def _configure_output(self, output, spline, parameter, spline_num,
                          x_location):
        '''_configure_output evaluates the given spline for the given parameter
        taking into account if the desired connecting point is in the parameter
        dataset or not'''
        if x_location == 'missing':
            if spline_num == 0:
                output = np.append(output, 0)
                index = np.where(parameter < self._a[1])
                self._temporary_x_location = index[0][-1]+1
            parameter = np.insert(parameter, self._temporary_x_location,
                                  self._a[1])
            output += spline(parameter)
            parameter = np.delete(parameter, self._temporary_x_location)
            if spline_num == 1:
                output = np.delete(output, self._temporary_x_location)
        elif x_location == 'found':
            output += spline(parameter)

        return output

    def __call__(self, parameter):
        output = np.zeros(len(parameter))
        for i in range(2):
            spline = HermiteSpline(self._p0[i], self._p1[i],
                                   self._m0[i], self._m1[i],
                                   self._a[i], self._b[i])

            # Output is computed if x is in data set or not
            if not (parameter == self._a[1]).any():
                x_location = 'missing'
            else:
                x_location = 'found'

            output = self._configure_output(output, spline, parameter, i,
                                            x_location)

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
        if temperatures[i] >= 0:
            f = 1.0007+(3.46e-6*pressures[i])
            sat_vps = f*0.61121*np.exp(17.502*temperatures[i]/(240.97+temperatures[i]))
        elif temperatures[i] > -50:
            f = 1.0003+(4.18e-6*pressures[i])
            sat_vps = f*0.61115*np.exp(22.452*temperatures[i]/(272.55+temperatures[i]))
        else:
            f = 1.0003+(4.18e-6*pressures[i])
            sat_vps = f*0.61115*np.exp(22.542*temperatures[i]/(273.48+temperatures[i]))
        saturation_vps.append(sat_vps)
    actual_vps = [humidities[i]/100*saturation_vps[i] for i in range(len(humidities))]
    return actual_vps, saturation_vps


def combine_profiles(data, day, month, year, hour, alt, profile_type='humidity',
                     latitude={'min': 13, 'max': 59},
                     longitude={'min': -144, 'max': -52},
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
    all_profiles_in_day = open(path + '.p', 'wb')
    pickle.dump(profiles, all_profiles_in_day)
    all_profiles_in_day.close()


def convert_to_celcius(temperature_F):
    if type(temperature_F) == list:
        temperature_F = np.array(temperature_F)
    temperature_C = (5/9)*(temperature_F-32)
    return temperature_C


def prepare_standard_profiles(standard_profiles_path='./../../../../data/weather/standard_profiles/standard_profiles.p'):
    standard_profiles = pickle.load(open(standard_profiles_path, 'rb'))

    # Interpolate temperature values at altitudes in relative humidity profile
    temperature_altitudes, temperatures = package_data(standard_profiles['temperature'])
    rh_altitudes, rhs = package_data(standard_profiles['relative humidity'])
    pressure_altitudes, pressures = package_data(standard_profiles['pressure'])

    standard_profiles['original_pressures'] = standard_profiles['pressure'][:]
    standard_profiles['original_temperatures'] = standard_profiles['temperature'][:]

    fun_temperature = interp1d(temperature_altitudes, temperatures)
    fun_pressure = interp1d(pressure_altitudes, pressures)

    temperatures = fun_temperature(rh_altitudes)
    pressures = fun_pressure(rh_altitudes)

    standard_profiles['temperature'] = package_data(rh_altitudes,
                                                    temperatures, method='pack')
    standard_profiles['pressure'] = package_data(rh_altitudes, pressures,
                                                 method='pack')

    return standard_profiles

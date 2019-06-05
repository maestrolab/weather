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
# Simple example for learning HermiteSpline
# import matplotlib.pyplot as plt
#
# a = HermiteSpline(p0=1000,m0=-400,m1=-30,a=0,b=10)
# x_values = np.linspace(0,10,50)
# output = a.__call__(parameter=x_values)
#
# x = [a._a,a._b]
# y = [a._p0,a._p1]
# fig = plt.figure()
# plt.plot(x_values,output)
# plt.scatter(x,y)
# plt.grid(True)
# plt.show()

################################################################################
# Functions to work with loaded profile data.
def package_data(data1, data2=None, method='unpack'):
    '''package_data packs or unpacks data in the form [[data1, data2]]'''
    if method == 'pack':
        packed_data = [[data1[i],data2[i]] for i in range(len(data1))]
        return packed_data
    elif method == 'unpack':
        unpacked_data_1 = [d[0] for d in data1]
        unpacked_data_2 = [d[1] for d in data1]
        return unpacked_data_1, unpacked_data_2

def calculate_vapor_pressures(humidity_profile, temperature_profile):
    saturation_vapor_pressures = []
    for t in temperature_profile:
       if t>=0:
           saturation_vapor_pressures.append(0.61121*np.exp((18.678-
                                                (t/234.5))*(t/(257.14+t))))
       else:
           saturation_vapor_pressures.append(0.61115*np.exp((23.036-
                                                (t/333.7))*(t/(279.82+t))))

    actual_vapor_pressures = [humidity_profile[i]/100*
                              saturation_vapor_pressures[i] for i in
                              range(len(humidity_profile))]
    return(actual_vapor_pressures, saturation_vapor_pressures)

def calculate_humidity_profile(vapor_pressures, saturation_vapor_pressures):
    relative_humidity_profile = [100*vapor_pressures[i]/
                                 saturation_vapor_pressures[i] for i in
                                 range(len(vapor_pressures))]

    return relative_humidity_profile

def convert_to_celcius(temperature_F):
    if type(temperature_F) == list:
        temperature_F = np.array(temperature_F)
    temperature_C = (5/9)*(temperature_F-32)
    return temperature_C
################################################################################
# Load data from pickle file and prepare for spline interpolation
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

# Written so loading process is faster
path = '../../../data/weather/twister/%s_%s_%s_%sdata.p' % (year,month,day,hour)
data = pickle.load(open(path,'rb'))

data['key'] = '%s, %s' % (lat, lon)
data['index'] = list(data['data'].keys()).index(data['key'])

# Unpack data into useful format
altitudes, relative_humidity_profile = package_data(data['data'][data['key']]['humidity'])
altitudes, temperature_profile = package_data(data['data'][data['key']]['temperature'])

temperature_profile = convert_to_celcius(temperature_profile)

# Calculate the actual and saturation vapor pressure profile
actual_vps, saturation_vps = calculate_vapor_pressures(relative_humidity_profile,
                                                       temperature_profile)
################################################################################
import matplotlib.pyplot as plt

# Solve for m1 from original vapor profile
x0 = actual_vps[1]
y0 = altitudes[1]
x1 = actual_vps[0]
y1 = altitudes[0]
y0 = actual_vps[1]
x0 = altitudes[1]
y1 = actual_vps[0]
x1 = altitudes[0]
m0 = (y1-y0)/(x1-x0)

# spline = HermiteSpline(p0=6000,p1=0,m0=-600,m1=m1,a=0,b=max(actual_vps))
# domain = np.linspace(0, actual_vps[0], 5)
# new_profile = spline(parameter=domain)

spline = HermiteSpline(p0=max(actual_vps),p1=0,m0=m0,m1=-1/10000,a=0,b=8000)
domain = np.linspace(0, 8000, 5)
new_profile = spline(parameter=domain)

# parametrized_profile = package_data(new_profile, domain, method='pack') # domain: vapor pressures
# parametrized_profile.reverse()
# new_altitudes, new_vapor = package_data(parametrized_profile)

parametrized_profile = package_data(domain, new_profile, method='pack') # domain: altitudes
#parametrized_profile.reverse()
new_altitudes, new_vapor = package_data(parametrized_profile)

################################################################################
# Interpolate saturation vapor pressures for modified profile
interpolated_saturation_vps = np.interp(new_altitudes, altitudes, saturation_vps)

################################################################################
# Plot original and parametrized profiles
fig = plt.figure()
plt.plot(altitudes, actual_vps, '-o')
#plt.plot(actual_vps, altitudes,'-o')
#plt.plot(saturation_vps, altitudes, '-o')
plt.plot(altitudes, saturation_vps)
#plt.plot(interpolated_saturation_vps, new_altitudes, '-o')
plt.plot(domain, new_profile)
#plt.plot(new_profile, domain)
#plt.xlim(0,35)
#plt.ylim(0,17500)
plt.show()

# Convert vapor pressure profile back to relative humidity
interpolated_rh = calculate_humidity_profile(new_vapor, interpolated_saturation_vps)

interpolated_rh_profile = package_data(new_altitudes, interpolated_rh, method='pack')
################################################################################
# Calculate the perceived loudness for both profiles
# Height to ground (HAG)
index = data['index']
height_to_ground = data['altitudes'][index] / 0.3048  # In feet
sBoom_data = prepare_weather_sBoom(data['data'], index)
[temperature, wind, humidity] = sBoom_data
noise = {'original':0, 'parametrized':0}
noise['original'] = boom_runner(sBoom_data, height_to_ground)

cruise_altitude = data['altitudes'][index]
interpolated_rh_profile += [[cruise_altitude, 0]] # Change last altitude to cruise altitude of aircraft

# # Height to ground the same?
sBoom_data_parametrized = list(sBoom_data[0:2]) + [interpolated_rh_profile]
noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground)
noise['difference'] = noise['original']-noise['parametrized']
print(noise)

################################################################################
# Plot relative humidity profiles
# inter_alts, inter_rh_profile = package_data(interpolated_rh_profile)
#
# fig = plt.figure()
# plt.plot(relative_humidity_profile, altitudes,'-o', label='Original')
# #plt.plot(saturation_vps, altitudes, '-o')
# #plt.plot(interpolated_saturation_vps, new_altitudes, '-o')
# plt.plot(inter_rh_profile, inter_alts, '-o', label='Parametrized')
# plt.xlim(-5,100)
# plt.ylim(0,17500)
# plt.xlabel('Relative Humidity [%]')
# plt.ylabel('Altitude [m]')
# plt.legend(title='Profiles')
# plt.show()

################################################################################
# Calculate parametrized profile for same domain as original profile
original_domain = np.array(altitudes)
parametrized_at_original_domain = spline(parameter=original_domain)
parametrized_at_original_domain = list(parametrized_at_original_domain)
#parametrized_at_original_domain.reverse()
#parametrized_at_original_domain += [[cruise_altitude, 0]] # Change last altitude to cruise altitude of aircraft
fig = plt.figure()
plt.plot(parametrized_at_original_domain)
plt.plot(actual_vps)
plt.show()
# Determine RMSD of parametrized profile
from sklearn.metrics import mean_squared_error

RMSD = mean_squared_error(actual_vps, parametrized_at_original_domain)
print(RMSD)

import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.interpolate import interp1d
from twister.humidity_spline_parametrization.misc_humidity import calculate_vapor_pressures,\
                            prepare_standard_profiles, package_data, convert_to_celcius

def prepare_vapor_pressures(humidity_profiles, temperature_profiles, pressure_profiles):
    '''prepare_vapor_pressures calculates the vapor pressure profile for a set
    of humidity profiles
    '''
    vapor_pressure_profiles = []
    for i in range(len(temperature_profiles)):
        temps = temperature_profiles[i]
        rhs = humidity_profiles[i]
        pres = pressure_profiles[i]
        altitudes, relative_humidities = package_data(rhs)
        altitudes_, temperatures = package_data(temps)
        altitudes__, pressures = package_data(pres)
        vapor_pressures, null = calculate_vapor_pressures(relative_humidities, temperatures, pressures)
        vapor_pressure_profile = package_data(altitudes, vapor_pressures, method='pack')
        vapor_pressure_profiles.append(vapor_pressure_profile)

    return vapor_pressure_profiles

YEAR = '2018'
MONTH = '06'
DAY = '18'
HOUR = '00'
altitude = 50000
directory = './'
locations = ['72469']  # Corresponds to Fort Worth/Dallas

f = open(locations[0] + '.p', 'rb')
data = pickle.load(f)
f.close()

# Interpolate for pressure data
path = './../../data/weather/standard_profiles/standard_profiles.p'
standard_profiles = prepare_standard_profiles(standard_profiles_path=path)
pressure_alts, pressures = package_data(standard_profiles['original'])
fun = interp1d(pressure_alts, pressures)
data['pressure'] = [[] for i in range(len(data['temperature']))]

for i in range(len(data['temperature'])):
    alts, temps = package_data(data['temperature'][i])
    temps_c = convert_to_celcius(temps)
    data['temperature'][i] = package_data(alts, list(temps_c), method='pack')
    profile_pressures = fun(alts)
    data['pressure'][i] = package_data(alts, profile_pressures, method='pack')

data['vapor_pressures'] = prepare_vapor_pressures(data['humidity'],
                                                  data['temperature'],
                                                  data['pressure'])

plt.figure()
for i in range(len(np.array(data['noise']))):
    alt, property = np.array(data['vapor_pressures'][i]).T
    plt.plot(property, alt, 'k', alpha=0.05)
plt.ylabel('Altitude (m)')
plt.xlabel('Vapor Pressure (kPa)')

plt.show()

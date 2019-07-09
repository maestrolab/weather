'''Script to parametrize the humidity profile using one Hermite spline in the
log(altitude) domain.'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

from weather.boom import boom_runner, prepare_weather_sBoom
from weather.scraper.twister import process_data

from parametrize_humidity import ParametrizeHumidity
from misc_humidity import package_data, convert_to_celcius


def refine_profiles(data, keys, altitudes):
    output = []
    for key in keys:
        original_altitudes, original_property = np.array(weather_data[key]).T
        f = interp1d(original_altitudes, original_property, kind='cubic')
        refined_altitudes = f(altitudes)
        output.append(refined_altitudes)
    return output


day = '18'
month = '06'
year = '2018'
hour = '12'
lat = 52
lon = -80
alt_ft = 45000.
alt = alt_ft * 0.3048

nearfield_file = './../../../../data/nearfield/25D_M16_RL5.p'
profile_type = 'vapor_pressures'
geometry_type = 'spline_bump'
if geometry_type == 'spline_log':
    bounds = [[0.5, 3], [0., 0.01], [-1., 1.], [-0.01, 0.], [6, 14], [0, 7]]
elif geometry_type == 'spline':
    bounds = [[1., 4.], [0, 0.001], [-0.1, 0.], [-0.1, 0.], [8000, 16000]]
elif geometry_type == 'spline_bump_log':
    bounds = [[0.5, 3.], [0., 0.5], [-0.1, 0.], [0., 0.], [-0.1, 0.], [7, 10],
              [0., 0.5], [0, 7], [8, 12]]
elif geometry_type == 'spline_bump':
    bounds = [[0.5, 3.], [0., 0.5], [-0.1, 0.], [0., 0.], [-0.1, 0.], [2000, 12000],
              [0., 0.5], [8000, 18000]]
n_altitude = 100

# Extract data
data, altitudes = process_data(day, month, year, hour, alt,
                               directory='./../../../../data/weather/twister/',
                               convert_celcius_to_fahrenheit=True)

key = '%i, %i' % (lat, lon)
weather_data = data[key]
index = list(data.keys()).index(key)
height_to_ground = altitudes[index] / 0.3048  # In feet

# Refine profiles
refined_altitudes = np.linspace(0, height_to_ground * 0.3048, n_altitude)
output = refine_profiles(weather_data, ['humidity', 'temperature', 'pressure'],
                         refined_altitudes)
relative_humidities, temperatures, pressures = output

# bounds = [p0, p1, m0, m1, b, a]

p_profile = ParametrizeHumidity(refined_altitudes, relative_humidities,
                                convert_to_celcius(temperatures), pressures, bounds=bounds,
                                geometry_type=geometry_type)
# plt.figure()
# plt.plot(p_profile.vps, refined_altitudes, label='old')
# plt.plot(np.exp(-p_profile.vps), refined_altitudes, label='new')
# plt.plot(np.exp(p_profile.vps), refined_altitudes, label='new1')
# plt.plot(-np.log(p_profile.vps), refined_altitudes, label='new3')
# plt.plot(np.log(-p_profile.vps), refined_altitudes, label='new4')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(p_profile.vps, max(p_profile.vps)*np.exp(-refined_altitudes/height_to_ground), label='new')
# print(np.exp(-refined_altitudes))
# plt.legend()
# plt.show()
# BREAK
# Optimize profile
fun = p_profile.RMSE
bounds_normalized = [(0, 1) for i in range(len(bounds))]
res = differential_evolution(fun, bounds=bounds_normalized,
                             args=[profile_type], popsize=100,
                             updating='deferred')

# Plot optimized profile
x = p_profile.normalize_inputs(res.x)
p_profile.geometry(x)
p_profile.calculate_humidity_profile()
p_profile.RMSE(res.x, profile_type=profile_type, print_rmse=False)
p_profile.plot()
p_profile.plot(profile_type='relative_humidities')
p_profile.plot(profile_type='log')
plt.show()

# Structuring data for sBoom
temperature_profile = np.array([refined_altitudes, temperatures]).T
humidity_profile = np.array([refined_altitudes, relative_humidities]).T
p_humidity_profile = np.array([p_profile.alts, p_profile.p_rhs]).T
sBoom_data = [list(temperature_profile), 0, list(humidity_profile)]
sBoom_data_parametrized = [list(temperature_profile), 0, list(p_humidity_profile)]

# Noise calculations
noise = {'original': 0, 'parametrized': 0, 'difference': 0}
noise['original'] = boom_runner(sBoom_data, height_to_ground,
                                nearfield_file=nearfield_file)
noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground,
                                    nearfield_file=nearfield_file)
noise['difference'] = noise['original'] - noise['parametrized']
print(noise)

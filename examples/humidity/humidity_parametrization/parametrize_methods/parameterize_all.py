'''Script to parametrize the humidity profile using one Hermite spline in the
log(altitude) domain.'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
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
# latitudes = np.linspace(13, 58, 10)  # 46
latitudes = np.arange(13, 59, dtype='float')
# longitudes = np.linspace(-144, -53, 14)  # 92
longitudes = np.arange(-144, -52, dtype='float')
alt_ft = 45000.
alt = alt_ft * 0.3048

nearfield_file = './../../../../data/nearfield/25D_M16_RL5.p'
profile_type = 'relative_humidities'
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
n_altitude = 20

# Extract data
data, altitudes = process_data(day, month, year, hour, alt,
                               directory='./../../../../data/weather/twister/',
                               convert_celcius_to_fahrenheit=True)
print(list(data.keys())[0], list(data.keys())[-1])
f = open('error.txt', 'w')
for lat in latitudes:
    for lon in longitudes:
        key = '%i, %i' % (lat, lon)
        weather_data = data[key]
        index = list(data.keys()).index(key)
        height_to_ground = altitudes[index] / 0.3048  # In feet

        # Refine profiles
        refined_altitudes = np.linspace(0, height_to_ground * 0.3048, n_altitude)
        output = refine_profiles(weather_data, ['humidity', 'temperature', 'pressure'],
                                 refined_altitudes)
        relative_humidities, temperatures, pressures = output

        p_profile = ParametrizeHumidity(refined_altitudes, relative_humidities,
                                        convert_to_celcius(temperatures), pressures, bounds=bounds,
                                        geometry_type=geometry_type)

        # Optimize profile
        fun = p_profile.RMSE
        bounds_normalized = [(0, 1) for i in range(len(bounds))]
        res = differential_evolution(fun, bounds=bounds_normalized,
                                     args=[profile_type], popsize=200,
                                     updating='deferred')

        # Plot optimized profile
        x = p_profile.normalize_inputs(res.x)
        p_profile.geometry(x)
        p_profile.calculate_humidity_profile()
        p_profile.RMSE(res.x, profile_type=profile_type, print_rmse=False)

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
        print(lat, lon, noise['difference'])
        f.write('%f\t%f\t%f\t%f\t%f\n' % (lat, lon, noise['original'],
                                          noise['parametrized'],
                                          noise['difference']))
f.close()

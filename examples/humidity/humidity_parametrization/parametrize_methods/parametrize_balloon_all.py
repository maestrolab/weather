import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d

from weather.boom import boom_runner, prepare_weather_sBoom
from weather.scraper.twister import process_data

from parametrize_humidity import ParametrizeHumidity
from misc_humidity import package_data, convert_to_celcius, prepare_standard_profiles


def refine_profiles(data, keys, altitudes):
    output = []
    for key in keys:
        original_altitudes, original_property = np.array(weather_data[key]).T
        f = interp1d(original_altitudes, original_property, kind='cubic')
        refined_altitudes = f(altitudes)
        output.append(refined_altitudes)
    return output

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
    bounds = [[0.5, 3.], [0., 0.5], [-0.1, 0.], [0., 0.], [-2.0095108695652176e-05, -2.0095108695652176e-05], [2000, 12000],
              [0., 0.5], [8000, 18000]]

# Load standard pressure profile
standard_path = './../../../../data/weather/standard_profiles/standard_profiles.p'
standard_profiles = prepare_standard_profiles(standard_profiles_path = standard_path)
pres = np.array(standard_profiles['original_pressures'])

# Load balloon data
cruise_alt = 13500
n_altitude = 20

filename = '72469_2000-2018'
balloon_data_path = './../../autoencoder/balloon_data/' + filename
data = pickle.load(open(balloon_data_path + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])

filepath = 'all_parametrize_balloon/txt_files/' + filename + '.txt'
f = open(filepath, 'w')

for i in range(6003,len(rh)):
    weather_data = {'humidity':rh[i], 'temperature':temp[i], 'pressure':pres}

    height_to_ground = (cruise_alt - data['height'][i][0]) / 0.3048

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
    print(i, noise['difference'])

    f.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (i, noise['original'],
                                                          noise['parametrized'],
                                                          noise['difference'],
                                                          x[0], x[1], x[2], x[5],
                                                          x[6], x[7]))

f.close()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

from weather.boom import boom_runner, prepare_weather_sBoom
from weather.scraper.twister import process_data

from parametrize_humidity import ParametrizeHumidity
from misc_humidity import package_data, convert_to_celcius, initialize_sample_weights

day = '18'
month = '06'
year = '2018'
hour = '12_'
lat = 47
lon = -115
alt_ft = 45000.
alt = alt_ft * 0.3048

data, altitudes = process_data(day, month, year, hour, alt,
                               directory='../../../../data/weather/twister/',
                               convert_celcius_to_fahrenheit=True)

key = '%i, %i' % (lat, lon)
weather_data = data[key]
index = list(data.keys()).index(key)
height_to_ground = altitudes[index] / 0.3048  # In feet

# Parametrization process
profile_altitudes, relative_humidities = package_data(weather_data['humidity'])
profile_altitudes, temperatures = package_data(weather_data['temperature'])
profile_altitudes, pressures = package_data(weather_data['pressure'])
temperatures = convert_to_celcius(temperatures)

# bounds = [a, b]
bounds = [[-0.5,0.],[0.5,3]]
p_profile = ParametrizeHumidity(profile_altitudes, relative_humidities,
                                temperatures, pressures, bounds = bounds,
                                geometry_type = 'log')

# Apply sample weights
sample_weights = initialize_sample_weights(profile_altitudes, type = 'quartic')

# Optimize profile
fun = lambda x: p_profile.RMSE(x, sample_weights = sample_weights)
bounds_normalized = [(0,1) for i in range(len(bounds))]
res = differential_evolution(fun, bounds = bounds_normalized)

# Plot optimized profile
x = p_profile.normalize_inputs(res.x)
# print(x)
p_profile.geometry(x)
p_profile.calculate_humidity_profile()
p_profile.RMSE(res.x, sample_weights = sample_weights, print_rmse = 'True')
p_profile.plot()
p_profile.plot(profile_type = 'relative_humidities')
p_profile.plot(profile_type = 'log')
plt.show()

# Calculate noise
p_humidity_profile = package_data(p_profile.alts, p_profile.p_rhs, method='pack')
noise = {'original':0,'parametrized':0,'difference':0}

# Noise calculations (original profile)
nearfield_file='../../../../data/nearfield/25D_M16_RL5.p'
sBoom_data = [weather_data['temperature'], 0, weather_data['humidity']]
height_to_ground = p_profile.alts[-1] / 0.3048
noise['original'] = boom_runner(sBoom_data, height_to_ground, nearfield_file=nearfield_file)

# Noise calculations (parametrized profile)
sBoom_data_parametrized = list(sBoom_data[0:2]) + [p_humidity_profile]
noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground, nearfield_file=nearfield_file)

noise['difference'] = noise['original']-noise['parametrized']

print(noise)

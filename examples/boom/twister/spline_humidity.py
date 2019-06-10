import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

from weather.boom import boom_runner, prepare_weather_sBoom
from weather.scraper.twister import process_data

from examples.boom.twister.parametrize_humidity import ParametrizeHumidity
from examples.boom.twister.misc_humidity import package_data, convert_to_celcius

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
profile_altitudes, pressures = package_data(weather_data['pressure'])
temperatures = convert_to_celcius(temperatures)

# bounds = [p0, m0, m1, b]
bounds = [[1.,4.], [-0.1,0.], [-0.1,0.], [8000,13000]]
p_profile = ParametrizeHumidity(profile_altitudes, relative_humidities,
                                temperatures, pressures, bounds=bounds,
                                geometry_type='spline')

# Optimize profile
fun = p_profile.RMSE
bounds_normalized = [(0,1) for i in range(len(bounds))]
res = differential_evolution(fun, bounds=bounds_normalized)

# Plot optimized profile
print(p_profile.normalize_inputs(res.x))
p_profile.geometry(res.x)
p_profile.calculate_humidity_profile()
p_profile.plot()
p_profile.plot(profile_type='relative_humidities')
plt.show()

# Calculate noise
p_humidity_profile = package_data(p_profile.alts, p_profile.p_rhs, method='pack')
noise = {'original':0,'parametrized':0,'difference':0}

# Noise calculations (original profile)
sBoom_data = prepare_weather_sBoom(data, index)
noise['original'] = boom_runner(sBoom_data, height_to_ground)

# Noise calculations (parametrized profile)
sBoom_data_parametrized = list(sBoom_data[0:2]) + [p_humidity_profile]
noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground)

noise['difference'] = noise['original']-noise['parametrized']

print(noise)

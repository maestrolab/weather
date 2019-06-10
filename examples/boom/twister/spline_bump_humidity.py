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
lat = 50
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

n_points = 1000
parametrize_domain = np.linspace(profile_altitudes[0], profile_altitudes[-1], n_points)

# bounds = [p0, m0, m1, m, x, y, w2]
bounds = [[0.5,3.], [-0.1,0.], [0.,0.], [-0.1,0.], [0.25*n_points,0.75*n_points], \
          [0.,0.5], [8000,18000]]
p_profile = ParametrizeHumidity(profile_altitudes, relative_humidities,
                                temperatures, pressures, bounds = bounds,
                                parametrize_altitudes = parametrize_domain,
                                geometry_type = 'spline_bump')

# Optimize profile
fun = p_profile.RMSE
bounds_normalized = [(0,1) for i in range(len(bounds))]
res = differential_evolution(fun,bounds=bounds_normalized)

# Plot optimized profile
x = p_profile.normalize_inputs(res.x)
print(x)
p_profile.geometry(x)
try:
    p_profile.calculate_humidity_profile()
    p_profile.plot(profile_type='relative_humidities')
except:
    pass
p_profile.plot()
plt.show()

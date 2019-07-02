import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from weather.boom import boom_runner, prepare_weather_sBoom
from weather.scraper.twister import process_data

from parametrize_humidity import ParametrizeHumidity
from misc_humidity import package_data, convert_to_celcius
from choose_cluster_percent_area import choose_method

day = '18'
month = '06'
year = '2018'
hour = '12_'
alt_ft = 45000.
alt = alt_ft * 0.3048

data, altitudes = process_data(day, month, year, hour, alt,
                               directory='./../../../../../data/weather/twister/',
                               convert_celcius_to_fahrenheit=True)

latitudes = list(range(13,59))
longitudes = list(range(-144,-52))

clusters = {i:0 for i in range(8)}

for lat in latitudes:
    for lon in longitudes:
        key = '%i, %i' % (lat, lon)
        weather_data = data[key]
        index = list(data.keys()).index(key)
        height_to_ground = altitudes[index] / 0.3048  # In feet

        # Parametrization process
        profile_altitudes, relative_humidities = package_data(weather_data['humidity'])
        profile_altitudes, temperatures = package_data(weather_data['temperature'])
        profile_altitudes, pressures = package_data(weather_data['pressure'])
        temperatures = convert_to_celcius(temperatures)

        method, RMSE_method, bounds, cluster = choose_method(weather_data['humidity'],
                                                             return_cluster = True)

        clusters[cluster[0]] += 1

        # print(method, RMSE_method)

keys = list(clusters.keys())
for key in keys:
    print(key, clusters[key])

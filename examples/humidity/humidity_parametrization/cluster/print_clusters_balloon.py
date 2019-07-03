import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from misc_humidity import package_data, convert_to_celcius, prepare_standard_profiles
from misc_cluster import truncate_at_altitude
from choose_cluster import choose_method

data = pickle.load(open('./../../../72469_profiles.p','rb'))

# Prepare standard pressure profile
path = './../../../../../data/weather/standard_profiles/standard_profiles.p'
standard_profiles = prepare_standard_profiles(standard_profiles_path = path)
standard_alts, standard_pressures = package_data(standard_profiles['pressure'])
fun = interp1d(standard_alts, standard_pressures)

clusters = {i:0 for i in range(8)}

for day in range(len(data['humidity'])):
    # Truncate profiles
    humidity = truncate_at_altitude(data['humidity'][day])
    temperature = truncate_at_altitude(data['temperature'][day])

    profile_altitudes, relative_humidities = package_data(humidity)
    # profile_altitudes, temperatures = package_data(temperature)
    # pressures = fun(profile_altitudes)
    # temperatures = convert_to_celcius(temperatures)

    method, RMSE_method, bounds, cluster = choose_method(humidity,
                                                         return_cluster = True)

    clusters[cluster[0]] += 1

keys = list(clusters.keys())
for key in keys:
    print(key, clusters[key])

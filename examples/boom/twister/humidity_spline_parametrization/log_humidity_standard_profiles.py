import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from weather.boom import boom_runner, prepare_weather_sBoom

from parametrize_humidity import ParametrizeHumidity
from misc_humidity import package_data, prepare_standard_profiles

# Parametrization process
path = './../../../../data/weather/standard_profiles/standard_profiles.p'
standard_profiles = prepare_standard_profiles(standard_profiles_path=path)
profile_altitudes, relative_humidities = package_data(standard_profiles['relative humidity'])
profile_altitudes, temperatures = package_data(standard_profiles['temperature'])
profile_altitudes, pressures = package_data(standard_profiles['pressure'])

# bounds = [a, b]
bounds = [[-0.5,0.],[0.5,3]]
p_profile = ParametrizeHumidity(profile_altitudes, relative_humidities,
                                temperatures, pressures, bounds = bounds,
                                geometry_type = 'log')

# Apply sample weights
# sample_weights = np.linspace(0,1,len(profile_altitudes))
# sample_weights = -16*(sample_weights-0.5)**4+1
sample_weights = None

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
sBoom_data = [standard_profiles['temperature'], 0, standard_profiles['relative humidity']]
height_to_ground = p_profile.alts[-1] / 0.3048
noise['original'] = boom_runner(sBoom_data, height_to_ground, nearfield_file=nearfield_file)

# Noise calculations (parametrized profile)
sBoom_data_parametrized = list(sBoom_data[0:2]) + [p_humidity_profile]
noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground, nearfield_file=nearfield_file)

noise['difference'] = noise['original']-noise['parametrized']

print(noise)

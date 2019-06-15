import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

from weather.boom import boom_runner, prepare_weather_sBoom

from parametrize_humidity import ParametrizeHumidity
from misc_humidity import package_data, convert_to_celcius, prepare_standard_profiles

# Parametrization process
standard_profiles = prepare_standard_profiles()
profile_altitudes, relative_humidities = package_data(standard_profiles['relative humidity'])
profile_altitudes, temperatures = package_data(standard_profiles['temperature'])
profile_altitudes, pressures = package_data(standard_profiles['pressure'])

# bounds = [p0, p1, m0, m1, b]
bounds = [[1.,4.], [0, 0.0001], [-0.1,0.], [-0.1,0.], [4000,16000]]
p_profile = ParametrizeHumidity(profile_altitudes, relative_humidities,
                                temperatures, pressures, bounds = bounds,
                                geometry_type = 'spline')

# Assign weights for root mean squared error calculation (emphasis on points as
#   high altitudes in profile)
sample_weights = np.linspace(0,1,len(profile_altitudes))
sample_weights = -1/2*(sample_weights+0.2)**2+sample_weights+1

# Optimize profile
fun = lambda x: p_profile.RMSE(x, sample_weights = sample_weights)
bounds_normalized = [(0,1) for i in range(len(bounds))]
res = differential_evolution(fun, bounds = bounds_normalized)

# Plot optimized profile
x = p_profile.normalize_inputs(res.x)
# print(x)
p_profile.geometry(x)
p_profile.calculate_humidity_profile()
p_profile.RMSE(res.x, sample_weights = sample_weights, print_rsme = 'True')
p_profile.plot()
p_profile.plot(profile_type='relative_humidities')
# p_profile.plot_percent_difference()
plt.show()

# Calculate noise
p_humidity_profile = package_data(p_profile.alts, p_profile.p_rhs, method='pack')
noise = {'original':0,'parametrized':0,'difference':0}

# Noise calculations (original profile)
sBoom_data = [standard_profiles['temperature'], 0, standard_profiles['relative humidity']]
height_to_ground = p_profile.alts[-1] / 0.3048
noise['original'] = boom_runner(sBoom_data, height_to_ground)

# Noise calculations (parametrized profile)
sBoom_data_parametrized = list(sBoom_data[0:2]) + [p_humidity_profile]
noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground)

noise['difference'] = noise['original']-noise['parametrized']

print(noise)

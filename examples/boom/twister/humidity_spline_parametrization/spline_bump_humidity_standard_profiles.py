import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

from weather.boom import boom_runner, prepare_weather_sBoom

from parametrize_humidity import ParametrizeHumidity
from misc_humidity import package_data, convert_to_celcius,\
                          prepare_standard_profiles, initialize_sample_weights

# Parametrization process
path = './../../../../data/weather/standard_profiles/standard_profiles.p'
standard_profiles = prepare_standard_profiles(standard_profiles_path=path)
profile_altitudes, relative_humidities = package_data(standard_profiles['relative humidity'])
profile_altitudes, temperatures = package_data(standard_profiles['temperature'])
profile_altitudes, pressures = package_data(standard_profiles['pressure'])

parametrize_domain = np.linspace(profile_altitudes[0], profile_altitudes[-1], 1000)

# bounds = [p0, p1, m0, m1, m, x, y, b]
bounds = [[0.5,3.], [0.,0.005], [-0.1,0.], [0.00,0.00], [-0.1,0.], [3000,12000], \
          [0.,0.5], [16000,18000]]
p_profile = ParametrizeHumidity(profile_altitudes, relative_humidities,
                                temperatures, pressures, bounds = bounds,
                                # parametrize_altitudes = parametrize_domain,
                                geometry_type = 'spline_bump')

# Assign weights for root mean squared error calculation
sample_weights = initialize_sample_weights(profile_altitudes, type = 'quartic')

# Optimize profile
fun = lambda x: p_profile.RMSE(x, sample_weights = sample_weights, print_rmse = 'False')
bounds_normalized = [(0,1) for i in range(len(bounds))]
res = differential_evolution(fun, bounds = bounds_normalized, polish = 'True')

# Plot optimized profile
x = p_profile.normalize_inputs(res.x)
print('Result: ', x)
p_profile.geometry(x)
try:
    p_profile.calculate_humidity_profile()
    p_profile.plot(profile_type='relative_humidities')
    p_profile.plot_percent_difference(x[-3])
except:
    pass
p_profile.RMSE(res.x, sample_weights = sample_weights, print_rmse = 'True')
p_profile.plot()
plt.show()

# Calculate noise
p_humidity_profile = package_data(p_profile.alts, p_profile.p_rhs, method='pack')
noise = {'original':0,'parametrized':0,'difference':0}

# Noise calculations (original profile)
nearfield_file='../../../../data/nearfield/25D_M16_RL5.p'
sBoom_data = [standard_profiles['temperature'], 0, standard_profiles['relative humidity']]
height_to_ground = p_profile.alts[-1] / 0.3048
noise['original'] = boom_runner(sBoom_data, height_to_ground, nearfield_file = nearfield_file)

# Noise calculations (parametrized profile)
sBoom_data_parametrized = list(sBoom_data[0:2]) + [p_humidity_profile]
noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground, nearfield_file = nearfield_file)

noise['difference'] = noise['original']-noise['parametrized']

print(noise)

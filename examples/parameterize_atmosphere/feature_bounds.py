import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from weather.parameterize_atmosphere.autoencoder import *

# Load data
locations_path = '../../data/atmosphere_models/test_data/US_2017_2018.p'

data = pickle.load(open(locations_path,'rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

# Interpolate profiles
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated,
                                                temp_interpolated, elevations)

rh_features = rh_interpolated[:,:,1]
temp_features = temp_interpolated[:,:,1]

# Plot boxplots
labelsize = 8

plt.figure()
rh_bp = plt.boxplot(rh_features, vert = False)
plt.xlabel('Relative Humidity [%]')
plt.ylabel('Feature')
plt.tick_params(labelsize = labelsize)

plt.figure()
temp_bp = plt.boxplot(temp_features, vert = False)
plt.xlabel('Temperature [F]')
plt.ylabel('Feature')
plt.tick_params(labelsize = labelsize)
# plt.show()

rh_feature_bounds = {i:[] for i in range(len(rh_interpolated[0,:,1]))}
temp_feature_bounds = {i:[] for i in range(len(temp_interpolated[0,:,1]))}
j = 0
for i in range(0,len(rh_bp['caps']),2):
    rh_lb = rh_bp['caps'][i].get_xdata()[0]
    rh_ub = rh_bp['caps'][i+1].get_xdata()[0]
    print('RH: [%.2f, %.2f]' % (rh_lb, rh_ub))
    rh_feature_bounds[j] = np.array([rh_lb, rh_ub])

    t_lb = temp_bp['caps'][i].get_xdata()[0]
    t_ub = temp_bp['caps'][i+1].get_xdata()[0]
    print('Temp: [%.2f, %.2f]' % (t_lb, t_ub))
    temp_feature_bounds[j] = np.array([t_lb, t_ub])

    j += 1

print(len(rh_feature_bounds), len(temp_feature_bounds))

# handle = open('feature_bounds.p','wb')
# pickle.dump({'rh':rh_feature_bounds, 'temp':temp_feature_bounds}, handle)
# handle.close()

# for i in range(0,len(rh_bp['caps'])-1,2):
#     # print(rh_bp['caps'][i].get_xdata())
#     print(rh_bp['caps'][i].get_xdata())
#     rh_lb = rh_bp['caps'][i].get_ydata()[0]
#     rh_ub = rh_bp['caps'][i+1].get_ydata()[0]
#     # if rh_ub > 100:
#     #     rh_ub = 100
#     rh_feature_bounds = np.array([rh_lb, rh_ub])
#     # print(rh_feature_bounds)

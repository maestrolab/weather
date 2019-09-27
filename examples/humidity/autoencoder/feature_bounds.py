import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autoencoder import *

# Load data
locations_path = 'balloon_data/2017+2018/US_2017_2018'
# locations_path = 'balloon_data/72469_all_years/72469_2017'
data = pickle.load(open(locations_path + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])

# Interpolate profiles
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)
rh_features = rh_interpolated[:,:,1]
temp_features = temp_interpolated[:,:,1]

# Plot boxplots
plt.figure()
rh_bp = plt.boxplot(rh_features)
plt.xlabel('Feature')
plt.ylabel('Relative Humidity [%]')

plt.figure()
temp_bp = plt.boxplot(temp_features)
plt.xlabel('Feature')
plt.ylabel('Temperature [F]')
plt.show()

# for i in range(0,len(rh_bp['caps'])-1,2):
#     # print(rh_bp['caps'][i].get_xdata())
#     rh_lb = rh_bp['caps'][i].get_ydata()[0]
#     rh_ub = rh_bp['caps'][i+1].get_ydata()[0]
#     if rh_ub > 100:
#         rh_ub = 100
#     rh_feature_bounds = np.array([rh_lb, rh_ub])
#     # print(feature_bounds)

import pickle
import numpy as np

from autoencoder import *

################################################################################
# Load profile data
balloon_data  ='balloon_data/2017+2018/US_2017_2018'

data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

# Interpolate profiles
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)

# Remove outliers from dataset
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated, temp_interpolated, elevations)

print('RH: [%.2f, %.2f]' % (np.min(rh_interpolated[:,:,1]), np.max(rh_interpolated[:,:,1])))
print('Temp: [%.2f, %.2f]' % (np.min(temp_interpolated[:,:,1]), np.max(temp_interpolated[:,:,1])))

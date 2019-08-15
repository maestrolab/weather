import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from autoencoder import *

# Load profile data
balloon_data = 'balloon_data/2017+2018/US_2017_2018'
# balloon_data = 'balloon_data/72469_all_years/72469_2015'
data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array(data['height'])
elevations = np.array([n[0] for n in elevations])

# Interpolate profiles
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)

# Minimum and maximum of profiles
rh_maxes = np.array([np.max(p) for p in rh_interpolated[:,:,1]])
rh_mins = np.array([np.min(p) for p in rh_interpolated[:,:,1]])
temp_maxes = np.array([np.max(p) for p in temp_interpolated[:,:,1]])
temp_mins = np.array([np.min(p) for p in temp_interpolated[:,:,1]])

# Load encoder model
n_params = 9
encoder_path = 'multi-year_vs_single_year/trained_models/'
encoder_name = '%i_params_E' % n_params
encoder = tf.keras.models.load_model(encoder_path + encoder_name + '.h5')

# Prepare data for encoder prediction
predict_data = np.hstack((rh_interpolated[:,:,1][:], temp_interpolated[:,:,1][:]))
bounds = define_bounds(predict_data, n, type = [['min','max'],['min','max']])
print(len(predict_data))
predict_data, bounds, index = bounds_check(predict_data, bounds, index = True)
print(len(predict_data))
y = np.array([normalize_inputs(predict_data[i],n,bounds[i],elevations[i]) for i in
             range(predict_data.shape[0])])

# Compute encoded representation
encoded_rep = encoder.predict(y)
x = encoded_rep

# Delete invalid indices (from bounds_check() above)
elevations = np.delete(elevations, index)
rh_maxes = np.delete(rh_maxes, index)
rh_mins = np.delete(rh_mins, index)
temp_maxes = np.delete(temp_maxes, index)
temp_mins = np.delete(temp_mins, index)

# Delete invalid indices (from elevations check below)
elevation_index = np.where(elevations > 5000)
elevations = np.delete(elevations, elevation_index)
rh_maxes = np.delete(rh_maxes, elevation_index)
rh_mins = np.delete(rh_mins, elevation_index)
temp_maxes = np.delete(temp_maxes, elevation_index)
temp_mins = np.delete(temp_mins, elevation_index)
x = np.delete(x, elevation_index, axis = 0)

# Define parameters
#   {h: elevation at ground, }
parameters = {'h':elevations, 'RH_max':rh_maxes, 'RH_min':rh_mins,
              'T_max':temp_maxes, 'T_min':temp_mins, 'L0':x[:,0],
              'L1':x[:,1], 'L2':x[:,2], 'L3':x[:,3]}

# Plot
df = pd.DataFrame(parameters)
# g = sns.pairplot(df, diag_kind="kde")
# plt.tight_layout()
# plt.show()

means = {}
medians = {}
keys = df.keys()
for key in keys:
    means[key] = df[key].mean()
    medians[key] = df[key].median()

# print(means)
# print(medians)
#
# print(elevations[elevations > 5000])
#
# print(rh_mins[rh_mins > 7])
# print(len(rh_mins[rh_mins > 7]))

# Find number of values at different cutoff points
cutoffs = np.array([1,2,3,4,5,6,7,10,12,15,20])
counts = {c:0 for c in cutoffs}
for cutoff in cutoffs:
    for val in rh_mins:
        if val < cutoff:
            counts[cutoff] += 1

plt.figure()
for key in counts.keys():
    plt.scatter(key, counts[key], color = 'dodgerblue')

plt.xticks(ticks = cutoffs, labels = [str(c) for c in cutoffs])
plt.xlabel('Minimum Relative Humidity [%]')
plt.ylabel('Number of Profiles Below')
plt.plot(range(21), np.array([len(elevations) for i in range(21)]))
plt.show()
